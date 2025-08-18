import os
import paramiko
import requests
import shlex
import time
from typing import Tuple
from backend_module.database import DataBaseManager
from backend_module.config import load_surreal_config
from query import encode_job_query, ml_inference_job_query


def _cd_command(path: str) -> str:
    """Build a safe cd command with home (~) expansion preserved."""
    if path == "~":
        return "cd ~"
    if path.startswith("~/"):
        return f"cd ~/{shlex.quote(path[2:])}"
    return f"cd {shlex.quote(path)}"


def ssh_docker_compose_up(hostname: str, username: str, password: str, workdir: str = "~/mlops-cloud") -> Tuple[str, str, int]:
    """
    SSHでサーバーに接続し、指定ディレクトリで docker compose up -d を実行する
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname, username=username, password=password)
        cd_cmd = _cd_command(workdir)
        # Compose v2 supports --pull always to always pull images when starting
        command = f"{cd_cmd} && docker compose up -d --pull always"
        _, stdout, stderr = client.exec_command(command)

        out = stdout.read().decode()
        err = stderr.read().decode()
        status = stdout.channel.recv_exit_status()

        print("===== SSH COMPOSE UP STDOUT =====")
        print(out)
        print("===== SSH COMPOSE UP STDERR =====")
        print(err)

        return out, err, status
    finally:
        client.close()


def get_latest_github_release(owner: str, repo: str):
    """
    GitHub APIから最新リリース情報を取得する
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    response = requests.get(url)

    if response.status_code == 200:
        latest_release = response.json()
        info = {
            "tag": latest_release.get("tag_name"),
            "name": latest_release.get("name"),
            "url": latest_release.get("html_url")
        }
        print("===== GitHub Release Info =====")
        print("タグ:", info["tag"])
        print("リリース名:", info["name"])
        print("URL:", info["url"])
        return info
    else:
        print("Error:", response.status_code, response.text)
        return None


def ssh_git_switch_detach(
    hostname: str,
    username: str,
    password: str,
    repo_dir: str,
    release: str,
) -> Tuple[str, str, int]:
    """
    指定サーバの指定ディレクトリで、指定リリースへ `git switch --detach` を実行する。

    - 事前に git リポジトリであることを確認し、`git fetch --tags --prune --all` を実行してから切り替えます。
    - 戻り値は (stdout, stderr, exit_status)。非0のときはエラー内容を stderr に含みます。

    Args:
        hostname: SSH 接続先ホスト名/IP。
        username: SSH ユーザー名。
        password: SSH パスワード。
        repo_dir: リポジトリのディレクトリパス。
        release: 切り替え先（タグ名/ブランチ名/コミットSHA）。
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname, username=username, password=password)

        # ~ 先頭のパスはシェル展開を許可しつつ、それ以外は安全にクオート
        if repo_dir == "~":
            cd_cmd = "cd ~"
        elif repo_dir.startswith("~/"):
            cd_cmd = f"cd ~/{shlex.quote(repo_dir[2:])}"
        else:
            cd_cmd = f"cd {shlex.quote(repo_dir)}"

        release_q = shlex.quote(release)

        # 注意: 破壊的操作は避け、fetch と switch のみを行う
        command = " && ".join(
            [
                cd_cmd,
                "test -d .git || { echo 'Not a git repository: '$(pwd) >&2; exit 2; }",
                "git remote -v >/dev/null 2>&1 || { echo 'No git remotes configured' >&2; exit 2; }",
                "git fetch --tags --prune --all",
                f"git switch --detach {release_q}",
                # 状態確認用に HEAD を出力
                "echo 'Switched to:' $(git rev-parse --short HEAD)",
            ]
        )

        _, stdout, stderr = client.exec_command(command)
        out = stdout.read().decode()
        err = stderr.read().decode()
        exit_status = stdout.channel.recv_exit_status()

        print("===== SSH GIT SWITCH STDOUT =====")
        print(out)
        print("===== SSH GIT SWITCH STDERR =====")
        print(err)

        return out, err, exit_status
    finally:
        client.close()


def main():
    # DB client (for job status checks)
    sconf = load_surreal_config()
    dbm = DataBaseManager(
        endpoint_url=sconf["endpoint_url"],
        username=sconf["username"],
        password=sconf["password"],
        namespace=sconf["namespace"],
        database=sconf["database"],
    )

    while True:
        try:
            enc_running = encode_job_query.has_in_progress_job(dbm)
            inf_running = ml_inference_job_query.has_in_progress_job(dbm)

            print("===== Job Progress Check =====")
            print("Encode job in progress:", enc_running)
            print("ML inference job in progress:", inf_running)

            if not enc_running and not inf_running:
                # No running jobs -> deploy latest release
                rel = get_latest_github_release(owner="takanotaiga", repo="mlops-cloud")
                if not rel or not rel.get("tag"):
                    print("Skip deploy: failed to fetch latest release")
                else:
                    tag = rel["tag"]
                    ssh_user = os.getenv("SSH_USERNAME")
                    ssh_pass = os.getenv("SSH_PASSWORD")
                    if not ssh_user or not ssh_pass:
                        print("Skip deploy: SSH_USERNAME or SSH_PASSWORD not set")
                        continue
                    # Switch repo to release tag
                    _, _, status = ssh_git_switch_detach(
                        hostname="172.17.0.1",
                        username=ssh_user,
                        password=ssh_pass,
                        repo_dir="~/mlops-cloud",
                        release=tag,
                    )
                    if status == 0:
                        # Bring up services with always-pull
                        ssh_docker_compose_up(
                            hostname="172.17.0.1",
                            username=ssh_user,
                            password=ssh_pass,
                            workdir="~/mlops-cloud",
                        )
                    else:
                        print("Deploy skipped: git switch failed (see logs above)")
            else:
                print("Running jobs detected; skip deploy this cycle")
        except Exception as e:
            print("Cycle error:", e)

        # Wait 15 minutes
        time.sleep(15 * 60)

if __name__ == "__main__":
    main()
