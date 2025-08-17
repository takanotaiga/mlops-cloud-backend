#!/usr/bin/env bash
# build_and_push.sh
set -Eeuo pipefail

# ===== 設定 =====
NAMESPACE="taigatakano"
PROJECT="mlops-cloud-backend"
VARIANTS=(cv mlx)             # ビルド対象
CONTEXT="."                   # ビルドコンテキスト
VERSION="${VERSION:-${1:-v1.0}}"   # 環境変数 VERSION または第1引数で上書き可
BUILDKIT="${BUILDKIT:-1}"     # DOCKER_BUILDKIT=1 をデフォルト

trap 'echo "❌ エラー: 行 $LINENO で失敗しました" >&2' ERR

build_and_push() {
  local variant="$1"
  local repo="${NAMESPACE}/${PROJECT}-${variant}"
  local dockerfile="./Dockerfile.${variant}"

  [[ -f "$dockerfile" ]] || { echo "Dockerfile が見つかりません: $dockerfile" >&2; return 1; }

  echo "🚧 Build: ${repo} (tags: latest, ${VERSION}) from ${dockerfile}"
  DOCKER_BUILDKIT="$BUILDKIT" docker build \
    -f "$dockerfile" \
    -t "${repo}:latest" \
    -t "${repo}:${VERSION}" \
    "$CONTEXT"

  echo "⬆️  Push: ${repo}:latest, ${repo}:${VERSION}"
  docker push "${repo}:latest"
  docker push "${repo}:${VERSION}"
}

for v in "${VARIANTS[@]}"; do
  build_and_push "$v"
done

echo "✅ 完了: ${VARIANTS[*]} に latest / ${VERSION} をビルド & プッシュしました。"
# VERSION=v1.2 ./build.sh