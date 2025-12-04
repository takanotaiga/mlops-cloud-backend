import asyncio
import json
import os
import signal
from dataclasses import dataclass
from typing import Optional

import paramiko
import websockets
from websockets.server import WebSocketServerProtocol


# Websocket bind settings
DEFAULT_WS_HOST = os.getenv("TERMINAL_WS_HOST", "0.0.0.0")
DEFAULT_WS_PORT = int(os.getenv("TERMINAL_WS_PORT", "8765"))


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    return val if val not in (None, "") else default


# SSH target (host machine)
DEFAULT_SSH_HOST = _get_env("TERMINAL_SSH_HOST") or _get_env("SSH_HOST") or _get_env("DOCKER_HOST_IP") or "172.17.0.1"
DEFAULT_SSH_PORT = int(_get_env("SSH_PORT", "22"))
DEFAULT_TERM = _get_env("TERMINAL_TERM", "xterm-256color")
DEFAULT_ROWS = int(_get_env("TERMINAL_ROWS", "24"))
DEFAULT_COLS = int(_get_env("TERMINAL_COLS", "80"))


@dataclass
class TerminalSession:
    """Represents a single SSH channel bound to a websocket connection."""

    client: paramiko.SSHClient
    channel: paramiko.Channel
    session_id: str
    closed: bool = False


def _connect_ssh(username: str, password: str, cols: int, rows: int) -> TerminalSession:
    """Establish an SSH session to the host and start an interactive shell with PTY."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=DEFAULT_SSH_HOST,
        port=DEFAULT_SSH_PORT,
        username=username,
        password=password,
        allow_agent=False,
        look_for_keys=False,
    )

    chan = client.get_transport().open_session()
    chan.get_pty(term=DEFAULT_TERM, width=cols, height=rows)
    chan.invoke_shell()

    session_id = os.urandom(8).hex()
    return TerminalSession(client=client, channel=chan, session_id=session_id)


async def _pump_pty_output(session: TerminalSession, websocket: WebSocketServerProtocol) -> None:
    """Stream SSH channel output to the websocket."""
    chan = session.channel
    loop = asyncio.get_running_loop()
    try:
        while True:
            data = await loop.run_in_executor(None, chan.recv, 1024)
            if not data:
                break
            await websocket.send(
                json.dumps({"type": "output", "sessionId": session.session_id, "data": data.decode(errors="ignore")})
            )
    finally:
        try:
            exit_code = chan.recv_exit_status()
        except Exception:
            exit_code = None
        await websocket.send(json.dumps({"type": "exit", "sessionId": session.session_id, "code": exit_code}))
        await websocket.close()


async def _handle_client_messages(session: TerminalSession, websocket: WebSocketServerProtocol) -> None:
    """Handle inbound websocket messages and forward them to the SSH channel."""
    chan = session.channel
    loop = asyncio.get_running_loop()
    async for raw in websocket:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        msg_type = msg.get("type")
        if msg_type == "input":
            data = msg.get("data", "")
            if not isinstance(data, str):
                continue
            await loop.run_in_executor(None, chan.send, data)
        elif msg_type == "resize":
            cols = int(msg.get("cols", DEFAULT_COLS))
            rows = int(msg.get("rows", DEFAULT_ROWS))
            try:
                chan.resize_pty(width=cols, height=rows)
            except Exception:
                pass
        elif msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong", "sessionId": session.session_id}))


async def _require_auth(websocket: WebSocketServerProtocol) -> Optional[dict]:
    """Parse the initial auth frame containing SSH credentials."""
    try:
        raw = await websocket.recv()
    except websockets.exceptions.ConnectionClosed:
        return None

    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.send(json.dumps({"type": "error", "message": "invalid_json"}))
        await websocket.close()
        return None

    if msg.get("type") != "auth":
        await websocket.send(json.dumps({"type": "error", "message": "expected_auth"}))
        await websocket.close()
        return None

    username = msg.get("username")
    password = msg.get("password")
    if not isinstance(username, str) or not isinstance(password, str) or not username or not password:
        await websocket.send(json.dumps({"type": "error", "message": "missing_credentials"}))
        await websocket.close()
        return None

    try:
        cols = int(msg.get("cols", DEFAULT_COLS))
        rows = int(msg.get("rows", DEFAULT_ROWS))
    except (TypeError, ValueError):
        cols = DEFAULT_COLS
        rows = DEFAULT_ROWS
    return {"username": username, "password": password, "cols": cols, "rows": rows}


async def _handle_connection(websocket: WebSocketServerProtocol) -> None:
    """Accept a websocket connection and bind it to a new SSH shell session."""
    auth = await _require_auth(websocket)
    if not auth:
        return

    try:
        session = _connect_ssh(
            username=auth["username"],
            password=auth["password"],
            cols=auth["cols"],
            rows=auth["rows"],
        )
    except Exception as exc:
        await websocket.send(json.dumps({"type": "error", "message": f"ssh_connect_failed: {exc}"}))
        await websocket.close()
        return

    await websocket.send(json.dumps({"type": "ready", "sessionId": session.session_id}))

    output_task = asyncio.create_task(_pump_pty_output(session, websocket))
    input_task = asyncio.create_task(_handle_client_messages(session, websocket))

    done, pending = await asyncio.wait(
        {output_task, input_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()

    try:
        session.channel.close()
    except Exception:
        pass
    try:
        session.client.close()
    except Exception:
        pass
    session.closed = True


async def main(host: str = DEFAULT_WS_HOST, port: int = DEFAULT_WS_PORT) -> None:
    """Run a websocket server that exposes a shell per connection via SSH back to the host."""
    async with websockets.serve(_handle_connection, host, port, max_size=None):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
