# Terminal WebSocket Protocol

Lightweight, unauthenticated websocket bridge that spawns a local shell for each connection. Built for quick frontend integration; every websocket connection gets its own pseudo-TTY running `/bin/bash` by default.

## Connection
- URL: `ws://<host>:<port>` (defaults: host `0.0.0.0`, port `8765`)
- Environment overrides (on the server process):
  - `TERMINAL_WS_HOST`: bind address
  - `TERMINAL_WS_PORT`: listen port
  - `SSH_HOST`, `TERMINAL_SSH_HOST`, or `DOCKER_HOST_IP`: SSH target (fallback order), then `host.docker.internal`, then `172.17.0.1`
  - `SSH_PORT`: SSH port (default `22`)
  - `SSH_USERNAME`, `SSH_PASSWORD`: SSH credentials
  - `TERMINAL_WORKDIR`: working directory sent after login (default `~`)
  - `TERMINAL_TERM`, `TERMINAL_ROWS`, `TERMINAL_COLS`: PTY term/size defaults
- Frames are text-only JSON. No authentication, encryption, or rate limiting.

## Message Shapes

### Server → Client
- `ready`: sent once after the shell is spawned.
  ```json
  {"type": "ready", "sessionId": "1f2a4c8e"}
  ```
- `output`: arbitrary chunks of stdout/stderr from the shell.
  ```json
  {"type": "output", "sessionId": "1f2a4c8e", "data": "ls -la\nREADME.md\n"}
  ```
- `exit`: emitted when the shell exits. Connection closes immediately after sending.
  ```json
  {"type": "exit", "sessionId": "1f2a4c8e", "code": 0}
  ```
- `pong`: response to a `ping` frame from the client.
  ```json
  {"type": "pong", "sessionId": "1f2a4c8e"}
  ```

### Client → Server
- `input`: write raw bytes to the shell’s stdin. Newlines must be provided by the client as `\n` when needed.
  ```json
  {"type": "input", "data": "ls -la\n"}
  ```
- `resize`: update terminal columns/rows for proper TTY formatting.
  ```json
  {"type": "resize", "cols": 120, "rows": 36}
  ```
- `ping`: optional keepalive; server replies with `pong`.
  ```json
  {"type": "ping"}
  ```

## Interaction Flow
1. Open websocket to `ws://<host>:<port>`.
2. Wait for `ready`, capture `sessionId` for routing client state.
3. Send `resize` with the current terminal size, then send `input` frames as the user types.
4. Render `output` chunks in order; chunks are not line-buffered.
5. When `exit` arrives or the socket closes, discard the session and reconnect for a fresh shell.

## Notes
- One websocket connection = one shell process. Reconnecting starts a new shell.
- The server ignores malformed JSON or unknown `type` values.
- Security is intentionally omitted; frontends should wrap this behind their own guardrails if needed.
