FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies
RUN pip install --no-cache-dir \
    websockets==12.0 \
    paramiko==3.5.0

# Copy only what we need
COPY terminal_manager.py ./terminal_manager.py

# Default envs (override at runtime)
# ENV TERMINAL_WS_HOST=0.0.0.0 \
#     TERMINAL_WS_PORT=8765 \
#     SSH_HOST=host.docker.internal \
#     SSH_PORT=22 \
#     SSH_USERNAME=user \
#     SSH_PASSWORD=pass \
#     TERMINAL_WORKDIR=/app

EXPOSE 8765

CMD ["python", "terminal_manager.py"]
