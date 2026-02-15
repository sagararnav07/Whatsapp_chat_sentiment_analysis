# Rasa Analyser — Dockerfile
# ============================
# Multi-stage build for Rasa server with custom actions.

FROM rasa/rasa:3.6.20-full AS rasa-server

# Copy Rasa project files
WORKDIR /app
COPY config.yml .
COPY domain.yml .
COPY credentials.yml .
COPY endpoints.yml .
COPY data/ ./data/

# Train the model during build
RUN rasa train --fixed-model-name rasa_analyser

# Expose Rasa API port
EXPOSE 5005

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:5005/ || exit 1

# Start Rasa server with REST channel
CMD ["run", "--enable-api", "--cors", "*", "--port", "5005"]


# ── Action Server ────────────────────────────────────────────────────────────
FROM rasa/rasa-sdk:3.9.0 AS action-server

WORKDIR /app

# Copy custom actions
COPY actions/ ./actions/

# Expose action server port
EXPOSE 5055

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:5055/health || exit 1
