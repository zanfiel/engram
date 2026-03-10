FROM oven/bun:1.1-alpine

WORKDIR /app

# Install build dependencies for transformers.js native bindings
RUN apk add --no-cache python3 make g++

# Copy package files first for layer caching
COPY package.json tsconfig.json ./

# Install dependencies
RUN bun install --production

# Copy source
COPY server.ts ./
COPY engram-gui.html ./
COPY engram-login.html ./

# Create data directory
RUN mkdir -p /app/data

# Environment defaults
ENV ENGRAM_PORT=4200
ENV ENGRAM_HOST=0.0.0.0
ENV ENGRAM_GUI_PASSWORD=changeme

# Expose port
EXPOSE 4200

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD wget -qO- http://localhost:4200/health || exit 1

# Data volume
VOLUME /app/data

# Run
CMD ["bun", "run", "server.ts"]
