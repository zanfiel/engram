FROM oven/bun:1.1-debian

WORKDIR /app

COPY package.json tsconfig.json ./
RUN bun install --production

COPY server.ts engram-gui.html engram-login.html ./
RUN mkdir -p /app/data

ENV ENGRAM_PORT=4200
ENV ENGRAM_HOST=0.0.0.0

EXPOSE 4200

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -sf http://localhost:4200/health || exit 1

VOLUME /app/data
CMD ["bun", "run", "server.ts"]
