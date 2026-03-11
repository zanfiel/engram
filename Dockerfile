FROM node:22-slim

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci --omit=dev 2>/dev/null || npm install --omit=dev

COPY server.ts engram-gui.html engram-login.html engram-graph.html ./

RUN mkdir -p /app/data
VOLUME /app/data

ENV ENGRAM_PORT=4200
ENV ENGRAM_HOST=0.0.0.0

EXPOSE 4200

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:4200/health').then(r=>{if(!r.ok)process.exit(1)}).catch(()=>process.exit(1))"

CMD ["node", "--experimental-strip-types", "server.ts"]
