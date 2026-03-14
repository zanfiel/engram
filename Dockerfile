FROM node:22-slim

WORKDIR /app

# Install deps first for layer caching
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev 2>/dev/null || npm install --omit=dev

# Copy app files
COPY server-split.ts ./
COPY src/ ./src/
COPY engram-gui.html engram-login.html ./

# Data volume
RUN mkdir -p /app/data
VOLUME /app/data

EXPOSE 4200

ENV ENGRAM_PORT=4200
ENV ENGRAM_HOST=0.0.0.0

CMD ["node", "--experimental-strip-types", "server-split.ts"]
