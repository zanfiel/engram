FROM node:22

WORKDIR /app

# Install deps first for layer caching
COPY package.json package-lock.json* ./
RUN npm install --include=optional && npm install @libsql/linux-x64-gnu@0.5.22 sharp

# Copy app files
COPY server.ts engram-gui.html engram-login.html ./

# Data volume
RUN mkdir -p /app/data
VOLUME /app/data

EXPOSE 4200

ENV ENGRAM_PORT=4200
ENV ENGRAM_HOST=0.0.0.0

CMD ["node", "--experimental-strip-types", "server.ts"]
