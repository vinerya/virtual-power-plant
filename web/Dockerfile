# ============================================================================
# VPP web console (Next.js) - multi-stage build using `output: "standalone"`
#
#   docker build -t vpp-web web/
#   docker run -p 3000:3000 \
#     -e API_BASE_URL=http://vpp-api:8000 \
#     -e WS_PUBLIC_URL=wss://vpp.example.com/api/v1/ws \
#     vpp-web
#
# Runtime env (read by the Next.js server): API_BASE_URL, WS_PUBLIC_URL,
# AUTH_COOKIE_NAME. NEXT_PUBLIC_* values are inlined at BUILD time, so they
# are build args here. See web/.env.example.
#
# The session cookie is `Secure` in production builds: serve the console over
# HTTPS (or open it on http://localhost, which browsers treat as secure).
# ============================================================================
FROM node:22-alpine AS base
ENV NEXT_TELEMETRY_DISABLED=1
RUN corepack enable
WORKDIR /app

# ---------------------------------------------------------------------------
FROM base AS deps
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

# ---------------------------------------------------------------------------
FROM base AS builder
ARG NEXT_PUBLIC_WS_URL=""
ARG NEXT_PUBLIC_USE_MOCKS=""
ENV NEXT_PUBLIC_WS_URL=${NEXT_PUBLIC_WS_URL} \
    NEXT_PUBLIC_USE_MOCKS=${NEXT_PUBLIC_USE_MOCKS}
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# `prebuild` copies Monaco into public/monaco (self-hosted editor assets).
RUN pnpm build

# ---------------------------------------------------------------------------
FROM node:22-alpine AS runner
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0
WORKDIR /app

RUN addgroup -S nextjs && adduser -S -G nextjs nextjs

# Standalone server + the static assets it does not copy on its own.
COPY --from=builder --chown=nextjs:nextjs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nextjs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nextjs /app/public ./public

USER nextjs
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:3000/login || exit 1

CMD ["node", "server.js"]
