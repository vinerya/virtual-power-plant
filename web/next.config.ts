import type { NextConfig } from "next";

// Security headers that do not depend on the request. The Content-Security-
// Policy (per-request nonce) and the optional HSTS header are set by
// middleware.ts; see lib/security-headers.ts.
const SECURITY_HEADERS = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  {
    key: "Permissions-Policy",
    value: "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
  },
];

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // Emit .next/standalone (a minimal server.js + traced node_modules) so the
  // Docker image (web/Dockerfile) does not need the full dependency tree.
  // `next start` and `next dev` are unaffected.
  output: "standalone",
  experimental: {
    typedRoutes: false,
  },
  async headers() {
    return [{ source: "/:path*", headers: SECURITY_HEADERS }];
  },
};

export default nextConfig;
