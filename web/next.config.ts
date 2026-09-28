import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Emit .next/standalone (a minimal server.js + traced node_modules) so the
  // Docker image (web/Dockerfile) does not need the full dependency tree.
  // `next start` and `next dev` are unaffected.
  output: "standalone",
  experimental: {
    typedRoutes: false,
  },
};

export default nextConfig;
