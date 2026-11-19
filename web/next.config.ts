import type { NextConfig } from "next";

/** Python execution service (FastAPI). Override with PY_API_URL. */
const PY_API_URL = process.env.PY_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    root: process.cwd(),
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async rewrites() {
    return [{ source: "/api/py/:path*", destination: `${PY_API_URL}/:path*` }];
  },
};

export default nextConfig;
