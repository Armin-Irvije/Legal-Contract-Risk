import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone output for a slim Docker image (see web/Dockerfile).
  output: "standalone",
};

export default nextConfig;
