import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable React strict mode for better development debugging
  reactStrictMode: true,
  
  // Disable x-powered-by header
  poweredByHeader: false,
  
  // Use Turbopack (Next.js 16 default)
  turbopack: {},
};

export default nextConfig;
