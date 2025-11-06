/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Fix WebSocket connection for HMR
  webpack: (config, { dev }) => {
    if (dev) {
      config.watchOptions = {
        poll: 1000,
        aggregateTimeout: 300,
      }
    }
    return config
  },
  // Suppress WebSocket warnings
  devIndicators: {
    buildActivityPosition: 'bottom-right',
  },
}

module.exports = nextConfig
