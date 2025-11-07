/**
 * Minimal Next.js config to avoid the Turbopack vs webpack warning during dev
 * and to whitelist the local network origin (used by Next's dev server).
 *
 * Two options to silence the message:
 *  - Add an explicit `turbopack: {}` entry (keeps Turbopack enabled but empty config)
 *  - Or run `next dev --webpack` if you want to force webpack instead
 *
 * This file chooses the first option and also sets `allowedDevOrigins` so
 * the dev server won't warn when you access via the machine IP on your LAN.
 */

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Keep Turbopack enabled but explicitly configured (empty) to silence the warning
  turbopack: {},
  // Whitelist the common dev origins (localhost + network IP). Add more if needed.
  allowedDevOrigins: [
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'http://10.16.4.123:3000'
  ],
  // Proxy common API routes to the backend in development so the browser sees
  // same-origin requests (helps avoid extension blocking and CORS during dev).
  async rewrites() {
    return [
      { source: '/api/plot', destination: 'http://127.0.0.1:8000/plot' },
  { source: '/api/report', destination: 'http://127.0.0.1:8000/report' },
      { source: '/api/sample', destination: 'http://127.0.0.1:8000/sample' },
      { source: '/api/models', destination: 'http://127.0.0.1:8000/models' },
      { source: '/api/predict', destination: 'http://127.0.0.1:8000/predict' },
      // awareness PDF generator (multi-language) -> FastAPI
      { source: '/api/awareness', destination: 'http://127.0.0.1:8000/awareness' },
      // pass-through for static files served by the backend
      { source: '/files/:path*', destination: 'http://127.0.0.1:8000/files/:path*' }
    ]
  }
}

module.exports = nextConfig
