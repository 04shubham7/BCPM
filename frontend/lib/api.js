// Centralized API base resolution for frontend
// Priority:
// 1. Use NEXT_PUBLIC_API_BASE if provided (must include protocol in production)
// 2. Else fall back to '/api' so that local dev (Next.js rewrites) or Vercel rewrites can proxy
//    to the backend. This keeps behavior consistent across all pages/components.
// NOTE: When deploying to Vercel without rewrites, you MUST set NEXT_PUBLIC_API_BASE to the full
// backend URL (e.g. https://your-backend.onrender.com) so calls go cross-origin directly.

const raw = process.env.NEXT_PUBLIC_API_BASE || ''
// Trim trailing slash for cleanliness
const trimmed = raw.replace(/\/$/, '')
// If no explicit base supplied, rely on '/api' so that rewrites (dev or vercel.json) can work.
export const API_BASE = trimmed || '/api'

// Helper to build full URL; ensures single slash separation
export function apiUrl(path = '') {
  if (!path.startsWith('/')) path = '/' + path
  return `${API_BASE}${path}`
}

// (Optional) tiny runtime debug you can invoke from the console:
//   window.__SHOW_API_BASE && window.__SHOW_API_BASE()
if (typeof window !== 'undefined') {
  window.__SHOW_API_BASE = () => { console.log('[API_BASE]', API_BASE); return API_BASE }
}
