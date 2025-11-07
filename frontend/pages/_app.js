import '../styles/globals.css'
import { Toaster } from 'react-hot-toast'

export default function App({ Component, pageProps }) {
  return (
    <>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 5000,
          style: {
            background: '#2d1b4e',
            color: '#e9d5ff',
            border: '1px solid rgba(168, 85, 247, 0.3)',
            boxShadow: '0 10px 15px -3px rgba(139, 92, 246, 0.2)',
          },
          success: {
            iconTheme: {
              primary: '#a78bfa',
              secondary: '#2d1b4e',
            },
          },
          error: {
            iconTheme: {
              primary: '#f87171',
              secondary: '#2d1b4e',
            },
          },
        }}
      />
      <Component {...pageProps} />
    </>
  )
}
