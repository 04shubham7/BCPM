import '../styles/globals.css'
import { Toaster } from 'react-hot-toast'

export default function App({ Component, pageProps }){
  return (
    <>
      <Toaster position="top-right" toastOptions={{duration: 5000}} />
      <Component {...pageProps} />
    </>
  )
}
