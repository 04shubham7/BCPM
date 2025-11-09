import Link from 'next/link'
import { apiUrl } from '../lib/api'

export default function SiteFooter(){
  return (
    <footer className="mt-16 bg-gradient-to-b from-purple-950/40 to-purple-950/60 border-t border-purple-500/20">
      <div className="max-w-6xl mx-auto px-6 py-10 grid md:grid-cols-3 gap-8">
        <div>
          <Link href="/" className="flex items-center gap-3 mb-3 no-underline">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-purple-600 to-violet-600 text-white font-bold flex items-center justify-center shadow shadow-purple-900/40">SY</div>
            <span className="text-purple-100 font-semibold">Shyamati</span>
          </Link>
          <p className="text-sm text-purple-300/70 leading-relaxed">
            A modern demo that turns ML pipelines into explainable predictions with on-demand reports. For education only — not medical advice.
          </p>
        </div>

        <div>
          <h4 className="text-purple-100 font-semibold mb-3">Explore</h4>
          <ul className="space-y-2 text-sm">
            <li><Link className="text-purple-300 hover:text-purple-200" href="/demo">Demo</Link></li>
            <li><Link className="text-purple-300 hover:text-purple-200" href="/learn">Getting Started</Link></li>
            <li><a className="text-purple-300 hover:text-purple-200" href={apiUrl('awareness?lang=en')} target="_blank" rel="noreferrer">Awareness PDF</a></li>
            <li><a className="text-purple-300 hover:text-purple-200" href="/files/model_pipeline.joblib">Model pipeline</a></li>
          </ul>
        </div>

        <div>
          <h4 className="text-purple-100 font-semibold mb-3">Resources</h4>
          <ul className="space-y-2 text-sm">
            <li><a className="text-purple-300 hover:text-purple-200" href="https://www.who.int/news-room/fact-sheets/detail/breast-cancer" target="_blank" rel="noreferrer">WHO — Breast cancer</a></li>
            <li><a className="text-purple-300 hover:text-purple-200" href="https://www.cancer.gov/types/breast" target="_blank" rel="noreferrer">NCI — Types</a></li>
            <li><a className="text-purple-300 hover:text-purple-200" href="https://www.cdc.gov/cancer/breast/basic_info/index.htm" target="_blank" rel="noreferrer">CDC — Basics</a></li>
          </ul>
        </div>
      </div>

      <div className="border-t border-purple-500/10">
        <div className="max-w-6xl mx-auto px-6 py-5 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="text-xs text-purple-300/60">© {new Date().getFullYear()} Shyamati. Built with FastAPI · Next.js · scikit-learn · Optional TensorFlow</div>
          <div className="flex items-center gap-3 text-xs">
            <a className="text-purple-300 hover:text-purple-200" href="/">Home</a>
            <span className="text-purple-500/40">•</span>
            <a className="text-purple-300 hover:text-purple-200" href="/learn">Learn</a>
            <span className="text-purple-500/40">•</span>
            <a className="text-purple-300 hover:text-purple-200" href="/demo">Demo</a>
          </div>
        </div>
      </div>
    </footer>
  )
}
