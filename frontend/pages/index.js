import Head from 'next/head'
import Link from 'next/link'
import VideoEmbed from '../components/VideoEmbed'
import SiteFooter from '../components/SiteFooter'
import { apiUrl } from '../lib/api'

export default function Home(){
  const doShare = async () => {
    try {
      const shareUrl = typeof window !== 'undefined' ? window.location.href : 'http://localhost:3000'
        if (navigator.share) {
        await navigator.share({
          title: 'Shyamati — Learn the signs',
          text: 'Know the signs of breast cancer and spread awareness.',
          url: shareUrl,
        })
      } else if (navigator.clipboard) {
        await navigator.clipboard.writeText(shareUrl)
        alert('Link copied!')
      }
    } catch (e) { /* user canceled or unsupported */ }
  }

  return (
    <div className="min-h-screen">
      <Head>
        <title>Shyamati — Demo</title>
        <meta name="description" content="Shyamati — explainable breast cancer prediction demo — FastAPI + Next.js" />
        <meta name="theme-color" content="#2d1b4e" />
      </Head>

      <header className="flex items-center justify-between px-8 py-6 bg-purple-900/20 backdrop-blur-md border-b border-purple-500/20">
        <Link href='/' className="flex items-center gap-4 no-underline">
          <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-purple-600 to-violet-600 flex items-center justify-center text-white font-bold shadow-lg shadow-purple-500/30">
            SY
          </div>
          <div>
            <div className="font-semibold text-lg text-purple-100">Shyamati</div>
            <div className="text-sm text-purple-300/70">Diagnosis demo & model studio</div>
          </div>
        </Link>
        <nav className="flex items-center gap-4">
          <Link href='/' className="text-purple-200 hover:text-purple-100 transition-colors">Home</Link>
          <Link href='/learn' className="text-purple-200 hover:text-purple-100 transition-colors">Getting Started</Link>
          <Link href='/demo' className="px-4 py-2 rounded-lg bg-gradient-to-r from-purple-600 to-violet-600 text-white shadow-lg shadow-purple-500/30 hover:shadow-xl transition-all">
            Try the Demo
          </Link>
        </nav>
      </header>

      <main className="max-w-6xl mx-auto grid lg:grid-cols-[1fr_420px] gap-8 p-6">
        <section className="py-8">
          <h1 className="text-5xl font-extrabold leading-tight bg-gradient-to-r from-purple-300 via-violet-300 to-fuchsia-300 bg-clip-text text-transparent">
            A modern, explainable breast cancer prediction demo
          </h1>
          <p className="text-purple-200/80 mt-4 text-lg">
            We converted a notebook into a reproducible training pipeline with model comparison, stacking, and optional deep-learning. Deployable with FastAPI and demoed with this Next.js frontend. Visuals are generated on-demand from the API, no static artifacts required.
          </p>

          <div className="flex flex-wrap gap-3 mt-6">
            <Link href='/demo' className="inline-block px-6 py-3 rounded-xl bg-gradient-to-r from-purple-600 to-violet-600 text-white shadow-lg shadow-purple-500/40 hover:shadow-xl transition-all hover:scale-105 font-semibold">
              Launch Demo
            </Link>
            <Link href='/learn' className="inline-block px-6 py-3 rounded-xl border-2 border-purple-500/40 text-purple-200 hover:bg-purple-800/30 transition-all font-semibold">
              What Is Breast Cancer?
            </Link>
            <a href={apiUrl('files/report.pdf')} target='_blank' rel='noreferrer' className="inline-block px-6 py-3 rounded-xl border-2 border-purple-500/40 text-purple-200 hover:bg-purple-800/30 transition-all font-semibold">
              Download Example Report
            </a>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-8">
            <FeatureCard 
              title='Production-ready pipeline' 
              desc='Imputer → Scaler → SelectKBest → candidate models + GridSearchCV + stacking' 
            />
            <FeatureCard 
              title='On-demand plots' 
              desc='ROC, PR, confusion, feature importances and SHAP are streamed as images' 
            />
            <FeatureCard 
              title='Explainable' 
              desc='SHAP-based explanations (when available) or local contributions inlined' 
            />
          </div>

          <div className="mt-10">
            <h2 className="text-2xl font-semibold text-purple-100 mb-4">Awareness Spotlight</h2>
            <VideoEmbed
              videoId="DcJDOsEF3-g"
              title="Why Early Detection Matters — Short Overview"
              accent="emerald"
            />
            <p className="text-xs text-purple-300/70 mt-3">Embedded via YouTube privacy-enhanced mode to reduce tracking.</p>
          </div>
        </section>

        <aside className="rounded-xl p-6 glass-card shadow-2xl shadow-purple-900/20">
          <h3 className="text-lg font-semibold text-purple-100 mb-4">Quick Start</h3>
          <ol className="mt-3 text-purple-200/70 list-decimal list-inside space-y-2 text-sm">
            <li>Run the training script: <code className="bg-purple-900/40 px-2 py-1 rounded text-purple-200">python train_model.py</code></li>
            <li>Start the API: <code className="bg-purple-900/40 px-2 py-1 rounded text-purple-200">python -m uvicorn app.main:APP --reload --port 8000</code></li>
            <li>Open this site and click <strong className="text-purple-100">Launch Demo</strong></li>
          </ol>

          <div className="mt-6 grid grid-cols-2 gap-2 text-sm">
            <div className="text-purple-300/70">Model files</div>
            <div className="text-right"><a className="text-purple-400 hover:text-purple-300 transition-colors" href={apiUrl('files/model_pipeline.joblib')}>pipeline</a></div>
            <div className="text-purple-300/70">Stacking</div>
            <div className="text-right"><a className="text-purple-400 hover:text-purple-300 transition-colors" href={apiUrl('files/model_pipeline_stacking.joblib')}>stacking</a></div>
            <div className="text-purple-300/70">Deep Learning</div>
            <div className="text-right"><a className="text-purple-400 hover:text-purple-300 transition-colors" href={apiUrl('files/dl_model.h5')}>dl_model.h5</a></div>
          </div>

          {/* Awareness mini-block to avoid empty space */}
          <div className="mt-8 pt-6 border-t border-purple-500/20 fade-in-up delay-1">
            <h4 className="text-purple-100 font-semibold mb-3">Know the signs</h4>
            <ul className="text-sm text-purple-200/80 space-y-1">
              <li>🫱 New lump in breast or underarm</li>
              <li>🎯 Nipple changes or discharge</li>
              <li>🟣 Skin dimpling, redness, or scaling</li>
              <li>📏 Any sudden change in size or shape</li>
            </ul>
            {/* Tiny stat highlight */}
            <div className="mt-4" aria-live="polite">
              <div className="flex items-center justify-between text-xs text-purple-300/80">
                <span>Early detection saves lives</span>
                <span aria-hidden="true">85%</span>
              </div>
              <div className="h-2 rounded-full bg-purple-900/50 border border-purple-500/30 overflow-hidden mt-1" role="img" aria-label="Estimated survival improves to around 85% with early detection">
                <div className="h-full bg-gradient-to-r from-purple-500 to-violet-600" style={{width: '85%'}} />
              </div>
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              <Link href='/learn' className="px-3 py-2 rounded-lg border border-purple-500/30 text-purple-100 hover:bg-purple-800/40 text-sm">Learn more</Link>
              <a href={apiUrl('awareness?lang=en')} target="_blank" rel="noreferrer" className="px-3 py-2 rounded-lg bg-gradient-to-r from-purple-600 to-violet-600 text-white text-sm shadow shadow-purple-800/40 hover:shadow-purple-700/50">Download guide PDF</a>
              <button onClick={doShare} className="px-3 py-2 rounded-lg border border-purple-500/30 text-purple-100 hover:bg-purple-800/40 text-sm inline-flex items-center gap-1">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <circle cx="18" cy="5" r="3"/>
                  <circle cx="6" cy="12" r="3"/>
                  <circle cx="18" cy="19" r="3"/>
                  <path d="M8.59 13.51l6.83 3.98M15.41 6.51L8.59 10.49"/>
                </svg>
                Share
              </button>
            </div>
            <div className="mt-2 text-xs text-purple-300/60">Educational only — not medical advice.</div>
          </div>

          {/* How the demo works */}
          <div className="mt-8 pt-6 border-t border-purple-500/20 fade-in-up delay-2">
            <h4 className="text-purple-100 font-semibold mb-3">How this demo works</h4>
            <ol className="text-sm text-purple-200/80 space-y-2 list-decimal list-inside">
              <li>Enter test values or use a sample on the Demo page.</li>
              <li>The API standardizes inputs and runs trained models.</li>
              <li>Results and a printable PDF report are generated on the fly.</li>
            </ol>
            <div className="mt-3 grid grid-cols-3 gap-2 text-center">
              <MiniStat label="Inputs" value="30+"/>
              <MiniStat label="Models" value="Stacked"/>
              <MiniStat label="Report" value="PDF"/>
            </div>
          </div>

          {/* Reduce your risk */}
          <div className="mt-8 pt-6 border-t border-purple-500/20 fade-in-up delay-3">
            <h4 className="text-purple-100 font-semibold mb-3">Reduce your risk</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <Tag text="🚶 Stay active"/>
              <Tag text="🥗 Balanced diet"/>
              <Tag text="🥂 Limit alcohol"/>
              <Tag text="🚭 Don’t smoke"/>
            </div>
            <div className="mt-3 text-xs text-purple-300/70">Talk to a healthcare professional for personalized guidance.</div>
            <div className="mt-3 text-xs space-x-3">
              <a className="text-purple-300 hover:text-purple-200 underline" href="https://www.who.int/news-room/fact-sheets/detail/breast-cancer" target="_blank" rel="noreferrer">WHO</a>
              <a className="text-purple-300 hover:text-purple-200 underline" href="https://www.cancer.gov/types/breast" target="_blank" rel="noreferrer">NCI</a>
              <a className="text-purple-300 hover:text-purple-200 underline" href="https://www.cdc.gov/cancer/breast/basic_info/index.htm" target="_blank" rel="noreferrer">CDC</a>
            </div>
          </div>
        </aside>
      </main>

      <SiteFooter />
    </div>
  )
}

function FeatureCard({title, desc}){
  return (
    <div className="p-5 rounded-xl glass-card border border-purple-500/20 hover:border-purple-400/40 transition-all">
      <h4 className="font-semibold mb-2 text-purple-100">{title}</h4>
      <div className="text-sm text-purple-200/70">{desc}</div>
    </div>
  )
}

function MiniStat({label, value}){
  return (
    <div className="rounded-lg bg-purple-950/40 border border-purple-500/30 py-3 flex flex-col items-center justify-center">
      <div className="text-xs text-purple-300/70">{label}</div>
      <div className="text-sm font-semibold text-purple-100 mt-1">{value}</div>
    </div>
  )
}

function Tag({text}){
  return <div className="px-2.5 py-1 rounded-md bg-purple-900/40 border border-purple-500/30 text-purple-100 text-xs">{text}</div>
}
