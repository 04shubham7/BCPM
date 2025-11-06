import Link from 'next/link'

export default function Home(){
  return (
    <div className="min-h-screen font-sans text-slate-900">
      <header className="flex items-center justify-between px-8 py-6 bg-white/60 backdrop-blur-md border-b">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-indigo-600 to-blue-500 flex items-center justify-center text-white font-bold">BC</div>
          <div>
            <div className="font-semibold text-lg">BreastAI</div>
            <div className="text-sm text-slate-500">Diagnosis demo & model studio</div>
          </div>
        </div>
        <nav className="flex items-center gap-4">
          <Link href='/' className="text-slate-900">Home</Link>
          <Link href='/demo' className="px-3 py-2 rounded-md bg-blue-50 text-blue-700">Try the Demo</Link>
        </nav>
      </header>

      <main className="max-w-6xl mx-auto grid lg:grid-cols-[1fr_420px] gap-8 p-6">
        <section className="py-8">
          <h1 className="text-4xl font-extrabold leading-tight">A modern, explainable breast cancer prediction demo</h1>
          <p className="text-slate-600 mt-4">We converted a notebook into a reproducible training pipeline with model comparison, stacking, and optional deep-learning. Deployable with FastAPI and demoed with this Next.js frontend. Visuals are generated on-demand from the API, no static artifacts required.</p>

          <div className="flex gap-3 mt-6">
            <Link href='/demo' className="inline-block px-4 py-2 rounded-lg bg-gradient-to-r from-indigo-600 to-blue-500 text-white shadow-lg">Launch Demo</Link>
            <a href='http://localhost:8000/files/report.pdf' target='_blank' rel='noreferrer' className="inline-block px-4 py-2 rounded-lg border">Download Report</a>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-8">
            <FeatureCard title='Production-ready pipeline' desc='Imputer → Scaler → SelectKBest → candidate models + GridSearchCV + stacking' />
            <FeatureCard title='On-demand plots' desc='ROC, PR, confusion, feature importances and SHAP are streamed as images' />
            <FeatureCard title='Explainable' desc='SHAP-based explanations (when available) or local contributions inlined' />
          </div>
        </section>

        <aside className="rounded-xl p-6 bg-white/60 backdrop-blur-md shadow-md">
          <h3 className="text-lg font-semibold">Quick Start</h3>
          <ol className="mt-3 text-slate-600 list-decimal list-inside">
            <li>Run the training script: <code>python train_model.py</code></li>
            <li>Start the API: <code>python -m uvicorn app.main:APP --reload --port 8000</code></li>
            <li>Open this site and click <strong>Launch Demo</strong></li>
          </ol>

          <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
            <div className="text-slate-500">Model files</div>
            <div className="text-right"><a className="text-blue-600" href='/files/model_pipeline.joblib'>pipeline</a></div>
            <div className="text-slate-500">Stacking</div>
            <div className="text-right"><a className="text-blue-600" href='/files/model_pipeline_stacking.joblib'>stacking</a></div>
            <div className="text-slate-500">Deep Learning</div>
            <div className="text-right"><a className="text-blue-600" href='/files/dl_model.h5'>dl_model.h5</a></div>
          </div>
        </aside>
      </main>

      <footer className="text-center py-6 text-slate-500 border-t">Built with FastAPI · Next.js · scikit-learn · Optional TensorFlow</footer>
    </div>
  )
}

function FeatureCard({title,desc}){
  return (
    <div className="p-4 rounded-lg bg-white/70 border">
      <h4 className="font-medium mb-1">{title}</h4>
      <div className="text-sm text-slate-600">{desc}</div>
    </div>
  )
}
