import { useState, useEffect } from 'react'
import toast from 'react-hot-toast'
import { motion, AnimatePresence } from 'framer-motion'
import Link from 'next/link'
import SiteFooter from '../components/SiteFooter'

export default function Demo() {
  const FEATURE_COUNT = 30
  const [features, setFeatures] = useState(Array(FEATURE_COUNT).fill(''))
  const [featureNames, setFeatureNames] = useState([])
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loadingSample, setLoadingSample] = useState(false)
  const [predicting, setPredicting] = useState(false)
  const [modelType, setModelType] = useState('sklearn')
  const [modelsAvailable, setModelsAvailable] = useState({ sklearn: true, dl: false, stacking: false })
  const [explain, setExplain] = useState(false)
  const [modalOpen, setModalOpen] = useState(false)
  // UI no longer renders inline plot tiles; reports contain all plots.

  const plotSupport = {
    sklearn: { roc: true, confusion: true, fi: true, pr: true },
    stacking: { roc: true, confusion: true, fi: false, pr: true },
    dl: { roc: true, confusion: true, fi: false, pr: true }
  }

  const handleChange = (i, v) => {
    const arr = [...features]
    arr[i] = v
    setFeatures(arr)
  }

  const pasteValues = (e) => {
    const txt = e.target.value
    const parts = txt.split(/[,\s]+/).filter(Boolean)
    if (parts.length >= FEATURE_COUNT) {
      const trimmed = parts.slice(0, FEATURE_COUNT)
      const sanitized = trimmed.map(p => {
        const n = parseFloat(p)
        return Number.isFinite(n) ? String(n) : ''
      })
      setFeatures(sanitized)
      toast.success('✅ Values pasted successfully!')
    }
  }

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') setModalOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Helper: generate & open PDF report (called after prediction or manually)
  const openReport = async (parsedFeatures, mType, preOpenedWin = null) => {
    const reportWin = preOpenedWin || (typeof window !== 'undefined' ? window.open('', '_blank') : null)
    try {
      const res = await fetch('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: parsedFeatures, model_type: mType })
      })
      if (!res.ok) {
        const errText = await res.text()
        toast.error('Report generation failed')
        console.error('Report error', errText)
        if (reportWin) reportWin.close()
        return
      }
      const blob = await res.blob()
      if (!blob || blob.size === 0) {
        toast.error('Empty report')
        if (reportWin) reportWin.close()
        return
      }
      const url = URL.createObjectURL(blob)
      if (reportWin) {
        reportWin.location.href = url
      } else {
        window.open(url, '_blank')
      }
      setTimeout(() => URL.revokeObjectURL(url), 60_000)
      toast.success('📄 Report opened in new tab')
    } catch (e) {
      console.error(e)
      toast.error('Failed to open report')
      if (reportWin) reportWin.close()
    }
  }

  const submit = async () => {
    setError('')
    setPredicting(true)
    const parsed = features.map(x => parseFloat(x))
    if (parsed.some(v => Number.isNaN(v))) {
      const msg = `Please enter ${FEATURE_COUNT} valid numeric feature values.`
      setError(msg)
      toast.error(msg)
      setPredicting(false)
      return
    }
    // Pre-open a tab to avoid popup blockers when we later stream the PDF
    const preWin = typeof window !== 'undefined' ? window.open('', '_blank') : null
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: parsed, model_type: modelType, explain })
      })
      if (!res.ok) {
        let msg = 'API error'
        try {
          const err = await res.json()
          if (err && err.detail) {
            if (Array.isArray(err.detail)) {
              msg = err.detail.map(d => d.msg || JSON.stringify(d)).join('; ')
            } else if (typeof err.detail === 'string') {
              msg = err.detail
            } else {
              msg = JSON.stringify(err.detail)
            }
          } else if (err.message) {
            msg = err.message
          } else if (err.error) {
            msg = err.error
          } else {
            msg = JSON.stringify(err)
          }
        } catch (_) {
          try { msg = await res.text() } catch (e) { }
        }
        setError(msg)
        toast.error(String(msg))
        setPredicting(false)
        if (preWin) preWin.close()
        return
      }
      const data = await res.json()
      if (data.warning) {
        toast(data.warning)
      } else {
        toast.success('✅ Prediction completed!')
      }
      setResult(data)
      // Immediately open PDF report constructed from current features
      openReport(parsed, modelType, preWin)
    } catch (err) {
      const s = String(err)
      setError(s)
      toast.error(s)
      if (preWin) preWin.close()
    } finally {
      setPredicting(false)
    }
  }

  const loadSample = async () => {
    setLoadingSample(true)
    setError('')
    try {
  const res = await fetch('/api/sample')
      if (!res.ok) {
        toast.error('Could not fetch sample')
        setLoadingSample(false)
        return
      }
      const j = await res.json()
      if (j.features && j.features.length >= FEATURE_COUNT) {
        setFeatures(j.features.slice(0, FEATURE_COUNT).map(v => String(v)))
        toast.success('📋 Sample loaded!')
      }
      if (j.feature_names) {
        setFeatureNames(j.feature_names.slice(0, FEATURE_COUNT))
      }
    } catch (err) {
      setError(String(err))
      toast.error('Failed to load sample')
    }
    setLoadingSample(false)
  }

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch('/api/models')
        if (!res.ok) return
        const j = await res.json()
        // Normalize to the frontend shape: { sklearn, dl, stacking, dl_file, dl_runtime }
        setModelsAvailable({
          sklearn: !!j.sklearn,
          stacking: !!j.stacking,
          // allow DL only if both the model file exists and runtime is available
          dl: !!(j.dl_file && j.dl_runtime),
          dl_file: !!j.dl_file,
          dl_runtime: !!j.dl_runtime
        })
        // Do NOT auto-switch the user's selected model. Previously we
        // forced 'stacking' when available which caused confusing UX
        // when the user explicitly chose 'sklearn'. Keep the current
        // selection intact and only update availability.
      } catch (e) { }
    }
    check()
  }, [])

  const probabilityPercent = result && result.probability ? Math.round(result.probability * 100) : 0

  // Plot tiles removed from the UI — all plots are generated inside the PDF report

  return (
    <>
    <main className="min-h-screen p-4 md:p-6">
  <div className="max-w-6xl mx-auto">
        <header className="flex flex-col md:flex-row items-start md:items-center justify-between mb-8 gap-4">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-purple-300 via-violet-300 to-fuchsia-300 bg-clip-text text-transparent">
              Breast Cancer Predictor
            </h1>
            <p className="text-sm text-purple-200/70 mt-2 flex items-center gap-2">
              <span className="w-2 h-2 bg-purple-400 rounded-full animate-pulse" />
              AI-powered diagnostic support system
            </p>
          </div>
          <nav className="flex gap-3">
            <Link href='/' className="px-5 py-2.5 rounded-xl text-purple-200 hover:bg-purple-800/40 transition-all text-sm font-medium border border-purple-500/20 hover:border-purple-400/40">
              Home
            </Link>
            <Link href='/learn' className="px-5 py-2.5 rounded-xl text-purple-200 hover:bg-purple-800/40 transition-all text-sm font-medium border border-purple-500/20 hover:border-purple-400/40">
              Getting Started
            </Link>
            <Link href='/demo' className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 to-violet-600 text-white font-medium shadow-lg shadow-purple-500/30 text-sm border border-purple-400/30">
              Demo
            </Link>
          </nav>
        </header>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          <div className="xl:col-span-2 space-y-6">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="glass-card rounded-2xl p-6 shadow-2xl shadow-purple-900/20">
              <h2 className="text-2xl font-semibold mb-4 flex items-center gap-3 text-purple-100">
                <div className="p-2 rounded-lg bg-purple-600/40 border border-purple-400/30">
                  <svg className="w-6 h-6 text-purple-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                </div>
                Input Features
              </h2>
              <p className="text-purple-200/60 text-sm mb-4">Enter {FEATURE_COUNT} numeric features or paste values</p>

              <textarea
                onBlur={pasteValues}
                placeholder={`Paste ${FEATURE_COUNT} values here...`}
                className="w-full h-28 rounded-xl p-4 border-2 border-purple-500/30 focus:border-purple-400/60 focus:ring-4 focus:ring-purple-500/20 transition-all outline-none font-mono text-sm resize-none bg-purple-900/20 text-purple-100 placeholder-purple-300/30"
              />

              <div className="flex flex-wrap items-center gap-4 mt-6 p-4 bg-purple-900/30 rounded-xl border border-purple-500/20">
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="text-sm font-semibold text-purple-200">Model:</span>
                  <label className="flex items-center gap-2 cursor-pointer group">
                    <input type='radio' name='model' value='sklearn' checked={modelType === 'sklearn'} onChange={() => setModelType('sklearn')} className="text-purple-500" />
                    <span className="text-sm font-medium text-purple-200 group-hover:text-purple-100">Sklearn</span>
                  </label>
                  <label className={`flex items-center gap-2 ${modelsAvailable.stacking ? 'cursor-pointer group' : 'opacity-40 cursor-not-allowed'}`}>
                    <input type='radio' name='model' value='stacking' checked={modelType === 'stacking'} onChange={() => modelsAvailable.stacking && setModelType('stacking')} disabled={!modelsAvailable.stacking} className="text-purple-500" />
                    <span className="text-sm font-medium text-purple-200 group-hover:text-purple-100">Stacking</span>
                  </label>
                  <label className={`flex items-center gap-2 ${modelsAvailable.dl ? 'cursor-pointer group' : 'opacity-40 cursor-not-allowed'}`}>
                    <input type='radio' name='model' value='dl' checked={modelType === 'dl'} onChange={() => modelsAvailable.dl && setModelType('dl')} disabled={!modelsAvailable.dl} className="text-purple-500" />
                    <span className="text-sm font-medium text-purple-200 group-hover:text-purple-100">Deep Learning</span>
                  </label>
                </div>

                {/* Show helpful note if DL model exists but runtime missing */}
                {modelsAvailable.dl_file && !modelsAvailable.dl_runtime && (
                  <div className="w-full mt-2 text-xs text-blue-200/80">⚠️ A deep-learning model file is present on the server but TensorFlow is not installed in the backend environment. To enable, install TensorFlow on the server: <code className="px-1 py-0.5 rounded bg-purple-900/30">pip install tensorflow-cpu</code></div>
                )}

                <label className="flex items-center gap-2 cursor-pointer ml-auto group">
                  <input type='checkbox' checked={explain} onChange={e => setExplain(e.target.checked)} className="text-purple-500 rounded" />
                  <span className="text-sm font-medium text-purple-200 group-hover:text-purple-100">SHAP</span>
                </label>
              </div>

              <div className="flex flex-wrap gap-3 mt-6">
                <button
                  onClick={submit}
                  disabled={predicting}
                  className="px-6 py-3 rounded-xl bg-gradient-to-r from-purple-600 to-violet-600 text-white font-semibold shadow-lg shadow-purple-500/40 hover:shadow-xl transition-all hover:scale-105 active:scale-95 flex items-center gap-2 disabled:opacity-50 border border-purple-400/30">
                  {predicting ? (
                    <>
                      <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Predicting...
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      Predict
                    </>
                  )}
                </button>
                <button
                  onClick={() => { setFeatures(Array(FEATURE_COUNT).fill('')); setResult(null); setError(''); toast.success('Reset') }}
                  className="px-6 py-3 rounded-xl border-2 border-purple-500/40 bg-purple-900/20 hover:bg-purple-800/30 transition-all font-medium text-purple-200">
                  Reset
                </button>
                <button
                  onClick={loadSample}
                  disabled={loadingSample}
                  className="px-6 py-3 rounded-xl border-2 border-dashed border-purple-500/40 bg-purple-900/20 hover:bg-purple-800/30 transition-all disabled:opacity-50 font-medium text-purple-200">
                  {loadingSample ? '⏳ Loading...' : '📋 Load Sample'}
                </button>
              </div>

              {error && (
                <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="mt-4 p-4 rounded-xl bg-red-900/30 border-2 border-red-500/40 text-red-200 text-sm">
                  {error}
                </motion.div>
              )}

              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 mt-6">
                {features.map((v, i) => (
                  <div key={i} className="flex flex-col">
                    <label className="text-xs text-purple-300/70 mb-1.5 truncate font-medium" title={featureNames[i] || `Feature ${i + 1}`}>
                      {featureNames[i] || `f${i + 1}`}
                    </label>
                    <input
                      inputMode='decimal'
                      type='number'
                      step="any"
                      value={v}
                      onChange={e => handleChange(i, e.target.value)}
                      placeholder={`${i + 1}`}
                      className="p-2.5 text-sm rounded-lg border-2 border-purple-500/30 focus:border-purple-400/60 focus:ring-2 focus:ring-purple-500/30 transition-all outline-none bg-purple-900/20 text-purple-100 placeholder-purple-300/30"
                    />
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }} className="glass-card rounded-2xl p-6 shadow-2xl shadow-purple-900/20">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-semibold flex items-center gap-3 text-purple-100">
                  <div className="p-2 rounded-lg bg-purple-600/40 border border-purple-400/30">
                    <svg className="w-6 h-6 text-purple-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  Performance Metrics
                </h2>
                <button
                  onClick={async () => {
                    const parsedNow = features.map(x => parseFloat(x))
                    if (parsedNow.some(v => Number.isNaN(v))) {
                      toast.error('Please provide all feature values before downloading the report')
                      return
                    }
                    openReport(parsedNow, modelType)
                  }}
                  className="px-4 py-2 rounded-lg bg-purple-600/40 hover:bg-purple-600/60 text-purple-100 text-sm font-medium transition-all border border-purple-400/30">
                  Download Reports
                </button>
              </div>
              
              <div className="p-6 rounded-xl bg-purple-900/20 border border-purple-500/20">
                <p className="text-sm text-purple-200/60">Plots and performance visualizations are no longer shown inline. A full, per-prediction PDF report (including ROC, Confusion Matrix, Precision-Recall and Feature Importance where available) is generated after each prediction. Click "Performance Metrics (PDF)" to open the report.</p>
              </div>
            </motion.div>
          </div>

          <div className="xl:col-span-1">
            <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.5, delay: 0.2 }} className="glass-card rounded-2xl p-6 shadow-2xl shadow-purple-900/20 sticky top-6">
              <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3 text-purple-100">
                <div className="p-2 rounded-lg bg-purple-600/40 border border-purple-400/30">
                  <svg className="w-6 h-6 text-purple-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                Result
              </h2>

              {result ? (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
                  <div className="p-6 rounded-xl bg-gradient-to-br from-purple-900/40 via-purple-800/30 to-purple-900/40 border-2 border-purple-500/30 shadow-lg">
                    <div className="flex justify-between items-center mb-5">
                      <div>
                        <div className="text-xs text-purple-300/70 uppercase tracking-wide font-semibold mb-1">Model</div>
                        <div className="font-bold text-purple-100 text-lg capitalize">{result.model_type}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs text-purple-300/70 uppercase tracking-wide font-semibold mb-1">Diagnosis</div>
                        <div className={`font-bold text-xl ${result.prediction === 1 ? 'text-red-400' : 'text-green-400'}`}>
                          {result.prediction === 1 ? (
                            <>⚠️ Malignant</>
                          ) : (
                            <>✅ Benign</>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <div className="text-xs text-purple-300/70 uppercase tracking-wide font-semibold">Confidence</div>
                        <div className="font-bold text-purple-100 text-xl">{probabilityPercent}%</div>
                      </div>
                      <div className="h-5 bg-purple-950/50 rounded-full overflow-hidden shadow-inner border border-purple-500/30">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${probabilityPercent}%` }}
                          transition={{ duration: 1, ease: "easeOut" }}
                          className={`h-full rounded-full ${result.prediction === 1 ? 'bg-gradient-to-r from-red-500 to-red-600' : 'bg-gradient-to-r from-green-500 to-green-600'}`}
                        />
                      </div>
                      <div className="text-xs text-purple-300/50 font-mono text-center pt-1">Raw: {result.probability?.toFixed(6)}</div>
                    </div>
                  </div>

                  {result.explanation && (
                    <div className="space-y-3">
                      <div className="text-sm font-semibold text-purple-200">SHAP Explanation</div>
                      <img
                        onClick={() => setModalOpen(true)}
                        src={result.explanation}
                        alt='SHAP'
                        className="w-full rounded-xl border-2 border-purple-500/40 cursor-zoom-in hover:border-purple-400/70 transition-all"
                      />
                    </div>
                  )}

                  {result.warning && (
                    <div className="p-4 rounded-xl bg-blue-900/30 border-2 border-blue-500/40 text-blue-200 text-sm">
                      ℹ️ {result.warning}
                    </div>
                  )}
                </motion.div>
              ) : (
                <div className="text-center py-20 text-purple-300/40">
                  <svg className="w-24 h-24 mx-auto mb-4 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-base font-medium mb-2">No prediction yet</p>
                  <p className="text-xs text-purple-300/30">Load sample or enter values</p>
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {modalOpen && result && result.explanation && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 p-4" onClick={() => setModalOpen(false)}>
            <motion.div initial={{ scale: 0.9 }} animate={{ scale: 1 }} exit={{ scale: 0.9 }} className="relative" onClick={e => e.stopPropagation()}>
              <button onClick={() => setModalOpen(false)} className="absolute -top-14 right-0 text-white px-4 py-2 rounded-lg bg-purple-900/50 border border-purple-500/30 hover:bg-purple-800/60 transition-all">
                Close ✕
              </button>
              <img src={result.explanation} alt="SHAP enlarged" className="max-w-full max-h-[85vh] rounded-2xl shadow-2xl border-4 border-purple-500/40" />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </main>
    <SiteFooter />
    </>
  )
}
