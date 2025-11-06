import { useState, useEffect } from 'react'
import toast from 'react-hot-toast'
import { motion, AnimatePresence } from 'framer-motion'

export default function Demo() {
  const FEATURE_COUNT = 30
  const [features, setFeatures] = useState(Array(FEATURE_COUNT).fill(''))
  const [featureNames, setFeatureNames] = useState([])
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loadingSample, setLoadingSample] = useState(false)
  const [modelType, setModelType] = useState('sklearn')
  const [modelsAvailable, setModelsAvailable] = useState({ sklearn: true, dl: false, stacking: false })
  const [explain, setExplain] = useState(false)
  const [modalOpen, setModalOpen] = useState(false)

  // Map which plot types are supported for each model
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
    }
  }

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') setModalOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const submit = async () => {
    setError('')
    const parsed = features.map(x => parseFloat(x))
    if (parsed.some(v => Number.isNaN(v))) {
      const msg = `Please enter ${FEATURE_COUNT} valid numeric feature values.`
      setError(msg)
      toast.error(msg)
      return
    }
    try {
      const res = await fetch('http://localhost:8000/predict', {
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
          try { msg = await res.text() } catch (e) { /* ignore */ }
        }
        setError(msg)
        toast.error(String(msg))
        return
      }
      const data = await res.json()
      if (data.warning) {
        toast((t) => (<div className="text-sm">{data.warning}</div>))
      } else {
        toast.success('Prediction successful')
      }
      setResult(data)
    } catch (err) {
      const s = String(err)
      setError(s)
      toast.error(s)
    }
  }

  const loadSample = async () => {
    setLoadingSample(true)
    setError('')
    try {
      const res = await fetch('http://localhost:8000/sample')
      if (!res.ok) {
        setError('Could not fetch sample from server')
        setLoadingSample(false)
        return
      }
      const j = await res.json()
      if (j.features && j.features.length >= FEATURE_COUNT) {
        setFeatures(j.features.slice(0, FEATURE_COUNT).map(v => String(v)))
      } else if (j.features) {
        const arr = Array.from({ length: FEATURE_COUNT }, (_, i) => j.features[i] ? String(j.features[i]) : '')
        setFeatures(arr)
      }
      if (j.feature_names) {
        setFeatureNames(j.feature_names.slice(0, FEATURE_COUNT))
      }
    } catch (err) {
      setError(String(err))
    }
    setLoadingSample(false)
  }

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch('http://localhost:8000/models')
        if (!res.ok) return
        const j = await res.json()
        setModelsAvailable(j)
        if (j.stacking) setModelType('stacking')
        else if (!j.dl && modelType === 'dl') setModelType('sklearn')
      } catch (e) {
        // ignore
      }
    }
    check()
  }, [])

  const probabilityPercent = result && result.probability ? Math.round(result.probability * 100) : 0

  // Component to render plot with error handling and loading states
  const PlotImage = ({ type, label, model }) => {
    const [imgError, setImgError] = useState(false)
    const [loading, setLoading] = useState(true)
    const isSupported = plotSupport[model]?.[type] !== false

    if (!isSupported) {
      return (
        <div className="w-full aspect-square rounded-lg border-2 border-dashed border-slate-200 flex items-center justify-center bg-slate-50">
          <div className="text-center p-4">
            <svg className="w-10 h-10 mx-auto mb-2 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
            </svg>
            <p className="text-xs text-slate-400 font-medium">Not supported</p>
            <p className="text-xs text-slate-400 mt-1">{model} model</p>
          </div>
        </div>
      )
    }

    if (imgError) {
      return (
        <div className="w-full aspect-square rounded-lg border-2 border-red-100 flex items-center justify-center bg-red-50">
          <div className="text-center p-4">
            <svg className="w-10 h-10 mx-auto mb-2 text-red-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-xs text-red-500 font-medium">Failed to load</p>
          </div>
        </div>
      )
    }

    return (
      <div className="relative group">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-50 rounded-lg border-2 border-slate-200 animate-pulse">
            <svg className="w-8 h-8 text-slate-300 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </div>
        )}
        <img
          alt={label}
          src={`http://localhost:8000/plot?type=${type}&model=${model}`}
          onLoad={() => setLoading(false)}
          onError={() => { setImgError(true); setLoading(false) }}
          className={`w-full rounded-lg border-2 border-slate-200 transition-all duration-300 ${loading ? 'opacity-0' : 'opacity-100'} hover:border-indigo-300 hover:shadow-md`}
        />
        {!loading && !imgError && (
          <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">
            {label}
          </div>
        )}
      </div>
    )
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-4 md:p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6 gap-4">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-indigo-600 to-blue-500 bg-clip-text text-transparent">
              Breast Cancer Predictor
            </h1>
            <p className="text-sm text-slate-600 mt-1">ML-powered diagnostic support system</p>
          </div>
          <nav className="flex gap-3">
            <a href='/' className="px-4 py-2 rounded-lg text-slate-600 hover:bg-white/60 transition-colors text-sm font-medium">
              Home
            </a>
            <a href='/demo' className="px-4 py-2 rounded-lg bg-white/80 text-indigo-600 font-medium shadow-sm text-sm">
              Demo
            </a>
          </nav>
        </header>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Input Section */}
          <div className="xl:col-span-2 space-y-6">
            <div className="bg-white/80 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-white/20">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                Input Features
              </h2>
              <p className="text-slate-600 text-sm mb-4">Enter {FEATURE_COUNT} numeric features or paste values below</p>

              <textarea
                onBlur={pasteValues}
                placeholder={`Paste ${FEATURE_COUNT} comma or space-separated values here`}
                className="w-full h-24 rounded-lg p-4 border-2 border-slate-200 focus:border-indigo-400 focus:ring-4 focus:ring-indigo-100 transition-all outline-none font-mono text-sm resize-none"
              />

              <div className="flex flex-wrap items-center gap-4 mt-4 p-4 bg-gradient-to-r from-slate-50 to-blue-50 rounded-lg">
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="text-sm font-medium text-slate-700">Model:</span>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input type='radio' name='model' value='sklearn' checked={modelType === 'sklearn'} onChange={() => setModelType('sklearn')} className="text-indigo-600 focus:ring-indigo-500" />
                    <span className="text-sm font-medium">Sklearn</span>
                  </label>
                  <label className={`flex items-center gap-2 ${modelsAvailable.stacking ? 'cursor-pointer' : 'opacity-40 cursor-not-allowed'}`}>
                    <input type='radio' name='model' value='stacking' checked={modelType === 'stacking'} onChange={() => modelsAvailable.stacking && setModelType('stacking')} disabled={!modelsAvailable.stacking} className="text-indigo-600 focus:ring-indigo-500" />
                    <span className="text-sm font-medium">Stacking</span>
                  </label>
                  <label className={`flex items-center gap-2 ${modelsAvailable.dl ? 'cursor-pointer' : 'opacity-40 cursor-not-allowed'}`}>
                    <input type='radio' name='model' value='dl' checked={modelType === 'dl'} onChange={() => modelsAvailable.dl && setModelType('dl')} disabled={!modelsAvailable.dl} className="text-indigo-600 focus:ring-indigo-500" />
                    <span className="text-sm font-medium">Deep Learning</span>
                  </label>
                </div>

                <label className="flex items-center gap-2 cursor-pointer ml-auto">
                  <input type='checkbox' checked={explain} onChange={e => setExplain(e.target.checked)} className="text-indigo-600 focus:ring-indigo-500 rounded" />
                  <span className="text-sm font-medium">SHAP Explanation</span>
                </label>
              </div>

              <div className="flex flex-wrap gap-3 mt-6">
                <button
                  onClick={submit}
                  className="px-6 py-3 rounded-lg bg-gradient-to-r from-indigo-600 to-blue-500 text-white font-semibold shadow-lg hover:shadow-xl transition-all hover:scale-105 active:scale-95 flex items-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Predict
                </button>
                <button
                  onClick={() => { setFeatures(Array(FEATURE_COUNT).fill('')); setResult(null); setError('') }}
                  className="px-6 py-3 rounded-lg border-2 border-slate-300 bg-white hover:bg-slate-50 transition-all font-medium"
                >
                  Reset
                </button>
                <button
                  onClick={loadSample}
                  disabled={loadingSample}
                  className="px-6 py-3 rounded-lg border-2 border-dashed border-slate-300 bg-white hover:bg-slate-50 transition-all disabled:opacity-50 font-medium"
                >
                  {loadingSample ? '⏳ Loading...' : '📋 Load Sample'}
                </button>
              </div>

              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-4 rounded-lg bg-red-50 border-2 border-red-200 text-red-700 text-sm flex items-start gap-3"
                >
                  <svg className="w-5 h-5 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                  <span>{error}</span>
                </motion.div>
              )}

              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 mt-6">
                {features.map((v, i) => (
                  <div key={i} className="flex flex-col">
                    <label className="text-xs text-slate-500 mb-1 truncate font-medium" title={featureNames[i] || `Feature ${i + 1}`}>
                      {featureNames[i] || `f${i + 1}`}
                    </label>
                    <input
                      inputMode='decimal'
                      type='number'
                      step="any"
                      value={v}
                      onChange={e => handleChange(i, e.target.value)}
                      placeholder={`${i + 1}`}
                      className="p-2 text-sm rounded-md border-2 border-slate-200 focus:border-indigo-400 focus:ring-2 focus:ring-indigo-200 transition-all outline-none"
                    />
                  </div>
                ))}
              </div>
            </div>

            {/* Visualization Grid */}
            <div className="bg-white/80 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-white/20">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Model Performance Metrics
              </h2>
              <div className="grid grid-cols-2 gap-4">
                <PlotImage key={`roc-${modelType}`} type="roc" label="ROC Curve" model={modelType} />
                <PlotImage key={`confusion-${modelType}`} type="confusion" label="Confusion Matrix" model={modelType} />
                <PlotImage key={`fi-${modelType}`} type="fi" label="Feature Importance" model={modelType} />
                <PlotImage key={`pr-${modelType}`} type="pr" label="Precision-Recall" model={modelType} />
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="xl:col-span-1">
            <div className="bg-white/80 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-white/20 sticky top-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <svg className="w-6 h-6 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Prediction Result
              </h2>

              {result ? (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
                  <div className="p-5 rounded-xl bg-gradient-to-br from-slate-50 via-white to-slate-50 border-2 border-slate-100">
                    <div className="flex justify-between items-center mb-4">
                      <div>
                        <div className="text-xs text-slate-500 uppercase tracking-wide font-semibold">Model</div>
                        <div className="font-bold text-slate-900 text-lg">{result.model_type}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs text-slate-500 uppercase tracking-wide font-semibold">Diagnosis</div>
                        <div className={`font-bold text-xl ${result.prediction === 1 ? 'text-red-600' : 'text-green-600'}`}>
                          {result.prediction === 1 ? '⚠️ Malignant' : '✅ Benign'}
                        </div>
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <div className="text-xs text-slate-500 uppercase tracking-wide font-semibold">Confidence</div>
                        <div className="font-bold text-slate-900 text-lg">{probabilityPercent}%</div>
                      </div>
                      <div className="h-4 bg-slate-200 rounded-full overflow-hidden shadow-inner">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${probabilityPercent}%` }}
                          transition={{ duration: 0.8, ease: "easeOut" }}
                          className={`h-full rounded-full ${result.prediction === 1 ? 'bg-gradient-to-r from-red-500 to-red-600' : 'bg-gradient-to-r from-green-500 to-green-600'}`}
                        />
                      </div>
                      <div className="text-xs text-slate-500 mt-2 font-mono">Raw: {result.probability?.toFixed(6)}</div>
                    </div>
                  </div>

                  {result.explanation && (
                    <div className="space-y-2">
                      <div className="text-sm font-semibold text-slate-700 flex items-center gap-2">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        SHAP Explanation
                      </div>
                      <img
                        onClick={() => setModalOpen(true)}
                        src={result.explanation}
                        alt='SHAP explanation'
                        className="w-full rounded-lg border-2 border-slate-200 cursor-zoom-in hover:border-indigo-400 transition-all shadow-md hover:shadow-lg"
                      />
                      <p className="text-xs text-slate-500 text-center italic">Click to enlarge</p>
                    </div>
                  )}

                  {result.explanation_error && (
                    <div className="p-3 rounded-lg bg-amber-50 border-2 border-amber-200 text-amber-800 text-sm">
                      <strong className="font-semibold">Note:</strong> {result.explanation_error}
                    </div>
                  )}

                  {result.warning && (
                    <div className="p-3 rounded-lg bg-blue-50 border-2 border-blue-200 text-blue-800 text-sm flex items-start gap-2">
                      <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                      </svg>
                      <span>{result.warning}</span>
                    </div>
                  )}
                </motion.div>
              ) : (
                <div className="text-center py-16 text-slate-400">
                  <svg className="w-20 h-20 mx-auto mb-4 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-sm font-medium">No prediction yet</p>
                  <p className="text-xs mt-2">Load a sample or enter values to get started</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Modal for enlarged explanation */}
      <AnimatePresence>
        {modalOpen && result && result.explanation && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
            onClick={() => setModalOpen(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="relative max-w-5xl max-h-[90vh]"
              onClick={e => e.stopPropagation()}
            >
              <button
                onClick={() => setModalOpen(false)}
                className="absolute -top-12 right-0 text-white hover:text-slate-300 transition-colors flex items-center gap-2 text-sm font-medium"
              >
                <span>Close</span>
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              <img
                src={result.explanation}
                alt="SHAP explanation enlarged"
                className="max-w-full max-h-[85vh] rounded-2xl shadow-2xl border-4 border-white/20"
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  )
}
