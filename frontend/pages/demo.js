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
  const [isMounted, setIsMounted] = useState(false)

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
    setIsMounted(true)
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
    const [imgSrc, setImgSrc] = useState('')
    const isSupported = plotSupport[model]?.[type] !== false

    // Generate new image URL when model or type changes
    useEffect(() => {
      if (isSupported) {
        setLoading(true)
        setImgError(false)
        // Add timestamp to prevent caching
        const url = `http://localhost:8000/plot?type=${type}&model=${model}&t=${Date.now()}`
        console.log(`[PlotImage] Loading ${label}:`, url)
        setImgSrc(url)
      }
    }, [model, type, isSupported])

    if (!isSupported) {
      return (
        <div className="w-full aspect-square rounded-lg border-2 border-dashed border-purple-400/30 flex items-center justify-center bg-white/5 backdrop-blur-sm">
          <div className="text-center p-4">
            <svg className="w-10 h-10 mx-auto mb-2 text-purple-400/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M18.364 18.364A9 9 0 0 0 5.636 5.636m12.728 12.728A9 9 0 0 1 5.636 5.636m12.728 12.728L5.636 5.636" />
            </svg>
            <p className="text-xs text-purple-300 font-medium">Not supported</p>
            <p className="text-xs text-purple-400/70 mt-1">{model} model</p>
          </div>
        </div>
      )
    }

    if (imgError) {
      return (
        <div className="w-full aspect-square rounded-lg border-2 border-red-400/50 flex items-center justify-center bg-red-500/10 backdrop-blur-sm">
          <div className="text-center p-4">
            <svg className="w-10 h-10 mx-auto mb-2 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-xs text-red-300 font-medium">Failed to load</p>
            <p className="text-xs text-red-400/70 mt-1">Click to retry</p>
          </div>
        </div>
      )
    }

    return (
      <div className="relative w-full bg-white rounded-lg border-2 border-purple-400/30 overflow-hidden hover:border-cyan-400 hover:shadow-lg hover:shadow-cyan-500/20 transition-all duration-300 min-h-[200px]">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white z-10">
            <div className="text-center">
              <svg className="w-8 h-8 text-purple-400 animate-spin mx-auto" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <p className="text-xs text-purple-600 mt-2 font-medium">Loading...</p>
            </div>
          </div>
        )}
        {imgSrc && (
          <img
            key={imgSrc}
            alt={label}
            src={imgSrc}
            onLoad={(e) => {
              console.log(`[PlotImage] ${label} loaded successfully`, e.target.naturalWidth, 'x', e.target.naturalHeight)
              setLoading(false)
            }}
            onError={(e) => {
              console.error(`[PlotImage] ${label} failed to load`, e)
              setImgError(true)
              setLoading(false)
            }}
            className="w-full h-auto block"
          />
        )}
        {!loading && !imgError && imgSrc && (
          <div className="absolute bottom-2 left-2 bg-black/80 text-white text-xs px-3 py-1.5 rounded-md backdrop-blur-sm border border-purple-400/30">
            {label}
          </div>
        )}
      </div>
    )
  }


  return (
    <main className="min-h-screen bg-gradient-to-br from-purple-900 via-indigo-900 to-cyan-900 p-4 md:p-6">
      {/* Animated background particles */}
      {isMounted && (
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {[...Array(15)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-2 h-2 bg-cyan-400 rounded-full opacity-10"
              animate={{
                x: [Math.random() * window.innerWidth, Math.random() * window.innerWidth],
                y: [Math.random() * window.innerHeight, Math.random() * window.innerHeight],
              }}
              transition={{
                duration: Math.random() * 10 + 15,
                repeat: Infinity,
                ease: "linear"
              }}
              style={{
                left: Math.random() * 100 + '%',
                top: Math.random() * 100 + '%',
              }}
            />
          ))}
        </div>
      )}

      <div className="max-w-7xl mx-auto relative z-10">
        {/* Header */}
        <motion.header 
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6 gap-4"
        >
          <div className="flex items-center gap-4">
            <motion.div 
              className="w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-500 via-pink-500 to-cyan-500 flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-purple-500/50"
              animate={{ 
                boxShadow: [
                  "0 0 20px rgba(168, 85, 247, 0.5)",
                  "0 0 40px rgba(236, 72, 153, 0.5)",
                  "0 0 20px rgba(6, 182, 212, 0.5)",
                  "0 0 40px rgba(168, 85, 247, 0.5)",
                ]
              }}
              transition={{ duration: 3, repeat: Infinity }}
            >
              BC
            </motion.div>
            <div>
              <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent">
                BCPM Demo
              </h1>
              <p className="text-sm text-purple-300 mt-1">ML-powered breast cancer prediction system</p>
            </div>
          </div>
          <nav className="flex gap-3">
            <motion.a 
              href='/' 
              className="px-4 py-2 rounded-lg text-white/90 hover:text-white hover:bg-white/10 transition-colors text-sm font-medium backdrop-blur-sm"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Home
            </motion.a>
            <motion.a 
              href='/demo' 
              className="px-4 py-2 rounded-lg bg-gradient-to-r from-purple-500/20 to-cyan-500/20 backdrop-blur-md text-white font-medium shadow-sm text-sm border border-purple-400/30"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Demo
            </motion.a>
          </nav>
        </motion.header>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Input Section */}
          <div className="xl:col-span-2 space-y-6">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="bg-white/10 backdrop-blur-xl rounded-2xl p-6 shadow-2xl border border-white/20 shadow-purple-500/20"
            >
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-white">
                <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                Input Features
              </h2>
              <p className="text-purple-200 text-sm mb-4">Enter {FEATURE_COUNT} numeric features or paste values below</p>

              <textarea
                onBlur={pasteValues}
                placeholder={`Paste ${FEATURE_COUNT} comma or space-separated values here`}
                className="w-full h-24 rounded-lg p-4 border-2 border-purple-400/30 bg-white/5 text-white placeholder-purple-300/50 focus:border-cyan-400 focus:ring-4 focus:ring-cyan-400/20 transition-all outline-none font-mono text-sm resize-none backdrop-blur-sm"
              />

              <div className="flex flex-wrap items-center gap-4 mt-4 p-4 bg-gradient-to-r from-purple-500/10 to-cyan-500/10 rounded-lg border border-purple-400/30 backdrop-blur-sm">
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="text-sm font-medium text-purple-200">Model:</span>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input type='radio' name='model' value='sklearn' checked={modelType === 'sklearn'} onChange={() => setModelType('sklearn')} className="text-purple-500 focus:ring-purple-500" />
                    <span className="text-sm font-medium text-white">Sklearn</span>
                  </label>
                  <label className={`flex items-center gap-2 ${modelsAvailable.stacking ? 'cursor-pointer' : 'opacity-40 cursor-not-allowed'}`}>
                    <input type='radio' name='model' value='stacking' checked={modelType === 'stacking'} onChange={() => modelsAvailable.stacking && setModelType('stacking')} disabled={!modelsAvailable.stacking} className="text-purple-500 focus:ring-purple-500" />
                    <span className="text-sm font-medium text-white">Stacking</span>
                  </label>
                  <label className={`flex items-center gap-2 ${modelsAvailable.dl ? 'cursor-pointer' : 'opacity-40 cursor-not-allowed'}`}>
                    <input type='radio' name='model' value='dl' checked={modelType === 'dl'} onChange={() => modelsAvailable.dl && setModelType('dl')} disabled={!modelsAvailable.dl} className="text-purple-500 focus:ring-purple-500" />
                    <span className="text-sm font-medium text-white">Deep Learning</span>
                  </label>
                </div>

                <label className="flex items-center gap-2 cursor-pointer ml-auto">
                  <input type='checkbox' checked={explain} onChange={e => setExplain(e.target.checked)} className="text-purple-500 focus:ring-purple-500 rounded" />
                  <span className="text-sm font-medium text-white">SHAP Explanation</span>
                </label>
              </div>

              <div className="flex flex-wrap gap-3 mt-6">
                <motion.button
                  onClick={submit}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-6 py-3 rounded-lg bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 text-white font-semibold shadow-lg hover:shadow-xl transition-all flex items-center gap-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Predict
                </motion.button>
                <motion.button
                  onClick={() => { setFeatures(Array(FEATURE_COUNT).fill('')); setResult(null); setError('') }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-6 py-3 rounded-lg border-2 border-purple-400/30 bg-white/5 hover:bg-white/10 text-white backdrop-blur-sm transition-all font-medium"
                >
                  Reset
                </motion.button>
                <motion.button
                  onClick={loadSample}
                  disabled={loadingSample}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-6 py-3 rounded-lg border-2 border-dashed border-purple-400/30 bg-white/5 hover:bg-white/10 text-white backdrop-blur-sm transition-all disabled:opacity-50 font-medium"
                >
                  {loadingSample ? '⏳ Loading...' : '📋 Load Sample'}
                </motion.button>
              </div>

              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-4 rounded-lg bg-red-500/20 border-2 border-red-400/50 text-red-200 text-sm flex items-start gap-3 backdrop-blur-sm"
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
                    <label className="text-xs text-purple-300 mb-1 truncate font-medium" title={featureNames[i] || `Feature ${i + 1}`}>
                      {featureNames[i] || `f${i + 1}`}
                    </label>
                    <input
                      inputMode='decimal'
                      type='number'
                      step="any"
                      value={v}
                      onChange={e => handleChange(i, e.target.value)}
                      placeholder={`${i + 1}`}
                      className="p-2 text-sm rounded-md border-2 border-purple-400/30 bg-white/5 text-white placeholder-purple-300/30 focus:border-cyan-400 focus:ring-2 focus:ring-cyan-400/20 transition-all outline-none backdrop-blur-sm"
                    />
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Visualization Grid */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="bg-white/10 backdrop-blur-xl rounded-2xl p-6 shadow-2xl border border-white/20 shadow-purple-500/20"
            >
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-white">
                <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
            </motion.div>
          </div>

          {/* Results Section */}
          <div className="xl:col-span-1">
            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="bg-white/10 backdrop-blur-xl rounded-2xl p-6 shadow-2xl border border-white/20 sticky top-6 shadow-purple-500/20"
            >
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-white">
                <svg className="w-6 h-6 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Prediction Result
              </h2>

              {result ? (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
                  <div className="p-5 rounded-xl bg-gradient-to-br from-purple-500/20 via-white/10 to-cyan-500/20 border-2 border-purple-400/30 backdrop-blur-sm">
                    <div className="flex justify-between items-center mb-4">
                      <div>
                        <div className="text-xs text-purple-300 uppercase tracking-wide font-semibold">Model</div>
                        <div className="font-bold text-white text-lg">{result.model_type}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs text-purple-300 uppercase tracking-wide font-semibold">Diagnosis</div>
                        <div className={`font-bold text-xl ${result.prediction === 1 ? 'text-red-400' : 'text-green-400'}`}>
                          {result.prediction === 1 ? '⚠️ Malignant' : '✅ Benign'}
                        </div>
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <div className="text-xs text-purple-300 uppercase tracking-wide font-semibold">Confidence</div>
                        <div className="font-bold text-white text-lg">{probabilityPercent}%</div>
                      </div>
                      <div className="h-4 bg-white/10 rounded-full overflow-hidden shadow-inner backdrop-blur-sm">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${probabilityPercent}%` }}
                          transition={{ duration: 0.8, ease: "easeOut" }}
                          className={`h-full rounded-full ${result.prediction === 1 ? 'bg-gradient-to-r from-red-500 to-pink-600' : 'bg-gradient-to-r from-green-400 to-cyan-500'}`}
                        />
                      </div>
                      <div className="text-xs text-purple-200 mt-2 font-mono">Raw: {result.probability?.toFixed(6)}</div>
                    </div>
                  </div>

                  {result.explanation && (
                    <div className="space-y-2">
                      <div className="text-sm font-semibold text-purple-300 flex items-center gap-2">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        SHAP Explanation
                      </div>
                      <img
                        onClick={() => setModalOpen(true)}
                        src={result.explanation}
                        alt='SHAP explanation'
                        className="w-full rounded-lg border-2 border-purple-400/50 cursor-zoom-in hover:border-cyan-400 transition-all shadow-md hover:shadow-lg hover:shadow-cyan-500/50"
                      />
                      <p className="text-xs text-purple-300 text-center italic">Click to enlarge</p>
                    </div>
                  )}

                  {result.explanation_error && (
                    <div className="p-3 rounded-lg bg-amber-500/20 border-2 border-amber-400/50 text-amber-200 text-sm backdrop-blur-sm">
                      <strong className="font-semibold">Note:</strong> {result.explanation_error}
                    </div>
                  )}

                  {result.warning && (
                    <div className="p-3 rounded-lg bg-blue-500/20 border-2 border-blue-400/50 text-blue-200 text-sm flex items-start gap-2 backdrop-blur-sm">
                      <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                      </svg>
                      <span>{result.warning}</span>
                    </div>
                  )}
                </motion.div>
              ) : (
                <div className="text-center py-16 text-purple-300/60">
                  <svg className="w-20 h-20 mx-auto mb-4 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-sm font-medium">No prediction yet</p>
                  <p className="text-xs mt-2">Load a sample or enter values to get started</p>
                </div>
              )}
            </motion.div>
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
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-md p-4"
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
                className="absolute -top-12 right-0 text-white hover:text-purple-300 transition-colors flex items-center gap-2 text-sm font-medium"
              >
                <span>Close</span>
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              <img
                src={result.explanation}
                alt="SHAP explanation enlarged"
                className="max-w-full max-h-[85vh] rounded-2xl shadow-2xl border-4 border-purple-500/40"
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  )
}
