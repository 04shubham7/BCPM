import Link from 'next/link'
import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'

export default function Home() {
  const [models, setModels] = useState({ sklearn: false, stacking: false, dl: false })
  const [scrollY, setScrollY] = useState(0)
  const [isMounted, setIsMounted] = useState(false)

  useEffect(() => {
    setIsMounted(true)
    const handleScroll = () => setScrollY(window.scrollY)
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  useEffect(() => {
    fetch('http://localhost:8000/models')
      .then(res => res.json())
      .then(data => setModels(data))
      .catch(() => {})
  }, [])

  return (
    <div className="min-h-screen font-sans bg-gradient-to-br from-purple-900 via-indigo-900 to-cyan-900 text-white overflow-hidden relative">
      {/* Animated background particles */}
      {isMounted && (
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-2 h-2 bg-cyan-400 rounded-full opacity-20"
              animate={{
                x: [Math.random() * window.innerWidth, Math.random() * window.innerWidth],
                y: [Math.random() * window.innerHeight, Math.random() * window.innerHeight],
              }}
              transition={{
                duration: Math.random() * 10 + 20,
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

      {/* Glassmorphic Header */}
      <motion.header 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, type: "spring" }}
        className={`sticky top-0 z-50 transition-all duration-300 ${
          scrollY > 50 
            ? 'bg-black/30 backdrop-blur-xl border-b border-white/10 shadow-lg shadow-purple-500/20' 
            : 'bg-white/5 backdrop-blur-md border-b border-white/5'
        }`}
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between px-6 lg:px-8 py-4">
          <motion.div 
            className="flex items-center gap-4"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400 }}
          >
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
              <div className="font-bold text-xl bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent">
                BCPM
              </div>
              <div className="text-sm text-purple-300">Breast Cancer Prediction Model</div>
            </div>
          </motion.div>
          
          <nav className="flex items-center gap-6">
            <Link href='/' className="text-white/90 hover:text-white transition-colors font-medium">
              Home
            </Link>
            <Link href='#features' className="text-white/90 hover:text-white transition-colors font-medium">
              Features
            </Link>
            <Link href='#models' className="text-white/90 hover:text-white transition-colors font-medium">
              Models
            </Link>
            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Link 
                href='/demo' 
                className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 text-white font-semibold shadow-lg shadow-purple-500/50 hover:shadow-xl hover:shadow-pink-500/50 transition-all"
              >
                Launch Demo
              </Link>
            </motion.div>
          </nav>
        </div>
      </motion.header>

      {/* Hero Section */}
      <main className="relative">
        <section className="max-w-7xl mx-auto px-6 lg:px-8 py-20 lg:py-32">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left: Text Content */}
            <div className="space-y-8">
              <motion.div
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
              >
                <motion.span 
                  className="inline-block px-4 py-2 rounded-full bg-purple-500/20 backdrop-blur-sm border border-purple-400/30 text-purple-300 text-sm font-semibold mb-6"
                  animate={{ 
                    boxShadow: [
                      "0 0 20px rgba(168, 85, 247, 0.3)",
                      "0 0 30px rgba(168, 85, 247, 0.5)",
                      "0 0 20px rgba(168, 85, 247, 0.3)",
                    ]
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  🏥 AI-Powered Medical Diagnostics
                </motion.span>
                
                <h1 className="text-5xl lg:text-7xl font-extrabold leading-tight mb-6">
                  <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent">
                    Predict Breast Cancer
                  </span>
                  <br />
                  <span className="text-white">
                    With ML Precision
                  </span>
                </h1>
                
                <p className="text-xl text-purple-200 leading-relaxed">
                  Advanced machine learning system achieving <span className="text-cyan-400 font-bold">98.25% accuracy</span> with 
                  ensemble stacking, deep learning, and SHAP explainability. Production-ready FastAPI backend with 
                  modern Next.js frontend.
                </p>
              </motion.div>

              <motion.div 
                className="flex flex-wrap gap-4"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
              >
                <motion.div whileHover={{ scale: 1.05, y: -5 }} whileTap={{ scale: 0.95 }}>
                  <Link 
                    href='/demo' 
                    className="inline-flex items-center gap-3 px-8 py-4 rounded-2xl bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 text-white font-bold text-lg shadow-2xl shadow-purple-500/50 hover:shadow-pink-500/50 transition-all group"
                  >
                    <span>Launch Demo</span>
                    <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                    </svg>
                  </Link>
                </motion.div>
                
                <motion.div whileHover={{ scale: 1.05, y: -5 }} whileTap={{ scale: 0.95 }}>
                  <a 
                    href='http://localhost:8000/files/report.pdf' 
                    target='_blank' 
                    rel='noreferrer' 
                    className="inline-flex items-center gap-3 px-8 py-4 rounded-2xl bg-white/10 backdrop-blur-md border-2 border-white/20 text-white font-bold text-lg hover:bg-white/20 transition-all group"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <span>Download Report</span>
                  </a>
                </motion.div>
              </motion.div>

              {/* Stats */}
              <motion.div 
                className="grid grid-cols-3 gap-6 pt-8"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.8, delay: 0.6 }}
              >
                <StatCard number="98.25%" label="Accuracy" />
                <StatCard number="100%" label="Precision" />
                <StatCard number="3+" label="Models" />
              </motion.div>
            </div>

            {/* Right: Glassmorphic Card with Model Info */}
            <motion.div
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="relative"
            >
              <motion.div 
                className="relative rounded-3xl p-8 bg-white/10 backdrop-blur-xl border border-white/20 shadow-2xl shadow-purple-500/20"
                whileHover={{ scale: 1.02, y: -10 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                {/* Glow effect */}
                <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-purple-500/20 via-pink-500/20 to-cyan-500/20 blur-2xl -z-10"></div>
                
                <h3 className="text-2xl font-bold mb-6 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
                  Available Models
                </h3>
                
                <div className="space-y-4">
                  <ModelStatus 
                    name="Pipeline Model" 
                    file="model_pipeline.joblib" 
                    available={models.sklearn}
                    description="RandomForest with feature selection"
                  />
                  <ModelStatus 
                    name="Stacking Ensemble" 
                    file="model_pipeline_stacking.joblib" 
                    available={models.stacking}
                    description="5 base models + meta-learner"
                  />
                  <ModelStatus 
                    name="Deep Learning" 
                    file="dl_model.h5" 
                    available={models.dl}
                    description="Neural network with TensorFlow"
                  />
                </div>

                <div className="mt-8 p-4 rounded-2xl bg-gradient-to-r from-purple-500/10 to-cyan-500/10 border border-purple-400/30">
                  <h4 className="font-semibold text-purple-300 mb-3">Quick Start</h4>
                  <ol className="space-y-2 text-sm text-purple-200">
                    <li className="flex items-start gap-2">
                      <span className="flex-shrink-0 w-6 h-6 rounded-full bg-purple-500/30 flex items-center justify-center text-xs font-bold">1</span>
                      <span>Train: <code className="text-cyan-400 font-mono">python train_model.py</code></span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="flex-shrink-0 w-6 h-6 rounded-full bg-purple-500/30 flex items-center justify-center text-xs font-bold">2</span>
                      <span>Start API: <code className="text-cyan-400 font-mono">uvicorn app.main:APP</code></span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="flex-shrink-0 w-6 h-6 rounded-full bg-purple-500/30 flex items-center justify-center text-xs font-bold">3</span>
                      <span>Launch the demo and start predicting!</span>
                    </li>
                  </ol>
                </div>
              </motion.div>
            </motion.div>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="max-w-7xl mx-auto px-6 lg:px-8 py-20">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
              Powerful Features
            </h2>
            <p className="text-xl text-purple-200">
              Production-ready ML pipeline with explainability
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <FeatureCard 
              icon="🎯"
              title="High Accuracy"
              description="Achieves 98.25% accuracy with ensemble stacking and perfect precision (100%)"
              delay={0.1}
            />
            <FeatureCard 
              icon="📊"
              title="Real-time Visualization"
              description="ROC curves, confusion matrices, feature importance, and SHAP plots generated on-demand"
              delay={0.2}
            />
            <FeatureCard 
              icon="🔍"
              title="SHAP Explainability"
              description="Understand model decisions with SHAP values and local feature contributions"
              delay={0.3}
            />
            <FeatureCard 
              icon="⚡"
              title="Fast API Backend"
              description="RESTful endpoints with FastAPI for predictions and visualizations"
              delay={0.4}
            />
            <FeatureCard 
              icon="🎨"
              title="Modern UI"
              description="Beautiful Next.js frontend with Tailwind CSS and Framer Motion animations"
              delay={0.5}
            />
            <FeatureCard 
              icon="🔬"
              title="Multiple Models"
              description="Compare sklearn pipeline, stacking ensemble, and deep learning models"
              delay={0.6}
            />
          </div>
        </section>

        {/* Model Comparison Section */}
        <section id="models" className="max-w-7xl mx-auto px-6 lg:px-8 py-20">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl lg:text-5xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
              Model Architecture
            </h2>
            <p className="text-xl text-purple-200">
              Ensemble of 5 base models with stacking meta-learner
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8">
            <ModelArchCard 
              title="Base Models"
              models={[
                "Logistic Regression (98.25% acc)",
                "Random Forest (96.49% acc)",
                "XGBoost (95.61% acc)",
                "LightGBM (95.61% acc)",
                "HistGradientBoosting (95.61% acc)"
              ]}
            />
            <ModelArchCard 
              title="Ensemble & Deep Learning"
              models={[
                "Stacking Ensemble (98.25% acc)",
                "Neural Network (TensorFlow/Keras)",
                "Feature Selection (SelectKBest k=8)",
                "GridSearchCV Optimization"
              ]}
            />
          </div>
        </section>

        {/* CTA Section */}
        <section className="max-w-7xl mx-auto px-6 lg:px-8 py-20">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="relative rounded-3xl p-12 bg-gradient-to-r from-purple-600/20 via-pink-600/20 to-cyan-600/20 backdrop-blur-xl border border-white/20 text-center overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 via-pink-500/10 to-cyan-500/10 blur-3xl"></div>
            
            <div className="relative z-10">
              <h2 className="text-4xl lg:text-5xl font-bold mb-6">
                Ready to Experience AI Diagnosis?
              </h2>
              <p className="text-xl text-purple-200 mb-8 max-w-2xl mx-auto">
                Try our interactive demo with real-time predictions, visualizations, and SHAP explanations
              </p>
              
              <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                <Link 
                  href='/demo'
                  className="inline-flex items-center gap-3 px-10 py-5 rounded-2xl bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 text-white font-bold text-xl shadow-2xl shadow-purple-500/50 hover:shadow-pink-500/50 transition-all"
                >
                  <span>Launch Demo Now</span>
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                </Link>
              </motion.div>
            </div>
          </motion.div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/10 bg-black/20 backdrop-blur-md py-8">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="text-purple-300 text-sm">
              © 2025 BCPM - Breast Cancer Prediction Model
            </div>
            <div className="flex items-center gap-6 text-sm text-purple-300">
              <span>Built with FastAPI</span>
              <span>•</span>
              <span>Next.js</span>
              <span>•</span>
              <span>Scikit-learn</span>
              <span>•</span>
              <span>TensorFlow</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

function StatCard({ number, label }) {
  return (
    <motion.div 
      className="text-center"
      whileHover={{ scale: 1.1, y: -5 }}
      transition={{ type: "spring", stiffness: 400 }}
    >
      <div className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
        {number}
      </div>
      <div className="text-sm text-purple-300 mt-1">{label}</div>
    </motion.div>
  )
}

function ModelStatus({ name, file, available, description }) {
  return (
    <motion.div 
      className="flex items-start gap-3 p-3 rounded-xl bg-white/5 hover:bg-white/10 transition-all group"
      whileHover={{ x: 5 }}
    >
      <div className={`flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center ${
        available 
          ? 'bg-green-500/20 text-green-400' 
          : 'bg-red-500/20 text-red-400'
      }`}>
        {available ? (
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        ) : (
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className="font-semibold text-white">{name}</div>
        <div className="text-xs text-purple-300 truncate">{file}</div>
        <div className="text-xs text-purple-400 mt-1">{description}</div>
      </div>
    </motion.div>
  )
}

function FeatureCard({ icon, title, description, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6, delay }}
      whileHover={{ scale: 1.05, y: -10 }}
      className="relative group"
    >
      <div className="relative p-6 rounded-2xl bg-white/10 backdrop-blur-md border border-white/20 hover:bg-white/15 transition-all h-full">
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-purple-500/0 via-pink-500/0 to-cyan-500/0 group-hover:from-purple-500/10 group-hover:via-pink-500/10 group-hover:to-cyan-500/10 transition-all"></div>
        
        <div className="relative z-10">
          <div className="text-4xl mb-4">{icon}</div>
          <h3 className="text-xl font-bold mb-2 text-white">{title}</h3>
          <p className="text-purple-200 text-sm leading-relaxed">{description}</p>
        </div>
      </div>
    </motion.div>
  )
}

function ModelArchCard({ title, models }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6 }}
      className="p-8 rounded-2xl bg-white/10 backdrop-blur-md border border-white/20"
    >
      <h3 className="text-2xl font-bold mb-6 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
        {title}
      </h3>
      <ul className="space-y-3">
        {models.map((model, idx) => (
          <motion.li 
            key={idx}
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: idx * 0.1 }}
            className="flex items-center gap-3 text-purple-200"
          >
            <svg className="w-5 h-5 text-cyan-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{model}</span>
          </motion.li>
        ))}
      </ul>
    </motion.div>
  )
}
