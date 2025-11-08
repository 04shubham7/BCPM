import Head from 'next/head'
import Link from 'next/link'
import { useState } from 'react'
import VideoEmbed from '../components/VideoEmbed'
import SiteFooter from '../components/SiteFooter'
import { apiUrl } from '../lib/api'

export default function Learn(){
  const [lang, setLang] = useState('en')
  const [openFaq, setOpenFaq] = useState(null)

  const T = translations[lang]

  return (
    <div className="min-h-screen">
      <Head>
        <title>Getting Started — Understand Breast Cancer</title>
        <meta name="description" content="A friendly, visual guide to breast cancer for everyone." />
      </Head>

      <header className="flex items-center justify-between px-8 py-6 bg-purple-900/20 backdrop-blur-md border-b border-purple-500/20">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-purple-600 to-violet-600 flex items-center justify-center text-white font-bold shadow-lg shadow-purple-500/30">
            BC
          </div>
          <div>
            <div className="font-semibold text-lg text-purple-100">BreastAI</div>
            <div className="text-sm text-purple-300/70">Learn • Prevent • Act</div>
          </div>
        </div>
        <nav className="flex items-center gap-4">
          <Link href='/' className="text-purple-200 hover:text-purple-100 transition-colors">Home</Link>
          <Link href='/learn' className="text-purple-100 font-semibold">Getting Started</Link>
          <Link href='/demo' className="px-4 py-2 rounded-lg bg-gradient-to-r from-purple-600 to-violet-600 text-white shadow-lg shadow-purple-500/30 hover:shadow-xl transition-all">
            Try the Demo
          </Link>
        </nav>
      </header>

      <main className="max-w-6xl mx-auto p-6 space-y-10">
  <section className="rounded-2xl p-6 glass-card shadow-2xl shadow-purple-900/20 overflow-hidden fade-in-up">
          <div className="grid md:grid-cols-2 gap-6 items-center">
            <div>
              <h1 className="text-4xl md:text-5xl font-extrabold leading-tight bg-gradient-to-r from-purple-300 via-violet-300 to-fuchsia-300 bg-clip-text text-transparent">
                {T.title}
              </h1>
              <p className="text-purple-200/80 mt-4" dangerouslySetInnerHTML={{__html: T.intro}} />
              <div className="mt-4 text-sm text-purple-300/70" dangerouslySetInnerHTML={{__html: T.subintro}} />
              <LanguageToggle lang={lang} setLang={setLang} />
            </div>
            <div className="relative">
              <div className="aspect-[4/3] w-full rounded-2xl bg-gradient-to-br from-purple-900/40 to-purple-800/30 border border-purple-500/30 p-4 flex items-center justify-center">
                <div className="grid grid-cols-2 gap-3 w-full">
                  <DiagramCard title="Healthy Cell" color="from-green-500/70 to-emerald-500/60" note="Grows normally" emoji="🧬"/>
                  <DiagramCard title="Abnormal Cell" color="from-yellow-500/70 to-amber-500/60" note="Grows differently" emoji="🧫"/>
                  <DiagramCard title="Benign Lump" color="from-blue-500/70 to-sky-500/60" note="Does not spread" emoji="🫧"/>
                  <DiagramCard title="Malignant Tumor" color="from-red-500/70 to-rose-500/60" note="Can spread" emoji="⚠️"/>
                </div>
              </div>
              <div className="absolute -bottom-3 right-4 text-xs text-purple-300/60">Illustrative diagram</div>
            </div>
          </div>
        </section>

        {/* Getting Started Videos */}
        <section className="rounded-2xl p-6 glass-card shadow-2xl shadow-purple-900/20 fade-in-up delay-1">
          <h2 className="text-2xl font-semibold text-purple-100 mb-4">Getting Started — Watch & Learn</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <VideoEmbed
              videoId="Y9Q9b_RtbXc"
              title="Breast Cancer Basics: Understand the Fundamentals"
              accent="rose"
            />
            <VideoEmbed
              videoId="-ygucOBbKJA"
              title="Self-Check & Early Detection — Practical Guide"
              accent="sky"
            />
          </div>
          <p className="text-xs text-purple-300/70 mt-3">Videos open in-page using YouTube’s privacy-enhanced mode.</p>
        </section>

  <section className="rounded-2xl p-6 glass-card shadow-2xl shadow-purple-900/20 fade-in-up delay-1">
          <h2 className="text-2xl font-semibold text-purple-100 mb-4">{T.symptomsHeading}</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            <SymptomCard title="New lump in the breast or underarm" icon="🫱"/>
            <SymptomCard title="Thickening or swelling of part of the breast" icon="💧"/>
            <SymptomCard title="Irritation or dimpling of breast skin" icon="🟣"/>
            <SymptomCard title="Redness or flaky skin in the nipple area" icon="🎯"/>
            <SymptomCard title="Pulling in of the nipple" icon="↩️"/>
            <SymptomCard title="Any change in size or shape" icon="📏"/>
          </div>
          <p className="text-xs text-purple-300/70 mt-3" dangerouslySetInnerHTML={{__html: T.symptomNote}} />
        </section>

  <section className="rounded-2xl p-6 glass-card shadow-2xl shadow-purple-900/20 fade-in-up delay-2">
          <h2 className="text-2xl font-semibold text-purple-100 mb-4">{T.screeningHeading}</h2>
          <div className="grid md:grid-cols-3 gap-4">
            <StepCard n={1} title="Self-Awareness" desc="Know your normal. If anything changes, talk to a doctor."/>
            <StepCard n={2} title="Clinical Exam" desc="A health worker checks and feels for changes."/>
            <StepCard n={3} title="Imaging" desc="Tests like mammogram, ultrasound, or MRI help see inside."/>
          </div>
          <div className="mt-4 text-purple-200/80 text-sm" dangerouslySetInnerHTML={{__html: T.screeningNote}} />
        </section>

  <section className="rounded-2xl p-6 glass-card shadow-2xl shadow-purple-900/20 fade-in-up delay-3">
          <h2 className="text-2xl font-semibold text-purple-100 mb-4">{T.riskHeading}</h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {riskItems[lang].map((r, idx) => (
              <TipCard key={idx} title={r.title} desc={r.desc} emoji={r.emoji} />
            ))}
          </div>
        </section>

        <section className="rounded-2xl p-6 glass-card shadow-2xl shadow-purple-900/20">
          <h2 className="text-2xl font-semibold text-purple-100 mb-4">{T.appHeading}</h2>
          <div className="grid md:grid-cols-2 gap-6 items-center">
            <div className="space-y-3 text-purple-200/80">
              <p dangerouslySetInnerHTML={{__html: T.appIntro}} />
              <ul className="list-disc list-inside space-y-1 text-sm text-purple-200/70">
                <li>It takes numbers (from tests) as input.</li>
                <li>It outputs a suggestion: likely benign or likely malignant.</li>
                <li>It generates a clear, printable PDF report with simple charts.</li>
              </ul>
              <p className="text-xs text-purple-300/60" dangerouslySetInnerHTML={{__html: T.disclaimer}} />
            </div>
            <div className="rounded-2xl bg-purple-900/30 border border-purple-500/30 p-4">
              <div className="grid grid-cols-2 gap-3">
                <ChartBlock title="Benign vs Malignant" percent={62} />
                <ChartBlock title="Early Detection Saves Lives" percent={85} />
                <ChartBlock title="Confidence Score" percent={78} />
                <ChartBlock title="Awareness Level" percent={45} />
              </div>
            </div>
          </div>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link href='/demo' className="inline-block px-5 py-3 rounded-xl bg-gradient-to-r from-purple-600 to-violet-600 text-white font-semibold shadow-lg shadow-purple-500/40 hover:shadow-xl transition-all">
              Try the Demo
            </Link>
            <a href="#resources" className="inline-block px-5 py-3 rounded-xl border-2 border-purple-500/40 text-purple-200 hover:bg-purple-800/30 transition-all">
              Jump to Resources
            </a>
            <DownloadAwarenessButton lang={lang} />
          </div>
        </section>

        <section id="faq" className="rounded-2xl p-6 glass-card shadow-2xl shadow-purple-900/20">
          <h2 className="text-2xl font-semibold text-purple-100 mb-4">{T.faqHeading}</h2>
          <div className="space-y-3">
            {faqMap[lang].map((f, idx) => (
              <div key={idx} className="rounded-xl border border-purple-500/30 bg-purple-900/30">
                <button onClick={() => setOpenFaq(openFaq === idx ? null : idx)} className="w-full text-left px-4 py-3 flex items-center justify-between">
                  <span className="font-medium text-purple-100">{f.q}</span>
                  <span className="text-purple-300">{openFaq === idx ? '−' : '+'}</span>
                </button>
                {openFaq === idx && (
                  <div className="px-4 pb-4 text-sm text-purple-200/80" dangerouslySetInnerHTML={{__html: f.a}} />
                )}
              </div>
            ))}
          </div>
        </section>

        <section id="resources" className="rounded-2xl p-6 glass-card shadow-2xl shadow-purple-900/20">
          <h2 className="text-2xl font-semibold text-purple-100 mb-4">{T.resourcesHeading}</h2>
          <ul className="space-y-2 text-purple-200/80">
            <li><a className="text-purple-300 hover:text-purple-200 underline" href="https://www.who.int/news-room/fact-sheets/detail/breast-cancer" target="_blank" rel="noreferrer">WHO: Breast Cancer Facts</a></li>
            <li><a className="text-purple-300 hover:text-purple-200 underline" href="https://www.cancer.gov/types/breast" target="_blank" rel="noreferrer">NCI: Types of Breast Cancer</a></li>
            <li><a className="text-purple-300 hover:text-purple-200 underline" href="https://www.cdc.gov/cancer/breast/basic_info/index.htm" target="_blank" rel="noreferrer">CDC: Basics About Breast Cancer</a></li>
          </ul>
          <ShareBlock />
        </section>
      </main>

      <SiteFooter />
    </div>
  )
}

function DiagramCard({title, color, note, emoji}){
  return (
    <div className={`rounded-xl p-4 bg-gradient-to-br ${color} text-white border border-white/20`}> 
      <div className="text-xl font-semibold mb-1 flex items-center gap-2"><span>{emoji}</span>{title}</div>
      <div className="text-sm opacity-90">{note}</div>
    </div>
  )
}

function SymptomCard({title, icon}){
  return (
    <div className="rounded-xl p-4 bg-purple-900/30 border border-purple-500/30">
      <div className="flex items-start gap-3">
        <div className="text-purple-200">
          <SymptomIcon icon={icon} />
        </div>
        <div className="text-purple-100 text-sm">{title}</div>
      </div>
    </div>
  )
}

function StepCard({n, title, desc}){
  return (
    <div className="rounded-xl p-4 bg-purple-900/30 border border-purple-500/30">
      <div className="flex items-start gap-3">
        <div className="mt-0.5"><StepIcon n={n} /></div>
        <div>
          <div className="text-xs text-purple-300/70">STEP {n}</div>
          <div className="text-purple-100 font-semibold">{title}</div>
          <div className="text-sm text-purple-200/70 mt-1">{desc}</div>
        </div>
      </div>
    </div>
  )
}

function TipCard({title, desc, emoji}){
  return (
    <div className="rounded-xl p-4 bg-purple-900/30 border border-purple-500/30">
      <div className="text-lg text-purple-100 font-semibold flex items-center gap-2"><span>{emoji}</span>{title}</div>
      <div className="text-sm text-purple-200/70 mt-1">{desc}</div>
    </div>
  )
}

function ChartBlock({title, percent}){
  return (
    <div className="rounded-xl p-4 bg-purple-950/40 border border-purple-500/30">
      <div className="text-sm text-purple-200/80 mb-2">{title}</div>
      <div className="h-3 rounded-full bg-purple-900/40 overflow-hidden border border-purple-500/30">
        <div className="h-full bg-gradient-to-r from-purple-500 to-violet-600" style={{width: `${percent}%`}} />
      </div>
      <div className="text-xs text-purple-300/60 mt-1">{percent}%</div>
    </div>
  )
}

function SymptomIcon({icon}){
  // Map to a minimal set of inline SVGs for clarity in print and screen
  switch(icon){
    case '🫱': // lump
      return (
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 21c-4.5 0-8-3.5-8-8 0-2.2.9-4.2 2.3-5.7 1.5-1.4 3.5-2.3 5.7-2.3 4.5 0 8 3.5 8 8 0 2.2-.9 4.2-2.3 5.7"/>
          <circle cx="12" cy="13" r="3" />
        </svg>
      )
    case '💧': // swelling
      return (
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 3s6 6.4 6 10a6 6 0 1 1-12 0c0-3.6 6-10 6-10z"/>
        </svg>
      )
    case '🟣': // skin dimpling
      return (
        <svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor">
          <circle cx="6" cy="12" r="2"/>
          <circle cx="12" cy="12" r="2"/>
          <circle cx="18" cy="12" r="2"/>
        </svg>
      )
    case '🎯': // nipple area
      return (
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
          <circle cx="12" cy="12" r="7"/>
          <circle cx="12" cy="12" r="3"/>
        </svg>
      )
    case '↩️': // pulling inward
      return (
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M9 10l-4 4 4 4"/>
          <path d="M20 12H6"/>
        </svg>
      )
    case '📏': // size/shape change
      return (
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M3 8h18"/>
          <path d="M3 12h10"/>
          <path d="M3 16h6"/>
        </svg>
      )
    default:
      return <span className="text-2xl">{icon}</span>
  }
}

function StepIcon({n}){
  // Simple numbered circle icon
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9"/>
      <text x="12" y="16" textAnchor="middle" fontSize="10" fill="currentColor">{n}</text>
    </svg>
  )
}

function LanguageToggle({lang, setLang}){
  const Btn = ({code, label}) => (
    <button onClick={() => setLang(code)} className={`text-xs px-2.5 py-1 rounded-lg border ${lang===code? 'bg-purple-600/40 border-purple-400/50 text-purple-50':'bg-purple-900/30 border-purple-500/30 text-purple-200 hover:bg-purple-800/40'}`}>
      {label}
    </button>
  )
  return (
    <div className="mt-4 flex items-center gap-2">
      <span className="text-xs text-purple-300/70">Language:</span>
      <Btn code='en' label='English' />
      <Btn code='hi' label='हिंदी' />
      <Btn code='mr' label='मराठी' />
      <button onClick={() => window.print()} className="ml-auto text-xs px-2.5 py-1 rounded-lg border bg-purple-900/30 border-purple-500/30 text-purple-200 hover:bg-purple-800/40 no-print">Print</button>
    </div>
  )
}

function DownloadAwarenessButton({lang}){
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const doDownload = async () => {
    setError(null)
    setLoading(true)
    try {
      const res = await fetch(apiUrl(`awareness?lang=${lang}`))
      if(!res.ok){
        throw new Error(`Download failed (${res.status})`)
      }
      const blob = await res.blob()
      if(blob.size === 0){
        throw new Error('Empty PDF response')
      }
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `breast_cancer_guide_${lang}.pdf`
      document.body.appendChild(a)
      a.click()
      a.remove()
      setTimeout(()=>URL.revokeObjectURL(url), 30000)
    } catch(e){
      console.error(e)
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }
  return (
    <div className="relative">
      <button disabled={loading} onClick={doDownload} className={`inline-block px-5 py-3 rounded-xl border-2 border-purple-500/40 text-purple-200 transition-all ${loading? 'opacity-60 cursor-not-allowed':'hover:bg-purple-800/30'}`}>
        {loading ? 'Preparing PDF…' : 'Download as PDF'}
      </button>
      {error && <div className="mt-2 text-xs text-red-300">{error}</div>}
    </div>
  )
}

function ShareBlock(){
  const shareUrl = typeof window !== 'undefined' ? window.location.href : 'http://localhost:3000/learn'
  const copy = async () => {
    try { await navigator.clipboard.writeText(shareUrl); alert('Link copied!') } catch(e) {}
  }
  const btn = 'px-3 py-2 rounded-lg border border-purple-500/30 bg-purple-900/30 hover:bg-purple-800/40 text-sm text-purple-100'
  return (
    <div className="mt-6 flex flex-wrap gap-2 items-center no-print">
      <span className="text-sm text-purple-300/80 mr-2">Share:</span>
      <button className={btn} onClick={copy}>Copy Link</button>
      <a className={btn} target="_blank" rel="noreferrer" href={`https://twitter.com/intent/tweet?url=${encodeURIComponent(shareUrl)}&text=${encodeURIComponent('Learn the basics of breast cancer in a simple visual guide')}`}>X/Twitter</a>
      <a className={btn} target="_blank" rel="noreferrer" href={`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`}>Facebook</a>
      <a className={btn} target="_blank" rel="noreferrer" href={`https://wa.me/?text=${encodeURIComponent('Learn about breast cancer: ' + shareUrl)}`}>WhatsApp</a>
    </div>
  )
}

const translations = {
  en: {
    title: 'What is Breast Cancer?',
    intro: 'Breast cancer happens when some cells in the breast grow faster than they should and form a lump (called a tumor). Some tumors are <span class="text-green-300">benign (not dangerous)</span>. Some are <span class="text-red-300">malignant (can spread)</span>. Early finding saves lives.',
    subintro: 'This page explains the basics in simple language with visuals so anyone can understand. Share it with family and friends.',
    symptomsHeading: 'Common Signs & Symptoms',
    symptomNote: 'Note: These signs can have other causes. Only a doctor can tell for sure.',
    screeningHeading: 'How Screening Works',
    screeningNote: 'Screening does not prevent cancer, but it helps find it early when it’s easier to treat.',
    riskHeading: 'Reducing Risk',
    appHeading: 'How This App Helps',
    appIntro: 'Our demo shows how AI can support doctors by providing a second opinion on data. It does not replace a doctor.',
    disclaimer: 'Disclaimer: For education only. Not medical advice.',
    faqHeading: 'Frequently Asked Questions',
    resourcesHeading: 'Helpful Resources',
  },
  hi: {
    title: 'स्तन कैंसर क्या है?',
    intro: 'स्तन के कुछ कोशिकाएँ सामान्य से तेज़ बढ़ने लगती हैं और एक गांठ बना सकती हैं (ट्यूमर)। कुछ ट्यूमर <span class="text-green-300">सौम्य (खतरनाक नहीं)</span> होते हैं। कुछ <span class="text-red-300">घातक (फैल सकते हैं)</span> होते हैं। जल्दी पता चलना जीवन बचाता है।',
    subintro: 'यह पृष्ठ सरल भाषा और चित्रों के साथ मूल बातें समझाता है। इसे परिवार और दोस्तों के साथ साझा करें।',
    symptomsHeading: 'सामान्य संकेत और लक्षण',
    symptomNote: 'नोट: इन संकेतों के अन्य कारण भी हो सकते हैं। निश्चित रूप से डॉक्टर ही बता सकते हैं।',
    screeningHeading: 'जांच (स्क्रीनिंग) कैसे होती है',
    screeningNote: 'स्क्रीनिंग कैंसर को रोकती नहीं है, लेकिन उसे जल्दी ढूंढने में मदद करती है जब इलाज आसान होता है।',
    riskHeading: 'जोखिम कैसे कम करें',
    appHeading: 'यह ऐप कैसे मदद करता है',
    appIntro: 'यह डेमो दिखाता है कि एआई डेटा के आधार पर डॉक्टरों को दूसरा मत देने में कैसे मदद कर सकता है। यह डॉक्टर का स्थान नहीं लेता।',
    disclaimer: 'अस्वीकरण: केवल शिक्षा के उद्देश्य से। यह चिकित्सीय सलाह नहीं है।',
    faqHeading: 'अक्सर पूछे जाने वाले प्रश्न',
    resourcesHeading: 'सहायक संसाधन',
  },
  mr: {
    title: 'स्तनाचा कर्करोग म्हणजे काय?',
    intro: 'स्तनातील काही पेशी सामान्यपेक्षा जलद वाढतात आणि गाठ (ट्यूमर) तयार होऊ शकते. काही ट्यूमर <span class="text-green-300">सौम्य (धोकादायक नाही)</span> असतात. काही <span class="text-red-300">घातक (पसरू शकतात)</span> असतात. लवकर निदान जीव वाचवते.',
    subintro: 'ही पान सोप्या भाषेत आणि चित्रांसह माहिती देते. कुटुंबीय व मित्रांसोबत शेअर करा.',
    symptomsHeading: 'सामान्य लक्षणे',
    symptomNote: 'टीप: या लक्षणांचे इतर कारणेही असू शकतात. निश्चित सांगू शकतो तो डॉक्टरच.',
    screeningHeading: 'तपासणी (स्क्रीनिंग) कशी होते',
    screeningNote: 'स्क्रीनिंग कर्करोग रोखत नाही; परंतु तो लवकर शोधण्यास मदत करते जेव्हा उपचार सोपे असतात.',
    riskHeading: 'जोखीम कशी कमी करावी',
    appHeading: 'हा ॲप कशी मदत करतो',
    appIntro: 'हा डेमो दाखवतो की एआय डॉक्टरांना डेटावर आधारित दुसरे मत देण्यात कशी मदत करू शकते. तो डॉक्टरची जागा घेणार नाही.',
    disclaimer: 'सूचना: ही शैक्षणिक माहिती आहे. वैद्यकीय सल्ला नाही.',
    faqHeading: 'नेहमी विचारले जाणारे प्रश्न',
    resourcesHeading: 'उपयुक्त स्त्रोत',
  }
}

const faqMap = {
  en: [
    { q: 'Does a lump always mean cancer?', a: 'No. Many lumps are benign. But any new lump should be checked by a healthcare professional.' },
    { q: 'Can men get breast cancer?', a: 'Yes, men can get breast cancer too, though it is less common than in women.' },
    { q: 'What age should screening start?', a: 'It depends on national guidelines and personal risk. Talk to your doctor about when to begin and how often.' },
    { q: 'Does screening hurt?', a: 'Some tests (like mammograms) can be uncomfortable but are quick. The benefits of early detection are significant.' },
  ],
  hi: [
    { q: 'क्या हर गांठ कैंसर होती है?', a: 'नहीं। कई गांठें सौम्य होती हैं। फिर भी नई गांठ दिखे तो स्वास्थ्यकर्मी से जाँच कराएँ।' },
    { q: 'क्या पुरुषों को भी स्तन कैंसर हो सकता है?', a: 'हाँ, पुरुषों में भी हो सकता है; बस महिलाओं की तुलना में कम होता है।' },
    { q: 'स्क्रीनिंग कब शुरू करनी चाहिए?', a: 'यह दिशानिर्देश और व्यक्तिगत जोखिम पर निर्भर करता है। अपने डॉक्टर से समय और आवृत्ति पूछें।' },
    { q: 'क्या स्क्रीनिंग में दर्द होता है?', a: 'कुछ जाँचें (जैसे मैमोग्राम) असुविधाजनक हो सकती हैं, पर जल्दी खत्म हो जाती हैं। जल्दी पता चलने के फायदे महत्वपूर्ण हैं।' },
  ],
  mr: [
    { q: 'प्रत्येक गाठ कर्करोग असते का?', a: 'नाही. अनेक गाठी सौम्य असतात. तरीही नवी गाठ दिसल्यास आरोग्यतज्ज्ञांकडून तपासणी करा.' },
    { q: 'पुरुषांनाही स्तनाचा कर्करोग होतो का?', a: 'हो, पुरुषांमध्येही होऊ शकतो; पण महिलांच्या तुलनेत कमी सामान्य आहे.' },
    { q: 'स्क्रीनिंग कधी सुरू करावी?', a: 'हे मार्गदर्शक तत्त्वे आणि वैयक्तिक जोखमीवर अवलंबून असते. आपल्या डॉक्टरांना विचारा.' },
    { q: 'स्क्रीनिंगमध्ये वेदना होतात का?', a: 'काही चाचण्या (उदा. मॅमोग्रॅम) अस्वस्थ वाटू शकतात; पण लवकर होतात. लवकर निदानाचे फायदे महत्वाचे आहेत.' },
  ],
}

const riskItems = {
  en: [
    { title: 'Stay Active', desc: 'Aim for regular movement each day.', emoji: '🚶' },
    { title: 'Eat Balanced', desc: 'More fruits, veggies, and whole foods.', emoji: '🥗' },
    { title: 'Limit Alcohol', desc: 'Less alcohol reduces risk.', emoji: '🥂' },
    { title: 'Don’t Smoke', desc: 'Quitting helps your whole body.', emoji: '🚭' },
  ],
  hi: [
    { title: 'सक्रिय रहें', desc: 'हर दिन थोड़ा चलना-फिरना शामिल करें।', emoji: '🚶' },
    { title: 'संतुलित भोजन', desc: 'फल, सब्ज़ियाँ और संपूर्ण आहार अधिक लें।', emoji: '🥗' },
    { title: 'शराब सीमित करें', desc: 'कम शराब से जोखिम घटता है।', emoji: '🥂' },
    { title: 'धूम्रपान न करें', desc: 'छोड़ने से पूरे शरीर को फायदा।', emoji: '🚭' },
  ],
  mr: [
    { title: 'सक्रिय राहा', desc: 'दररोज थोडी हालचाल करा.', emoji: '🚶' },
    { title: 'संतुलित आहार', desc: 'फळे, भाजीपाला आणि पौष्टिक अन्न अधिक घ्या.', emoji: '🥗' },
    { title: 'मद्यपान कमी करा', desc: 'कमी मद्यपानाने जोखीम घटते.', emoji: '🥂' },
    { title: 'धूम्रपान टाळा', desc: 'सोडल्याने सर्वांगीण आरोग्य सुधारते.', emoji: '🚭' },
  ],
}
