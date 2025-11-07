export default function VideoEmbed({ videoId, title, accent = 'purple' }){
  // Accent styles mapped to Tailwind classes (kept explicit for safelisting by Tailwind)
  const accents = {
    purple: {
      border: 'border-purple-400/40',
      shadow: 'shadow-purple-900/30',
      ring: 'ring-purple-500/20',
    },
    rose: {
      border: 'border-rose-400/40',
      shadow: 'shadow-rose-900/30',
      ring: 'ring-rose-500/20',
    },
    sky: {
      border: 'border-sky-400/40',
      shadow: 'shadow-sky-900/30',
      ring: 'ring-sky-500/20',
    },
    emerald: {
      border: 'border-emerald-400/40',
      shadow: 'shadow-emerald-900/30',
      ring: 'ring-emerald-500/20',
    },
  }
  const a = accents[accent] || accents.purple

  const src = `https://www.youtube-nocookie.com/embed/${videoId}?modestbranding=1&rel=0&color=white`

  return (
    <div className={`rounded-2xl overflow-hidden border ${a.border} bg-purple-950/30 ${a.shadow} ring-1 ${a.ring}`}> 
      <div className="video-aspect">
        {/* External link button overlay */}
        <a
          href={`https://www.youtube.com/watch?v=${videoId}`}
          target="_blank"
          rel="noreferrer"
          aria-label="Open on YouTube"
          className="absolute top-2 right-2 z-10 inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-purple-900/70 hover:bg-purple-800/80 border border-purple-500/40 text-[12px] text-purple-100 backdrop-blur-sm no-print"
        >
          <YouTubeIcon />
          <span className="hidden sm:inline">Open on YouTube</span>
        </a>
        <iframe
          src={src}
          title={title}
          loading="lazy"
          referrerPolicy="strict-origin-when-cross-origin"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          allowFullScreen
        />
      </div>
      {title && (
        <div className="px-4 py-3 text-sm text-purple-100/90 bg-gradient-to-r from-purple-950/60 to-transparent border-t border-purple-500/20">
          {title}
        </div>
      )}
    </div>
  )
}

function YouTubeIcon(){
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M22.54 6.42a2.78 2.78 0 0 0-1.95-2C18.88 4 12 4 12 4s-6.88 0-8.59.42a2.78 2.78 0 0 0-1.95 2A29 29 0 0 0 1 12a29 29 0 0 0 .46 5.58 2.78 2.78 0 0 0 1.95 2C5.12 20 12 20 12 20s6.88 0 8.59-.42a2.78 2.78 0 0 0 1.95-2A29 29 0 0 0 23 12a29 29 0 0 0-.46-5.58z"/>
      <path d="M10 15l5-3-5-3v6z" fill="currentColor" stroke="none"/>
    </svg>
  )
}
