/**
 * Stocker Analytics mark — concentric neon hexagons around a green candlestick.
 * SVG recreation of the supplied logo; uses the brand/bull tokens so it themes
 * (green→teal in both light and dark). Swap in a raster from /public if desired.
 */
export function StockerLogo({ size = 36, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      className={className}
      role="img"
      aria-label="Stocker Analytics"
    >
      <defs>
        <linearGradient id="sk-candle" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgb(var(--bull))" />
          <stop offset="100%" stopColor="rgb(var(--brand))" />
        </linearGradient>
        <filter id="sk-glow" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="1.3" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <g filter="url(#sk-glow)" strokeLinejoin="round" fill="none">
        <polygon points="32,6 54.52,19 54.52,45 32,58 9.48,45 9.48,19" stroke="rgb(var(--brand))" strokeOpacity="0.3" strokeWidth="1.8" />
        <polygon points="32,11 50.19,21.5 50.19,42.5 32,53 13.81,42.5 13.81,21.5" stroke="rgb(var(--brand-2))" strokeOpacity="0.6" strokeWidth="1.8" />
        <polygon points="32,16 45.86,24 45.86,40 32,48 18.14,40 18.14,24" stroke="rgb(var(--brand))" strokeWidth="2" />
      </g>

      {/* candlestick: wick + body */}
      <line x1="32" y1="17" x2="32" y2="47" stroke="url(#sk-candle)" strokeWidth="2.4" strokeLinecap="round" />
      <rect x="27.4" y="23" width="9.2" height="18" rx="2.2" fill="url(#sk-candle)" />
    </svg>
  );
}
