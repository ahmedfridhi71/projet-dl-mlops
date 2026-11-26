import { IconSparkles, Spinner } from "./Icons.jsx";

// Bouton principal : dégradé bleu/violet, reflet animé, se soulève au survol.
export default function AnalyzeButton({ onClick, disabled, loading }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="group relative overflow-hidden rounded-xl bg-linear-to-r from-blue-600 via-indigo-600 to-violet-600 px-6 py-4 font-semibold text-white
        shadow-lg shadow-indigo-950/50 transition-all duration-300 ease-out
        hover:-translate-y-0.5 hover:shadow-xl hover:shadow-violet-600/30 hover:brightness-110
        active:translate-y-0 active:scale-[0.98]
        focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950
        disabled:pointer-events-none disabled:from-slate-800 disabled:via-slate-800 disabled:to-slate-800 disabled:text-slate-500 disabled:shadow-none"
    >
      {!disabled && (
        <span className="pointer-events-none absolute inset-y-0 left-0 w-1/4 animate-shine bg-linear-to-r from-transparent via-white/25 to-transparent" />
      )}
      <span className="relative flex items-center justify-center gap-2">
        {loading ? <Spinner /> : <IconSparkles className="h-5 w-5 transition-transform duration-300 group-hover:rotate-12 group-hover:scale-110" />}
        {loading ? "Analyse en cours…" : "Analyser"}
      </span>
    </button>
  );
}
