import { IconEye, IconInfo } from "./Icons.jsx";

// Carte Grad-CAM : zones rouges = zones qui ont le plus poussé vers « fissure ».
export default function GradCamView({ status, overlay, error }) {
  return (
    <div className="animate-fade-up rounded-2xl border border-white/10 bg-white/[0.03] p-6 backdrop-blur [animation-delay:120ms]">
      <div className="flex items-center gap-2">
        <IconEye className="h-5 w-5 text-violet-300" />
        <p className="font-semibold text-slate-100">Grad-CAM</p>
        <p className="text-sm text-slate-500">· où le modèle a regardé</p>
      </div>

      {status === "loading" && (
        <div className="mt-5 flex flex-col items-center gap-3">
          <div className="aspect-square w-full max-w-64 animate-shimmer rounded-xl bg-[linear-gradient(110deg,rgba(255,255,255,0.03)_30%,rgba(139,92,246,0.15)_50%,rgba(255,255,255,0.03)_70%)] bg-[length:200%_100%] ring-1 ring-white/10" />
          <p className="text-sm text-slate-400">Calcul de la carte… (plus long la première fois)</p>
        </div>
      )}

      {status === "done" && (
        <div className="mt-5 flex flex-col items-center">
          <img
            src={`data:image/png;base64,${overlay}`}
            alt="Carte Grad-CAM superposée à l'image"
            className="aspect-square w-full max-w-64 animate-pop rounded-xl shadow-2xl shadow-black/50 ring-1 ring-white/10"
          />
          <div className="mt-4 w-full max-w-64">
            <div className="h-1.5 rounded-full bg-[linear-gradient(to_right,#00f,#0ff,#ff0,#f00)]" />
            <div className="mt-1.5 flex justify-between text-[11px] text-slate-500">
              <span>peu important</span>
              <span>très important</span>
            </div>
          </div>
          <p className="mt-4 flex gap-2 text-xs leading-relaxed text-slate-500">
            <IconInfo className="h-4 w-4 shrink-0" />
            Carte relative : il y a toujours une zone « chaude », même sans fissure. Elle montre où
            le modèle a regardé, pas la preuve d'une fissure.
          </p>
        </div>
      )}

      {status === "error" && (
        <p className="mt-4 flex gap-2 text-sm text-slate-400">
          <IconInfo className="h-5 w-5 shrink-0 text-slate-500" />
          Grad-CAM indisponible : {error}
        </p>
      )}
    </div>
  );
}
