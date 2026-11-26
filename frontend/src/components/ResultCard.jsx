import { useEffect, useState } from "react";
import { IconAlert, IconClock, IconCube, IconShieldCheck } from "./Icons.jsx";

// Fait monter un nombre de 0 à sa valeur (animation du pourcentage).
function useCountUp(cible, duree = 900) {
  const [valeur, setValeur] = useState(0);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return setValeur(cible);
    let raf;
    const debut = performance.now();
    const pas = (t) => {
      const k = Math.min((t - debut) / duree, 1);
      setValeur(cible * (1 - Math.pow(1 - k, 3))); // ralentit à la fin
      if (k < 1) raf = requestAnimationFrame(pas);
    };
    raf = requestAnimationFrame(pas);
    return () => cancelAnimationFrame(raf);
  }, [cible, duree]);
  return valeur;
}

// Résultat : label, probabilité de fissure et barre de confiance avec le seuil.
export default function ResultCard({ result }) {
  const fissure = result.label === "fissure";
  const pct = result.probability * 100;
  const seuilPct = result.threshold * 100;
  const affiche = useCountUp(pct);

  const t = fissure
    ? {
        carte: "border-rose-500/20 from-rose-500/10",
        icone: "bg-rose-500/15 text-rose-300 ring-rose-400/30",
        titre: "text-rose-200",
        barre: "from-orange-400 via-rose-500 to-fuchsia-500",
        Icone: IconAlert,
        titreTexte: "Fissure détectée",
        sousTitre: "Le modèle recommande une inspection de cette surface.",
      }
    : {
        carte: "border-emerald-500/20 from-emerald-500/10",
        icone: "bg-emerald-500/15 text-emerald-300 ring-emerald-400/30",
        titre: "text-emerald-200",
        barre: "from-emerald-400 to-teal-400",
        Icone: IconShieldCheck,
        titreTexte: "Sans fissure",
        sousTitre: "Aucune fissure détectée au seuil choisi.",
      };

  return (
    <div className={`relative animate-fade-up overflow-hidden rounded-2xl border bg-linear-to-br to-transparent p-6 backdrop-blur ${t.carte}`}>
      <div className="flex items-start gap-4">
        <div className={`grid h-12 w-12 shrink-0 animate-pop place-items-center rounded-xl ring-1 ${t.icone}`}>
          <t.Icone className="h-6 w-6" />
        </div>
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">Résultat</p>
          <p className={`mt-0.5 text-2xl font-bold tracking-tight ${t.titre}`}>{t.titreTexte}</p>
          <p className="mt-1 text-sm text-slate-400">{t.sousTitre}</p>
        </div>
      </div>

      <div className="mt-7">
        <div className="flex items-end justify-between">
          <span className="text-sm text-slate-400">Probabilité de fissure</span>
          <span className="text-4xl font-extrabold tabular-nums tracking-tight text-white">
            {affiche.toFixed(1)}
            <span className="ml-0.5 text-xl font-semibold text-slate-400">%</span>
          </span>
        </div>

        {/* Barre 0-100 % ; le repère blanc marque le seuil de décision. */}
        <div className="relative mt-6">
          <span
            className="absolute -top-5 -translate-x-1/2 whitespace-nowrap text-[11px] font-medium text-slate-300"
            style={{ left: `${seuilPct}%` }}
          >
            seuil {seuilPct.toFixed(0)} %
          </span>
          <div className="h-3 overflow-hidden rounded-full bg-white/5 ring-1 ring-white/10">
            <div
              className={`h-full animate-grow rounded-full bg-linear-to-r ${t.barre} shadow-[0_0_16px] shadow-current`}
              style={{ width: `${Math.max(pct, 1)}%` }}
            />
          </div>
          <div
            className="absolute -top-1 h-5 w-0.5 -translate-x-1/2 rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)]"
            style={{ left: `${seuilPct}%` }}
          />
          <div className="mt-1.5 flex justify-between text-[11px] text-slate-500">
            <span>0 %</span>
            <span>100 %</span>
          </div>
        </div>

        <p className="mt-3 text-xs leading-relaxed text-slate-500">
          Fissure si la probabilité ≥ {seuilPct.toFixed(0)} %. Seuil bas choisi sur la validation
          pour rater le moins de fissures possible.
        </p>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-3 border-t border-white/5 pt-5 text-sm">
        <div className="flex items-center gap-2 text-slate-400">
          <IconClock className="h-4 w-4 text-slate-500" />
          Inférence <span className="font-semibold tabular-nums text-slate-200">{result.inference_ms.toFixed(1)} ms</span>
        </div>
        <div className="flex items-center gap-2 text-slate-400">
          <IconCube className="h-4 w-4 text-slate-500" />
          Moteur <span className="font-semibold text-slate-200">ONNX Runtime</span>
        </div>
      </div>
    </div>
  );
}
