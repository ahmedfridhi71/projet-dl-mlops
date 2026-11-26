// Résultat : label, probabilité de fissure et barre de confiance avec le seuil.
export default function ResultCard({ result }) {
  const fissure = result.label === "fissure";
  const pct = result.probability * 100;
  const seuilPct = result.threshold * 100;

  return (
    <div className={`rounded-xl border p-5 ${fissure ? "border-red-200 bg-red-50" : "border-emerald-200 bg-emerald-50"}`}>
      <p className="text-sm font-medium text-slate-500">Résultat</p>
      <p className={`mt-1 text-3xl font-bold ${fissure ? "text-red-700" : "text-emerald-700"}`}>
        {fissure ? "Fissure détectée" : "Sans fissure"}
      </p>

      <div className="mt-5">
        <div className="mb-1 flex justify-between text-sm text-slate-600">
          <span>Probabilité de fissure</span>
          <span className="font-semibold tabular-nums">{pct.toFixed(1)} %</span>
        </div>
        {/* Barre 0-100 % ; le trait noir marque le seuil de décision. */}
        <div className="relative h-3 rounded-full bg-slate-200">
          <div
            className={`h-3 rounded-full ${fissure ? "bg-red-500" : "bg-emerald-500"}`}
            style={{ width: `${Math.max(pct, 0.5)}%` }}
          />
          <div className="absolute -top-1 h-5 w-0.5 bg-slate-900" style={{ left: `${seuilPct}%` }} title="Seuil" />
        </div>
        <p className="mt-2 text-xs text-slate-500">
          Fissure si la probabilité ≥ {seuilPct.toFixed(0)} % (trait noir). Seuil bas choisi
          pour rater le moins de fissures possible.
        </p>
      </div>

      <p className="mt-4 text-xs text-slate-400">Inférence ONNX : {result.inference_ms.toFixed(1)} ms</p>
    </div>
  );
}
