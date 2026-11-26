import { IconEye, IconSparkles, IconUpload } from "./Icons.jsx";

const ETAPES = [
  { Icone: IconUpload, titre: "Déposez une photo", texte: "Une surface de béton, en JPEG ou PNG." },
  { Icone: IconSparkles, titre: "Lancez l'analyse", texte: "Le modèle ConvNeXt-Tiny donne la probabilité de fissure." },
  { Icone: IconEye, titre: "Comprenez la décision", texte: "La carte Grad-CAM montre où le modèle a regardé." },
];

// Panneau affiché avant la première analyse.
export default function EmptyState() {
  return (
    <div className="flex h-full min-h-80 animate-fade-up flex-col justify-center gap-5 rounded-2xl border border-white/10 bg-white/[0.03] p-6 backdrop-blur">
      {ETAPES.map(({ Icone, titre, texte }, i) => (
        <div key={titre} className="flex items-start gap-4">
          <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-white/5 text-violet-300 ring-1 ring-white/10">
            <Icone className="h-5 w-5" />
          </div>
          <div>
            <p className="font-medium text-slate-100">
              <span className="mr-2 text-slate-500">{i + 1}.</span>
              {titre}
            </p>
            <p className="text-sm text-slate-400">{texte}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
