import { IconLogo } from "./Icons.jsx";

// En-tête : logo, titre et état du backend.
export default function Header({ online, metadata }) {
  const etat =
    online === null
      ? { dot: "bg-slate-400", ring: "border-white/10 text-slate-400", text: "Connexion…" }
      : online
        ? { dot: "bg-emerald-400", ring: "border-emerald-400/20 bg-emerald-400/10 text-emerald-300", text: "Backend connecté" }
        : { dot: "bg-rose-400", ring: "border-rose-400/20 bg-rose-400/10 text-rose-300", text: "Backend hors ligne" };

  return (
    <header className="sticky top-0 z-20 border-b border-white/5 bg-slate-950/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="grid h-10 w-10 place-items-center rounded-xl bg-linear-to-br from-blue-500 to-violet-600 text-white shadow-lg shadow-violet-600/30">
            <IconLogo className="h-5 w-5" />
          </div>
          <div className="leading-tight">
            <p className="font-semibold tracking-tight text-white">Détection de fissures</p>
            <p className="text-xs text-slate-400">
              {metadata ? `${metadata.model} · seuil ${metadata.threshold}` : "Béton · SDNET2018"}
            </p>
          </div>
        </div>

        <span className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium ${etat.ring}`}>
          <span className="relative flex h-2 w-2">
            {online && <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${etat.dot}`} />}
            <span className={`relative inline-flex h-2 w-2 rounded-full ${etat.dot}`} />
          </span>
          {etat.text}
        </span>
      </div>
    </header>
  );
}
