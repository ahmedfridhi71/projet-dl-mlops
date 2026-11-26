// En-tête : titre, modèle utilisé et état du backend.
export default function Header({ online, metadata }) {
  const etat = {
    null: ["bg-slate-300", "Connexion…"],
    true: ["bg-emerald-500", "Backend connecté"],
    false: ["bg-red-500", "Backend hors ligne"],
  }[online];

  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-6 py-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-900">Détection de fissures du béton</h1>
          <p className="text-sm text-slate-500">
            {metadata
              ? `Modèle ${metadata.model} · seuil ${metadata.threshold}`
              : "Classification d'images SDNET2018"}
          </p>
        </div>
        <span className="flex items-center gap-2 text-sm text-slate-600">
          <span className={`h-2.5 w-2.5 rounded-full ${etat[0]}`} />
          {etat[1]}
        </span>
      </div>
    </header>
  );
}
