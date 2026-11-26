import { useRef, useState } from "react";
import { IconPhoto, IconUpload } from "./Icons.jsx";

const TYPES = ["image/jpeg", "image/png"];
const MAX_OCTETS = 10 * 1024 * 1024; // même limite que le backend

function taille(octets) {
  return octets < 1024 * 1024 ? `${(octets / 1024).toFixed(0)} Ko` : `${(octets / 1024 / 1024).toFixed(1)} Mo`;
}

// Zone de dépôt (glisser-déposer ou clic) avec aperçu de l'image.
export default function ImageUploader({ file, previewUrl, onSelect, onError, disabled }) {
  const input = useRef(null);
  const [survol, setSurvol] = useState(false);

  function choisir(f) {
    if (!f) return;
    if (!TYPES.includes(f.type)) return onError("Format refusé : JPEG ou PNG seulement.");
    if (f.size > MAX_OCTETS) return onError("Fichier trop gros : 10 Mo maximum.");
    onSelect(f);
  }
  const ouvrir = () => !disabled && input.current.click();

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Choisir une image"
      onClick={ouvrir}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && ouvrir()}
      onDragOver={(e) => { e.preventDefault(); setSurvol(true); }}
      onDragLeave={() => setSurvol(false)}
      onDrop={(e) => {
        e.preventDefault();
        setSurvol(false);
        if (!disabled) choisir(e.dataTransfer.files[0]);
      }}
      className={`group relative flex min-h-80 flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed p-5 outline-none transition-all duration-300
        focus-visible:ring-2 focus-visible:ring-violet-500/60
        ${survol
          ? "scale-[1.01] border-violet-400 bg-violet-500/10"
          : "border-white/10 bg-white/[0.02] hover:border-violet-400/50 hover:bg-white/[0.04]"}
        ${disabled ? "cursor-wait opacity-70" : "cursor-pointer"}`}
    >
      <input
        ref={input}
        type="file"
        accept="image/jpeg,image/png"
        className="hidden"
        onChange={(e) => { choisir(e.target.files[0]); e.target.value = ""; }}
      />

      {previewUrl ? (
        <>
          <div className="relative">
            <img
              key={previewUrl}
              src={previewUrl}
              alt="Image choisie"
              className="max-h-64 animate-pop rounded-xl object-contain shadow-2xl shadow-black/50 ring-1 ring-white/10"
            />
          </div>
          <div className="flex items-center gap-2 text-sm">
            <IconPhoto className="h-4 w-4 text-slate-500" />
            <span className="max-w-48 truncate font-medium text-slate-300">{file.name}</span>
            <span className="text-slate-500">· {taille(file.size)}</span>
            <span className="ml-1 rounded-md border border-white/10 px-2 py-0.5 text-xs text-slate-400 transition group-hover:border-violet-400/40 group-hover:text-violet-300">
              Changer
            </span>
          </div>
        </>
      ) : (
        <>
          <div className="grid h-16 w-16 place-items-center rounded-2xl bg-linear-to-br from-blue-500/20 to-violet-600/20 text-violet-300 ring-1 ring-violet-400/20 transition-transform duration-300 group-hover:-translate-y-1">
            <IconUpload className="h-7 w-7" />
          </div>
          <div className="text-center">
            <p className="font-semibold text-slate-100">Glissez une image ici</p>
            <p className="mt-1 text-sm text-slate-400">
              ou <span className="font-medium text-violet-300">cliquez pour parcourir</span>
            </p>
          </div>
          <p className="text-xs text-slate-500">JPEG ou PNG · 10 Mo maximum · toute taille</p>
        </>
      )}
    </div>
  );
}
