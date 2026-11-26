import { useRef, useState } from "react";

const TYPES = ["image/jpeg", "image/png"];
const MAX_OCTETS = 10 * 1024 * 1024; // même limite que le backend

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

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => !disabled && input.current.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !disabled && input.current.click()}
      onDragOver={(e) => { e.preventDefault(); setSurvol(true); }}
      onDragLeave={() => setSurvol(false)}
      onDrop={(e) => {
        e.preventDefault();
        setSurvol(false);
        if (!disabled) choisir(e.dataTransfer.files[0]);
      }}
      className={`flex min-h-72 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-4 transition
        ${survol ? "border-sky-500 bg-sky-50" : "border-slate-300 bg-white hover:border-slate-400"}
        ${disabled ? "cursor-wait opacity-70" : ""}`}
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
          <img src={previewUrl} alt="Image choisie" className="max-h-64 rounded-lg object-contain" />
          <p className="text-sm text-slate-500">{file.name} · cliquer pour changer</p>
        </>
      ) : (
        <>
          <svg className="h-10 w-10 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
          <p className="font-medium text-slate-700">Glissez une image ici ou cliquez</p>
          <p className="text-sm text-slate-500">JPEG ou PNG, 10 Mo maximum</p>
        </>
      )}
    </div>
  );
}
