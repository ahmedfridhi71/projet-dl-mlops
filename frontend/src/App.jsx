import { useEffect, useRef, useState } from "react";
import AnalyzeButton from "./components/AnalyzeButton.jsx";
import EmptyState from "./components/EmptyState.jsx";
import GradCamView from "./components/GradCamView.jsx";
import Header from "./components/Header.jsx";
import { IconError } from "./components/Icons.jsx";
import ImageUploader from "./components/ImageUploader.jsx";
import ResultCard from "./components/ResultCard.jsx";
import { getHealth, getMetadata, predict, predictVisualize } from "./services/api.js";

const pourcent = (x) => `${(x * 100).toFixed(1)} %`;

export default function App() {
  const [online, setOnline] = useState(null); // null = on ne sait pas encore
  const [metadata, setMetadata] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [cam, setCam] = useState({ status: "idle" });
  const analyseId = useRef(0); // ignore les réponses d'une ancienne analyse

  // Au chargement : état du backend et infos du modèle.
  useEffect(() => {
    getHealth().then(() => setOnline(true)).catch(() => setOnline(false));
    getMetadata().then(setMetadata).catch(() => {});
  }, []);

  // Libère l'URL d'aperçu quand l'image change.
  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);

  function choisirImage(f) {
    analyseId.current += 1;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    setCam({ status: "idle" });
    setLoading(false);
  }

  async function analyser() {
    const id = ++analyseId.current;
    setLoading(true);
    setError(null);
    setResult(null);
    setCam({ status: "idle" });
    try {
      // 1) Résultat rapide (ONNX).
      const r = await predict(file);
      if (id !== analyseId.current) return;
      setResult({ ...r, id });
      setOnline(true);
    } catch (e) {
      if (id === analyseId.current) setError(e.message);
      return;
    } finally {
      if (id === analyseId.current) setLoading(false);
    }
    // 2) Grad-CAM, plus lent (PyTorch) : affiché quand il arrive.
    setCam({ status: "loading" });
    try {
      const v = await predictVisualize(file);
      if (id === analyseId.current) setCam({ status: "done", overlay: v.overlay_png_base64 });
    } catch (e) {
      if (id === analyseId.current) setCam({ status: "error", error: e.message });
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Halos de couleur en arrière-plan */}
      <div className="pointer-events-none absolute inset-0 -z-10" aria-hidden="true">
        <div className="absolute -top-48 left-1/4 h-[32rem] w-[32rem] rounded-full bg-blue-600/20 blur-3xl" />
        <div className="absolute -top-24 right-0 h-[28rem] w-[28rem] rounded-full bg-violet-600/20 blur-3xl" />
        <div className="absolute inset-0 bg-[radial-gradient(rgba(255,255,255,0.04)_1px,transparent_1px)] [background-size:24px_24px] [mask-image:linear-gradient(to_bottom,black,transparent_70%)]" />
      </div>

      <Header online={online} metadata={metadata} />

      <main className="mx-auto max-w-6xl px-6 pb-16 pt-12">
        <div className="mb-10 max-w-2xl animate-fade-up">
          <h1 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl">
            Analysez une surface de{" "}
            <span className="bg-linear-to-r from-blue-400 via-indigo-400 to-violet-400 bg-clip-text text-transparent">
              béton
            </span>
          </h1>
          <p className="mt-4 text-lg leading-relaxed text-slate-400">
            Déposez une photo : le modèle indique s'il voit une fissure, avec sa probabilité,
            et montre où il a regardé.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          <section className="flex flex-col gap-4">
            <ImageUploader
              file={file}
              previewUrl={previewUrl}
              onSelect={choisirImage}
              onError={setError}
              disabled={loading}
            />
            <AnalyzeButton onClick={analyser} disabled={!file || loading} loading={loading} />
            {error && (
              <p role="alert" className="flex animate-fade-up items-start gap-2 rounded-xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                <IconError className="h-5 w-5 shrink-0" />
                {error}
              </p>
            )}
          </section>

          <section className="flex flex-col gap-6">
            {result ? (
              <>
                <ResultCard key={result.id} result={result} />
                {cam.status !== "idle" && <GradCamView {...cam} />}
              </>
            ) : (
              <EmptyState />
            )}
          </section>
        </div>
      </main>

      {metadata && (
        <footer className="border-t border-white/5">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-3 px-6 py-6 text-xs text-slate-500">
            <span>Métriques au seuil {metadata.threshold}</span>
            {[["Validation", metadata.metrics.validation], ["Test", metadata.metrics.test]].map(([nom, m]) => (
              <span key={nom} className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1">
                <span className="font-medium text-slate-300">{nom}</span> · rappel {pourcent(m.recall)} · précision {pourcent(m.precision)} · F1 {pourcent(m.f1)}
              </span>
            ))}
          </div>
        </footer>
      )}
    </div>
  );
}
