import { useEffect, useRef, useState } from "react";
import Header from "./components/Header.jsx";
import ImageUploader from "./components/ImageUploader.jsx";
import ResultCard from "./components/ResultCard.jsx";
import GradCamView from "./components/GradCamView.jsx";
import { getHealth, getMetadata, predict, predictVisualize } from "./services/api.js";

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
      setResult(r);
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
    <div className="min-h-screen">
      <Header online={online} metadata={metadata} />

      <main className="mx-auto grid max-w-5xl gap-6 px-6 py-8 md:grid-cols-2">
        <section className="flex flex-col gap-4">
          <ImageUploader
            file={file}
            previewUrl={previewUrl}
            onSelect={choisirImage}
            onError={setError}
            disabled={loading}
          />
          <button
            onClick={analyser}
            disabled={!file || loading}
            className="rounded-lg bg-sky-600 px-4 py-3 font-semibold text-white transition hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {loading ? "Analyse en cours…" : "Analyser"}
          </button>
          {error && (
            <p className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p>
          )}
        </section>

        <section className="flex flex-col gap-4">
          {result ? (
            <>
              <ResultCard result={result} />
              {cam.status !== "idle" && <GradCamView {...cam} />}
            </>
          ) : (
            <div className="flex min-h-72 items-center justify-center rounded-xl border border-slate-200 bg-white p-6 text-center text-slate-400">
              Choisissez une image puis cliquez sur « Analyser ».
            </div>
          )}
        </section>
      </main>

      {metadata && (
        <footer className="mx-auto max-w-5xl px-6 pb-8 text-xs text-slate-400">
          Métriques au seuil {metadata.threshold} — validation : rappel{" "}
          {metadata.metrics.validation.recall}, précision {metadata.metrics.validation.precision} · test :
          rappel {metadata.metrics.test.recall}, précision {metadata.metrics.test.precision}
        </footer>
      )}
    </div>
  );
}
