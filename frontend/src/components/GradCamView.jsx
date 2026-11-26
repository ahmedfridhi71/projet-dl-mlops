// Carte Grad-CAM : zones rouges = zones qui ont le plus poussé vers « fissure ».
export default function GradCamView({ status, overlay, error }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5">
      <p className="text-sm font-medium text-slate-500">Grad-CAM (où le modèle a regardé)</p>

      {status === "loading" && (
        <div className="mt-4 flex h-56 items-center justify-center gap-3 text-sm text-slate-500">
          <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-sky-500" />
          Calcul de la carte… (plus long la première fois)
        </div>
      )}

      {status === "done" && (
        <>
          <img
            src={`data:image/png;base64,${overlay}`}
            alt="Carte Grad-CAM superposée à l'image"
            className="mx-auto mt-4 w-56 rounded-lg [image-rendering:auto]"
          />
          <div className="mx-auto mt-3 w-56">
            <div className="h-2 rounded-full bg-[linear-gradient(to_right,#00f,#0ff,#ff0,#f00)]" />
            <div className="mt-1 flex justify-between text-xs text-slate-500">
              <span>peu important</span>
              <span>très important</span>
            </div>
          </div>
          <p className="mt-3 text-xs text-slate-500">
            Carte relative : il y a toujours une zone « chaude », même sans fissure.
            Elle montre où le modèle a regardé, pas la preuve d'une fissure.
          </p>
        </>
      )}

      {status === "error" && (
        <p className="mt-4 text-sm text-slate-500">Grad-CAM indisponible : {error}</p>
      )}
    </div>
  );
}
