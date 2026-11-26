// Appels au backend FastAPI.
export const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

// Envoie une requête et renvoie le JSON, ou lève une erreur avec le message du backend.
async function request(path, options) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, options);
  } catch {
    throw new Error(`Backend injoignable (${API_URL}). Est-il lancé ?`);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail ?? `Erreur HTTP ${res.status}`);
  }
  return data;
}

function imageForm(file) {
  const form = new FormData();
  form.append("file", file); // le backend attend le champ "file"
  return { method: "POST", body: form };
}

export const getHealth = () => request("/health");
export const getMetadata = () => request("/model/metadata");
export const predict = (file) => request("/predict", imageForm(file));
export const predictVisualize = (file) => request("/predict/visualize", imageForm(file));
