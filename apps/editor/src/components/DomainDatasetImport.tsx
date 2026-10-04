import { useState } from "react";
import { api, errorText } from "../api";

export interface ImportedDataset {
  id: string; path: string; family: "vision" | "nlp" | "speech"; kind: string; synthetic: boolean;
  license: { declaration: string; authority: string }; contract: unknown;
  sources: { path: string; sha256: string; bytes: number }[]; payloadSha256: string;
}

export function DomainDatasetImport({ area, onUse }: { area: string; onUse: (d: ImportedDataset) => void }) {
  const kind = area === "Vision" ? "coco" : area === "Speech" ? "wav" : "conll";
  const [path, setPath] = useState(""); const [root, setRoot] = useState("");
  const [license, setLicense] = useState(""); const [synthetic, setSynthetic] = useState("");
  const [pairs, setPairs] = useState("[]"); const [token, setToken] = useState(0); const [label, setLabel] = useState(-1);
  const [busy, setBusy] = useState(false); const [error, setError] = useState("");
  const [dataset, setDataset] = useState<ImportedDataset | null>(null); const [applied, setApplied] = useState(false);
  const convert = async () => {
    setBusy(true); setError(""); setDataset(null); setApplied(false);
    try {
      setDataset(await api.post<ImportedDataset>("/api/domain/datasets", { kind, path, root: kind === "conll" ? null : root,
        license, synthetic: synthetic === "true", flip_pairs: kind === "coco" ? JSON.parse(pairs) : [], token_column: token, label_column: label }));
    } catch (e) { setError(errorText(e)); } finally { setBusy(false); }
  };
  return <section className="prod-panel" aria-label="domain dataset import">
    <h3>Import {kind === "coco" ? "COCO segmentation" : kind === "wav" ? "WAV and transcripts" : "CoNLL IOB2"} dataset</h3>
    <p>Use files on the backend machine. Import preserves source snapshots and hashes; license and synthetic status are your declarations. Limits: 2–256 records, 64 MiB source and decoded data. Changing data requires fresh training.</p>
    {kind === "coco" && <p>RGB PNG/JPEG, uniform 4–512 pixel dimensions, 1–4 categories, polygon/RLE masks; no crowd or box-only annotations. COCO boxes are preserved; overlapping instance masks remain separate. Semantic training uses the later annotation at overlaps.</p>}
    {kind === "wav" && <p>JSON array of {"{id, file, text}"}; relative WAV paths. Uniform rate and channels, 512–32,000 samples/channel, 1–4 channels. No resampling, trimming or invented timing labels. Feature extraction averages channels as declared by the audio feature operation.</p>}
    {kind === "conll" && <p>UTF-8 whitespace columns, blank-line sentences, strict IOB2. Text is reconstructed with single spaces; entity character offsets refer to that text. Original tokens, tags and line numbers are preserved.</p>}
    <label>Source {kind === "conll" ? "CoNLL file" : "JSON file"} path <input aria-label="dataset source path" value={path} onChange={(e) => setPath(e.target.value)} /></label>
    {kind !== "conll" && <label>Member directory <input aria-label="dataset member directory" value={root} onChange={(e) => setRoot(e.target.value)} /></label>}
    <label>License / permitted-use declaration <input aria-label="dataset license declaration" value={license} onChange={(e) => setLicense(e.target.value)} /></label>
    <label>Source status <select aria-label="dataset synthetic declaration" value={synthetic} onChange={(e) => setSynthetic(e.target.value)}>
      <option value="">Choose a declaration</option><option value="true">SYNTHETIC fixture</option><option value="false">Non-synthetic source (user declared)</option>
    </select></label>
    {kind === "coco" && <label>Keypoint flip index pairs (JSON) <input aria-label="dataset flip pairs" value={pairs} onChange={(e) => setPairs(e.target.value)} /></label>}
    {kind === "conll" && <div className="row"><label>Token column <input aria-label="dataset token column" type="number" min={-128} max={127} value={token} onChange={(e) => setToken(Number(e.target.value))} /></label>
      <label>IOB2 column <input aria-label="dataset label column" type="number" min={-128} max={127} value={label} onChange={(e) => setLabel(Number(e.target.value))} /></label></div>}
    <button disabled={busy || !path.trim() || !license.trim() || !synthetic || (kind !== "conll" && !root.trim())} onClick={convert}>{busy ? "Importing…" : "Import and verify dataset"}</button>
    {error && <p role="alert">{error}</p>}
    {dataset && <><p>{dataset.synthetic ? "SYNTHETIC" : "Non-synthetic (user declared)"} · {dataset.license.declaration} · dataset {dataset.id.slice(0, 16)}</p>
      <details><summary>Imported contract, source hashes and license</summary><pre className="prod-recorded">{JSON.stringify(dataset, null, 2)}</pre></details>
      <button onClick={() => { onUse(dataset); setApplied(true); }}>Use imported dataset in this project</button>
      {applied && <p role="status">Draft source updated. Checkpoint continuation and fitted-tokenizer selection cleared; run to train a fresh model. Previous runs retain their provenance.</p>}
    </>}
  </section>;
}
