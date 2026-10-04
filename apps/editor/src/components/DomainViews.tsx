import { useState } from "react";
import { Heatmap, ColorBar } from "./Heatmap";
import { TabProv, useNodeResult } from "./Tabular";
import { NotRecorded } from "./Provenance";
import { fmtNum } from "../util";

type RecordData = Record<string, any>;
const num = (v: unknown) => typeof v === "number" && Number.isFinite(v) ? fmtNum(v, 5) : "not recorded";

function Facts({ value, title }: { value: RecordData; title: string }) {
  return <details className="domain-facts"><summary>{title}</summary><pre>{JSON.stringify(value, null, 2)}</pre></details>;
}

function SamplePicker({ samples, index, setIndex }: { samples: RecordData[]; index: number; setIndex: (i: number) => void }) {
  return <label>Recorded sample <select aria-label="domain sample" value={Math.min(index, samples.length - 1)} onChange={(e) => setIndex(Number(e.target.value))}>
    {samples.map((s, i) => <option value={i} key={i}>{s.id ?? s.before?.id ?? i}</option>)}
  </select></label>;
}

function VisionOverlay({ sample, prediction = false, label }: { sample: RecordData; prediction?: boolean; label: string }) {
  const [masks, setMasks] = useState(true);
  const [h, w] = sample.size;
  const boxes: number[][] = prediction ? sample.predBoxes : sample.gtBoxes ?? sample.boxesXyxyPixel;
  const kps: number[][][] = prediction ? [] : sample.gtKeypoints ?? sample.keypoints ?? [];
  const mask = prediction ? sample.predMask : sample.gtMask ?? sample.labelMap;
  return <figure className="domain-image"><figcaption>{label}</figcaption>
    <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${label}: ${sample.id}`}>
      <image href={`data:image/png;base64,${sample.image}`} width={w} height={h} />
      {masks && mask && <image href={`data:image/png;base64,${mask}`} width={w} height={h} opacity="0.45" />}
      {(boxes ?? []).map(([x1, y1, x2, y2], i) => <g key={i}><rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke={prediction ? "#ffd84d" : "#fff"} strokeWidth="0.5" />
        <text x={x1} y={Math.max(3, y1 - 1)} fontSize="3" fill="#fff">{(prediction ? sample.predLabels : sample.gtLabels ?? sample.labels)?.[i]}{prediction ? ` (${num(sample.predScores?.[i])})` : ""}</text></g>)}
      {kps.map((row, i) => row.map(([x, y, v], j) => v > 0 && <circle key={`${i}-${j}`} cx={x} cy={y} r="0.9" fill={v === 2 ? "#ffda50" : "none"} stroke="#ffda50" strokeWidth="0.4"><title>keypoint {j}, visibility {v}: ({x}, {y})</title></circle>))}
    </svg>
    {mask && <label><input type="checkbox" checked={masks} onChange={(e) => setMasks(e.target.checked)} /> Mask overlay</label>}
  </figure>;
}

function Curve({ curve }: { curve?: RecordData[] }) {
  if (!curve?.length) return null;
  return <details><summary>Recorded training and validation curve ({curve.length} epochs)</summary><div className="tablewrap"><table><thead><tr><th>Epoch</th><th>Train loss</th><th>Validation loss</th><th>Validation IoU</th></tr></thead><tbody>
    {curve.map((r) => <tr key={r.epoch}><td>{r.epoch}</td><td>{num(r.trainLoss)}</td><td>{num(r.valLoss)}</td><td>{num(r.valMeanIoU)}</td></tr>)}
  </tbody></table></div></details>;
}

function VisionView({ data, kind }: { data: RecordData; kind: string }) {
  const [index, setIndex] = useState(0);
  const samples = data.samples ?? data.examples ?? [];
  const s = samples[Math.min(index, samples.length - 1)];
  return <>
    {data.segmentation && <><h3>Validation segmentation</h3><p>Mean IoU {num(data.segmentation.meanIoU)} · Dice {num(data.segmentation.meanDice)} · pixel accuracy {num(data.segmentation.pixelAccuracy)}</p>
      <p className="muted small">{data.segmentation.provenance}</p><Facts title="Per-class metrics" value={data.segmentation} /></>}
    {data.detection && <><p>Derived detection mAP {num(data.detection.map)} · AP50 {num(data.detection.map50)}</p><p className="muted small">{data.detection.provenance}</p></>}
    {data.consistencyAfter && <Facts title="Annotation consistency after transforms" value={data.consistencyAfter} />}
    <Curve curve={data.curve} />
    {s && <><SamplePicker samples={samples} index={index} setIndex={setIndex} />
      <div className="domain-pair">{kind === "vision_transform" ? <><VisionOverlay sample={s.before} label="Before: ground truth" /><VisionOverlay sample={s.after} label="After: ground truth" /></>
        : <><VisionOverlay sample={s} label="Ground truth" />{kind === "vision_seg" && <VisionOverlay sample={s} prediction label="Learned prediction" />}</>}</div>
      <div className="prov">Sample {s.id ?? s.before?.id} · {kind === "vision_seg" ? `validation, epoch ${data.config.epochs}` : "source / recorded transform"}</div>
      <Facts title={s.log ? "Actual transform parameters and removed instances" : "Recorded coordinates, visibility and sample metrics"} value={s.log ?? Object.fromEntries(Object.entries(s).filter(([k]) => !["image", "gtMask", "predMask", "labelMap"].includes(k)))} />
    </>}
    <Facts title="Image and annotation contract" value={data.contract} />
    {data.policy && <Facts title="Declared clipping and removal policy" value={data.policy} />}
  </>;
}

function NlpView({ data }: { data: RecordData }) {
  const [index, setIndex] = useState(0), [token, setToken] = useState(1);
  const samples = data.samples ?? data.batch?.examples ?? data.examples ?? [], s = samples[Math.min(index, samples.length - 1)];
  const t = s?.offsets?.[token];
  return <>
    <p>{data.policyText ?? (data.contract?.labelPolicy ? `Label policy: ${data.contract.labelPolicy}; ignore index ${data.contract.ignoreIndex}` : "Source character spans (end exclusive)")}</p>
    {data.spanMetrics && <><h3>Validation span evaluation</h3><p>Micro F1 {num(data.spanMetrics.micro.f1)} · precision {num(data.spanMetrics.micro.precision)} · recall {num(data.spanMetrics.micro.recall)}</p><p className="muted small">{data.spanMetrics.provenance}</p><Facts title="Per-entity and strict IOB2 metrics" value={{ selected: data.spanMetrics, strict: data.strictMetrics, seqeval: data.crossCheck, tokenAccuracy: data.tokenAccuracy }} /></>}
    <Curve curve={data.curve} />
    {s && <><SamplePicker samples={samples} index={index} setIndex={(i) => { setIndex(i); setToken(1); }} />
      <p className="domain-text">{t && t[1] > t[0] ? <>{s.text.slice(0, t[0])}<mark>{s.text.slice(t[0], t[1])}</mark>{s.text.slice(t[1])}</> : s.text}</p>
      {s.tokens && <><p className="muted small">Select a subword to highlight its original character span. Predictions at unscored positions are excluded from evaluation.</p>
        <div className="tablewrap"><table className="dtable"><thead><tr><th>Position / subword</th><th>ID</th><th>Source span</th><th>Word</th><th>Attention</th><th>Gold / loss label</th><th>Prediction</th></tr></thead><tbody>
          {s.tokens.map((piece: string, i: number) => <tr key={i} className={i === token ? "sel" : ""}><td><button onClick={() => setToken(i)} aria-label={`subword ${i}`}>{i}: {piece}</button></td><td>{s.ids[i]}</td><td>[{s.offsets[i].join(", ")})</td><td>{s.wordIds[i] ?? "special"}</td><td>{s.attention[i]}</td><td>{s.labels[i] ?? `ignore (${s.labelIds[i]})`}</td><td>{s.predTokens?.[i] ?? "not recorded"}</td></tr>)}
        </tbody></table></div></>}
      <div className="prov">Sample {s.id} · {s.split ?? "validation"} · {data.config ? `epoch ${data.config.epochs}` : "recorded tokenization / source"}</div>
      <Facts title="Character entities and word-level labels / predictions" value={{ spans: s.spans, words: s.words }} />
    </>}
    <Facts title="Tokenizer / mask / label contract" value={data.contract} />
    {data.batch && <><p>{data.batch.note}</p><Facts title="Actual padded batch shape and attention lengths" value={{ shape: data.batch.shape, lengths: data.batch.lengths }} /></>}
    {data.tokenizer && <Facts title="Fitted tokenizer identity and normalization" value={data.tokenizer} />}
    {data.maskCheck && <Facts title="Padding invariance measured on the trained model" value={data.maskCheck} />}
  </>;
}

function Waveform({ envelope, duration, selectedTime }: { envelope: RecordData; duration: number; selectedTime?: number }) {
  const n = envelope.bins;
  const points = [...envelope.max.map((v: number, i: number) => `${i / n * 700},${45 - v * 40}`), ...envelope.min.map((v: number, i: number) => `${i / n * 700},${45 - v * 40}`).reverse()].join(" ");
  return <><svg className="domain-wave" viewBox="0 0 700 90" role="img" aria-label="Recorded waveform min/max envelope"><polygon points={points} fill="#387aa5" /><line x1="0" x2="700" y1="45" y2="45" stroke="#888" />
    {selectedTime != null && <line x1={selectedTime / duration * 700} x2={selectedTime / duration * 700} y1="0" y2="90" stroke="#d44" />}</svg>
    <div className="domain-axis"><span>0 s</span><span>{num(duration)} s</span></div><p className="muted small">Min/max envelope over {envelope.samples} samples, {envelope.bins} bins. Time = sample index / sample rate. Source waveform displays channel 0; spectral features use the declared channel policy.</p></>;
}

function Spectrogram({ values }: { values: number[][] }) {
  const transposed = values[0].map((_, m) => values.map((f) => f[m])).reverse();
  let min = Infinity, max = -Infinity;
  for (const row of values) for (const v of row) { min = Math.min(min, v); max = Math.max(max, v); }
  const scale = { mode: "sequential" as const, min, max };
  return <><p className="small">Mel bands (high ↑), time →. Recorded feature values; scale below.</p><div className="domain-spectrum"><Heatmap values={transposed} scale={scale} cell={3} title="Recorded mel spectrogram" /></div><ColorBar scale={scale} /></>;
}

function SpeechView({ data, kind }: { data: RecordData; kind: string }) {
  const [index, setIndex] = useState(0), [frame, setFrame] = useState(0);
  const samples = data.samples ?? data.examples ?? data.clips ?? [], s = samples[Math.min(index, samples.length - 1)];
  const rate = data.contract?.sampleRate ?? data.config?.sampleRate;
  const selected = s?.frameTimes?.[frame];
  const vocabulary: string[] = data.alphabet ? ["blank", ...data.alphabet] : [];
  return <>
    {data.teaching && <section className="domain-teaching"><h3>Teaching signal: exactly 32,000 samples/channel</h3><p>{data.teaching.what}</p><Waveform envelope={data.teaching.envelope} duration={data.teaching.durationSeconds} /><Facts title="Teaching sample rate, channels, duration and samples" value={Object.fromEntries(Object.entries(data.teaching).filter(([k]) => k !== "envelope" && k !== "rawHead"))} /></section>}
    {data.rates && <><h3>Validation decoding</h3><p>CER {num(data.rates.cer.rate)} · WER {num(data.rates.wer.rate)}</p><p className="muted small">{data.rates.provenance}</p><Facts title="CTC lengths, blank, reduction and alignment constraints" value={data.ctc} /></>}
    {data.config?.formula && <p>{data.config.formula}</p>}
    {data.batch && <Facts title="Padded batch, real sequence lengths and mask counts" value={data.batch} />}
    {data.frameTable && <Facts title="Actual STFT frame counts against the formula" value={{ checks: data.formulaChecks, clips: data.frameTable, stft: data.stftCheck }} />}
    <Curve curve={data.curve} />
    {s && <><SamplePicker samples={samples} index={index} setIndex={(i) => { setIndex(i); setFrame(0); }} />
      <p>Reference: <code>{s.ref ?? s.text ?? "not recorded"}</code>{kind === "speech_ctc" && <> · greedy hypothesis: <code>{s.hyp || "(empty)"}</code></>}</p>
      {s.envelope && <Waveform envelope={s.envelope} duration={s.durationSeconds ?? s.samples / rate} selectedTime={selected} />}
      {s.spectrogram && <Spectrogram values={s.spectrogram} />}
      {s.frameTimes && <><label>Inspect {kind === "speech_ctc" ? "output" : "feature"} frame <input type="range" aria-label="speech frame" min="0" max={s.frameTimes.length - 1} value={frame} onChange={(e) => setFrame(Number(e.target.value))} /></label>
        <p>Frame {frame} · time {num(selected)} s{rate && <> · sample index ≈ {Math.round(selected * rate)}</>}{s.framePath && <> · best label {vocabulary[s.framePath[frame]] === " " ? "space" : vocabulary[s.framePath[frame]]} · probability {num(s.frameProb[frame])}</>}</p></>}
      {s.tokenFrames && <><p className="muted small">Greedy token onset frames are model alignment observations; they are not forced alignment or word timestamps.</p><Facts title="Decoded tokens → output frames → time; gold timing regions (not recorded when unavailable)" value={{ tokens: [...s.hyp], frames: s.tokenFrames, timesSeconds: s.tokenTimes, gold: s.gold }} /></>}
      {s.align && <><h4>Edit-distance alignment</h4>{["cer", "wer"].map((k) => <div key={k}><b>{k.toUpperCase()}</b>: {num(s.align[k].rate)} · S {s.align[k].substitutions} / D {s.align[k].deletions} / I {s.align[k].insertions}
        <div className="domain-edits">{s.align[k].ops.map((o: RecordData, i: number) => <span key={i} className={`edit-${o.op}`} title={o.op}><b>{o.op}</b><code>{o.ref === null ? "∅" : o.ref === " " ? "␠" : o.ref}</code><code>{o.hyp === null ? "∅" : o.hyp === " " ? "␠" : o.hyp}</code></span>)}</div></div>)}</>}
      <div className="prov">Sample {s.id} · {data.config?.epochs ? `validation, epoch ${data.config.epochs}` : "source / recorded features"}</div>
      <Facts title="Recorded sample lengths and time contract" value={Object.fromEntries(Object.entries(s).filter(([k]) => !["spectrogram", "envelope", "framePath", "frameProb", "frameTimes", "rawHead"].includes(k)))} />
    </>}
    <Facts title="Audio / feature contract" value={data.contract} />
    {data.config && <Facts title="Declared feature / model configuration" value={data.config} />}
  </>;
}

export function DomainResultView({ runId, node }: { runId: string | null; node: string }) {
  const st = useNodeResult<RecordData>(runId, "summary", node);
  if (!runId) return <NotRecorded message="No run selected. Domain values are recorded from real execution." />;
  if (st.error) return <div className="error pre">{st.error}</div>;
  if (!st.data) return <div className="muted">Loading recorded result…</div>;
  if (!st.data.available) return <><NotRecorded message={st.data.message} /><TabProv p={st.data.provenance} /></>;
  const { data, summaryKind: kind, provenance } = st.data;
  return <div className="domain-result"><TabProv p={provenance} label={`summary artifact ${(provenance as any)?.summarySha256?.slice(0, 12) ?? "not recorded"}`} />
    {data.note && <p className={data.synthetic === true ? "notice-inline synthetic" : "notice-inline"}>{data.note}</p>}
    {kind.startsWith("vision") ? <VisionView key={`${runId}/${node}`} data={data} kind={kind} /> : kind.startsWith("nlp") ? <NlpView key={`${runId}/${node}`} data={data} /> : <SpeechView key={`${runId}/${node}`} data={data} kind={kind} />}
    <Facts title="Recorded provenance and library / model identity" value={data.provenance} />
  </div>;
}
