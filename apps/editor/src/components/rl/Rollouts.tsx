import { useEffect, useState } from "react";
import { fmtInt } from "../../util";
import { fmt, useGet } from "./common";
import type { CapturedEpisode, CapturedEv, EvalReport, Setup } from "./types";

const ACTION_NAMES: Record<number, string[]> = { 2: ["left", "right"], 4: ["up", "down", "left", "right"] };

function QBars({ q, chosen, names }: { q: number[]; chosen: number; names: string[] }) {
  const lo = Math.min(0, ...q), hi = Math.max(0.0001, ...q), span = hi - lo || 1;
  return (
    <div className="qbars" aria-label="Q-values">
      {q.map((v, i) => (
        <div key={i} className={i === chosen ? "q chosen" : "q"}><span className="qn">{names[i] ?? `a${i}`}</span>
          <span className="qb"><i style={{ width: `${((v - lo) / span) * 100}%` }} /></span><span className="qv">{fmt(v, 4)}</span></div>
      ))}
    </div>
  );
}

interface Viewable { title: string; frames: string[]; steps: { k: number; text: string; detail?: any }[]; note: string; meta: string }

function Viewer({ runId, view, onTrace, names }: { runId: string; view: Viewable; onTrace?: (tid: number) => void; names: string[] }) {
  const [k, setK] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => { setK(0); setPlaying(false); }, [view.title]);
  useEffect(() => {
    if (!playing) return;
    const h = setInterval(() => setK((x) => { if (x >= view.frames.length - 1) { setPlaying(false); return x; } return x + 1; }), 160);
    return () => clearInterval(h);
  }, [playing, view.frames.length]);
  const step = view.steps.find((s) => s.k === k);
  const d = step?.detail;
  return (
    <div className="rollout" aria-label="rollout viewer">
      <div className="rollframe">
        <img alt={`frame ${k}`} src={`/api/rl/runs/${runId}/frames/${view.frames[Math.min(k, view.frames.length - 1)]}.png`} />
        <div className="scrub">
          <button onClick={() => setK(Math.max(0, k - 1))} aria-label="previous frame">◀</button>
          <button onClick={() => setPlaying((p) => !p)} aria-label="play">{playing ? "Pause" : "Play"}</button>
          <button onClick={() => setK(Math.min(view.frames.length - 1, k + 1))} aria-label="next frame">▶</button>
          <input type="range" aria-label="frame scrubber" min={0} max={Math.max(0, view.frames.length - 1)} value={k} onChange={(e) => setK(Number(e.target.value))} />
          <span className="small">frame {k} / {view.frames.length - 1}</span>
        </div>
        <div className="small muted">{view.note}</div>
      </div>
      <div className="rollstep">
        <div className="small">{view.meta}</div>
        {step && d ? (
          <>
            <h4>Step {step.k}: state before the action is frame {k}</h4>
            {d.qValues && <QBars q={d.qValues} chosen={d.action} names={names} />}
            <table className="kv"><tbody>
              <tr><td>action</td><td>{names[d.action] ?? d.action} <span className="muted">({d.actionSource}{d.epsilon != null ? `, ε ${fmt(d.epsilon, 3)}` : ""}{d.policyVersion != null ? `, policy v${d.policyVersion}` : ""})</span></td></tr>
              {d.reward != null && <tr><td>reward</td><td>{fmt(d.reward)} = {Object.entries(d.components ?? {}).map(([c, v]) => `${c} ${fmt(v as number)}`).join(" + ")}</td></tr>}
              {d.rawComponents && <tr><td>raw components</td><td>{Object.entries(d.rawComponents).map(([c, v]) => `${c} ${fmt(v as number)}`).join(", ")}</td></tr>}
              {d.terminated != null && <tr><td>flags</td><td>terminated <b>{String(d.terminated)}</b> · truncated <b>{String(d.truncated)}</b></td></tr>}
              {d.obs && <tr><td>observation</td><td className="small"><code>[{d.obs.map((x: number) => fmt(x, 3)).join(", ")}]</code></td></tr>}
              {d.nextObs && <tr><td>next observation</td><td className="small"><code>[{d.nextObs.map((x: number) => fmt(x, 3)).join(", ")}]</code> <span className="muted">(the true next state, also when the episode ended)</span></td></tr>}
              {d.tid != null && <tr><td>transition id</td><td><b>{d.tid}</b> {onTrace && <button onClick={() => onTrace(d.tid)}>Trace into the update</button>}</td></tr>}
            </tbody></table>
          </>
        ) : <div className="muted small">{step ? "" : k >= view.steps.length ? "Final frame: the state after the last step." : "No step detail recorded for this frame."}</div>}
      </div>
    </div>
  );
}

export function RolloutsTab({ runId, captured, setup, evalReport, onTrace }: { runId: string | null; captured: CapturedEv[]; setup: Setup | null; evalReport: EvalReport | null; onTrace: (tid: number) => void }) {
  const [src, setSrc] = useState<string>("");   // "train:<sha>" | "eval:<index>"
  const names = ACTION_NAMES[setup?.nActions ?? 0] ?? [];
  const opts = [...captured.map((c) => ({ v: `train:${c.sha256}`, label: `training episode ${c.episodeId} · starts at step ${fmtInt(c.startTick)} · ${c.length ?? "unfinished"} steps · ${c.capturedSteps} captured` })),
    ...((evalReport?.final.capturedEpisodes ?? []).map((e, i) => ({ v: `eval:${i}`, label: `final EVALUATION episode seed ${e.seed} · ${e.length} steps · task return ${fmt(e.taskReturn)} (greedy)` })))];
  useEffect(() => { if (!src && opts.length) setSrc(opts[0].v); }, [opts.length]); // eslint-disable-line react-hooks/exhaustive-deps
  const ep = useGet<CapturedEpisode>(runId && src.startsWith("train:") ? `/api/rl/runs/${runId}/episodes/${src.slice(6)}` : null);
  if (!runId) return <div className="empty pad">No run selected. Rollout frames are real <code>env.render()</code> images recorded during a run.</div>;
  let view: Viewable | null = null;
  if (src.startsWith("train:") && ep.data) {
    const e = ep.data;
    view = { title: src, frames: e.frames, note: `${e.frameNote}. ${e.boundedCapture ? `BOUNDED CAPTURE: ${e.capturedSteps} of ${e.length ?? "?"} steps were captured.` : ""}`,
      meta: `Training episode ${e.episodeId} of environment ${e.envIndex} · behavior policy (ε-greedy) · ${e.length != null ? `ended ${e.terminated ? "by termination" : "by truncation"}, return ${fmt(e.return)}, task return ${fmt(e.taskReturn)}` : "unfinished when training stopped"}`,
      steps: e.steps.map((s) => ({ k: s.k, text: "", detail: s })) };
  } else if (src.startsWith("eval:") && evalReport) {
    const e = evalReport.final.capturedEpisodes![Number(src.slice(5))];
    view = { title: src, frames: e.frames, note: "Frame k is the state before step k; the last frame is the final state. Greedy policy on a separate environment instance.",
      meta: `Evaluation episode, seed ${e.seed}, ${e.length} steps, ended ${e.terminated ? "by termination" : "by truncation"}, task return ${fmt(e.taskReturn)}${e.framesCapped ? " · frames capped" : ""}`,
      steps: e.actions.map((a, i) => ({ k: i, text: "", detail: { action: a, qValues: e.qValues[i], actionSource: "greedy" } })) };
  }
  return (
    <div className="scroll pad rlroll">
      <h3>Rollout viewer</h3>
      <div className="small muted">Frames are rendered by the environment during the run (rendering is separate from stepping and never advances the state). Pick an episode, scrub through it, and open any step's transition id in the trace.</div>
      {opts.length === 0 && <div className="empty pad">This run has no captured episodes yet (capture is bounded and configured on the Learner tab).</div>}
      {opts.length > 0 && <select aria-label="rollout source" value={src} onChange={(e) => setSrc(e.target.value)}>{opts.map((o) => <option key={o.v} value={o.v}>{o.label}</option>)}</select>}
      {ep.error && <div className="error">{ep.error}</div>}
      {view && view.frames.length > 0 && <Viewer runId={runId} view={view} onTrace={onTrace} names={names} />}
    </div>
  );
}
