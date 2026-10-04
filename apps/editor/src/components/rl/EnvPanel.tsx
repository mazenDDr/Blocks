import { useEffect, useMemo, useState } from "react";
import { api, errorText } from "../../api";
import type { Graph, UiDoc, Validation } from "../../types";
import { Check, Num, Row, Sel } from "../agent/common";
import { Collapsible, Fact, nodeOfType, patchConfig, useGet } from "./common";
import type { Catalog, CatalogEntry, SpaceInfo } from "./types";

export function SpaceView({ title, sp }: { title: string; sp: SpaceInfo | undefined }) {
  if (!sp) return <div className="muted small">{title}: not typed yet (fix the errors).</div>;
  return (
    <div className="spacebox" aria-label={`${title} space`}>
      <b>{title}</b>: <code>{sp.type}</code>{" "}
      {sp.type === "Discrete" && <>n = {sp.n}{sp.start ? `, start ${sp.start}` : ""} <span className="muted">({sp.dtype})</span></>}
      {sp.type === "Box" && <>shape [{sp.shape?.join(", ")}] <span className="muted">({sp.dtype}{sp.bounded ? ", bounded" : ", has unbounded dimensions"})</span>
        <div className="small muted">low {JSON.stringify(sp.low)} · high {JSON.stringify(sp.high)}</div></>}
      {sp.type !== "Discrete" && sp.type !== "Box" && <span className="muted">{JSON.stringify(sp)}</span>}
    </div>
  );
}

const GRID_TOOLS = ["wall", "start", "goal", "free"] as const;

/** Visual grid-world builder: click cells to paint walls / start / goal. The schematic is drawn from the configured data; the frame beside it is a real env.render(). */
function GridBuilder({ kwargs, onChange }: { kwargs: Record<string, any>; onChange: (k: Record<string, any>) => void }) {
  const w = Number(kwargs.width ?? 5), h = Number(kwargs.height ?? 5);
  const walls: number[][] = kwargs.walls ?? [];
  const start: number[] = kwargs.start ?? [0, 0];
  const goal: number[] = kwargs.goal ?? [w - 1, h - 1];
  const [tool, setTool] = useState<(typeof GRID_TOOLS)[number]>("wall");
  const [frame, setFrame] = useState<{ png: string; src: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const key = JSON.stringify([w, h, walls, start, goal]);
  useEffect(() => {
    const t = setTimeout(() => {
      api.post<{ framePng: string; frameSource: string }>("/api/rl/grid/preview", { kwargs: { width: w, height: h, walls, start, goal } })
        .then((r) => { setFrame({ png: r.framePng, src: r.frameSource }); setErr(null); }).catch((e) => { setFrame(null); setErr(errorText(e)); });
    }, 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const isWall = (x: number, y: number) => walls.some(([a, b]) => a === x && b === y);
  const paint = (x: number, y: number) => {
    const nw = walls.filter(([a, b]) => !(a === x && b === y));
    if (tool === "wall") { if (!(x === start[0] && y === start[1]) && !(x === goal[0] && y === goal[1])) onChange({ ...kwargs, walls: isWall(x, y) ? nw : [...nw, [x, y]] }); }
    else if (tool === "free") onChange({ ...kwargs, walls: nw });
    else if (tool === "start") onChange({ ...kwargs, walls: nw, start: [x, y] });
    else onChange({ ...kwargs, walls: nw, goal: [x, y] });
  };
  const resize = (nw_: number, nh_: number) => onChange({ ...kwargs, width: nw_, height: nh_, walls: walls.filter(([a, b]) => a < nw_ && b < nh_), start: start[0] < nw_ && start[1] < nh_ ? start : [0, 0], goal: goal[0] < nw_ && goal[1] < nh_ ? goal : [nw_ - 1, nh_ - 1] });
  return (
    <div className="gridbuilder" aria-label="grid world builder">
      <h4>Environment builder · grid world</h4>
      <Row label="Size"><Num label="grid width" integer min={2} max={12} value={w} onChange={(n) => resize(Math.max(2, Math.min(12, n ?? w)), h)} /> ×
        <Num label="grid height" integer min={2} max={12} value={h} onChange={(n) => resize(w, Math.max(2, Math.min(12, n ?? h)))} /> cells</Row>
      <Row label="Observation"><Sel label="grid observation" value={kwargs.observation ?? "onehot"} options={[{ value: "onehot", label: "one-hot cell (width x height features)" }, { value: "xy", label: "normalized (x, y) (2 features)" }]}
        onChange={(v) => onChange({ ...kwargs, observation: v })} />
        <label className="small"><Check label="grid random start" value={!!kwargs.random_start} onChange={(b) => onChange({ ...kwargs, random_start: b })} /> random start cell (seeded)</label></Row>
      <Row label="Paint"><span role="radiogroup" aria-label="paint tool">{GRID_TOOLS.map((t) => <button key={t} className={tool === t ? "primary" : ""} onClick={() => setTool(t)}>{t}</button>)}</span></Row>
      <div className="gridrow">
        <div>
          <div className="gridschem" style={{ gridTemplateColumns: `repeat(${w}, 26px)` }} aria-label="grid schematic">
            {Array.from({ length: h }).flatMap((_, y) => Array.from({ length: w }).map((__, x) => {
              const cls = isWall(x, y) ? "wall" : x === goal[0] && y === goal[1] ? "goal" : x === start[0] && y === start[1] ? "start" : "";
              return <button key={`${x},${y}`} className={`gcell ${cls}`} title={`(${x}, ${y})`} aria-label={`cell ${x},${y} ${cls || "free"}`} onClick={() => paint(x, y)}>{cls === "goal" ? "G" : cls === "start" ? "S" : cls === "wall" ? "#" : ""}</button>;
            }))}
          </div>
          <div className="small muted">Schematic drawn from the configured data. # wall · S start · G goal (terminates the episode). Actions: up, down, left, right. A move into a wall or the border keeps the agent in place and records a <code>bump</code>.</div>
        </div>
        <div>
          {frame ? <img className="gridframe" alt="rendered grid world" src={`data:image/png;base64,${frame.png}`} /> : <div className="muted small">{err ?? "rendering…"}</div>}
          <div className="small muted">{frame?.src ?? "Real frame of the reset state, rendered by the environment (stepping and rendering are separate operations; this render did not advance anything)."}</div>
        </div>
      </div>
    </div>
  );
}

export function EnvPanel({ graph, setGraph, validation, ui }: { graph: Graph; setGraph: (f: (g: Graph) => Graph) => void; validation: Validation | null; ui: UiDoc }) {
  const cat = useGet<Catalog>("/api/rl/catalog");
  const env = nodeOfType(graph, "rl.environment");
  const reward = nodeOfType(graph, "rl.reward");
  const cfg = (env?.config ?? {}) as Record<string, any>;
  const entry: CatalogEntry | undefined = cat.data?.entries.find((e) => e.id === (cfg.env_id ?? "CartPole-v1"));
  const rv = validation?.rl?.environment;
  const comps = rv?.components ?? entry?.rewardComponents ?? [];
  const defaults = rv?.defaultWeights ?? entry?.defaultWeights ?? {};
  const listed: { name: string; weight: number; enabled: boolean }[] = useMemo(() => ((reward?.config.components as any[]) ?? []), [reward]);
  const cur = (name: string) => listed.find((c) => c.name === name) ?? { name, weight: defaults[name] ?? 1, enabled: true };
  const setComp = (name: string, patch: Partial<{ weight: number; enabled: boolean }>) => {
    const next = comps.map((c) => (c === name ? { ...cur(c), ...patch } : cur(c))).filter((c) => c.weight !== defaults[c.name] || !c.enabled);
    patchConfig(setGraph, "rl.reward", { components: next });
  };
  const diags = (validation?.diagnostics ?? []).filter((d) => d.nodeId === env?.id || d.nodeId === reward?.id);
  const wrappers: { type: string; min: number; max: number }[] = cfg.wrappers ?? [];
  return (
    <div className="scroll pad rlenv">
      <h3>Environment <span className="badge">Gymnasium {cat.data?.gymnasium ?? "…"}</span> {entry && <span className={`badge ${entry.status === "verified" ? "ok" : "warnb"}`}>{entry.status}</span>}</h3>
      {cat.error && <div className="error">{cat.error}</div>}
      {diags.map((d, i) => <div key={i} className={d.severity === "error" ? "errbadge" : "warn"}><b>{d.code}</b> {d.message}{d.fixes.map((f, j) => <div key={j} className="small">Fix: {f.label}</div>)}</div>)}
      <div className="rlcols">
        <section>
          <Row label="Environment"><Sel label="environment id" value={cfg.env_id ?? "CartPole-v1"} options={(cat.data?.entries ?? []).map((e) => ({ value: e.id, label: `${e.id} (${e.status})` }))}
            onChange={(id) => { patchConfig(setGraph, "rl.environment", { env_id: id, kwargs: id === "Void/GridWorld-v0" ? { width: 5, height: 5, walls: [], start: [0, 0], goal: [4, 4] } : {}, max_episode_steps: null }); patchConfig(setGraph, "rl.reward", { components: [] }); }} /></Row>
          <Row label="Time limit" hint={`TimeLimit applied by gym.make; registered default ${rv?.timeLimit ?? entry?.timeLimit ?? "none"}; reaching it TRUNCATES the episode`}>
            <Num label="max episode steps" integer min={1} nullable value={cfg.max_episode_steps ?? null} onChange={(n) => patchConfig(setGraph, "rl.environment", { max_episode_steps: n })} /></Row>
          <Row label="Seed" hint="seeds the training environments (seed + environment index); evaluation uses its own seeds"><Num label="environment seed" integer value={cfg.seed ?? 0} onChange={(n) => patchConfig(setGraph, "rl.environment", { seed: n ?? 0 })} /></Row>
          <Row label="Autoreset" hint="Gymnasium 1.x vector-env contract">
            <Sel label="autoreset mode" value={cfg.autoreset_mode ?? "NextStep"} options={[{ value: "NextStep", label: "NextStep (Gymnasium default)" }, { value: "SameStep", label: "SameStep" }]}
              onChange={(m) => patchConfig(setGraph, "rl.environment", { autoreset_mode: m })} /></Row>
          <div className="small muted">{(cfg.autoreset_mode ?? "NextStep") === "NextStep"
            ? "NextStep: the step that ends an episode returns the TRUE final observation; the following step() only resets (action ignored, reward 0). That reset step is not stored as a transition."
            : "SameStep: the ending step resets immediately; the true final observation arrives in info['final_obs']. The final rendered frame of an episode is therefore unavailable."}</div>
          <h4>Wrapper chain (outermost first)</h4>
          <ol className="wrapchain" aria-label="wrapper chain">
            {(rv?.wrapperChain ?? entry?.wrapperChain ?? []).map((w, i) => <li key={i}><b>{w.name}</b>{w.maxEpisodeSteps ? ` (${w.maxEpisodeSteps} steps)` : ""}{w.range ? ` [${w.range.join(", ")}]` : ""}{w.weights ? ` weights ${JSON.stringify(w.weights)}` : ""}{w.base ? " · base environment" : ""}<div className="small muted">{w.effect}</div></li>)}
          </ol>
          <div className="small muted">Wrappers change the task actually being learned. Observation pipeline: identity (no observation wrappers are configured).</div>
          <h4>Reward wrappers</h4>
          {wrappers.map((w, i) => (
            <div key={i} className="arow2">ClipReward min <Num label={`clip min ${i}`} value={w.min} onChange={(n) => patchConfig(setGraph, "rl.environment", { wrappers: wrappers.map((x, j) => (j === i ? { ...x, min: n ?? -1 } : x)) })} />
              max <Num label={`clip max ${i}`} value={w.max} onChange={(n) => patchConfig(setGraph, "rl.environment", { wrappers: wrappers.map((x, j) => (j === i ? { ...x, max: n ?? 1 } : x)) })} />
              <button className="danger" onClick={() => patchConfig(setGraph, "rl.environment", { wrappers: wrappers.filter((_, j) => j !== i) })}>Remove</button></div>
          ))}
          <button onClick={() => patchConfig(setGraph, "rl.environment", { wrappers: [...wrappers, { type: "ClipReward", min: -1, max: 1 }] })}>Add ClipReward</button>
        </section>
        <section>
          <h4>Spaces (typed, from the real environment)</h4>
          <SpaceView title="Observation" sp={rv?.observationSpace ?? entry?.observationSpace} />
          <SpaceView title="Action" sp={rv?.actionSpace ?? entry?.actionSpace} />
          <h4>Reward components</h4>
          <table className="dtable" aria-label="reward components"><thead><tr><th>component</th><th>default weight</th><th>weight</th><th>enabled</th></tr></thead>
            <tbody>{comps.map((c) => { const v = cur(c); return (
              <tr key={c}><td><b>{c}</b></td><td className="num">{defaults[c]}</td>
                <td><Num label={`weight ${c}`} value={v.weight} onChange={(n) => setComp(c, { weight: n ?? 0 })} /></td>
                <td><Check label={`enabled ${c}`} value={v.enabled} onChange={(b) => setComp(c, { enabled: b })} /></td></tr>); })}</tbody></table>
          <div className="small muted">reward_t = sum over enabled components of weight × raw component. Raw components are recorded separately in every transition, so toggling one changes what the learner optimizes, not what is logged.
            {rv?.rewardModified && <b> The training reward differs from the environment default: evaluation reports both the task return (default reward) and the training-reward return.</b>}</div>
          {entry && (
            <Collapsible title="Catalog facts" open>
              <table className="kv"><tbody>
                <Fact k="version">{entry.id} (v{entry.version}) · Gymnasium {entry.gymnasium}</Fact>
                <Fact k="reward definition">{entry.rewardDefinition}</Fact><Fact k="termination">{entry.termination}</Fact><Fact k="truncation">{entry.truncation}</Fact>
                <Fact k="success rule">{entry.successText}</Fact><Fact k="rendering">{entry.renderModes.join(", ")}</Fact><Fact k="dependencies">{entry.dependencies.join(", ")}</Fact>
                <Fact k="license">{entry.license}</Fact><Fact k="resources">{entry.resources}</Fact><Fact k="seeding">{entry.seeding}</Fact><Fact k="snapshots">{entry.snapshot}</Fact>
              </tbody></table>
            </Collapsible>
          )}
        </section>
      </div>
      {cfg.env_id === "Void/GridWorld-v0" && <GridBuilder kwargs={cfg.kwargs ?? {}} onChange={(k) => patchConfig(setGraph, "rl.environment", { kwargs: k })} />}
      {ui.description && <div className="muted small">{ui.description}</div>}
    </div>
  );
}
