import { useEffect, useState, type ReactNode } from "react";
import { api } from "../api";

/** The Blocks mark: two pieces resting on each other and a small cap piece, the way blocks stack on a play-room floor. */
export function BlocksMark({ size = 28 }: { size?: number }) {
  return (
    <svg className="blocks-mark" width={size} height={size} viewBox="0 0 40 40" aria-hidden="true">
      <rect x="3" y="15" width="16" height="16" rx="5" fill="var(--accent)" />
      <rect x="21" y="9" width="16" height="16" rx="5" fill="var(--accent)" opacity="0.5" />
      <rect x="12" y="4" width="16" height="9" rx="4.5" fill="var(--ink)" />
    </svg>
  );
}

const PATHS: Record<string, string> = {
  graph: "M5 7h5v5H5zM14 12h5v5h-5zM10 9.5h2a2 2 0 0 1 2 2V12",
  data: "M4 6h16M4 12h16M4 18h16M9 6v12",
  experiments: "M9 3h6M10 3v6l-5 9a2 2 0 0 0 1.8 3h10.4a2 2 0 0 0 1.8-3l-5-9V3",
  training: "M4 17l5-6 4 3 7-8",
  debug: "M12 20a6 6 0 0 0 6-6V10a6 6 0 0 0-12 0v4a6 6 0 0 0 6 6zM12 4V2M6 12H3M21 12h-3M7 18l-2 2M17 18l2 2",
  attention: "M4 12c3-5 13-5 16 0-3 5-13 5-16 0zM12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z",
  backends: "M5 5h14v5H5zM5 14h14v5H5zM8 7.5h.01M8 16.5h.01",
  coverage: "M4 20V10M10 20V4M16 20v-7M22 20H2",
  domain: "M3 7l9-4 9 4-9 4zM3 12l9 4 9-4M3 17l9 4 9-4",
  scale: "M12 3v18M3 12h18M6 6l12 12M18 6L6 18",
  production: "M5 19l4-4M14 4l6 6-8 8-6-6zM16 8h.01",
  records: "M6 3h9l4 4v14H6zM15 3v4h4M9 12h7M9 16h7",
};

/** Workspace icons: one stroke set at the text's weight (1.6px at 18px), so they read as part of the label, not decoration. */
export function WorkspaceIcon({ view }: { view: string }) {
  const d = PATHS[view] ?? PATHS.graph;
  return (
    <svg className="ws-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={d} />
    </svg>
  );
}

type Theme = "system" | "light" | "dark";
const NEXT: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };

function readTheme(): Theme {
  try { const t = localStorage.getItem("blocks-theme"); return t === "light" || t === "dark" ? t : "system"; } catch { return "system"; }
}

/** Light, dark or follow the system. The choice is a per-viewer convenience kept in localStorage; without it the system setting applies. */
export function ThemeSwitch() {
  const [theme, setTheme] = useState<Theme>(readTheme);
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") delete root.dataset.theme; else root.dataset.theme = theme;
    try { if (theme === "system") localStorage.removeItem("blocks-theme"); else localStorage.setItem("blocks-theme", theme); } catch { /* storage unavailable */ }
  }, [theme]);
  const label = theme === "system" ? "Theme: follows system" : theme === "light" ? "Theme: light" : "Theme: dark";
  return (
    <button className="theme-switch" aria-label={`${label}. Switch theme`} title={label} onClick={() => setTheme(NEXT[theme])}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden="true">
        {theme === "dark" ? <path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z" />
          : theme === "light" ? <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4 12H2M22 12h-2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5" /></>
          : <><circle cx="12" cy="12" r="8" /><path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor" /></>}
      </svg>
      <span className="theme-label">{theme === "system" ? "Auto" : theme === "light" ? "Light" : "Dark"}</span>
    </button>
  );
}

/* ---------------------------------------------------------------- direction 2: liquid glass (design/DIRECTION.md) */

// Displacement maps for the glass rim: red bends the backdrop sideways near the left/right edges, green near the top/bottom,
// neutral (#80) across the middle, so a control lenses what is behind its rim and leaves its centre clear.
const RIM = (x2: number, y2: number) => `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100" preserveAspectRatio="none"><linearGradient id="g" x2="${x2}" y2="${y2}"><stop offset="0" stop-color="${x2 ? "#ff0000" : "#00ff00"}"/><stop offset=".22" stop-color="${x2 ? "#800000" : "#008000"}"/><stop offset=".78" stop-color="${x2 ? "#800000" : "#008000"}"/><stop offset="1" stop-color="#000000"/></linearGradient><rect width="100" height="100" fill="url(#g)"/></svg>`)}`;

/** The wallpaper the glass bends (the product's own pieces, softened) and the refraction filter small glass controls use. */
export function GlassBackdrop() {
  return (
    <>
      <svg className="wallpaper" aria-hidden="true" focusable="false" preserveAspectRatio="xMidYMid slice" viewBox="0 0 1600 1000">
        <defs><filter id="wall-soft" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="70" /></filter></defs>
        <rect width="1600" height="1000" style={{ fill: "var(--wall-base)" }} />
        <g filter="url(#wall-soft)">
          <rect x="-80" y="420" width="620" height="620" rx="150" style={{ fill: "var(--wall-a)" }} opacity="0.55" transform="rotate(-12 230 730)" />
          <rect x="980" y="-160" width="560" height="560" rx="140" style={{ fill: "var(--wall-b)" }} opacity="0.5" transform="rotate(14 1260 120)" />
          <rect x="600" y="620" width="420" height="300" rx="110" style={{ fill: "var(--wall-c)" }} opacity="0.4" />
        </g>
      </svg>
      <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden="true" focusable="false">
        <filter id="liquid-rim" x="0" y="0" width="100%" height="100%" colorInterpolationFilters="sRGB">
          <feImage href={RIM(1, 0)} preserveAspectRatio="none" result="mx" />
          <feImage href={RIM(0, 1)} preserveAspectRatio="none" result="my" />
          <feComposite in="mx" in2="my" operator="arithmetic" k2="1" k3="1" result="map" />
          <feDisplacementMap in="SourceGraphic" in2="map" scale="22" xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </svg>
    </>
  );
}

/** Whether the control service answers: polled every 15 s on a cheap read the editor already makes, like Docker's engine light. */
export function useServiceState(): "running" | "unreachable" | "checking" {
  const [state, setState] = useState<"running" | "unreachable" | "checking">("checking");
  useEffect(() => {
    let live = true;
    const check = (first = false) => { if (document.hidden && !first) return; api.get("/api/backends").then(() => live && setState("running")).catch(() => live && setState("unreachable")); };
    check(true);
    const t = window.setInterval(check, 15000);
    return () => { live = false; window.clearInterval(t); };
  }, []);
  return state;
}

/** The window's footer, in Docker Desktop's manner: the service light on the left, what this project is doing on the right. */
export function StatusBar({ service, backend, working, kind, children }: { service: "running" | "unreachable" | "checking"; backend?: string; working: number; kind: string; children?: ReactNode }) {
  return (
    <footer className="statusbar" aria-label="Status">
      <span className={`sb-item sb-service ${service}`}><i className="dot" aria-hidden="true" />{service === "running" ? "Control service running" : service === "unreachable" ? "Control service unreachable" : "Checking control service…"}</span>
      {backend && <span className="sb-item">Backend <b>{backend}</b></span>}
      <span className={`sb-item ${working ? "sb-working" : ""}`}><i className={`dot ${working ? "working" : "idle"}`} aria-hidden="true" />{working ? `${working} run${working > 1 ? "s" : ""} working` : "No run working"}</span>
      <span className="sb-item">{kind} graph</span>
      <span className="sb-spacer" />
      {children}
    </footer>
  );
}
