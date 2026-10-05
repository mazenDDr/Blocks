import { useEffect, useMemo, useRef, useState } from "react";
import { findGraphCommands, type GraphCommand } from "../graphCommands";

export function GraphCommandMenu({ commands, context, onRun }: { commands: GraphCommand[]; context: string; onRun: (command: GraphCommand) => void }) {
  const [open, setOpen] = useState(false), [query, setQuery] = useState("");
  const [page, setPage] = useState(0), [active, setActive] = useState(0);
  const dialog = useRef<HTMLDialogElement>(null), search = useRef<HTMLInputElement>(null);
  const matches = useMemo(() => findGraphCommands(commands, query), [commands, query]);
  const last = Math.max(0, Math.ceil(matches.length / 25) - 1), current = Math.min(page, last);
  const rows = matches.slice(current * 25, (current + 1) * 25), index = Math.min(active, Math.max(0, rows.length - 1));
  const show = () => { setQuery(""); setPage(0); setActive(0); setOpen(true); };
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.altKey || e.shiftKey || !(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== "k") return;
      const target = e.target instanceof Element ? e.target : null;
      if (target?.closest('input, textarea, select, [contenteditable], .cm-editor') || document.querySelector('dialog[open]')) return;
      e.preventDefault();setQuery("");setPage(0);setActive(0);setOpen(true);
    };
    window.addEventListener("keydown", key);return () => window.removeEventListener("keydown", key);
  }, []);
  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (open) { if (!element.open) element.showModal();search.current?.focus(); }
    else if (element.open) element.close();
  }, [open]);
  const run = (command: GraphCommand) => { if (!command.disabled) { dialog.current?.close();setOpen(false);onRun(command); } };
  return <>
    <button aria-label="open graph commands" onClick={show}>Commands <small>⌘/Ctrl K</small></button>
    <dialog ref={dialog} className="command-menu" aria-labelledby="command-menu-title" onClose={() => { if (!dialog.current?.open) setOpen(false); }} onCancel={() => setOpen(false)} onKeyDown={e => {
        if (e.key !== "Tab") return;
        const controls = Array.from(e.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled)'));
        const first = controls[0], lastControl = controls[controls.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault();lastControl?.focus(); }
        else if (!e.shiftKey && document.activeElement === lastControl) { e.preventDefault();first?.focus(); }
      }}>
      <div className="row"><h2 id="command-menu-title">Project commands</h2><button aria-label="close graph commands" onClick={() => setOpen(false)}>Close</button></div>
      <p>{context}. Search uses literal lowercase text. Arrow keys select a result; Enter runs it, Escape closes. Tab reaches result buttons and pages. Typing shortcuts in editors are left to those editors.</p>
      <label>Find workspace, node or block <input ref={search} aria-label="find graph commands" maxLength={200} value={query}
        onChange={e => { setQuery(e.target.value);setPage(0);setActive(0); }} onKeyDown={e => {
          if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault();if (rows.length) setActive((index + (e.key === "ArrowDown" ? 1 : -1) + rows.length) % rows.length); }
          if (e.key === "Enter" && rows[index]) { e.preventDefault();run(rows[index]); }
        }} /></label>
      <p role="status">{matches.length} matching commands · page {current + 1} of {last + 1} · at most 25 results per page.{rows[index] && ` Selected: ${rows[index].label}${rows[index].disabled ? ` · ${rows[index].disabled}` : ""}.`}</p>
      <div className="command-results">{rows.map((command, i) => <button key={command.id} aria-label={`run command ${command.id}`} aria-pressed={i === index} disabled={!!command.disabled} onClick={() => run(command)}>
        <b>{command.label}</b><span>{command.detail}</span>{command.disabled && <span>{command.disabled}</span>}
      </button>)}</div>
      {!rows.length && <p>No existing commands match this search.</p>}
      <div className="row"><button disabled={current === 0} onClick={() => { setPage(current - 1);setActive(0);search.current?.focus(); }}>Previous commands</button><button disabled={current === last} onClick={() => { setPage(current + 1);setActive(0);search.current?.focus(); }}>Next commands</button></div>
    </dialog>
  </>;
}
