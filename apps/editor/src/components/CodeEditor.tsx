import { useEffect, useRef } from "react";
import { autocompletion, closeBrackets } from "@codemirror/autocomplete";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { bracketMatching, defaultHighlightStyle, indentOnInput, syntaxHighlighting } from "@codemirror/language";
import { lintGutter, setDiagnostics, type Diagnostic as CMDiagnostic } from "@codemirror/lint";
import { EditorState } from "@codemirror/state";
import { EditorView, highlightActiveLine, keymap, lineNumbers } from "@codemirror/view";
import { python } from "@codemirror/lang-python";

export interface SourceDiagnostic { line: number; message: string; severity: "error" | "warning" | "info" }

/** CodeMirror 6 Python editor. `value` is the source of truth held by the parent; `diagnostics` are shown inline (stack frames, syntax errors);
 *  `jump` moves the cursor to a line (used when a stack frame is clicked). */
export function CodeEditor({ value, onChange, diagnostics, jump, readOnly }: {
  value: string; onChange: (v: string) => void; diagnostics: SourceDiagnostic[]; jump?: { line: number; nonce: number } | null; readOnly?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const cb = useRef(onChange);
  cb.current = onChange;

  useEffect(() => {
    if (!host.current) return;
    const v = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: value,
        extensions: [
          lineNumbers(), history(), indentOnInput(), bracketMatching(), closeBrackets(), highlightActiveLine(), autocompletion(), lintGutter(), python(),
          syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
          keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
          EditorState.readOnly.of(!!readOnly),
          EditorView.updateListener.of((u) => { if (u.docChanged) cb.current(u.state.doc.toString()); }),
          EditorView.theme({ "&": { height: "100%", fontSize: "12.5px", border: "1px solid #d9dde3", borderRadius: "4px" }, ".cm-scroller": { fontFamily: "ui-monospace, Menlo, monospace" } }),
        ],
      }),
    });
    view.current = v;
    return () => { v.destroy(); view.current = null; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // external changes (generated template, loading another block) replace the document; typing is not echoed back
  useEffect(() => {
    const v = view.current;
    if (v && v.state.doc.toString() !== value) v.dispatch({ changes: { from: 0, to: v.state.doc.length, insert: value } });
  }, [value]);

  useEffect(() => {
    const v = view.current;
    if (!v) return;
    const ds: CMDiagnostic[] = diagnostics.filter((d) => d.line >= 1 && d.line <= v.state.doc.lines).map((d) => {
      const l = v.state.doc.line(d.line);
      return { from: l.from, to: l.to, severity: d.severity, message: d.message };
    });
    v.dispatch(setDiagnostics(v.state, ds));
  }, [diagnostics, value]);

  useEffect(() => {
    const v = view.current;
    if (!v || !jump || jump.line < 1 || jump.line > v.state.doc.lines) return;
    const l = v.state.doc.line(jump.line);
    v.dispatch({ selection: { anchor: l.from }, scrollIntoView: true });
    v.focus();
  }, [jump?.nonce]); // eslint-disable-line react-hooks/exhaustive-deps

  return <div ref={host} className="cmhost" aria-label="code editor" />;
}
