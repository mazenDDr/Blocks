/** User-authored UI metadata. Native validation identities are references, never results. */
export interface NodeAnnotation {
  text: string; author: string; authoredAt: string;
  source: { nodeId: string; nodeType: string; nodeVersion: string; kind: "graph" | "module"; hash: string };
}
export type NodeAnnotations = Record<string, NodeAnnotation>;
const MAX_NOTES = 200, MAX_BYTES = 2 * 1024 * 1024;
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value);
export function readAnnotation(value: unknown): NodeAnnotation | null {
  if (value === undefined) return null;
  const source = record(value) && record(value.source) ? value.source : null;
  if (!record(value) || typeof value.text !== "string" || !value.text.trim() || value.text.length > 4000 || typeof value.author !== "string" || value.author.length > 120
    || typeof value.authoredAt !== "string" || !/^\d{4}-\d{2}-\d{2}T/.test(value.authoredAt) || !Number.isFinite(Date.parse(value.authoredAt)) || !record(value.source)
    || !source || !["nodeId", "nodeType", "nodeVersion"].every(key => typeof source[key] === "string" && !!source[key])
    || !["graph", "module"].includes(String(value.source.kind)) || typeof value.source.hash !== "string" || !/^[0-9a-f]{64}$/.test(value.source.hash))
    throw Error("E_ANNOTATION_FORMAT: This saved node comment is malformed; review or remove it explicitly.");
  return value as unknown as NodeAnnotation;
}
export function annotationsOf(value: unknown): Record<string, unknown> {
  if (value === undefined) return {};
  if (!record(value)) throw Error("E_ANNOTATION_FORMAT: Saved node comments must be a keyed metadata object.");
  return value;
}
export function saveAnnotation(value: unknown, key: string, note: NodeAnnotation): Record<string, unknown> {
  const before = annotationsOf(value);readAnnotation(note);
  const previous = readAnnotation(Object.hasOwn(before, key) ? before[key] : undefined);
  const next = { ...before, [key]: { ...previous, ...note, source: { ...previous?.source, ...note.source } } };
  if (Object.keys(next).length > MAX_NOTES || JSON.stringify(next).length * 2 > MAX_BYTES)
    throw Error("E_ANNOTATION_LIMIT: At most 200 node comments and 2 MiB estimated metadata per project.");
  return next;
}
export function removeAnnotation(value: unknown, key: string): Record<string, unknown> {
  const before = annotationsOf(value);const { [key]: ignored, ...next } = before;return next;
}
export function renameAnnotation(value: unknown, oldKey: string, newKey: string): Record<string, unknown> {
  const before = annotationsOf(value);
  if (!Object.hasOwn(before, oldKey)) return before;
  if (Object.hasOwn(before, newKey)) throw Error("E_ANNOTATION_CONFLICT: A saved comment already belongs to the target node key.");
  return { ...removeAnnotation(before, oldKey), [newKey]: before[oldKey] };
}
