/** Layout-only operations on actual measured canvas rectangles. No graph mutation. */
export type Arrangement = "left" | "right" | "top" | "bottom" | "horizontal-center" | "vertical-center" | "horizontal-spacing" | "vertical-spacing";
export interface LayoutBox { id: string; key: string; x: number; y: number; width: number; height: number }
export const ARRANGEMENTS: { value: Arrangement; label: string; minimum: number }[] = [
  { value: "left", label: "Align left", minimum: 2 },
  { value: "right", label: "Align right", minimum: 2 },
  { value: "top", label: "Align top", minimum: 2 },
  { value: "bottom", label: "Align bottom", minimum: 2 },
  { value: "horizontal-center", label: "Center horizontally", minimum: 2 },
  { value: "vertical-center", label: "Center vertically", minimum: 2 },
  { value: "horizontal-spacing", label: "Distribute horizontally", minimum: 3 },
  { value: "vertical-spacing", label: "Distribute vertically", minimum: 3 },
];
export function arrangeGraphNodes(boxes: LayoutBox[], operation: Arrangement): Record<string, { x: number; y: number }> {
  const definition = ARRANGEMENTS.find(o => o.value === operation);
  if (!definition) throw Error("E_LAYOUT_OPERATION: Unsupported arrangement.");
  if (boxes.length < definition.minimum || boxes.length > 100 || new Set(boxes.map(b => b.key)).size !== boxes.length || new Set(boxes.map(b => b.id)).size !== boxes.length)
    throw Error(`E_LAYOUT_SELECTION: Select ${definition.minimum}–100 distinct nodes in this scope.`);
  if (boxes.some(b => !Number.isFinite(b.x) || !Number.isFinite(b.y))) throw Error("E_LAYOUT_POSITION: A selected node has an invalid position.");
  if (boxes.some(b => !Number.isFinite(b.width) || !Number.isFinite(b.height) || b.width <= 0 || b.height <= 0))
    throw Error("E_LAYOUT_MEASUREMENT: Wait for actual selected-node dimensions.");
  const horizontal = ["left", "right", "horizontal-center", "horizontal-spacing"].includes(operation);
  const axis = horizontal ? "x" : "y", size = horizontal ? "width" : "height";
  const low = Math.min(...boxes.map(b => b[axis]));
  const high = Math.max(...boxes.map(b => b[axis] + b[size]));
  const output = Object.fromEntries(boxes.map(b => [b.key, { x: b.x, y: b.y }]));
  if (operation.endsWith("spacing")) {
    const ordered = [...boxes].sort((a, b) => a[axis] - b[axis]);
    const first = ordered[0], last = ordered[ordered.length - 1];
    const available = last[axis] - first[axis] - ordered.slice(0, -1).reduce((sum, b) => sum + b[size], 0);
    if (available < 0) throw Error("E_LAYOUT_OVERLAP: Move the end nodes farther apart to distribute without overlap.");
    const gap = available / (ordered.length - 1);
    let cursor = first[axis] + first[size] + gap;
    for (const box of ordered.slice(1, -1)) {
      output[box.key][axis] = cursor;
      cursor += box[size] + gap;
    }
  } else {
    for (const box of boxes) output[box.key][axis] = operation.endsWith("center") ? (low + high - box[size]) / 2
      : operation === "right" || operation === "bottom" ? high - box[size] : low;
  }
  if (Object.values(output).some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y))) throw Error("E_LAYOUT_POSITION: Arrangement exceeds finite canvas coordinates.");
  return output;
}
