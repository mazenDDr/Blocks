import { useCallback, useState } from "react";
import type { NodeChange } from "@xyflow/react";

type Measurements = Record<string, { width: number; height: number }>;
/** Actual transient DOM sizes required by controlled React Flow; never saved in graph/UI. */
export function useFlowMeasurements() {
  const [measurements, setMeasurements] = useState<Measurements>({});
  const rememberDimensions = useCallback((changes: NodeChange[]) => {
    const sizes = changes.filter(c => c.type === "dimensions" && c.dimensions);
    if (!sizes.length) return;
    setMeasurements(old => {
      let next = old;
      for (const change of sizes) {
        if (change.type !== "dimensions" || !change.dimensions) continue;
        const size = change.dimensions;
        if (next[change.id]?.width === size.width && next[change.id]?.height === size.height) continue;
        if (next === old) next = { ...old };
        next[change.id] = { width: size.width, height: size.height };
      }
      return next;
    });
  }, []);
  const pruneMeasurements = useCallback((ids: string[]) => {
    const present = new Set(ids);
    setMeasurements(old => Object.keys(old).every(id => present.has(id)) ? old : Object.fromEntries(Object.entries(old).filter(([id]) => present.has(id))));
  }, []);
  return { measurements, rememberDimensions, pruneMeasurements };
}
