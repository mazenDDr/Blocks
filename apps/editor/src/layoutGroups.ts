/** Persistent layout groups: named frames around cards, stored in the UI document only (never in the graph or its hash). */
import type { UiDoc } from "./types";

export interface LayoutGroup { id: string; label: string; scope: string; members: string[] }
export interface Box { id: string; x: number; y: number; width: number; height: number }
export interface Frame { id: string; label: string; x: number; y: number; width: number; height: number; members: string[]; missing: string[] }

export const GROUP_LIMIT = 100, MEMBER_LIMIT = 500, LABEL_LIMIT = 80, PAD = 24, HEAD = 28;
const fail = (code: string, message: string): never => { throw Error(`${code}: ${message}`); };

/** Read groups defensively: malformed entries are ignored rather than trusted. */
export function groupsOf(ui: UiDoc): LayoutGroup[] {
  const raw = (ui as { layoutGroups?: unknown }).layoutGroups;
  if (!Array.isArray(raw)) return [];
  return raw.filter((g): g is LayoutGroup => !!g && typeof g === "object" && typeof g.id === "string" && typeof g.label === "string"
    && typeof g.scope === "string" && Array.isArray(g.members) && g.members.every((m: unknown) => typeof m === "string"));
}

function write(ui: UiDoc, groups: LayoutGroup[]): UiDoc {
  return { ...ui, layoutGroups: groups } as UiDoc;
}

function cleanLabel(label: string) {
  const t = label.trim();
  if (!t || t.length > LABEL_LIMIT) fail("E_GROUP_LABEL", `Name a group with 1–${LABEL_LIMIT} characters.`);
  return t;
}

function cleanMembers(members: string[], existing: Set<string>) {
  const unique = [...new Set(members)];
  if (!unique.length || unique.length > MEMBER_LIMIT) fail("E_GROUP_MEMBERS", `Select 1–${MEMBER_LIMIT} cards for a group.`);
  if (unique.some(m => !existing.has(m))) fail("E_GROUP_MEMBERS", "Groups hold only cards that exist in this scope.");
  return unique;
}

export function createGroup(ui: UiDoc, scope: string, label: string, members: string[], existing: Set<string>): { ui: UiDoc; id: string } {
  const groups = groupsOf(ui);
  if (groups.length >= GROUP_LIMIT) fail("E_GROUP_LIMIT", `At most ${GROUP_LIMIT} groups per project.`);
  const taken = new Set(groups.map(g => g.id));
  let n = groups.length + 1;
  while (taken.has(`group_${n}`)) n++;
  const id = `group_${n}`;
  return { ui: write(ui, [...groups, { id, label: cleanLabel(label), scope, members: cleanMembers(members, existing) }]), id };
}

export function updateGroup(ui: UiDoc, id: string, change: { label?: string; members?: string[] }, existing: Set<string>): UiDoc {
  const groups = groupsOf(ui), g = groups.find(x => x.id === id) ?? fail("E_GROUP_MISSING", "That group no longer exists.");
  const next = { ...g, ...(change.label !== undefined ? { label: cleanLabel(change.label) } : {}),
    ...(change.members !== undefined ? { members: cleanMembers(change.members, existing) } : {}) };
  return write(ui, groups.map(x => x.id === id ? next : x));
}

export function deleteGroup(ui: UiDoc, id: string): UiDoc {
  return write(ui, groupsOf(ui).filter(g => g.id !== id));
}

/** Frames for one scope from actual card rectangles; members without a rectangle (deleted or unmeasured) are reported missing. */
export function frames(ui: UiDoc, scope: string, boxes: Box[]): Frame[] {
  const byId = new Map(boxes.map(b => [b.id, b]));
  return groupsOf(ui).filter(g => g.scope === scope).flatMap(g => {
    const present = g.members.map(m => byId.get(m)).filter((b): b is Box => !!b && [b.x, b.y, b.width, b.height].every(Number.isFinite));
    const missing = g.members.filter(m => !byId.has(m));
    if (!present.length) return [];
    const left = Math.min(...present.map(b => b.x)) - PAD, top = Math.min(...present.map(b => b.y)) - PAD - HEAD;
    const right = Math.max(...present.map(b => b.x + b.width)) + PAD, bottom = Math.max(...present.map(b => b.y + b.height)) + PAD;
    return [{ id: g.id, label: g.label, x: left, y: top, width: right - left, height: bottom - top, members: present.map(b => b.id), missing }];
  });
}

/** Keep memberships attached when a card is renamed in this scope. */
export function renameMember(ui: UiDoc, scope: string, oldId: string, newId: string): UiDoc {
  const groups = groupsOf(ui);
  if (!groups.some(g => g.scope === scope && g.members.includes(oldId))) return ui;
  return write(ui, groups.map(g => g.scope === scope ? { ...g, members: g.members.map(m => m === oldId ? newId : m) } : g));
}
