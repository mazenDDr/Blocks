/** React Flow's background grid and its control-button icons are decorative; hide them from assistive technology (ADR 0072).
 * The buttons keep their own accessible labels. React Flow exposes no prop for this, so a single observer marks them as they render. */
const DECORATIVE = ".react-flow__background, .react-flow__controls-button svg";

export function hideDecorativeSvgs(root: ParentNode = document): void {
  root.querySelectorAll(DECORATIVE).forEach(el => { if (el.getAttribute("aria-hidden") !== "true") el.setAttribute("aria-hidden", "true"); });
}

export function observeDecorativeSvgs(): () => void {
  hideDecorativeSvgs();
  const mo = new MutationObserver(() => hideDecorativeSvgs());
  mo.observe(document.body, { childList: true, subtree: true });
  return () => mo.disconnect();
}
