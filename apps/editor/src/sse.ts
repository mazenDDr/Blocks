/** Incremental server-sent-event parser for `event:`/`data:` frames separated by a blank line. */
export interface SseEvent { event: string; data: string }

export function sseParser(onEvent: (e: SseEvent) => void) {
  let buffer = "";
  return (chunk: string) => {
    buffer += chunk.replace(/\r\n/g, "\n");
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, end); buffer = buffer.slice(end + 2);
      let event = "message"; const data: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length) onEvent({ event, data: data.join("\n") });
    }
  };
}

/** POST JSON and deliver each event; resolves when the stream closes. Non-2xx responses reject with the body text. */
export async function postEventStream(url: string, body: unknown, onEvent: (e: SseEvent) => void, signal?: AbortSignal) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" }, body: JSON.stringify(body), signal });
  if (!r.ok || !r.body) throw Error(`${r.status}: ${(await r.text()).slice(0, 500)}`);
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader(), feed = sseParser(onEvent);
  for (;;) { const { done, value } = await reader.read(); if (done) break; feed(value); }
}
