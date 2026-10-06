import {test} from 'node:test';
import assert from 'node:assert/strict';
import {sseParser} from '../src/sse.ts';

test('frames split across arbitrary chunks, CRLF and multi-line data are reassembled in order',()=>{
  const seen=[];const feed=sseParser(e=>seen.push(e));
  const wire='event: token\r\ndata: {"delta":"Re"}\r\n\r\nevent: token\ndata: {"delta":"d, "}\n\nevent: result\ndata: line1\ndata: line2\n\nevent: end\ndata: {}\n\n';
  for(let i=0;i<wire.length;i+=3)feed(wire.slice(i,i+3));
  assert.deepEqual(seen,[{event:'token',data:'{"delta":"Re"}'},{event:'token',data:'{"delta":"d, "}'},{event:'result',data:'line1\nline2'},{event:'end',data:'{}'}]);
});

test('incomplete trailing frame waits; frames without data are ignored; default event name is message',()=>{
  const seen=[];const feed=sseParser(e=>seen.push(e));
  feed(': comment\n\ndata: x\n\nevent: token\ndata: partial');assert.deepEqual(seen,[{event:'message',data:'x'}]);
  feed('\n\n');assert.equal(seen.at(-1).data,'partial');
});
