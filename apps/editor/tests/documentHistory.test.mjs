import { test } from 'node:test';
import assert from 'node:assert/strict';
import { initialHistory, reduceHistory, HISTORY_LIMIT, HISTORY_BYTES } from '../src/documentHistory.ts';
const edit = (h,transaction,update) => reduceHistory(h,{type:'edit',transaction,update});
const move = (h,type) => reduceHistory(h,{type});
test('paired graph/layout edit restores one complete document and redo invalidates on new edit', () => {
  const original={graph:{nodes:['source'],edges:[]},ui:{positions:{source:[0,0]}}};
  let h=initialHistory(original);
  h=edit(h,1,d=>({...d,graph:{...d.graph,nodes:[...d.graph.nodes,'new']}}));
  h=edit(h,1,d=>({...d,ui:{positions:{...d.ui.positions,new:[50,50]}}}));
  assert.equal(h.past.length,1);
  const modified=h.current;
  h=move(h,'undo'); assert.deepEqual(h.current,original);
  h=move(h,'redo'); assert.deepEqual(h.current,modified);
  h=move(h,'undo');h=edit(h,2,d=>({...d,graph:{nodes:['different'],edges:[]}}));
  assert.equal(h.future.length,0);assert.equal(move(h,'redo'),h);
  assert.deepEqual(original.graph.nodes,['source']);
});
test('no-op selection-equivalent updates preserve redo and reset prevents cross-project undo', () => {
  let h=edit(initialHistory({n:0}),1,d=>({n:d.n+1}));h=move(h,'undo');
  assert.equal(edit(h,2,d=>({...d})),h);
  h=reduceHistory(h,{type:'reset',value:{n:42}});
  assert.equal(h.past.length,0);assert.equal(h.future.length,0);assert.equal(move(h,'undo'),h);
});
test('ordered multiple undo/redo and bounded oldest eviction', () => {
  let h=initialHistory({n:0});
  for(let i=1;i<=HISTORY_LIMIT+3;i++)h=edit(h,i,()=>({n:i}));
  assert.equal(h.past.length,HISTORY_LIMIT);
  for(let i=0;i<HISTORY_LIMIT;i++)h=move(h,'undo');
  assert.equal(h.current.n,3);
  for(let i=0;i<HISTORY_LIMIT;i++)h=move(h,'redo');
  assert.equal(h.current.n,HISTORY_LIMIT+3);
});
test('large documents evict snapshot history rather than refuse draft changes', () => {
  const big='x'.repeat(HISTORY_BYTES/2);
  const h=edit(initialHistory({n:0}),1,()=>({n:1,big}));
  assert.equal(h.current.big,big);assert.equal(h.past.length,0);
});
test('intermediate gesture frames and terminal frame undo as one layout transaction', () => {
  let h=initialHistory({graph:{nodes:['n']},ui:{positions:{n:{x:0,y:0}}}});
  const original=h.current;
  for(let i=1;i<=20;i++)h=edit(h,7,d=>({...d,ui:{positions:{n:{x:i,y:i*2}}}}));
  assert.equal(h.past.length,1);
  assert.deepEqual(move(h,'undo').current,original);
  assert.deepEqual(move(move(h,'undo'),'redo').current,h.current);
});
