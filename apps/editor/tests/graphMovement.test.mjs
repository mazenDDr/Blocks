import {test} from 'node:test';
import assert from 'node:assert/strict';
import {moveGraphNodes} from '../src/graphMovement.ts';
test('same finite displacement preserves independently checked relative origins and original scoped data',()=>{
 const points=[{id:'constructor',key:'m@1/constructor',x:-40,y:0},{id:'next',key:'m@1/next',x:180,y:500}],snapshot=structuredClone(points);
 const result=moveGraphNodes(points,23.125,-80);
 assert.deepEqual(result,{'m@1/constructor':{x:-16.875,y:-80},'m@1/next':{x:203.125,y:420}});
 assert.equal(result['m@1/next'].x-result['m@1/constructor'].x,220);assert.equal(result['m@1/next'].y-result['m@1/constructor'].y,500);
 assert.deepEqual(points,snapshot);assert.deepEqual(moveGraphNodes(points,0,0),{});
});
test('ambiguous/empty/oversized selection and nonfinite/out-of-bounds layout refuse without partial result',()=>{
 const p={id:'a',key:'a',x:0,y:10};
 assert.throws(()=>moveGraphNodes([],1,2),e=>e.code==='E_MOVE_SELECTION');assert.throws(()=>moveGraphNodes([p,p],1,2),e=>e.code==='E_MOVE_SELECTION');
 assert.throws(()=>moveGraphNodes(Array.from({length:101},(_,i)=>({...p,id:String(i),key:String(i)})),1,2),e=>e.code==='E_MOVE_SELECTION');
 assert.throws(()=>moveGraphNodes([p],NaN,0),e=>e.code==='E_MOVE_DELTA');assert.throws(()=>moveGraphNodes([p],1_000_001,0),e=>e.code==='E_MOVE_DELTA');
 assert.throws(()=>moveGraphNodes([{...p,x:1_000_000}],1,0),e=>e.code==='E_MOVE_POSITION');assert.throws(()=>moveGraphNodes([{...p,y:Infinity}],1,0),e=>e.code==='E_MOVE_POSITION');
});
