import {test} from 'node:test';
import assert from 'node:assert/strict';
import {graphOutline} from '../src/graphOutline.ts';
const graph={nodes:[{id:'source',type:'core.tensor_input',config:{}},{id:'m',type:'core.composite',config:{module:'encoder',version:'2.0.0',share:'other'}}],edges:[{id:'e',kind:'tensor',from:{node:'source',port:'value'},to:{node:'m',port:'x'}}]};
const error={nodeId:'m/inner',code:'E_SYNTHETIC',severity:'error',message:'SYNTHETIC declared error'};
const report={graphHash:'a'.repeat(64),nodes:{source:{typed:true,outputShapes:{value:{kind:'tensor',shape:[2,4]}}}},diagnostics:[error,{...error,nodeId:null,message:'SYNTHETIC global error'}]};
test('outline links actual typed wires, structural identities, native shapes and nested diagnostics',()=>{
  const view=graphOutline(graph,{'core.tensor_input':{displayName:'Tensor input'}},report,false);
  assert.equal(view.rows[0].title,'Tensor input');assert.equal(view.rows[0].outgoing[0],'value → m.x (tensor)');
  assert.equal(view.rows[1].incoming[0],'source.value → x (tensor)');assert.equal(view.rows[1].module,'encoder@2.0.0');assert.equal(view.rows[1].sharing,'other');
  assert.deepEqual(view.rows[1].diagnostics,[error]);assert.equal(view.rows[0].nativeView,report.nodes.source);assert.equal(view.globalDiagnostics.length,1);
});
test('pending validation never exposes stale errors, shapes or provenance',()=>{
  const view=graphOutline(graph,{},report,true);assert.equal(view.graphHash,null);assert.deepEqual(view.globalDiagnostics,[]);
  assert(view.rows.every(row=>row.nativeView===null&&row.diagnostics.length===0));assert.deepEqual(graphOutline(graph,{},report,true,'',true).rows,[]);
});
test('literal search and native-error filter preserve graph order without changing any source data',()=>{
  const before=JSON.stringify({graph,report});assert.deepEqual(graphOutline(graph,{},report,false,'encoder',true).rows.map(n=>n.id),['m']);
  assert.deepEqual(graphOutline(graph,{},report,false,'E_SYNTHETIC').rows.map(n=>n.id),['m']);
  assert.deepEqual(graphOutline(graph,{},report,false,'%').rows,[]);assert.equal(JSON.stringify({graph,report}),before);
});
