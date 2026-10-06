import {test} from 'node:test';import assert from 'node:assert/strict';
import {rlOutline} from '../src/rlOutline.ts';
// SYNTHETIC helper contract fixture; not a training result or measured environment.
const graph={nodes:[{id:'constructor',type:'rl.environment',config:{env_id:'SYNTHETIC'}},{id:'q',type:'rl.q_network',config:{network:{nodes:[{id:'inner',type:'core.tensor_input'}]}}}],edges:[{id:'e',kind:'env',from:{node:'constructor',port:'env'},to:{node:'q',port:'env'}}]};
const report={graphHash:'a'.repeat(64),nodes:{constructor:{typed:true,inputShapes:{},outputShapes:{env:{kind:'env',observationSpace:{type:'Box',shape:[4]}}}},q:{typed:false}},diagnostics:[{nodeId:'q',severity:'error',code:'E_SYNTHETIC',message:'SYNTHETIC contract test only'}]};
test('RL native contracts, declared nested configuration/wires and exact unique targets are searchable without edits',()=>{
 const before=structuredClone({graph,report});const result=rlOutline(graph,{},report,false);assert.equal(result.graphHash,report.graphHash);assert.deepEqual(result.rows[0].nativeView,report.nodes.constructor);assert.equal(result.rows[0].panel,'env');assert.equal(result.rows[1].panel,'learner');assert.equal(result.rows[1].incoming[0],'constructor.env → env (env)');
 assert.deepEqual(rlOutline(graph,{},report,false,'observationSpace').rows.map(r=>r.id),['constructor']);assert.deepEqual(rlOutline(graph,{},report,false,'inner').rows.map(r=>r.id),['q']);assert.deepEqual(rlOutline(graph,{},report,false,'E_SYNTHETIC',true).rows.map(r=>r.id),['q']);assert.deepEqual({graph,report},before);
 const repeated={...graph,nodes:[...graph.nodes,{...graph.nodes[0],id:'other'}]};assert.equal(rlOutline(repeated,{},null,false).rows[0].panel,null);
 const duplicate={...graph,nodes:[...graph.nodes,graph.nodes[0]]};assert(rlOutline(duplicate,{},report,false).rows.filter(r=>r.id==='constructor').every(r=>r.node===null&&r.panel===null));
});
test('pending reports withhold prior RL contracts/errors/hash but preserve declared configuration and exact own-property handling',()=>{
 const result=rlOutline(graph,{},report,true);assert.equal(result.graphHash,null);assert(result.rows.every(r=>r.nativeView===null&&r.diagnostics.length===0));assert.equal(rlOutline(graph,{},report,true,'observationSpace').rows.length,0);assert.equal(rlOutline(graph,{},report,true,'inner').rows.length,1);assert.equal(rlOutline(graph,{},report,true,'',true).rows.length,0);
 assert.equal(rlOutline(graph,{}, {...report,nodes:{}},false).rows[0].nativeView,null);assert.equal(rlOutline(graph,{},report,false,'[not a regex').rows.length,0);
});
