import {test} from 'node:test';
import assert from 'node:assert/strict';
import {insertOnWire} from '../src/graphInsertion.ts';
const op={type:'diag.probe',version:'1.0.0',backend:'pytorch',graphKind:'model',inputs:['input'],outputs:['output'],inputKinds:{input:'tensor'},outputKinds:{output:'tensor'},defaults:{label:''}};
const graph={schemaVersion:'1.0.0',graphKind:'model',backend:'pytorch',nodes:[{id:'constructor',type:'core.tensor_input',version:'1.0.0',config:{}},{id:'dest',type:'pytorch.nn.relu',version:'1.0.0',config:{}}],edges:[{id:'selected',kind:'tensor',from:{node:'constructor',port:'output'},to:{node:'dest',port:'input'},extension:{SYNTHETIC:'original consumer metadata'}}],extension:{preserved:true}};
const ui={schemaVersion:'1.0.0',positions:{constructor:{x:10,y:30},dest:{x:90,y:50},probe_1:{x:200,y:20}},nodeComments:{probe_2:{text:'SYNTHETIC retained orphan'}},extension:{preserved:true}};
test('fresh identities avoid orphan notes/layout and wire IDs; metadata stays at original consumer; originals immutable',()=>{
 const g=structuredClone(graph),u=structuredClone(ui);const r=insertOnWire(graph,ui,'selected',op,'input','output');
 assert.equal(r.id,'probe_3');assert.deepEqual(r.ui.positions.probe_3,{x:50,y:40});assert.deepEqual(r.ui.nodeComments,ui.nodeComments);
 assert.deepEqual(r.graph.edges[1].extension,graph.edges[0].extension);assert(!Object.hasOwn(r.graph.edges[0],'extension'));
 assert.deepEqual(r.graph.extension,graph.extension);assert.deepEqual(graph,g);assert.deepEqual(ui,u);
 const again=insertOnWire(r.graph,r.ui,r.graph.edges[0].id,op,'input','output');assert.equal(again.id,'probe_4');assert.equal(new Set(again.graph.edges.map(e=>e.id)).size,again.graph.edges.length);
});
test('ambiguous producers, missing wires, wrong ports/backend/kind and nonfinite positions refuse with stable codes',()=>{
 const check=(g,u,o,e='selected',input='input')=>insertOnWire(g,u,e,o,input,'output');
 assert.throws(()=>check({...graph,edges:[...graph.edges,{...graph.edges[0],id:'duplicate'}]},ui,op),e=>e.code==='E_INSERT_WIRE');
 assert.throws(()=>check(graph,ui,op,'gone'),e=>e.code==='E_INSERT_WIRE');
 assert.throws(()=>check(graph,ui,op,'selected','unknown'),e=>e.code==='E_INSERT_PORT');
 assert.throws(()=>check(graph,ui,{...op,backend:'jax'}),e=>e.code==='E_INSERT_OPERATION');
 assert.throws(()=>check(graph,ui,{...op,inputKinds:{input:'table'}}),e=>e.code==='E_INSERT_KIND');
 assert.throws(()=>check(graph,{...ui,positions:{...ui.positions,dest:{x:Infinity,y:40}}},op),e=>e.code==='E_INSERT_POSITION');
});
