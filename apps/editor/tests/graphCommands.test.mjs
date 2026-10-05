import {test} from 'node:test';
import assert from 'node:assert/strict';
import {graphCommands,workspaceChoices,findGraphCommands} from '../src/graphCommands.ts';
const graph={graphKind:'model',backend:'pytorch',nodes:[{id:'constructor',type:'core.tensor_input',version:'1.0.0',config:{}}],edges:[]};
const op={type:'diag.probe',version:'1.0.0',backend:'pytorch',graphKind:'model',displayName:'SYNTHETIC Straße probe',purpose:'Actual catalogue test fixture only; no native execution claim'};
test('catalogue exposes actual supported workspace/scopes and current history, with stable target IDs and backend labels',()=>{
 const cmds=graphCommands(graph,graph,[op,{...op,type:'jax.future',backend:'jax'}],false,true,true);
 assert(cmds.some(c=>c.id==='inspect:constructor'&&c.node==='constructor'));
 assert(cmds.some(c=>c.id==='create:diag.probe'&&c.detail.includes('pytorch')));assert(!cmds.some(c=>c.id==='create:jax.future'));
 assert(cmds.find(c=>c.id==='undo').disabled);assert(!cmds.find(c=>c.id==='redo').disabled);assert(cmds.some(c=>c.id==='root'));
 const agent=graphCommands({...graph,graphKind:'agent'},graph,[op],false,false,false);
 assert(!agent.some(c=>['create','inspect'].includes(c.action)));assert.deepEqual(workspaceChoices('rl').map(c=>c[0]),['scale','production','records','graph']);
 assert(!workspaceChoices('tabular').some(([key])=>key==='training'));assert(workspaceChoices('domain').some(([key])=>key==='domain'));
});
test('literal Unicode search does not invent semantic/regex matches or edit the original catalogue',()=>{
 const cmds=graphCommands(graph,graph,[op],true,false,false),snapshot=structuredClone(cmds);
 assert.equal(findGraphCommands(cmds,'straße')[0].id,'create:diag.probe');assert.equal(findGraphCommands(cmds,'STRASSE').length,0);
 assert.equal(findGraphCommands(cmds,'.*').length,0);assert.equal(findGraphCommands(cmds,'core.tensor_input')[0].node,'constructor');
 assert.deepEqual(cmds,snapshot);
});


test('actual reusable definition roundtrip and disconnected append preserve all original interface/edge metadata and identities',async()=>{
 const fs=await import('node:fs');const {defToView,viewToDef}=await import('../src/modules.ts');
 const fixture=JSON.parse(fs.readFileSync(new URL('../../../examples/residual_cnn.project.json',import.meta.url),'utf8'));
 const definition=fixture.modules[0],snapshot=structuredClone(definition),view=defToView(definition);
 assert.deepEqual(viewToDef(view,definition),definition);
 const next=viewToDef({...view,nodes:[...view.nodes,{id:'SYNTHETIC_probe',type:'diag.probe',version:'1.0.0',config:{}}]},definition);
 assert.deepEqual(next.edges,definition.edges);assert.deepEqual(next.inputs,definition.inputs);assert.deepEqual(next.outputs,definition.outputs);
 assert.deepEqual(definition,snapshot);
});
