import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {copyModuleNodes,pasteModuleNodes} from '../src/moduleClipboard.ts';
import {copyGraphNodes,pasteGraphNodes} from '../src/graphClipboard.ts';
const bundled=JSON.parse(fs.readFileSync(new URL('../../../examples/residual_cnn.project.json',import.meta.url),'utf8'));
const g=bundled,d=g.modules[0],prefix=d.id+'@'+d.version+'/',hash='a'.repeat(64);
const ui={schemaVersion:"1.0.0",positions:{[prefix+'conv_a']:{x:12,y:24},[prefix+'conv_a_copy']:{x:7,y:8}},nodeComments:{[prefix+'conv_a']:{text:'SYNTHETIC original note'},[prefix+'conv_a_copy_2']:{text:'SYNTHETIC orphan note'}},unknown:{retained:true}};
test('module-local internal wires and actual scoped origins transfer, boundaries omitted and native source scope declared',()=>{
 const before=structuredClone(g),copy=copyModuleNodes(g,ui,d,['conv_a','relu_a'],'SYNTHETIC module',hash);
 assert.equal(copy.sourceGraphHash,hash);assert.deepEqual(copy.sourceScope,{kind:'module',id:d.id,version:d.version});
 assert.deepEqual(copy.nodes.map(n=>n.id),['conv_a','relu_a']);assert.equal(copy.edges.length,1);assert.equal(copy.omittedBoundaryEdges,2);assert.deepEqual(copy.positions.conv_a,{x:12,y:24});
 const result=pasteModuleNodes(g,ui,d,copy);assert.equal(result.mapping.conv_a,'conv_a_copy_3');const next=result.graph.modules[0];
 assert.deepEqual(result.graph.nodes,g.nodes);assert.deepEqual(result.graph.edges,g.edges);assert.deepEqual(next.inputs,d.inputs);assert.deepEqual(next.outputs,d.outputs);assert.deepEqual(next.params,d.params);assert.deepEqual(next.edges.slice(0,d.edges.length),d.edges);assert.deepEqual(next.nodes.slice(0,d.nodes.length),d.nodes);
 assert.deepEqual(next.edges.at(-1).from,{node:'conv_a_copy_3',port:'output'});assert.deepEqual(next.edges.at(-1).to,{node:'relu_a_copy',port:'input'});
 assert.deepEqual(result.ui.positions[prefix+'conv_a_copy_3'],{x:92,y:104});assert.deepEqual(result.ui.nodeComments,ui.nodeComments);assert.deepEqual(result.ui.unknown,ui.unknown);assert.deepEqual(g,before);
 assert(!next.edges.some(e=>e.from.node==='$in'&&e.to.node==='conv_a_copy_3'));
});
test('module output boundaries omitted, pseudo/state/ambiguous scopes refuse and root notes/layout reserve fresh identities',()=>{
 const output=copyModuleNodes(g,ui,d,['relu_out'],'SYNTHETIC module',hash);assert.equal(output.omittedBoundaryEdges,2);
 assert.throws(()=>copyModuleNodes(g,ui,d,['__in:x'],'SYNTHETIC',hash),e=>e.code==='E_CLIPBOARD_SELECTION');
 assert.throws(()=>copyModuleNodes({...g,modules:[d,d]},ui,d,['conv_a'],'SYNTHETIC',hash),e=>e.code==='E_CLIPBOARD_SCOPE');
 assert.throws(()=>copyModuleNodes({...g,modules:[{...d,nodes:d.nodes.map(n=>n.id==='conv_a'?{...n,stateRef:'model/conv_a'}:n)}]},ui,d,['conv_a'],'SYNTHETIC',hash),e=>e.code==='E_CLIPBOARD_STATE_REF');
 const root={...g,nodes:[{id:'constructor',type:'core.tensor_input',version:'1.0.0',config:{shape:['N',4],dtype:'float32'}}],edges:[]};
 const rootUi={schemaVersion:'1.0.0',positions:{constructor_copy:{x:4,y:8}},nodeComments:{constructor_copy_2:{text:'SYNTHETIC orphan'}}};const copied=copyGraphNodes(root,rootUi,['constructor'],'SYNTHETIC',hash),pasted=pasteGraphNodes(root,rootUi,copied);
 assert.deepEqual(copied.positions.constructor,{x:60,y:120});assert.equal(pasted.mapping.constructor,'constructor_copy_3');assert.deepEqual(pasted.ui.nodeComments,rootUi.nodeComments);
});
test('valid constructor module ID uses actual topology fallback rather than inherited object properties',()=>{
 const rename=id=>id==='conv_a'?'constructor':id;
 const definition={...d,nodes:d.nodes.map(n=>({...n,id:rename(n.id)})),edges:d.edges.map(e=>({...e,from:{...e.from,node:rename(e.from.node)},to:{...e.to,node:rename(e.to.node)}}))};
 const graph={...g,modules:[definition]};const copy=copyModuleNodes(graph,{schemaVersion:'1.0.0',positions:{}},definition,['constructor'],'SYNTHETIC',hash);
 assert.deepEqual(copy.positions.constructor,{x:290,y:40});assert.equal(copy.omittedBoundaryEdges,2);
});
