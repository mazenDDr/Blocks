import {test} from 'node:test';
import assert from 'node:assert/strict';
import {copyGraphNodes,pasteGraphNodes} from '../src/graphClipboard.ts';
const N=(id,sharedWith=null)=>({id,type:'pytorch.nn.linear',version:'1.0.0',config:{in_features:4,out_features:4},sharedWith,stateRef:null,unknown:{preserved:true}});
const graph={schemaVersion:'1.0.0',graphKind:'model',backend:'pytorch',nodes:[N('a'),N('b','a'),N('outside')],edges:[{id:'inside',kind:'tensor',from:{node:'a',port:'output'},to:{node:'b',port:'input'}},{id:'boundary',kind:'tensor',from:{node:'outside',port:'output'},to:{node:'a',port:'input'}}]};
const ui={schemaVersion:'1.0.0',positions:{a:{x:10,y:20},b:{x:50,y:20}}};
const copy=(g=graph,ids=['a','b'])=>copyGraphNodes(g,ui,ids,'SYNTHETIC source','a'.repeat(64));
test('copies internal typed wires/config/unknown fields and rebinds independent sharing; source is immutable',()=>{
  const original=structuredClone(graph);const clip=copy();assert.equal(clip.omittedBoundaryEdges,1);assert.equal(clip.edges.length,1);
  const pasted=pasteGraphNodes(graph,ui,clip);
  assert.equal(pasted.mapping.a,'a_copy');assert.equal(pasted.mapping.b,'b_copy');
  assert.equal(pasted.graph.nodes.at(-1).sharedWith,'a_copy');assert.deepEqual(pasted.graph.nodes.at(-1).unknown,{preserved:true});
  assert.deepEqual(pasted.graph.edges.at(-1).from,{node:'a_copy',port:'output'});assert.deepEqual(pasted.ui.positions.a_copy,{x:90,y:100});
  clip.nodes[0].config.out_features=100;assert.deepEqual(graph,original);assert.equal(pasted.graph.nodes.at(-2).config.out_features,4);
});
test('repeated paste generates collision-free node/edge identities and leaves boundary disconnected',()=>{
  const clip=copy();const first=pasteGraphNodes(graph,ui,clip);const second=pasteGraphNodes(first.graph,first.ui,clip,160);
  assert.equal(second.mapping.a,'a_copy_2');assert.equal(new Set(second.graph.edges.map(e=>e.id)).size,second.graph.edges.length);
  assert(!second.graph.edges.some(e=>e.from.node==='outside'&&e.to.node==='a_copy_2'));
});
test('external sharing/opaque state/generated selections/incompatible kinds refuse',()=>{
  assert.throws(()=>copy(graph,['b']),e=>e.code==='E_CLIPBOARD_SHARE');
  assert.throws(()=>copy({...graph,nodes:[{...N('a'),stateRef:'opaque'}]},['a']),e=>e.code==='E_CLIPBOARD_STATE_REF');
  assert.throws(()=>copy(graph,['generated/child']),e=>e.code==='E_CLIPBOARD_SELECTION');
  assert.throws(()=>copy({...graph,graphKind:'agent'}),e=>e.code==='E_CLIPBOARD_KIND');
  assert.throws(()=>pasteGraphNodes({...graph,backend:'jax'},ui,copy()),e=>e.code==='E_CLIPBOARD_KIND');
});
test('transitive module/code definitions preserve provenance and conflicting versions refuse',()=>{
  const code={id:'native',version:'1.0.0',source:'SYNTHETIC pinned code',origin:{commit:'SYNTHETIC-test-origin-marker'},fixtures:[]};
  const leaf={id:'leaf',version:'1.0.0',nodes:[{...N('inside'),type:'code.block',config:{block:'native',version:'1.0.0'}}],edges:[]};
  const root={id:'root',version:'1.0.0',nodes:[{...N('call'),type:'core.composite',config:{module:'leaf'}}],edges:[]};
  const g={...graph,nodes:[{...N('call'),type:'core.composite',config:{module:'root'}}],edges:[],modules:[root,leaf],codeBlocks:[code]};
  const clip=copy(g,['call']);assert.deepEqual(clip.modules.map(m=>m.id),['root','leaf']);assert.equal(clip.codeBlocks[0].origin.commit,'SYNTHETIC-test-origin-marker');
  const pasted=pasteGraphNodes({...g,nodes:[],modules:[],codeBlocks:[]},ui,clip);assert.equal(pasted.graph.modules.length,2);assert.equal(pasted.graph.codeBlocks.length,1);
  assert.throws(()=>pasteGraphNodes({...g,modules:[{...root,description:'different'},leaf]},ui,clip),e=>e.code==='E_CLIPBOARD_DEFINITION_CONFLICT');
  assert.throws(()=>copy({...g,modules:[root]},['call']),e=>e.code==='E_CLIPBOARD_DEFINITION');
});
test('copy payload is bounded; unknown op values are preserved without execution claim',()=>{
  assert.throws(()=>copy({...graph,nodes:[{...N('a'),config:{huge:'x'.repeat(300000)}}]},['a']),e=>e.code==='E_CLIPBOARD_BOUNDS');
  const clip=copy({...graph,nodes:[{...N('a'),type:'future.unknown',config:{declared:7}}],edges:[]},['a']);
  assert.equal(pasteGraphNodes({...graph,nodes:[],edges:[]},ui,clip).graph.nodes[0].type,'future.unknown');
});
test('declared local draft state references rebind and package identities never disappear or conflict silently',()=>{
  const dependency={operation:'SYNTHETIC.plugin',version:'1.0.0',implementation:{sha256:'a'.repeat(64)}};
  const g={...graph,nodes:[{...N('a'),stateRef:'model/a'}],edges:[],packageDependencies:[dependency]};
  const clip=copy(g,['a']);const pasted=pasteGraphNodes({...g,nodes:[],packageDependencies:[]},ui,clip);
  assert.equal(pasted.graph.nodes[0].stateRef,'model/a_copy');assert.deepEqual(pasted.graph.packageDependencies,[dependency]);
  assert.throws(()=>pasteGraphNodes({...g,packageDependencies:[{...dependency,version:'2.0.0'}]},ui,clip),e=>e.code==='E_CLIPBOARD_DEFINITION_CONFLICT');
});
test('required module state remains opaque and refuses rather than guessing a rebind',()=>{
  const def={id:'m',version:'1.0.0',nodes:[{...N('inner'),stateRef:'model/inner'}],edges:[]};
  const g={...graph,nodes:[{...N('call'),type:'core.composite',config:{module:'m'}}],edges:[],modules:[def]};
  assert.throws(()=>copy(g,['call']),e=>e.code==='E_CLIPBOARD_STATE_REF');
});
test('composite instance sharing requires the source selection and rebinds the target',()=>{
  const def={id:'m',version:'1.0.0',nodes:[],edges:[]};
  const g={...graph,nodes:[{...N('a'),type:'core.composite',config:{module:'m',share:'clone'}},{...N('b'),type:'core.composite',sharedWith:null,config:{module:'m',share:'a'}}],edges:[],modules:[def]};
  assert.throws(()=>copy(g,['b']),e=>e.code==='E_CLIPBOARD_SHARE');
  assert.equal(pasteGraphNodes(g,ui,copy(g)).graph.nodes.at(-1).config.share,'a_copy');
});
test('missing saved positions use the actual root canvas fallback at original graph index',()=>{
  const clip=copyGraphNodes(graph,{schemaVersion:'1.0.0',positions:{}},['outside'],'SYNTHETIC source','a'.repeat(64));
  assert.deepEqual(clip.positions.outside,{x:580,y:120});
  assert.deepEqual(pasteGraphNodes(graph,ui,clip).ui.positions.outside_copy,{x:660,y:200});
});
