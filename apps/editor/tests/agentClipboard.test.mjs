import {test} from 'node:test';
import assert from 'node:assert/strict';
import {copyAgentNodes,pasteAgentNodes} from '../src/agentClipboard.ts';
const field=(name,type='text')=>({name,type,reducer:{kind:'replace'},scope:'turn'});
const N=(id,config={})=>({id,type:'agent.set_state',version:'1.0.0',config});
const E=(a,b)=>({id:`${a}__${b}`,kind:'control',from:{node:a,port:'out'},to:{node:b,port:'in'}});
const graph={schemaVersion:'1.0.0',graphKind:'agent',backend:'langgraph',nodes:[N('plan'),N('check'),N('answer',{index:'docs'}),N('outside')],
  edges:[E('START','plan'),E('plan','check'),E('answer','END'),E('outside','plan')],
  agent:{state:[field('question'),field('draft'),field('score','number'),field('unused')],
    routes:[{id:'route_check',from:'check',cases:[{id:'good',label:'',when:{all:[{field:'score',op:'gt',value:0.5}]},to:'answer'}],default:'plan',defaultLabel:'retry'},
      {id:'route_out',from:'outside',cases:[{id:'c',label:'',when:{always:true},to:'check'}],default:'END'}],
    joins:[{node:'answer',waitFor:['plan','check']},{node:'plan',waitFor:['outside']}],limits:{maxSteps:25},
    indexes:[{id:'docs',loader:{directory:'d',glob:'*.md'},splitter:{strategy:'fixed',chunkSize:100,chunkOverlap:0},embeddings:{provider:'local_hash',model:'h',dimension:8,normalize:true}}],policies:[]}};
const ui={schemaVersion:'1.0.0',positions:{plan:{x:10,y:20},check:{x:300,y:20}}};
const access={plan:{reads:['question'],writes:['draft']},check:{reads:['draft'],writes:['score']},answer:{reads:['draft.text'],writes:[]},outside:{reads:[],writes:['unused']}};
const copy=(ids=['plan','check','answer'],g=graph)=>copyAgentNodes(g,ui,ids,access,'SYNTHETIC source','a'.repeat(64));

test('copies internal transitions, routes, joins and the exact state/index definitions the selection uses',()=>{
  const original=structuredClone(graph);const clip=copy();
  assert.deepEqual(clip.edges.map(e=>e.id),['plan__check','answer__END']);
  assert.deepEqual(clip.routes.map(r=>r.id),['route_check']);assert.deepEqual(clip.joins,[{node:'answer',waitFor:['plan','check']}]);
  assert.deepEqual(clip.state.map(f=>f.name),['draft','question','score']);assert.deepEqual(clip.indexes.map(i=>i.id),['docs']);
  assert.deepEqual(clip.omitted,{transitions:2,routes:1,joins:1});
  assert.deepEqual(graph,original);
});

test('paste uses fresh ids, rebinds routes/joins/transitions to copies and keeps END; source graph unchanged',()=>{
  const clip=copy();const before=structuredClone(graph);const pasted=pasteAgentNodes(graph,ui,clip);
  assert.deepEqual(pasted.mapping,{plan:'plan_copy',check:'check_copy',answer:'answer_copy'});
  const route=pasted.graph.agent.routes.at(-1);
  assert.equal(route.from,'check_copy');assert.equal(route.cases[0].to,'answer_copy');assert.equal(route.default,'plan_copy');assert.notEqual(route.id,'route_check');
  assert.deepEqual(pasted.graph.agent.joins.at(-1),{node:'answer_copy',waitFor:['plan_copy','check_copy']});
  assert.deepEqual(pasted.graph.edges.slice(-2).map(e=>[e.id,e.from.node,e.to.node]),[['plan_copy__check_copy','plan_copy','check_copy'],['answer_copy__END','answer_copy','END']]);
  assert.equal(pasted.graph.agent.state.length,4);assert.equal(pasted.graph.agent.indexes.length,1);
  assert.deepEqual(pasted.ui.positions.plan_copy,{x:90,y:100});assert.deepEqual(graph,before);
});

test('paste into another agent graph adds missing definitions and refuses differing ones',()=>{
  const clip=copy();
  const empty={schemaVersion:'1.0.0',graphKind:'agent',backend:'langgraph',nodes:[],edges:[]};
  const pasted=pasteAgentNodes(empty,{schemaVersion:'1.0.0',positions:{}},clip);
  assert.deepEqual(pasted.graph.agent.state.map(f=>f.name),['draft','question','score']);assert.equal(pasted.graph.agent.indexes[0].id,'docs');
  assert.deepEqual(pasted.mapping.plan,'plan_copy');
  const conflicting={...empty,agent:{state:[field('score','text')]}};
  assert.throws(()=>pasteAgentNodes(conflicting,{schemaVersion:'1.0.0',positions:{}},clip),/E_CLIPBOARD_DEFINITION_CONFLICT/);
});

test('refuses terminals, unknown nodes, missing native access, undeclared fields and other graph kinds',()=>{
  assert.throws(()=>copy(['START']),/E_CLIPBOARD_SELECTION/);assert.throws(()=>copy(['nope']),/E_CLIPBOARD_SELECTION/);
  assert.throws(()=>copyAgentNodes(graph,ui,['plan'],{},'s','h'),/E_CLIPBOARD_PENDING/);
  const missing=structuredClone(graph);missing.agent.state=missing.agent.state.filter(f=>f.name!=='draft');
  assert.throws(()=>copy(['plan'],missing),/E_CLIPBOARD_STATE/);
  assert.throws(()=>copy(['plan'],{...graph,graphKind:'model'}),/E_CLIPBOARD_KIND/);
  assert.throws(()=>pasteAgentNodes({...graph,graphKind:'model'},ui,copy()),/E_CLIPBOARD_KIND/);
});

test('repeated paste never reuses ids',()=>{
  const clip=copy(['plan']);const once=pasteAgentNodes(graph,ui,clip);const twice=pasteAgentNodes(once.graph,once.ui,clip);
  assert.equal(twice.mapping.plan,'plan_copy_2');
});
