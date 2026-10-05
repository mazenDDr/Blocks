import {test} from 'node:test';import assert from 'node:assert/strict';
import {agentOutline} from '../src/agentOutline.ts';
const g={graphKind:'agent',backend:'langgraph',nodes:[{id:'constructor',type:'agent.set_state',config:{}},{id:'next',type:'agent.set_state',config:{}}],edges:[{id:'entry',kind:'control',from:{node:'START',port:'out'},to:{node:'constructor',port:'in'}}],agent:{routes:[{id:'route',from:'constructor',cases:[{id:'yes',label:'SYNTHETIC declared condition',when:{field:'n',op:'>',value:0},to:'next'}],default:'END'}],joins:[{node:'next',waitFor:['constructor']}]}};
const diagnostic={code:'E_SYNTHETIC',severity:'error',nodeId:'next',message:'SYNTHETIC contract test only'};
const report={graphHash:'a'.repeat(64),nodes:{constructor:{typed:true,reads:['n'],writes:['answer'],effects:['file_write']},next:{typed:false,reads:[],writes:[],effects:[]}},diagnostics:[diagnostic]};
test('declared routes/joins/control wires and actual reads/writes/effects searchable without changing native/graph data',()=>{
 const before=structuredClone({g,report});const rows=agentOutline(g,{},report,false).rows;assert.deepEqual(rows[0].routes,g.agent.routes);assert.deepEqual(rows[1].joins,g.agent.joins);assert.deepEqual(rows[0].native,report.nodes.constructor);assert.equal(rows[0].incoming[0],'START.out → in (control)');
 assert.deepEqual(agentOutline(g,{},report,false,'file_write').rows.map(r=>r.id),['constructor']);assert.deepEqual(agentOutline(g,{},report,false,'answer').rows.map(r=>r.id),['constructor']);assert.deepEqual(agentOutline(g,{},report,false,'SYNTHETIC declared condition').rows.map(r=>r.id),['constructor']);assert.deepEqual(agentOutline(g,{},report,false,'E_SYNTHETIC',true).rows.map(r=>r.id),['next']);assert.deepEqual({g,report},before);
});
test('pending agent reports withhold native fields/diagnostics/hash but retain explicit declared structure and refuse literal misses',()=>{
 const result=agentOutline(g,{},report,true);assert.equal(result.graphHash,null);assert(result.rows.every(r=>r.native===null&&r.diagnostics.length===0));assert.deepEqual(result.rows[0].routes,g.agent.routes);assert.deepEqual(agentOutline(g,{},report,true,'',true).rows,[]);assert.deepEqual(agentOutline(g,{},report,false,'%').rows,[]);
});
