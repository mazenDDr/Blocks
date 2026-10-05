import {test} from 'node:test';
import assert from 'node:assert/strict';
import {diagnosticTarget} from '../src/diagnosticNavigation.ts';
const node=(id,type='diag.probe',config={})=>({id,type,config,version:'1.0.0'});
const leaf={id:'leaf',version:'1.0.0',nodes:[node('constructor')]};
const middle={id:'middle',version:'1.0.0',nodes:[node('loop','core.repeat',{module:'leaf',count:2})]};
const graph={graphKind:'model',backend:'pytorch',nodes:[node('call','core.composite',{module:'middle'}),node('pick','core.select',{then:{module:'leaf'},otherwise:{module:'middle'}}),node('plain')],modules:[middle,leaf]};
test('declared nested composite/repeat/select diagnostic paths resolve exact stored definition/node without edits',()=>{
 const before=structuredClone(graph);assert.deepEqual(diagnosticTarget({...graph,nodes:[node('loop','core.repeat',{module:'leaf',count:64})]},'loop/it63/constructor'),{node:'constructor',scopes:[{module:'leaf',version:'1.0.0',via:'loop/it63'}]});assert.deepEqual(diagnosticTarget(graph,'call/loop/it1/constructor'),{node:'constructor',scopes:[{module:'middle',version:'1.0.0',via:'call'},{module:'leaf',version:'1.0.0',via:'loop/it1'}]});
 assert.deepEqual(diagnosticTarget(graph,'pick/otherwise/loop/it0/constructor'),{node:'constructor',scopes:[{module:'middle',version:'1.0.0',via:'pick/otherwise'},{module:'leaf',version:'1.0.0',via:'loop/it0'}]});
 assert.deepEqual(diagnosticTarget(graph,'pick/then/constructor'),{node:'constructor',scopes:[{module:'leaf',version:'1.0.0',via:'pick/then'}]});
 assert.deepEqual(diagnosticTarget(graph,'constructor',{id:'leaf',version:'1.0.0'}),{node:'constructor',scopes:[]});assert.deepEqual(diagnosticTarget(graph,'plain'),{node:'plain',scopes:[]});assert.deepEqual(graph,before);
});
test('missing/ambiguous/recursive scopes and invalid repeat/branch/pseudo paths never navigate to a guessed node',()=>{
 for(const path of [null,'','call//loop','call/loop/constructor','call/loop/it2/constructor','call/loop/it01/constructor','pick/no/constructor','plain/constructor','__in:x','call/no'])assert.equal(diagnosticTarget(graph,path),null,path);
 assert.equal(diagnosticTarget({...graph,modules:[middle,leaf,leaf]},'pick/then/constructor'),null);
 assert.equal(diagnosticTarget(graph,'constructor',{id:'missing',version:'1.0.0'}),null);
 assert.equal(diagnosticTarget({...graph,nodes:[node('plain'),node('plain')]},'plain'),null);
 assert.equal(diagnosticTarget({...graph,modules:[{...middle,nodes:[node('again','core.composite',{module:'middle'})]}]},'call/again/plain'),null);
 assert.equal(diagnosticTarget({...graph,graphKind:'agent'},'plain'),null);
});
