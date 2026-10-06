import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createGroup,updateGroup,deleteGroup,frames,groupsOf,renameMember,PAD,HEAD} from '../src/layoutGroups.ts';
const ui={schemaVersion:'1.0.0',positions:{},other:{kept:true}};
const ids=new Set(['a','b','c']);
const B=(id,x,y,w=100,h=50)=>({id,x,y,width:w,height:h});

test('create, rename, change members and delete are pure edits of the UI document only',()=>{
  const {ui:one,id}=createGroup(ui,'root','  Encoder  ',['a','b','a'],ids);
  assert.equal(id,'group_1');assert.deepEqual(groupsOf(one),[{id:'group_1',label:'Encoder',scope:'root',members:['a','b']}]);
  assert.deepEqual(one.other,{kept:true});assert.equal(groupsOf(ui).length,0);
  const two=updateGroup(one,id,{label:'Stem',members:['c']},ids);assert.deepEqual(groupsOf(two)[0],{id,label:'Stem',scope:'root',members:['c']});
  const {id:second}=createGroup(two,'module:m@1.0.0','Inner',['a'],ids);assert.equal(second,'group_2');
  assert.deepEqual(groupsOf(deleteGroup(two,id)),[]);
});

test('frames enclose measured member cards with padding and a title band, per scope, reporting missing members',()=>{
  let {ui:g}=createGroup(ui,'root','G',['a','b','c'],ids);
  g=createGroup(g,'module:m@1','Other',['a'],ids).ui;
  const f=frames(g,'root',[B('a',0,100),B('b',300,40,200,80)]);
  assert.deepEqual(f,[{id:'group_1',label:'G',x:-PAD,y:40-PAD-HEAD,width:500+2*PAD,height:(150-40)+2*PAD+HEAD,members:['a','b'],missing:['c']}]);
  assert.deepEqual(frames(g,'root',[]),[]);assert.equal(frames(g,'module:m@1',[B('a',0,0)])[0].label,'Other');
});

test('renaming a card keeps its membership only in the same scope',()=>{
  let {ui:g}=createGroup(ui,'root','G',['a'],ids);g=createGroup(g,'module:m@1','M',['a'],ids).ui;
  const r=renameMember(g,'root','a','z');assert.deepEqual(groupsOf(r).map(x=>x.members),[['z'],['a']]);
  assert.equal(renameMember(g,'root','nope','z'),g);
});

test('refuses empty labels, unknown members, empty groups, missing groups and malformed stored data is ignored',()=>{
  assert.throws(()=>createGroup(ui,'root',' ',['a'],ids),/E_GROUP_LABEL/);
  assert.throws(()=>createGroup(ui,'root','G',['x'],ids),/E_GROUP_MEMBERS/);
  assert.throws(()=>createGroup(ui,'root','G',[],ids),/E_GROUP_MEMBERS/);
  assert.throws(()=>updateGroup(ui,'group_9',{label:'x'},ids),/E_GROUP_MISSING/);
  assert.deepEqual(groupsOf({...ui,layoutGroups:[{id:1},null,{id:'g',label:'L',scope:'root',members:['a',2]},{id:'ok',label:'L',scope:'root',members:['a']}]}).map(g=>g.id),['ok']);
});
