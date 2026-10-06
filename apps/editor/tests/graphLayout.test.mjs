import {test} from 'node:test';
import assert from 'node:assert/strict';
import {layeredLayout} from '../src/graphLayout.ts';
const C=(id,x=0,y=0,width=180,height=60)=>({id,key:'k/'+id,x,y,width,height});
const L=(from,to)=>({from,to});
const overlaps=(cards,pos)=>{for(let i=0;i<cards.length;i++)for(let j=i+1;j<cards.length;j++){const a=pos[cards[i].key],b=pos[cards[j].key];
  if(a.x<b.x+cards[j].width&&b.x<a.x+cards[i].width&&a.y<b.y+cards[j].height&&b.y<a.y+cards[i].height)return true;}return false;};

test('chain is laid out left to right from the current top-left corner, keyed by layout key',()=>{
  const cards=[C('c',500,40),C('a',900,300),C('b',100,10)];
  const pos=layeredLayout(cards,[L('a','b'),L('b','c')]);
  assert.deepEqual(pos,{'k/a':{x:100,y:10},'k/b':{x:360,y:10},'k/c':{x:620,y:10}});
});

test('branches stack by measured heights, layers space by widest card, nothing overlaps',()=>{
  const cards=[C('in'),C('wide',0,0,400,90),C('tall',0,0,120,200),C('out')];
  const pos=layeredLayout(cards,[L('in','wide'),L('in','tall'),L('wide','out'),L('tall','out')],80,40);
  assert.equal(pos['k/wide'].x,pos['k/tall'].x);assert.equal(pos['k/out'].x,0+180+80+400+80);
  assert.equal(Math.abs(pos['k/tall'].y-pos['k/wide'].y),pos['k/tall'].y>pos['k/wide'].y?90+40:200+40);
  assert(!overlaps(cards,pos));
  for(const [a,b] of [['in','wide'],['wide','out']])assert(pos['k/'+a].x+cards.find(c=>c.id===a).width<=pos['k/'+b].x);
});

test('cycles, self loops, duplicates, unknown endpoints and isolated cards are handled deterministically',()=>{
  const cards=[C('a'),C('b'),C('c'),C('alone')];
  const links=[L('a','b'),L('b','c'),L('c','a'),L('b','b'),L('a','b'),L('a','ghost')];
  const first=layeredLayout(cards,links),second=layeredLayout(structuredClone(cards),structuredClone(links));
  assert.deepEqual(first,second);assert(!overlaps(cards,first));
  assert(first['k/a'].x<first['k/b'].x&&first['k/b'].x<first['k/c'].x);assert.equal(first['k/alone'].x,0);
});

test('barycenter ordering removes an avoidable crossing',()=>{
  // Document order puts y1 above y0 while their parents are x0 above x1.
  const cards=[C('x0'),C('x1'),C('y1'),C('y0')];
  const pos=layeredLayout(cards,[L('x0','y0'),L('x1','y1')]);
  assert(pos['k/x0'].y<pos['k/x1'].y);assert(pos['k/y0'].y<pos['k/y1'].y);
});

test('refuses missing measurements, invalid positions, duplicates, empty and oversized scopes',()=>{
  assert.throws(()=>layeredLayout([C('a',0,0,NaN)],[]),/E_LAYOUT_MEASUREMENT/);
  assert.throws(()=>layeredLayout([C('a',NaN)],[]),/E_LAYOUT_POSITION/);
  assert.throws(()=>layeredLayout([C('a'),C('a')],[]),/E_LAYOUT_SELECTION/);
  assert.throws(()=>layeredLayout([],[]),/E_LAYOUT_SELECTION/);
  assert.throws(()=>layeredLayout(Array.from({length:1001},(_,i)=>C('n'+i)),[]),/E_LAYOUT_SELECTION/);
});

test('1000-node chain stays linear-time enough and overlap free',()=>{
  const cards=Array.from({length:1000},(_,i)=>C('n'+i,(i%25)*220,Math.floor(i/25)*130));
  const links=cards.slice(1).map((c,i)=>L(cards[i].id,c.id));
  const start=performance.now();const pos=layeredLayout(cards,links);const ms=performance.now()-start;
  assert.equal(pos['k/n999'].x,999*(180+80));assert(ms<2000,String(ms));
});
