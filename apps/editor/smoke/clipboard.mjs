// Real owned Chrome, real saved drafts and API validation. No invented graph evidence.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const evidence={status:'failed',fixture:'SYNTHETIC native graph clipboard',runtimeErrors:[],apiErrors:[],consoleWarnings:[]};
const browser=await puppeteer.launch({executablePath:process.env.VOID_SMOKE_CHROME,headless:true,
  args:process.platform==='linux'?['--no-sandbox']:[],defaultViewport:{width:1600,height:1100}});
process.once('SIGTERM',()=>void browser.close());process.once('SIGINT',()=>void browser.close());
const page=await browser.newPage();page.setDefaultTimeout(45000);
page.on('pageerror',e=>evidence.runtimeErrors.push(e.message));
page.on('dialog',d=>d.accept());
page.on('console',m=>{if(['warning','error'].includes(m.type()))evidence.consoleWarnings.push({type:m.type(),text:m.text()});});
page.on('response',r=>{if(new URL(r.url()).pathname.startsWith('/api/')&&r.status()>=400)evidence.apiErrors.push({status:r.status(),url:r.url()});});
const fill=async(selector,value)=>{
  const el=await page.waitForSelector(selector);
  await el.evaluate((el,v)=>{Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(el,v);el.dispatchEvent(new Event('input',{bubbles:true}));},String(value));
  await el.press('Tab');
};
const click=async(label)=>{
  await page.waitForFunction(t=>[...document.querySelectorAll('button')].some(b=>b.textContent.trim()===t&&!b.disabled),{},label);
  for(const h of await page.$$('button'))if(await h.evaluate((e,t)=>e.textContent.trim()===t&&!e.disabled,label))return h.click();
};
const disabled=selector=>page.$eval(selector,e=>e.disabled);
const save=async()=>{
  const pending=page.waitForResponse(r=>r.request().method()==='PUT'&&new URL(r.url()).pathname==='/api/projects/SYNTHETIC_clipboard');
  for(const h of await page.$$('.topbar button'))if(await h.evaluate(e=>e.textContent.startsWith('Save'))){await h.click();break;}
  const response=await pending;assert.equal(response.status(),200);
  return page.evaluate(async()=>{const r=await fetch('/api/projects/SYNTHETIC_clipboard');if(!r.ok)throw Error('Read saved draft '+r.status);return r.json();});
};
const shortcut=async(redo=false)=>{
  await page.evaluate(()=>document.activeElement?.blur());
  await page.keyboard.down(process.platform==='darwin'?'Meta':'Control');
  if(redo)await page.keyboard.down('Shift');
  await page.keyboard.press('z');
  if(redo)await page.keyboard.up('Shift');
  await page.keyboard.up(process.platform==='darwin'?'Meta':'Control');
};
try{
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn' && document.querySelectorAll('.react-flow__node').length===10);
  await fill('input[aria-label="project id"]','SYNTHETIC_clipboard');
  const source=await save();evidence.source=source;evidence.stage='copy complete native graph';
  await page.click('.clipboard-tools summary');await click('Select all for copy');await click('Copy selected nodes');
  await page.waitForFunction(h=>document.querySelector('.clipboard-provenance')?.textContent.includes(h),{},source.graphHash);
  assert((await page.$eval('.clipboard-provenance',e=>e.textContent)).includes('0 boundary wires omitted'));
  await click('New model graph');await page.waitForFunction(()=>document.querySelectorAll('.react-flow__node').length===0);
  await fill('input[aria-label="project id"]','SYNTHETIC_clipboard');
  if(!(await page.$eval('.clipboard-tools',e=>e.open)))await page.click('.clipboard-tools summary');
  const empty=await save();await click('Paste copied nodes');
  await page.waitForFunction(n=>document.querySelectorAll('.react-flow__node').length===n,{},source.graph.nodes.length);
  const pasted=await save();evidence.pasted=pasted;
  assert.equal(pasted.graph.nodes.length,source.graph.nodes.length);assert.equal(pasted.graph.edges.length,source.graph.edges.length);
  const mapping=Object.fromEntries(source.graph.nodes.map((n,i)=>[n.id,pasted.graph.nodes[i].id]));
  for(const [i,n] of source.graph.nodes.entries()){
    const copy=pasted.graph.nodes[i];assert.equal(copy.type,n.type);assert.deepEqual(copy.config,n.config);assert.notEqual(copy.id,n.id);
    if(n.stateRef)assert.equal(copy.stateRef,`model/${copy.id}`);
    const point=source.ui.positions[n.id];if(point)assert.deepEqual(pasted.ui.positions[copy.id],{x:point.x+80,y:point.y+80});
  }
  for(const [i,e] of source.graph.edges.entries()){
    const c=pasted.graph.edges[i];assert.equal(c.kind,e.kind);assert.deepEqual(c.from,{...e.from,node:mapping[e.from.node]});assert.deepEqual(c.to,{...e.to,node:mapping[e.to.node]});
  }
  const validated=await page.evaluate(async graph=>(await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})})).json(),pasted.graph);
  assert(validated.ok);assert.equal(validated.totalParams,20042);evidence.validation=validated;
  await page.screenshot({path:path.join(output,'copied-native-model.png'),fullPage:true});
  await click('Undo');const undone=await save();assert.deepEqual(undone.graph,empty.graph);assert.deepEqual(undone.ui,empty.ui);
  await click('Redo');const redone=await save();assert.deepEqual(redone.graph,pasted.graph);assert.deepEqual(redone.ui,pasted.ui);
  await click('Paste copied nodes');await page.waitForFunction(n=>document.querySelectorAll('.react-flow__node').length===2*n,{},source.graph.nodes.length);
  const second=await save();assert.equal(new Set(second.graph.nodes.map(n=>n.id)).size,20);assert.equal(new Set(second.graph.edges.map(e=>e.id)).size,second.graph.edges.length);
  evidence.repeated=second.graph.nodes.map(n=>n.id);
  // A partial copy declares its dropped boundary and remains natively invalid until connected.
  await click('Clear copy selection');await page.click('input[aria-label="copy node conv_1_copy"]');await click('Copy selected nodes');
  await page.waitForFunction(()=>document.querySelector('.clipboard-provenance')?.textContent.includes('2 boundary wires omitted'));
  await click('New model graph');await page.waitForFunction(()=>document.querySelectorAll('.react-flow__node').length===0);
  if(!(await page.$eval('.clipboard-tools',e=>e.open)))await page.click('.clipboard-tools summary');
  await click('Paste copied nodes');await page.waitForFunction(()=>document.querySelectorAll('.react-flow__node').length===1);
  await fill('input[aria-label="project id"]','SYNTHETIC_clipboard');const partial=await save();
  const refused=await page.evaluate(async graph=>(await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})})).json(),partial.graph);
  assert(!refused.ok);assert(refused.diagnostics.some(d=>d.code==='E_MISSING_INPUT'));evidence.partial={graph:partial.graph,validation:refused};
  await click('New tabular graph');await page.waitForFunction(()=>document.querySelector('.kind')?.textContent.includes('tabular'));
  if(!(await page.$eval('.clipboard-tools',e=>e.open)))await page.click('.clipboard-tools summary');
  assert(await page.$$eval('button',buttons=>buttons.find(b=>b.textContent.trim()==='Paste copied nodes').disabled));
  await click('Clear graph clipboard');assert.equal(await page.$('.clipboard-provenance'),null);
  await page.screenshot({path:path.join(output,'graph-clipboard.png'),fullPage:true});
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings.filter(m=>m.type==='error'),[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';
  console.log('PASS native whole graph copy → independent fresh typed nodes/config/layout → one undo/redo → unique repeat → partial boundary diagnostics → incompatible-kind refusal');
}catch(error){evidence.error=String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
