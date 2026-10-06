// Real owned Chrome, real saved drafts and native validation. Layout-only evidence; no model-quality claim.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const evidence={status:'failed',fixture:'SYNTHETIC scrambled layout of the native reference_cnn example',runtimeErrors:[],apiErrors:[],consoleWarnings:[]};
const browser=await puppeteer.launch({executablePath:process.env.VOID_SMOKE_CHROME,headless:true,
  args:process.platform==='linux'?['--no-sandbox']:[],defaultViewport:{width:1600,height:1100}});
process.once('SIGTERM',()=>void browser.close());process.once('SIGINT',()=>void browser.close());
const page=await browser.newPage();page.setDefaultTimeout(45000);
page.on('pageerror',e=>evidence.runtimeErrors.push(e.message));
page.on('dialog',d=>d.accept());
page.on('console',m=>{if(['warn','warning','error'].includes(m.type()))evidence.consoleWarnings.push({type:m.type(),text:m.text()});});
page.on('response',r=>{if(new URL(r.url()).pathname.startsWith('/api/')&&r.status()>=400)evidence.apiErrors.push({status:r.status(),url:r.url()});});
const ID='SYNTHETIC_auto_layout';
const fill=async(selector,value)=>{
  const el=await page.waitForSelector(selector);
  await el.evaluate((el,v)=>{Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(el,v);el.dispatchEvent(new Event('input',{bubbles:true}));},String(value));
  await el.press('Tab');
};
const click=async(label)=>{
  await page.waitForFunction(t=>[...document.querySelectorAll('button')].some(b=>b.textContent.trim()===t&&!b.disabled),{},label);
  for(const h of await page.$$('button'))if(await h.evaluate((e,t)=>e.textContent.trim()===t&&!e.disabled,label))return h.click();
};
const toast=text=>page.waitForFunction(t=>[...document.querySelectorAll('.toast')].some(e=>e.textContent.includes(t)),{},text);
const read=()=>page.evaluate(async id=>{const r=await fetch('/api/projects/'+id);if(!r.ok)throw Error('Read saved draft '+r.status);return r.json();},ID);
const save=async()=>{
  const pending=page.waitForResponse(r=>r.request().method()==='PUT'&&new URL(r.url()).pathname==='/api/projects/'+ID);
  for(const h of await page.$$('.topbar button'))if(await h.evaluate(e=>e.textContent.startsWith('Save'))){await h.click();break;}
  assert.equal((await pending).status(),200);return read();
};
const shortcut=async(redo=false)=>{
  await page.evaluate(()=>document.activeElement?.blur());
  await page.keyboard.down(process.platform==='darwin'?'Meta':'Control');if(redo)await page.keyboard.down('Shift');
  await page.keyboard.press('z');if(redo)await page.keyboard.up('Shift');await page.keyboard.up(process.platform==='darwin'?'Meta':'Control');
};
const sizes=()=>page.$$eval('.react-flow__node',ns=>Object.fromEntries(ns.map(n=>[n.dataset.id,{width:n.offsetWidth,height:n.offsetHeight}])));
try{
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length===10);
  await fill('input[aria-label="project id"]',ID);const initial=await save();
  // Scramble: reverse document order along a diagonal so every wire initially points backwards.
  const ids=initial.graph.nodes.map(n=>n.id);
  const ui={...initial.ui,synthetic:true,description:'SYNTHETIC scrambled layout; layout-only evidence',positions:Object.fromEntries(ids.map((id,i)=>[id,{x:(ids.length-1-i)*300+37,y:((i*7)%5)*90+11}]))};
  await page.evaluate(async({id,data})=>{const r=await fetch('/api/projects/'+id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!r.ok)throw Error('Store '+r.status);},{id:ID,data:{graph:initial.graph,ui}});
  await page.reload();await page.waitForSelector(`select[aria-label="open project"] option[value="project:${ID}"]`);
  await page.select('select[aria-label="open project"]','project:'+ID);await toast(`Loaded project '${ID}'`);
  await page.waitForFunction(n=>document.querySelectorAll('.react-flow__node').length===n,{},ids.length);
  const baseline=await read();assert.deepEqual(baseline.ui.positions,ui.positions);
  const native=await page.evaluate(async graph=>(await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})})).json(),baseline.graph);
  assert(native.ok);assert.equal(native.graphHash,baseline.graphHash);

  evidence.stage='auto-arrange: every wire points right, no card overlap, graph/hash unchanged';
  if(!(await page.$eval('.arrangement-tools',e=>e.open)))await page.click('.arrangement-tools summary');
  await click('Auto-arrange whole layout');await toast('Auto-arranged this layout left to right');
  const arranged=await save();const dims=await sizes();
  assert.deepEqual(arranged.graph,baseline.graph);assert.equal(arranged.graphHash,baseline.graphHash);
  assert.notDeepEqual(arranged.ui.positions,baseline.ui.positions);
  const p=arranged.ui.positions;
  for(const e of arranged.graph.edges)assert(p[e.from.node].x+dims[e.from.node].width<=p[e.to.node].x,`wire ${e.id} does not point right`);
  for(let i=0;i<ids.length;i++)for(let j=i+1;j<ids.length;j++){const a=ids[i],b=ids[j];
    assert(!(p[a].x<p[b].x+dims[b].width&&p[b].x<p[a].x+dims[a].width&&p[a].y<p[b].y+dims[b].height&&p[b].y<p[a].y+dims[a].height),`${a} overlaps ${b}`);}
  assert.equal(Math.min(...ids.map(id=>p[id].x)),Math.min(...ids.map(id=>baseline.ui.positions[id].x)));
  assert.equal(Math.min(...ids.map(id=>p[id].y)),Math.min(...ids.map(id=>baseline.ui.positions[id].y)));
  await new Promise(r=>setTimeout(r,600));// the view refits 150ms after arranging; screenshot only
  await page.screenshot({path:path.join(output,'auto-arranged.png'),fullPage:true,captureBeyondViewport:false});

  evidence.stage='one Undo restores the scrambled layout, Redo reapplies, repeat is a no-op';
  await shortcut();const undone=await save();assert.deepEqual(undone.ui,baseline.ui);assert.deepEqual(undone.graph,baseline.graph);
  await shortcut(true);const redone=await save();assert.deepEqual(redone.ui,arranged.ui);
  await click('Auto-arrange whole layout');await toast('Auto-arranged this layout left to right');
  assert.deepEqual((await save()).ui,arranged.ui);
  evidence.result={nodes:ids.length,edges:arranged.graph.edges.length,graphHash:arranged.graphHash,positions:arranged.ui.positions,dimensions:dims};

  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';
  console.log('PASS scrambled native layout → auto-arrange (rightward wires, no overlap, unchanged graph/hash) → Undo/Redo → idempotent repeat');
}catch(error){evidence.error=String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true,captureBeyondViewport:false}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
