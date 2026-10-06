// Real owned Chrome, real saved drafts and API validation. No invented graph evidence.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const evidence={status:'failed',fixture:'SYNTHETIC native selected-group layout movement',runtimeErrors:[],apiErrors:[],consoleWarnings:[]};
const browser=await puppeteer.launch({executablePath:process.env.VOID_SMOKE_CHROME,headless:true,
  args:process.platform==='linux'?['--no-sandbox']:[],defaultViewport:{width:1600,height:1100}});
process.once('SIGTERM',()=>void browser.close());process.once('SIGINT',()=>void browser.close());
const page=await browser.newPage();page.setDefaultTimeout(45000);
page.on('pageerror',e=>evidence.runtimeErrors.push(e.message));
page.on('dialog',d=>d.accept());
page.on('console',m=>{if(['warn','warning','error'].includes(m.type()))evidence.consoleWarnings.push({type:m.type(),text:m.text()});});
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
  const pending=page.waitForResponse(r=>r.request().method()==='PUT'&&new URL(r.url()).pathname==='/api/projects/SYNTHETIC_movement');
  for(const h of await page.$$('.topbar button'))if(await h.evaluate(e=>e.textContent.startsWith('Save'))){await h.click();break;}
  const response=await pending;assert.equal(response.status(),200);
  return page.evaluate(async()=>{const r=await fetch('/api/projects/SYNTHETIC_movement');if(!r.ok)throw Error('Read saved draft '+r.status);return r.json();});
};
const shortcut=async(redo=false)=>{
  await page.evaluate(()=>document.activeElement?.blur());
  await page.keyboard.down(process.platform==='darwin'?'Meta':'Control');
  if(redo)await page.keyboard.down('Shift');
  await page.keyboard.press('z');
  if(redo)await page.keyboard.up('Shift');
  await page.keyboard.up(process.platform==='darwin'?'Meta':'Control');
};

const choose=async ids=>{
 if(!(await page.$eval('.movement-tools',e=>e.open)))await page.click('.movement-tools summary');await click('Clear movement selection');
 for(const id of ids)await page.click(`input[aria-label="move node ${id}"]`);
};
const offset=async(dx,dy)=>{await fill('input[aria-label="group horizontal offset"]',dx);await fill('input[aria-label="group vertical offset"]',dy);};
const read=()=>page.evaluate(async()=>(await fetch('/api/projects/SYNTHETIC_movement')).json());
const loadStored=async source=>{
 await page.evaluate(async source=>{const r=await fetch('/api/projects/SYNTHETIC_movement',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(source)});if(!r.ok)throw Error('Seed actual movement layout '+r.status);},source);
 await page.select('select[aria-label="open project"]','project:SYNTHETIC_movement');await page.waitForFunction(()=>document.querySelector('.toast[role="status"]')?.textContent.includes("Loaded project 'SYNTHETIC_movement'"));await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='SYNTHETIC_movement'&&document.querySelector('button[aria-label="undo draft edit"]').disabled);
 for(const h of await page.$$('button[role="tab"]'))if(await h.evaluate(e=>e.textContent.trim()==='Graph')){await h.focus();await h.press('Enter');break;}
 await page.waitForSelector('.movement-tools');
};
const native=graph=>page.evaluate(async graph=>{const r=await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})});if(!r.ok)throw Error('Native validation '+r.status);return r.json();},graph);
// Chrome serializes CSS coordinates with limited precision; saved API coordinates are checked exactly.
const origin=ids=>page.evaluate(ids=>Object.fromEntries(ids.map(id=>{const el=[...document.querySelectorAll('.react-flow__node')].find(el=>el.dataset.id===id);if(!el)throw Error('Missing actual card '+id);const t=new DOMMatrix(el.style.transform);return[id,{x:t.e,y:t.f}];})),ids);
const verify=(before,after,ids,dx,dy,prefix='')=>{
 assert.deepEqual(after.graph,before.graph);assert.equal(after.graphHash,before.graphHash);
 assert.deepEqual({...after.ui,positions:null},{...before.ui,positions:null});
 assert.deepEqual(Object.fromEntries(Object.entries(after.ui.positions).filter(([k])=>!ids.some(id=>k===prefix+id))),Object.fromEntries(Object.entries(before.ui.positions).filter(([k])=>!ids.some(id=>k===prefix+id))));
 for(const id of ids){const key=prefix+id,p=before.ui.positions[key];assert.deepEqual(after.ui.positions[key],{x:p.x+dx,y:p.y+dy});}
 for(const a of ids)for(const b of ids){assert.equal(after.ui.positions[prefix+a].x-after.ui.positions[prefix+b].x,before.ui.positions[prefix+a].x-before.ui.positions[prefix+b].x);assert.equal(after.ui.positions[prefix+a].y-after.ui.positions[prefix+b].y,before.ui.positions[prefix+a].y-before.ui.positions[prefix+b].y);}
};
const openModule=async()=>{if(!(await page.$eval('.graph-outline',e=>e.open)))await page.click('.graph-outline summary');const button=await page.waitForSelector('[aria-label="module outline res1"]');await button.focus();await button.press('Enter');await page.waitForSelector('[aria-label="breadcrumb"]');};
try{
 await page.goto(process.env.VOID_SMOKE_URL);await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length===10);
 await fill('input[aria-label="project id"]','SYNTHETIC_movement');const original=await save();
 const ids=['conv_1','pool_1','fc'],source={graph:original.graph,ui:{...original.ui,positions:{...original.ui.positions,conv_1:{x:120,y:50},pool_1:{x:660,y:340},fc:{x:1480,y:720}},movementEvidence:{label:'SYNTHETIC user-declared layout origins; no training claim',preserved:true}}};await loadStored(source);const baseline=await save();assert.deepEqual(baseline.graph,source.graph);assert.deepEqual(baseline.ui,source.ui);evidence.baseline=baseline;const baselineNative=await native(baseline.graph);assert(baselineNative.ok);assert.equal(baselineNative.totalParams,20042);
 evidence.stage='actual same-offset origins, unchanged native report and one whole-layout Undo';await choose(ids);await offset(23.125,-80);await click('Move selected layout cards');const moved=await save();verify(baseline,moved,ids,23.125,-80);assert.deepEqual(await native(moved.graph),baselineNative);
 evidence.domBeforeCheck=await origin(ids);evidence.expectedOrigins=ids.map(id=>[id,moved.ui.positions[id]]);await page.waitForFunction(points=>points.every(([id,p])=>{const el=[...document.querySelectorAll('.react-flow__node')].find(el=>el.dataset.id===id);if(!el)return false;const exact=new DOMMatrix(el.style.transform), rendered=new DOMMatrix(getComputedStyle(el).transform);return Math.abs(exact.e-p.x)<0.01&&Math.abs(exact.f-p.y)<0.01&&Math.abs(rendered.e-p.x)<0.01&&Math.abs(rendered.f-p.y)<0.01;}),{},ids.map(id=>[id,moved.ui.positions[id]]));evidence.root={baseline,moved,native:baselineNative,domOrigins:await origin(ids)};
 await click('Undo');assert.deepEqual((await save()).ui,baseline.ui);assert(await page.$eval('button[aria-label="undo draft edit"]',b=>b.disabled));await click('Redo');assert.deepEqual((await save()).ui,moved.ui);await click('Undo');assert.deepEqual((await save()).ui,baseline.ui);
 evidence.stage='zero offset adds no position/history entry';await choose(ids);await offset(0,0);await click('Move selected layout cards');assert.deepEqual((await save()).ui,baseline.ui);assert(await page.$eval('button[aria-label="undo draft edit"]',b=>b.disabled));
 evidence.stage='empty input refuses and result bounds refuse without partial edit';await offset('',20);assert(await page.$$eval('.movement-actions button',bs=>bs.find(b=>b.textContent==='Move selected layout cards').disabled));assert.deepEqual((await save()).ui,baseline.ui);
 const bounded={graph:baseline.graph,ui:{...baseline.ui,positions:{...baseline.ui.positions,conv_1:{x:1000000,y:50}}}};await loadStored(bounded);const boundedRead=await save();await choose(['conv_1']);await offset(1,0);await click('Move selected layout cards');await page.waitForFunction(()=>document.querySelector('.toast[role="status"]')?.textContent.includes('E_MOVE_POSITION'));assert.deepEqual((await save()).ui,boundedRead.ui);assert(await page.$eval('button[aria-label="undo draft edit"]',b=>b.disabled));evidence.refusal=boundedRead;
 evidence.stage='module layout scope, unchanged interfaces/graph/identity and root keys';await page.select('select[aria-label="open project"]','example:residual_cnn');await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='residual_cnn');await fill('input[aria-label="project id"]','SYNTHETIC_movement');const moduleSource=await save();
 const moduleIds=['conv_a','relu_a','conv_b'],prefix='residual_block@1.0.0/';await loadStored({graph:moduleSource.graph,ui:{...moduleSource.ui,positions:{...moduleSource.ui.positions,...Object.fromEntries(moduleIds.map((id,i)=>[prefix+id,{x:100+i*600,y:60+i*350}]))}}});const moduleBaseline=await save();await openModule();await choose(moduleIds);await offset(-23.125,80);await click('Move selected layout cards');const moduleMoved=await save();verify(moduleBaseline,moduleMoved,moduleIds,-23.125,80,prefix);
 const moduleNative=await page.evaluate(async graph=>(await fetch('/api/modules/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph,moduleId:'residual_block',version:'1.0.0'})})).json(),moduleMoved.graph);assert(moduleNative.ok);assert.deepEqual(await native(moduleMoved.graph),await native(moduleBaseline.graph));evidence.module={baseline:moduleBaseline,moved:moduleMoved,native:moduleNative};await page.screenshot({path:path.join(output,'module-group-movement.png'),fullPage:true,captureBeyondViewport:false});await click('Undo');assert.deepEqual((await save()).ui,moduleBaseline.ui);
 evidence.stage='expanded root interiors require explicit collapse';if(!(await page.$eval('.graph-outline',e=>e.open)))await page.click('.graph-outline summary');await page.click('[aria-label="inspect outline res1"]');await click('Expand in place');await choose(['res1']);await offset(20,30);await page.waitForFunction(()=>document.querySelector('.movement-tools [role="status"]')?.textContent.includes('Collapse expanded modules'));assert(await page.$$eval('.movement-actions button',bs=>bs.find(b=>b.textContent==='Move selected layout cards').disabled));await click('Collapse modules for movement');await click('Move selected layout cards');const collapsedMoved=await save();assert.deepEqual(collapsedMoved.graph,moduleBaseline.graph);await click('Undo');assert.deepEqual((await save()).ui,moduleBaseline.ui);
 evidence.families=[];
 const fixtures=JSON.parse(process.env.VOID_MOVEMENT_FIXTURES);
 for(const example of ['tabular_regression','vision_segmentation_synthetic','nlp_token_classification','speech_ctc_tones']){
  evidence.stage='native family '+example;await page.select('select[aria-label="open project"]','example:'+example);await page.waitForFunction(id=>document.querySelector('input[aria-label="project id"]').value===id,{},example);await fill('input[aria-label="project id"]','SYNTHETIC_movement');const familySource=await save();const familyIds=familySource.graph.nodes.slice(0,3).map(n=>n.id);
  const sourceType={vision_segmentation_synthetic:'domain.vision_source',nlp_token_classification:'domain.nlp_source',speech_ctc_tones:'domain.audio_source'}[example];const graph=sourceType?{...familySource.graph,nodes:familySource.graph.nodes.map(n=>n.type===sourceType?{...n,config:{...n.config,path:fixtures[example]}}:n)}:familySource.graph;
  await loadStored({graph,ui:{...familySource.ui,positions:{...familySource.ui.positions,...Object.fromEntries(familyIds.map((id,i)=>[id,{x:100+i*700,y:60+i*350}]))}}});const familyBaseline=await save(),report=await native(familyBaseline.graph);assert(report.ok);await choose(familyIds);await offset(23.125,-80);await click('Move selected layout cards');const familyMoved=await save();verify(familyBaseline,familyMoved,familyIds,23.125,-80);const after=await native(familyMoved.graph);assert.deepEqual(after,report);evidence.families.push({example,baseline:familyBaseline,moved:familyMoved,native:after});await click('Undo');assert.deepEqual((await save()).ui,familyBaseline.ui);
 }
 evidence.stage='actual native constructor ID with absent stored layout and zero no-op';const prototype={graph:{schemaVersion:'1.0.0',graphKind:'model',backend:'pytorch',nodes:[{id:'constructor',type:'core.tensor_input',version:'1.0.0',config:{shape:['N',4],dtype:'float32'}}],edges:[]},ui:{schemaVersion:'1.0.0',synthetic:true,description:'SYNTHETIC native input; absent saved position is actual canvas fallback',positions:{}}};await loadStored(prototype);const prototypeRead=await save();await choose(['constructor']);const actualOrigin=await origin(['constructor']);await offset(0,0);await click('Move selected layout cards');assert.deepEqual((await save()).ui,prototypeRead.ui);assert(await page.$eval('button[aria-label="undo draft edit"]',b=>b.disabled));await offset(23.125,-80);await click('Move selected layout cards');const prototypeMoved=await save();assert.deepEqual(prototypeMoved.ui.positions.constructor,{x:actualOrigin.constructor.x+23.125,y:actualOrigin.constructor.y-80});assert.deepEqual(prototypeMoved.graph,prototypeRead.graph);assert.deepEqual(await native(prototypeMoved.graph),await native(prototypeRead.graph));evidence.prototype={baseline:prototypeRead,moved:prototypeMoved,native:await native(prototypeMoved.graph),actualOrigin};await click('Undo');assert.deepEqual((await save()).ui,prototypeRead.ui);
 await page.screenshot({path:path.join(output,'selected-group-movement.png'),fullPage:true,captureBeyondViewport:false});assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);evidence.browserVersion=await browser.version();evidence.status='passed';console.log('PASS selected-group movement → actual same-offset origins → exact graph/UI/native reports → single Undo/Redo/zero no-op → bounds refusal → module scope/collapse → all native families/constructor fallback');
}catch(error){evidence.error=String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true,captureBeyondViewport:false}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
