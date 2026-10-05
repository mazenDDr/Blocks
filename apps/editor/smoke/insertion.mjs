// Real owned Chrome, real saved drafts and API validation. No invented graph evidence.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const evidence={status:'failed',fixture:'SYNTHETIC native typed wire insertion',runtimeErrors:[],apiErrors:[],consoleWarnings:[]};
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
  await el.evaluate((el,v)=>{Object.getOwnPropertyDescriptor(el.tagName==='TEXTAREA'?HTMLTextAreaElement.prototype:HTMLInputElement.prototype,'value').set.call(el,v);el.dispatchEvent(new Event('input',{bubbles:true}));},String(value));
  await el.press('Tab');
};
const click=async(label)=>{
  await page.waitForFunction(t=>[...document.querySelectorAll('button')].some(b=>b.textContent.trim()===t&&!b.disabled),{},label);
  for(const h of await page.$$('button'))if(await h.evaluate((e,t)=>e.textContent.trim()===t&&!e.disabled,label))return h.click();
};
const disabled=selector=>page.$eval(selector,e=>e.disabled);
const save=async()=>{
  const pending=page.waitForResponse(r=>r.request().method()==='PUT'&&new URL(r.url()).pathname==='/api/projects/SYNTHETIC_insertion');
  for(const h of await page.$$('.topbar button'))if(await h.evaluate(e=>e.textContent.startsWith('Save'))){await h.click();break;}
  const response=await pending;assert.equal(response.status(),200);
  return page.evaluate(async()=>{const r=await fetch('/api/projects/SYNTHETIC_insertion');if(!r.ok)throw Error('Read saved draft '+r.status);return r.json();});
};
const shortcut=async(redo=false)=>{
  await page.evaluate(()=>document.activeElement?.blur());
  await page.keyboard.down(process.platform==='darwin'?'Meta':'Control');
  if(redo)await page.keyboard.down('Shift');
  await page.keyboard.press('z');
  if(redo)await page.keyboard.up('Shift');
  await page.keyboard.up(process.platform==='darwin'?'Meta':'Control');
};

const validate=graph=>page.evaluate(async graph=>{const r=await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})});assertOK(r);return r.json();function assertOK(r){if(!r.ok)throw Error('Native validation '+r.status);}},graph);
const tools=async()=>{if(!(await page.$eval('.insertion-tools',e=>e.open)))await page.click('.insertion-tools summary');};
const select=async(edge,op,input,output)=>{await tools();await page.select('select[aria-label="insertion wire"]',edge);await page.select('select[aria-label="insertion operation"]',op);await page.select('select[aria-label="insertion input"]',input);await page.select('select[aria-label="insertion output"]',output);};
const openModule=async()=>{ evidence.stage='module native identity ready';await page.waitForFunction(()=>document.querySelector('.graph-outline .provenance')?.textContent.includes('Native validation identity'));evidence.stage='module outline details activation';if(!(await page.$eval('.graph-outline',e=>e.open))){const summary=await page.$('.graph-outline summary');await summary.focus();await summary.press('Enter');}await page.waitForFunction(()=>document.querySelector('.graph-outline').open);await page.waitForFunction(()=>document.querySelector('.graph-outline .provenance')?.textContent.includes('Native validation identity'));evidence.stage='module button activation';const moduleButton=await page.waitForSelector('[aria-label="module outline res1"]');await moduleButton.focus();await moduleButton.press('Enter');evidence.stage='module breadcrumb after Enter';await page.waitForSelector('[aria-label="breadcrumb"]'); };
const verifySplit=(before,after,edge,id,op)=>{
 const node=after.graph.nodes.find(n=>n.id===id);assert(node);assert.equal(node.type,op);
 assert.deepEqual(after.graph.nodes.filter(n=>n.id!==id),before.graph.nodes);
 assert.deepEqual(after.graph.edges.filter(e=>e.from.node!==id&&e.to.node!==id),before.graph.edges.filter(e=>e.id!==edge.id));
 const incoming=after.graph.edges.find(e=>e.to.node===id),outgoing=after.graph.edges.find(e=>e.from.node===id);
 assert.deepEqual(incoming.from,edge.from);assert.deepEqual(outgoing.to,edge.to);assert.equal(incoming.kind,edge.kind);assert.equal(outgoing.kind,edge.kind);
 assert.equal(new Set(after.graph.edges.map(e=>e.id)).size,after.graph.edges.length);
 assert.deepEqual(Object.fromEntries(Object.entries(after.ui.positions).filter(([k])=>k!==id)),before.ui.positions);
 assert.deepEqual(after.ui.nodeComments,before.ui.nodeComments);assert.notEqual(after.graphHash,before.graphHash);
};
try{
 await page.goto(process.env.VOID_SMOKE_URL);await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length===10);
 await fill('input[aria-label="project id"]','SYNTHETIC_insertion');const baseline=await save();evidence.baseline=baseline;
 const edge=baseline.graph.edges[0];
 evidence.stage='explicit operation/ports and one paired Undo';await tools();assert(await page.$$eval('button',bs=>bs.find(b=>b.textContent==='Insert selected block').disabled));
 await select(edge.id,'diag.probe','input','output');await click('Insert selected block');const inserted=await save();evidence.inserted=inserted;verifySplit(baseline,inserted,edge,'probe_1','diag.probe');
 const report=await validate(inserted.graph);assert(report.ok);assert.equal(report.totalParams,20042);assert.equal(report.graphHash,inserted.graphHash);assert.equal(report.nodes.probe_1.observationOnly,true);evidence.native=report;
 await click('Undo');assert.deepEqual((await save()).graph,baseline.graph);assert.deepEqual((await save()).ui,baseline.ui);
 await click('Redo');assert.deepEqual((await save()).graph,inserted.graph);assert.deepEqual((await save()).ui,inserted.ui);
 await click('Undo');
 evidence.stage='additional input stays missing, real native error shown';await select(edge.id,'tensor.add','a','output');await click('Insert selected block');const missing=await save();const invalid=await validate(missing.graph);assert(!invalid.ok);assert(invalid.diagnostics.some(d=>d.code==='E_MISSING_INPUT'&&d.nodeId==='add_1'&&d.port==='b'));assert(!missing.graph.edges.some(e=>e.to.node==='add_1'&&e.to.port==='b'));await page.waitForFunction(()=>document.querySelector('.node-inspector')?.textContent.includes('E_MISSING_INPUT'));evidence.missing={saved:missing,native:invalid};await click('Undo');assert.deepEqual((await save()).graph,baseline.graph);
 await page.screenshot({path:path.join(output,'typed-wire-insertion.png'),fullPage:true});
 evidence.stage='module input and internal wires preserve all other definitions/interfaces';await page.select('select[aria-label="open project"]','example:residual_cnn');await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='residual_cnn');await fill('input[aria-label="project id"]','SYNTHETIC_insertion');const moduleSource=await save();evidence.moduleSource=moduleSource;
 await openModule();
 const def=moduleSource.graph.modules.find(d=>d.id==='residual_block');evidence.modules=[];
 for(const selected of def.edges.slice(0,2)){evidence.stage='module stored edge '+selected.id;
  await select(selected.id,'diag.probe','input','output');await click('Insert selected block');const saved=await save();const changed=saved.graph.modules.find(d=>d.id===def.id);
  assert.deepEqual(saved.graph.nodes,moduleSource.graph.nodes);assert.deepEqual(saved.graph.edges,moduleSource.graph.edges);assert.deepEqual(changed.inputs,def.inputs);assert.deepEqual(changed.outputs,def.outputs);assert.deepEqual(changed.nodes.slice(0,-1),def.nodes);
  assert.deepEqual(changed.edges.filter(e=>e.from.node!=='probe_1'&&e.to.node!=='probe_1'),def.edges.filter(e=>e.id!==selected.id));
  assert.deepEqual(changed.edges.find(e=>e.to.node==='probe_1').from,selected.from);assert.deepEqual(changed.edges.find(e=>e.from.node==='probe_1').to,selected.to);
  assert.deepEqual(Object.fromEntries(Object.entries(saved.ui.positions).filter(([key])=>key!=='residual_block@1.0.0/probe_1')),moduleSource.ui.positions);
  const native=await validate(saved.graph);assert(native.ok);assert.equal(native.totalParams,(await validate(moduleSource.graph)).totalParams);
  evidence.modules.push({originalEdge:selected,saved,native});await click('Undo');assert.deepEqual((await save()).graph,moduleSource.graph);assert.deepEqual((await save()).ui,moduleSource.ui);await openModule();
 }
 await tools();await page.select('select[aria-label="insertion wire"]','__outwire_y');await page.waitForFunction(()=>document.querySelector('.insertion-tools [role="status"]')?.textContent.includes('E_INSERT_SCOPE'));assert(await page.$$eval('button',bs=>bs.find(b=>b.textContent==='Insert selected block').disabled));assert.deepEqual((await save()).graph,moduleSource.graph);await page.screenshot({path:path.join(output,'module-wire-insertion.png'),fullPage:true});
 evidence.stage='native tabular library labels and typed-kind refusal';await page.select('select[aria-label="open project"]','example:tabular_regression');await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='tabular_regression');await fill('input[aria-label="project id"]','SYNTHETIC_insertion');const table=await save();
 const fitted=table.graph.edges.find(e=>e.kind==='fit_state');await select(fitted.id,'tabular.profile','table','table');await page.waitForFunction(()=>document.querySelector('.insertion-tools [role="status"]')?.textContent.includes('E_INSERT_KIND'));assert(await page.$$eval('button',bs=>bs.find(b=>b.textContent==='Insert selected block').disabled));assert.deepEqual((await save()).graph,table.graph);
 const tableEdge=table.graph.edges.find(e=>e.kind==='table');await select(tableEdge.id,'tabular.profile','table','table');await click('Insert selected block');const profiled=await save();verifySplit(table,profiled,tableEdge,'profile_1','tabular.profile');const tableNative=await validate(profiled.graph);assert(tableNative.ok);evidence.table={baseline:table,saved:profiled,native:tableNative};await click('Undo');assert.deepEqual((await save()).graph,table.graph);assert.deepEqual((await save()).ui,table.ui);
 evidence.stage='owned synthetic vision native annotation wire';await page.select('select[aria-label="open project"]','example:vision_segmentation_synthetic');await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='vision_segmentation_synthetic');
 const fixture=JSON.parse(process.env.VOID_INSERTION_FIXTURES);const source=await page.evaluate(async()=>{const r=await fetch('/api/examples/vision_segmentation_synthetic');if(!r.ok)throw Error('Example '+r.status);return r.json();});
 source.graph.nodes.find(n=>n.type==='domain.vision_source').config.path=fixture.vision;
 await page.evaluate(async data=>{const r=await fetch('/api/projects/SYNTHETIC_insertion',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph:data.graph,ui:data.ui})});if(!r.ok)throw Error('Store labelled fixture '+r.status);},source);
 await page.select('select[aria-label="open project"]','project:SYNTHETIC_insertion');await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='SYNTHETIC_insertion'&&document.querySelector('button[aria-label="undo draft edit"]').disabled);for(const h of await page.$$('button[role="tab"]'))if(await h.evaluate(e=>e.textContent.trim()==='Graph')){await h.focus();await h.press('Enter');break;}await page.waitForSelector('.insertion-tools');const vision=await save();assert((await validate(vision.graph)).ok);
 const annotation=vision.graph.edges.find(e=>e.kind==='image_batch+annotations');await select(annotation.id,'domain.vision_box_convert','data','data');await click('Insert selected block');const converted=await save();verifySplit(vision,converted,annotation,'vision_box_convert_1','domain.vision_box_convert');const visionNative=await validate(converted.graph);assert(visionNative.ok);evidence.vision={baseline:vision,saved:converted,native:visionNative};await click('Undo');assert.deepEqual((await save()).graph,vision.graph);assert.deepEqual((await save()).ui,vision.ui);
 assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);evidence.browserVersion=await browser.version();evidence.status='passed';console.log('PASS typed wire insertion → exact unrelated graph/UI → one Undo/Redo → real native missing-input/kind refusal → module input/internal → native tabular/vision');
}catch(error){evidence.error=String(error);evidence.uiState=await page.evaluate(()=>({outlineOpen:document.querySelector('.graph-outline')?.open,provenance:document.querySelector('.graph-outline .provenance')?.textContent,outlineText:document.querySelector('.graph-outline')?.textContent,breadcrumb:document.querySelector('[aria-label="breadcrumb"]')?.textContent,wireOptions:[...document.querySelectorAll('[aria-label="insertion wire"] option')].map(o=>[o.value,o.textContent])})).catch(()=>null);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
