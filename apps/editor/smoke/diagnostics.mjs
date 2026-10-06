// Real owned Chrome, real saved drafts and API validation. No invented graph evidence.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const evidence={status:'failed',fixture:'SYNTHETIC native diagnostic navigation',runtimeErrors:[],apiErrors:[],consoleWarnings:[]};
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
  const pending=page.waitForResponse(r=>r.request().method()==='PUT'&&new URL(r.url()).pathname==='/api/projects/SYNTHETIC_diagnostics');
  for(const h of await page.$$('.topbar button'))if(await h.evaluate(e=>e.textContent.startsWith('Save'))){await h.click();break;}
  const response=await pending;assert.equal(response.status(),200);
  return page.evaluate(async()=>{const r=await fetch('/api/projects/SYNTHETIC_diagnostics');if(!r.ok)throw Error('Read saved draft '+r.status);return r.json();});
};
const shortcut=async(redo=false)=>{
  await page.evaluate(()=>document.activeElement?.blur());
  await page.keyboard.down(process.platform==='darwin'?'Meta':'Control');
  if(redo)await page.keyboard.down('Shift');
  await page.keyboard.press('z');
  if(redo)await page.keyboard.up('Shift');
  await page.keyboard.up(process.platform==='darwin'?'Meta':'Control');
};
const native=graph=>page.evaluate(async graph=>(await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})})).json(),graph);
const moduleNative=(graph,moduleId)=>page.evaluate(async data=>(await fetch('/api/modules/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)})).json(),{graph,moduleId,version:'1.0.0'});
const load=async data=>{
 await page.evaluate(async data=>{const r=await fetch('/api/projects/SYNTHETIC_diagnostics',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!r.ok)throw Error('Stored diagnostic fixture '+r.status);},data);
 await page.waitForSelector('select[aria-label="open project"] option[value="project:SYNTHETIC_diagnostics"]');await page.select('select[aria-label="open project"]','project:SYNTHETIC_diagnostics');await page.waitForFunction(()=>document.querySelector('.toast[role="status"]')?.textContent.includes("Loaded project 'SYNTHETIC_diagnostics'"));await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='SYNTHETIC_diagnostics'&&document.querySelector('button[aria-label="undo draft edit"]').disabled);
};
const root=async()=>{const b=await page.waitForSelector('[aria-label="breadcrumb"] button');await b.focus();await b.press('Enter');await page.waitForFunction(()=>!document.querySelector('[aria-label="breadcrumb"]'));};
const openOutline=async()=>{if(!(await page.$eval('.graph-outline',e=>e.open))){const b=await page.$('.graph-outline summary');await b.focus();await b.press('Enter');}};
const activate=async id=>{const b=await page.waitForSelector(`[aria-label="inspect diagnostic ${id} E_MISSING_INPUT"]`);await b.focus();await b.press('Enter');await page.waitForFunction(()=>document.querySelector('input[aria-label="node id"]')?.value==='constructor');};
try{
 await page.goto(process.env.VOID_SMOKE_URL);await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length===10);await fill('input[aria-label="project id"]','SYNTHETIC_diagnostics');await save();
 const graphs=JSON.parse(process.env.VOID_DIAGNOSTIC_FIXTURES);evidence.cases=[];
 for(const [kind,graph]of Object.entries(graphs)){
  evidence.stage='current actual native '+kind+' fixture';await load({graph,ui:{schemaVersion:'1.0.0',positions:{},synthetic:true,description:'SYNTHETIC intentionally unconnected constructor.b; no trained values',unknown:{preserved:kind}}});const before=await save(),report=await native(before.graph);assert(!report.ok);await openOutline();await page.waitForFunction(hash=>document.querySelector('.graph-outline .provenance')?.textContent.includes(hash),{},report.graphHash);const paths=report.diagnostics.filter(d=>d.code==='E_MISSING_INPUT').map(d=>d.nodeId);assert.equal(paths.length,2);
  const journeys=[];
  for(const id of paths){
   evidence.stage='actual native path '+id;await activate(id);const moduleId=id.includes('/otherwise/')?'alternate':'leaf';const expected=await moduleNative(before.graph,moduleId);await openOutline();await page.waitForFunction(hash=>document.querySelector('.graph-outline .provenance')?.textContent.includes(hash),{},expected.moduleHash);assert((await page.$eval('[aria-label="breadcrumb"]',e=>e.textContent)).includes(moduleId));assert((await page.$eval('[aria-label="breadcrumb"]',e=>e.textContent)).includes(kind==='select'?(id.includes('/otherwise/')?'pick/otherwise':'pick/then'):'loop/'+id.split('/').at(-2)));if(kind==='nested')assert((await page.$eval('[aria-label="breadcrumb"]',e=>e.textContent)).includes('middle'));const after=await save();assert.deepEqual(after.graph,before.graph);assert.deepEqual(after.ui,before.ui);assert.equal(after.graphHash,before.graphHash);assert(await disabled('button[aria-label="undo draft edit"]'));assert.deepEqual(await native(after.graph),report);journeys.push({id,moduleId,moduleNative:expected,saved:after,breadcrumb:await page.$eval('[aria-label="breadcrumb"]',e=>e.textContent)});await page.screenshot({path:path.join(output,kind+'-'+(id.includes('/otherwise/')?'otherwise':id.includes('/then/')?'then':id.split('/').at(-2))+'-module-inspector.png'),fullPage:true,captureBeyondViewport:false});
   // A direct module-local diagnostic selects the same stored node without adding a scope or draft edit.
   const crumb=await page.$eval('[aria-label="breadcrumb"]',e=>e.textContent);await activate('constructor');assert.equal(await page.$eval('[aria-label="breadcrumb"]',e=>e.textContent),crumb);assert.deepEqual((await save()).ui,before.ui);await root();await openOutline();await page.waitForFunction(hash=>document.querySelector('.graph-outline .provenance')?.textContent.includes(hash),{},report.graphHash);
  }
  evidence.cases.push({kind,before,native:report,journeys});await page.screenshot({path:path.join(output,kind+'-native-diagnostics.png'),fullPage:true,captureBeyondViewport:false});
 }
 evidence.stage='existing draft history survives navigation';const previous=evidence.cases.at(-1).before;await page.click('[aria-label="inspect outline input"]');const held=[];let reached;const gateSeen=new Promise(resolve=>{reached=resolve;});await page.setRequestInterception(true);const gate=request=>{const body=request.postData();if(new URL(request.url()).pathname==='/api/validate'&&request.method()==='POST'&&body&&JSON.parse(body).graph.nodes.some(n=>n.id==='input'&&n.config.shape?.[1]===5)){held.push(request);reached();}else void request.continue();};page.on('request',gate);await fill('input[aria-label="shape"]','N, 5');await gateSeen;await page.waitForFunction(()=>document.querySelector('.graph-outline .provenance')?.textContent.includes('Native validation pending'));assert.equal((await page.$$('[aria-label^="inspect diagnostic "]')).length,0);evidence.pending={actualRequestsHeld:held.length,visible:await page.$eval('.graph-outline .provenance',e=>e.textContent),diagnosticControls:0};for(const request of held)await request.continue();page.off('request',gate);await page.setRequestInterception(false);const edited=await save();assert.notEqual(edited.graphHash,previous.graphHash);await openOutline();await page.waitForFunction(()=>document.querySelector('[aria-label^="inspect diagnostic pick/then/constructor"]'));const candidate=await page.$('[aria-label^="inspect diagnostic pick/then/constructor"]');await candidate.focus();await candidate.press('Enter');await page.waitForFunction(()=>document.querySelector('input[aria-label="node id"]')?.value==='constructor');assert(await page.$eval('button[aria-label="undo draft edit"]',b=>!b.disabled));await click('Undo');assert.deepEqual((await save()).graph,previous.graph);assert.deepEqual((await save()).ui,previous.ui);assert(await disabled('button[aria-label="undo draft edit"]'));
 assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);evidence.browserVersion=await browser.version();evidence.status='passed';console.log('PASS exact actual native diagnostic paths → nested/repeat/both branches → shared definition inspector/provenance → local diagnostic/breadcrumb → no graph/UI/history edit');
}catch(error){evidence.error=String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true,captureBeyondViewport:false}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
