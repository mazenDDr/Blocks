// Real owned Chrome, real saved drafts and API validation. No invented graph evidence.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const evidence={status:'failed',fixture:'SYNTHETIC native structured outline',runtimeErrors:[],apiErrors:[],consoleWarnings:[]};
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
  const pending=page.waitForResponse(r=>r.request().method()==='PUT'&&new URL(r.url()).pathname==='/api/projects/SYNTHETIC_outline');
  for(const h of await page.$$('.topbar button'))if(await h.evaluate(e=>e.textContent.startsWith('Save'))){await h.click();break;}
  const response=await pending;assert.equal(response.status(),200);
  return page.evaluate(async()=>{const r=await fetch('/api/projects/SYNTHETIC_outline');if(!r.ok)throw Error('Read saved draft '+r.status);return r.json();});
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
  await fill('input[aria-label="project id"]','SYNTHETIC_outline');const original=await save();evidence.original=original;
  await page.click('.graph-outline summary');
  await page.waitForFunction(h=>document.querySelector('.graph-outline .provenance')?.textContent.includes(h),{},original.graphHash);
  assert.equal(await page.$$eval('.outline-table tbody tr',rows=>rows.length),10);
  await fill('input[aria-label="outline search"]','conv_1');
  // Wire search includes endpoints in neighbouring rows, so inspect the exact actual node.
  await page.click('[aria-label="inspect outline conv_1"]');
  await page.waitForFunction(()=>document.querySelector('input[aria-label="node id"]')?.value==='conv_1');
  await page.click('[aria-label="center outline conv_1"]');
  await page.waitForFunction(()=>{
    const n=document.querySelector('.react-flow__node[data-id="conv_1"]'),c=document.querySelector('main.center');if(!n||!c)return false;
    const a=n.getBoundingClientRect(),b=c.getBoundingClientRect();return a.left>=b.left&&a.right<=b.right&&a.top>=b.top&&a.bottom<=b.bottom;
  });
  const navigated=await save();assert.deepEqual(navigated.graph,original.graph);assert.deepEqual(navigated.ui,original.ui);
  evidence.stage='actual native configuration error';await fill('input[aria-label="out_channels"]','0');
  await page.waitForFunction(()=>document.querySelector('.outline-table tbody')?.textContent.includes('E_CONFIG'));
  await page.click('input[aria-label="outline native errors only"]');
  const invalid=await save();const native=await page.evaluate(async graph=>(await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})})).json(),invalid.graph);
  assert(!native.ok);const codes=native.diagnostics.filter(d=>d.nodeId==='conv_1'&&d.severity==='error').map(d=>d.code);assert(codes.length);
  const content=await page.$eval('.outline-table tbody',e=>e.textContent);assert(codes.every(code=>content.includes(code)));evidence.nativeError={graph:invalid.graph,validation:native};
  await page.click('[aria-label="inspect outline conv_1"]');await click('Undo');
  await page.waitForFunction(()=>!document.querySelector('.outline-table tbody'));
  const restored=await save();assert.deepEqual(restored.graph,original.graph);assert.deepEqual(restored.ui,original.ui);
  await page.click('input[aria-label="outline native errors only"]');await fill('input[aria-label="outline search"]','');
  await page.waitForSelector('[aria-label="center outline conv_1"]');
  await page.screenshot({path:path.join(output,'native-outline.png'),fullPage:true,captureBeyondViewport:false});
  evidence.stage='native module scope';await page.select('select[aria-label="open project"]','example:residual_cnn');
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='residual_cnn');
  await fill('input[aria-label="project id"]','SYNTHETIC_outline');const moduleSource=await save();
  if(!(await page.$eval('.graph-outline',e=>e.open)))await page.click('.graph-outline summary');
  await page.waitForSelector('[aria-label="module outline res1"]');const moduleResponse=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/modules/validate'&&r.status()===200);await page.click('[aria-label="module outline res1"]');const moduleValidation=await (await moduleResponse).json();
  if(!(await page.$eval('.graph-outline',e=>e.open)))await page.click('.graph-outline summary');
  await page.waitForFunction(()=>document.querySelector('.graph-outline .provenance')?.textContent.includes('Module residual_block@1.0.0')&&!document.querySelector('.graph-outline .provenance')?.textContent.includes('pending'));
  assert(moduleValidation.ok);await page.waitForFunction(h=>document.querySelector('.graph-outline .provenance')?.textContent.includes(h),{},moduleValidation.moduleHash);evidence.moduleValidation=moduleValidation;await page.click('[aria-label="inspect outline conv_a"]');await page.waitForFunction(()=>document.querySelector('input[aria-label="node id"]')?.value==='conv_a');
  const moduleNavigated=await save();assert.deepEqual(moduleNavigated.graph,moduleSource.graph);assert.deepEqual(moduleNavigated.ui,moduleSource.ui);evidence.module={source:moduleSource,visible:await page.$eval('.outline-table',e=>e.textContent)};
  await page.screenshot({path:path.join(output,'module-outline.png'),fullPage:true,captureBeyondViewport:false});
  evidence.stage='bounded real native graph pages';
  const pages={schemaVersion:'1.0.0',graphKind:'model',backend:'pytorch',nodes:Array.from({length:75},(_,i)=>({id:`node_${String(i).padStart(3,'0')}`,type:'core.tensor_input',version:'1.0.0',config:{shape:['N',4],dtype:'float32'}})),edges:[]};
  const pageUi={schemaVersion:'1.0.0',positions:{},synthetic:true,description:'SYNTHETIC 75 independent tensor inputs for outline pagination; no trained model or quality claim.'};
  const pageSource=await page.evaluate(async data=>{const r=await fetch('/api/projects/SYNTHETIC_outline_pages',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!r.ok)throw Error('Seed real graph '+r.status);return r.json();},{graph:pages,ui:pageUi});
  const pageSeed=await page.evaluate(async()=>(await fetch('/api/projects/SYNTHETIC_outline_pages')).json());
  await save();await page.waitForFunction(()=>[...document.querySelector('select[aria-label="open project"]').options].some(o=>o.value==='project:SYNTHETIC_outline_pages'));
  await page.select('select[aria-label="open project"]','project:SYNTHETIC_outline_pages');await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='SYNTHETIC_outline_pages');
  if(!(await page.$eval('.graph-outline',e=>e.open)))await page.click('.graph-outline summary');
  await page.waitForFunction(h=>document.querySelector('.graph-outline .provenance')?.textContent.includes(h),{},pageSource.graphHash);
  const ids=()=>page.$$eval('.outline-table tbody tr th',cells=>cells.map(c=>c.childNodes[0].textContent));
  const first=await ids();assert.equal(first.length,50);assert.equal(first[0],'node_000');assert.equal(first.at(-1),'node_049');
  await click('Next outline nodes');const second=await ids();assert.equal(second.length,25);assert.equal(second[0],'node_050');assert.equal(second.at(-1),'node_074');
  await click('Previous outline nodes');assert.deepEqual(await ids(),first);
  await click('Next outline nodes');await fill('input[aria-label="outline search"]','node_074');assert.deepEqual(await ids(),['node_074']);
  assert(await page.$$eval('button',buttons=>buttons.find(b=>b.textContent.trim()==='Previous outline nodes').disabled));
  await page.click('[aria-label="inspect outline node_074"]');await page.waitForFunction(()=>document.querySelector('input[aria-label="node id"]')?.value==='node_074');
  const pageRead=await page.evaluate(async()=>(await fetch('/api/projects/SYNTHETIC_outline_pages')).json());assert.deepEqual(pageRead.graph,pageSeed.graph);assert.deepEqual(pageRead.ui,pageSeed.ui);evidence.pages={graphHash:pageSource.graphHash,first,second,search:await ids()};
  await page.screenshot({path:path.join(output,'outline-pages.png'),fullPage:true,captureBeyondViewport:false});
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings.filter(m=>m.type==='error'),[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';console.log('PASS native outline contracts/wires → explicit selection/center → real native error navigation → exact undo → module scope without graph/layout mutations');
}catch(error){evidence.error=String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true,captureBeyondViewport:false}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
