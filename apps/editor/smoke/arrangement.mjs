// Real owned Chrome, real saved drafts and API validation. No invented graph evidence.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const fixturePaths=JSON.parse(process.env.VOID_ARRANGEMENT_FIXTURES);
const evidence={status:'failed',fixture:'SYNTHETIC native measured graph arrangement',runtimeErrors:[],apiErrors:[],consoleWarnings:[]};
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
  const pending=page.waitForResponse(r=>r.request().method()==='PUT'&&new URL(r.url()).pathname==='/api/projects/SYNTHETIC_arrangement');
  for(const h of await page.$$('.topbar button'))if(await h.evaluate(e=>e.textContent.startsWith('Save'))){await h.click();break;}
  const response=await pending;assert.equal(response.status(),200);
  return page.evaluate(async()=>{const r=await fetch('/api/projects/SYNTHETIC_arrangement');if(!r.ok)throw Error('Read saved draft '+r.status);return r.json();});
};
const shortcut=async(redo=false)=>{
  await page.evaluate(()=>document.activeElement?.blur());
  await page.keyboard.down(process.platform==='darwin'?'Meta':'Control');
  if(redo)await page.keyboard.down('Shift');
  await page.keyboard.press('z');
  if(redo)await page.keyboard.up('Shift');
  await page.keyboard.up(process.platform==='darwin'?'Meta':'Control');
};

const select=async ids=>{
  if(!(await page.$eval('.arrangement-tools',e=>e.open)))await page.click('.arrangement-tools summary');
  await click('Clear arrangement selection');
  for(const id of ids)await page.click(`input[aria-label="arrange node ${id}"]`);
};
const boxes=async ids=>page.evaluate(ids=>Object.fromEntries(ids.map(id=>{
  const n=[...document.querySelectorAll('.react-flow__node')].find(n=>n.dataset.id===id);
  if(!n||getComputedStyle(n).visibility!=='visible')throw Error('Actual measured card unavailable '+id);
  return [id,{width:n.offsetWidth,height:n.offsetHeight}];
})),ids);
const native=graph=>page.evaluate(async graph=>{
  const r=await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})});
  assertResponse(r);return r.json();function assertResponse(r){if(!r.ok)throw Error('Native validation '+r.status);}
},graph);
const checkGeometry=(before,after,ids,sizes,op,prefix='')=>{
  const horizontal=['Align left','Align right','Center horizontally','Distribute horizontally'].includes(op), axis=horizontal?'x':'y', other=horizontal?'y':'x', size=horizontal?'width':'height';
  for(const id of ids)assert.equal(after.ui.positions[prefix+id][other],before.ui.positions[prefix+id][other]);
  const positions={...before.ui.positions};for(const id of ids)positions[prefix+id]=after.ui.positions[prefix+id];
  assert.deepEqual(after.ui,{...before.ui,positions});assert.deepEqual(after.graph,before.graph);assert.equal(after.graphHash,before.graphHash);
  const points=ids.map(id=>({id,start:after.ui.positions[prefix+id][axis],length:sizes[id][size]}));
  const original=ids.map(id=>({id,start:before.ui.positions[prefix+id][axis],length:sizes[id][size]}));
  const near=(a,b)=>assert(Math.abs(a-b)<1e-7,`${op}: ${a} != ${b}`);
  if(op.startsWith('Distribute')){
    const order=[...original].sort((a,b)=>a.start-b.start);
    near(points.find(p=>p.id===order[0].id).start,order[0].start);near(points.find(p=>p.id===order.at(-1).id).start,order.at(-1).start);
    const moved=order.map(p=>points.find(q=>q.id===p.id));const gaps=moved.slice(1).map((p,i)=>p.start-moved[i].start-moved[i].length);assert(gaps.every(g=>g>=-1e-7));for(const g of gaps)near(g,gaps[0]);
  }else if(op.startsWith('Center')){
    const center=(Math.min(...original.map(p=>p.start))+Math.max(...original.map(p=>p.start+p.length)))/2;
    for(const p of points)near(p.start+p.length/2,center);
  }else{
    const end=op==='Align right'||op==='Align bottom';const target=end?Math.max(...original.map(p=>p.start+p.length)):Math.min(...original.map(p=>p.start));
    for(const p of points)near(p.start+(end?p.length:0),target);
  }
};
try{
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length===10);
  await fill('input[aria-label="project id"]','SYNTHETIC_arrangement');let baseline=await save();
  const ids=['conv_1','pool_1','fc'];
  const source={graph:baseline.graph,ui:{...baseline.ui,positions:{...baseline.ui.positions,conv_1:{x:120,y:50},pool_1:{x:660,y:340},fc:{x:1480,y:720}},arrangementEvidence:'SYNTHETIC user-authored layout fixture; no training/quality claim'}};
  await page.evaluate(async source=>{const r=await fetch('/api/projects/SYNTHETIC_arrangement',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(source)});if(!r.ok)throw Error('Seed layout '+r.status);},source);
  const expected=await page.evaluate(async()=>{const r=await fetch('/api/projects/SYNTHETIC_arrangement');if(!r.ok)throw Error('Read native layout seed '+r.status);return r.json();});
  if(await page.$('.toast[role="status"]')){await page.click('.toast[role="status"]');await page.waitForFunction(()=>!document.querySelector('.toast[role="status"]'));}
  await page.select('select[aria-label="open project"]','project:SYNTHETIC_arrangement');
  await page.waitForFunction(()=>document.querySelector('.toast[role="status"]')?.textContent.includes("Loaded project 'SYNTHETIC_arrangement'"));
  await page.waitForFunction(()=>document.querySelector('button[aria-label="undo draft edit"]').disabled);
  baseline=await save();assert.deepEqual(baseline.graph,expected.graph);assert.deepEqual(baseline.ui,expected.ui);assert.equal(baseline.graphHash,expected.graphHash);
  const validated=await native(baseline.graph);assert(validated.ok);assert.equal(validated.totalParams,20042);assert.equal(validated.graphHash,baseline.graphHash);evidence.baseline=baseline;evidence.native=validated;evidence.actions=[];
  for(const op of ['Align left','Align right','Align top','Align bottom','Center horizontally','Center vertically','Distribute horizontally','Distribute vertically']){
    evidence.stage=op;await select(ids);await page.waitForFunction(()=>[...document.querySelectorAll('.arrangement-actions button')].every(b=>!b.disabled));const dimensions=await boxes(ids);
    await click(op);const moved=await save();checkGeometry(baseline,moved,ids,dimensions,op);
    // A repeated already-aligned action must not create a second Undo entry.
    await click(op);const repeated=await save();assert.deepEqual(repeated.ui,moved.ui);
    await click('Undo');const undone=await save();assert.deepEqual(undone.graph,baseline.graph);assert.deepEqual(undone.ui,baseline.ui);
    await click('Redo');const redone=await save();assert.deepEqual(redone.ui,moved.ui);assert.deepEqual(redone.graph,baseline.graph);
    await click('Undo');assert.deepEqual((await save()).ui,baseline.ui);
    evidence.actions.push({operation:op,dimensions,positions:moved.ui.positions,graphHash:moved.graphHash});
  }
  await page.screenshot({path:path.join(output,'native-arrangement.png'),fullPage:true});
  evidence.stage='invalid overlap refuses without an edit';
  // Actual end-to-end refusal on measured cards whose end span cannot fit their widths.
  const crowded={graph:baseline.graph,ui:{...baseline.ui,positions:{...baseline.ui.positions,conv_1:{x:100,y:50},pool_1:{x:110,y:340},fc:{x:120,y:720}}}};
  await page.evaluate(async source=>{const r=await fetch('/api/projects/SYNTHETIC_arrangement_crowded',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(source)});if(!r.ok)throw Error('Seed crowded layout '+r.status);},crowded);
  await save();await page.waitForFunction(()=>[...document.querySelector('select[aria-label="open project"]').options].some(o=>o.value==='project:SYNTHETIC_arrangement_crowded'));
  await page.select('select[aria-label="open project"]','project:SYNTHETIC_arrangement_crowded');await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='SYNTHETIC_arrangement_crowded');
  await select(ids);await click('Distribute horizontally');await page.waitForFunction(()=>document.querySelector('.toast[role="status"]')?.textContent.includes('E_LAYOUT_OVERLAP'));
  assert(await page.$eval('button[aria-label="undo draft edit"]',b=>b.disabled));evidence.refused=await page.evaluate(async()=>(await fetch('/api/projects/SYNTHETIC_arrangement_crowded')).json());
  assert.deepEqual(evidence.refused.graph,baseline.graph);assert.deepEqual(evidence.refused.ui,crowded.ui);
  evidence.stage='module-scoped layout';await page.select('select[aria-label="open project"]','example:residual_cnn');await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='residual_cnn');
  await fill('input[aria-label="project id"]','SYNTHETIC_arrangement');let moduleSource=await save();
  const moduleIds=['conv_a','relu_a','conv_b'],prefix='residual_block@1.0.0/';const moduleUi={...moduleSource.ui,positions:{...moduleSource.ui.positions,[prefix+'conv_a']:{x:100,y:60},[prefix+'relu_a']:{x:700,y:360},[prefix+'conv_b']:{x:1400,y:760}}};
  await page.evaluate(async source=>{const r=await fetch('/api/projects/SYNTHETIC_arrangement',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(source)});if(!r.ok)throw Error('Seed module layout '+r.status);},{graph:moduleSource.graph,ui:moduleUi});
  await page.select('select[aria-label="open project"]','project:SYNTHETIC_arrangement');await page.waitForFunction(()=>document.querySelector('button[aria-label="undo draft edit"]').disabled);moduleSource=await save();
  if(!(await page.$eval('.graph-outline',e=>e.open)))await page.click('.graph-outline summary');await page.waitForSelector('[aria-label="module outline res1"]');await page.click('[aria-label="module outline res1"]');
  await select(moduleIds);await click('Align right');const moduleDimensions=await boxes(moduleIds);const moduleMoved=await save();checkGeometry(moduleSource,moduleMoved,moduleIds,moduleDimensions,'Align right',prefix);
  const moduleValidation=await page.evaluate(async graph=>(await fetch('/api/modules/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph,moduleId:'residual_block',version:'1.0.0'})})).json(),moduleMoved.graph);
  assert(moduleValidation.ok);evidence.module={source:moduleSource,moved:moduleMoved,dimensions:moduleDimensions,validation:moduleValidation};
  await page.screenshot({path:path.join(output,'module-arrangement.png'),fullPage:true});await click('Undo');assert.deepEqual((await save()).ui,moduleSource.ui);
  evidence.families=[];
  for(const example of ['tabular_regression','vision_segmentation_synthetic','nlp_token_classification','speech_ctc_tones']){
    evidence.stage='native family '+example;await page.select('select[aria-label="open project"]','example:'+example);await page.waitForFunction(id=>document.querySelector('input[aria-label="project id"]').value===id,{},example);
    // Choose the actual Graph workspace by its displayed label.
    for(const h of await page.$$('button[role="tab"]'))if(await h.evaluate(e=>e.textContent.trim()==='Graph')){await h.click();break;}
    await fill('input[aria-label="project id"]','SYNTHETIC_arrangement');const familySource=await save();
    const familyIds=familySource.graph.nodes.slice(0,3).map(n=>n.id);assert.equal(familyIds.length,3);
    const familyUi={...familySource.ui,positions:{...familySource.ui.positions,...Object.fromEntries(familyIds.map((id,i)=>[id,{x:120+i*700,y:60+i*350}]))}};
    const sourceType={vision_segmentation_synthetic:'domain.vision_source',nlp_token_classification:'domain.nlp_source',speech_ctc_tones:'domain.audio_source'}[example];
    const familyGraph=sourceType?{...familySource.graph,nodes:familySource.graph.nodes.map(n=>n.type===sourceType?{...n,config:{...n.config,path:fixturePaths[example]}}:n)}:familySource.graph;
    await page.evaluate(async data=>{const r=await fetch('/api/projects/SYNTHETIC_arrangement',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!r.ok)throw Error('Seed family '+r.status);},{graph:familyGraph,ui:familyUi});
    await page.select('select[aria-label="open project"]','project:SYNTHETIC_arrangement');await page.waitForFunction(()=>document.querySelector('button[aria-label="undo draft edit"]').disabled);
    for(const h of await page.$$('button[role="tab"]'))if(await h.evaluate(e=>e.textContent.trim()==='Graph')){await h.click();break;}
    const familyBaseline=await save();const baselineValidation=await native(familyBaseline.graph);evidence.currentFamily={example,baseline:familyBaseline,validation:baselineValidation};assert(baselineValidation.ok);await select(familyIds);await click('Align bottom');const familyDimensions=await boxes(familyIds);const familyMoved=await save();
    checkGeometry(familyBaseline,familyMoved,familyIds,familyDimensions,'Align bottom');const familyValidation=await native(familyMoved.graph);assert(familyValidation.ok);assert.deepEqual(familyValidation,baselineValidation);assert.equal(familyValidation.graphHash,familyMoved.graphHash);
    await click('Undo');assert.deepEqual((await save()).ui,familyBaseline.ui);evidence.families.push({example,baseline:familyBaseline,moved:familyMoved,dimensions:familyDimensions,validation:familyValidation});
  }
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';console.log('PASS actual measured align/distribute → exact semantic identity → single Undo/Redo → scoped module layout → measured overlap refusal');
}catch(error){evidence.error=String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
