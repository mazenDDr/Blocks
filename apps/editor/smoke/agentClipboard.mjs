// Real owned Chrome, real saved drafts and native agent validation. No invented graph evidence.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const evidence={status:'failed',fixture:'SYNTHETIC native agent clipboard',runtimeErrors:[],apiErrors:[],consoleWarnings:[]};
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
const toast=text=>page.waitForFunction(t=>[...document.querySelectorAll('.toast')].some(e=>e.textContent.includes(t)),{},text);
const read=id=>page.evaluate(async id=>{const r=await fetch('/api/projects/'+id);if(!r.ok)throw Error('Read saved draft '+r.status);return r.json();},id);
const save=async id=>{
  const pending=page.waitForResponse(r=>r.request().method()==='PUT'&&new URL(r.url()).pathname==='/api/projects/'+id);
  for(const h of await page.$$('.topbar button'))if(await h.evaluate(e=>e.textContent.startsWith('Save'))){await h.click();break;}
  assert.equal((await pending).status(),200);return read(id);
};
const undo=async()=>{
  await page.evaluate(()=>document.activeElement?.blur());
  await page.keyboard.down(process.platform==='darwin'?'Meta':'Control');await page.keyboard.press('z');await page.keyboard.up(process.platform==='darwin'?'Meta':'Control');
};
const native=graph=>page.evaluate(async graph=>(await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})})).json(),graph);
const store=(id,data)=>page.evaluate(async({id,data})=>{const r=await fetch('/api/projects/'+id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!r.ok)throw Error('Store '+id+' '+r.status);},{id,data});
const open=async(id,reload=false)=>{
  // Reloading clears the page-memory clipboard, so only the first open reloads to list stored drafts.
  if(reload)await page.reload();await page.waitForSelector(`select[aria-label="open project"] option[value="project:${id}"]`);
  await page.select('select[aria-label="open project"]','project:'+id);await toast(`Loaded project '${id}'`);await page.waitForSelector('.agent-clipboard');
  if(!(await page.$eval('.agent-clipboard',e=>e.open)))await page.click('.agent-clipboard summary');
};
const errors=report=>report.diagnostics.filter(d=>d.severity==='error').map(d=>[d.code,d.nodeId]).sort();
const SRC='SYNTHETIC_agent_clipboard_src',DST='SYNTHETIC_agent_clipboard_dst',BAD='SYNTHETIC_agent_clipboard_conflict';
try{
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length===10);
  await page.waitForSelector('select[aria-label="open project"] option[value="example:serving_state"]');await page.select('select[aria-label="open project"]','example:serving_state');
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='serving_state');
  await fill('input[aria-label="project id"]',SRC);const initial=await save(SRC);
  const graph={...initial.graph,edges:initial.graph.edges.filter(e=>e.from.node!=='tick'),agent:{...initial.graph.agent,
    routes:[{id:'declared_route',from:'tick',cases:[{id:'positive',label:'SYNTHETIC declared route',when:{field:'n',op:'>',value:0},to:'answer'}],default:'END'}],joins:[]}};
  const ui={...initial.ui,description:'SYNTHETIC agent clipboard fixture; no execution/model-quality claim',synthetic:true};
  await store(SRC,{graph,ui});
  const empty={schemaVersion:'1.0.0',graphKind:'agent',backend:'langgraph',nodes:[],edges:[],agent:{state:[],routes:[],joins:[],limits:{maxSteps:8},indexes:[],policies:[]}};
  await store(DST,{graph:empty,ui:{schemaVersion:'1.0.0',positions:{},synthetic:true,description:'SYNTHETIC empty agent paste target'}});
  const differing=structuredClone(empty);differing.agent.state=[{name:'n',type:'text',reducer:{kind:'replace'},scope:'turn',description:''}];
  await store(BAD,{graph:differing,ui:{schemaVersion:'1.0.0',positions:{},synthetic:true,description:'SYNTHETIC conflicting agent paste target'}});
  await open(SRC,true);const baseline=await read(SRC),baseReport=await native(baseline.graph);assert(baseReport.ok,JSON.stringify(baseReport.diagnostics));

  evidence.stage='copy two nodes with their declared route using actual native state access';
  await page.click('input[aria-label="copy agent node tick"]');await page.click('input[aria-label="copy agent node answer"]');
  await click('Copy selected agent nodes');await toast(`Copied 2 agent nodes from '${SRC}'`);
  const provenance=await page.$eval('.agent-clipboard .clipboard-provenance',e=>e.textContent);
  assert(provenance.includes(baseReport.graphHash),provenance);assert(provenance.includes('2 nodes, 1 transitions, 1 routes, 0 joins, 5 state fields'),provenance);
  evidence.provenance=provenance;

  evidence.stage='same-graph paste: fresh ids, rebound route, only unreachable-entry errors, connected copy natively valid';
  await click('Paste copied agent nodes');await toast('Pasted 2 agent nodes with fresh IDs');
  const pasted=await save(SRC);const pastedIds=pasted.graph.nodes.map(n=>n.id);assert.deepEqual(pastedIds,['tick','answer','tick_copy','answer_copy']);
  assert.deepEqual(pasted.graph.nodes.slice(0,2),baseline.graph.nodes);
  const route=pasted.graph.agent.routes.find(r=>r.from==='tick_copy');assert.equal(route.cases[0].to,'answer_copy');assert.equal(route.default,'END');
  assert(pasted.graph.edges.some(e=>e.from.node==='answer_copy'&&e.to.node==='END'));
  assert.deepEqual(pasted.graph.agent.state,baseline.graph.agent.state);
  const pastedReport=await native(pasted.graph);assert.deepEqual(errors(pastedReport),[['E_UNREACHABLE_NODE','answer_copy'],['E_UNREACHABLE_NODE','tick_copy']]);
  const entered={...pasted.graph,edges:[...pasted.graph.edges,{id:'START__tick_copy',kind:'control',from:{node:'START',port:'out'},to:{node:'tick_copy',port:'in'}}]};
  const enteredReport=await native(entered);assert(enteredReport.ok,JSON.stringify(enteredReport.diagnostics));
  await page.screenshot({path:path.join(output,'agent-paste-same-graph.png'),fullPage:true,captureBeyondViewport:false});
  await undo();const restored=await save(SRC);assert.deepEqual(restored.graph,baseline.graph);
  evidence.sameGraph={pastedIds,errors:errors(pastedReport),connectedCopyNativeOk:enteredReport.ok,undoRestoresBaseline:true};

  evidence.stage='cross-project paste adds exactly the used state fields';
  await open(DST);await click('Paste copied agent nodes');await toast('Pasted 2 agent nodes with fresh IDs');
  const target=await save(DST);assert.deepEqual(target.graph.nodes.map(n=>n.id),['tick_copy','answer_copy']);
  assert.deepEqual(target.graph.agent.state.map(f=>f.name).sort(),['answer','history','n','question','turns']);
  for(const f of target.graph.agent.state)assert.deepEqual(f,baseline.graph.agent.state.find(s=>s.name===f.name));
  const targetEntered={...target.graph,edges:[...target.graph.edges,{id:'START__tick_copy',kind:'control',from:{node:'START',port:'out'},to:{node:'tick_copy',port:'in'}}]};
  const targetReport=await native(targetEntered);assert(targetReport.ok,JSON.stringify(targetReport.diagnostics));
  await page.screenshot({path:path.join(output,'agent-paste-other-project.png'),fullPage:true,captureBeyondViewport:false});
  evidence.crossProject={nodes:target.graph.nodes.map(n=>n.id),state:target.graph.agent.state.map(f=>f.name),connectedNativeOk:targetReport.ok};

  evidence.stage='differing state definition refuses without editing the draft';
  await open(BAD);const before=await read(BAD);await click('Paste copied agent nodes');await toast('E_CLIPBOARD_DEFINITION_CONFLICT');
  assert(await page.$eval('button[aria-label="undo draft edit"]',e=>e.disabled));assert.deepEqual((await save(BAD)).graph,before.graph);
  evidence.conflictRefused=true;

  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';
  console.log('PASS native agent copy → same-graph paste/route rebinding/Undo → cross-project state transfer → conflict refusal');
}catch(error){evidence.error=String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true,captureBeyondViewport:false}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
