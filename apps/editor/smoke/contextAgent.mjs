// Durable browser-test source. The runner copies and executes it outside the repo.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const {default: puppeteer} = await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output = process.env.VOID_SMOKE_OUTPUT;
const evidence = {status: 'failed', fixture: 'SYNTHETIC pinned retrieval + short-term native policy' , runtimeErrors: [], apiErrors: [], consoleWarnings: [], checkpoints: {}};
const browser = await puppeteer.launch({executablePath: process.env.VOID_SMOKE_CHROME, headless: true,
  args: process.platform === 'linux' ? ['--no-sandbox'] : [], defaultViewport: {width: 1600, height: 1100}});
// On runner timeout/interruption, close owned Chrome; the pending page operation then fails.
process.once('SIGTERM', () => { void browser.close(); });
process.once('SIGINT', () => { void browser.close(); });
const page = await browser.newPage();
page.setDefaultTimeout(45000);
page.on('pageerror', e => evidence.runtimeErrors.push(e.message));
page.on('dialog', d => d.accept());
page.on('console', m => {
  if (['warning', 'error'].includes(m.type())) evidence.consoleWarnings.push({type: m.type(), text: m.text()});
});
page.on('response', r => {
  if (new URL(r.url()).pathname.startsWith('/api/') && r.status() >= 400) evidence.apiErrors.push({status: r.status(), url: r.url()});
});
const click = async (text, scope = '') => {
  const selector = scope ? scope+' button' : 'button';
  await page.waitForFunction((s,t) => [...document.querySelectorAll(s)].some(b => b.textContent.trim() === t && !b.disabled), {}, selector, text);
  for (const h of await page.$$(selector)) {
    if (await h.evaluate((e,t) => e.textContent.trim() === t && !e.disabled, text)) return h.click();
  }
  throw Error('No enabled button: '+text);
};
const text = t => page.waitForFunction(t => document.body.innerText.includes(t), {}, t);
const fillElement = async (h, value) => {
  assert(h, 'Missing input');
  await h.evaluate((el,v) => {
    const prototype = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', {bubbles: true}));
  }, String(value));
  await h.press('Tab');
};
const fill = async (selector,value) => fillElement(await page.waitForSelector(selector),value);
const labelFill = async (name,value,tag='input') => {
  const label = await page.evaluateHandle(n => [...document.querySelectorAll('.production-workspace label')].find(l => l.innerText.trim().startsWith(n)),name);
  return fillElement(await label.asElement()?.$(tag),value);
};
const response = (suffix, method='POST') => page.waitForResponse(r => r.request().method() === method && new URL(r.url()).pathname.endsWith(suffix));
const capture = file => page.screenshot({path: path.join(output,file+'.png'), fullPage: true});

const get = async url => page.evaluate(async url => {
  const r=await fetch(url);if(!r.ok)throw Error(`Native GET ${url}: ${r.status}`);return r.json();
},url);

const jsonOutput=process.env.VOID_CONTEXT_JSON==='1';
const checkpoint=release=>get(`/api/production/releases/${release.id}/conversation?user=local-user&session=investigation`);
const valid=(trace,turn,question)=>{
  assert.equal(trace.status,200);assert.equal(trace.conversationState.revision,turn);
  const a=trace.result.agent;assert.equal(a.sourceRunId,evidence.sourceRun);
  assert.equal(a.finalState.turns,turn);assert.equal(a.finalState.history.length,Math.min(turn,3));
  assert.equal(a.finalState.history.at(-1).content,question);
  assert.equal(a.retrievals.length,1);assert.equal(a.retrievals[0].included.length,1);
  assert(a.events.some(e=>e.type==='index_ready'&&e.data.action==='pinned'));
  const app=a.memorySelections[0];assert(app.applicationSha256);assert.equal(app.value.stages[0].op,'retrieve');
  assert.equal(app.value.stages.at(-1).op,'budget');assert(app.value.final.length);
  assert(Object.values(app.value.records).every(r=>r.store==='short_term'));
  assert.equal(a.modelCalls,jsonOutput?1:0);
  if(jsonOutput){
    assert.deepEqual(trace.result.predictions,[{colour:'blue',count:9}]);const c=a.contexts[0].value;
    assert.equal(c.provider,'ollama');assert.equal(c.fixture,false);assert.equal(c.usage.source,'provider');
    assert(JSON.stringify(c.providerRequest).includes('SYNTHETIC retrieved record: colour blue; count 9.'));
    assert(JSON.stringify(c.providerRequest).includes(question));assert(Object.keys(c.applications).length&&Object.keys(c.retrievals).length);
  }else assert.deepEqual(trace.result.predictions,[`SYNTHETIC ${question}: ${turn}`]);
};
try{
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length===10);
  const restored=process.env.VOID_CONTEXT_RECOVERY_SEED?JSON.parse(fs.readFileSync(process.env.VOID_CONTEXT_RECOVERY_SEED,'utf8')):null;
  let version,release;
  if(restored){
    evidence.sourceRun=restored.sourceRun;await click('Production');const overview=await get('/api/production');
    version=overview.versions.find(v=>v.id===restored.version.id);release=overview.releases.find(r=>r.id===restored.release.id);
    assert.deepEqual(version,restored.version);assert.deepEqual(release,restored.release);
    assert.deepEqual(await get(`/api/agent/runs/${evidence.sourceRun}/final-state`),restored.sourceFinal);
    assert.deepEqual(await get(`/api/production/requests/${restored.trace.requestId}?user=local-user`),restored.trace);
    assert.deepEqual(await checkpoint(release),restored.checkpoint);
    assert.deepEqual(await get(`/api/production/releases/${release.id}/monitor`),restored.monitor);
    evidence.recovered={version:true,release:true,source:true,trace:true,checkpoint:true,monitor:true};
  }else{
    const project=jsonOutput?'serving_context_json':'serving_context';
    await page.waitForSelector(`select[aria-label="open project"] option[value="example:${project}"]`);
    await page.select('select[aria-label="open project"]','example:'+project);await page.waitForSelector('.aworkspace');
    await click('Run & trace');await fill('textarea[aria-label="input question"]','SYNTHETIC source: read the record.');
    let pending=response('/api/runs');await click('Run graph');const source=await(await pending).json();evidence.sourceRun=source.runId;
    await page.waitForFunction(r=>[...document.querySelectorAll('.rundetail h3')].some(h=>h.innerText.includes(r)&&h.innerText.includes('completed')),{},source.runId);
    await click('Production');const candidate=source.runId+(jsonOutput?':__agent_context_json_conversation__':':__agent_context_conversation__');
    await page.waitForFunction(k=>[...document.querySelector('select[aria-label="registration candidate"]').options].some(o=>o.value===k),{},candidate);
    await page.select('select[aria-label="registration candidate"]',candidate);
    await labelFill('Name','SYNTHETIC native context');await labelFill('Intended use','Pinned retrieval / short-term policy recovery verification','textarea');
    await labelFill('Limitations','Lexical hashing; private bounded state; no quality benchmark','textarea');
    pending=response('/api/production/versions');await click('Register version');version=await(await pending).json();
  }
  assert.equal(version.adapter,jsonOutput?'conversation_context_json':'conversation_context');evidence.version=version;
  assert.equal(version.manifest.indexes.notes.chunks,1);
  await page.waitForFunction(id=>[...document.querySelector('select[aria-label="registered version"]').options].some(o=>o.value===id),{},version.id);
  await page.select('select[aria-label="registered version"]',version.id);await click('Release','.production-workspace');
  if(!restored){
    assert.equal(await page.$eval('select[aria-label="serving session mode"]',e=>e.value),'conversation');
    assert.equal(await page.$eval('input[aria-label="serving maxBatch"]',e=>e.value),'1');
    await labelFill('Namespace',jsonOutput?'context-json-browser-smoke':'context-browser-smoke');await fill('input[aria-label="serving timeoutSeconds"]',30);
    await page.$eval('.production-workspace input[type="checkbox"]',e=>e.click());
    let pending=response('/api/production/releases');await click('Preview release candidate');release=await(await pending).json();
    await text('ready for explicit deployment');await click('Deploy selected release');await text('Local endpoint routes this exact release');
  }else{
    await page.waitForFunction(id=>[...document.querySelector('select[aria-label="production release"]').options].some(o=>o.value===id),{},release.id);
    await page.select('select[aria-label="production release"]',release.id);
  }
  evidence.release=release;evidence.sourceFinal=await get(`/api/agent/runs/${evidence.sourceRun}/final-state`);
  await click('Requests','.production-workspace');
  let pending=response(`/api/production/versions/${version.id}/reference-input`,'GET');await click('Load recorded source-turn input');assert.equal((await(await pending).json()).observedLabels,null);
  const base=restored?restored.checkpoint.head.revision:0;
  for(let step=1;step<=2;step++){
    const turn=base+step,question=`SYNTHETIC browser question ${turn}: read the record.`;
    await fill('textarea[aria-label="prediction records"]',JSON.stringify([{question}]));
    pending=response('/predict');await click('Send prediction request');const trace=await(await pending).json();valid(trace,turn,question);evidence.trace=trace;
    await page.waitForFunction(sha=>document.querySelector('.production-workspace').innerText.includes(sha),{},trace.traceSha256);
    await page.waitForSelector('[aria-label="served retrieval and memory policies"]');
    await text(trace.result.agent.memorySelections[0].applicationSha256);
    await capture(restored?'recovered-policy-'+step:'policy-'+step);
    if(jsonOutput&&step===2) assert(JSON.stringify(trace.result.agent.contexts[0].value.providerRequest).includes(`SYNTHETIC browser question ${turn-1}:`));
  }
  pending=response(`/api/production/releases/${release.id}/conversation`,'GET');await click('Inspect conversation checkpoint');evidence.checkpoint=await(await pending).json();
  assert.equal(evidence.checkpoint.state.turns,base+2);assert.equal(evidence.checkpoint.head.revision,base+2);
  const before=evidence.checkpoint;
  pending=response(`/api/production/requests/${evidence.trace.requestId}/replay`);await click('Replay captured inputs in isolation');evidence.replay=await(await pending).json();
  assert.deepEqual(evidence.replay.result.predictions,evidence.trace.result.predictions);assert.deepEqual(await checkpoint(release),before);
  const labels=jsonOutput?[{count:9,colour:'blue'}]:evidence.trace.result.predictions;
  await fill('textarea[aria-label="ground truth labels"]',JSON.stringify(labels));pending=response(`/api/production/requests/${evidence.trace.requestId}/labels`);await click('Record ground truth');assert.equal((await pending).status(),200);
  pending=response(`/api/production/releases/${release.id}/monitor`,'GET');await click('Monitoring','.production-workspace');evidence.monitor=await(await pending).json();
  assert.equal(evidence.monitor.usage.successfulTurnModelCalls,jsonOutput?base+2:0);
  assert.equal(evidence.monitor.labelBasedQuality.values[jsonOutput?'exactJsonAgreement':'exactStringAgreement'],1);
  assert.deepEqual(await get(`/api/agent/runs/${evidence.sourceRun}/final-state`),evidence.sourceFinal);
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings.filter(m=>m.type==='error'),[]);
  evidence.browserVersion=await browser.version();evidence.jsonOutput=jsonOutput;evidence.status='passed';
  console.log('PASS native pinned retrieval + short-term conversation'+(jsonOutput?' + actual Ollama JSON':' (zero model calls)')+(restored?' after physical source deletion':''));
}catch(error){evidence.error=error.stack;console.error(error);try{await capture('failure');evidence.visibleFailureText=await page.evaluate(()=>document.body.innerText);}catch{}process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2)+'\n');await browser.close();}
