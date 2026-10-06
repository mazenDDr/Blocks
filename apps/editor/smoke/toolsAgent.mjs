// Durable browser-test source. The runner copies and executes it outside the repo.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const {default: puppeteer} = await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output = process.env.VOID_SMOKE_OUTPUT;
const evidence = {status: 'failed', fixture: 'SYNTHETIC native pure calculator serving; zero model calls', runtimeErrors: [], apiErrors: [], consoleWarnings: [], checkpoints: {}};
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
const validResult=result=>{
  assert.equal(result.family,'agent_turn');assert.deepEqual(result.predictions,['Native calculator: 42']);
  const a=result.agent;assert.equal(a.toolCalls,1);assert.equal(a.modelCalls,0);assert.equal(a.contexts.length,0);
  assert.equal(a.tools[0].tool,'calculator');assert.deepEqual(a.tools[0].effects,[]);
  assert.equal(a.tools[0].result.value,42);assert.equal(a.finalState.calculation.result.value,42);
  assert(a.events.some(e=>e.type==='tool_call'&&e.data.status==='ok'));return a;
};
try{
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length===10);
  let version,release;
  const restored=process.env.VOID_TOOLS_RECOVERY_SEED?JSON.parse(fs.readFileSync(process.env.VOID_TOOLS_RECOVERY_SEED,'utf8')):null;
  if(restored){
    evidence.sourceRun=restored.sourceRun;await click('Production');const overview=await get('/api/production');
    version=overview.versions.find(v=>v.id===restored.version.id);release=overview.releases.find(r=>r.id===restored.release.id);
    assert.deepEqual(version,restored.version);assert.deepEqual(release,restored.release);
    assert.deepEqual(await get(`/api/agent/runs/${evidence.sourceRun}/final-state`),restored.sourceFinal);
    assert.deepEqual(await get(`/api/production/requests/${restored.trace.requestId}?user=${restored.trace.user}`),restored.trace);
    evidence.recovered={version:true,release:true,source:true,trace:true};
  }else{
    await page.waitForSelector('select[aria-label="open project"] option[value="example:serving_tools"]');
    await page.select('select[aria-label="open project"]','example:serving_tools');await page.waitForSelector('.aworkspace');
    await click('Run & trace');await fill('textarea[aria-label="input question"]','6*7');
    let pending=response('/api/runs');await click('Run graph');const source=await(await pending).json();evidence.sourceRun=source.runId;
    await page.waitForFunction(r=>[...document.querySelectorAll('.rundetail h3')].some(h=>h.innerText.includes(r)&&h.innerText.includes('completed')),{},source.runId);
    await text('The run reached END.');await click('Production');const candidate=source.runId+':__agent_tools_graph__';
    await page.waitForFunction(k=>[...document.querySelector('select[aria-label="registration candidate"]').options].some(o=>o.value===k),{},candidate);
    await page.select('select[aria-label="registration candidate"]',candidate);
    await labelFill('Name','SYNTHETIC native calculator');await labelFill('Intended use','Native tool/capture/recovery verification','textarea');
    await labelFill('Limitations','Bounded arithmetic; no file effects or model quality claim','textarea');
    pending=response('/api/production/versions');await click('Register version');version=await(await pending).json();
  }
  assert.equal(version.adapter,'agent_tools');assert(version.manifest.implementation['agent/tools.py']);evidence.version=version;
  await page.waitForFunction(id=>[...document.querySelector('select[aria-label="registered version"]').options].some(o=>o.value===id),{},version.id);
  await page.select('select[aria-label="registered version"]',version.id);await page.waitForFunction(()=>[...document.querySelectorAll('h4')].some(h=>h.textContent.includes('Pinned native agent')));await capture('registry');
  await click('Release','.production-workspace');
  if(!restored){
    assert.equal(await page.$eval('select[aria-label="serving session mode"]',e=>e.value),'stateless');
    assert.equal(await page.$eval('input[aria-label="serving maxBatch"]',e=>e.value),'1');
    await labelFill('Namespace','tools-browser-smoke');await page.$eval('.production-workspace input[type="checkbox"]',e=>e.click());
    let pending=response('/api/production/releases');await click('Preview release candidate');release=await(await pending).json();
    await text('ready for explicit deployment');await click('Deploy selected release');await text('Local endpoint routes this exact release');
  }else{
    await page.waitForFunction(id=>[...document.querySelector('select[aria-label="production release"]').options].some(o=>o.value===id),{},release.id);
    await page.select('select[aria-label="production release"]',release.id);
  }
  evidence.release=release;assert.equal(release.config.captureInputs,true);
  evidence.sourceFinal=await get(`/api/agent/runs/${evidence.sourceRun}/final-state`);
  await click('Requests','.production-workspace');let pending=response(`/api/production/versions/${version.id}/reference-input`,'GET');
  await click('Load recorded source-turn input');const reference=await(await pending).json();assert.equal(reference.observedLabels,null);
  await fill('textarea[aria-label="prediction records"]','[{"question":"6*7"}]');
  pending=response('/predict');await click('Send prediction request');const trace=await(await pending).json();assert.equal(trace.status,200);
  validResult(trace.result);evidence.trace=trace;await page.waitForFunction(sha=>document.body.innerText.includes(sha),{},trace.traceSha256);await capture('request');
  pending=response(`/api/production/requests/${trace.requestId}/replay`);await click('Replay captured inputs in isolation');const replay=await(await pending).json();
  validResult(replay.result);assert.notEqual(replay.result.agent.executionId,trace.result.agent.executionId);evidence.replay=replay;
  await fill('textarea[aria-label="ground truth labels"]','["Native calculator: 42"]');
  pending=response(`/api/production/requests/${trace.requestId}/labels`);await click('Record ground truth');assert.equal((await pending).status(),200);
  const before=await get('/api/production/requests?release='+release.id);
  pending=response(`/api/production/releases/${release.id}/monitor`,'GET');await click('Monitoring','.production-workspace');const monitor=await(await pending).json();
  assert.equal(monitor.labelBasedQuality.values.exactStringAgreement,1);assert.equal(monitor.usage.successfulTurnModelCalls,0);evidence.monitor=monitor;
  assert.deepEqual(await get('/api/production/requests?release='+release.id),before);
  assert.deepEqual(await get(`/api/agent/runs/${evidence.sourceRun}/final-state`),evidence.sourceFinal);await capture('monitor');
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings.filter(m=>m.type==='error'),[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';console.log('PASS native calculator source → registry → warmup/deploy → request → replay → independent label → read-only monitor'+(restored?' after source deletion':''));
}catch(error){evidence.error=error.stack;console.error(error);try{await capture('failure');evidence.visibleFailureText=await page.evaluate(()=>document.body.innerText);}catch{}process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2)+'\n');await browser.close();}
