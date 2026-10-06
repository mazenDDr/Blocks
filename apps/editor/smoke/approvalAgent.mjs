// Durable browser-test source. The runner copies and executes it outside the repo.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const {default: puppeteer} = await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output = process.env.VOID_SMOKE_OUTPUT;
const evidence = {status: 'failed', fixture: 'SYNTHETIC native approval checkpoints; zero model calls', runtimeErrors: [], apiErrors: [], consoleWarnings: [], checkpoints: {}};
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
  if (['warn', 'warning', 'error'].includes(m.type())) evidence.consoleWarnings.push({type: m.type(), text: m.text()});
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
const capture = file => page.screenshot({path: path.join(output,file+'.png'), fullPage: true, captureBeyondViewport: false});

const get = async url => page.evaluate(async url => {
  const r=await fetch(url);if(!r.ok)throw Error(`Native GET ${url}: ${r.status}`);return r.json();
},url);
const approval = (release) => get(`/api/production/releases/${release.id}/approval?user=local-user&session=investigation`);
try {
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length===10);
  let version,release;
  const restored=process.env.VOID_APPROVAL_RECOVERY_SEED?JSON.parse(fs.readFileSync(process.env.VOID_APPROVAL_RECOVERY_SEED,'utf8')):null;
  if(restored){
    evidence.sourceRun=restored.sourceRun;await click('Production');const overview=await get('/api/production');
    version=overview.versions.find(v=>v.id===restored.version.id);release=overview.releases.find(r=>r.id===restored.release.id);
    assert.deepEqual(version,restored.version);assert.deepEqual(release,restored.release);
    assert.deepEqual(await get(`/api/agent/runs/${evidence.sourceRun}/final-state`),restored.sourceFinal);
    assert.deepEqual(await get(`/api/production/requests/${restored.paused.requestId}?user=local-user`),restored.paused);
    assert.deepEqual(await approval(release),restored.pendingSnapshot);
    evidence.recovered={version:true,release:true,source:true,pausedTrace:true,pausedCheckpoint:true};
  }else{
    await page.waitForSelector('select[aria-label="open project"] option[value="example:serving_approval"]');
    await page.select('select[aria-label="open project"]','example:serving_approval');await page.waitForSelector('.aworkspace');
    await click('Run & trace');await fill('textarea[aria-label="input question"]','SYNTHETIC browser proposal');
    let pending=response('/api/runs');await click('Run graph');const source=await(await pending).json();evidence.sourceRun=source.runId;
    await page.waitForFunction(r=>[...document.querySelectorAll('.rundetail h3')].some(h=>h.innerText.includes(r)&&h.innerText.includes('paused')),{},source.runId);
    pending=response(`/api/runs/${source.runId}/resume`);await click('Approve');assert.equal((await pending).status(),200);
    await page.waitForFunction(r=>[...document.querySelectorAll('.rundetail h3')].some(h=>h.innerText.includes(r)&&h.innerText.includes('completed')),{},source.runId);
    await click('Production');const candidate=source.runId+':__agent_approval_conversation__';
    await page.waitForFunction(k=>[...document.querySelector('select[aria-label="registration candidate"]').options].some(o=>o.value===k),{},candidate);
    await page.select('select[aria-label="registration candidate"]',candidate);
    await labelFill('Name','SYNTHETIC native approval');await labelFill('Intended use','Native paused checkpoint recovery verification','textarea');
    await labelFill('Limitations','Effect-free; no model quality claim','textarea');
    pending=response('/api/production/versions');await click('Register version');version=await(await pending).json();
  }
  evidence.version=version;assert.equal(version.adapter,'conversation_approval');
  await page.waitForFunction(id=>[...document.querySelector('select[aria-label="registered version"]').options].some(o=>o.value===id),{},version.id);
  await page.select('select[aria-label="registered version"]',version.id);await click('Release','.production-workspace');
  if(!restored){
    assert.equal(await page.$eval('select[aria-label="serving session mode"]',e=>e.value),'conversation');
    assert.equal(await page.$eval('input[aria-label="serving maxBatch"]',e=>e.value),'1');
    await labelFill('Namespace','approval-browser-smoke');await page.$eval('.production-workspace input[type="checkbox"]',e=>e.click());
    let pending=response('/api/production/releases');await click('Preview release candidate');release=await(await pending).json();
    await text('ready for explicit deployment');await click('Deploy selected release');await text('Local endpoint routes this exact release');
  }else{
    await page.waitForFunction(id=>[...document.querySelector('select[aria-label="production release"]').options].some(o=>o.value===id),{},release.id);
    await page.select('select[aria-label="production release"]',release.id);
  }
  evidence.release=release;evidence.sourceFinal=await get(`/api/agent/runs/${evidence.sourceRun}/final-state`);
  await click('Requests','.production-workspace');
  if(!restored){
    let pending=response(`/api/production/versions/${version.id}/reference-input`,'GET');await click('Load recorded source-turn input');assert.equal((await(await pending).json()).observedLabels,null);
    await fill('textarea[aria-label="prediction records"]','[{"question":"SYNTHETIC browser proposal"}]');
    pending=response('/predict');await click('Send prediction request');const paused=await(await pending).json();assert.equal(paused.status,202);
    assert.deepEqual(paused.result.predictions,[]);assert.equal(paused.result.agent.modelCalls,0);assert.equal(paused.conversationState.revision,1);evidence.paused=paused;
  }else{evidence.paused=restored.paused;}
  let pending=response(`/api/production/releases/${release.id}/approval`,'GET');await click('Inspect pending approval');const snap=await(await pending).json();
  assert(snap.pending);assert.equal(snap.state.n,restored?4:1);assert.equal(snap.pending.value.proposed,'SYNTHETIC browser proposal');evidence.pendingSnapshot=snap;
  await text('pending review');await capture(restored?'recovered-paused':'paused');
  await fill('textarea[aria-label="approval edit value"]','reviewed browser text');pending=response('/resume');await click('Edit and resume');const resumed=await(await pending).json();
  assert.equal(resumed.status,200);assert.deepEqual(resumed.result.predictions,[`edit: reviewed browser text / ${restored?4:1}`]);assert.equal(resumed.conversationState.revision,restored?8:2);assert.equal(resumed.result.agent.modelCalls,0);
  evidence.resumed=resumed;await text('no pending review');await capture('resumed');
  // Approve/reject are actual native decisions on subsequent turns, retaining thread state.
  for(const [index,action] of ['approve','reject'].entries()){
    await fill('textarea[aria-label="prediction records"]','[{"question":"SYNTHETIC browser proposal"}]');
    pending=response('/predict');await click('Send prediction request');assert.equal((await(await pending).json()).status,202);
    pending=response(`/api/production/releases/${release.id}/approval`,'GET');await click('Inspect pending approval');const review=await(await pending).json();assert.equal(review.state.n,index+(restored?5:2));
    pending=response('/resume');await click(action==='approve'?'Approve and resume':'Reject and resume');const trace=await(await pending).json();
    assert.equal(trace.status,200);assert.deepEqual(trace.result.predictions,[`${action}: SYNTHETIC browser proposal / ${index+(restored?5:2)}`]);
    evidence[action]=trace;
  }
  // Leave a genuine paused native checkpoint in the seeded workbench for backup recovery.
  pending=response('/predict');await click('Send prediction request');const paused=await(await pending).json();assert.equal(paused.status,202);
  evidence.paused=paused;evidence.pendingSnapshot=await approval(release);assert.equal(evidence.pendingSnapshot.state.n,restored?7:4);
  assert.deepEqual(await get(`/api/agent/runs/${evidence.sourceRun}/final-state`),evidence.sourceFinal);
  const monitor=await get(`/api/production/releases/${release.id}/monitor`);assert.equal(monitor.health.errors,0);assert.equal(monitor.usage.successfulTurnModelCalls,0);evidence.monitor=monitor;
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings.filter(m=>m.type==='error'),[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';console.log('PASS native pending checkpoint → edit/approve/reject → retained native thread'+(restored?' after physical source deletion':''));
}catch(error){evidence.error=error.stack;console.error(error);try{await capture('failure');evidence.visibleFailureText=await page.evaluate(()=>document.body.innerText);}catch{}process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2)+'\n');await browser.close();}
