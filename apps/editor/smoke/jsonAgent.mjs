// Durable browser-test source. The runner copies and executes it outside the repo.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const {default: puppeteer} = await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output = process.env.VOID_SMOKE_OUTPUT;
const evidence = {status: 'failed', fixture: 'SYNTHETIC declared colour/count text; actual non-fixture Ollama structured output', runtimeErrors: [], apiErrors: [], consoleWarnings: [], checkpoints: {}};
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
const validResult = result => {
  assert.equal(result.family,'agent_json');
  assert.deepEqual(result.predictions,[{colour:'blue',count:9}]);
  const a=result.agent;assert.equal(a.modelCalls,1);assert.equal(a.contexts.length,1);
  assert.equal(a.threadId,a.executionId);assert.equal(a.sourceRunId,evidence.sourceRun);
  const c=a.contexts[0].value;assert.equal(c.provider,'ollama');assert.equal(c.fixture,false);
  assert.equal(c.purpose,'structured_output');assert.equal(c.usage.source,'provider');assert(c.usage.outputTokens>0);
  assert(JSON.stringify(c.providerRequest).includes('colour blue, count 9'));
  assert(a.events.some(e=>e.type==='structured_attempt'&&e.data.valid===true&&e.data.callId===c.callId));
  assert.deepEqual(a.finalState.result,{colour:'blue',count:9});return a;
};
try {
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForSelector('select[aria-label="open project"]');
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length>0);
  let version,release;
  const restored=process.env.VOID_JSON_RECOVERY_SEED?JSON.parse(fs.readFileSync(process.env.VOID_JSON_RECOVERY_SEED,'utf8')):null;
  if(restored){
    evidence.sourceRun=restored.sourceRun;
    await click('Production');
    const overview=await get('/api/production');
    version=overview.versions.find(v=>v.id===restored.version.id);release=overview.releases.find(r=>r.id===restored.release.id);
    assert.deepEqual(version,restored.version);assert.deepEqual(release,restored.release);
    assert.deepEqual(await get(`/api/agent/runs/${evidence.sourceRun}/final-state`),restored.sourceFinal);
    assert.deepEqual(await get(`/api/agent/runs/${evidence.sourceRun}/model-calls`),restored.sourceCalls);
    const recorded=await get(`/api/production/requests/${restored.trace.requestId}?user=${restored.trace.user}`);
    assert.deepEqual(recorded,restored.trace);assert.equal(recorded.status,200);validResult(recorded.result);
    const mon=await get(`/api/production/releases/${release.id}/monitor`);
    assert.deepEqual(mon,restored.monitor);evidence.recovered={version:true,release:true,sourceFinal:true,sourceCalls:true,trace:true,monitor:true};
  }else{
    await page.waitForFunction(()=>[...document.querySelector('select[aria-label="open project"]').options].some(o=>o.value==='example:serving_json_agent'));
    await page.select('select[aria-label="open project"]','example:serving_json_agent');await page.waitForSelector('.aworkspace');
    await text('SYNTHETIC declared colour/count extraction teaching text');await click('Run & trace');
    await fill('textarea[aria-label="input question"]','SYNTHETIC teaching input: colour red, count 3.');
    let pending=response('/api/runs');await click('Run graph');const source=await(await pending).json();evidence.sourceRun=source.runId;
    await page.waitForFunction(r=>[...document.querySelectorAll('.rundetail h3')].some(h=>h.innerText.includes(r)&&h.innerText.includes('completed')),{},source.runId);await text('The run reached END.');
    await click('Production');const candidate=source.runId+':__agent_json_graph__';
    await page.waitForFunction(k=>[...document.querySelector('select[aria-label="registration candidate"]').options].some(o=>o.value===k),{},candidate);
    await page.select('select[aria-label="registration candidate"]',candidate);
    await labelFill('Name','SYNTHETIC native JSON extraction');await labelFill('Intended use','Schema/context/recovery verification with actual local Ollama','textarea');
    await labelFill('Limitations','Declared teaching text; no real-world model quality claim','textarea');
    pending=response('/api/production/versions');await click('Register version');version=await(await pending).json();
  }
  assert.equal(version.adapter,'agent_json');assert.equal(version.family,'agent_json');
  assert.equal(version.manifest.adapter,'native-langgraph-json-local');assert.equal(version.manifest.outputField,'result');
  assert(version.manifest.provider.digest);assert.equal(version.manifest.source.contextSha256.length,1);
  assert.deepEqual(version.manifest.outputSchema.jsonSchema.required,['colour','count']);
  evidence.version=version;
  await page.waitForFunction(id=>[...document.querySelector('select[aria-label="registered version"]').options].some(o=>o.value===id),{},version.id);
  await page.select('select[aria-label="registered version"]',version.id);
  await page.waitForFunction(()=>[...document.querySelectorAll('.production-workspace h4')].some(h=>h.textContent==='Pinned native agent: text input → LangGraph turn → schema-validated JSON'));await capture(restored?'recovered-registry':'registry');
  await click('Release','.production-workspace');
  if(!restored){
    assert.equal(await page.$eval('select[aria-label="serving session mode"]',e=>e.value),'stateless');
    assert.equal(await page.$eval('input[aria-label="serving maxBatch"]',e=>e.value),'1');
    await labelFill('Namespace','json-browser-smoke');await fill('input[aria-label="serving timeoutSeconds"]',30);
    await page.$eval('.production-workspace input[type="checkbox"]',e=>e.click());
    const pending=response('/api/production/releases');await click('Preview release candidate');release=await(await pending).json();
    assert.equal(release.config.captureInputs,true);assert.equal(release.config.sessionMode,'stateless');
    await text('ready for explicit deployment');await click('Deploy selected release');await text('Local endpoint routes this exact release');
  }else{
    await page.waitForFunction(id=>[...document.querySelector('select[aria-label="production release"]').options].some(o=>o.value===id),{},release.id);
    await page.select('select[aria-label="production release"]',release.id);
  }
  evidence.release=release;await capture(restored?'recovered-release':'release');
  evidence.sourceFinal=await get(`/api/agent/runs/${evidence.sourceRun}/final-state`);
  evidence.sourceCalls=await get(`/api/agent/runs/${evidence.sourceRun}/model-calls`);
  assert.deepEqual(evidence.sourceFinal.values.result,{colour:'red',count:3});
  assert.equal(evidence.sourceCalls.calls.length,1);assert.equal(evidence.sourceCalls.calls[0].fixture,false);
  await click('Requests','.production-workspace');
  const referencePending=response(`/api/production/versions/${version.id}/reference-input`,'GET');await click('Load recorded source-turn input');
  const reference=await(await referencePending).json();assert.equal(reference.observedLabels,null);assert.equal(reference.family,'agent_json');evidence.reference=reference;
  await fill('textarea[aria-label="prediction records"]',JSON.stringify([{question:'SYNTHETIC teaching input: colour blue, count 9.'}]));
  let pending=response('/predict');await click('Send prediction request');const traceResponse=await pending;assert.equal(traceResponse.status(),200);const trace=await traceResponse.json();assert.equal(trace.status,200);validResult(trace.result);evidence.trace=trace;
  await page.waitForFunction(id=>document.querySelector('.production-workspace').innerText.includes(id),{},trace.traceSha256);
  await capture(restored?'recovered-real-json-request':'real-json-request');
  pending=response(`/api/production/requests/${trace.requestId}/replay`);await click('Replay captured inputs in isolation');const replayResponse=await pending;assert.equal(replayResponse.status(),200);const replay=await replayResponse.json();
  assert.equal(replay.sourceTraceSha256,trace.traceSha256);assert.equal(replay.releaseId,release.id);validResult(replay.result);assert.notEqual(replay.result.agent.executionId,trace.result.agent.executionId);evidence.replay=replay;
  await fill('textarea[aria-label="ground truth labels"]','[{"count":9,"colour":"blue"}]');
  pending=response(`/api/production/requests/${trace.requestId}/labels`);await click('Record ground truth');assert.equal((await pending).status(),200);
  await text('Ground truth recorded with label delay');
  const before=await get(`/api/production/requests?release=${release.id}`);
  pending=response(`/api/production/releases/${release.id}/monitor`,'GET');await click('Monitoring','.production-workspace');
  const monitor=await(await pending).json();assert.equal(monitor.family,'agent_json');assert.equal(monitor.labelBasedQuality.values.exactJsonAgreement,1);
  assert.equal(monitor.usage.successfulTurnModelCalls,restored?restored.monitor.usage.successfulTurnModelCalls+1:1);evidence.monitor=monitor;
  await text('exactJsonAgreement');await capture(restored?'recovered-monitor':'monitor');
  assert.deepEqual(await get(`/api/production/requests?release=${release.id}`),before);
  assert.deepEqual(await get(`/api/agent/runs/${evidence.sourceRun}/final-state`),evidence.sourceFinal);
  assert.deepEqual(await get(`/api/agent/runs/${evidence.sourceRun}/model-calls`),evidence.sourceCalls);
  // Streamed request through the editor: provisional live text must equal the recorded provider response of that request.
  await click('Requests','.production-workspace');
  await fill('textarea[aria-label="prediction records"]',JSON.stringify([{question:'SYNTHETIC teaching input: colour blue, count 9.'}]));
  pending=response('/predict/stream');await click('Stream prediction request');assert.equal((await pending).status(),200);
  const streamedId=await page.waitForFunction(old=>{const m=document.querySelector('.production-workspace').innerText.match(/Current request ([0-9a-f-]{36})/);return m&&m[1]!==old&&m[1];},{},trace.requestId).then(h=>h.jsonValue());
  const streamedTrace=await page.waitForFunction(async id=>{const r=await fetch('/api/production/requests/'+id);if(!r.ok)return null;const t=await r.json();return t.status===200&&t.result?t:null;},{polling:250},streamedId).then(h=>h.jsonValue());
  validResult(streamedTrace.result);
  await page.waitForFunction(sha=>document.querySelector('.production-workspace').innerText.includes(sha),{},streamedTrace.traceSha256);
  const shown=await page.$eval('[aria-label="streamed provider text"] pre',e=>e.textContent);
  assert.equal(shown,streamedTrace.result.agent.contexts[0].value.response);assert(shown.length>0);
  evidence.streamed={requestId:streamedId,chars:shown.length,traceSha256:streamedTrace.traceSha256};
  await capture(restored?'recovered-streamed-request':'streamed-request');
  // Recovery compares the complete seeded workbench, including this streamed
  // request. Capture its actual final monitor before the offline backup.
  evidence.monitor=await get(`/api/production/releases/${release.id}/monitor`);
  assert.equal(evidence.monitor.usage.successfulTurnModelCalls,monitor.usage.successfulTurnModelCalls+1);
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings.filter(m=>m.type==='error'),[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';
  console.log('PASS actual Ollama source → pinned JSON version → native warmup → release → real schema/context request → isolated replay → independent labels → read-only monitor'+(restored?' after physical source deletion':''));
} catch(error){
  evidence.error=error.stack;console.error(error);try{await capture('failure');evidence.visibleFailureText=await page.evaluate(()=>document.body.innerText);}catch{}
  process.exitCode=1;
} finally {fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2)+'\n');await browser.close();}
