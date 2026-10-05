// Durable browser-test source. The runner copies and executes it outside the repo.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const {default: puppeteer} = await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output = process.env.VOID_SMOKE_OUTPUT;
const evidence = {status: 'failed', fixture: 'SYNTHETIC actual model-free LangGraph', runtimeErrors: [], apiErrors: [], consoleWarnings: [], checkpoints: {}};
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

try {
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForSelector('select[aria-label="open project"]');
  await page.waitForFunction(() => document.querySelector('input[aria-label="project id"]').value === 'reference_cnn' && document.querySelectorAll('.react-flow__node').length > 0);
  await page.waitForFunction(() => [...document.querySelector('select[aria-label="open project"]').options].some(o => o.value === 'example:serving_state'));
  await page.select('select[aria-label="open project"]','example:serving_state');
  await page.waitForSelector('.aworkspace');
  await text('SYNTHETIC native state teaching workflow');
  await click('Run & trace');
  await fill('textarea[aria-label="input question"]','SYNTHETIC source');
  let pending = response('/api/runs');
  await click('Run graph');
  const source = await (await pending).json();
  evidence.sourceRun = source.runId;
  await page.waitForFunction(r => [...document.querySelectorAll('.rundetail h3')].some(h => h.innerText.includes(r) && h.innerText.includes('completed')), {}, source.runId);
  await text('The run reached END.');
  await click('Production');
  const candidate = source.runId+':__agent_conversation__';
  await page.waitForFunction(k => [...document.querySelector('select[aria-label="registration candidate"]').options].some(o => o.value === k), {}, candidate);
  await page.select('select[aria-label="registration candidate"]',candidate);
  await labelFill('Name','SYNTHETIC native state smoke');
  await labelFill('Intended use','Native counters, checkpoint and editor evidence','textarea');
  await labelFill('Limitations','No language model or answer accuracy claim','textarea');
  pending = response('/api/production/versions');
  await click('Register version');
  const version = await (await pending).json();
  assert.equal(version.adapter,'conversation');
  evidence.version = version.id;
  await click('Release','.production-workspace');
  assert.equal(await page.$eval('select[aria-label="serving session mode"]',e => e.value),'conversation');
  await labelFill('Namespace','native-browser-smoke');
  await page.$eval('.production-workspace input[type="checkbox"]',e => e.click());
  pending = response('/api/production/releases');
  await click('Preview release candidate');
  const release = await (await pending).json();
  evidence.release = release.id;
  assert.equal(release.config.captureInputs,true);
  await text('ready for explicit deployment');
  await click('Deploy selected release');
  await text('Local endpoint routes this exact release');
  await click('Requests','.production-workspace');
  const turn = async (question, expected) => {
    await fill('textarea[aria-label="prediction records"]',JSON.stringify([{question}]));
    const pending = response('/predict');
    await click('Send prediction request');
    const trace = await (await pending).json();
    assert.equal(trace.status,200);
    assert.deepEqual(trace.result.predictions,[expected]);
    assert.equal(trace.result.agent.modelCalls,0);
    assert.equal(trace.result.agent.contexts.length,0);
    await text('This native workflow made no model calls.');
    return trace;
  };
  const first = await turn('SYNTHETIC first','SYNTHETIC first: 1/1');
  const second = await turn('SYNTHETIC second','SYNTHETIC second: 2/1');
  assert.equal(second.conversationState.threadId,first.conversationState.threadId);
  const inspect = async session => {
    await labelFill('Session',session);
    const pending = response('/conversation','GET');
    await click('Inspect conversation checkpoint');
    return (await pending).json();
  };
  const parent = await inspect('investigation');
  assert.deepEqual(parent.state,{question:'SYNTHETIC second',n:2,history:['SYNTHETIC first','SYNTHETIC second'],turns:1,answer:'SYNTHETIC second: 2/1'});
  assert.equal(parent.head.revision,2);
  evidence.checkpoints.parent = parent;
  await fill('input[aria-label="conversation action reason"]','SYNTHETIC native browser fork');
  await fill('input[aria-label="fork destination session"]','alternative');
  assert(await page.evaluate(() => [...document.querySelectorAll('button')].find(b => b.textContent.trim() === 'Reset inspected conversation').disabled));
  await capture('review');
  pending = response('/conversation/fork');
  await click('Fork inspected checkpoint');
  const fork = await (await pending).json();
  evidence.fork = fork;
  await text('Fork created; immutable action receipt');
  assert.deepEqual((await inspect('alternative')).state,parent.state);
  const branch = await turn('SYNTHETIC branch','SYNTHETIC branch: 3/1');
  assert.equal(branch.conversationState.threadId,fork.threadId);
  assert.notEqual(fork.threadId,first.conversationState.threadId);
  assert.deepEqual(await inspect('investigation'),parent);
  await fill('input[aria-label="conversation action reason"]','SYNTHETIC native browser reset');
  await page.click('input[aria-label="confirm conversation reset"]');
  pending = response('/conversation/reset');
  await click('Reset inspected conversation');
  evidence.reset = await (await pending).json();
  await text('Conversation reset; immutable action receipt');
  const empty = await inspect('investigation');
  assert.deepEqual(empty.head,{revision:3,checkpointSha256:null,lastRequestId:null});
  assert.equal(empty.state,null);
  evidence.checkpoints.reset = empty;
  await capture('reset');
  // Select the original successful turn through the actual recorded-request list.
  const requestLabel = second.requestId.slice(0,12);
  await page.waitForFunction(id => [...document.querySelectorAll('.prod-list button')].some(b => b.textContent.includes(id)), {}, requestLabel);
  for (const h of await page.$$('.prod-list button')) if (await h.evaluate((e,id) => e.textContent.includes(id),requestLabel)) { await h.click(); break; }
  await text(second.traceSha256);
  pending = response('/replay');
  await click('Replay captured inputs in isolation');
  const replay = await (await pending).json();
  assert.deepEqual(replay.result.predictions,second.result.predictions);
  assert.deepEqual(await inspect('investigation'),empty);
  const fresh = await turn('SYNTHETIC fresh','SYNTHETIC fresh: 1/1');
  assert.equal(fresh.conversationState.revision,4);
  assert.notEqual(fresh.conversationState.threadId,first.conversationState.threadId);
  evidence.checkpoints.fresh = fresh.conversationState;
  evidence.traces = [first,second,branch,fresh];
  pending = page.waitForResponse(r => new URL(r.url()).pathname.endsWith('/monitor'));
  await click('Monitoring','.production-workspace');
  const monitor = await (await pending).json();
  assert.equal(monitor.health.requests,4);
  assert.equal(monitor.health.errors,0);
  assert.equal(monitor.usage.successfulTurnModelCalls,0);
  assert.equal(monitor.labelBasedQuality.available,false);
  evidence.monitor = monitor;
  await capture('monitor');
  // User-authored metadata is separate from the measured native run.
  await click('Records');
  pending = response('/api/research/runs','GET');
  await click('Search recorded runs');
  const catalogue = await (await pending).json();
  assert(catalogue.runs.some(r => r.runId === source.runId && r.kind === 'agent' && r.status === 'completed'));
  pending = response('/api/research/runs/'+source.runId,'GET');
  await page.click(`[aria-label="inspect research ${source.runId}"]`);
  const initialRecord = await (await pending).json();
  assert.equal(initialRecord.annotation.revision,0);
  assert.equal(initialRecord.graphHash,source.graphHash);
  await fill('input[aria-label="research annotation author"]','SYNTHETIC native smoke researcher');
  await fill('textarea[aria-label="research annotation note"]','SYNTHETIC counter run observation; actual native source retained.');
  await fill('textarea[aria-label="research annotation tags"]','teaching-recovery\nStraße');
  pending = response('/annotation','PUT');
  await click('Save reviewed annotation');
  const noted = await (await pending).json();
  assert.equal(noted.annotation.revision,1);
  await fill('textarea[aria-label="research annotation note"]','SYNTHETIC revised observation; authored metadata, no model-quality claim.');
  pending = response('/annotation','PUT');
  await click('Save reviewed annotation');
  const revised = await (await pending).json();
  assert.equal(revised.annotation.revision,2);
  pending = response('/annotation/history','GET');
  await click('Read annotation revisions');
  const revisions = await (await pending).json();
  assert.deepEqual(revisions.revisions,[noted.annotation,revised.annotation]);
  await fill('input[aria-label="research search text"]','STRASSE');
  await fill('input[aria-label="research exact tag"]','teaching-recovery');
  pending = response('/api/research/runs','GET');
  await click('Search recorded runs');
  const found = await (await pending).json();
  assert.deepEqual(found.runs.map(r => r.runId),[source.runId]);
  assert.deepEqual(found.runs[0].annotation,revised.annotation);
  evidence.research = {noted,revised,revisions,found};
  await capture('research-records');
  await click('Open original run');
  await page.waitForFunction(r => [...document.querySelectorAll('.rundetail h3')].some(h => h.innerText.includes(r) && h.innerText.includes('completed')), {},source.runId);
  await text('The run reached END.');
  assert.deepEqual(evidence.runtimeErrors,[]);
  assert.deepEqual(evidence.apiErrors,[]);
  // Preserve known initial CNN warnings, but never silently discard unexpected console errors.
  assert.deepEqual(evidence.consoleWarnings.filter(m => m.type === 'error'),[]);
  evidence.browserVersion = await browser.version();
  evidence.status = 'passed';
  console.log('PASS source worker → native registry/release → two turns → reviewed fork/reset → historical replay → fresh thread → monitoring; zero model calls');
} catch (error) {
  evidence.error = error.stack;
  console.error(error);
  try { await capture('failure'); evidence.visibleFailureText = await page.evaluate(() => document.body.innerText); } catch { /* closed by timeout */ }
  process.exitCode = 1;
} finally {
  fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2)+'\n');
  await browser.close();
}
