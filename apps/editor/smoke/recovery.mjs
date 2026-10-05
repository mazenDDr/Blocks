// Executed only from an external copy by tools/recovery_smoke.py (ADR0027).
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const {default: puppeteer} = await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output = process.env.VOID_SMOKE_OUTPUT;
const seed = JSON.parse(fs.readFileSync(process.env.VOID_RECOVERY_SEED, 'utf8'));
assert.equal(seed.status, 'passed');
const evidence = {status: 'failed', fixture: 'SYNTHETIC native restored conversation', runtimeErrors: [], apiErrors: [], consoleWarnings: []};
const browser = await puppeteer.launch({executablePath: process.env.VOID_SMOKE_CHROME, headless: true,
  args: process.platform === 'linux' ? ['--no-sandbox'] : [], defaultViewport: {width: 1600, height: 1100}});
process.once('SIGTERM', () => { void browser.close(); });
process.once('SIGINT', () => { void browser.close(); });
const page = await browser.newPage();
page.setDefaultTimeout(45000);
page.on('pageerror', e => evidence.runtimeErrors.push(e.message));
page.on('dialog', d => d.accept());
page.on('console', m => { if (['warning', 'error'].includes(m.type())) evidence.consoleWarnings.push({type: m.type(), text: m.text()}); });
page.on('response', r => { if (new URL(r.url()).pathname.startsWith('/api/') && r.status() >= 400) evidence.apiErrors.push({status: r.status(), url: r.url()}); });
const click = async (text, scope='') => {
  const selector = scope ? scope+' button' : 'button';
  await page.waitForFunction((s,t) => [...document.querySelectorAll(s)].some(b => b.textContent.trim() === t && !b.disabled), {}, selector, text);
  for (const h of await page.$$(selector)) if (await h.evaluate((e,t) => e.textContent.trim() === t && !e.disabled, text)) return h.click();
  throw Error('No enabled button: '+text);
};
const text = t => page.waitForFunction(t => document.body.innerText.includes(t), {}, t);
const fillElement = async (h,value) => {
  assert(h);
  await h.evaluate((el,v) => {
    const prototype = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype,'value').set.call(el,v);
    el.dispatchEvent(new Event('input',{bubbles:true}));
  }, String(value));
  await h.press('Tab');
};
const fill = async (s,v) => fillElement(await page.waitForSelector(s),v);
const labelFill = async (name,value) => {
  const label = await page.evaluateHandle(n => [...document.querySelectorAll('.production-workspace label')].find(l => l.innerText.trim().startsWith(n)),name);
  return fillElement(await label.asElement()?.$('input'),value);
};
const response = (suffix,method='POST') => page.waitForResponse(r => r.request().method() === method && new URL(r.url()).pathname.endsWith(suffix));
const capture = file => page.screenshot({path:path.join(output,file+'.png'),fullPage:true});
const inspect = async session => {
  await labelFill('Session',session);
  const pending = response('/conversation','GET');
  await click('Inspect conversation checkpoint');
  return (await pending).json();
};
const turn = async (question, expected) => {
  await fill('textarea[aria-label="prediction records"]',JSON.stringify([{question}]));
  const pending = response('/predict');
  await click('Send prediction request');
  const trace = await (await pending).json();
  assert.equal(trace.status,200);
  assert.deepEqual(trace.result.predictions,[expected]);
  assert.equal(trace.result.agent.modelCalls,0);
  assert.equal(trace.versionId,seed.version);
  assert.equal(trace.releaseId,seed.release);
  await text('This native workflow made no model calls.');
  return trace;
};
try {
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForSelector('select[aria-label="open project"]');
  await page.waitForFunction(() => [...document.querySelector('select[aria-label="open project"]').options].some(o => o.value === 'example:serving_state'));
  await page.select('select[aria-label="open project"]','example:serving_state');
  await page.waitForSelector('.aworkspace');
  let pending = response('/api/production','GET');
  await click('Production');
  const registry = await (await pending).json();
  assert(registry.versions.some(v => v.id === seed.version && v.adapter === 'conversation'));
  assert(registry.releases.some(r => r.id === seed.release && r.versionId === seed.version));
  assert(registry.routes.some(r => r.release === seed.release && r.namespace === 'native-browser-smoke'));
  await page.waitForSelector('select[aria-label="registered version"]');
  await page.waitForFunction(id => [...document.querySelector('select[aria-label="registered version"]').options].some(o => o.value === id), {},seed.version);
  await page.select('select[aria-label="registered version"]',seed.version);
  await text(seed.version);
  await click('Requests','.production-workspace');
  await page.select('select[aria-label="production release"]',seed.release);
  const original = await inspect('investigation');
  const fresh = seed.traces[3];
  assert.deepEqual(original.head,{revision:4,checkpointSha256:fresh.conversationState.checkpointSha256,lastRequestId:fresh.requestId});
  assert.equal(original.state.n,1);
  evidence.original = original;
  const branch = await inspect('alternative');
  assert.equal(branch.head.checkpointSha256,seed.traces[2].conversationState.checkpointSha256);
  assert.equal(branch.state.n,3);
  evidence.branch = branch;
  await capture('restored-review');
  const nextBranch = await turn('SYNTHETIC recovered branch','SYNTHETIC recovered branch: 4/1');
  assert.equal(nextBranch.conversationState.threadId,seed.fork.threadId);
  evidence.nextBranch = nextBranch;
  assert.deepEqual(await inspect('investigation'),original);
  const next = await turn('SYNTHETIC recovered fresh','SYNTHETIC recovered fresh: 2/1');
  assert.equal(next.conversationState.threadId,fresh.conversationState.threadId);
  assert.equal(next.conversationState.revision,5);
  evidence.next = next;
  const beforeReplay = await inspect('investigation');
  const second = seed.traces[1];
  await page.waitForFunction(id => [...document.querySelectorAll('.prod-list button')].some(b => b.textContent.includes(id)), {},second.requestId.slice(0,12));
  for (const h of await page.$$('.prod-list button')) if (await h.evaluate((e,id) => e.textContent.includes(id),second.requestId.slice(0,12))) { await h.click(); break; }
  await text(second.traceSha256);
  pending = response('/replay');
  await click('Replay captured inputs in isolation');
  evidence.replay = await (await pending).json();
  assert.deepEqual(evidence.replay.result.predictions,second.result.predictions);
  assert.deepEqual(await inspect('investigation'),beforeReplay);
  await capture('restored-replay');
  pending = response('/monitor','GET');
  await click('Monitoring','.production-workspace');
  evidence.monitor = await (await pending).json();
  assert.equal(evidence.monitor.health.requests,6);
  assert.equal(evidence.monitor.health.errors,0);
  await capture('restored-monitor');
  if (process.env.VOID_RECOVERY_TRACKERS) {
    const trackerSeed = JSON.parse(fs.readFileSync(process.env.VOID_RECOVERY_TRACKERS, 'utf8'));
    pending = response('/api/integrations', 'GET');
    await click('Integrations');
    const integrations = await (await pending).json();
    for (const expected of trackerSeed.exports) {
      assert.deepEqual(integrations.exports.find(x => x.id === expected.id), expected);
    }
    await click('Trackers');
    const observed = [];
    for (const expected of trackerSeed.exports) {
      await text(expected.adapter+' · confirmed · '+expected.runId);
      const article = await page.evaluateHandle(adapter => [...document.querySelectorAll('article')].find(a => a.innerText.startsWith(adapter+' · confirmed')), expected.adapter);
      const button = await article.asElement().$('button');
      pending = response('/api/integrations/exports/'+expected.id+'/sync');
      await button.click();
      observed.push(await (await pending).json());
      assert.deepEqual(observed.at(-1), expected);
    }
    evidence.trackerExports = observed;
    await capture('restored-trackers');
  }
  assert.deepEqual(evidence.runtimeErrors,[]);
  assert.deepEqual(evidence.apiErrors,[]);
  assert.deepEqual(evidence.consoleWarnings.filter(m => m.type === 'error'),[]);
  evidence.version = seed.version;
  evidence.release = seed.release;
  evidence.browserVersion = await browser.version();
  evidence.status = 'passed';
  console.log('PASS restored route/version/checkpoints → independent branch/fresh continuations → historical replay → monitor; source workbench deleted');
} catch (error) {
  evidence.error = error.stack;
  console.error(error);
  try { await capture('failure'); evidence.visibleFailureText = await page.evaluate(() => document.body.innerText); } catch { /* bounded runner cleanup */ }
  process.exitCode = 1;
} finally {
  fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2)+'\n');
  await browser.close();
}
