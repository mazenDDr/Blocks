// Durable browser-test source. The runner copies and executes it outside the repo.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const {default: puppeteer} = await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output = process.env.VOID_SMOKE_OUTPUT;
const evidence = {status: 'failed', fixture: 'SYNTHETIC release long-term memory, zero model calls', runtimeErrors: [], apiErrors: [], consoleWarnings: []};
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

const predict = async (note) => {
  await fill('textarea[aria-label="prediction records"]',JSON.stringify([{note}]));
  const pending=response('/predict');await click('Send prediction request');const trace=await(await pending).json();
  assert.equal(trace.status,200,JSON.stringify(trace.error));return trace;
};
const rows = () => page.$$eval('[aria-label="memory records"] tbody tr td:first-child',tds=>tds.map(td=>td.childNodes[0].textContent));
try {
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForSelector('select[aria-label="open project"] option[value="example:serving_memory"]');
  await page.select('select[aria-label="open project"]','example:serving_memory');await page.waitForSelector('.aworkspace');
  await click('Run & trace');await fill('textarea[aria-label="input note"]','SYNTHETIC source note');
  let pending=response('/api/runs');await click('Run graph');const source=await(await pending).json();evidence.sourceRun=source.runId;
  await page.waitForFunction(r=>[...document.querySelectorAll('.rundetail h3')].some(h=>h.innerText.includes(r)&&h.innerText.includes('completed')),{},source.runId);
  await click('Production');const candidate=source.runId+':__agent_memory_graph__';
  await page.waitForFunction(k=>[...document.querySelector('select[aria-label="registration candidate"]').options].some(o=>o.value===k),{},candidate);
  await page.select('select[aria-label="registration candidate"]',candidate);
  await labelFill('Name','SYNTHETIC release memory');await labelFill('Intended use','Per-user long-term memory verification','textarea');
  await labelFill('Limitations','Lexical hashing; zero model calls; no quality benchmark','textarea');
  pending=response('/api/production/versions');await click('Register version');const version=await(await pending).json();
  assert.equal(version.adapter,'agent_memory');assert.equal(version.manifest.memory.maxRecordsPerUser,200);evidence.version={id:version.id,memory:version.manifest.memory};
  await page.waitForFunction(id=>[...document.querySelector('select[aria-label="registered version"]').options].some(o=>o.value===id),{},version.id);
  await page.select('select[aria-label="registered version"]',version.id);await click('Release','.production-workspace');
  assert.equal(await page.$eval('select[aria-label="serving session mode"]',e=>e.value),'stateless');
  await labelFill('Namespace','memory-browser-smoke');await fill('input[aria-label="serving timeoutSeconds"]',30);
  pending=response('/api/production/releases');await click('Preview release candidate');const release=await(await pending).json();
  await text('ready for explicit deployment');await click('Deploy selected release');await text('Local endpoint routes this exact release');
  await click('Requests','.production-workspace');
  await labelFill('User namespace','alice');
  await page.waitForSelector('[aria-label="release memory"]');await text('No records yet.');
  const t1=await predict('SYNTHETIC alice likes teal');assert.match(t1.result.predictions[0],/recalled 0 /);
  const t2=await predict('SYNTHETIC alice owns a bicycle');assert.match(t2.result.predictions[0],/recalled 1 /);
  await page.waitForFunction(()=>document.querySelectorAll('[aria-label="memory records"] tbody tr').length===2);
  assert.deepEqual(await rows(),['SYNTHETIC alice likes teal','SYNTHETIC alice owns a bicycle']);
  await capture('alice-memory');
  await labelFill('User namespace','bob');
  await text('No records yet.');
  const b1=await predict('SYNTHETIC bob likes red');assert.match(b1.result.predictions[0],/recalled 0 /);
  await page.waitForFunction(()=>document.querySelectorAll('[aria-label="memory records"] tbody tr').length===1);
  await labelFill('User namespace','alice');
  await page.waitForFunction(()=>document.querySelectorAll('[aria-label="memory records"] tbody tr').length===2);
  const first=(await get(`/api/production/releases/${release.id}/memory?user=alice`)).records[0];
  pending=response(`/memory/${first.id}`,'DELETE');await (await page.waitForSelector(`button[aria-label="delete memory record ${first.id}"]`)).click();
  assert.equal((await pending).status(),200);
  await page.waitForFunction(()=>document.querySelectorAll('[aria-label="memory records"] tbody tr').length===1);
  const t3=await predict('SYNTHETIC alice reads poems');assert.match(t3.result.predictions[0],/recalled 1 /);
  await capture('alice-after-delete');
  evidence.memory={alice:(await get(`/api/production/releases/${release.id}/memory?user=alice`)).records.map(r=>r.text),bob:(await get(`/api/production/releases/${release.id}/memory?user=bob`)).records.map(r=>r.text)};
  assert.deepEqual(evidence.memory,{alice:['SYNTHETIC alice owns a bicycle','SYNTHETIC alice reads poems'],bob:['SYNTHETIC bob likes red']});
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);
  evidence.status='passed';
  console.log('PASS release memory: source run → register → deploy → two users isolated → delete from the panel');
} catch (error) {
  evidence.error=String(error);await capture('failure').catch(()=>{});process.exitCode=1;
} finally {
  fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();
}
