// Durable browser-test source. The runner copies and executes it outside the repo.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const {default: puppeteer} = await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output = process.env.VOID_SMOKE_OUTPUT;
const evidence = {status: 'failed', fixture: 'SYNTHETIC native regression/cache retention', runtimeErrors: [], apiErrors: [], consoleWarnings: [], checkpoints: {}};
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

const project='SYNTHETIC_cache_retention',url='/api/cache/nodes/policies/'+project;
const get=async url=>page.evaluate(async url=>{const r=await fetch(url);if(!r.ok)throw Error(`NativeGET ${url}: ${r.status}`);return r.json();},url);
try{
  await page.goto(process.env.VOID_SMOKE_URL);await page.waitForSelector('select[aria-label="open project"]');
  await page.waitForFunction(p=>[...document.querySelector('select[aria-label="open project"]').options].some(o=>o.value==='project:'+p),{},project);
  await page.select('select[aria-label="open project"]','project:'+project);
  await text(`Loaded project '${project}'.`);await page.waitForFunction(p=>document.querySelector('input[aria-label="project id"]').value===p,{},project);
  await page.waitForSelector('.cache-retention-policy');await page.click('.cache-retention-policy summary');
  const restored=process.env.VOID_CACHE_RECOVERY_SEED?JSON.parse(fs.readFileSync(process.env.VOID_CACHE_RECOVERY_SEED,'utf8')):null;
  if(restored){
    assert.deepEqual(await get('/api/projects/'+project),restored.projectBefore);
    assert.deepEqual(await Promise.all(['SYNTHETIC-cache-original','SYNTHETIC-cache-variant'].map(r=>get('/api/runs/'+r))),restored.runsBefore);
    assert.deepEqual(await get('/api/cache/nodes'),restored.cacheAfter);
    const actual=await get(url);assert.deepEqual(actual.policy,restored.disabled.policy);assert.deepEqual(actual.receipts,restored.disabled.receipts);assert.equal(actual.running,true);
    await text('stored policy revision 2');await page.waitForSelector('[aria-label="cache retention receipts"]');await capture('restored-policy-native-receipt');
    evidence.restored={project:true,runs:true,cache:true,policy:true,receipts:true,ownSchedulerRunning:true};
  }else{
  await text('no policy recorded');evidence.projectBefore=await get('/api/projects/'+project);
  evidence.runsBefore=await Promise.all(['SYNTHETIC-cache-original','SYNTHETIC-cache-variant'].map(r=>get('/api/runs/'+r)));
  const before=await get('/api/cache/nodes');assert.equal(before.projects.find(p=>p.projectId===project).entries,19);evidence.cacheBefore=before;
  await click('Edit retention policy','.cache-retention-policy');assert.equal(await page.$eval('[aria-label="automatic cache retention enabled"]',e=>e.checked),false);
  await fill('[aria-label="cache retention olderThanHours"]',0);await fill('[aria-label="cache retention intervalSeconds"]',5);
  await page.click('[aria-label="automatic cache retention enabled"]');
  let pending=response(url+'/preview');await click('Preview retention candidates','.cache-retention-policy');const preview=await(await pending).json();assert.equal(preview.removed,3);assert.equal(preview.dryRun,true);assert.equal(preview.bytesFreed,0);evidence.preview=preview;
  assert.deepEqual(await get('/api/cache/nodes'),before);await capture('reviewed-native-preview');
  pending=page.waitForResponse(r=>r.request().method()==='PUT'&&new URL(r.url()).pathname===url);
  await click('Save retention policy','.cache-retention-policy');const stored=await(await pending).json();assert.equal(stored.policy.revision,1);assert.equal(stored.policy.config.enabled,true);assert.equal(stored.policy.config.intervalSeconds,5);assert.equal(stored.running,true);evidence.enabled=stored;
  await page.waitForFunction(async url=>{const r=await fetch(url);if(!r.ok)return false;const s=await r.json();return s.receipts.some(r=>r.result?.removed===3&&r.error===null);},{polling:300},url);
  const observed=await get(url);const receipt=observed.receipts[0];assert.equal(receipt.revision,1);assert.equal(receipt.error,null);assert.equal(receipt.result.removed,3);assert(receipt.result.bytesFreed>0);assert.equal(receipt.result.entriesTruncated,false);assert.equal(receipt.result.entriesSha256.length,64);assert.deepEqual(receipt.result.entries.map(e=>e.node_id).sort(),['metrics','ols','predictions']);evidence.receipt=receipt;
  const after=await get('/api/cache/nodes');assert.equal(after.projects.find(p=>p.projectId===project).entries,16);evidence.cacheAfter=after;
  await page.waitForFunction(()=>document.querySelector('[aria-label="cache retention receipts"]')?.textContent.includes('"removed": 3'));await capture('actual-scheduled-prune');
  assert.deepEqual(await get('/api/projects/'+project),evidence.projectBefore);assert.deepEqual(await Promise.all(['SYNTHETIC-cache-original','SYNTHETIC-cache-variant'].map(r=>get('/api/runs/'+r))),evidence.runsBefore);
  await click('Edit retention policy','.cache-retention-policy');await page.click('[aria-label="automatic cache retention enabled"]');
  pending=page.waitForResponse(r=>r.request().method()==='PUT'&&new URL(r.url()).pathname===url);await click('Save retention policy','.cache-retention-policy');const disabled=await(await pending).json();assert.equal(disabled.policy.config.enabled,false);assert.equal(disabled.policy.revision,2);assert.deepEqual(disabled.receipts,observed.receipts);evidence.disabled=disabled;
  await text('stored policy revision 2');await capture('disabled-policy');
  // Actual incomplete editor IDs should not launch invalid background requests.
  await fill('input[aria-label="project id"]','');await page.waitForSelector('.cache-retention-policy');
  if(!(await page.$eval('.cache-retention-policy',e=>e.open)))await page.click('.cache-retention-policy summary');
  await text('Use a valid project ID');
  await fill('input[aria-label="project id"]',project);
  await page.waitForFunction(()=>document.querySelector('.cache-retention-policy')?.textContent.includes('stored policy revision 2'));
  assert.deepEqual(await get('/api/projects/'+project),evidence.projectBefore);evidence.invalidDraftScopeGuard=true;
  }
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings.filter(m=>m.type==='error'),[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';console.log('PASS actual native cache19 → dry-run3 → explicitly enabled scheduled prune3 → cache16 → unchanged native run/project records → disabled revision2');
}catch(error){evidence.error=error.stack;console.error(error);try{await capture('failure');evidence.visibleFailureText=await page.evaluate(()=>document.body.innerText);}catch{}process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2)+'\n');await browser.close();}
