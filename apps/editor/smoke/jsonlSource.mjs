// Durable browser-test source. The runner copies and executes it outside the repo.
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const {default: puppeteer} = await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output = process.env.VOID_SMOKE_OUTPUT;
const evidence = {status: 'failed', fixture: 'SYNTHETIC flat JSONL native regression' , runtimeErrors: [], apiErrors: [], consoleWarnings: [], checkpoints: {}};
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

const stableMonitor=m=>{const value=structuredClone(m);delete value.window.until;return value;};
const inspect=(rid,kind)=>page.evaluate(async ({rid,kind})=>{const r=await fetch(`/api/runs/${rid}/inspect`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind,node:'housing',limit:5})});if(!r.ok)throw Error('Native inspect '+r.status);return r.json();},{rid,kind});
const completed=rid=>page.waitForFunction(async rid=>{const r=await fetch('/api/runs/'+rid);if(!r.ok)return false;const v=await r.json();if(v.status==='failed')throw Error(JSON.stringify(v.failure));return v.status==='completed';},{polling:250},rid);
try{
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length===10);
  let version,release;
  const restored=process.env.VOID_JSONL_RECOVERY_SEED?JSON.parse(fs.readFileSync(process.env.VOID_JSONL_RECOVERY_SEED,'utf8')):null;
  if(restored){
    evidence.sourceRun=restored.sourceRun;await click('Production');const overview=await get('/api/production');
    version=overview.versions.find(v=>v.id===restored.version.id);release=overview.releases.find(r=>r.id===restored.release.id);
    assert.deepEqual(version,restored.version);assert.deepEqual(release,restored.release);
    assert.deepEqual(await get('/api/runs/'+evidence.sourceRun),restored.source);
    assert.deepEqual(await inspect(evidence.sourceRun,'summary'),restored.sourceSummary);
    assert.deepEqual(await inspect(evidence.sourceRun,'table'),restored.sourceTable);
    assert.deepEqual(await get(`/api/production/requests/${restored.trace.requestId}?user=local-user`),restored.trace);
    assert.deepEqual(stableMonitor(await get(`/api/production/releases/${release.id}/monitor`)),stableMonitor(restored.monitor));
    assert.deepEqual(await get('/api/projects/SYNTHETIC_jsonl_imported'),restored.importedProject);
    evidence.source=restored.source;evidence.sourceSummary=restored.sourceSummary;evidence.sourceTable=restored.sourceTable;evidence.package=restored.package;evidence.importedProject=restored.importedProject;
    evidence.recovered={version:true,release:true,source:true,table:true,summary:true,trace:true,stableMonitor:true,inertPackage:true};
  }else{
    await page.waitForSelector('select[aria-label="open project"] option[value="example:jsonl_regression"]');
    await page.select('select[aria-label="open project"]','example:jsonl_regression');await page.waitForSelector('.tabrun');
    await text('SYNTHETIC housing teaching data in flat JSONL');
    await page.$eval('.tabrun input[type="checkbox"]',e=>{if(!e.checked)e.click();});
    let pending=response('/api/runs');await click('Run graph');evidence.sourceRun=(await(await pending).json()).runId;
    await completed(evidence.sourceRun);evidence.source=await get('/api/runs/'+evidence.sourceRun);
    assert.equal(evidence.source.progress.nodesDone,17);assert.equal(evidence.source.sources[0].rows,412);
    assert.equal(evidence.source.nodes.find(n=>n.node==='housing').cache.status,'bypass');
    evidence.sourceSummary=await inspect(evidence.sourceRun,'summary');evidence.sourceTable=await inspect(evidence.sourceRun,'table');
    const d=evidence.sourceSummary.data;assert.equal(d.format,'jsonl');assert.equal(d.rows,412);
    const raw=fs.readFileSync(d.path);assert.equal(createHash('sha256').update(raw).digest('hex'),d.sha256);
    assert.equal(evidence.sourceTable.provenance.sourceSha256,d.sha256);assert.deepEqual(evidence.sourceTable.rowIds,[0,1,2,3,4]);
    pending=response('/api/runs');await click('Run graph');const cached=(await(await pending).json()).runId;await completed(cached);evidence.cached=await get('/api/runs/'+cached);
    assert.equal(evidence.cached.nodes.filter(n=>n.cache?.status==='hit').length,16);assert.equal(evidence.cached.nodes.find(n=>n.node==='housing').cache.status,'bypass');
    await page.waitForFunction(rid=>document.querySelector('.tabrun')?.innerText.includes(rid),{},cached);
    await page.waitForSelector('.tabrun .runmark[title^="finished · cache bypass"]');
    await page.click('.tabrun .runmark[title^="finished · cache bypass"]');await page.waitForFunction(()=>document.querySelector('input[aria-label="node id"]')?.value==='housing');
    pending=response('/inspect');await click('Source');const shown=await(await pending).json();assert.equal(shown.data.sha256,d.sha256);await text(d.sha256);await capture('native-jsonl-source');
    await click('Integrations');await click('Packages');
    assert.equal(await page.$eval('.prod-panel input[type="checkbox"]',e=>e.checked),false);
    pending=response('/api/packages/export');await click('Export current project package');const inert=await(await pending).json();assert.deepEqual(inert.resources,{});
    await page.$eval('.prod-panel input[type="checkbox"]',e=>e.click());pending=response('/api/packages/export');await click('Export current project package');evidence.package=await(await pending).json();
    const keys=Object.keys(evidence.package.resources);assert.equal(keys.length,1);assert(keys[0].endsWith('.jsonl'));assert.equal(evidence.package.resources[keys[0]].sha256,d.sha256);
    const label=await page.evaluateHandle(()=>[...document.querySelectorAll('label')].find(l=>l.innerText.trim().startsWith('Import project ID')));
    await fillElement(await label.asElement().$('input'),'SYNTHETIC_jsonl_imported');
    pending=response('/api/packages/import');await click('Import and open package');const imported=await(await pending).json();assert.equal(imported.installsOrExecutesCode,false);assert.equal(imported.resources,1);
    evidence.importedProject=await get('/api/projects/SYNTHETIC_jsonl_imported');
    await click('Production');const candidate=evidence.sourceRun+':ols';
    await page.waitForFunction(k=>[...document.querySelector('select[aria-label="registration candidate"]').options].some(o=>o.value===k),{},candidate);
    await page.select('select[aria-label="registration candidate"]',candidate);
    await labelFill('Name','SYNTHETIC JSONL regression');await labelFill('Intended use','Native JSONL source and fitted-pipeline recovery verification','textarea');
    await labelFill('Limitations','Teaching fixture; training-reference labels; no real-world housing benchmark','textarea');
    pending=response('/api/production/versions');await click('Register version');version=await(await pending).json();
  }
  evidence.version=version;assert.equal(version.manifest.source.type,'tabular.jsonl_source');assert.equal(version.manifest.source.summary.sha256,evidence.sourceSummary.data.sha256);
  await page.waitForFunction(id=>[...document.querySelector('select[aria-label="registered version"]').options].some(o=>o.value===id),{},version.id);
  await page.select('select[aria-label="registered version"]',version.id);await click('Release','.production-workspace');
  if(!restored){
    await labelFill('Namespace','jsonl-browser-smoke');await page.$eval('.production-workspace input[type="checkbox"]',e=>e.click());
    let pending=response('/api/production/releases');await click('Preview release candidate');release=await(await pending).json();
    await text('ready for explicit deployment');await click('Deploy selected release');await text('Local endpoint routes this exact release');
  }else{
    await page.waitForFunction(id=>[...document.querySelector('select[aria-label="production release"]').options].some(o=>o.value===id),{},release.id);
    await page.select('select[aria-label="production release"]',release.id);
  }
  evidence.release=release;await click('Requests','.production-workspace');
  let pending=response(`/api/production/versions/${version.id}/reference-input`,'GET');await click('Load recorded training inputs');const reference=await(await pending).json();assert(reference.labelNote.includes('in-sample'));
  await fill('textarea[aria-label="prediction records"]',JSON.stringify(reference.records.slice(0,2)));
  pending=response('/predict');await click('Send prediction request');evidence.trace=await(await pending).json();assert.equal(evidence.trace.status,200);
  assert.equal(evidence.trace.result.predictions.length,2);assert(evidence.trace.result.predictions.every(Number.isFinite));
  assert.equal(evidence.trace.lineage.source.type,'tabular.jsonl_source');assert.equal(evidence.trace.lineage.source.summary.sha256,evidence.sourceSummary.data.sha256);
  await page.waitForFunction(sha=>document.querySelector('.production-workspace').innerText.includes(sha),{},evidence.trace.traceSha256);await capture(restored?'recovered-jsonl-pipeline':'jsonl-pipeline');
  pending=response(`/api/production/requests/${evidence.trace.requestId}/replay`);await click('Replay captured inputs in isolation');evidence.replay=await(await pending).json();assert.deepEqual(evidence.replay.result,evidence.trace.result);
  await fill('textarea[aria-label="ground truth labels"]',JSON.stringify(reference.observedLabels.slice(0,2)));pending=response(`/api/production/requests/${evidence.trace.requestId}/labels`);await click('Record ground truth');assert.equal((await pending).status(),200);
  pending=response(`/api/production/releases/${release.id}/monitor`,'GET');await click('Monitoring','.production-workspace');evidence.monitor=await(await pending).json();
  assert.equal(evidence.monitor.labelBasedQuality.labelledRows,restored?4:2);assert(Number.isFinite(evidence.monitor.labelBasedQuality.values.mse));
  assert.deepEqual(await get('/api/runs/'+evidence.sourceRun),evidence.source);
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings.filter(m=>m.type==='error'),[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';console.log('PASS actual JSONL source → native cache → explicit inert package → fitted regression → replay → in-sample labelled monitor'+(restored?' after physical source-workbench deletion':''));
}catch(error){evidence.error=error.stack;console.error(error);try{await capture('failure');evidence.visibleFailureText=await page.evaluate(()=>document.body.innerText);}catch{}process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2)+'\n');await browser.close();}
