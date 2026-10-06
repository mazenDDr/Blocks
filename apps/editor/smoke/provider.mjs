// Durable browser-test source. The runner copies and executes it outside the repo.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const {default: puppeteer} = await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output = process.env.VOID_SMOKE_OUTPUT;
const evidence = {status: 'failed', fixture: 'serving_agent example; provider settings only, no model call', runtimeErrors: [], apiErrors: [], consoleWarnings: []};
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

const saved = () => page.evaluate(async()=>(await (await fetch('/api/projects/serving_agent')).json()).graph.nodes.find(n=>n.id==='reply').config.model);
const validation = g => page.evaluate(async g=>(await (await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph:g})})).json()),g);
try {
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForSelector('select[aria-label="open project"] option[value="example:serving_agent"]');
  await page.select('select[aria-label="open project"]','example:serving_agent');await page.waitForSelector('.aworkspace');
  // The load toast stays until clicked and can cover the workspace tabs on narrower font metrics (seen on CI); dismiss it like a user would.
  await page.waitForSelector('.toast');await page.$eval('.toast',e=>e.click());await page.waitForFunction(()=>!document.querySelector('.toast'));
  await (await page.waitForSelector('.react-flow__node[data-id="reply"]')).click();
  await page.waitForSelector('select[aria-label="provider"]');
  // Omitted settings show the op's declared defaults (the example relies on them), never "(not declared)" / "undefined".
  assert.equal(await page.$eval('select[aria-label="Messages from"]',e=>e.value),'prompt_messages');
  assert(!(await page.$$eval('.react-flow__node',ns=>ns.map(n=>n.textContent).join(' '))).includes('undefined'));
  await page.select('select[aria-label="provider"]','openai_compatible');
  assert.equal(await page.$eval('input[aria-label="base url"]',e=>e.value),'http://127.0.0.1:11434/v1');
  await page.select('select[aria-label="reasoning effort"]','none');
  await capture('provider-form');
  for(const h of await page.$$('button'))if(await h.evaluate(e=>e.textContent.trim().startsWith('Save')&&!e.disabled)){await h.click();break;}await text("Saved 'serving_agent'");
  const model=await saved();evidence.saved=model;
  assert.deepEqual([model.provider,model.base_url,model.reasoning_effort,model.api_key??null],['openai_compatible','http://127.0.0.1:11434/v1','none',null]);
  const graph=await page.evaluate(async()=>(await (await fetch('/api/projects/serving_agent')).json()).graph);
  const ok=await validation(graph);evidence.validation={ok:ok.ok,codes:(ok.diagnostics??[]).map(d=>d.code)};
  assert(!(ok.diagnostics??[]).some(d=>d.code==='E_MODEL_ENDPOINT'),JSON.stringify(evidence.validation));
  const reply=graph.nodes.find(n=>n.id==='reply');reply.config.model={...reply.config.model,base_url:'http://api.example.invalid/v1',api_key:{kind:'env',name:'OPENAI_API_KEY'}};
  const bad=await validation(graph);evidence.insecure=(bad.diagnostics??[]).filter(d=>d.code==='E_MODEL_ENDPOINT').map(d=>d.message);
  assert.equal(evidence.insecure.length,1);
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);
  evidence.status='passed';
  console.log('PASS provider: OpenAI-compatible endpoint chosen in the form, saved, validated; key over plain http to another host refused');
} catch (error) {
  evidence.error=String(error);await capture('failure').catch(()=>{});process.exitCode=1;
} finally {
  fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();
}
