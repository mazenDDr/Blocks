// Durable browser-test source. The runner copies and executes it outside the repo.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const {default: puppeteer} = await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output = process.env.VOID_SMOKE_OUTPUT;
const evidence = {status: 'failed', fixture: 'serving_state example (model-free); 3 SYNTHETIC cases, one designed to fail', runtimeErrors: [], apiErrors: [], consoleWarnings: []};
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

const CASES=[{id:'c1',input:{question:'SYNTHETIC alpha'},checks:[{field:'answer',kind:'contains',value:'SYNTHETIC alpha:'}]},
  {id:'c2',input:{question:'SYNTHETIC beta'},checks:[{field:'answer',kind:'regex',value:'^SYNTHETIC beta: \\d+/\\d+$'}]},
  {id:'c3',input:{question:'SYNTHETIC gamma'},checks:[{field:'answer',kind:'equals',value:'designed to fail'}]}];
try {
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForSelector('select[aria-label="open project"] option[value="example:serving_state"]');
  await page.select('select[aria-label="open project"]','example:serving_state');await page.waitForSelector('.aworkspace');
  await click('Evaluate');
  await fill('textarea[aria-label="evaluation cases"]',JSON.stringify(CASES));await fill('input[aria-label="evaluation name"]','SYNTHETIC browser evaluation');
  const posted=response('/api/runs');await click('Run evaluation');const sub=await(await posted).json();assert.equal(sub.cases,3);evidence.evaluation=sub.runId;
  await page.waitForFunction(()=>document.querySelector('.eval-score')?.textContent.includes('2 / 3'));
  const rows=await page.$$eval('[aria-label="evaluation cases result"] tbody tr',trs=>trs.map(tr=>[...tr.children].slice(0,3).map(td=>td.textContent.trim())));
  evidence.rows=rows;
  assert.deepEqual(rows.map(r=>r[1]),['pass','pass','fail']);assert(rows[2][2].includes('"designed to fail"')&&rows[2][2].includes('SYNTHETIC gamma: '));
  const report=await get(`/api/agent/evals/${sub.runId}`);evidence.report={passed:report.report.passed,passRate:report.report.passRate,wilson95:report.report.wilson95};
  assert.deepEqual([report.report.passed,report.report.passRate],[2,0.6667]);
  await capture('evaluation');
  const child=report.cases[0].childRunId;
  await (await page.waitForSelector('button[aria-label="open run of case c1"]')).click();
  await page.waitForFunction(id=>document.body.innerText.includes(id),{},child);
  evidence.openedChild=child;
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);
  evidence.status='passed';
  console.log('PASS agent evaluation: cases entered → evaluation run → 2/3 with the recorded failure → child run opened');
} catch (error) {
  evidence.error=String(error);await capture('failure').catch(()=>{});process.exitCode=1;
} finally {
  fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();
}
