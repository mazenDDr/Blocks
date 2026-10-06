// Durable browser-test source. The runner copies and executes it outside the repo.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';

const {default: puppeteer} = await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output = process.env.VOID_SMOKE_OUTPUT;
const evidence = {status: 'failed', fixture: 'tool_agent_calculator example; block settings only, no model call', runtimeErrors: [], apiErrors: [], consoleWarnings: []};
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

const saved = () => page.evaluate(async()=>(await (await fetch('/api/projects/tool_agent_calculator')).json()).graph.nodes.find(n=>n.id==='solve').config);
try {
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForSelector('select[aria-label="open project"] option[value="example:tool_agent_calculator"]');
  await page.select('select[aria-label="open project"]','example:tool_agent_calculator');await page.waitForSelector('.aworkspace');
  // The load toast stays until clicked and can cover the workspace tabs on narrower font metrics (seen on CI); dismiss it like a user would.
  await page.waitForSelector('.toast');await page.$eval('.toast',e=>e.click());await page.waitForFunction(()=>!document.querySelector('.toast'));
  const card=await page.$eval('.react-flow__node[data-id="solve"]',e=>e.textContent);assert(card.includes('may call calculator')&&card.includes('≤3 calls'),card);
  await (await page.waitForSelector('.react-flow__node[data-id="solve"]')).click();
  assert.equal(await page.$eval('input[aria-label="offer tool calculator"]',e=>e.checked),true);
  await (await page.waitForSelector('input[aria-label="offer tool read_text_file"]')).click();
  await text('read_text_file needs the directory it may read');
  await fill('textarea[aria-label="allowed directory"]','examples/fixtures');
  await page.waitForFunction(()=>!document.body.innerText.includes('read_text_file needs the directory it may read'));
  await fill('input[aria-label="max tool calls"]','2');
  await capture('tool-agent-form');
  for(const h of await page.$$('button'))if(await h.evaluate(e=>e.textContent.trim().startsWith('Save')&&!e.disabled)){await h.click();break;}await text("Saved 'tool_agent_calculator'");
  const cfg=await saved();evidence.saved={tools:cfg.tools,allowed_dir:cfg.allowed_dir,max_tool_calls:cfg.max_tool_calls};
  assert.deepEqual(evidence.saved,{tools:['calculator','read_text_file'],allowed_dir:'examples/fixtures',max_tool_calls:2});
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);
  evidence.status='passed';
  console.log('PASS tool agent: card summary, tool offers, read-directory diagnostic, limit, saved config');
} catch (error) {
  evidence.error=String(error);await capture('failure').catch(()=>{});process.exitCode=1;
} finally {
  fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();
}
