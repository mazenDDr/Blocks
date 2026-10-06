// Real owned Chrome, native worker process, real Gymnasium Pendulum-v1. Short budget: workflow evidence, not a learning-quality claim.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const evidence={status:'failed',fixture:'rl_pendulum_td3 example with a 600-step budget',runtimeErrors:[],apiErrors:[],consoleWarnings:[]};
const browser=await puppeteer.launch({executablePath:process.env.VOID_SMOKE_CHROME,headless:true,args:process.platform==='linux'?['--no-sandbox']:[],defaultViewport:{width:1600,height:1100}});
process.once('SIGTERM',()=>void browser.close());process.once('SIGINT',()=>void browser.close());
const page=await browser.newPage();page.setDefaultTimeout(90000);
page.on('pageerror',e=>evidence.runtimeErrors.push(e.message));
page.on('dialog',d=>d.accept());
page.on('console',m=>{if(['warning','error'].includes(m.type()))evidence.consoleWarnings.push({type:m.type(),text:m.text()});});
page.on('response',r=>{if(new URL(r.url()).pathname.startsWith('/api/')&&r.status()>=400)evidence.apiErrors.push({status:r.status(),url:r.url()});});
const click=async(label)=>{
  await page.waitForFunction(t=>[...document.querySelectorAll('button')].some(b=>b.textContent.trim()===t&&!b.disabled),{},label);
  for(const h of await page.$$('button'))if(await h.evaluate((e,t)=>e.textContent.trim()===t&&!e.disabled,label))return h.click();
};
const setNum=async(label,value)=>{
  const el=await page.waitForSelector(`input[aria-label="${label}"]`);
  await el.evaluate((el,v)=>{Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(el,v);el.dispatchEvent(new Event('input',{bubbles:true}));},String(value));
  await el.press('Tab');
};
try{
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForFunction(()=>document.querySelectorAll('.react-flow__node').length===10);
  await page.waitForSelector('select[aria-label="open project"] option[value="example:rl_pendulum_td3"]');
  await page.select('select[aria-label="open project"]','example:rl_pendulum_td3');
  await page.waitForFunction(()=>document.querySelector('.rlbar')?.textContent.includes('RL · TD3'));
  await click('Learner');
  await page.waitForFunction(()=>document.querySelector('.rllearner h3')?.textContent.includes('TD3'));
  await setNum('Environment steps',600);await setNum('Random steps before learning',200);await setNum('Evaluate every (steps)',300);await setNum('Batch size',64);
  await setNum('td3 hidden sizes','32, 32');
  const shown=await page.$$eval('.rllearner input',els=>Object.fromEntries(els.map(e=>[e.getAttribute('aria-label'),e.value])));
  assert.equal(shown['Discount gamma'],'0.99');assert.equal(shown['Critic updates per actor update'],'2');  // defaults shown for omitted fields
  evidence.shownFields=shown;
  await page.screenshot({path:path.join(output,'td3-learner.png'),fullPage:true});
  const posted=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/runs'&&r.request().method()==='POST');
  await click('Run');
  const runId=(await (await posted).json()).runId;assert(runId);
  const final=await page.waitForFunction(async id=>{const r=await fetch('/api/runs/'+id);const j=await r.json();return ['completed','failed','cancelled'].includes(j.status)?j:null;},{polling:500,timeout:120000},runId).then(h=>h.jsonValue());
  assert.equal(final.status,'completed',JSON.stringify(final.error));
  const curves=await page.evaluate(async id=>(await fetch('/api/rl/runs/'+id+'/curves')).json(),runId);
  assert.equal(curves.totalSteps,600);assert.equal(curves.evals.length,2);assert.equal(curves.episodes.length,3);
  const saved=await page.evaluate(async()=>(await fetch('/api/projects/rl_pendulum_td3')).json());
  const learner=saved.graph.nodes.find(n=>n.type==='rl.td3_learner');
  assert.deepEqual([learner.config.total_steps,learner.config.learning_starts,learner.config.eval_every,learner.config.batch_size,learner.config.hidden],[600,200,300,64,[32,32]]);
  await click('Evaluation');await new Promise(r=>setTimeout(r,800));
  await page.screenshot({path:path.join(output,'td3-evaluation.png'),fullPage:true});
  evidence.run={runId,status:final.status,evalTicks:curves.evals.map(e=>e.tick),finalTaskReturn:curves.evals.at(-1).taskReturn};
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';
  console.log('PASS TD3 example → learner edits saved → native worker run on Pendulum-v1 → curves/evaluation in the editor');
}catch(error){evidence.error=String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
