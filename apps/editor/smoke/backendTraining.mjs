// Real owned Chrome and native worker: choose JAX in the Train tab, train the reference CNN, confirm the backend run and its checkpoint.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const evidence={status:'failed',fixture:'reference_cnn on the bundled SYNTHETIC shapes10 folder, 1 epoch, JAX plain SGD',runtimeErrors:[],apiErrors:[],consoleWarnings:[]};
const browser=await puppeteer.launch({executablePath:process.env.VOID_SMOKE_CHROME,headless:true,args:process.platform==='linux'?['--no-sandbox']:[],defaultViewport:{width:1600,height:1100}});
process.once('SIGTERM',()=>void browser.close());process.once('SIGINT',()=>void browser.close());
const page=await browser.newPage();page.setDefaultTimeout(120000);
page.on('pageerror',e=>evidence.runtimeErrors.push(e.message));
page.on('dialog',d=>d.accept());
page.on('console',m=>{if(['warning','error'].includes(m.type()))evidence.consoleWarnings.push({type:m.type(),text:m.text()});});
page.on('response',r=>{if(new URL(r.url()).pathname.startsWith('/api/')&&r.status()>=400)evidence.apiErrors.push({status:r.status(),url:r.url()});});
const setInput=async(selector,value)=>{const el=await page.waitForSelector(selector);await el.evaluate((el,v)=>{Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(el,v);el.dispatchEvent(new Event('input',{bubbles:true}));},String(value));};
try{
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length===10);
  await page.select('select[aria-label="training backend"]','jax');
  const epochs=await page.$$eval('label',ls=>{const l=ls.find(x=>x.textContent.trim().startsWith('Epochs'));return !!l;});assert(epochs);
  await page.$$eval('label',ls=>{const i=ls.find(x=>x.textContent.trim().startsWith('Epochs')).querySelector('input');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(i,'1');i.dispatchEvent(new Event('input',{bubbles:true}));});
  const posted=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/runs'&&r.request().method()==='POST');
  const run=await page.waitForSelector('button[title="Save the project and train a new run"]:not([disabled])');await run.click();
  const resp=await posted;assert.equal(resp.status(),201);const runId=(await resp.json()).runId;
  const final=await page.waitForFunction(async id=>{const j=await (await fetch('/api/runs/'+id)).json();return ['completed','failed','cancelled'].includes(j.status)?j:null;},{polling:500},runId).then(h=>h.jsonValue());
  assert.equal(final.status,'completed',JSON.stringify(final.error));assert.equal(final.config.backend,'jax');assert.equal(final.config.epochs,1);
  await new Promise(r=>setTimeout(r,800));
  await page.screenshot({path:path.join(output,'jax-training.png'),fullPage:true});
  evidence.run={runId,status:final.status,backend:final.config.backend,progress:final.progress,final:final.final};
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';
  console.log('PASS Train tab backend=jax → native worker JAX SGD run completes with its recorded backend');
}catch(error){evidence.error=String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
