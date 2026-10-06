// Automated accessibility audit (ADR 0072): Chrome's own accessibility tree for every workspace of four example kinds.
// Counts interactive controls without an accessible name and images without alternative text. Not a substitute for a human review.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const strict=process.env.VOID_A11Y_STRICT!=='0';
const evidence={status:'failed',method:'CDP Accessibility.getFullAXTree after each workspace tab; roles button/link/textbox/searchbox/combobox/listbox/checkbox/radio/slider/spinbutton/switch/tab/menuitem/option with an empty computed name, and img without a name',
  runtimeErrors:[],apiErrors:[],views:[],violations:[]};
const browser=await puppeteer.launch({executablePath:process.env.VOID_SMOKE_CHROME,headless:true,args:process.platform==='linux'?['--no-sandbox']:[],defaultViewport:{width:1600,height:1100}});
process.once('SIGTERM',()=>void browser.close());process.once('SIGINT',()=>void browser.close());
const page=await browser.newPage();page.setDefaultTimeout(60000);
page.on('pageerror',e=>evidence.runtimeErrors.push(e.message));
page.on('dialog',d=>d.accept());
page.on('response',r=>{if(new URL(r.url()).pathname.startsWith('/api/')&&r.status()>=400)evidence.apiErrors.push({status:r.status(),url:r.url()});});
const cdp=await page.createCDPSession();await cdp.send('Accessibility.enable');await cdp.send('DOM.enable');
const ROLES=new Set(['button','link','textbox','searchbox','combobox','listbox','checkbox','radio','slider','spinbutton','switch','tab','menuitem','option']);
async function audit(label){
  const {nodes}=await cdp.send('Accessibility.getFullAXTree');
  const found=[];
  for(const n of nodes){
    if(n.ignored)continue;
    const role=n.role?.value, name=(n.name?.value??'').trim();
    if(!((ROLES.has(role)&&!name)||((role==='img'||role==='image')&&!name)))continue;
    let where=null;
    if(n.backendDOMNodeId){try{const d=await cdp.send('DOM.describeNode',{backendNodeId:n.backendDOMNodeId});const a=d.node.attributes??[];const at={};for(let i=0;i<a.length;i+=2)at[a[i]]=a[i+1];
      where={tag:d.node.nodeName.toLowerCase(),class:at.class??null,id:at.id??null,type:at.type??null,placeholder:at.placeholder??null,value:at.value??null};}catch{}}
    found.push({view:label,role,where});
  }
  evidence.views.push({view:label,unnamed:found.length});evidence.violations.push(...found);
}
async function openProject(value,check){
  await page.select('select[aria-label="open project"]',value);await page.waitForFunction(check);await new Promise(r=>setTimeout(r,700));
}
async function eachTab(project){
  const tabs=await page.$$eval('[role="tablist"][aria-label="workspace"] button',bs=>bs.map(b=>b.textContent.trim()));
  for(const t of tabs){
    for(const h of await page.$$('[role="tablist"][aria-label="workspace"] button'))if(await h.evaluate((e,x)=>e.textContent.trim()===x,t)){await h.click();break;}
    await new Promise(r=>setTimeout(r,900));
    await audit(`${project} · ${t}`);
  }
}
try{
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForFunction(()=>document.querySelectorAll('.react-flow__node').length===10);
  // Self-check: the audit must flag deliberately unnamed controls before its zero counts mean anything.
  await page.evaluate(()=>{const d=document.createElement('div');d.id='a11y-self-check';d.innerHTML='<button></button><input type="text"><img src="data:image/gif;base64,R0lGODlhAQABAAAAACw=">';document.body.appendChild(d);});
  await new Promise(r=>setTimeout(r,200));
  await audit('self-check (3 injected unnamed controls)');
  const injected=evidence.violations.filter(v=>v.view.startsWith('self-check'));
  evidence.selfCheck={flagged:injected.length,roles:injected.map(v=>v.role)};
  assert(injected.length>=3,'audit failed to flag injected unnamed controls: '+JSON.stringify(injected));
  evidence.violations=evidence.violations.filter(v=>!v.view.startsWith('self-check'));evidence.views=evidence.views.filter(v=>!v.view.startsWith('self-check'));
  await page.evaluate(()=>document.getElementById('a11y-self-check').remove());
  await eachTab('model reference_cnn');
  await openProject('example:production_sensors',()=>document.querySelector('input[aria-label="project id"]').value==='production_sensors');await eachTab('tabular production_sensors');
  await openProject('example:serving_state',()=>document.querySelector('input[aria-label="project id"]').value==='serving_state');await eachTab('agent serving_state');
  await openProject('example:rl_cartpole_dqn',()=>document.querySelector('input[aria-label="project id"]').value==='rl_cartpole_dqn');await eachTab('rl rl_cartpole_dqn');
  const byKind={};for(const v of evidence.violations){const k=`${v.role}:${v.where?.tag}:${v.where?.type??''}:${v.where?.class??''}`;byKind[k]=(byKind[k]??0)+1;}
  evidence.summary={views:evidence.views.length,violations:evidence.violations.length,byKind};
  fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));
  assert.deepEqual(evidence.runtimeErrors,[]);
  if(strict)assert.equal(evidence.violations.length,0,JSON.stringify(byKind));
  evidence.status='passed';
  console.log(`PASS accessibility audit: ${evidence.views.length} views, ${evidence.violations.length} unnamed controls`);
}catch(error){evidence.error=String(error);process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
