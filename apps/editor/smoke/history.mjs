// Real owned Chrome, real saved drafts and API validation. No invented graph evidence.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const evidence={status:'failed',fixture:'SYNTHETIC native graph draft history',runtimeErrors:[],apiErrors:[],consoleWarnings:[]};
const browser=await puppeteer.launch({executablePath:process.env.VOID_SMOKE_CHROME,headless:true,
  args:process.platform==='linux'?['--no-sandbox']:[],defaultViewport:{width:1600,height:1100}});
process.once('SIGTERM',()=>void browser.close());process.once('SIGINT',()=>void browser.close());
const page=await browser.newPage();page.setDefaultTimeout(45000);
page.on('pageerror',e=>evidence.runtimeErrors.push(e.message));
page.on('dialog',d=>d.accept());
page.on('console',m=>{if(['warning','error'].includes(m.type()))evidence.consoleWarnings.push({type:m.type(),text:m.text()});});
page.on('response',r=>{if(new URL(r.url()).pathname.startsWith('/api/')&&r.status()>=400)evidence.apiErrors.push({status:r.status(),url:r.url()});});
const fill=async(selector,value)=>{
  const el=await page.waitForSelector(selector);
  await el.evaluate((el,v)=>{Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(el,v);el.dispatchEvent(new Event('input',{bubbles:true}));},String(value));
  await el.press('Tab');
};
const click=async(label)=>{
  await page.waitForFunction(t=>[...document.querySelectorAll('button')].some(b=>b.textContent.trim()===t&&!b.disabled),{},label);
  for(const h of await page.$$('button'))if(await h.evaluate((e,t)=>e.textContent.trim()===t&&!e.disabled,label))return h.click();
};
const disabled=selector=>page.$eval(selector,e=>e.disabled);
const save=async()=>{
  const pending=page.waitForResponse(r=>r.request().method()==='PUT'&&new URL(r.url()).pathname==='/api/projects/SYNTHETIC_undo');
  for(const h of await page.$$('.topbar button'))if(await h.evaluate(e=>e.textContent.startsWith('Save'))){await h.click();break;}
  const response=await pending;assert.equal(response.status(),200);
  return page.evaluate(async()=>{const r=await fetch('/api/projects/SYNTHETIC_undo');if(!r.ok)throw Error('Read saved draft '+r.status);return r.json();});
};
const shortcut=async(redo=false)=>{
  await page.evaluate(()=>document.activeElement?.blur());
  await page.keyboard.down(process.platform==='darwin'?'Meta':'Control');
  if(redo)await page.keyboard.down('Shift');
  await page.keyboard.press('z');
  if(redo)await page.keyboard.up('Shift');
  await page.keyboard.up(process.platform==='darwin'?'Meta':'Control');
};
try{
  await page.goto(process.env.VOID_SMOKE_URL);await page.waitForSelector('select[aria-label="open project"]');
  evidence.stage='wait initial boot';
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn' && document.querySelectorAll('.react-flow__node').length>0);
  await page.waitForFunction(()=>[...document.querySelector('select[aria-label="open project"]').options].some(o=>o.value==='example:serving_state'));
  await page.select('select[aria-label="open project"]','example:serving_state');await page.waitForSelector('.aworkspace');
  await fill('input[aria-label="project id"]','SYNTHETIC_undo');
  assert(await disabled('[aria-label="undo draft edit"]'));assert(await disabled('[aria-label="redo draft edit"]'));
  evidence.stage='save baseline';const baseline=await save();evidence.baseline=baseline;
  await fill('input[aria-label="Search blocks"]','agent.set_state');
  await page.click('.lib-item');await page.waitForFunction(n=>document.querySelectorAll('.react-flow__node').length===n+3,{},baseline.graph.nodes.length);
  evidence.stage='save added';const added=await save();evidence.added=added;
  assert.equal(added.graph.nodes.length,baseline.graph.nodes.length+1);
  assert(Object.keys(added.ui.positions).length>Object.keys(baseline.ui.positions).length);
  await click('Undo');const undone=await save();assert.deepEqual(undone.graph,baseline.graph);assert.deepEqual(undone.ui,baseline.ui);
  await shortcut(true);const redone=await save();assert.deepEqual(redone.graph,added.graph);assert.deepEqual(redone.ui,added.ui);
  await shortcut();const again=await save();assert.deepEqual(again.graph,baseline.graph);assert.deepEqual(again.ui,baseline.ui);
  // A full drag is one edit, independent of animation frame count.
  evidence.stage='drag';const node=await page.$('.react-flow__node[data-id="tick"]');assert(node);
  // A fitted canvas must fill the workspace and the drag must hit this actual card.
  await page.click('.react-flow__controls-fitview');
  await page.waitForFunction(()=>{
    const transform=document.querySelector('.react-flow__viewport')?.style.transform;
    if(window.__historyFit?.transform!==transform)window.__historyFit={transform,since:performance.now()};
    return window.__historyFit && performance.now()-window.__historyFit.since>350;
  });
  await page.waitForFunction(()=>{
    const n=document.querySelector('.react-flow__node[data-id="tick"]'), c=document.querySelector('.acanvas .center');
    if(!n||!c||c.getBoundingClientRect().height<400)return false;
    const b=n.getBoundingClientRect();return document.elementFromPoint(b.x+b.width/2,b.y+b.height/2)?.closest('.react-flow__node')===n;
  });
  const box=await node.boundingBox();assert(box);
  evidence.dragTarget=await node.evaluate(n=>({node:n.getBoundingClientRect().toJSON(),canvas:n.closest('.center').getBoundingClientRect().toJSON()}));
  await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();
  await page.mouse.move(box.x+box.width/2+80,box.y+box.height/2+80,{steps:15});await page.mouse.up();
  await page.waitForFunction(()=>!document.querySelector('[aria-label="undo draft edit"]').disabled);
  const moved=await save();assert.deepEqual(moved.graph,baseline.graph);assert.notDeepEqual(moved.ui.positions.tick,baseline.ui.positions.tick);
  assert(await disabled('[aria-label="redo draft edit"]'));
  await click('Undo');const beforeDrag=await save();assert.deepEqual(beforeDrag.ui,baseline.ui);assert.deepEqual(beforeDrag.graph,baseline.graph);
  await click('Redo');const afterDrag=await save();assert.deepEqual(afterDrag.ui,moved.ui);evidence.drag={before:baseline.ui,after:moved.ui};
  // Draft settings on a different graph family use the same history and clear on load.
  await page.select('select[aria-label="open project"]','example:reference_cnn');await page.waitForSelector('select[aria-label="backend"]');
  assert(await disabled('[aria-label="undo draft edit"]'));assert(await disabled('[aria-label="redo draft edit"]'));
  await fill('input[aria-label="project id"]','SYNTHETIC_undo');
  evidence.stage='model settings';const model=await save();await page.select('select[aria-label="backend"]','jax');
  const changed=await save();assert.equal(changed.graph.backend,'jax');
  await click('Undo');const modelUndo=await save();assert.deepEqual(modelUndo.graph,model.graph);assert.deepEqual(modelUndo.ui,model.ui);
  await page.focus('input[aria-label="project id"]');await page.keyboard.down(process.platform==='darwin'?'Meta':'Control');await page.keyboard.press('z');await page.keyboard.up(process.platform==='darwin'?'Meta':'Control');
  assert(!(await disabled('[aria-label="redo draft edit"]'))); // Text-field shortcut cannot consume draft redo.
  evidence.model={before:model.graph.backend,after:changed.graph.backend};
  await page.screenshot({path:path.join(output,'draft-history.png'),fullPage:true});
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);
  assert.deepEqual(evidence.consoleWarnings.filter(m=>m.type==='error'),[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';
  console.log('PASS add/settings/layout → whole-document undo/redo → grouped drag → new-edit invalidation → project reset/text focus');
}catch(error){evidence.error=String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
