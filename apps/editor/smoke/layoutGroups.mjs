// Real owned Chrome, real saved drafts and native validation. Layout-only evidence; no model-quality claim.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const evidence={status:'failed',fixture:'SYNTHETIC layout group on the native reference_cnn example',runtimeErrors:[],apiErrors:[],consoleWarnings:[]};
const browser=await puppeteer.launch({executablePath:process.env.VOID_SMOKE_CHROME,headless:true,
  args:process.platform==='linux'?['--no-sandbox']:[],defaultViewport:{width:1600,height:1100}});
process.once('SIGTERM',()=>void browser.close());process.once('SIGINT',()=>void browser.close());
const page=await browser.newPage();page.setDefaultTimeout(45000);
page.on('pageerror',e=>evidence.runtimeErrors.push(e.message));
page.on('dialog',d=>d.accept());
page.on('console',m=>{if(['warning','error'].includes(m.type()))evidence.consoleWarnings.push({type:m.type(),text:m.text()});});
page.on('response',r=>{if(new URL(r.url()).pathname.startsWith('/api/')&&r.status()>=400)evidence.apiErrors.push({status:r.status(),url:r.url()});});
const ID='SYNTHETIC_layout_groups',LABEL='SYNTHETIC stem',MEMBERS=['conv_1','relu_1','pool_1'];
const fill=async(selector,value)=>{
  const el=await page.waitForSelector(selector);
  await el.evaluate((el,v)=>{Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(el,v);el.dispatchEvent(new Event('input',{bubbles:true}));},String(value));
};
const click=async(label)=>{
  await page.waitForFunction(t=>[...document.querySelectorAll('button')].some(b=>b.textContent.trim().startsWith(t)&&!b.disabled),{},label);
  for(const h of await page.$$('button'))if(await h.evaluate((e,t)=>e.textContent.trim().startsWith(t)&&!e.disabled,label))return h.click();
};
const toast=text=>page.waitForFunction(t=>[...document.querySelectorAll('.toast')].some(e=>e.textContent.includes(t)),{},text);
const read=()=>page.evaluate(async id=>{const r=await fetch('/api/projects/'+id);if(!r.ok)throw Error('Read saved draft '+r.status);return r.json();},ID);
const save=async()=>{
  const pending=page.waitForResponse(r=>r.request().method()==='PUT'&&new URL(r.url()).pathname==='/api/projects/'+ID);
  for(const h of await page.$$('.topbar button'))if(await h.evaluate(e=>e.textContent.startsWith('Save'))){await h.click();break;}
  assert.equal((await pending).status(),200);return read();
};
const undo=async()=>{await page.evaluate(()=>document.activeElement?.blur());await page.keyboard.down(process.platform==='darwin'?'Meta':'Control');await page.keyboard.press('z');await page.keyboard.up(process.platform==='darwin'?'Meta':'Control');};
const open=async sel=>{if(!(await page.$eval(sel,e=>e.open)))await page.click(sel+' summary');};
const rect=sel=>page.$eval(sel,e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height};});
const frameSel=`[aria-label="layout group ${LABEL}"]`;
try{
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length===10);
  await fill('input[aria-label="project id"]',ID);await page.keyboard.press('Tab');const baseline=await save();
  evidence.stage='group three selected cards: layout-only edit, frame encloses its cards';
  await open('.arrangement-tools');for(const m of MEMBERS)await page.click(`input[aria-label="arrange node ${m}"]`);
  await open('.layout-groups');await fill('input[aria-label="new layout group name"]',LABEL);
  await click('Group selected cards (3)');await toast(`Created layout group '${LABEL}'`);
  const grouped=await save();
  assert.deepEqual(grouped.graph,baseline.graph);assert.equal(grouped.graphHash,baseline.graphHash);
  assert.deepEqual(grouped.ui.layoutGroups,[{id:'group_1',label:LABEL,scope:'root',members:MEMBERS}]);
  await page.waitForSelector(frameSel);const frame=await rect(frameSel);
  for(const m of MEMBERS){const c=await rect(`.react-flow__node[data-id="${m}"]`);assert(c.x>=frame.x&&c.y>=frame.y&&c.x+c.w<=frame.x+frame.w&&c.y+c.h<=frame.y+frame.h,`${m} outside frame`);}
  await page.screenshot({path:path.join(output,'layout-group.png'),fullPage:true});
  evidence.stage='drag the frame head: exactly the members move by one shared offset';
  const head=await rect(frameSel+' .layout-frame-head');
  await page.mouse.move(head.x+10,head.y+8);await page.mouse.down();await page.mouse.move(head.x+70,head.y+48,{steps:8});await page.mouse.move(head.x+130,head.y+88,{steps:8});await page.mouse.up();
  const dragged=await save();const before=grouped.ui.positions,after=dragged.ui.positions;
  const deltas=MEMBERS.map(m=>[after[m].x-before[m].x,after[m].y-before[m].y]);
  assert(deltas.every(d=>d[0]===deltas[0][0]&&d[1]===deltas[0][1]),JSON.stringify(deltas));assert(deltas[0][0]>0&&deltas[0][1]>0,JSON.stringify(deltas));
  for(const id of Object.keys(before).filter(k=>!MEMBERS.includes(k)))assert.deepEqual(after[id],before[id],id);
  assert.deepEqual(dragged.graph,baseline.graph);assert.deepEqual(dragged.ui.layoutGroups,grouped.ui.layoutGroups);
  assert(!Object.keys(after).some(k=>k.startsWith('frame:')),'frame leaked into positions');
  evidence.dragOffset=deltas[0];
  evidence.stage='one Undo restores the pre-drag layout; reload keeps the group';
  await undo();assert.deepEqual((await save()).ui,grouped.ui);
  await page.reload();await page.waitForSelector(`select[aria-label="open project"] option[value="project:${ID}"]`);
  await page.select('select[aria-label="open project"]','project:'+ID);await toast(`Loaded project '${ID}'`);await page.waitForSelector(frameSel);
  evidence.stage='clicking the frame selects its cards; deleting the group leaves cards in place';
  const h2=await rect(frameSel+' .layout-frame-head');await page.mouse.click(h2.x+h2.w-20,h2.y+h2.h/2);// the header band: wires can cross the frame body
  await open('.arrangement-tools');await page.waitForFunction(()=>document.querySelector('.arrangement-tools legend')?.textContent.includes('(3)'));
  await open('.layout-groups');await page.click('button[aria-label="delete layout group group_1"]');await toast('Deleted the layout group');
  const removed=await save();assert.deepEqual(removed.ui.layoutGroups,[]);assert.deepEqual(removed.ui.positions,grouped.ui.positions);
  assert.equal(await page.$(frameSel),null);
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);
  evidence.browserVersion=await browser.version();evidence.status='passed';
  console.log('PASS layout group → enclosing frame → frame drag moves members only → Undo → reload → frame selection → delete');
}catch(error){evidence.error=String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
