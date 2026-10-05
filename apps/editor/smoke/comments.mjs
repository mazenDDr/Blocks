// Real owned Chrome, real saved drafts and API validation. No invented graph evidence.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const evidence={status:'failed',fixture:'SYNTHETIC native authored node comments',runtimeErrors:[],apiErrors:[],consoleWarnings:[]};
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
  await el.evaluate((el,v)=>{Object.getOwnPropertyDescriptor(el.tagName==='TEXTAREA'?HTMLTextAreaElement.prototype:HTMLInputElement.prototype,'value').set.call(el,v);el.dispatchEvent(new Event('input',{bubbles:true}));},String(value));
  await el.press('Tab');
};
const click=async(label)=>{
  await page.waitForFunction(t=>[...document.querySelectorAll('button')].some(b=>b.textContent.trim()===t&&!b.disabled),{},label);
  for(const h of await page.$$('button'))if(await h.evaluate((e,t)=>e.textContent.trim()===t&&!e.disabled,label))return h.click();
};
const disabled=selector=>page.$eval(selector,e=>e.disabled);
const save=async()=>{
  const pending=page.waitForResponse(r=>r.request().method()==='PUT'&&new URL(r.url()).pathname==='/api/projects/SYNTHETIC_comments');
  for(const h of await page.$$('.topbar button'))if(await h.evaluate(e=>e.textContent.startsWith('Save'))){await h.click();break;}
  const response=await pending;assert.equal(response.status(),200);
  return page.evaluate(async()=>{const r=await fetch('/api/projects/SYNTHETIC_comments');if(!r.ok)throw Error('Read saved draft '+r.status);return r.json();});
};
const shortcut=async(redo=false)=>{
  await page.evaluate(()=>document.activeElement?.blur());
  await page.keyboard.down(process.platform==='darwin'?'Meta':'Control');
  if(redo)await page.keyboard.down('Shift');
  await page.keyboard.press('z');
  if(redo)await page.keyboard.up('Shift');
  await page.keyboard.up(process.platform==='darwin'?'Meta':'Control');
};

const choose=async id=>{
  if(!(await page.$eval('.graph-outline',e=>e.open)))await page.click('.graph-outline summary');
  await page.waitForSelector(`[aria-label="inspect outline ${id}"]`);await page.click(`[aria-label="inspect outline ${id}"]`);
  if(!(await page.$eval('.node-comments',e=>e.open)))await page.click('.node-comments summary');
  await fill('input[aria-label="find node comments"]','');
  if(!(await page.$eval('.node-comment',e=>e.open)))await page.click('.node-comment summary');
};
const apply=async(text,author='SYNTHETIC declared researcher')=>{
  await fill('textarea[aria-label="node comment text"]',text);await fill('input[aria-label="node comment author"]',author);await click('Apply node comment');
};
const loadStored=async(data,id='SYNTHETIC_comments')=>{
  await page.evaluate(async ({id,data})=>{const r=await fetch('/api/projects/'+id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph:data.graph,ui:data.ui})});if(!r.ok)throw Error('Seed native project '+r.status);},{id,data});
  await page.select('select[aria-label="open project"]','project:'+id);await page.waitForFunction(()=>document.querySelector('button[aria-label="undo draft edit"]').disabled);
};
const textVisible=text=>page.waitForFunction(t=>document.querySelector('.node-comment')?.textContent.includes(t),{},text);
try{
  await page.goto(process.env.VOID_SMOKE_URL);await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='reference_cnn'&&document.querySelectorAll('.react-flow__node').length===10);
  await fill('input[aria-label="project id"]','SYNTHETIC_comments');const baseline=await save();evidence.baseline=baseline;
  evidence.stage='authored metadata with native provenance';await choose('conv_1');
  const authored='SYNTHETIC authored observation — Straße 🧪\n<img src=x onerror="window.syntheticCommentExecuted=true">';
  await apply(authored);const annotated=await save();evidence.annotated=annotated;
  assert.deepEqual(annotated.graph,baseline.graph);assert.equal(annotated.graphHash,baseline.graphHash);assert.deepEqual(annotated.ui.positions,baseline.ui.positions);
  const note=annotated.ui.nodeComments.conv_1;assert.equal(note.text,authored);assert.equal(note.author,'SYNTHETIC declared researcher');assert.equal(note.source.hash,baseline.graphHash);assert.equal(note.source.nodeId,'conv_1');assert.equal(note.source.nodeType,'pytorch.nn.conv2d');assert.equal(note.source.kind,'graph');assert(Number.isFinite(Date.parse(note.authoredAt)));
  await choose('conv_1');await textVisible('Comment refers to the current native identity.');assert.equal(await page.$$eval('.node-comment img',els=>els.length),0);assert(!(await page.evaluate(()=>window.syntheticCommentExecuted)));
  await click('Undo');assert.deepEqual((await save()).ui,baseline.ui);await click('Redo');assert.deepEqual((await save()).ui,annotated.ui);
  await choose('conv_1');await click('Remove node comment');const removed=await save();assert.deepEqual(removed.ui.nodeComments,{});assert.deepEqual(removed.graph,baseline.graph);await click('Undo');assert.deepEqual((await save()).ui,annotated.ui);
  evidence.stage='earlier identity review after real native graph edit';await choose('conv_1');await fill('input[aria-label="out_channels"]','64');await textVisible('Comment refers to an earlier node or native identity.');const changed=await save();assert.notEqual(changed.graphHash,baseline.graphHash);assert.deepEqual(changed.ui.nodeComments.conv_1,note);
  await click('Apply node comment');const reviewed=await save();assert.equal(reviewed.ui.nodeComments.conv_1.source.hash,changed.graphHash);assert.equal(reviewed.ui.nodeComments.conv_1.text,note.text);
  await click('Undo');assert.deepEqual((await save()).ui,changed.ui);await click('Undo');assert.deepEqual((await save()).graph,baseline.graph);
  evidence.stage='rename/delete preserve exact original provenance through one Undo';await choose('conv_1');await fill('input[aria-label="node id"]','conv_renamed');const renamed=await save();assert.deepEqual(renamed.ui.nodeComments.conv_renamed,note);assert(!Object.hasOwn(renamed.ui.nodeComments,'conv_1'));assert(!Object.hasOwn(renamed.ui.positions,'conv_1'));await choose('conv_renamed');await textVisible('Comment refers to an earlier node or native identity.');
  await click('Delete node');const deleted=await save();assert(!deleted.graph.nodes.some(n=>n.id==='conv_renamed'));assert(!Object.hasOwn(deleted.ui.nodeComments,'conv_renamed'));
  await click('Undo');const deleteUndo=await save();assert.deepEqual(deleteUndo.graph,renamed.graph);assert.deepEqual(deleteUndo.ui,renamed.ui);
  await click('Undo');const renameUndo=await save();assert.deepEqual(renameUndo.graph,annotated.graph);assert.deepEqual(renameUndo.ui,annotated.ui);evidence.rename={renamed,deleted};
  evidence.stage='comments stay with original copied node';await choose('conv_1');if(!(await page.$eval('.clipboard-tools',e=>e.open)))await page.click('.clipboard-tools summary');await click('Copy selected nodes');await page.waitForSelector('.clipboard-provenance');await click('Paste copied nodes');const copied=await save();assert.deepEqual(copied.ui.nodeComments,annotated.ui.nodeComments);assert(copied.graph.nodes.some(n=>n.id==='conv_1_copy'));await click('Undo');assert.deepEqual((await save()).ui,annotated.ui);
  evidence.stage='literal Unicode search and original inspector';if(!(await page.$eval('.node-comments',e=>e.open)))await page.click('.node-comments summary');await fill('input[aria-label="find node comments"]','straße');assert.equal(await page.$$eval('.node-comments tbody tr',rows=>rows.length),1);await page.click('[aria-label="inspect comment conv_1"]');await page.waitForFunction(()=>document.querySelector('input[aria-label="node id"]').value==='conv_1');await page.screenshot({path:path.join(output,'authored-node-comment.png'),fullPage:true});
  evidence.stage='orphan target metadata prevents silent rename overwrite';const conflict={...annotated,ui:{...annotated.ui,nodeComments:{...annotated.ui.nodeComments,conv_renamed:{...note,text:'SYNTHETIC orphan comment retained'}}}};await loadStored(conflict);await choose('conv_1');await fill('input[aria-label="node id"]','conv_renamed');await page.waitForFunction(()=>document.querySelector('.node-inspector .error')?.textContent.includes('E_ANNOTATION_CONFLICT'));const conflictRead=await save();assert.deepEqual(conflictRead.graph,conflict.graph);assert.deepEqual(conflictRead.ui,conflict.ui);assert(await disabled('button[aria-label="undo draft edit"]'));evidence.conflict=conflictRead;
  if(!(await page.$eval('.node-comments',e=>e.open)))await page.click('.node-comments summary');assert(await disabled('[aria-label="inspect comment conv_renamed"]'));await page.click('[aria-label="remove comment conv_renamed"]');assert(!Object.hasOwn((await save()).ui.nodeComments,'conv_renamed'));await click('Undo');assert.deepEqual((await save()).ui,conflict.ui);
  evidence.stage='malformed metadata is visible and never silently repaired';const malformed={...annotated,ui:{...annotated.ui,nodeComments:{conv_1:'SYNTHETIC invalid imported comment'}}};await loadStored(malformed);await choose('conv_1');await textVisible('E_ANNOTATION_FORMAT');assert.deepEqual((await save()).ui,malformed.ui);await click('Remove node comment');assert.deepEqual((await save()).ui.nodeComments,{});await click('Undo');assert.deepEqual((await save()).ui,malformed.ui);
  const collection={...annotated,ui:{...annotated.ui,nodeComments:null}};await loadStored(collection);if(!(await page.$eval('.node-comments',e=>e.open)))await page.click('.node-comments summary');await page.waitForFunction(()=>document.querySelector('.node-comments [role="alert"]')?.textContent.includes('E_ANNOTATION_FORMAT'));await click('Remove malformed comments collection');assert.deepEqual((await save()).ui.nodeComments,{});await click('Undo');assert.deepEqual((await save()).ui,collection.ui);
  evidence.stage='bounded metadata pages and no silent eviction';
  const many={...annotated,ui:{...annotated.ui,nodeComments:Object.fromEntries(Array.from({length:200},(_,i)=>[`orphan_${String(i).padStart(3,'0')}`,{...note,text:`SYNTHETIC orphan authored record ${i}`}]))}};
  await loadStored(many);await choose('conv_1');assert.equal(await page.$$eval('.node-comments tbody tr',rows=>rows.length),25);
  await click('Next comments');assert((await page.$eval('.node-comments tbody tr th',e=>e.textContent)).startsWith('orphan_025'));await click('Previous comments');
  await apply('SYNTHETIC attempted comment201');await page.waitForFunction(()=>document.querySelector('.node-comment [role="alert"]')?.textContent.includes('E_ANNOTATION_LIMIT'));
  assert.deepEqual((await save()).ui,many.ui);assert(await disabled('button[aria-label="undo draft edit"]'));evidence.limits={count:Object.keys(many.ui.nodeComments).length,rowsPerPage:25,noEviction:true};
  evidence.stage='valid native node ID shadows Object prototype without losing metadata';
  const prototypeSource={graph:{schemaVersion:'1.0.0',graphKind:'model',backend:'pytorch',nodes:[{id:'constructor',type:'core.tensor_input',version:'1.0.0',config:{shape:['N',4],dtype:'float32'}}],edges:[]},ui:{schemaVersion:'1.0.0',synthetic:true,description:'SYNTHETIC native input to verify comment keys; no training claim',positions:{constructor:{x:100,y:100}},nodeComments:{}}};
  await loadStored(prototypeSource);const prototypeBaseline=await save();await choose('constructor');await apply('SYNTHETIC own comment for valid native constructor ID');const prototypeAnnotated=await save();assert(Object.hasOwn(prototypeAnnotated.ui.nodeComments,'constructor'));assert.equal(prototypeAnnotated.ui.nodeComments.constructor.source.hash,prototypeBaseline.graphHash);assert.deepEqual(prototypeAnnotated.graph,prototypeBaseline.graph);evidence.prototype=prototypeAnnotated;
  evidence.stage='module-definition identity and layout remain separate';await page.select('select[aria-label="open project"]','example:residual_cnn');await page.waitForFunction(()=>document.querySelector('input[aria-label="project id"]').value==='residual_cnn');await fill('input[aria-label="project id"]','SYNTHETIC_comments');const moduleSource=await save();
  if(!(await page.$eval('.graph-outline',e=>e.open)))await page.click('.graph-outline summary');const response=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/modules/validate'&&r.status()===200);await page.click('[aria-label="module outline res1"]');const moduleValidation=await(await response).json();assert(moduleValidation.ok);await choose('conv_a');await apply('SYNTHETIC reusable definition comment');const moduleAnnotated=await save();const key='residual_block@1.0.0/conv_a';assert.equal(moduleAnnotated.ui.nodeComments[key].source.kind,'module');assert.equal(moduleAnnotated.ui.nodeComments[key].source.hash,moduleValidation.moduleHash);assert.deepEqual(moduleAnnotated.graph,moduleSource.graph);assert.deepEqual(moduleAnnotated.ui.positions,moduleSource.ui.positions);await choose('conv_a');await textVisible('Comment refers to the current native identity.');evidence.module={source:moduleSource,annotated:moduleAnnotated,validation:moduleValidation};await page.screenshot({path:path.join(output,'module-node-comment.png'),fullPage:true});await click('Undo');assert.deepEqual((await save()).ui,moduleSource.ui);
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);evidence.browserVersion=await browser.version();evidence.status='passed';console.log('PASS authored node comments → exact native provenance → stale review → rename/delete Undo → copy isolation → conflict/malformed refusal → scoped module identity');
}catch(error){evidence.error=String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
