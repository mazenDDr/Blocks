// Descriptive timings from actual owned Chrome and native API; no latency target assertions.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const {default:puppeteer}=await import(pathToFileURL(process.env.VOID_SMOKE_PUPPETEER).href);
const output=process.env.VOID_SMOKE_OUTPUT;
const evidence={status:'failed',fixture:'SYNTHETIC tensor input → ReLU chains; metadata only, no native execution/model quality claim',
  methodology:{sizes:[100,500,1000],validationSamples:20,loadSamples:10,selectionSamples:20,
    percentile:'nearest rank: sorted[ceil(p*n)-1]',
    validation:'Warm browser fetch POST /api/validate through Vite proxy, including HTTP/JSON; no data scan/dry run.',
    load:'Existing project select change → matching project ID/load receipt/current native hash/all canvas node DOMs/two animation frames; includes native validation debounce and observer frames. App already warm.',
    selection:'Existing outline Inspect-node button click → exact inspector ID/two animation frames. Does not measure add/update, pointer travel or canvas pan/zoom.',
    exclusions:'One headless browser/laptop/dev build/CPU; bounded synthetic chain only; no CPU isolation, existing user services remain running. Animation-frame observers add readiness latency. No independent cold process repetitions, representative trained architectures, user productivity, heap/RSS, GPU/frame-time or universal platform certification.'},
  runtimeErrors:[],apiErrors:[],consoleWarnings:[],measurements:[],
  environment:{revision:process.env.VOID_BENCHMARK_REVISION,nativeDependencies:JSON.parse(process.env.VOID_BENCHMARK_NATIVE_ENV),node:process.version,platform:os.platform(),release:os.release(),arch:os.arch(),cpu:os.cpus()[0]?.model,logicalCpus:os.cpus().length,totalMemoryBytes:os.totalmem(),viewport:{width:1600,height:1100},editor:'Vite dev server; React development mode'}};
const browser=await puppeteer.launch({executablePath:process.env.VOID_SMOKE_CHROME,headless:true,args:process.platform==='linux'?['--no-sandbox']:[],defaultViewport:evidence.environment.viewport});
process.once('SIGTERM',()=>void browser.close());process.once('SIGINT',()=>void browser.close());
const page=await browser.newPage();page.setDefaultTimeout(60000);
// Only this runner's disposable teaching draft is ever open.
page.on('dialog',d=>d.accept());
page.on('pageerror',e=>evidence.runtimeErrors.push(e.message));
page.on('console',m=>{if(['warning','error'].includes(m.type()))evidence.consoleWarnings.push({type:m.type(),text:m.text()});});
page.on('response',r=>{if(new URL(r.url()).pathname.startsWith('/api/')&&r.status()>=400)evidence.apiErrors.push({status:r.status(),url:r.url()});});
const stats=values=>{const a=[...values].sort((a,b)=>a-b);return{samples:values,unit:'ms',n:a.length,min:a[0],median:a[Math.ceil(.5*a.length)-1],p95:a[Math.ceil(.95*a.length)-1],max:a.at(-1)};};
try{
  await page.goto(process.env.VOID_SMOKE_URL);
  await page.waitForFunction(()=>document.querySelectorAll('.react-flow__node').length===10);
  evidence.environment.browser=await browser.version();
  evidence.environment.native=await page.evaluate(async()=>{const r=await fetch('/api/production');if(!r.ok)throw Error('Native environment '+r.status);return r.json();});
  for(const size of evidence.methodology.sizes){
    const ids=Array.from({length:size},(_,i)=>`node_${String(i).padStart(4,'0')}`);
    const graph={schemaVersion:'1.0.0',graphKind:'model',backend:'pytorch',nodes:ids.map((id,i)=>({id,type:i?'pytorch.nn.relu':'core.tensor_input',version:'1.0.0',config:i?{}:{shape:['N',4],dtype:'float32'},stateRef:null})),
      edges:ids.slice(1).map((id,i)=>({id:`edge_${i}`,kind:'tensor',from:{node:ids[i],port:i?'output':'value'},to:{node:id,port:'input'}}))};
    const ui={schemaVersion:'1.0.0',synthetic:true,description:'SYNTHETIC metadata-only editor timing chain; no training or model quality claim.',positions:Object.fromEntries(ids.map((id,i)=>[id,{x:(i%25)*220,y:Math.floor(i/25)*130}]))};
    const projectIds=[`SYNTHETIC_perf_${size}_a`,`SYNTHETIC_perf_${size}_b`];
    const seeded=await page.evaluate(async({graph,ui,projectIds})=>{
      const seeds=[];for(const id of projectIds){const r=await fetch('/api/projects/'+id,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph,ui})});if(!r.ok)throw Error('Seed '+r.status);const s=await fetch('/api/projects/'+id);if(!s.ok)throw Error('Read seed '+s.status);seeds.push(await s.json());}
      return seeds;
    },{graph,ui,projectIds});
    const native=await page.evaluate(async graph=>{const r=await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})});if(!r.ok)throw Error('Validate '+r.status);return r.json();},graph);
    assert(native.ok,JSON.stringify(native.diagnostics));assert.equal(native.graphHash,seeded[0].graphHash);
    await page.reload();await page.waitForFunction(ids=>ids.every(id=>[...document.querySelector('select[aria-label="open project"]').options].some(o=>o.value==='project:'+id)),{},projectIds);
    const loads=[];
    for(let i=0;i<11;i++){
      const id=projectIds[i%2];
      const elapsed=await page.evaluate(async({id,hash,size})=>{
        const old=document.querySelector('.toast[role="status"]');if(old)old.click();
        const select=document.querySelector('select[aria-label="open project"]');const start=performance.now();select.value='project:'+id;select.dispatchEvent(new Event('change',{bubbles:true}));
        return await new Promise((resolve,reject)=>{let readyFrames=0;const check=()=>{
          const outline=document.querySelector('.graph-outline');if(outline&&!outline.open)outline.open=true;
          const ready=document.querySelector('input[aria-label="project id"]')?.value===id&&document.querySelector('.toast[role="status"]')?.textContent.includes(`Loaded project '${id}'`)&&outline?.querySelector('.provenance')?.textContent.includes(hash)&&document.querySelectorAll('.react-flow__node').length===size;
          readyFrames=ready?readyFrames+1:0;if(readyFrames>=2)return resolve(performance.now()-start);
          if(performance.now()-start>60000)return reject(Error('Actual load readiness timeout '+id));requestAnimationFrame(check);
        };requestAnimationFrame(check);});
      },{id,hash:native.graphHash,size});
      if(i)loads.push(elapsed);
    }
    const selection=[];
    for(let i=0;i<21;i++){
      const id=ids[i%2?49:0];
      const elapsed=await page.evaluate(async id=>{
        const button=document.querySelector(`[aria-label="inspect outline ${id}"]`);if(!button)throw Error('Missing native outline node '+id);const start=performance.now();button.click();
        return await new Promise((resolve,reject)=>{let frames=0;const check=()=>{const ready=document.querySelector('input[aria-label="node id"]')?.value===id&&button.getAttribute('aria-pressed')==='true';frames=ready?frames+1:0;
          if(frames>=2)return resolve(performance.now()-start);if(performance.now()-start>60000)return reject(Error('Selection timeout '+id));requestAnimationFrame(check);};requestAnimationFrame(check);});
      },id);
      if(i)selection.push(elapsed);
    }
    const validations=await page.evaluate(async graph=>{
      const samples=[];for(let i=0;i<21;i++){const start=performance.now();const r=await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({graph})});const result=await r.json();if(!r.ok||!result.ok)throw Error('Native validation failed');if(i)samples.push(performance.now()-start);}return samples;
    },graph);
    const after=await page.evaluate(async projectIds=>Promise.all(projectIds.map(async id=>{const r=await fetch('/api/projects/'+id);if(!r.ok)throw Error('Read unchanged '+r.status);return r.json();})),projectIds);
    for(let i=0;i<2;i++)assert.deepEqual(after[i],seeded[i]);
    assert(await page.$eval('button[aria-label="undo draft edit"]',e=>e.disabled));
    const measurement={nodes:size,edges:size-1,graphHash:native.graphHash,nativeOk:native.ok,canvasNodes:await page.$$eval('.react-flow__node',a=>a.length),outlineRows:await page.$$eval('.outline-table tbody tr',a=>a.length),validation:stats(validations),projectLoad:stats(loads),outlineSelection:stats(selection)};
    evidence.measurements.push(measurement);
    fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));
    await page.screenshot({path:path.join(output,`actual-${size}-nodes.png`),fullPage:true});
    console.log(JSON.stringify({nodes:size,validationP95Ms:measurement.validation.p95,loadP95Ms:measurement.projectLoad.p95,selectionP95Ms:measurement.outlineSelection.p95}));
  }
  assert.deepEqual(evidence.runtimeErrors,[]);assert.deepEqual(evidence.apiErrors,[]);assert.deepEqual(evidence.consoleWarnings,[]);
  evidence.status='passed';
}catch(error){evidence.error=String(error);await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});process.exitCode=1;}
finally{fs.writeFileSync(path.join(output,'evidence.json'),JSON.stringify(evidence,null,2));await browser.close();}
