const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require('node:path').join(__dirname, '../web/dev.html'), 'utf8');
const code = html.split('// ---- Repaired graph identities ----')[1].split('// ---- End repaired graph identities ----')[0];
const manifest = {version:1,graph_repaired:true,paths_ids_file:'ids-hash.bin'};
const paths = {version:3,authors:4,files:{ids:'ids-hash.bin'}};
const records = {
 'M123': {id:'M123',display_name:'Michael Boyle',orcid:null,work_ids:['W1'],identity_status:'partitioned',institution:'St Andrews',papers:{A8:{title:'Political paper'}}},
 'M124': {id:'M124',display_name:'Michael Boyle',orcid:'0000-0002-7287-1353',work_ids:['W2'],identity_status:'partitioned',institution:'Hawaii'},
 'M125': {id:'M125',identity_status:'quarantined',work_ids:['W3']},
 'A8': {id:'A8',status:'retained',work_ids:['W2'],orcid:'0000-0002-7287-1353'}
};
function harness(overrides={}) {
 const requests=[];
 const c=vm.createContext({DATA_BASE:'',DataView,stripId:id=>String(id).split('/').pop(),idBucketNum:id=>Number(id.slice(1))%1000,
  pathsMeta:async()=>paths, evidenceOrcid:x=>x,
  fetchJsonOr404:async url=>{requests.push(url);
   if(url.endsWith('/manifest.json'))return manifest;
   if(url.includes('/profiles/'))return records[url.split('/').pop().replace('.json','')]||null;
   if(url.includes('/aliases/'))return {A7:['M123','M124','M125'],A9:['M123']};
   if(url.endsWith('/flags.json'))return {paths_ids_file:'ids-hash.bin',entries:[{id:'M125',n:3,kind:'quarantined'}]};
   if(url.includes('/institutions/'))return Object.values(records).filter(p=>p.id[0]==='M');
   throw new Error('Unexpected URL '+url);
  },...overrides});
 vm.runInContext(code,c);return {c,requests};
}
const plain=x=>JSON.parse(JSON.stringify(x));
test('tagged binary IDs preserve A/M namespaces, including equal numeric suffixes',()=>{
 const {c}=harness();const b=new ArrayBuffer(16),v=new DataView(b);v.setBigUint64(0,123n,true);v.setBigUint64(8,(1n<<63n)|123n,true);
 assert.equal(c.decodeMapIdentity(v,0),'A123');assert.equal(c.decodeMapIdentity(v,8),'M123');
});
test('a DOI selects its publication partition, while ORCID selects only deposited support',async()=>{
 const {c}=harness();const r={candidates:[{id:'A7',name:'Michael Boyle'}],paper:{id:'W1'},message:'Found.'};
 let out=await c.expandRepairedCandidates(r,'doi','10.1/paper');assert.deepEqual(plain(out.candidates.map(x=>x.id)),['M123']);
 out=await c.expandRepairedCandidates(r,'orcid','0000-0002-7287-1353');assert.deepEqual(plain(out.candidates.map(x=>x.id)),['M124']);
 assert.equal(out.candidates[0].affiliation,'Hawaii');
});
test('one-to-one old links migrate, split links request an explicit choice, and M IDs stay intact',async()=>{
 const {c}=harness();assert.equal(await c.resolveMapAlias('A9'),'M123');assert.equal(await c.resolveMapAlias('M123'),'M123');
 await assert.rejects(c.resolveMapAlias('A7'),/several identities/);
});
test('repaired details and papers come only from release metadata, never a synthetic OpenAlex request',async()=>{
 const {c,requests}=harness();const p=await c.repairedAuthorDetails('M123');assert.equal(p.last_known_institutions[0].display_name,'St Andrews');
 assert.equal(p.papers.A8.title,'Political paper');await c.repairedProfile('M123');
 assert.equal(requests.filter(x=>x.includes('/profiles/')).length,1);assert.ok(requests.every(x=>x.startsWith('/data/identities/')));
 await assert.rejects(c.repairedProfile('M999'),/Couldn't load/);
});
test('quarantine flags bind to this exact path ID artifact and institution starts exclude quarantine',async()=>{
 const {c}=harness();const flags=await c.repairedGraphFlags();assert.equal(flags.byId.get('M125').kind,'quarantined');
 assert.equal(flags.byNumber.get(3).id,'M125');assert.equal((await c.repairedInstitutionCandidates('I1')).length,2);
 const bad=harness({pathsMeta:async()=>({...paths,files:{ids:'different.bin'}})}).c;
 await assert.rejects(bad.repairedGraphFlags(),/do not match/);
});
test('v3 graphs fail closed without identity metadata; original A-only releases remain compatible',async()=>{
 const missing=harness({fetchJsonOr404:async()=>null}).c;
 await assert.rejects(missing.repairedIdentityManifest(),/metadata is missing/);
 const legacy=harness({pathsMeta:async()=>({version:2}),fetchJsonOr404:async()=>null}).c;
 assert.equal(await legacy.repairedProfile('A7'),null);assert.deepEqual(plain(await legacy.repairedAliases('A7')),['A7']);
 await assert.rejects(legacy.repairedProfile('M123'),/unavailable/);
});
test('a profile for another identity is rejected and failed lookups can retry',async()=>{
 let wrong=true;
 const {c}=harness({fetchJsonOr404:async url=>url.endsWith('/manifest.json')?manifest:wrong?{id:'M999'}:records.M123});
 await assert.rejects(c.repairedProfile('M123'),/does not match/);wrong=false;assert.equal((await c.repairedProfile('M123')).id,'M123');
});
test('the flags loader propagates a missing repaired manifest instead of swallowing it as legacy',async()=>{
 const {c}=harness({fetchJsonOr404:async()=>null});
 c.location={href:'http://localhost/index.html'};c.URL=URL;
 const flags=html.split('// ---- R46: data flags ----')[1].split('// ---- End data flags ----')[0];
 vm.runInContext(flags,c);
 await assert.rejects(c.dataFlags(),/metadata is missing/);
 await assert.rejects(c.dataFlags(),/metadata is missing/);
});
test('Connect refuses a quarantined endpoint and blocks quarantine during normal graph search',async()=>{
 let searches=0,avoided,status;
 const {c}=harness();
 Object.assign(c,{pathGen:1,connectA:{id:'M123',name:'Political Boyle'},connectB:{id:'M125',name:'Unresolved'},
  stripId:x=>x,authorPathNumber:async id=>id==='M125'?3:0,loadPathIndex:()=>{},performance:{now:()=>0},
  dataFlags:async()=>({byNumber:new Map([[3,{kind:'quarantined'}]])}),setPathStatus:x=>{status=x},
  browserShortestPath:async(_m,_a,_b,_p,options)=>{searches++;avoided=options.avoidNodes;return null},
  PATH_MAX_HOPS:14});
 const connect=html.slice(html.indexOf('async function connectInBrowser('),html.indexOf('connectGo.addEventListener("click"'));
 vm.runInContext(connect,c);
 await c.connectInBrowser(paths,'M123','M125',1);assert.equal(searches,0);assert.match(status,/unresolved/);
 c.authorPathNumber=async id=>id==='A8'?1:0;c.connectB={id:'A8',name:'Coauthor'};
 await c.connectInBrowser(paths,'M123','A8',1);assert.equal(searches,1);assert.equal(avoided.get(3).kind,'quarantined');
});
test('ORCID lookup expands real release aliases before checking membership, preserving a small repaired identity',async()=>{
 const {c}=harness();
 const localFetch=c.fetchJsonOr404;
 Object.assign(c,{URL,URLSearchParams,normalize:x=>x.toLowerCase(),
  pooled:async tasks=>Promise.all(tasks.map(task=>task())),
  fetchIdBucket:async()=>new Map([['M124',[.4,.5,1]]]),
  fetchJsonOr404:async url=>url.startsWith('https://api.openalex.org/authors?')
   ? {results:[{id:'A7',display_name:'Michael Boyle',orcid:'0000-0002-7287-1353',works_count:2}]}
   :localFetch(url)});
 vm.runInContext(html.split('// ---- Identifier lookup helpers ----')[1].split('// ---- End identifier lookup helpers ----')[0],c);
 const result=await c.fetchIdentifierCandidates('0000-0002-7287-1353','orcid');
 assert.deepEqual(plain(result.candidates.map(x=>x.id)),['M124']);
 assert.match(result.message,/1 mapped profile/);
});


test('a retained source keeps its URL and its DOI results use only clean memberships', async () => {
 const {c}=harness();
 assert.equal(await c.resolveMapAlias('A8'),'A8');
 const result={candidates:[{id:'A8',name:'Retained researcher'}],paper:{id:'W3'},message:'Found.'};
 assert.equal((await c.expandRepairedCandidates(result,'doi','10.1/ambiguous')).candidates.length,0);
 result.paper.id='W2';
 assert.deepEqual(plain((await c.expandRepairedCandidates(result,'doi','10.1/clean')).candidates.map(x=>x.id)),['A8']);
});

test('graph manifests select their own immutable generation for every identity request', async () => {
 const generation='a'.repeat(64);
 const {c,requests}=harness({pathsMeta:async()=>({...paths,identity_generation:generation})});
 await c.repairedGraphFlags();
 await c.repairedAliases('A7');
 await c.repairedProfile('M123');
 await c.repairedInstitutionCandidates('I1');
 assert.equal(requests.length,5);
 assert.ok(requests.every(url=>url.startsWith('/data/identities/'+generation+'/')));
 const missing=harness({pathsMeta:async()=>({version:2,identity_generation:generation}),fetchJsonOr404:async()=>null}).c;
 await assert.rejects(missing.repairedIdentityManifest(),/metadata is missing/);
});
