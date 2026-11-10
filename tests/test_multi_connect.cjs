const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const vm = require('node:vm');
const html = readFileSync(join(__dirname, '../web/dev.html'), 'utf8');
const bfs = html.slice(html.indexOf('async function browserShortestPath('), html.indexOf('// Indices i whose link'));
const engine = html.slice(html.indexOf('// ---- Multi-endpoint Connect engine ----'), html.indexOf('// ---- End multi-endpoint Connect engine ----'));
const plain = x => JSON.parse(JSON.stringify(x));
function harness(overrides = {}) {
  const calls = {resolve:[], papers:[], names:[], searches:[]};
  const graph = overrides.graph || new Map([[1,[2]],[2,[1,3]],[3,[2,4]],[4,[3]]]);
  const c = vm.createContext({Map,Set,DOMException,Number,Promise,PATH_MAX_HOPS:12,PATH_MAX_LINKS:10000,
    stripId: id=>String(id).replace(/^https?:\/\/openalex\.org\//,''),
    authorPathNumber: async id=>{calls.resolve.push(id); return Number(id.slice(1));},
    resolveInstitutionStart: async (_m,e)=>{calls.resolve.push(e.id);return {nodes:e.members,names:new Map(e.memberNames || []),considered:e.members.length};},
    dataFlags: async ()=>({byNumber:new Map()}),
    eachPathList: async (_m,nodes,visit)=>{for(const n of [...nodes].sort((a,b)=>a-b)) if(visit(n,graph.get(n)||[]))return true;return false;},
    pathAuthorIds: async (_m,nodes)=>nodes.map(n=>'A'+n),
    fetchDisplayName: async id=>{calls.names.push(id);return 'Name '+id;},
    sharedPaper: async (a,b)=>{calls.papers.push([a,b]);return {title:a+' with '+b,url:'https://example.org/paper'};},
    ...overrides,
  });
  vm.runInContext(bfs+'\n'+engine,c);
  const real=c.browserShortestPath;
  c.browserShortestPath=async (...args)=>{calls.searches.push(args);return real(...args);};
  return {c,calls};
}
const endpoints = (...ids)=>ids.map(id=>({id,name:'Selected '+id}));
const meta={authors:100};

test('four endpoints resolve once and stream all six unordered pairs sequentially',async()=>{
 const {c,calls}=harness();const received=[];let pending=false;
 const results=await c.searchConnectGroup(meta,endpoints('A1','A2','A3','A4'),{onResult:async r=>{
   assert.equal(pending,false);pending=true;await Promise.resolve();received.push(r);pending=false;
 }});
 assert.deepEqual(plain(results.map(r=>[r.source,r.target,r.status,r.hops])),[[0,1,'found',1],[0,2,'found',2],[0,3,'found',3],[1,2,'found',1],[1,3,'found',2],[2,3,'found',1]]);
 assert.equal(received.length,6);assert.deepEqual(calls.resolve,['A1','A2','A3','A4']);
 assert.ok(calls.searches.every(x=>x[4].avoidSuspect===true));
});

test('both institution endpoints search complete mapped member pools and keep member names',async()=>{
 const {c,calls}=harness();const results=await c.searchConnectGroup(meta,[
  {id:'I10',name:'First university',members:[1,4],memberNames:[['A1','One'],['A4','Four']]},
  {id:'I20',name:'Second university',members:[2,3],memberNames:[['A2','Two'],['A3','Three']]},
 ]);
 assert.deepEqual(plain(calls.searches[0].slice(1,3)),[[1,4],[2,3]]);
 assert.equal(results[0].hops,1);assert.deepEqual(plain(results[0].steps.map(s=>s.name)),['One','Two']);
 assert.match(results[0].note,/mapped.*candidates/i);assert.equal(calls.names.length,0);
});

test('overlapping institution members produce a real zero-hop result',async()=>{
 const {c,calls}=harness();const r=(await c.searchConnectGroup(meta,[{id:'I1',members:[2,4],memberNames:[['A2','Shared member']]},{id:'I2',members:[2]}]))[0];
 assert.equal(r.status,'found');assert.equal(r.hops,0);assert.equal(r.steps.length,1);assert.equal(r.steps[0].name,'Shared member');assert.equal(calls.papers.length,0);
});

test('flagged intermediates and suspect edges are blocked without a permissive fallback',async()=>{
 const graph=new Map([[1,[9,~5,3]],[9,[1,2]],[5,[1,2]],[3,[1,4]],[4,[3,2]],[2,[9,5,4]]]);
 const {c}=harness({graph,dataFlags:async()=>({byNumber:new Map([[9,{kind:'merged'}]])})});
 const r=(await c.searchConnectGroup(meta,endpoints('A1','A2')))[0];
 assert.deepEqual(plain(r.steps.map(s=>s.id)),['A1','A3','A4','A2']);assert.equal(r.hops,3);
});

test('quarantine endpoints are unavailable and quarantined institution members are removed',async()=>{
 const {c,calls}=harness({dataFlags:async()=>({byNumber:new Map([[1,{kind:'quarantined'}]])})});
 const r=await c.searchConnectGroup(meta,[{id:'A1',name:'Unresolved'},{id:'I5',members:[1,4]},{id:'A3',name:'Three'}]);
 assert.deepEqual(plain(r.map(x=>x.status)),['unavailable','unavailable','found']);
 assert.equal(calls.searches.length,1);assert.deepEqual(plain(calls.searches[0][1]),[4]);
 assert.match(r[0].note,/unresolved|quarantin/i);
});

test('an endpoint resolution failure affects its pairs while other pairs complete',async()=>{
 const {c}=harness({authorPathNumber:async id=>{if(id==='A2')throw new Error('Index unavailable');return Number(id.slice(1));}});
 const r=await c.searchConnectGroup(meta,endpoints('A1','A2','A3'));
 assert.deepEqual(plain(r.map(x=>x.status)),['error','found','error']);assert.match(r[0].note,/Index unavailable/);
});

test('missing endpoints, pair errors, limits and disconnected searches have distinct states',async()=>{
 const {c}=harness({authorPathNumber:async id=>id==='A4'?null:Number(id.slice(1))});
 let n=0;c.browserShortestPath=async()=>{n++;if(n===1)throw new Error('Network problem');if(n===2)return {tooFar:true};return null;};
 const r=await c.searchConnectGroup(meta,endpoints('A1','A2','A3','A4'));
 assert.deepEqual(plain(r.map(x=>x.status)),['error','limit','unavailable','not-found','unavailable','unavailable']);
});

test('unavailable identity flags fail closed for every pair',async()=>{
 const {c,calls}=harness({dataFlags:async()=>{throw new Error('Identity metadata unavailable');}});
 const r=await c.searchConnectGroup(meta,endpoints('A1','A2','A3'));
 assert.ok(r.every(x=>x.status==='error'));assert.equal(calls.searches.length,0);
});

test('hydration failures preserve paths, cache shared edges and names, and report missing evidence',async()=>{
 const paperCalls=[],nameCalls=[];
 const graph=new Map([[1,[2]],[2,[1,3,4]],[3,[2]],[4,[2]]]);
 const {c}=harness({graph,sharedPaper:async(a,b)=>{paperCalls.push([a,b]);if(a==='A2'||b==='A2')throw new Error('No evidence');},
 fetchDisplayName:async id=>{nameCalls.push(id);throw new Error('No live name');}});
 const r=await c.searchConnectGroup(meta,endpoints('A1','A3','A4'));
 assert.ok(r.every(x=>x.status==='found'));assert.match(r[0].note,/shared-paper evidence/i);
 assert.equal(r[0].steps[1].name,'A2');assert.equal(nameCalls.filter(x=>x==='A2').length,1);
 assert.equal(paperCalls.length,3);assert.ok(r[0].steps.every(s=>s.paper===null));
});

test('stale groups abort after completed incremental results and do not publish late results',async()=>{
 const {c}=harness();let stale=false;const received=[];
 await assert.rejects(c.searchConnectGroup(meta,endpoints('A1','A2','A3'),{isStale:()=>stale,onResult:async r=>{received.push(r);stale=true;}}),{name:'AbortError'});
 assert.equal(received.length,1);
 const slow=harness({authorPathNumber:async()=>{stale=true;return 1;}});stale=false;
 await assert.rejects(slow.c.searchConnectGroup(meta,endpoints('A1','A2'),{isStale:()=>stale}),{name:'AbortError'});
});

test('limits and exact duplicate IDs reject before requests, distinct IDs are not deduplicated by names or aliases',async()=>{
 const {c,calls}=harness();
 await assert.rejects(c.searchConnectGroup(meta,endpoints('A1','https://openalex.org/A1')),/duplicate|already/i);
 await assert.rejects(c.searchConnectGroup(meta,endpoints(...Array.from({length:9},(_,i)=>'A'+i))),/8/);
 assert.equal(calls.resolve.length,0);
 const alias=harness({authorPathNumber:async()=>2});
 const r=await alias.c.searchConnectGroup(meta,[{id:'A1',name:'Same name'},{id:'A9',name:'Same name'}]);
 assert.equal(r.length,1);assert.equal(r[0].hops,0);
});

test('automatic institution representatives exclude every flagged profile while explicit warned researchers remain selectable',async()=>{
 const graph=new Map([[1,[2]],[2,[1,9]],[9,[2]]]);
 const {c,calls}=harness({graph,dataFlags:async()=>({byNumber:new Map([[9,{kind:'merged'}]])})});
 const pooledResult=await c.searchConnectGroup(meta,[{id:'I5',name:'University',members:[9,1]},...endpoints('A2')]);
 assert.deepEqual(plain(calls.searches[0][1]),[1]);assert.equal(pooledResult[0].status,'found');
 const explicit=(await c.searchConnectGroup(meta,endpoints('A9','A2')))[0];
 assert.equal(explicit.status,'found');assert.equal(explicit.steps[0].id,'A9');assert.match(explicit.note,/identity warning|identities unverified/);
 const empty=(await c.searchConnectGroup(meta,[{id:'I6',members:[9]},...endpoints('A2')]))[0];
 assert.equal(empty.status,'unavailable');assert.match(empty.note,/flags|flagged/);assert.doesNotMatch(empty.note,/quarantined|unresolved/);
});
