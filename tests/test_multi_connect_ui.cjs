const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(require('node:path').join(__dirname, '../web/dev.html'), 'utf8');
function section(start,end) { const a=html.indexOf(start),b=html.indexOf(end,a); assert.ok(a>=0&&b>a, start); return html.slice(a,b); }
function selectionHarness() {
 const c=vm.createContext({MULTI_CONNECT_LIMIT:8,stripId:id=>String(id).replace(/^https?:\/\/openalex\.org\//,'')});
 vm.runInContext(section('// ---- Multi-connect selection helpers ----','// ---- End multi-connect selection helpers ----'),c); return c;
}
const researcher=id=>({id,name:id});
test('all visible selections are required and duplicates normalize complete OpenAlex URLs',()=>{
 const c=selectionHarness();
 assert.match(c.validateConnectSelections([researcher('A1'),null]).error,/select/i);
 assert.match(c.validateConnectSelections([researcher('A1'),researcher('A2'),null]).error,/3/);
 assert.match(c.validateConnectSelections([researcher('A1'),researcher('https://openalex.org/A1')]).error,/already/i);
 assert.equal(c.validateConnectSelections([researcher('A1'),researcher('M1'),researcher('I1')]).error,null);
 assert.match(c.validateConnectSelections(Array.from({length:9},(_,i)=>researcher('A'+i))).error,/8/);
});
test('selection validation rejects non-author/non-institution IDs without changing caller objects',()=>{
 const c=selectionHarness();
 assert.match(c.validateConnectSelections([researcher('W1'),researcher('A2')]).error,/researcher|institution/);
 const selected=researcher('https://openalex.org/A1');
 const result=c.validateConnectSelections([selected,researcher('I2')]);
 assert.equal(result.endpoints[0].id,'A1');assert.equal(selected.id,'https://openalex.org/A1');
});
function parse(hash) {
 const c=vm.createContext({location:{hash},stripId:id=>String(id).replace(/^https?:\/\/openalex\.org\//,'')});
 vm.runInContext(section('function parseHash()','function writeHash()'),c);
 return JSON.parse(JSON.stringify(c.parseHash()));
}
test('group links accept two to eight mixed endpoints and reject malformed or duplicate groups',()=>{
 assert.deepEqual(parse('#1/0/0&g=A1,I2,M3').g,['A1','I2','M3']);
 for(const ids of ['A1','A1,A1','A1,W2','A1,garbage','A1,I2,','A1,A2,A3,A4,A5,A6,A7,A8,A9']) assert.equal(parse('#1/0/0&g='+ids).g,undefined);
 assert.deepEqual(parse('#1/0/0&p=I1,A2').p,['I1','A2']);
});

function restoreHarness(ids, overrides={}) {
 const selected=[],goSnapshots=[],statuses=[];
 let c;
 const extra=[];
 function lookup(slot) { return {pause(){},select(entry){
   selected.push([slot,entry]);
   // Production lookup.select invokes its normal callback, which cancels
   // pending work and clears paths. Restoring all rows must remain atomic.
   c.pendingG=null;c.pendingP=null;c.pathGen++;
 }}; }
 const existing={removed:false,lookup:{pause(){existing.paused=true;}},element:{remove(){existing.detached=true;}}};
 extra.push(existing);
 const aKind={value:'researcher'},bKind={value:'researcher'};
 c=vm.createContext({pendingG:ids,pendingA:'A99',pendingP:['A1','A2'],pathGen:3,activeA:null,activePath:null,
  normalize:name=>name.toLowerCase(),fetchConnectStartName:async id=>'Name '+id,
  connectFromKind:aKind,connectToKind:bKind,connectALookup:lookup('a'),connectBLookup:lookup('b'),
  extraConnectRows:extra,connectPanel:{classList:{add(){}}},connectToggle:{classList:{add(){}}},
  writeHash(){c.hashWrites=(c.hashWrites||0)+1;},setPathStatus:note=>statuses.push(note),
  addConnectRow(){c.pendingG=null;c.pathGen++;const row={kind:{value:'researcher'},lookup:lookup('extra'+extra.length)};extra.push(row);return row;},
  syncConnectStartUI(){},syncConnectGroupUI(){},connectGo:{click(){goSnapshots.push({selected:selected.map(x=>[x[0],x[1][2]]),kinds:[aKind.value,bKind.value,...extra.map(row=>row.kind.value)]});}},
  MULTI_CONNECT_LIMIT:8,stripId:id=>String(id).replace(/^https?:\/\/openalex\.org\//,''),...overrides});
 vm.runInContext(section('// ---- Multi-connect selection helpers ----','// ---- End multi-connect selection helpers ----'),c);
 vm.runInContext(section('async function restoreConnectGroup(','async function restoreFromHash('),c);
 return {c,selected,goSnapshots,statuses,existing,extra};
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));

test('editing during group name hydration discards late successful or failed responses',async()=>{
 for(const fails of [false,true]){
  const ids=['A1','I2'];let release,reject;
  const h=restoreHarness(ids,{fetchConnectStartName:id=>id==='A1'?new Promise((r,j)=>{release=r;reject=j;}):Promise.resolve('Institution')});
  const pending=h.c.restoreConnectGroup(ids);await flush();
  if(fails)h.c.pathGen++;else h.c.pendingG=null;
  if(fails)reject(new Error('Old response failed'));else release('Old name');
  await pending;assert.equal(h.selected.length,0);assert.equal(h.goSnapshots.length,0);assert.equal(h.statuses.length,0);
 }
});

test('mixed group restoration fills every control in one transaction before exactly one Go',async()=>{
 const ids=['A1','I2','A3','I4'];const h=restoreHarness(ids);
 await h.c.restoreConnectGroup(ids);
 assert.deepEqual(h.goSnapshots,[{selected:[['a','A1'],['b','I2'],['extra0','A3'],['extra1','I4']],kinds:['researcher','institution','researcher','institution']}]);
 assert.equal(h.existing.removed,true);assert.equal(h.existing.paused,true);assert.equal(h.existing.detached,true);
 assert.equal(h.c.pendingG,null);assert.equal(h.c.pendingA,null);assert.equal(h.c.pendingP,null);assert.ok(h.c.pathGen>3);assert.equal(h.statuses.length,0);
});

function highlightOwnershipHarness(state={}) {
 const hashes=[];let clearCalls=0,c;
 const element=()=>({append(){}});
 c=vm.createContext({activeG:null,pendingG:null,activeConnectGroup:null,activeA:null,pendingA:'A99',pathGen:1,
  document:{createElement:element},stripId:id=>String(id).split('/').pop(),
  markerOnDot:()=>({remove(){}}),toLngLat:(x,y)=>[x,y],overlayLabelAdd(){},
  clearSearchHighlight(){},updateSearchClearVisibility(){},drawLabels(){},highlightLegendForCoords(){},
  clearPath(){clearCalls++;c.activeG=null;c.pendingG=null;c.activeConnectGroup=null;c.pathGen++;},
  writeHash(){hashes.push({a:c.activeA,g:c.activeG,pending:c.pendingG});},...state});
 const start=html.indexOf('function setSearchHighlight(');
 const end=html.indexOf('\n}\n',start);
 assert.ok(start>=0&&end>start);
 vm.runInContext(html.slice(start,end+2),c);
 return {c,hashes,clearCalls:()=>clearCalls};
}

test('a researcher highlight cancels pending or active group ownership even before any route is drawn',()=>{
 for(const state of [{pendingG:['A1','A2']},{activeG:['A1','A2']},{activeConnectGroup:{gen:1,running:true}}]){
  const h=highlightOwnershipHarness(state);h.c.setSearchHighlight('https://openalex.org/A7','Chosen researcher',.3,.4);
  assert.equal(h.clearCalls(),1);assert.equal(h.c.pathGen,2);assert.equal(h.c.activeA,'A7');assert.equal(h.c.pendingA,null);
  assert.equal(h.c.activeG,null);assert.equal(h.c.pendingG,null);assert.equal(h.c.activeConnectGroup,null);
  assert.deepEqual(h.hashes,[{a:'A7',g:null,pending:null}]);
 }
 const ordinary=highlightOwnershipHarness();ordinary.c.setSearchHighlight('A8','Ordinary researcher',.1,.2);assert.equal(ordinary.clearCalls(),0);
});

test('discarding a group clears pending restoration as well as visible and running group ownership',()=>{
 let replaced=0;const c=vm.createContext({pendingG:['A1','I2'],activeG:['A1','I2'],activeConnectGroup:{gen:5},connectStop:{hidden:false},connectGroupResults:{hidden:false,replaceChildren(){replaced++;}}});
 vm.runInContext(section('function discardConnectGroup()','function invalidateConnectSelections()'),c);
 c.discardConnectGroup();assert.equal(c.pendingG,null);assert.equal(c.activeG,null);assert.equal(c.activeConnectGroup,null);
 assert.equal(c.connectStop.hidden,true);assert.equal(c.connectGroupResults.hidden,true);assert.equal(replaced,1);
});

test('copying an author card discards undrawn group ownership before writing the author link',async()=>{
 for(const state of [{pendingG:['A1','I2']},{activeG:['A1','I2']},{activeConnectGroup:{gen:1,running:true}}]){
  let click,cleared=0;const copies=[];let c;
  c=vm.createContext({infoCardData:{bareId:'A7',name:'Author seven',xw:.3,yw:.4},
   activePath:null,pathEventSource:null,pendingP:null,activeG:null,pendingG:null,activeConnectGroup:null,activeA:null,
   location:{href:'https://map.example/#g=A1,I2'},navigator:{clipboard:{writeText:async text=>copies.push(text)}},
   clearPath(){cleared++;c.activeG=null;c.pendingG=null;c.activeConnectGroup=null;},clearRays(){},
   setSearchHighlight(id){assert.equal(c.activeG,null);assert.equal(c.pendingG,null);assert.equal(c.activeConnectGroup,null);c.activeA=id;c.location.href='https://map.example/#a='+id;},
   infoCardCopyBtn:{addEventListener(_event,listener){click=listener;}},infoCardCopyTimer:null,clearTimeout(){},setTimeout(){return 1;},prompt(){assert.fail('clipboard should succeed');},...state});
  vm.runInContext(section('infoCardCopyBtn.addEventListener("click", async () => {','const tip = document.getElementById("tip");'),c);
  await click();assert.equal(cleared,1);assert.deepEqual(copies,['https://map.example/#a=A7']);
 }
});

test('Escape cancels a pending group restore and stops an active search from the map',()=>{
 let handler,stops=0,clears=0;
 const closed={classList:{contains:()=>false}};
 const lookup={isOpen:()=>false};
 const c=vm.createContext({document:{activeElement:null,addEventListener:(_event,fn)=>{handler=fn;}},
  identifierDialog:{open:false},shareOverlay:closed,infoCard:closed,helpOverlay:closed,searchDropdown:closed,
  connectALookup:lookup,connectBLookup:lookup,extraConnectRows:[],connectAInput:{},connectBInput:{},
  pendingG:['A1','A2','A3'],activeConnectGroup:null,connectStop:{click(){stops++;}},
  clearPath(){clears++;},activeA:null,rayMarkers:[],map:{getSource:()=>null},isolateCommunity:null});
 const start=html.lastIndexOf('document.addEventListener("keydown", ev => {');
 vm.runInContext(html.slice(start,html.indexOf('}, true);',start)+9),c);
 const event=()=>({key:'Escape',preventDefault(){},stopPropagation(){}});
 handler(event());assert.equal(c.pendingG,null);assert.equal(clears,1);
 c.activeConnectGroup={running:true};handler(event());assert.equal(stops,1);assert.equal(clears,1);
});


test('existing map group controls focus found pairs and preserve keyboard focus as other pairs finish',()=>{
 let doc;const focuses=[];
 class Element{
  constructor(tag){this.tagName=tag;this.children=[];this.attrs={};this.listeners={};this._text='';}
  set textContent(value){this._text=String(value);this.children=[];}get textContent(){return this._text+this.children.map(child=>child.textContent).join('');}
  append(...children){this.children.push(...children);}replaceChildren(...children){this.children=children;}
  setAttribute(key,value){this.attrs[key]=String(value);}getAttribute(key){return this.attrs[key]??null;}
  addEventListener(event,fn){this.listeners[event]=fn;}contains(node){return this===node||this.children.some(child=>child.contains(node));}
  querySelector(selector){const key=/data-connect-control="([^"]+)"/.exec(selector)?.[1];const walk=node=>node.getAttribute('data-connect-control')===key?node:node.children.map(walk).find(Boolean);return walk(this)||null;}
  focus(){doc.activeElement=this;}
 }
 doc={activeElement:null,createElement:tag=>new Element(tag),createElementNS(){assert.fail('group controls use the existing map overlay');}};
 const results=[{source:0,target:1,status:'found',hops:2},{source:0,target:2,status:'not-found',note:'No path within the search limit.'}];
 const group={endpoints:[researcher('A1'),researcher('A2'),researcher('I3')],results,focused:null,running:true};
 const container=new Element('div');const c=vm.createContext({document:doc,connectGroupResults:container,activeConnectGroup:group,groupPairKey:r=>`${r.source}:${r.target}`,hopsText:n=>`${n} hops`,focusConnectPair:key=>focuses.push(key)});
 vm.runInContext(section('function refreshConnectGroupResults()','async function focusConnectPair('),c);c.refreshConnectGroupResults();
 assert.match(container.textContent,/2\/3 pairs checked/);assert.match(container.textContent,/Waiting/);
 const foundPair=container.querySelector('[data-connect-control="pair-0:1"]');assert.equal(foundPair.disabled,false);
 assert.equal(container.querySelector('[data-connect-control="pair-0:2"]').disabled,true);assert.equal(container.querySelector('[data-connect-control="pair-1:2"]').disabled,true);
 foundPair.listeners.click();container.querySelector('[data-connect-control="all"]').listeners.click();assert.deepEqual(focuses,['0:1',null]);
 foundPair.focus();results.push({source:1,target:2,status:'found',hops:1});c.refreshConnectGroupResults();
 assert.notEqual(doc.activeElement,foundPair);assert.equal(doc.activeElement,container.querySelector('[data-connect-control="pair-0:1"]'));assert.match(container.textContent,/3\/3 pairs checked/);
 const outside=new Element('input');outside.focus();c.refreshConnectGroupResults();assert.equal(doc.activeElement,outside);assert.equal(group.results,results);
});

test('group map links preserve camera and endpoint IDs while ordinary author and pair links remain compatible',()=>{
 const writes=[],ids=['A1','I2','A3'];
 const c=vm.createContext({map:{getCenter:()=>({lat:0,lng:0}),getZoom:()=>2},history:{replaceState(_a,_b,hash){writes.push(hash);}},activeA:'A9',pendingA:null,activeP:['A8','A7'],pendingP:null,activeG:ids,pendingG:null});
 vm.runInContext(section('function writeHash()','const initialHash = parseHash();'),c);
 c.writeHash();assert.equal(writes.at(-1),'#2.00/0.00000/0.00000&g=A1,I2,A3');
 c.activeG=null;c.pendingG=ids;c.writeHash();assert.equal(writes.at(-1),'#2.00/0.00000/0.00000&g=A1,I2,A3');
 c.pendingG=null;c.writeHash();assert.equal(writes.at(-1),'#2.00/0.00000/0.00000&a=A9&p=A8,A7');assert.deepEqual(ids,['A1','I2','A3']);
});
