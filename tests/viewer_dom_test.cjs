// Lightweight DOM unit harness. This tests interaction logic, not browser layout.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(process.argv[2], 'utf8');
const payload = JSON.parse(html.match(/<script id="report-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const unreadable = {...payload.runs[0], id:payload.runs.length, report:null, error:'Metrics unavailable'};
payload.runs.push(unreadable);
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];
class Element {
  constructor(tag) {this.tagName=tag.toUpperCase();this.children=[];this.style={};this.dataset={};this.attributes={};this._text='';this.clientWidth=600;this.offsetLeft=0;}
  append(...nodes) {for(let n of nodes){if(typeof n==='string'){const t=new Element('#text');t.textContent=n;n=t;}n.parent=this;n.offsetLeft=this.children.length*85;this.children.push(n);}}
  replaceChildren(...nodes) {this.children=[];this._text='';this.append(...nodes);}
  replaceWith(node) {const i=this.parent.children.indexOf(this);this.parent.children[i]=node;node.parent=this.parent;}
  set textContent(value){this._text=String(value);this.children=[];}
  get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
  setAttribute(k,v){this.attributes[k]=String(v);if(k==='class')this.className=String(v);}
  getAttribute(k){return this.attributes[k];}
  getBoundingClientRect(){return {left:0,top:0,width:600,height:220};}
  contains(target){return this===target||this.children.some(c=>c.contains(target));}
  querySelectorAll(selector){return all(this).filter(e=>selector.startsWith('.')?(e.className||'').split(' ').includes(selector.slice(1)):selector==='[data-gallery]'?e.dataset.gallery!=null:e.tagName===selector.toUpperCase());}
  querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  showModal(){this.open=true;} close(){this.open=false;}
}
function all(root){return root.children.flatMap(child=>[child,...all(child)]);}
const document={root:new Element('body'),activeElement:null,createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),
  createTextNode:text=>{const e=new Element('#text');e.textContent=text;return e;},
  getElementById:id=>all(document.root).find(e=>e.id===id)||null,
  querySelectorAll:selector=>document.root.querySelectorAll(selector),listeners:{},addEventListener:(key,fn)=>document.listeners[key]=fn};
for(const id of ['app','home','report-data','image-dialog','close-dialog','dialog-content']){const e=new Element(id==='image-dialog'?'dialog':'div');e.id=id;document.root.append(e);}
document.getElementById('report-data').textContent=JSON.stringify(payload);
const context={document,Node:Element,URLSearchParams,history:{replaceState(){}},location:{hash:''},window:{scrollTo(){}}};
vm.runInNewContext(code,context);
const app=document.getElementById('app');
const findButton=text=>all(app).find(e=>e.tagName==='BUTTON'&&e.textContent===text);
const click=(button)=>{assert.ok(button,'expected button');button.onclick({preventDefault(){},stopPropagation(){}});};
assert.match(app.textContent,/Evolution records/);
const cards=app.querySelectorAll('.run-card');assert.equal(cards.length,payload.runs.length);
click(cards[0]);
assert.match(app.textContent,/RUN INSPECTOR/);
assert.ok(app.querySelectorAll('h3').some(e=>e.textContent==='Novelty'));
assert.ok(app.querySelectorAll('h3').some(e=>e.textContent==='Display novelty')); // previous-parent differs from display mean
assert.equal(app.querySelectorAll('.candidate').length,9);
const slider=document.getElementById('generation-slider');slider.value='0';slider.oninput();
assert.equal(document.getElementById('selection-index').textContent,'Selection 0 / 2');
assert.equal(document.getElementById('previous').disabled,true);
click(document.getElementById('next'));
assert.equal(document.getElementById('selection-index').textContent,'Selection 1 / 2');
document.listeners.keydown({key:'ArrowRight',preventDefault(){}});
assert.equal(document.getElementById('selection-index').textContent,'Selection 2 / 2');
const chart=app.querySelector('.chart');
chart.querySelector('svg').onclick({clientX:55});
assert.equal(document.getElementById('selection-index').textContent,'Selection 0 / 2');
assert.match(chart.querySelector('.chart-values').textContent,/—/); // unavailable is not zero
click(app.querySelector('.candidate'));
assert.equal(document.getElementById('image-dialog').open,true);
document.getElementById('close-dialog').onclick();
click(findButton('All generated images'));
assert.equal(document.getElementById('detail-gallery').querySelectorAll('img').length,25);
click(findButton('All display visits'));
assert.equal(document.getElementById('detail-gallery').querySelectorAll('.display-card').length,3);
click(findButton('← All runs'));
const human=payload.runs.find(r=>r.report?.selection_strategy.selection_strategy==='human'&&r.report.generations.length);
click(app.querySelectorAll('.run-card')[human.id]);
assert.match(app.textContent,/saved selection differs from the last chronological choice/);
click(findButton('Final ancestry'));
assert.equal(document.getElementById('detail-gallery').querySelectorAll('img').length,human.report.final_ancestry.length);
click(findButton('All display visits'));
assert.equal(document.getElementById('detail-gallery').querySelectorAll('.display-card').length,human.report.display_history.length);
click(findButton('← All runs'));
const empty=payload.runs.find(r=>r.report&&r.report.generations.length===0);
click(app.querySelectorAll('.run-card')[empty.id]);
assert.match(app.textContent,/No explicit selections/);
click(findButton('All generated images'));
assert.equal(document.getElementById('detail-gallery').querySelectorAll('img').length,9);
click(findButton('← All runs'));
click(app.querySelectorAll('.run-card')[unreadable.id]);
assert.match(app.textContent,/Metric records are unavailable/);
assert.equal(document.getElementById('detail-gallery').querySelectorAll('img').length,unreadable.images.length);
console.log('Viewer interaction unit checks passed: charts, slider, keyboard, candidates, modal, complete image galleries, human ancestry, and empty sessions.');
