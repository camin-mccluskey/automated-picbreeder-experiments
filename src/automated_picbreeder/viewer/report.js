/* Offline inspection only: no fetches, external scripts or changes to experiments. */
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('report-data').textContent);
  const app = document.getElementById('app');
  const dialog = document.getElementById('image-dialog');
  const colors = ['#35604d', '#b3623c', '#6586a0', '#9472a0', '#a18a36'];
  const imageIndexes = new Map(data.runs.map(run => [run.id, new Map(run.images.map(item => [item.path, item]))]));
  const state = {run: null, index: 0, gallery: 'timeline', page: 0, charts: []};
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = value => value == null ? '—' : typeof value === 'boolean' ? String(value) :
    !finite(value) ? String(value) : Number.isInteger(value) ? value.toLocaleString() :
      Math.abs(value) > 0 && Math.abs(value) < .0001 ? value.toExponential(2) : Number(value.toPrecision(4)).toString();
  const label = key => key.replaceAll('_', ' ').replace(/\b(mse|rmse|mae|rgb)\b/g, word => word.toUpperCase());
  const strategy = run => run.report?.selection_strategy?.selection_strategy || 'Unrecorded strategy';
  const rows = () => state.run?.report?.generations || [];
  const node = (tag, text, cls) => {
    const element = document.createElement(tag);
    if (text != null) element.textContent = text;
    if (cls) element.className = cls;
    return element;
  };
  const button = (text, action, cls) => {const b = node('button', text, cls); b.type = 'button'; b.onclick = action; return b;};
  const svg = (tag, attrs = {}, text) => {
    const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    if (text != null) element.textContent = text;
    return element;
  };
  function image(run, path, alt, eager = false) {
    const record = imageIndexes.get(run.id)?.get(path);
    if (!record) return node('div', 'No saved selection', 'placeholder');
    const img = node('img'); img.src = record.url; img.alt = alt; img.loading = eager ? 'eager' : 'lazy';
    img.onerror = () => img.replaceWith(node('div', 'Image file unavailable', 'placeholder'));
    return img;
  }
  function zoom(run, item, heading) {
    const content = document.getElementById('dialog-content'); content.replaceChildren();
    content.append(image(run, item.image || item.path, heading, true));
    const detail = node('div'); detail.append(node('h2', heading), node('pre', JSON.stringify(item, null, 2)));
    content.append(detail); if (!dialog.open) dialog.showModal();
  }
  document.getElementById('close-dialog').onclick = () => dialog.close();
  dialog.onclick = event => {if (event.target === dialog) dialog.close();};
  document.getElementById('home').onclick = event => {event.preventDefault(); overview();};
  function statBar(items) {
    const bar = node('div', null, 'stats');
    items.forEach(([name, value]) => {const part = node('div', null, 'stat'); part.append(node('strong', fmt(value)), node('span', name)); bar.append(part);});
    return bar;
  }
  function titleBlock(title, subtitle, eyebrow) {
    const heading = node('div', null, 'page-heading'); const copy = node('div');
    copy.append(node('div', eyebrow, 'eyebrow'), node('h1', title), node('p', subtitle)); heading.append(copy); return heading;
  }
  function selectControl(items, value, onChange, ariaLabel) {
    const select = node('select'); select.setAttribute('aria-label', ariaLabel);
    items.forEach(([key, name]) => {const option = node('option', name); option.value = key; select.append(option);});
    select.value = value; select.onchange = () => onChange(select.value); return select;
  }
  function table(headers, values) {
    const wrap = node('div', null, 'table-wrap'), t = node('table'), head = node('tr');
    headers.forEach(h => head.append(node('th', h))); const thead = node('thead'); thead.append(head); t.append(thead);
    const body = node('tbody'); values.forEach(row => {const tr = node('tr'); row.forEach((v, i) => {
      const td = node('td', null, i ? 'num' : ''); td.append(v instanceof Node ? v : document.createTextNode(fmt(v))); tr.append(td);
    }); body.append(tr);}); t.append(body); wrap.append(t); return wrap;
  }
  function basePlot(series, length, {band = [], overview = false} = {}) {
    const W = 600, H = 220, left = 55, right = 15, top = 16, bottom = 35;
    const all = series.flatMap(line => line.values.filter(finite)).concat(band.flatMap(b => [b?.min, b?.max]).filter(finite));
    let low = all.reduce((a,b)=>Math.min(a,b),0), high = all.reduce((a,b)=>Math.max(a,b),0);
    if (low === high) high = low + 1;
    const x = i => left + i / Math.max(1, length - 1) * (W-left-right);
    const y = value => H-bottom - (value-low)/(high-low)*(H-top-bottom);
    const root = svg('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Metric by chronological selection index'});
    for (let i = 0; i <= 3; i++) {
      const value = low + (high-low)*i/3;
      root.append(svg('line', {x1:left, x2:W-right, y1:y(value), y2:y(value), class:'gridline'}));
      root.append(svg('text', {x:left-8, y:y(value)+3, 'text-anchor':'end', class:'axis'}, fmt(value)));
    }
    [...new Set([0, Math.floor((length-1)/2), Math.max(0,length-1)])].forEach(i => root.append(svg('text', {x:x(i), y:H-15, 'text-anchor':'middle', class:'axis'}, String(i))));
    const path = values => {let d = '', gap = true; values.forEach((v, i) => {if (!finite(v)) {gap = true; return;} d += `${gap?'M':'L'}${x(i)},${y(v)} `; gap=false;}); return d;};
    // Split quartile bands at missing measurements, rather than bridging gaps.
    let segment = [];
    const flush = () => {
      if (segment.length) root.append(svg('path', {d:segment.map((b,i) => `${i?'L':'M'}${x(b.i)},${y(b.max)}`).join(' ') + ' ' + segment.slice().reverse().map(b => `L${x(b.i)},${y(b.min)}`).join(' ') + ' Z', fill:'#dce6cc'}));
      segment=[];
    };
    band.forEach((b,i) => {if (b && finite(b.min) && finite(b.max)) segment.push({...b,i}); else flush();}); flush();
    for (const line of series) {
      const g = svg('g');
      const p = svg('path', {d:path(line.values), fill:'none', stroke:line.color, 'stroke-width':line.faint?1.4:2, opacity:line.faint?.4:1});
      g.append(p);
      if (line.onClick) {
        const hit = svg('path', {d:path(line.values), fill:'none', stroke:'transparent', 'stroke-width':12, 'pointer-events':'stroke', cursor:'pointer'});
        hit.append(svg('title', {}, line.name));
        hit.onclick = event => {event.stopPropagation(); line.onClick(indexAt(event));};
        hit.onmouseenter = () => {p.setAttribute('opacity','1'); if(line.onHover) line.onHover();};
        hit.onmouseleave = () => p.setAttribute('opacity',line.faint?'.4':'1');
        g.append(hit);
      }
      // Single-point runs remain visible. Long trajectories do not need a dot per point.
      const measured = line.values.map((v,i) => ({v,i})).filter(p => finite(p.v));
      if (measured.length <= 16) measured.forEach(({v,i}) => {
        const point = svg('circle',{cx:x(i),cy:y(v),r:3,fill:line.color});
        point.append(svg('title',{},`${line.name} · ${i}: ${fmt(v)}`));
        if(line.onClick) point.onclick = event => {event.stopPropagation();line.onClick(i);};
        g.append(point);
      });
      root.append(g);
    }
    function indexAt(event) {const rect = root.getBoundingClientRect(); return Math.max(0,Math.min(length-1,Math.round(((event.clientX-rect.left)/rect.width*W-left)/(W-left-right)*Math.max(1,length-1))));}
    const cursor = svg('line',{x1:x(state.index),x2:x(state.index),y1:top,y2:H-bottom,class:'cursor'});
    if (!overview) {root.append(cursor); root.style.cursor='crosshair'; root.onclick=event=>setSelection(indexAt(event));}
    if (!all.length) root.append(svg('text',{x:W/2,y:H/2,'text-anchor':'middle',class:'axis'},'No measurements available'));
    return {root, cursor, x};
  }
  function overview() {
    state.run=null; app.replaceChildren(); history.replaceState(null,'','#');
    app.append(titleBlock('Evolution records', data.title, 'AUTOMATED + HUMAN EXPLORATION'));
    app.append(statBar([['Runs',data.runs.length],['Complete',data.runs.filter(r=>r.status==='complete').length],['Human sessions',data.runs.filter(r=>strategy(r)==='human').length],['Failed / interrupted',data.runs.filter(r=>['failed','interrupted'].includes(r.status)).length]]));
    const toolbar=node('div',null,'toolbar'), gallery=node('div',null,'run-gallery');
    const strategies=[...new Set(data.runs.map(strategy))];
    const filter=selectControl([['all','All strategies'],...strategies.map(s=>[s,s])],'all',drawGallery,'Filter runs by strategy'); toolbar.append(filter,node('span','Every final selection, including partial runs.','muted')); app.append(toolbar,gallery);
    function drawGallery() {
      gallery.replaceChildren();
      data.runs.filter(r=>filter.value==='all'||strategy(r)===filter.value).forEach(run=>{
        const card=button('',()=>openRun(run.id),'run-card'), wrap=node('div',null,'image-wrap');
        wrap.append(image(run,run.report?.final_image,`Final image, ${strategy(run)}, seed ${run.seed}`));
        wrap.append(node('span',run.status,`badge ${run.status}`)); card.append(wrap,node('h3',`${strategy(run)} · seed ${fmt(run.seed)}`),node('p',run.label));
        if(run.report) card.append(node('p',`${run.report.summary.decisions} selections · ${run.images.length} generated`));
        gallery.append(card);
      });
      if(!gallery.children.length) gallery.append(node('p','No runs in this collection.','empty'));
    } drawGallery();
    const groups=data.groups.filter(g=>g.aggregate && g.aggregate.completed_runs);
    if(groups.length) {
      app.append(node('h2','Across runs'));
      const controls=node('div',null,'toolbar'), chartHost=node('div',null,'overview-chart');
      let selectedGroup=groups[0];
      const groupSelect=selectControl(groups.map((g,i)=>[String(i),g.label]),'0',value=>{selectedGroup=groups[Number(value)];fillMetrics();},'Batch to compare');
      const metricSelect=selectControl([], '', drawOverviewChart,'Metric across runs');
      const show=node('input');show.type='checkbox';show.checked=true;show.onchange=drawOverviewChart;
      const showLabel=node('label','Individual runs ');showLabel.append(show); controls.append(groupSelect,metricSelect,showLabel); app.append(controls,chartHost);
      function fillMetrics() {
        metricSelect.replaceChildren();
        Object.keys(selectedGroup.aggregate.run_metrics).forEach(key=>{const option=node('option',label(key));option.value=key;metricSelect.append(option);});
        metricSelect.value='pixel_mse_nearest_earlier';drawOverviewChart();
      }
      function drawOverviewChart() {
        chartHost.replaceChildren(); const key=metricSelect.value, agg=selectedGroup.aggregate.generations;
        const card=node('div',null,'chart'), heading=node('div',null,'chart-header'); heading.append(node('h3',label(key)),node('small','SELECTION INDEX')); card.append(heading);
        const hint=node('p','Median and middle 50% of completed runs. Hover a run to identify it; click to inspect.','chart-note');
        const series=[];
        if(show.checked) selectedGroup.run_ids.map(id=>data.runs[id]).filter(r=>r.status==='complete'&&r.report).forEach((r,i)=>series.push({
          name:`Seed ${r.seed} · ${r.label}`,values:r.report.generations.map(row=>row.metrics[key]),color:colors[i%colors.length],faint:true,
          onClick:index=>openRun(r.id,index),onHover:()=>hint.textContent=`Seed ${r.seed} · ${r.label} — click its line to inspect.`}));
        series.push({name:'Median',values:agg.map(g=>g.metrics[key]?.median),color:'#263f30'});
        const plot=basePlot(series,agg.length,{overview:true,band:agg.map(g=>({min:g.metrics[key]?.q25,max:g.metrics[key]?.q75}))});
        card.append(plot.root,hint); const counts=agg.map(g=>g.metrics[key]?.count||0);
        card.append(node('p',`Contributing runs: ${Math.min(...counts)}–${Math.max(...counts)}. Missing values and failed runs excluded; runs carry equal weight. Quartiles are not confidence intervals.`,'chart-note')); chartHost.append(card);
      } fillMetrics();
    }
    app.append(node('h2','Run ledger'));
    app.append(table(['Run','Status','Selections','Presentations','Genomes','Run seconds','Human seconds'],data.runs.map(run=>[
      button(`${strategy(run)} / ${fmt(run.seed)}`,()=>openRun(run.id)),run.status,run.report?.summary.decisions,
      run.report?.summary.candidate_presentations,run.images.length,run.report?.summary.run_seconds,run.report?.summary.interaction_seconds])));
    app.append(node('p','Pixel change and classifier confidence are diagnostics, not ratings of interestingness. Human interaction time includes deliberation and idle time. Compare configurations and exploration budgets before interpreting differences.','footnote'));
  }
  const groups = [
    ['Visual change',[['pixel_mse_previous','Previous selection'],['pixel_mse_nearest_earlier','Nearest earlier selection']], 'Full-resolution RGB MSE. A visually different image is not necessarily meaningful novelty.'],
    ['Novelty',[['novelty_selected','Selected'],['novelty_grid_mean','Grid mean']], 'Strategy measurement against the previous grid mean; first grid unavailable.'],
    ['Display novelty',[['display_novelty_selected','Selected'],['display_novelty_grid_mean','Grid mean']], 'Shared diagnostic against the previous display visit. Includes Back and reset visits.'],
    ['ImageNet confidence',[['imagenet_selected_confidence','Selected maximum'],['imagenet_grid_mean_confidence','Mean of candidate maxima']], 'Frozen classifier measurements used during selection.'],
    ['Post-run ImageNet',[['posthoc_imagenet_selected_confidence','Selected maximum'],['posthoc_imagenet_grid_mean_confidence','Mean of candidate maxima']], 'Separate post-run evaluation; these scores did not determine the recorded choices.'],
    ['Weighted value components',[['value_novelty_contribution','Novelty'],['value_quality_contribution','Quality'],['value_offspring_contribution','Offspring']], 'Actual weighted contributions to the selected score, including warm-up weights.'],
    ['Prediction error',[['prediction_mse_selected','Selected image'],['prediction_mse_grid_mean','Grid mean']], 'Masked prediction MSE measured before the current observer update.'],
    ['Offspring value',[['offspring_forecast_selected','Forecast'],['offspring_target','Observed children']], 'Indexed by parent selection. Outcomes arrive at the next generation; final target is unobserved.'],
    ['Forecast accuracy',[['offspring_cumulative_rmse','Predictor'],['offspring_running_mean_cumulative_rmse','Running-mean baseline'],['offspring_current_value_cumulative_rmse','Current-value baseline']], 'Cumulative RMSE on observed eligible selected-parent transitions only.'],
    ['Exploratory choices',[['exploratory_choice','Random-choice event'],['exploratory_choice_frequency','Cumulative frequency']], 'A random choice still counts when it happens to select the greedy winner.'],
    ['Time per selection',[['rendering_seconds','Rendering'],['selection_seconds','Strategy'],['saving_seconds','Saving']], 'Host wall time. Strategy time includes inference and training; nested timing metrics are not additive.'],
  ];
  function makeChart(title, keys, note) {
    const available=keys.filter(([key])=>rows().some(row=>Object.hasOwn(row.metrics,key)));
    if(!available.length) return null;
    const card=node('div',null,'chart'), heading=node('div',null,'chart-header');heading.append(node('h3',title),node('small','SELECTION INDEX'));card.append(heading);
    const lines=available.map(([key,name],i)=>({key,name,color:colors[i%colors.length],values:rows().map(row=>row.metrics[key])}));
    const plot=basePlot(lines,rows().length), legend=node('div',null,'legend');
    lines.forEach(line=>{const entry=node('span'), swatch=node('i');swatch.style.background=line.color;entry.append(swatch,document.createTextNode(line.name));legend.append(entry);});
    const values=node('div',null,'chart-values'); card.append(plot.root,legend,values,node('p',note,'chart-note'));
    state.charts.push({plot,values,lines,card});return card;
  }
  function openRun(id, index) {
    const run=data.runs[id];if(!run)return;state.run=run;state.charts=[];state.gallery='timeline';state.page=0;
    state.index=Math.max(0,Math.min(Number.isInteger(index)?index:Math.max(0,(run.report?.generations.length||1)-1),(run.report?.generations.length||1)-1));
    app.replaceChildren(); app.append(button('← All runs',overview,'back'));
    const heading=titleBlock(`${strategy(run)} / ${fmt(run.seed)}`,run.label,'RUN INSPECTOR');
    const switcher=selectControl(data.runs.map(r=>[String(r.id),`${strategy(r)} · seed ${fmt(r.seed)} · ${r.label}`]),String(id),value=>openRun(Number(value)),'Choose run');heading.append(switcher);app.append(heading);
    if(run.error)app.append(node('p',run.error,'note error'));
    if(!run.report){
      app.append(node('p',`This run is ${run.status}. Metric records are unavailable.`,'empty'));
      const links=node('div',null,'links');Object.entries(run.files).forEach(([name,url])=>{const a=node('a',name);a.href=url;links.append(a);});app.append(links);
      if(run.images.length){
        app.append(node('h2','All generated images'));
        const gallery=node('div');gallery.id='detail-gallery';app.append(gallery);
        state.gallery='images';drawGallery();
      }
      history.replaceState(null,'',`#run=${run.id}`);window.scrollTo(0,0);return;
    }
    const report=run.report, s=report.summary;
    app.append(statBar([['Status',run.status],['Selections',s.decisions],['Presentations',s.candidate_presentations],['Genomes',run.images.length],['Back / reset',`${s.backtracks||0} / ${s.resets||0}`]]));
    const links=node('div',null,'links');Object.entries(run.files).forEach(([name,url])=>{const a=node('a',name);a.href=url;links.append(a);});app.append(links);
    const final=node('div',null,'final-mini');final.append(image(run,report.final_image,'Current saved selection'),node('span',`Saved selection: ${report.final_selected_id==null?'none':`genome #${report.final_selected_id}`}`));links.append(final);
    if(strategy(run)==='human')app.append(node('p','Human history: each selection click has a row. Grid visits and mutation rounds are separate. Backtracking and resets remain in the exploration budget.','note'));
    if(report.generations.length && report.final_selected_id!==report.generations.at(-1).selected_id)app.append(node('p','The saved selection differs from the last chronological choice. Use Final ancestry to inspect the restored image’s actual parent chain.','note'));
    if(rows().length) {
      const controls=node('div',null,'controls');
      const prev=button('←',()=>setSelection(state.index-1));prev.id='previous';prev.setAttribute('aria-label','Previous selection');
      const next=button('→',()=>setSelection(state.index+1));next.id='next';next.setAttribute('aria-label','Next selection');
      const output=node('output');output.id='selection-index';output.setAttribute('aria-live','polite');
      const slider=node('input');slider.type='range';slider.min=0;slider.max=rows().length-1;slider.step=1;slider.value=state.index;slider.id='generation-slider';slider.setAttribute('aria-label','Selection index');slider.oninput=()=>setSelection(Number(slider.value));
      controls.append(prev,slider,next,output);app.append(controls);
      const inspector=node('div',null,'inspector'), visual=node('section',null,'visual-panel');visual.id='visual-panel';visual.setAttribute('aria-label','Selected image and candidate grid');
      const plots=node('section',null,'plots');plots.setAttribute('aria-label','Run metric charts');inspector.append(visual,plots);app.append(inspector);
      groups.forEach(([title,keys,note])=>{if(title==='Display novelty' && rows().some(r=>'novelty_selected' in r.metrics))return;const chart=makeChart(title,keys,note);if(chart)plots.append(chart);});
      const extra=node('div',null,'toolbar'), extraHost=node('div');extraHost.id='extra-chart';
      const metricKeys=Object.keys(report.summary.metrics).sort();
      const choice=selectControl([['','Additional metric…'],...metricKeys.map(k=>[k,label(k)])],'',key=>{
        state.charts=state.charts.filter(c=>!extraHost.contains(c.card));extraHost.replaceChildren();
        if(key)extraHost.append(makeChart(label(key),[[key,label(key)]],'Values are recorded per selection; unavailable measurements remain gaps.'));updateCursors();
      },'Additional metric');extra.append(node('label','Inspect any metric'),choice);app.append(extra,extraHost);
      const classKey=rows().some(r=>r.selected_class)?'selected_class':rows().some(r=>r.posthoc_selected_class)?'posthoc_selected_class':null;
      if(classKey){const strip=node('div',null,'class-timeline');app.append(node('h3',classKey==='selected_class'?'Selected ImageNet classes':'Post-run selected classes'));rows().forEach((r,i)=>{const c=r[classKey];if(c)strip.append(button(`${i}: ${c.name} [${c.index}]`,()=>setSelection(i)));});app.append(strip);}
    } else app.append(node('p','No explicit selections were recorded. All generated images and display visits are available below.','empty'));
    const galleryHeading=node('div',null,'section-heading');galleryHeading.append(node('h2','Explore the images'));app.append(galleryHeading);
    const tabs=node('div',null,'gallery-tabs');
    [['timeline','Selected timeline'],['displays','All display visits'],['ancestry','Final ancestry'],['images','All generated images']].forEach(([key,name])=>{const b=button(name,()=>{state.gallery=key;state.page=0;drawGallery();});b.dataset.gallery=key;tabs.append(b);});app.append(tabs);
    const gallery=node('div');gallery.id='detail-gallery';app.append(gallery);
    const detail=node('details',null,'configuration');detail.append(node('summary','All scalar metrics at this selection'));const detailBody=node('div');detailBody.id='metric-table';detail.append(detailBody);app.append(detail);
    const summary=node('details',null,'configuration');summary.append(node('summary','Run summary and measurement counts'),node('pre',JSON.stringify(report.summary,null,2)));app.append(summary);
    const config=node('details',null,'configuration');config.append(node('summary','Configuration and provenance'),node('pre',JSON.stringify({settings:report.settings,strategy:report.selection_strategy,post_run_evaluation:report.posthoc_evaluation},null,2)));app.append(config);
    app.append(node('p','Indices start at 0. Charts follow recorded selection events; human rounds can repeat or move backwards. Images and diagnostics are evidence to inspect, not an overall interestingness score.','footnote'));
    if(rows().length)setSelection(state.index);else {drawGallery();history.replaceState(null,'',`#run=${run.id}`);}
    window.scrollTo(0,0);
  }
  function updateCursors() {
    const row=rows()[state.index];if(!row)return;
    state.charts.forEach(({plot,values,lines})=>{plot.cursor.setAttribute('x1',plot.x(state.index));plot.cursor.setAttribute('x2',plot.x(state.index));values.textContent=lines.map(line=>`${line.name}: ${fmt(row.metrics[line.key])}`).join(' · ');});
  }
  function setSelection(index) {
    if(!rows().length)return;state.index=Math.max(0,Math.min(rows().length-1,index));const row=rows()[state.index], run=state.run;
    document.getElementById('generation-slider').value=state.index;document.getElementById('selection-index').textContent=`Selection ${state.index} / ${rows().length-1}`;
    document.getElementById('previous').disabled=state.index===0;document.getElementById('next').disabled=state.index===rows().length-1;
    const panel=document.getElementById('visual-panel');panel.replaceChildren();
    const hero=button('',()=>zoom(run,{image:row.selected_image,genome_id:row.selected_id,...row.metrics},`Selected genome #${row.selected_id}`),'hero-image');hero.setAttribute('aria-label',`Enlarge selected genome ${row.selected_id}`);hero.append(image(run,row.selected_image,`Selected genome #${row.selected_id}`,true));panel.append(hero);
    const caption=node('div',null,'image-caption');caption.append(node('span',`SELECTED #${row.selected_id}`),node('span',`ROUND ${row.round} · DISPLAY ${row.display_index}`));panel.append(caption);
    const grid=node('div',null,'candidate-grid');row.candidates.forEach((candidate,i)=>{
      const b=button('',()=>zoom(run,candidate,`Candidate ${i} · genome #${candidate.genome_id}`),`candidate ${i===row.selected_position?'chosen':''}`);
      b.setAttribute('aria-label',`Candidate ${i}, genome ${candidate.genome_id}${i===row.selected_position?', selected':''}`);
      b.append(image(run,candidate.image,`Candidate ${i}`,true),node('span',`${i===row.selected_position?'● ':''}#${candidate.genome_id}`));
      const c=candidate.top_class||candidate.posthoc_top_class;if(c){const l=node('span',c.name);l.title=`${c.name} [${c.index}]`;b.append(l);}grid.append(b);
    });panel.append(grid);
    const notes=node('div',null,'row-details');notes.append(node('span',row.selection_mode,'badge'));
    (row.preceding_actions||[]).filter(a=>a==='back'||a==='reset').forEach(a=>notes.append(node('span',a,'badge')));
    if(row.metrics.comprehension_warmup_complete===0)notes.append(node('span','Observer warm-up','badge'));
    if(row.metrics.offspring_warmup_complete===0)notes.append(node('span','Predictor warm-up','badge'));
    if(row.metrics.offspring_forecast_used===1)notes.append(node('span','Forecast active','badge'));
    if(row.mutation)notes.append(node('p',`Mutation σ ${row.mutation.strength}; structure changes ${row.mutation.topology?'enabled':'disabled'}.`));
    if(row.offspring_outcome)notes.append(node('p',`Offspring outcome observed at selection ${row.offspring_outcome.received_generation}.`));panel.append(notes);
    updateCursors();
    const gallery=document.getElementById('detail-gallery');
    if(!gallery.children.length)drawGallery();
    else if(state.gallery==='timeline'){
      const strip=gallery.querySelector('.timeline-strip');
      if(strip){Array.from(strip.children).forEach((b,i)=>b.className=`thumb ${i===state.index?'active':''}`);const active=strip.children[state.index];if(active)strip.scrollLeft=Math.max(0,active.offsetLeft-strip.offsetLeft-strip.clientWidth/2);}
    }
    const metrics=document.getElementById('metric-table');metrics.replaceChildren(table(['Metric','Value'],Object.entries(row.metrics).map(([k,v])=>[label(k),v])));
    history.replaceState(null,'',`#run=${run.id}&generation=${state.index}`);
  }
  function thumbnail(item, text, action, active=false) {
    const b=button('',action,`thumb ${active?'active':''}`);b.append(image(state.run,item.image||item.path||item.selected_image,text),node('small',text));b.title=text;return b;
  }
  function drawGallery() {
    const host=document.getElementById('detail-gallery');if(!host)return;host.replaceChildren();
    document.querySelectorAll('[data-gallery]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.gallery===state.gallery)));
    const report=state.run.report;
    if(state.gallery==='timeline') {
      host.append(node('p','Every explicit selection in chronological order. Repeated choices and discarded branches stay visible.','gallery-caption'));
      const strip=node('div',null,'timeline-strip');rows().forEach((r,i)=>strip.append(thumbnail(r,`${i} · #${r.selected_id}`,()=>setSelection(i),i===state.index)));host.append(strip);
      if(!rows().length)host.append(node('p','No recorded selections.','empty'));
      const active=strip.querySelector('.active');if(active)strip.scrollLeft=Math.max(0,active.offsetLeft-strip.offsetLeft-strip.clientWidth/2);
      return;
    }
    if(state.gallery==='ancestry') {
      host.append(node('p','Root → current saved selection, following genome parent links. Repeated selections of the same genome are not extra ancestry nodes.','gallery-caption'));
      const strip=node('div',null,'timeline-strip');(report.final_ancestry||[]).forEach(item=>strip.append(thumbnail(item,`#${item.genome_id}`,()=>zoom(state.run,item,`Ancestor #${item.genome_id}`))));host.append(strip);
      if(!strip.children.length)host.append(node('p','There is no current saved selection. Earlier exploration is still available in the other views.','empty'));
      return;
    }
    const displays=state.gallery==='displays', items=displays?(report.display_history||[]):state.run.images, size=displays?8:60;
    const pages=Math.max(1,Math.ceil(items.length/size));state.page=Math.min(state.page,pages-1);
    host.append(node('p',displays?'Every grid visit, including Back, resets and grids without an explicit selection.':'Every generated genome, including rejected candidates and abandoned branches. Click an image to enlarge it.','gallery-caption'));
    const list=node('div',null,displays?'display-list':'all-images');
    items.slice(state.page*size,(state.page+1)*size).forEach(item=>{
      if(!displays){list.append(thumbnail(item,`#${item.genome_id}`,()=>zoom(state.run,item,`Genome #${item.genome_id}`)));return;}
      const card=node('div',null,'display-card');card.append(node('h3',`Display ${item.display_index} · ${item.action} · round ${item.round} · ${item.selection_generations.length} selection(s)`));
      const grid=node('div',null,'display-images');item.candidates.forEach(c=>grid.append(thumbnail(c,`#${c.genome_id}`,()=>zoom(state.run,c,`Display ${item.display_index} · genome #${c.genome_id}`),c.genome_id===item.selected_id)));card.append(grid);
      item.selection_generations.forEach(i=>card.append(button(`Selection ${i}`,()=>setSelection(i))));list.append(card);
    });host.append(list);
    const pager=node('div',null,'pager'), prev=button('← Previous',()=>{state.page--;drawGallery();}), next=button('Next →',()=>{state.page++;drawGallery();});prev.disabled=state.page===0;next.disabled=state.page===pages-1;pager.append(prev,node('span',`Page ${state.page+1} / ${pages} · ${items.length} ${displays?'visits':'images'}`),next);host.append(pager);
  }
  document.addEventListener('keydown',event=>{
    if(!state.run||dialog.open||['INPUT','SELECT','TEXTAREA'].includes(document.activeElement?.tagName)||event.altKey||event.ctrlKey||event.metaKey)return;
    if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();setSelection(state.index+(event.key==='ArrowRight'?1:-1));}
  });
  const initial=new URLSearchParams(location.hash.slice(1));
  if(initial.has('run')&&data.runs[Number(initial.get('run'))])openRun(Number(initial.get('run')),Number(initial.get('generation')||0));
  else if(data.runs.length===1)openRun(0);else overview();
})();
