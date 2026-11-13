function stage3Task(s){
 const c=card(qaName(s.qa_type),msg('qa',{split:splitName(s.split),group:responseName(s.response_group)}));
 para(c,'Question',s.question);
 para(c,'提问 / 最早回答',fmt(s.query_time)+' / '+(s.answer_time===null?msg('untriggered',{time:fmt(s.review_horizon)}):fmt(s.answer_time)));
 para(c,'标注画面',msg('resolution',{size:s.annotation_resolution.join(' × ')}));
 const label=el('label','决策时刻 '),selector=el('select');selector.id='decision-select';label.append(selector);c.append(label);
 const states=s.demo_decisions||[];
 states.forEach((d,i)=>option(selector,String(i),fmt(d.decision_time)+' · '+d.status));
 const detail=el('div');c.append(detail);
 function decision(){
  const d=states[Number(selector.value)||0];if(!d)return;
  detail.replaceChildren();para(detail,'输入视频前缀',range(d.video_window));
  para(detail,'监督目标 · 非模型输出',d.target);
  if(d.missing_evidence)para(detail,'Why WAIT',d.missing_evidence);
  playButton(detail,d.video_window);timeline(d.video_window,null,[s.query_time,d.decision_time]);seek(d.decision_time);
 }
 decisionSelection=Math.max(0,Math.min(states.length-1,decisionSelection));selector.value=String(decisionSelection);
 selector.onchange=()=>{decisionSelection=Number(selector.value)||0;decision();syncURL();};decision();
 const answer=card('参考答案与证据','审核信息 · 不属于模型输入');
 const disclose=el('details');disclose.append(el('summary','展开参考答案 / 原尺寸证据'));answer.append(disclose);
 para(disclose,'Reference answer',s.reference_answer||'No trigger within the reviewed horizon. Remain in WAIT.');
 if(s.trigger_rule)para(disclose,'Trigger rule',s.trigger_rule);
 if(s.query_time_missing_evidence)para(disclose,'Missing at query time',s.query_time_missing_evidence);
 if(s.answer_time_new_evidence)para(disclose,'New evidence at answer time',s.answer_time_new_evidence);
 if(s.historical_anchor)para(disclose,'历史证据范围',fmt(s.historical_anchor.start)+' – '+fmt(s.historical_anchor.end));
 for(const f of s.evidence||[]){
  const a=el('a');a.href=f.demo_url;a.target='_blank';a.rel='noopener';
  const img=el('img');img.src=f.demo_url;img.alt=msg('evidenceAlt',{time:fmt(f.time)});img.loading='lazy';img.style.width='100%';img.style.height='auto';
  a.append(img,el('span',msg('evidence',{time:fmt(f.time),size:f.width+'×'+f.height})));disclose.append(a);
 }
 para(disclose,'Task / event group',s.task_id+' / '+s.shared_event_group);
 para(disclose,'时序精度',s.answer_time_bracket.precision);
}

const $ = id => document.getElementById(id);
const player=$('video'), videoSelect=$('video-select'), stageSelect=$('stage-select'), sampleSelect=$('sample-select');
let catalog, current, samples=[], stopAt=null, autoSeeking=false, activeDuration=0, localSelection=0, decisionSelection=0, preservePlayback=false;
const fmt=t=>Number(t).toFixed(1)+'s';
const range=w=>'['+w.map(fmt).join(', ')+')';
const clock=t=>Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0');
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=tr(text);if(cls)e.className=cls;return e;}
function option(select,value,text){const e=el('option',text);e.value=value;select.append(e);}
function card(title,sub){const c=el('article',undefined,'col');c.append(el('h2',title),el('p',sub,'csub'));$('columns').append(c);return c;}
function para(c,title,text){
 c.append(el('p',title,'field-label'),el('p',text,'prose'));
 if(displayLanguage==='zh'&&typeof text==='string'&&contentZh[text]&&contentZh[text]!==text){
  const original=el('details',undefined,'original-text');original.append(el('summary','Show English source'));
  const body=document.createElement('p');body.className='prose';body.textContent=text;original.append(body);c.append(original);
 }
}
function syncURL(){
 const u=new URL(location.href);u.searchParams.set('video',current.video_id);u.searchParams.set('stage',stageSelect.value);u.searchParams.set('sample',sampleSelect.value||'0');
 if(stageSelect.value==='stage2')u.searchParams.set('local',String(localSelection));else u.searchParams.delete('local');
 u.searchParams.set('lang',displayLanguage);
 if(stageSelect.value==='stage3')u.searchParams.set('decision',String(decisionSelection));else u.searchParams.delete('decision');
 history.replaceState(null,'',u);
}
function timeline(window,target=null,frames=[]){
 const pct=t=>100*t/Math.max(1,activeDuration)+'%';
 $('tl-input').style.left=pct(window[0]);$('tl-input').style.width=pct(window[1]-window[0]);$('tl-sentence').hidden=!target;
 if(target){$('tl-sentence').style.left=pct(target[0]);$('tl-sentence').style.width=pct(target[1]-target[0]);}
 $('tl-frames').replaceChildren();for(const t of frames){const m=el('div',undefined,'fmark');m.style.left=pct(t);m.title=fmt(t);$('tl-frames').append(m);}
 $('window-label').textContent=range(window);
}
function seek(t){if(preservePlayback)return;stopAt=null;player.pause();player.currentTime=Math.min(t,activeDuration);}
function playButton(c,w,target=null,frames=[]){
 const b=el('button',msg('play',{window:range(w)}),'playbtn');b.type='button';c.append(b);
 b.onclick=()=>{timeline(w,target,frames);player.pause();stopAt=w[1];autoSeeking=true;player.currentTime=w[0];player.play().catch(()=>{stopAt=null;});};
}
function renderPresentation(){
 $('featured-videos').replaceChildren();const p=catalog.presentation;if(!p?.features?.length)return;
 $('presentation').hidden=false;$('presentation-title').textContent=tr(p.title);$('presentation-note').textContent=tr(p.note);
 for(const f of p.features){
  const b=el('button',undefined,'featured-card');b.type='button';b.dataset.video=f.video_id;b.setAttribute('aria-label',tr(f.title)+', '+tr(f.tag));
  const image=el('img');image.src=f.preview;image.alt=msg('preview',{title:tr(f.title),time:clock(f.preview_time)});image.loading='lazy';b.append(image);
  const body=el('span',undefined,'featured-body');body.append(el('span',(f.split==='eval'?msg('evalBadge'):'TRAIN')+' · '+tr(f.tag),'featured-tag'),el('strong',f.title),el('span',f.focus,'featured-focus'),el('span',clock(f.window[0])+' – '+clock(f.window[1])+' · '+f.video_id,'featured-time'));b.append(body);
  b.onclick=()=>{videoSelect.value=f.video_id;stageSelect.value='stage2';selectVideo(f.sample,f.local);};$('featured-videos').append(b);
 }
}
function selectVideo(sample=0,local=0){
 current=catalog.videos.find(v=>v.video_id===videoSelect.value);stopAt=null;player.pause();selectStage(sample,local);
}
function selectStage(sample=0,local=0){
 if(!current)return;
 const stage=current.stages[stageSelect.value];activeDuration=stage.duration??current.duration;
 $('sentence-legend').hidden=stageSelect.value!=='stage1';
 const sourceURL=new URL(stage.video_url||current.video_url,location.href).href;
 if(player.src!==sourceURL){player.pause();player.src=sourceURL;}
 $('timeline').setAttribute('aria-valuemax',activeDuration);samples=stage.samples||[];sampleSelect.replaceChildren();
 $('sample-label').hidden=!stage.available;$('empty').hidden=!!stage.available;$('empty').textContent=tr(stage.note);
 $('source').textContent=stage.available?stage.source+'\nSHA256: '+stage.sha256:tr('无来源文件');
 $('status').textContent=tr(stage.note);
 $('meta').textContent=current.video_id+' · '+clock(activeDuration)+' · '+msg('sampleCount',{count:samples.length})+(stage.frame_budget?' · '+msg('budget',{count:stage.frame_budget,size:stage.frame_size}):'');
 if(stageSelect.value==='stage2'&&stage.available){
  const tail=Math.max(0,activeDuration-stage.coverage_end);
  $('meta').textContent+=' · '+msg('coverage',{time:clock(stage.coverage_end)})+(tail>0?' · '+msg('tail',{time:tail.toFixed(1)}):'');
  if(stage.review){
   $('meta').textContent+=' · '+stage.review.split.toUpperCase()+' · '+msg('targets',{count:stage.review.targets});
   $('status').textContent=msg('review',{size:stage.review.original_size.join('×')});
   $('source').textContent+='\n'+msg('frozen',{hash:stage.review.release_manifest_sha256})+'\n'+msg('videoHash',{hash:stage.review.source_video_sha256});
  }
 }
 samples.forEach((s,i)=>option(sampleSelect,String(i),stageSelect.value==='stage1'?'#'+s.sentence_id+' '+range(s.sentence_window):stageSelect.value==='stage2'?'Block '+(i+1)+' · '+clock(s.block_window[0])+' – '+clock(s.block_window[1])+(!s.block_summary_target?' · '+msg('tailLocal'):''):qaName(s.qa_type||s.sample_type||'QA')+' · '+tr(s.question)));
 sampleSelect.disabled=!samples.length;sampleSelect.value=String(Math.max(0,Math.min(samples.length-1,Number(sample)||0)));
 document.querySelectorAll('.featured-card').forEach(b=>{const active=stageSelect.value==='stage2'&&b.dataset.video===current.video_id;b.classList.toggle('selected',active);b.setAttribute('aria-pressed',String(active));});
 selectSample(local);
}
function stage2Sample(s,requestedLocal){
 const c=card('Local','只概括选中的 10 秒');
 const label=el('label','选择 10 秒窗口 ','local-selector'),selector=el('select');selector.setAttribute('aria-label',tr('Local 10 秒窗口'));label.append(selector);c.append(label);
 const locals=s.local_sub_summaries;
 locals.forEach((l,i)=>option(selector,String(i),clock(l.window[0])+' – '+clock(l.window[1])));
 localSelection=Math.max(0,Math.min(locals.length-1,Number(requestedLocal)||0));selector.value=String(localSelection);
 const detail=el('div',undefined,'local-detail');c.append(detail);
 const list=el('details',undefined,'local-list');list.append(el('summary',msg('locals',{count:locals.length})));c.append(list);
 locals.forEach((l,i)=>{const box=el('div',undefined,'local-item');const b=el('button',clock(l.window[0])+' – '+clock(l.window[1]),'window-link');b.type='button';b.onclick=()=>{selector.value=String(i);changeLocal();};box.append(b,el('p',l.local_summary_target,'prose'));list.append(box);});
 function changeLocal(){
  localSelection=Number(selector.value)||0;const l=locals[localSelection];detail.replaceChildren();para(detail,'视频窗口',range(l.window));para(detail,'监督文本',l.local_summary_target);playButton(detail,l.window);
  timeline(l.window);seek(l.window[0]);syncURL();
 }
 selector.onchange=changeLocal;
 const block=card('Block','概括当前完整 60 秒，整合六条 Local');
 const global=card('Global','概括从视频开始到当前分钟结束');
 if(s.block_summary_target){
  para(block,'视频窗口',range(s.block_window));para(block,'监督文本',s.block_summary_target);playButton(block,s.block_window);
  para(global,'视频前缀',range([0,s.block_window[1]]));para(global,'监督文本',s.global_summary_target);playButton(global,[0,s.block_window[1]]);
 }else{
  para(block,'尾段未满 60 秒',msg('noBlock',{count:locals.length}));
  para(global,'该尾段没有新增 Global','Global 在完整分钟结束时生成。可选择前一个完整分钟查看其累计摘要。');
 }
 changeLocal();
}
function selectSample(local=0){
 if(!preservePlayback){stopAt=null;player.pause();}$('columns').replaceChildren();$('target-banner').hidden=true;localSelection=0;
 if(!samples.length){timeline([0,0]);syncURL();return;}
 const s=samples[Number(sampleSelect.value)||0];
 if(stageSelect.value==='stage1'){
  $('target-banner').hidden=false;$('target-text').textContent=tr(s.target);
  for(const r of s.conditions){const c=card(r.condition,r.condition==='through_with_asr'?'视频看到目标句结束':'视频只看到目标句开始');para(c,'视频输入',range(r.video_window)+' · '+msg('sampled',{count:r.frame_times.length}));para(c,'历史 ASR · 训练模板处理后',r.prompt_texts[0]);playButton(c,r.video_window,s.sentence_window,r.frame_times);c.onmouseenter=()=>timeline(r.video_window,s.sentence_window,r.frame_times);}
  timeline(s.conditions[0].video_window,s.sentence_window,s.conditions[0].frame_times);seek(s.sentence_window[0]);
 }else if(stageSelect.value==='stage2'){
  stage2Sample(s,local);
 }else if(s.schema_version==='stage3_task_v3_assistant_native'){
  stage3Task(s);
 }else{
  const c=card(s.sample_type||'QA','QA 监督样本 · 来源与审核状态见上方说明');para(c,'问题',s.question);para(c,'输入视频窗口',range(s.video_window));para(c,'监督目标 · WAIT / ANSWER',s.target);for(const [name,value] of Object.entries(s.meta||{}))para(c,name,typeof value==='object'?JSON.stringify(value):String(value));playButton(c,s.video_window);timeline(s.video_window);seek(s.video_window[0]);
 }
 syncURL();
}
player.addEventListener('timeupdate',()=>{if(!current)return;$('tl-playhead').style.left=100*player.currentTime/Math.max(1,activeDuration)+'%';$('timeline').setAttribute('aria-valuenow',player.currentTime);if(stopAt!==null&&player.currentTime>=stopAt){player.pause();stopAt=null;}});
player.addEventListener('seeking',()=>{if(!autoSeeking)stopAt=null;});player.addEventListener('seeked',()=>{autoSeeking=false;});
player.addEventListener('error',()=>{$('status').textContent+=' '+msg('videoError');});
$('timeline').onclick=e=>{const r=$('timeline').getBoundingClientRect();seek(Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*activeDuration);};
$('timeline').onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();seek(Math.max(0,player.currentTime+(e.key==='ArrowRight'?5:-5)));}};
videoSelect.onchange=()=>{decisionSelection=0;selectVideo();};stageSelect.onchange=()=>{decisionSelection=0;selectStage();};sampleSelect.onchange=()=>{decisionSelection=0;selectSample();};
(async()=>{try{
 await loadTranslations();applyStaticLanguage();
 const r=await fetch('catalog.json',{cache:'no-store'});if(!r.ok)throw Error(tr('请先运行 scripts/build_demo.py 生成 catalog.json'));catalog=await r.json();
 for(const v of catalog.videos)option(videoSelect,v.video_id,v.video_id+' · '+tr(v.title));
 renderPresentation();const params=new URLSearchParams(location.search);const first=catalog.presentation?.features?.[0];
 if(catalog.videos.some(v=>v.video_id===params.get('video')))videoSelect.value=params.get('video');else if(first)videoSelect.value=first.video_id;
 stageSelect.value=['stage1','stage2','stage3'].includes(params.get('stage'))?params.get('stage'):(first?'stage2':'stage1');
 decisionSelection=Math.max(0,Number(params.get('decision'))||0);
 const useFeaturedDefault=!params.has('video')&&!params.has('sample');selectVideo(useFeaturedDefault?first?.sample||0:Number(params.get('sample'))||0,useFeaturedDefault?first?.local||0:Number(params.get('local'))||0);
 videoSelect.disabled=false;stageSelect.disabled=false;
}catch(e){$('empty').hidden=false;$('empty').textContent=String(e);}})();

$('language-select').onchange=()=>{
 displayLanguage=$('language-select').value;
 try{localStorage.setItem('ultrasound-demo-language',displayLanguage);}catch{}
 applyStaticLanguage();if(!catalog||!current)return;
 const videoId=current.video_id,sample=sampleSelect.value,local=localSelection;
 const wasPlaying=!player.paused,time=player.currentTime,oldStop=stopAt;
 const opened=[...document.querySelectorAll('#columns details:not(.original-text)')].map(d=>d.open);
 preservePlayback=true;
 try{
  videoSelect.replaceChildren();for(const v of catalog.videos)option(videoSelect,v.video_id,v.video_id+' · '+tr(v.title));
  videoSelect.value=videoId;renderPresentation();selectStage(sample,local);
  document.querySelectorAll('#columns details:not(.original-text)').forEach((d,i)=>{if(!d.classList.contains('original-text'))d.open=opened[i]||false;});
 }finally{preservePlayback=false;stopAt=oldStop;}
 if(Math.abs(player.currentTime-time)>.1)player.currentTime=time;
 if(wasPlaying)player.play().catch(()=>{});
 syncURL();
};
