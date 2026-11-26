'use strict';
const $ = id => document.getElementById(id);
const types = {current_observation:'当前观察',history_recall:'历史回忆',event_notification:'事件通知',temporal_comparison:'时序比较'};
const state = {data:null,video:null,task:null,stop:null,pending:null,revision:0};
const player=$('player');
const segments = new URLSearchParams(location.search).get('section') === 'segments';
const v3 = new URLSearchParams(location.search).get('section') === 'qa-v3';
document.body.dataset.section = segments ? 'segments' : v3 ? 'qa-v3' : 'qa';
$(segments ? 'segments-nav' : v3 ? 'v3-nav' : 'qa-nav').setAttribute('aria-current', 'page');
function sectionVideos() {
  return segments ? state.data.videos.filter(v => v.visual_events?.length) : state.data.videos;
}
function videoSummary(video) {
  return fmt(video.duration) + ' · ' + (segments ? video.visual_events.length + ' 个片段' : video.tasks.length + ' QA');
}
function fmt(t){const n=Math.max(0,Math.floor(t||0));return `${String(Math.floor(n/60)).padStart(2,'0')}:${String(n%60).padStart(2,'0')}`;}
function stamp(t) {
  const ms = Math.round(t * 1000);
  return fmt(ms / 1000) + (ms % 1000 ? '.' + String(ms % 1000).padStart(3, '0').replace(/0+$/, '') : '');
}
function node(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
function bookmark(){const u=new URL(location.href);u.searchParams.set('video',state.video.id);if (segments) u.searchParams.delete('qa'); else u.searchParams.set('qa',state.task.task_id);history.replaceState(null,'',u);}
function renderVideos(){const list=$('video-list');list.replaceChildren();for(const v of sectionVideos()){const b=node('button','video-option'+(v===state.video?' active':''));b.type='button';b.dataset.video=v.id;b.setAttribute('aria-pressed',String(v===state.video));const img=node('img');img.src=v.poster;img.alt='';img.loading='lazy';const text=node('span');text.append(node('strong','',v.title),node('small','',videoSummary(v)));b.append(img,text);b.onclick=()=>selectVideo(v);list.append(b);}}
function renderTasks(){const term=$('search').value.trim().toLowerCase(),type=$('type-filter').value;const tasks=state.video.tasks.filter(t=>(!type||t.qa_type===type)&&(!term||(t.question+' '+(t.reference_answer || '')).toLowerCase().includes(term)));$('qa-count').textContent=`${tasks.length} / ${state.video.tasks.length}`;const list=$('qa-list');list.replaceChildren();if(!tasks.length){list.append(node('p','empty','没有匹配的 QA。试试其他关键词或类型。'));return;}for(const t of tasks){const b=node('button','qa-option'+(t===state.task?' active':''));b.type='button';b.dataset.task=t.task_id;b.setAttribute('aria-pressed',String(t===state.task));const times=node('span','qa-times');times.append(node('span','',`Q ${fmt(t.query_time)}`),node('span','',t.answer_time == null ? 'WAIT' : 'A ' + fmt(t.answer_time)));b.append(node('span','qa-type',types[t.qa_type]),node('span','qa-question',t.question),times);b.onclick=()=>selectTask(t);list.append(b);}}
function seek(time,play=false,stop=null){const revision=++state.revision;state.stop=stop;player.pause();state.pending={time,play,revision};applySeek();}
function applySeek(){const p=state.pending;if(!p||player.readyState<1)return;state.pending=null;player.currentTime=Math.max(0,Math.min(p.time,player.duration||state.video.duration));tick();if(p.play)player.play().catch(()=>{if(p.revision===state.revision){state.stop=null;$('context-note').textContent='浏览器暂停了自动播放，请点击视频中的播放按钮。';}});}
function renderVisualEvents(video) {
  const events = video.visual_events || [];
  $('visual-events').hidden = !segments || events.length === 0;
  const list = $('event-list');
  list.replaceChildren();
  for (const event of events) {
    const button = node('button', 'event-option');
    button.type = 'button';
    const image = node('img');
    image.src = event.thumbnail;
    image.alt = '';
    image.loading = 'lazy';
    button.append(image, node('strong', '', event.id),
      node('span', '', fmt(event.start) + '–' + fmt(event.end)));
    button.onclick = () => seek(event.start, true, event.end);
    list.append(button);
  }
}
function selectVideo(v,taskId){state.video=v;state.stop=null;state.pending=null;state.revision++;player.pause();$('media-error').hidden=true;$('search').value='';$('type-filter').value='';$('video-title').textContent=v.title;$('video-topic').textContent=v.title_en;$('video-meta').textContent=videoSummary(v);$('duration').textContent='/ '+fmt(v.duration);$('scrubber').max=v.duration;player.poster=v.poster;player.src=v.url;player.load();
 $('v3-nav').href = '?section=qa-v3&video=' + encodeURIComponent(v.id);
 $('qa-nav').href = '?video=' + encodeURIComponent(v.id);
 const clipVideo = v.visual_events?.length ? v : state.data.videos.find(item => item.visual_events?.length);
 if (clipVideo) $('segments-nav').href = '?section=segments&video=' + encodeURIComponent(clipVideo.id);
 renderVideos();renderVisualEvents(v);
 if (segments) { state.task=null; bookmark(); seek(0); return; }
 selectTask(v.tasks.find(t=>t.task_id===taskId)||v.tasks.find(t=>t.task_id===v.featured_task)||v.tasks[0]);}
function showPrediction(task, index) {
  const d = task.decisions[index];
  $('model-action').textContent = d.action;
  $('model-text').textContent = d.text;
  $('forced-row').hidden = !d.ready;
  $('forced-text').textContent = d.forced || '';
  $('match-label').textContent = task.different ? '与 GT 文本不同' : '与 GT 文本一致';
}
function showV3(task) {
  $('v3-review').hidden = false;
  $('query-decision').textContent = task.decision_at_query;
  $('query-reason').textContent = task.decision_reason_at_query;
  $('criterion-row').hidden = !task.trigger_criterion;
  $('criterion').textContent = task.trigger_criterion || '';
  const select = $('model-time');
  select.replaceChildren();
  for (const [i,d] of task.decisions.entries()) {
    select.add(new Option(stamp(d.time) + ' · GT ' + (d.ready ? 'ANSWER' : 'WAIT'), String(i)));
  }
  const answer = task.decisions.findIndex(d => d.ready);
  select.selectedIndex = Math.max(0, answer);
  showPrediction(task, select.selectedIndex);
  select.onchange = () => {
    showPrediction(task, select.selectedIndex);
    seek(task.decisions[select.selectedIndex].time);
  };
  $('jump-difference').value = task.different ? task.task_id : '';
}
function selectTask(t) {
  state.task = t;
  const q = t.query_time, a = t.answer_time, instant = a === q, open = a == null;
  const end = open ? t.coverage_end : a;
  $('type-badge').textContent = types[t.qa_type];
  $('wait-label').textContent = open ? '观察区间内持续 WAIT' : instant ? '可即时回答' : '等待 ' + (a-q) + ' 秒';
  $('question').textContent = t.question;
  $('answer').textContent = t.reference_answer ?? '该标注区间内没有出现可回答的证据。';
  $('query-time').textContent = stamp(q);
  $('answer-time').textContent = open ? '—' : stamp(a);
  $('go-answer').disabled = open;
  $('task-id').textContent = t.task_id;
  $('play-window-label').textContent = instant ? '播放时刻前后画面' : open ? '播放标注观察区间' : '播放 Q–A 区间';
  $('interval-duration').textContent = instant ? '前后各 3 秒' : String(Math.round((end-q)*1000)/1000) + ' 秒';
  $('context-note').textContent = instant ? '提问和回答发生在同一时刻。播放前后各 3 秒可查看上下文。' :
      open ? '标注观察区间止于 ' + stamp(end) + '。' : '播放从提问开始，到标注回答时暂停。';
  $('window-label').textContent = instant ? 'Q = A · ' + stamp(q) : stamp(q) + ' – ' + stamp(end);
  const left = 100*q/state.video.duration, right = 100*end/state.video.duration;
  $('q-marker').style.left = left+'%';
  $('a-marker').hidden = open;
  $('a-marker').style.left = right+'%';
  $('a-marker').classList.toggle('coincident',right-left<2);
  $('wait-range').style.left=left+'%';
  $('wait-range').style.width=(right-left)+'%';
  $('evidence-details').open = false;
  const ev=$('evidence'); ev.replaceChildren();
  if (v3) {
    showV3(t);
    for (const [start,stop] of t.evidence_windows) {
      const b = node('button','window-button',start===stop ? stamp(start) : stamp(start)+' – '+stamp(stop));
      b.type='button';
      b.onclick=()=>seek(start, stop>start, stop>start ? stop : null);
      ev.append(b);
    }
  } else {
    for (const e of t.evidence) {
      const b=node('button'), img=node('img'); b.type='button';
      img.src=e.url; img.alt=fmt(e.time)+' 的原始证据帧'; img.loading='lazy';
      b.append(img,node('span','',(e.time<q?'历史证据':'回答证据')+' · '+fmt(e.time)));
      b.onclick=()=>seek(e.time); ev.append(b);
    }
  }
  renderTasks();bookmark();seek(q);
}
function tick(){const t=player.currentTime||0;$('clock').textContent=fmt(t);$('scrubber').value=t;if(state.task){const task=state.task;const phase=t+.025<task.query_time?'before':task.answer_time == null ? (t > task.coverage_end ? 'outside' : 'wait') : t+.025<task.answer_time?'wait':'ready';$('phase').textContent=phase==='before'?'尚未发问':phase==='outside'?'标注区间已结束':phase==='wait'?'WAIT · 等待证据':'ANSWER · 已到回答时刻';$('phase').className='phase '+phase;}
 if(state.stop!==null&&!player.seeking&&t>=state.stop){const end=state.stop;state.stop=null;player.pause();player.currentTime=end;}}
$('go-query').onclick=()=>seek(state.task.query_time);$('go-answer').onclick=()=>seek(state.task.answer_time);
$('play-window').onclick=()=>{const t=state.task;const instant=t.answer_time===t.query_time;seek(instant?Math.max(0,t.query_time-3):t.query_time,true,Math.min(state.video.duration,instant?t.answer_time+3:(t.answer_time ?? t.coverage_end)));};
$('scrubber').oninput=()=>seek(Number($('scrubber').value));$('search').oninput=renderTasks;$('type-filter').onchange=renderTasks;
player.addEventListener('loadedmetadata',applySeek);player.addEventListener('seeked',tick);player.addEventListener('timeupdate',tick);player.addEventListener('ended',()=>{state.stop=null;tick();});
function animation(){if(!player.paused)tick();requestAnimationFrame(animation);}requestAnimationFrame(animation);
player.addEventListener('error',()=>{$('media-error').hidden=false;$('media-error').textContent='视频加载失败。请确认原视频文件可用，并通过附带的 HTTP 服务打开页面。';});
$('copy-link').onclick=async()=>{try{await navigator.clipboard.writeText(location.href);$('copy-link').textContent='链接已复制';setTimeout(()=>$('copy-link').textContent='复制此 QA 链接',1800);}catch{window.prompt('复制此 QA 链接',location.href);}};
function fromURL(){const p=new URLSearchParams(location.search);const videos=sectionVideos();const v=videos.find(v=>v.id===p.get('video'))||videos[0];if(!v){$('loading').hidden=false;$('loading').textContent='尚无已切分的视频。';$('explorer').hidden=true;return;}selectVideo(v,p.get('qa'));}
window.addEventListener('popstate',()=>{if(state.data)fromURL();});
async function loadV3() {
  const res = await fetch('qa-v3.json');
  if (!res.ok) throw new Error('QA v3 HTTP ' + res.status);
  const release = await res.json();
  const choices = $('jump-difference');
  for (const video of state.data.videos) {
    video.tasks = release.videos[video.id];
    video.featured_task = video.tasks[0].task_id;
    for (const t of video.tasks.filter(t => t.different)) choices.add(new Option(t.task_id,t.task_id));
  }
  state.data.counts.qa = state.data.videos.reduce((n,v)=>n+v.tasks.length,0);
  $('differences').hidden = false;
  choices.onchange = () => {
    if (choices.value) selectVideo(state.data.videos.find(v=>v.id===choices.value.split('/')[0]), choices.value);
  };
  document.querySelector('.collection').textContent = 'QA v3 · Epoch ' + release.epoch;
  document.querySelector('.intro h1').textContent = '新版 QA 与模型回答';
  document.querySelector('.intro .eyebrow').hidden = true;
  document.querySelector('.intro h1 + p').textContent = '回看标注证据，对照参考答案与模型在各标注时刻的输出。';
  document.querySelector('footer').textContent = '同集 overfit 评测 · Qwen3.5-2B 与双 Q-Former · 第 '+release.epoch+' 轮。模型输出按固定标注时刻生成。';
}
async function init(){try{const r=await fetch('data.json');if(!r.ok)throw new Error(`HTTP ${r.status}`);state.data=await r.json();if(!state.data.videos?.length)throw new Error('没有视频数据');if (v3) await loadV3();const c=state.data.counts;for(const [n,label] of [[c.videos,'视频案例'],[c.qa,'英文 QA'],[4,'问题类型']]){const s=node('div','stat');s.append(node('strong','',String(n)),node('span','',label));$('stats').append(s);}$('video-count').textContent=sectionVideos().length + " 个视频";$('loading').hidden=true;$('explorer').hidden=false;fromURL();}catch(e){$('loading').textContent=`加载失败：${e.message}。请先运行 python3 scripts/build_qa_examples.py，并通过 HTTP 服务访问。`;}}
init();
