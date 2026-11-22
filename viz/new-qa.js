/* Independent viewer: canonical English QA, Chinese reading translations, recorded decisions. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const section = $('new-version-qa');
  const player = $('nqa-video');
  const make = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  const language = () => typeof displayLanguage !== 'undefined' ? displayLanguage : 'en';
  const local = (zh, en) => language() === 'zh' ? zh : en;
  const clock = value => `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
  const types = {
    current_observation: ['当前观察', 'Current observation'],
    history_recall: ['历史回忆', 'History recall'],
    temporal_comparison: ['变化比较', 'Temporal comparison'],
    event_notification: ['事件通知', 'Event notification'],
  };
  const typeName = type => local(...types[type]);
  let data = null, video = null, task = null, decision = 0, pendingSeek = null;
  let loadError = null, mediaFailed = false;

  function syncURL() {
    const url = new URL(location.href);
    if (task) url.searchParams.set('nqa_task', task.task_id);
    else url.searchParams.delete('nqa_task');
    url.searchParams.set('lang', language());
    history.replaceState(null, '', url);
  }
  function seek(seconds) {
    player.pause();
    pendingSeek = seconds;
    if (player.readyState >= 1) {
      player.currentTime = Math.min(seconds, player.duration || video.duration);
      pendingSeek = null;
      renderClock();
    }
  }
  player.addEventListener('loadedmetadata', () => {
    if (pendingSeek !== null) seek(pendingSeek);
  });
  player.addEventListener('play', () => {
    document.querySelectorAll('video').forEach(other => { if (other !== player) other.pause(); });
  });
  document.addEventListener('play', event => {
    if (event.target instanceof HTMLVideoElement && event.target !== player) player.pause();
  }, true);
  function renderClock() {
    $('nqa-clock').textContent = mediaFailed
      ? local('视频加载失败，请检查原视频是否可访问。', 'Video could not load. Check source video access.')
      : `${local('播放位置', 'Playback position')} ${clock(player.currentTime)} / ${clock(video?.duration || 0)}`;
  }
  player.addEventListener('timeupdate', renderClock);
  player.addEventListener('error', () => { mediaFailed = true; renderClock(); });

  function renderCards() {
    $('nqa-videos').replaceChildren();
    if (!data) return;
    for (const item of data.videos) {
      const button = make('button', undefined, 'nqa-video-card');
      button.type = 'button';
      button.dataset.video = item.video_id;
      button.setAttribute('aria-pressed', String(video === item));
      const image = make('img');
      image.src = item.poster; image.alt = item.title[language()]; image.loading = 'lazy';
      const body = make('span');
      body.append(make('strong', item.title[language()]), make('span', `${item.video_id} · ${clock(item.duration)} · ${item.tasks.length} QA`, 'nqa-meta'));
      button.append(image, body);
      button.onclick = () => selectVideo(item);
      $('nqa-videos').append(button);
    }
  }
  function selectVideo(next, selected = null) {
    video = next;
    player.pause(); pendingSeek = null; mediaFailed = false;
    player.src = video.url; player.poster = video.poster;
    player.setAttribute('aria-label', video.title[language()]);
    $('nqa-search').value = ''; $('nqa-type').value = ''; $('nqa-subset').value = '';
    renderCards(); selectTask(selected || video.tasks[0]); renderClock();
  }
  function filteredTasks() {
    if (!video) return [];
    const search = $('nqa-search').value.trim().toLowerCase();
    return video.tasks.filter(t => (!$('nqa-type').value || t.qa_type === $('nqa-type').value)
      && (!$('nqa-subset').value || t.subset === $('nqa-subset').value)
      && [t.task_id, t.question.en, t.question.zh, t.answer.en, t.answer.zh].some(s => s.toLowerCase().includes(search)));
  }
  function renderList() {
    const list = $('nqa-tasks');
    const previousScroll = list.scrollTop;
    const tasks = filteredTasks();
    $('nqa-count').textContent = `${tasks.length} / ${video?.tasks.length || 0} QA`;
    $('nqa-tasks').replaceChildren();
    for (const item of tasks) {
      const button = make('button', undefined, 'nqa-task');
      button.type = 'button'; button.dataset.task = item.task_id;
      button.setAttribute('aria-pressed', String(task === item));
      button.append(make('span', `${item.task_id.split('_').pop()} · ${typeName(item.qa_type)} · ${clock(item.query_time)} → ${clock(item.answer_time)}`, 'nqa-meta'), make('span', item.question[language()]));
      button.onclick = () => selectTask(item);
      $('nqa-tasks').append(button);
    }
    if (!tasks.length) list.append(make('p', local('没有匹配的问答。', 'No matching QA.'), 'nqa-meta'));
    list.scrollTop = previousScroll;
    const active = list.querySelector('[aria-pressed=true]');
    if (active) {
      const bounds = list.getBoundingClientRect(), selected = active.getBoundingClientRect();
      if (selected.bottom > bounds.bottom) list.scrollTop += selected.bottom - bounds.bottom;
      else if (selected.top < bounds.top) list.scrollTop -= bounds.top - selected.top;
    }
  }
  function selectTask(next) {
    task = next; decision = 0;
    $('nqa-evidence').open = false;
    renderList(); renderTask();
    if (task) seek(task.query_time);
    syncURL();
  }
  function filter() {
    const tasks = filteredTasks();
    if (!tasks.includes(task)) selectTask(tasks[0] || null);
    else renderList();
  }
  function renderDecision() {
    if (!task) return;
    const current = task.decisions[decision];
    const badge = $('nqa-decision-status');
    badge.textContent = current.status;
    badge.className = current.status === 'WAIT' ? 'nqa-wait' : 'nqa-answer';
    $('nqa-prefix').textContent = `${local('标注决策时刻', 'Annotated decision time')} ${current.time}s · ${local('输入视频前缀', 'Video input prefix')} [0, ${current.time}]s`;
  }
  function renderTask() {
    $('nqa-detail').hidden = !task;
    if (!task) return;
    $('nqa-task-id').textContent = `${task.task_id} · ${typeName(task.qa_type)}`;
    $('nqa-source-kind').textContent = task.subset === 'visual_core'
      ? local('画面为主', 'Primarily visual') : local('文字 / 标注辅助', 'Text / overlay assisted');
    // English is always visible, including in Chinese display mode.
    $('nqa-question-en').textContent = task.question.en;
    $('nqa-question-zh').textContent = task.question.zh;
    $('nqa-answer-en').textContent = task.answer.en;
    $('nqa-answer-zh').textContent = task.answer.zh;
    section.querySelectorAll('.nqa-translation').forEach(node => { node.hidden = language() !== 'zh'; });
    $('nqa-query').textContent = `${local('提问', 'Query')} ${clock(task.query_time)} (${task.query_time}s)`;
    $('nqa-answer-time').textContent = `${local('回答', 'Answer')} ${clock(task.answer_time)} (${task.answer_time}s)`;
    const choices = $('nqa-decisions');
    choices.replaceChildren();
    task.decisions.forEach((item, index) => {
      const option = make('option', `${item.time}s · ${item.status}`);
      option.value = String(index); choices.append(option);
    });
    choices.value = String(decision);
    renderDecision();
    const frames = $('nqa-evidence-frames');
    frames.replaceChildren();
    for (const frame of task.evidence) {
      const figure = make('figure');
      const link = make('a'); link.href = frame.url; link.target = '_blank'; link.rel = 'noopener';
      const image = make('img'); image.src = frame.url; image.loading = 'lazy';
      image.alt = `${local('证据帧', 'Evidence frame')} ${frame.time}s`;
      const role = frame.role === 'trigger_absent' ? local('触发前', 'Before trigger') : local('答案依据', 'Answer support');
      link.append(image); figure.append(link, make('figcaption', `${frame.time}s · ${role}`));
      frames.append(figure);
    }
  }
  function renderLanguage() {
    section.querySelectorAll('[data-nqa-en]').forEach(node => { node.textContent = node.dataset[language() === 'zh' ? 'nqaZh' : 'nqaEn']; });
    $('nqa-jump').textContent = local('新版本 QA ↓', 'New QA ↓');
    $('nqa-search').placeholder = local('搜索中英文问题或答案', 'Search questions or answers');
    const select = $('nqa-type'), saved = select.value;
    select.replaceChildren(make('option', local('全部类型', 'All types'))); select.firstChild.value = '';
    for (const type of Object.keys(types)) { const option = make('option', typeName(type)); option.value = type; select.append(option); }
    select.value = saved;
    if (data) {
      $('nqa-stats').textContent = `${data.counts.videos} ${local('个训练视频', 'training videos')} · ${data.counts.qa} QA · ${data.counts.decisions} ${local('条决策样本', 'decisions')}`;
      renderCards(); renderList(); renderTask(); renderClock();
      if (video) player.setAttribute('aria-label', video.title[language()]);
    } else if (loadError) {
      $('nqa-loading').textContent = local('新版本 QA 加载失败，请重新构建展示数据。', 'New QA could not load. Rebuild the viewer data.');
    }
  }
  $('nqa-query').onclick = () => { decision = 0; renderTask(); seek(task.query_time); };
  $('nqa-answer-time').onclick = () => { decision = task.decisions.length - 1; renderTask(); seek(task.answer_time); };
  $('nqa-decisions').onchange = () => { decision = Number($('nqa-decisions').value); renderDecision(); seek(task.decisions[decision].time); };
  $('nqa-search').oninput = filter;
  $('nqa-type').onchange = filter;
  $('nqa-subset').onchange = filter;
  $('language-select').addEventListener('change', () => { renderLanguage(); syncURL(); });
  renderLanguage();
  (async () => {
    try {
      const response = await fetch('new-qa.json', {cache: 'no-store'});
      if (!response.ok) throw Error(`HTTP ${response.status}`);
      data = await response.json();
      $('nqa-review-link').href = data.review_url;
      $('nqa-training-link').href = data.samples_url;
      $('nqa-loading').hidden = true; $('nqa-browser').hidden = false;
      const requested = new URLSearchParams(location.search).get('nqa_task');
      const selectedVideo = data.videos.find(v => v.tasks.some(t => t.task_id === requested)) || data.videos[0];
      selectVideo(selectedVideo, selectedVideo.tasks.find(t => t.task_id === requested));
      renderLanguage();
    } catch (error) {
      loadError = error; $('nqa-browser').hidden = true; $('nqa-loading').hidden = false; renderLanguage();
      console.error('New QA:', error);
    }
  })();
})();
