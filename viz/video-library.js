/* Complete video and existing Stage-data browser; all source text stays unchanged. */
(() => {
  const byId = id => document.getElementById(id);
  const make = (tag, value, className) => {
    const node = document.createElement(tag);
    if (value !== undefined) node.textContent = value;
    if (className) node.className = className;
    return node;
  };
  const localized = (zh, en) => displayLanguage === 'zh' ? zh : en;
  const clock = seconds => {
    const value = Math.max(0, Number(seconds) || 0);
    return `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;
  };
  const interval = window => `${clock(window[0])}–${clock(window[1])}`;
  const ratingName = value => ({high: localized('高', 'High'), medium: localized('中', 'Medium'), low: localized('低', 'Low')})[value] || value;
  const typeName = value => ({ultrasound_tutorial: localized('超声教程', 'Ultrasound tutorial'), ppt_lecture: localized('PPT 讲座', 'Slide lecture'), mixed: localized('混合视频', 'Mixed'), pure_ultrasound: localized('纯超声', 'Pure ultrasound'), uncertain: localized('待确定', 'Uncertain')})[value] || value;
  const sourceText = value => typeof contentZh === 'object' && displayLanguage === 'zh' ? (contentZh[value] || value) : value;
  let library = null;
  let selectedId = null;
  let selectedRecord = 0;
  let view = 'stages';

  function urlState() {
    const url = new URL(location.href);
    url.searchParams.set('view', view);
    if (selectedId) url.searchParams.set('library_video', selectedId);
    url.searchParams.set('library_stage', byId('library-stage').value);
    if (byId('library-type').value) url.searchParams.set('library_type', byId('library-type').value); else url.searchParams.delete('library_type');
    if (byId('library-value').value) url.searchParams.set('library_value', byId('library-value').value); else url.searchParams.delete('library_value');
    if (byId('library-search').value) url.searchParams.set('library_q', byId('library-search').value); else url.searchParams.delete('library_q');
    if (byId('library-collection').value) url.searchParams.set('library_collection', byId('library-collection').value); else url.searchParams.delete('library_collection');
    history.replaceState(null, '', url);
  }

  function setView(next, update = true) {
    view = next;
    const showLibrary = next === 'library';
    byId('video-library').hidden = !showLibrary;
    document.body.classList.toggle('library-active', showLibrary);
    for (const [id, active] of [['tab-stages', !showLibrary], ['tab-library', showLibrary]]) {
      const tab = byId(id);
      tab.classList.toggle('active', active);
      tab.setAttribute('aria-selected', String(active));
    }
    if (showLibrary) byId('video').pause(); else byId('library-video').pause();
    if (update) urlState();
  }

  function allStageRows(video) {
    const stage = byId('library-stage').value;
    const rows = video.stage_data[stage];
    if (stage !== 'stage3') return rows;
    const collection = byId('library-collection').value;
    return collection ? rows.filter(row => row.collection === collection) : rows;
  }

  function appendText(parent, label, value) {
    if (value === null || value === undefined || value === '') return;
    const field = make('div', undefined, 'library-field');
    field.append(make('span', label, 'library-field-label'), make('p', sourceText(String(value)), 'library-field-value'));
    parent.append(field);
  }

  function seek(seconds) {
    const player = byId('library-video');
    player.currentTime = Math.max(0, Number(seconds) || 0);
    player.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  }

  function seekButton(parent, label, seconds) {
    const button = make('button', label, 'library-seek');
    button.type = 'button';
    button.onclick = () => seek(seconds);
    parent.append(button);
  }

  function renderRecord() {
    const video = library?.videos.find(v => v.video_id === selectedId);
    const host = byId('library-record-detail');
    host.replaceChildren();
    if (!video) return;
    const stage = byId('library-stage').value;
    const rows = allStageRows(video);
    const row = rows[selectedRecord];
    if (!row) return;
    if (stage === 'stage1') {
      host.append(make('h4', `${localized('句子', 'Sentence')} #${row.sentence_id} · ${interval(row.window)}`));
      seekButton(host, localized('跳到目标句', 'Go to target sentence'), row.window[0]);
      appendText(host, localized('视频输入窗口', 'Input video window'), interval(row.video_window));
      appendText(host, localized('监督目标', 'Target'), row.target);
      appendText(host, localized('历史 ASR', 'Historical ASR'), row.historical_asr);
      appendText(host, localized('训练条件', 'Training conditions'), row.conditions.join(' · '));
    } else if (stage === 'stage2') {
      host.append(make('h4', `${localized('分钟块', 'Minute block')} · ${interval(row.block_window)}`));
      row.local_sub_summaries.forEach((local, index) => {
        const card = make('div', undefined, 'library-local');
        seekButton(card, `Local ${index + 1} · ${interval(local.window)}`, local.window[0]);
        appendText(card, localized('监督文本', 'Supervision'), local.local_summary_target);
        host.append(card);
      });
      if (row.block_summary_target) {
        const block = make('div', undefined, 'library-local');
        seekButton(block, `Block · ${interval(row.block_window)}`, row.block_window[0]);
        appendText(block, localized('监督文本', 'Supervision'), row.block_summary_target);
        host.append(block);
      }
      if (row.global_summary_target) {
        const global = make('div', undefined, 'library-local');
        seekButton(global, `Global · ${interval([0, row.block_window[1]])}`, 0);
        appendText(global, localized('监督文本', 'Supervision'), row.global_summary_target);
        host.append(global);
      }
    } else {
      host.append(make('h4', `${row.qa_type.replaceAll('_', ' ')} · ${row.task_id}`));
      const times = make('div', undefined, 'library-time-actions');
      seekButton(times, `${localized('提问', 'Query')} ${clock(row.query_time)}`, row.query_time);
      if (row.answer_time !== null) seekButton(times, `${localized('回答', 'Answer')} ${clock(row.answer_time)}`, row.answer_time);
      host.append(times);
      appendText(host, localized('问题', 'Question'), row.question);
      appendText(host, localized('参考答案', 'Reference answer'), row.reference_answer);
      appendText(host, 'WAIT / ANSWER', row.response_group);
      appendText(host, localized('提问时缺少的证据', 'Missing at query time'), row.query_time_missing_evidence);
      appendText(host, localized('回答时新增的证据', 'New evidence at answer time'), row.answer_time_new_evidence);
      if (row.historical_anchor) appendText(host, localized('历史证据范围', 'Historical evidence'), interval([row.historical_anchor.start, row.historical_anchor.end]));
      appendText(host, localized('证据模态', 'Evidence modality'), row.evidence_modality);
      appendText(host, localized('审核状态', 'Review status'), row.review_status);
    }
  }

  function renderData() {
    const video = library?.videos.find(v => v.video_id === selectedId);
    const list = byId('library-data-list');
    list.replaceChildren();
    byId('library-record-detail').replaceChildren();
    if (!video) return;
    const stage = byId('library-stage').value;
    const collections = byId('library-collection');
    const previous = collections.value;
    collections.replaceChildren();
    const availableCollections = [...new Set(video.stage_data.stage3.map(row => row.collection))];
    if (stage === 'stage3' && availableCollections.length > 1) {
      for (const {id, title} of library.stage3_collections) {
        if (availableCollections.includes(id)) {
          const option = make('option', title); option.value = id; collections.append(option);
        }
      }
      collections.value = availableCollections.includes(previous) ? previous : availableCollections[0];
      collections.parentElement.hidden = false;
    } else {
      if (availableCollections.length) {
        const id = availableCollections[0];
        const option = make('option', library.stage3_collections.find(c => c.id === id)?.title || id);
        option.value = id; collections.append(option);
      }
      collections.parentElement.hidden = true;
    }
    const rows = allStageRows(video);
    byId('library-data-heading').textContent = `Stage ${stage.slice(-1)} · ${localized('时间段数据', 'time-aligned data')}`;
    byId('library-data-count').textContent = rows.length
      ? `${rows.length} ${localized('组记录', 'records')} · ${localized('点击时间段查看监督文本并跳转播放', 'select a time range to inspect supervision and seek')}`
      : localized('这个视频暂无本地可用的 Stage 数据；评级仍显示，但没有生成占位样本。', 'No local Stage data is available for this video. The original rating remains visible; no placeholder samples were created.');
    selectedRecord = Math.max(0, Math.min(selectedRecord, rows.length - 1));
    rows.forEach((row, index) => {
      let title;
      if (stage === 'stage1') title = `#${row.sentence_id} · ${interval(row.window)} · ${row.target}`;
      else if (stage === 'stage2') title = `Block ${index + 1} · ${interval(row.block_window)}`;
      else title = `${clock(row.query_time)}${row.answer_time !== null && row.answer_time !== row.query_time ? ` → ${clock(row.answer_time)}` : ''} · ${row.qa_type.replaceAll('_', ' ')} · ${row.question}`;
      const button = make('button', title, 'library-record');
      button.type = 'button';
      button.classList.toggle('active', index === selectedRecord);
      button.onclick = () => {
        selectedRecord = index;
        for (const node of list.children) node.classList.remove('active');
        button.classList.add('active');
        renderRecord();
        seek(stage === 'stage1' ? row.window[0] : stage === 'stage2' ? row.block_window[0] : row.query_time);
        urlState();
      };
      list.append(button);
    });
    renderRecord();
  }

  function renderVideo() {
    const video = library?.videos.find(v => v.video_id === selectedId);
    const info = byId('library-video-info');
    info.replaceChildren();
    if (!video) {
      info.append(make('p', localized('没有符合条件的视频。', 'No videos match these filters.')));
      byId('library-video').removeAttribute('src');
      byId('library-video').load();
      renderData();
      return;
    }
    const stage = byId('library-stage').value;
    const rating = video.stage_value[stage];
    info.append(make('h3', video.video_id));
    info.append(make('p', `${video.split.toUpperCase()} · ${typeName(video.video_type)} · ${clock(video.duration)} · ${video.topics.join(', ')}`, 'library-video-meta'));
    appendText(info, localized('视频类型原因', 'Video type reason'), video.video_type_reason);
    const value = make('div', undefined, 'library-rating');
    value.append(make('strong', `Stage ${stage.slice(-1)} · ${ratingName(rating.value)}`, `value-${rating.value}`));
    appendText(value, localized('评级原因', 'Rating reason'), rating.reason);
    info.append(value);
    const other = make('div', undefined, 'library-other-ratings');
    for (const s of ['stage1', 'stage2', 'stage3']) {
      other.append(make('span', `Stage ${s.slice(-1)}: ${ratingName(video.stage_value[s].value)}`, s === stage ? 'selected' : ''));
    }
    info.append(other);
    const player = byId('library-video');
    const source = new URL(video.video_url, location.href).href;
    if (player.src !== source) { player.pause(); player.src = source; player.load(); }
    renderData();
  }

  function renderList() {
    if (!library) return;
    const q = byId('library-search').value.trim().toLowerCase();
    const type = byId('library-type').value;
    const stage = byId('library-stage').value;
    const value = byId('library-value').value;
    const filtered = library.videos.filter(video =>
      (!type || video.video_type === type) &&
      (!value || video.stage_value[stage].value === value) &&
      (!q || `${video.video_id} ${video.topics.join(' ')}`.toLowerCase().includes(q)));
    byId('library-count').textContent = `${filtered.length} / ${library.videos.length} ${localized('个视频', 'videos')}`;
    if (!filtered.some(video => video.video_id === selectedId)) selectedId = filtered[0]?.video_id || null;
    const list = byId('library-rows');
    list.replaceChildren();
    const fragment = document.createDocumentFragment();
    for (const video of filtered) {
      const button = make('button', undefined, 'library-row');
      button.type = 'button';
      button.classList.toggle('active', video.video_id === selectedId);
      button.setAttribute('aria-pressed', String(video.video_id === selectedId));
      const title = make('span', video.video_id, 'library-row-title');
      const rating = make('span', ratingName(video.stage_value[stage].value), `library-value value-${video.stage_value[stage].value}`);
      const meta = make('span', `${typeName(video.video_type)} · ${video.split.toUpperCase()} · ${clock(video.duration)}`, 'library-row-meta');
      button.append(title, rating, meta);
      button.onclick = () => {
        selectedId = video.video_id;
        selectedRecord = 0;
        renderList();
        urlState();
      };
      fragment.append(button);
    }
    list.append(fragment);
    renderVideo();
    urlState();
  }

  function renderLanguage() {
    if (!library) return;
    const type = byId('library-type');
    const selectedType = type.value;
    type.replaceChildren();
    let option = make('option', localized('全部类型', 'All types')); option.value = ''; type.append(option);
    for (const name of [...new Set(library.videos.map(video => video.video_type))].sort()) {
      option = make('option', typeName(name)); option.value = name; type.append(option);
    }
    type.value = selectedType;
    const values = byId('library-value');
    const selectedValue = values.value;
    values.replaceChildren();
    for (const value of ['', 'high', 'medium', 'low']) {
      option = make('option', value ? ratingName(value) : localized('全部价值', 'All values'));
      option.value = value;
      values.append(option);
    }
    values.value = selectedValue;
    byId('library-total').textContent = `${library.videos.length} ${localized('个有原始标签的视频', 'originally labelled videos')}`;
    byId('library-provenance').textContent = localized(
      '原评级基于无音频、无 ASR 的抽样视觉审阅；原因保留原文。此页展示已有监督数据，不是模型输出。',
      'Original ratings came from sampled silent visual review, without audio or ASR. Reasons remain in their source language. This page shows existing supervision, not model output.');
    renderList();
  }

  byId('tab-stages').onclick = () => setView('stages');
  byId('tab-library').onclick = () => setView('library');
  byId('library-search').oninput = () => { selectedRecord = 0; renderList(); };
  byId('library-type').onchange = () => { selectedRecord = 0; renderList(); };
  byId('library-value').onchange = () => { selectedRecord = 0; renderList(); };
  byId('library-stage').onchange = () => { selectedRecord = 0; renderList(); };
  byId('library-collection').onchange = () => { selectedRecord = 0; renderData(); urlState(); };
  byId('language-select').addEventListener('change', () => renderLanguage());

  (async () => {
    try {
      const response = await fetch('video_library.json', {cache: 'no-store'});
      if (!response.ok) throw new Error('Run scripts/viz_demo.sh to build video_library.json');
      library = await response.json();
      const params = new URLSearchParams(location.search);
      byId('library-stage').value = ['stage1', 'stage2', 'stage3'].includes(params.get('library_stage')) ? params.get('library_stage') : 'stage3';
      selectedId = library.videos.some(v => v.video_id === params.get('library_video')) ? params.get('library_video') : (library.videos.some(v => v.video_id === 'P3FoSFIrfQA') ? 'P3FoSFIrfQA' : library.videos[0]?.video_id);
      byId('library-search').value = params.get('library_q') || '';
      renderLanguage();
      byId('library-type').value = params.get('library_type') || '';
      byId('library-value').value = params.get('library_value') || '';
      renderList();
      const requestedCollection = params.get('library_collection');
      if (requestedCollection && [...byId('library-collection').options].some(o => o.value === requestedCollection)) {
        byId('library-collection').value = requestedCollection;
        renderData();
      }
      setView(params.get('view') === 'library' ? 'library' : 'stages');
    } catch (error) {
      byId('library-count').textContent = String(error);
      setView(new URLSearchParams(location.search).get('view') === 'library' ? 'library' : 'stages');
    }
  })();
})();
