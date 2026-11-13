(() => {
  'use strict';
  const container = document.getElementById('os-examples');
  const make = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  const time = value => value === null ? 'Not specified / 未标注' : `${value}s`;
  const labels = {
    silence: 'Continue observing / 继续观察',
    standby: 'Relevant clues; keep waiting / 相关线索出现，继续等待',
    response: 'Reference response / 目标回复',
    answer: 'Reference answer / 标注答案'
  };

  function render(example) {
    const record = example.record;
    const card = make('article', undefined, 'os-card');
    card.id = `os-${example.key}`;
    card.append(make('h3', example.title));
    card.append(make('div', `${record.source} · ${record.classification.task} · ${example.duration.toFixed(1)}s · ${(example.bytes / 1e6).toFixed(2)} MB`, 'os-meta'));
    const video = make('video');
    video.id = `os-video-${example.key}`;
    video.controls = true;
    video.preload = 'metadata';
    video.playsInline = true;
    video.src = example.video;
    video.poster = example.poster;
    video.setAttribute('aria-label', `${example.title} source video`);
    card.append(video);
    const status = make('p', '', 'os-clock');
    status.setAttribute('aria-live', 'off');
    card.append(status);
    video.addEventListener('play', () => {
      container.querySelectorAll('video').forEach(other => { if (other !== video) other.pause(); });
    });
    video.addEventListener('error', () => { status.textContent = 'Video could not load / 视频加载失败'; });
    const prompt = record.prompts[0];
    card.append(make('p', `QUESTION · ${time(prompt.at)}`, 'field-label'));
    card.append(make('p', prompt.text, 'os-question'));
    card.append(make('p', example.translation.question, 'os-translation'));
    card.append(make('p', example.note, 'os-note'));

    const rows = [];
    const jump = target => {
      video.pause();
      if (video.readyState >= 1) video.currentTime = target;
      else video.addEventListener('loadedmetadata', () => { video.currentTime = target; }, {once:true});
    };
    if (record.timeline.some(event => event.interval)) {
      card.append(make('p', 'ANNOTATION TIMELINE · 标注时间线（点击跳转）', 'field-label'));
      const strip = make('div', undefined, 'os-strip');
      for (const event of record.timeline) {
        const duration = event.interval ? event.interval[1] - event.interval[0] : 1;
        const button = make('button', event.type === 'response' ? `↑${event.at}s` : event.type, `os-segment os-${event.type}`);
        button.type = 'button';
        button.style.flex = `${Math.max(1, duration)} 1 0`;
        button.title = `${event.type}: ${event.interval ? event.interval.join('–') + 's' : time(event.at)}`;
        button.setAttribute('aria-label', `Seek to ${event.interval ? event.interval[0] : event.at} seconds, ${event.type}`);
        button.addEventListener('click', () => jump(event.interval ? event.interval[0] : event.at));
        strip.append(button);
      }
      card.append(strip, make('p', 'Gray: Silence · Amber: Standby · Green: Response / 灰：静默，黄：待命，绿：回复', 'os-legend'));
    }
    let answerIndex = 0;
    for (const event of record.timeline) {
      const row = make('div', undefined, 'os-event');
      if (event.interval || event.at !== null) {
        const target = event.interval ? event.interval[0] : event.at;
        const button = make('button', event.interval ? `${event.interval[0]}–${event.interval[1]}s` : `${event.at}s ↗`, 'os-time');
        button.type = 'button';
        button.setAttribute('aria-label', `Jump to ${target} seconds: ${event.type}`);
        button.addEventListener('click', () => jump(target));
        row.append(button);
      } else {
        row.append(make('span', 'Time: null', 'os-meta'));
      }
      const body = make('div');
      body.append(make('span', event.type.toUpperCase(), 'os-kind'));
      if (event.text) {
        body.append(make('p', event.text, 'os-answer'));
        const translated = example.translation.answers[answerIndex++];
        if (translated) body.append(make('p', translated, 'os-translation'));
      } else {
        body.append(make('p', labels[event.type] || event.type, 'os-translation'));
      }
      row.append(body);
      rows.push({row, event});
      card.append(row);
    }
    video.addEventListener('timeupdate', () => {
      status.textContent = `Video position / 当前播放：${video.currentTime.toFixed(1)}s / ${example.duration.toFixed(1)}s`;
      rows.forEach(({row, event}) => {
        const active = event.interval ? video.currentTime >= event.interval[0] && video.currentTime < event.interval[1]
          : event.at !== null && video.currentTime >= event.at && video.currentTime < event.at + 1;
        row.classList.toggle('os-active', active);
      });
    });
    const links = make('div', undefined, 'os-links');
    for (const [label, href] of [['Original QA JSON / 原始标注', example.record_url], ['Video file / 视频文件', example.video], ['Source manifest / 来源清单', example.manifest_url]]) {
      const link = make('a', label);
      link.href = href;
      link.target = '_blank';
      link.rel = 'noopener';
      links.append(link);
    }
    card.append(links);
    const details = make('details');
    details.append(make('summary', 'Raw record & provenance / 展开原始数据与来源'));
    details.append(make('p', `${example.annotation_file} · line ${example.annotation_line}`, 'os-meta'));
    details.append(make('p', 'SHA256 matched to the official video manifest. Source annotations are displayed without independently validating every event. / 视频哈希与官方清单一致；标注原样展示，未逐项独立复核。', 'os-meta'));
    details.append(make('pre', JSON.stringify(record, null, 2)));
    card.append(details);
    return card;
  }

  fetch('data/onestreamer/examples.json', {cache:'no-store'})
    .then(response => { if (!response.ok) throw Error(`HTTP ${response.status}`); return response.json(); })
    .then(examples => {
      container.replaceChildren(...examples.map(render));
      if (location.hash === '#onestreamer-examples') {
        // The existing explorer loads its catalog asynchronously and changes
        // the height above this section. Jump after that layout is complete.
        const select = document.getElementById('video-select');
        const scroll = () => requestAnimationFrame(() => document.getElementById('onestreamer-examples').scrollIntoView());
        if (!select || !select.disabled) scroll();
        else {
          const observer = new MutationObserver(() => {
            if (!select.disabled) { observer.disconnect(); scroll(); }
          });
          observer.observe(select, {attributes:true, attributeFilter:['disabled']});
        }
      }
    })
    .catch(error => { container.replaceChildren(make('p', `Could not load examples / 样例加载失败：${error.message}`)); });
})();
