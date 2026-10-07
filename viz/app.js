const CONDITION_LABELS = {
  before_with_asr: ['before_with_asr', 'video up to sentence start, earlier ASR visible'],
  through_with_asr: ['through_with_asr', 'video through sentence end, earlier ASR visible'],
  before_mask_asr: ['before_mask_asr', 'video up to sentence start, earlier ASR MASKED'],
};
let DATA = null;
let sampling = 'uniform';
let currentSid = null;
let stopAt = null;
const video = document.getElementById('video');
const sentenceSelect = document.getElementById('sentence-select');
const samplingSelect = document.getElementById('sampling-select');
const columns = document.getElementById('columns');
const targetText = document.getElementById('target-text');
const meta = document.getElementById('meta');
const tlSentence = document.getElementById('tl-sentence');
const tlPlayhead = document.getElementById('tl-playhead');
const tlFrames = document.getElementById('tl-frames');
function fmt(t) { return Number(t).toFixed(2) + 's'; }
function pct(t) { return (100 * t / DATA.duration) + '%'; }

async function load() {
  DATA = await (await fetch('data.json')).json();
  video.src = DATA.video_url;
  meta.textContent = DATA.video_id + '  |  ' + DATA.sentences.length + ' sentences  |  ' + DATA.duration.toFixed(1) + 's  |  frame budget ' + DATA.max_frames;
  DATA.sentences.forEach(function (s) {
    const o = document.createElement('option');
    o.value = s.sentence_id;
    o.textContent = '#' + s.sentence_id + '  [' + s.sentence_window[0].toFixed(1) + '-' + s.sentence_window[1].toFixed(1) + 's]  ' + s.target.slice(0, 48);
    sentenceSelect.appendChild(o);
  });
  selectSentence(DATA.sentences[0].sentence_id);
}
function sentenceById(sid) {
  return DATA.sentences.find(function (s) { return s.sentence_id === sid; });
}

function selectSentence(sid) {
  currentSid = sid;
  const s = sentenceById(sid);
  targetText.textContent = s.target;
  renderColumns(s);
  renderTimeline(s, 'before_with_asr');
}
function renderColumns(s) {
  columns.innerHTML = '';
  DATA.condition_order.forEach(function (cond) {
    const c = s.conditions[cond];
    const frames = c.frames[sampling];
    const label = CONDITION_LABELS[cond];
    const asrText = c.prompt_texts[0];
    const masked = c.historical_asr_masked;
    const col = document.createElement('div');
    col.className = 'col';
    col.id = 'col-' + cond;
    col.innerHTML =
      '<h2>' + label[0] + '</h2>' +
      '<p class="csub">' + label[1] + '</p>' +
      '<p class="kv">video window: <b>[' + c.video_window[0].toFixed(2) + ', ' + c.video_window[1].toFixed(2) + ']</b></p>' +
      '<p class="kv">sampled frames: <span class="framecount">' + frames.length + '</span></p>' +
      '<p class="kv">ASR masked: <b>' + masked + '</b></p>' +
      '<div class="asr' + (masked ? ' masked' : '') + '">' + asrText + '</div>' +
      '<button class="playbtn" data-end="' + c.video_window[1] + '">Play 0 to ' + c.video_window[1].toFixed(1) + 's</button>';
    col.querySelector('.playbtn').addEventListener('click', function () {
      setActive(cond);
      renderTimeline(s, cond);
      stopAt = parseFloat(this.getAttribute('data-end'));
      video.currentTime = 0;
      video.play();
    });
    col.addEventListener('mouseenter', function () { renderTimeline(s, cond); });
    columns.appendChild(col);
  });
}
function setActive(cond) {
  DATA.condition_order.forEach(function (c) {
    const el = document.getElementById('col-' + c);
    if (el) el.classList.toggle('active', c === cond);
  });
}

function renderTimeline(s, cond) {
  const c = s.conditions[cond];
  tlSentence.style.left = pct(s.sentence_window[0]);
  tlSentence.style.width = pct(s.sentence_window[1] - s.sentence_window[0]);
  tlFrames.innerHTML = '';
  c.frames[sampling].forEach(function (t) {
    const m = document.createElement('div');
    m.className = 'fmark';
    m.style.left = pct(t);
    m.title = fmt(t);
    tlFrames.appendChild(m);
  });
}
video.addEventListener('timeupdate', function () {
  tlPlayhead.style.left = pct(video.currentTime);
  if (stopAt !== null && video.currentTime >= stopAt) {
    video.pause();
    stopAt = null;
  }
});

sentenceSelect.addEventListener('change', function () {
  selectSentence(parseInt(this.value, 10));
});
samplingSelect.addEventListener('change', function () {
  sampling = this.value;
  selectSentence(currentSid);
});

load();
