'use strict';
// Match viewer: video + tracking overlay + 2D pitch + label and clue timelines, all on one frame.
// Data comes from viewer/cache (see viewer/build_cache.py for the encodings).

const COLOR = { H: '#3b82f6', A: '#ef4444', D: '#9ca3af', clue: '#f59e0b', seen: '#22c55e', ball: '#facc15',
  looksDead: 'rgba(156,163,175,0.4)', Hsoft: 'rgba(59,130,246,0.45)', Asoft: 'rgba(239,68,68,0.45)',
  ignored: '#6b7280', ungraded: 'rgba(255,255,255,0.05)', grid: '#262b32', text: '#e5e7ea', muted: '#8f98a3' };
const LABEL_COLOR = [null, COLOR.H, COLOR.A, COLOR.D];
const LABEL_TEXT = ['–', 'H', 'A', 'D'];
// Kick-off frames, published in DATA.md and FAQ.md. Used only for the clock readout.
let KICKOFF = { 1: 11185, 2: 113742 };  // a bench match's meta.json gives its own
const ON_BALL_DISTANCE = 2.0;  // metres, same as model v1's clue A
const MARGIN = 4;              // metres of grass drawn around the pitch map
const LANE_W = 150;            // px reserved for lane names on the timelines

const PAUSE_BEFORE = 6;        // frames before a watched change where "slow down and pause" stops
const SLOW_AFTER = 25;         // frames after a change that stay in slow motion
const CHANGE_NOTE = 25;        // frames after a change that its badge stays up (the ring is that frame only)

const $ = (id) => document.getElementById(id);
// Which match: BAR-ATH (with video) by default; ?match=<id> opens a bench match, which has no video.
const MATCH = new URLSearchParams(location.search).get('match');
const BASE = MATCH ? `/cache/bench/${encodeURIComponent(MATCH)}/` : '/cache/';

// Stands in for the <video> element on matches without video: the same few properties, methods and
// events the page uses (play, pause, currentTime, playbackRate, muted, requestVideoFrameCallback),
// driven by the screen's refresh instead of a decoder.
class Clock extends EventTarget {
  constructor() { super(); this.paused = true; this.playbackRate = 1; this.muted = true; this.duration = Infinity; this._t = 0; this._last = null; this._callbacks = []; }
  get currentTime() { return this._t; }
  set currentTime(t) { this._t = Math.max(0, Math.min(t, this.duration)); this._last = null; queueMicrotask(() => this.dispatchEvent(new Event('seeked'))); }
  play() {
    if (this.paused) { this.paused = false; this._last = null; this.dispatchEvent(new Event('play')); requestAnimationFrame((n) => this._tick(n)); }
    return Promise.resolve();
  }
  pause() { if (!this.paused) { this.paused = true; this.dispatchEvent(new Event('pause')); } }
  requestVideoFrameCallback(callback) { this._callbacks.push(callback); }
  _tick(now) {
    if (this.paused) return;
    if (this._last !== null) this._t = Math.min(this.duration, this._t + ((now - this._last) / 1000) * this.playbackRate);
    this._last = now;
    const callbacks = this._callbacks; this._callbacks = [];
    for (const callback of callbacks) callback(now, { mediaTime: this._t });
    if (this._t >= this.duration) this.pause(); else requestAnimationFrame((n) => this._tick(n));
  }
}
const video = MATCH ? new Clock() : $('video');
if (MATCH) { $('video').removeAttribute('src'); $('video').load(); }   // don't download our video for a bench match
const TEAM_NAME = MATCH ? ['home', 'away'] : ['Barça', 'Athletic'];   // bench matches: their own home and away
const S = {
  frame: 0, zoom: 30, playing: false, notes: [],
  // Change watch: which label lanes to watch, and what to do as a change approaches.
  watch: { lanes: new Set(), mode: 'slow', slowRate: 0.1, lead: 2, userRate: 1, pausedFor: -1 },
};

// ------------------------------------------------------------------------------------------
// Loading and data access
// ------------------------------------------------------------------------------------------

async function buildMatchPicker(meta) {
  // The header's title becomes a picker: BAR-ATH plus every bench match with a cache.
  let bench = [];
  try { bench = await fetch('/cache/bench/index.json').then((r) => (r.ok ? r.json() : [])); } catch { bench = []; }
  if (!bench.length) return;
  const select = document.createElement('select');
  select.id = 'match';
  select.title = 'Match';
  select.append(new Option('BAR – ATH (ours, with video)', ''));
  for (const m of bench) select.append(new Option(`${m.id} · ${m.title}`, m.id));
  select.value = MATCH || '';
  select.addEventListener('change', () => { location.search = select.value ? `?match=${encodeURIComponent(select.value)}` : ''; });
  const brand = document.querySelector('.brand');
  brand.textContent = '';
  brand.append(select, Object.assign(document.createElement('span'), { textContent: MATCH ? 'bench match: true labels, no video' : 'match viewer' }));
}

function addCleanToggle() {
  const label = document.createElement('label');
  const input = Object.assign(document.createElement('input'), { type: 'checkbox', id: 'l-clean' });
  input.addEventListener('change', () => {
    const t = S.tracks[input.checked ? 'clean' : 'damaged'];
    Object.assign(S, { players: t.players, ball: t.ball });
    Object.assign(S.meta, { slots: t.slots, team: t.team });
    render();
  });
  label.append(input, ' clean tracking (as published, before our damage)');
  $('l-numbers').closest('label').after(label);
}

async function load() {
  const meta = await fetch(`${BASE}meta.json`).then((r) => r.json());
  const hasVideo = meta.video !== false;
  const names = ['players', 'ball', 'labels', 'clues'].concat(hasVideo ? ['homography'] : [])
    .concat(meta.clean ? ['players_clean', 'ball_clean'] : []);
  const buffers = Object.fromEntries(await Promise.all(names.map((n) => fetch(`${BASE}${n}.bin`).then((r) => {
    if (!r.ok) throw new Error(`missing cache file ${n}.bin`);
    return r.arrayBuffer().then((b) => [n, b]);
  }))));
  Object.assign(S, {
    meta, F: meta.frames, fps: meta.fps, L: meta.pitch[0], W: meta.pitch[1],
    players: new Int16Array(buffers.players), ball: new Int16Array(buffers.ball),
    H: hasVideo ? new Float32Array(buffers.homography) : null,
    labels: new Uint8Array(buffers.labels), clues: new Uint8Array(buffers.clues),
  });
  if (meta.kickoff) KICKOFF = meta.kickoff;
  if (!hasVideo) { document.body.classList.add('no-video'); video.duration = (S.F - 1) / S.fps; }
  if (meta.clean) {
    // A bench match: the damaged tracking the models see, and the clean original, one toggle apart.
    S.tracks = {
      damaged: { players: S.players, ball: S.ball, slots: meta.slots, team: meta.team },
      clean: { players: new Int16Array(buffers.players_clean), ball: new Int16Array(buffers.ball_clean),
               slots: meta.clean.slots, team: meta.clean.team },
    };
    addCleanToggle();
  }
  await buildMatchPicker(meta);
  S.graded = new Uint8Array(S.F);
  for (const [, first, last] of meta.halves) S.graded.fill(1, first, last + 1);
  S.lanes = buildLanes(meta);
  for (const lane of S.lanes) if (lane.kind === 'label') lane.changes = laneChanges(lane);
  buildWatchChips();
  await loadNotes();
  $('loading').hidden = true;
  resize();
  seek(meta.halves[0][1]);
}

// ------------------------------------------------------------------------------------------
// Change watch: slow down (and optionally pause) just before a watched lane changes label
// ------------------------------------------------------------------------------------------

function laneChanges(lane) {
  // Frames where the label differs from the frame before, between two real labels (not X/none).
  const out = [];
  let previous = lane.value(0);
  for (let f = 1; f < S.F; f++) {
    const v = lane.value(f);
    if (v !== previous && v && previous) out.push(f);
    previous = v;
  }
  return Int32Array.from(out);
}

function upperBound(array, f) {
  // Index of the first element > f.
  let lo = 0, hi = array.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (array[mid] <= f) lo = mid + 1; else hi = mid; }
  return lo;
}

const watchedLanes = () => S.lanes.filter((l) => l.kind === 'label' && S.watch.lanes.has(l.name));

function nextWatched(f) {
  let best = Infinity;
  for (const lane of watchedLanes()) {
    const i = upperBound(lane.changes, f);
    if (i < lane.changes.length) best = Math.min(best, lane.changes[i]);
  }
  return best;
}

function previousWatched(f) {
  let best = -Infinity;
  for (const lane of watchedLanes()) {
    const i = upperBound(lane.changes, f);
    if (i > 0) best = Math.max(best, lane.changes[i - 1]);
  }
  return best;
}

function describeChange(f) {
  return watchedLanes().filter((l) => l.value(f) !== l.value(f - 1))
    .map((l) => `${l.name} ${LABEL_TEXT[l.value(f - 1)]}→${LABEL_TEXT[l.value(f)]}`).join(', ');
}

function buildWatchChips() {
  const box = $('watch-lanes');
  box.textContent = '';
  for (const lane of S.lanes.filter((l) => l.kind === 'label')) {
    const label = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = !lane.name.startsWith('baseline');  // truth and our models by default
    if (input.checked) S.watch.lanes.add(lane.name);
    input.addEventListener('change', () => {
      if (input.checked) S.watch.lanes.add(lane.name); else S.watch.lanes.delete(lane.name);
      S.watch.pausedFor = -1;
      render();
    });
    label.append(input, lane.name.replace(' (first 5 min)', ''));
    box.append(label);
    $('frame-lane').append(new Option(lane.name.replace(' (first 5 min)', ''), lane.name));
  }
}

function drawFrameState(f) {
  // Ring around the video in the chosen lane's label colour, and the change-frame flash.
  const wrap = document.querySelector('.video-wrap');
  const lane = S.lanes.find((l) => l.name === $('frame-lane').value);
  wrap.style.setProperty('--frame', (lane && LABEL_COLOR[lane.value(f)]) || 'transparent');
  const last = previousWatched(f);  // latest watched change at or before this frame
  const onChange = last === f && $('watch-flash').checked;
  wrap.classList.toggle('flash', onChange);
  const badge = $('change-badge');
  badge.hidden = !$('watch-flash').checked || !(f - last <= CHANGE_NOTE);
  badge.classList.toggle('now', onChange);
  if (!badge.hidden) {
    badge.textContent = onChange ? `change at this frame (${f}): ${describeChange(f)}`
      : `changed ${f - last} frame${f - last === 1 ? '' : 's'} ago (${last}): ${describeChange(last)}`;
  }
}

function watchTick(f) {
  const w = S.watch;
  const next = nextWatched(f);
  const ahead = next - f;
  const since = f - previousWatched(f);
  $('next-change').textContent = Number.isFinite(next)
    ? `next change in ${(ahead / S.fps).toFixed(1)} s · ${describeChange(next)}` : 'no more watched changes';
  const slowZone = w.mode !== 'off' && (ahead <= w.lead * S.fps || since < SLOW_AFTER);
  $('slow-badge').hidden = !(slowZone && !video.paused);
  if (video.paused) return;
  const rate = slowZone ? Math.min(w.slowRate, w.userRate) : w.userRate;
  if (video.playbackRate !== rate) video.playbackRate = rate;
  if (w.mode === 'pause' && ahead <= PAUSE_BEFORE && w.pausedFor !== next) {
    w.pausedFor = next;  // pause once per change; pressing play continues through it slowly
    video.pause();
  }
}

function jumpToChange(direction) {
  // Land `lead` seconds before the next / previous watched change, so pressing play runs into it.
  const lead = S.watch.lead * S.fps;
  const anchor = S.frame + lead;
  const change = direction > 0 ? nextWatched(anchor) : previousWatched(anchor - 1);
  if (!Number.isFinite(change)) return;
  video.pause();
  seek(change - lead);
}

// ------------------------------------------------------------------------------------------
// Notes: what Jose sees, pinned to a frame. For understanding only (never labels; RULES.md).
// ------------------------------------------------------------------------------------------

async function loadNotes() {
  try { S.notes = await fetch('/notes').then((r) => r.json()); } catch { S.notes = []; }
  renderNotes();
}

function saveNotes() {
  fetch('/notes', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(S.notes) });
}

function shortClock(f) {
  if (f >= S.meta.halves[0][1] && f <= S.meta.halves[0][2]) return `1h ${fmt((f - KICKOFF[1]) / S.fps)}`;
  if (f >= S.meta.halves[1][1] && f <= S.meta.halves[1][2]) return `2h ${fmt(45 * 60 + (f - KICKOFF[2]) / S.fps)}`;
  return `frame ${f}`;
}

function renderNotes() {
  const list = $('note-list');
  list.textContent = '';
  S.notes.forEach((note, index) => {
    if ((note.match || 'BAR-ATH') !== (MATCH || 'BAR-ATH')) return;   // each match keeps its own notes
    const item = document.createElement('li');
    item.dataset.frame = note.frame;
    const time = Object.assign(document.createElement('span'), { className: 't', textContent: shortClock(note.frame) });
    const kind = Object.assign(document.createElement('span'), { className: 'kind', textContent: note.kind });
    const text = Object.assign(document.createElement('span'), { textContent: note.text });
    const remove = Object.assign(document.createElement('button'), { textContent: 'delete', title: 'Delete this note' });
    remove.addEventListener('click', (e) => { e.stopPropagation(); S.notes.splice(index, 1); saveNotes(); renderNotes(); render(); });
    item.addEventListener('click', () => { video.pause(); seek(note.frame); });
    item.append(time, kind, text, remove);
    list.append(item);
  });
}

function markNearNotes(f) {
  for (const item of $('note-list').children) item.classList.toggle('near', Math.abs(Number(item.dataset.frame) - f) <= S.fps);
}

function buildLanes(meta) {
  // Truth first, then our models (newest first), then the organisers' baselines (the floor first).
  const named = meta.labelLanes.map((name, index) => ({ name, index }));
  const truth = named.filter((l) => l.name.startsWith('truth'));
  const models = named.filter((l) => !l.name.startsWith('truth') && !l.name.startsWith('baseline')).reverse();
  const baselines = named.filter((l) => l.name.startsWith('baseline'))
    .sort((a, b) => (b.name === 'baseline_smooth50') - (a.name === 'baseline_smooth50'));
  const labelLanes = [...truth, ...models, ...baselines].map((l) => ({
    name: l.name, value: (f) => S.labels[l.index * S.F + f], color: (v) => LABEL_COLOR[v], kind: 'label' }));
  const clueLanes = meta.clueLanes.map((name, index) => ({
    name, kind: 'clue', value: (f) => S.clues[index * S.F + f],
    // v6's learned signals: a probability in %, drawn in bands (strong / weak / nothing)
    color: name === 'v4: P(dead)' ? (v) => (v > 80 ? COLOR.D : v > 50 ? COLOR.looksDead : null)
      : (name === 'v6: restart clue' || name === 'v7: PathCRF') ? (v) => (v === 0 ? null : v - 1 >= 80 ? COLOR.H : v - 1 > 50 ? COLOR.Hsoft
        : v - 1 <= 20 ? COLOR.A : v - 1 < 50 ? COLOR.Asoft : COLOR.muted)
      : name === 'ball visible' ? (v) => (v ? COLOR.seen : null)
      : (name.startsWith('A:') || name.startsWith('C:') || name === 'v8: keeper repaired' || name === 'v9: seen pass') ? (v) => [null, COLOR.H, COLOR.A][v]
        : name.endsWith('whistle') ? (v) => [null, COLOR.clue, COLOR.ignored][v]
          : (v) => (v ? COLOR.clue : null),
  }));
  return [...labelLanes, ...clueLanes];
}

function pos(array, i) {
  const x = array[i * 2];
  return x === S.meta.missing ? null : [x / S.meta.scale, array[i * 2 + 1] / S.meta.scale];
}
const playerAt = (f, slot) => pos(S.players, f * S.meta.slots + slot);
const ballAt = (f) => pos(S.ball, f);
const metres = (p) => [p[0] * S.L, p[1] * S.W];

function nearestByTeam(f) {
  // Nearest player of each team to the ball, in metres: [{slot, d}, {slot, d}] or null if no ball.
  const b = ballAt(f);
  if (!b) return null;
  const [bx, by] = metres(b);
  const best = [{ slot: -1, d: Infinity }, { slot: -1, d: Infinity }];
  for (let s = 0; s < S.meta.slots; s++) {
    const p = playerAt(f, s);
    if (!p) continue;
    const [x, y] = metres(p);
    const d = Math.hypot(x - bx, y - by);
    const t = S.meta.team[s];
    if (d < best[t].d) best[t] = { slot: s, d };
  }
  return best;
}

function ballSpeed(f) {
  const a = ballAt(f - 1), b = ballAt(f);
  if (!a || !b) return null;
  const [ax, ay] = metres(a), [bx, by] = metres(b);
  return Math.hypot(bx - ax, by - ay) * S.fps;
}

// ------------------------------------------------------------------------------------------
// Time
// ------------------------------------------------------------------------------------------

function seek(f) {
  S.watch.pausedFor = -1;  // after a jump, the next change may pause again
  S.frame = Math.max(0, Math.min(S.F - 1, Math.round(f)));
  video.currentTime = (S.frame + 0.5) / S.fps;  // middle of the frame, so the right one is shown
  render();
}

function onVideoFrame(now, metadata) {
  S.frame = Math.max(0, Math.min(S.F - 1, Math.round(metadata.mediaTime * S.fps)));
  render();
  video.requestVideoFrameCallback(onVideoFrame);
}

function fmt(seconds) {
  const sign = seconds < 0 ? '−' : '';
  const s = Math.abs(seconds);
  return `${sign}${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}

function matchClock(f) {
  if (f < S.meta.halves[0][1]) return 'before kick-off';
  if (f <= S.meta.halves[0][2]) return `1st half ${fmt((f - KICKOFF[1]) / S.fps)}`;
  if (f < S.meta.halves[1][1]) return 'half-time';
  if (f < KICKOFF[2]) return '2nd half, before kick-off';
  if (f <= S.meta.halves[1][2]) return `2nd half ${fmt(45 * 60 + (f - KICKOFF[2]) / S.fps)}`;
  return 'after the final whistle';
}

function parseJump(text) {
  const t = text.trim().toLowerCase();
  if (/^\d+$/.test(t)) return Number(t);
  const m = t.match(/^([12])\s*h\s*(\d+):(\d{1,2})$/);
  if (!m) return null;
  const half = Number(m[1]), seconds = Number(m[2]) * 60 + Number(m[3]);
  return KICKOFF[half] + (seconds - (half === 2 ? 45 * 60 : 0)) * S.fps;
}

// ------------------------------------------------------------------------------------------
// Drawing
// ------------------------------------------------------------------------------------------

function fitCanvas(canvas, cssHeight) {
  const ratio = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = cssHeight ?? canvas.clientHeight;
  if (cssHeight != null) canvas.style.height = `${h}px`;
  if (canvas.width !== Math.round(w * ratio) || canvas.height !== Math.round(h * ratio)) {
    canvas.width = Math.round(w * ratio);
    canvas.height = Math.round(h * ratio);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, w, h };
}

function dot(ctx, x, y, r, fill, stroke, width = 1.5) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) { ctx.lineWidth = width; ctx.strokeStyle = stroke; ctx.stroke(); }
}

function drawOverlay(f, near) {
  if (!S.H) return;  // no video on bench matches
  const { ctx, w, h } = fitCanvas($('overlay'));
  ctx.clearRect(0, 0, w, h);
  const b = ballAt(f);
  $('ball-badge').hidden = !!b;
  const m = S.H.subarray(f * 9, f * 9 + 9);
  if (Number.isNaN(m[0])) return;
  const k = w / 1920;
  const project = ([x, y]) => {
    const d = m[6] * x + m[7] * y + m[8];
    return [((m[0] * x + m[1] * y + m[2]) / d) * k, ((m[3] * x + m[4] * y + m[5]) / d) * k];
  };
  if ($('l-players').checked) {
    ctx.font = '10px system-ui';
    ctx.textAlign = 'center';
    for (let s = 0; s < S.meta.slots; s++) {
      const p = playerAt(f, s);
      if (!p) continue;
      const [x, y] = project(p);
      const team = S.meta.team[s];
      const onBall = near && near[team].slot === s && near[team].d <= ON_BALL_DISTANCE;
      dot(ctx, x, y, 5, team === 0 ? COLOR.H : COLOR.A, onBall ? COLOR.ball : '#fff', onBall ? 2.5 : 1.2);
      if ($('l-numbers').checked) { ctx.fillStyle = '#fff'; ctx.fillText(s + 1, x, y - 8); }
    }
  }
  if (b && $('l-ball').checked) {
    const [x, y] = project(b);
    dot(ctx, x, y, 4, COLOR.ball, '#000', 1.5);
  }
}

function drawPitchLines(ctx, X, Y, u) {
  const { L, W } = S;
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth = 1;
  const rect = (x, y, w, h) => ctx.strokeRect(X(x), Y(y), w * u, h * u);
  const circle = (x, y, r, a0 = 0, a1 = Math.PI * 2) => { ctx.beginPath(); ctx.arc(X(x), Y(y), r * u, a0, a1); ctx.stroke(); };
  rect(0, 0, L, W);
  ctx.beginPath(); ctx.moveTo(X(L / 2), Y(0)); ctx.lineTo(X(L / 2), Y(W)); ctx.stroke();
  circle(L / 2, W / 2, 9.15);
  for (const side of [0, 1]) {
    const x0 = side ? L - 16.5 : 0, g0 = side ? L - 5.5 : 0, spot = side ? L - 11 : 11;
    rect(x0, W / 2 - 20.16, 16.5, 40.32);
    rect(g0, W / 2 - 9.16, 5.5, 18.32);
    rect(side ? L : -2, W / 2 - 3.66, 2, 7.32);
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.fillRect(X(spot) - 1, Y(W / 2) - 1, 2, 2);
    const a = Math.acos(5.5 / 9.15);  // arc outside the box only
    circle(spot, W / 2, 9.15, side ? Math.PI - a : -a, side ? Math.PI + a : a);
  }
}

function drawPitch(f, near) {
  const canvas = $('pitch');
  const { ctx, w, h } = fitCanvas(canvas);
  const u = w / (S.L + 2 * MARGIN);
  const X = (x) => (x + MARGIN) * u, Y = (y) => (y + MARGIN) * u;
  ctx.fillStyle = '#0f2a20';
  ctx.fillRect(0, 0, w, h);
  drawPitchLines(ctx, X, Y, u);

  // Which way Barcelona attacks in this half (from the metadata), bottom-left.
  const half = f >= S.meta.halves[1][1] ? 2 : 1;
  const ltr = (S.meta.directions['0'] === 'left_to_right') === (half === 1);
  ctx.font = '11px system-ui';
  ctx.textAlign = 'left';
  ctx.fillStyle = COLOR.H;
  if (!MATCH) ctx.fillText(`Barcelona attacks ${ltr ? '→' : '←'}`, 6, h - 6);   // bench matches carry no directions

  // Ball trail over the last second.
  ctx.strokeStyle = 'rgba(250,204,21,0.5)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  let pen = false;
  for (let g = Math.max(0, f - S.fps); g <= f; g++) {
    const b = ballAt(g);
    if (!b) { pen = false; continue; }
    const [x, y] = metres(b);
    if (pen) ctx.lineTo(X(x), Y(y)); else ctx.moveTo(X(x), Y(y));
    pen = true;
  }
  ctx.stroke();

  for (let s = 0; s < S.meta.slots; s++) {
    const p = playerAt(f, s);
    if (!p) continue;
    const [x, y] = metres(p);
    const team = S.meta.team[s];
    const onBall = near && near[team].slot === s && near[team].d <= ON_BALL_DISTANCE;
    dot(ctx, X(x), Y(y), Math.max(3.5, 0.9 * u), team === 0 ? COLOR.H : COLOR.A, onBall ? COLOR.ball : 'rgba(255,255,255,0.8)', onBall ? 2.5 : 1);
  }

  const b = ballAt(f);
  if (b) {
    const [x, y] = metres(b);
    dot(ctx, X(x), Y(y), Math.max(3, 0.6 * u), COLOR.ball, '#000', 1.2);
  } else {
    let g = f;  // last known position, hollow, if seen within the last 10 s
    while (g > f - 10 * S.fps && g > 0 && !ballAt(g)) g--;
    const last = ballAt(g);
    if (last) {
      const [x, y] = metres(last);
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.arc(X(x), Y(y), Math.max(4, 0.8 * u), 0, Math.PI * 2);
      ctx.strokeStyle = COLOR.ball; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLOR.ball;
      ctx.textAlign = 'right';
      ctx.fillText(`ball lost ${((f - g) / S.fps).toFixed(1)} s ago`, w - 6, h - 6);
    }
  }
}

function chip(v, kind) {
  if (kind === 'label') return `<span class="chip ${LABEL_TEXT[v]}">${LABEL_TEXT[v]}</span>`;
  return v ? `<span class="chip on">on</span>` : `<span class="chip off">–</span>`;
}

function drawInfo(f, near) {
  const labelRows = S.lanes.filter((l) => l.kind === 'label')
    .map((l) => `<div class="row"><span class="k">${l.name}</span>${chip(l.value(f), 'label')}</div>`).join('');
  const clueRows = S.lanes.filter((l) => l.kind === 'clue').map((l) => {
    const v = l.value(f);
    const c = l.name === 'v4: P(dead)' ? `<span class="chip ${v > 50 ? 'D' : 'off'}">${v} %</span>`
      : (l.name === 'v6: restart clue' || l.name === 'v7: PathCRF') ? (v ? `<span class="chip ${v - 1 > 50 ? 'H' : 'A'}">${v - 1 > 50 ? TEAM_NAME[0] + ' ' + (v - 1) : TEAM_NAME[1] + ' ' + (101 - v)} %</span>` : chip(0))
      : (l.name.startsWith('A:') || l.name.startsWith('C:') || l.name === 'v8: keeper repaired' || l.name === 'v9: seen pass') ? (v ? `<span class="chip ${v === 1 ? 'H' : 'A'}">${v === 1 ? 'H' : 'A'}</span>` : chip(0))
      : l.name.endsWith('whistle') ? [chip(0), '<span class="chip on">counted</span>', '<span class="chip D">ignored</span>'][v]
        : chip(v);
    return `<div class="row"><span class="k">${l.name}</span>${c}</div>`;
  }).join('');
  const speed = ballSpeed(f);
  const ball = near
    ? `<div class="row"><span class="k">ball speed</span><span>${speed == null || Number.isNaN(speed) ? '–' : speed.toFixed(1) + ' m/s'}</span></div>
       <div class="row"><span class="k">nearest ${TEAM_NAME[0]}</span><span>${near[0].d === Infinity ? '–' : near[0].d.toFixed(1) + ' m'}</span></div>
       <div class="row"><span class="k">nearest ${TEAM_NAME[1]}</span><span>${near[1].d === Infinity ? '–' : near[1].d.toFixed(1) + ' m'}</span></div>`
    : '<div class="row"><span class="k">ball</span><span>not tracked</span></div>';
  const tags = S.meta.tags.filter((t) => t.start <= f && f < t.end)
    .map((t) => `<div class="row"><span class="k">${t.code.toLowerCase()}</span><span>${(t.Team || '').toLowerCase()}${t.Type ? ' · ' + t.Type.toLowerCase() : ''}</span></div>`).join('')
    || '<div class="row"><span class="k">none</span></div>';
  $('info').innerHTML = `
    <div class="card"><h3>Labels at this frame</h3>${labelRows}</div>
    <div class="card"><h3>Clues</h3>${clueRows}${ball}</div>
    <div class="card tags"><h3>Smart Tagging</h3>${tags}</div>`;
}

function laneRuns(ctx, lane, f0, f1, x0, ppf, y, h) {
  // One rectangle per run of equal colour, at least 1 px wide so short events stay visible.
  // (By colour, not value: the probability lanes change value every frame but colour rarely.)
  let start = f0, color = lane.color(lane.value(f0));
  for (let f = f0 + 1; f <= f1; f++) {
    const next = f < f1 ? lane.color(lane.value(f)) : undefined;
    if (next === color) continue;
    if (color) {
      ctx.fillStyle = color;
      ctx.fillRect(x0 + (start - f0) * ppf, y, Math.max(1, (f - start) * ppf), h);
    }
    start = f; color = next;
  }
}

const UNGRADED = { value: (f) => (S.graded[f] ? 0 : 1), color: (v) => (v ? COLOR.ungraded : null) };

function laneLayout(laneH, gap) {
  const labelCount = S.lanes.filter((l) => l.kind === 'label').length;
  return S.lanes.map((lane, i) => ({ lane, y: 4 + i * (laneH + gap) + (i >= labelCount ? 6 : 0) }));
}

function drawLaneNames(ctx, layout, laneH) {
  ctx.font = `${Math.min(11, laneH - 1)}px system-ui`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  for (const { lane, y } of layout) {
    ctx.fillStyle = lane.kind === 'label' ? COLOR.text : COLOR.muted;
    ctx.fillText(lane.name, 8, y + laneH / 2);
  }
}

function buildOverview() {
  const laneH = 11, gap = 2;
  const layout = laneLayout(laneH, gap);
  const height = layout[layout.length - 1].y + laneH + 4;
  const canvas = $('overview');
  const { w } = fitCanvas(canvas, height);
  const off = document.createElement('canvas');
  const ratio = window.devicePixelRatio || 1;
  off.width = Math.round(w * ratio); off.height = Math.round(height * ratio);
  const ctx = off.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const ppf = (w - LANE_W) / S.F;
  // Ungraded stretches (before kick-off, half-time, after the whistle) as grey bands.
  ctx.fillStyle = COLOR.ungraded;
  let start = 0;
  for (let f = 1; f <= S.F; f++) {
    if (f < S.F && S.graded[f] === S.graded[start]) continue;
    if (!S.graded[start]) ctx.fillRect(LANE_W + start * ppf, 0, (f - start) * ppf, height);
    start = f;
  }
  for (const { lane, y } of layout) laneRuns(ctx, lane, 0, S.F, LANE_W, ppf, y, laneH);
  drawLaneNames(ctx, layout, laneH);
  S.overview = { canvas: off, ppf, height };
}

function drawOverview(f) {
  const { ctx, w, h } = fitCanvas($('overview'), S.overview.height);
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(S.overview.canvas, 0, 0, w, h);
  const { ppf } = S.overview;
  ctx.fillStyle = 'rgba(255,255,255,0.12)';  // the detail window
  ctx.fillRect(LANE_W + (f - S.zoom * S.fps) * ppf, 0, Math.max(2, 2 * S.zoom * S.fps * ppf), h);
  ctx.fillStyle = '#fff';
  ctx.fillRect(LANE_W + f * ppf - 0.5, 0, 1.5, h);
  for (const note of S.notes) ctx.fillRect(LANE_W + note.frame * ppf - 0.5, 0, 1, 4);  // note ticks on top
}

function drawDetail(f) {
  const laneH = 14, gap = 3;
  const layout = laneLayout(laneH, gap);
  const height = layout[layout.length - 1].y + laneH + 22;
  const { ctx, w, h } = fitCanvas($('detail'), height);
  ctx.clearRect(0, 0, w, h);
  const span = S.zoom * S.fps;
  const f0 = Math.max(0, f - span), f1 = Math.min(S.F, f + span + 1);
  const ppf = (w - LANE_W) / (2 * span + 1);
  const x0 = LANE_W + (f0 - (f - span)) * ppf;
  // Ungraded frames shaded, then lanes.
  laneRuns(ctx, UNGRADED, f0, f1, x0, ppf, 0, h - 18);
  for (const { lane, y } of layout) laneRuns(ctx, lane, f0, f1, x0, ppf, y, laneH);
  drawLaneNames(ctx, layout, laneH);
  drawTicks(ctx, f, span, ppf, h);
}

function drawTicks(ctx, f, span, ppf, h) {
  // Seconds relative to the current frame along the bottom (every 1, 5 or 30 s), and the playhead.
  const step = (S.zoom <= 10 ? 1 : S.zoom <= 60 ? 5 : 30) * S.fps;
  ctx.fillStyle = COLOR.muted;
  ctx.font = '10px system-ui';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  for (let g = Math.ceil((f - span) / step) * step; g <= f + span; g += step) {
    const x = LANE_W + (g - (f - span)) * ppf;
    ctx.fillRect(x, h - 18, 1, 4);
    const seconds = Math.round((g - f) / S.fps);
    const text = Math.abs(seconds) < 60 ? `${seconds > 0 ? '+' : seconds < 0 ? '−' : ''}${Math.abs(seconds)} s`
      : `${seconds > 0 ? '+' : ''}${fmt(seconds)}`;
    ctx.fillText(text, x, h - 5);
  }
  const center = LANE_W + span * ppf + ppf / 2;
  ctx.fillStyle = '#fff';
  ctx.fillRect(center - 0.75, 0, 1.5, h - 18);
}

const STRIP_NOW = 34;  // px on the right of the strip for each lane's label at the current frame

function drawStrip(f) {
  // Label lanes only (truth, our models, baselines) around the current frame, right under the video.
  const laneH = 13, gap = 3;
  const lanes = S.lanes.filter((l) => l.kind === 'label');
  const layout = lanes.map((lane, i) => ({ lane, y: 5 + i * (laneH + gap) }));
  const height = layout[layout.length - 1].y + laneH + 22;
  const { ctx, w, h } = fitCanvas($('strip'), height);
  ctx.clearRect(0, 0, w, h);
  const span = S.zoom * S.fps;
  const f0 = Math.max(0, f - span), f1 = Math.min(S.F, f + span + 1);
  const ppf = (w - LANE_W - STRIP_NOW) / (2 * span + 1);
  const x0 = LANE_W + (f0 - (f - span)) * ppf;
  laneRuns(ctx, UNGRADED, f0, f1, x0, ppf, 0, h - 18);
  for (const { lane, y } of layout) {
    ctx.fillStyle = 'rgba(255,255,255,0.04)';  // empty lane background, so "no label" is visible
    ctx.fillRect(LANE_W, y, w - LANE_W - STRIP_NOW, laneH);
    laneRuns(ctx, lane, f0, f1, x0, ppf, y, laneH);
    const v = lane.value(f);  // the label at the playhead, as a letter on the right
    if (v) {
      ctx.fillStyle = LABEL_COLOR[v];
      ctx.fillRect(w - STRIP_NOW + 8, y, STRIP_NOW - 12, laneH);
      ctx.fillStyle = v === 3 ? '#111' : '#fff';
      ctx.font = '600 10px system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(LABEL_TEXT[v], w - STRIP_NOW + 8 + (STRIP_NOW - 12) / 2, y + laneH / 2 + 0.5);
    }
  }
  drawLaneNames(ctx, layout, laneH);
  drawTicks(ctx, f, span, ppf, h);
  ctx.fillStyle = '#fff';  // notes as thin white marks across the lanes
  for (const note of S.notes) {
    if (note.frame < f - span || note.frame > f + span) continue;
    ctx.fillRect(LANE_W + (note.frame - (f - span)) * ppf - 1, 0, 2, h - 18);
  }
}

function render() {
  if (!S.meta) return;
  const f = S.frame;
  const near = nearestByTeam(f);
  drawOverlay(f, near);
  drawStrip(f);
  watchTick(f);
  drawFrameState(f);
  markNearNotes(f);
  if (!document.body.classList.contains('review')) {  // hidden in review mode: skip the work
    drawPitch(f, near);
    drawInfo(f, near);
    drawOverview(f);
    drawDetail(f);
  }
  const t = f / S.fps;
  $('clock').textContent = `frame ${f} · ${MATCH ? 'time' : 'video'} ${fmt(t)}.${String(f % S.fps).padStart(2, '0')} · ${matchClock(f)}`;
  $('zoom-label').textContent = `±${S.zoom} s`;
  $('strip-zoom').textContent = `±${S.zoom} s`;
}

function resize() {
  if (!S.meta) return;
  buildOverview();
  render();
}

// ------------------------------------------------------------------------------------------
// Input
// ------------------------------------------------------------------------------------------

function togglePlay() {
  if (video.paused) video.play(); else video.pause();
}
video.addEventListener('play', () => { $('play').textContent = 'Pause'; });
video.addEventListener('pause', () => { $('play').textContent = 'Play'; });
video.addEventListener('seeked', () => { if (video.paused) { S.frame = Math.round(video.currentTime * S.fps - 0.5); render(); } });
if ('requestVideoFrameCallback' in video) {
  video.requestVideoFrameCallback(onVideoFrame);  // exact frame of what is on screen
} else {
  video.addEventListener('timeupdate', () => { S.frame = Math.floor(video.currentTime * S.fps); render(); });
}

$('play').addEventListener('click', togglePlay);
document.querySelectorAll('[data-step]').forEach((b) => b.addEventListener('click', () => {
  video.pause(); seek(S.frame + Number(b.dataset.step));
}));
$('rate').addEventListener('change', (e) => { S.watch.userRate = Number(e.target.value); video.playbackRate = S.watch.userRate; });
$('watch-mode').addEventListener('change', (e) => { S.watch.mode = e.target.value; S.watch.pausedFor = -1; render(); });
$('watch-rate').addEventListener('change', (e) => { S.watch.slowRate = Number(e.target.value); render(); });
$('watch-lead').addEventListener('change', (e) => { S.watch.lead = Number(e.target.value); render(); });
$('watch-flash').addEventListener('change', render);
$('frame-lane').addEventListener('change', render);

function toggleReview() {
  document.body.classList.toggle('review');
  requestAnimationFrame(resize);  // canvases follow the new layout
}
function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen().catch(() => {});
}
$('review').addEventListener('click', toggleReview);
$('fullscreen').addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', () => requestAnimationFrame(resize));

$('note-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const input = $('note-text');
  const text = input.value.trim();
  if (!text) { input.setCustomValidity('Write what you see first'); input.reportValidity(); return; }
  S.notes.push({ frame: S.frame, kind: $('note-kind').value, text, written: new Date().toISOString(),
                 match: MATCH || 'BAR-ATH' });
  S.notes.sort((a, b) => a.frame - b.frame);
  saveNotes();
  renderNotes();
  input.value = '';
  input.blur();
  render();
});
$('note-text').addEventListener('input', () => $('note-text').setCustomValidity(''));
$('l-sound').addEventListener('change', (e) => { video.muted = !e.target.checked; });
['l-players', 'l-ball', 'l-numbers'].forEach((id) => $(id).addEventListener('change', render));
$('jump-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const f = parseJump($('jump').value);
  if (f == null) { $('jump').setCustomValidity('Use a frame number, or 1h 12:30 / 2h 67:05'); $('jump').reportValidity(); return; }
  $('jump').setCustomValidity('');
  video.pause(); seek(f); $('jump').blur();
});
$('jump').addEventListener('input', () => $('jump').setCustomValidity(''));
const zoom = (factor) => { S.zoom = Math.max(5, Math.min(600, Math.round(S.zoom * factor))); render(); };
$('zoom-in').addEventListener('click', () => zoom(0.5));
$('zoom-out').addEventListener('click', () => zoom(2));

$('overview').addEventListener('click', (e) => {
  const x = e.offsetX - LANE_W;
  if (x < 0) return;
  video.pause(); seek(x / S.overview.ppf);
});
$('strip').addEventListener('click', (e) => {
  const w = $('strip').clientWidth;
  const span = S.zoom * S.fps;
  const x = e.offsetX - LANE_W;
  if (x < 0 || e.offsetX > w - STRIP_NOW) return;
  video.pause(); seek(S.frame - span + (x / (w - LANE_W - STRIP_NOW)) * (2 * span + 1));
});
$('detail').addEventListener('click', (e) => {
  const w = $('detail').clientWidth;
  const span = S.zoom * S.fps;
  const x = e.offsetX - LANE_W;
  if (x < 0) return;
  video.pause(); seek(S.frame - span + (x / (w - LANE_W)) * (2 * span + 1));
});

document.addEventListener('keydown', (e) => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) {
    if (e.key === 'Escape') e.target.blur();
    return;
  }
  const step = e.altKey ? 250 : e.shiftKey ? 25 : 1;
  if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); video.pause(); seek(S.frame + step); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); video.pause(); seek(S.frame - step); }
  else if (e.key === '[') zoom(0.5);
  else if (e.key === ']') zoom(2);
  else if (e.key === 'n') jumpToChange(1);
  else if (e.key === 'p') jumpToChange(-1);
  else if (e.key === 'r') toggleReview();
  else if (e.key === 'f') toggleFullscreen();
  else if (e.key === 'm') { e.preventDefault(); video.pause(); $('note-text').focus(); }
  else if (e.key === 'j') { e.preventDefault(); $('jump').focus(); }
});
window.addEventListener('resize', resize);

load().catch((error) => { $('loading').textContent = `Could not load the viewer data: ${error.message}`; });
