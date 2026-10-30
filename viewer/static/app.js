'use strict';
// Match viewer: video + tracking overlay + 2D pitch + label and clue timelines, all on one frame.
// Data comes from viewer/cache (see viewer/build_cache.py for the encodings).

const COLOR = { H: '#3b82f6', A: '#ef4444', D: '#9ca3af', clue: '#f59e0b', seen: '#22c55e', ball: '#facc15',
  ungraded: 'rgba(255,255,255,0.05)', grid: '#262b32', text: '#e5e7ea', muted: '#8f98a3' };
const LABEL_COLOR = [null, COLOR.H, COLOR.A, COLOR.D];
const LABEL_TEXT = ['–', 'H', 'A', 'D'];
// Kick-off frames, published in DATA.md and FAQ.md. Used only for the clock readout.
const KICKOFF = { 1: 11185, 2: 113742 };
const ON_BALL_DISTANCE = 2.0;  // metres, same as model v1's clue A
const MARGIN = 4;              // metres of grass drawn around the pitch map
const LANE_W = 150;            // px reserved for lane names on the timelines

const $ = (id) => document.getElementById(id);
const video = $('video');
const S = { frame: 0, zoom: 30, playing: false };

// ------------------------------------------------------------------------------------------
// Loading and data access
// ------------------------------------------------------------------------------------------

async function load() {
  const meta = await fetch('/cache/meta.json').then((r) => r.json());
  const names = ['players', 'ball', 'homography', 'labels', 'clues'];
  const buffers = await Promise.all(names.map((n) => fetch(`/cache/${n}.bin`).then((r) => {
    if (!r.ok) throw new Error(`missing cache file ${n}.bin`);
    return r.arrayBuffer();
  })));
  Object.assign(S, {
    meta, F: meta.frames, fps: meta.fps, L: meta.pitch[0], W: meta.pitch[1],
    players: new Int16Array(buffers[0]), ball: new Int16Array(buffers[1]), H: new Float32Array(buffers[2]),
    labels: new Uint8Array(buffers[3]), clues: new Uint8Array(buffers[4]),
  });
  S.graded = new Uint8Array(S.F);
  for (const [, first, last] of meta.halves) S.graded.fill(1, first, last + 1);
  S.lanes = buildLanes(meta);
  S.v1 = meta.labelLanes.indexOf('v1');
  $('loading').hidden = true;
  resize();
  seek(meta.halves[0][1]);
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
    color: name === 'ball visible' ? (v) => (v ? COLOR.seen : null)
      : (name.startsWith('A:') || name.startsWith('C:')) ? (v) => [null, COLOR.H, COLOR.A][v]
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

function nextChange(lane, f) {
  // First frame after f whose label differs from f's.
  const v = lane.value, current = v(f);
  let g = f + 1;
  while (g < S.F && v(g) === current) g++;
  return g < S.F ? g : f;
}

function previousChange(lane, f) {
  // Start of the run that contains f - 1: the start of the current run, or of the one before it.
  const v = lane.value;
  let g = f - 1;
  if (g < 0) return f;
  const current = v(g);
  while (g > 0 && v(g - 1) === current) g--;
  return g;
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
  ctx.fillText(`Barcelona attacks ${ltr ? '→' : '←'}`, 6, h - 6);

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
    const c = (l.name.startsWith('A:') || l.name.startsWith('C:')) ? (v ? `<span class="chip ${v === 1 ? 'H' : 'A'}">${v === 1 ? 'H' : 'A'}</span>` : chip(0)) : chip(v);
    return `<div class="row"><span class="k">${l.name}</span>${c}</div>`;
  }).join('');
  const speed = ballSpeed(f);
  const ball = near
    ? `<div class="row"><span class="k">ball speed</span><span>${speed == null ? '–' : speed.toFixed(1) + ' m/s'}</span></div>
       <div class="row"><span class="k">nearest Barcelona</span><span>${near[0].d === Infinity ? '–' : near[0].d.toFixed(1) + ' m'}</span></div>
       <div class="row"><span class="k">nearest Athletic</span><span>${near[1].d === Infinity ? '–' : near[1].d.toFixed(1) + ' m'}</span></div>`
    : '<div class="row"><span class="k">ball</span><span>not tracked</span></div>';
  const tags = S.meta.tags.filter((t) => t.start <= f && f < t.end)
    .map((t) => `<div class="row"><span class="k">${t.code.toLowerCase()}</span><span>${(t.Team || '').toLowerCase()}${t.Type ? ' · ' + t.Type.toLowerCase() : ''}</span></div>`).join('')
    || '<div class="row"><span class="k">none</span></div>';
  $('info').innerHTML = `
    <div class="card"><h3>Labels at this frame</h3>${labelRows}</div>
    <div class="card"><h3>v1 clues</h3>${clueRows}${ball}</div>
    <div class="card tags"><h3>Smart Tagging</h3>${tags}</div>`;
}

function laneRuns(ctx, lane, f0, f1, x0, ppf, y, h) {
  // One rectangle per run of equal values, at least 1 px wide so short events stay visible.
  let start = f0, v = lane.value(f0);
  for (let f = f0 + 1; f <= f1; f++) {
    const nv = f < f1 ? lane.value(f) : -1;
    if (nv === v) continue;
    const color = lane.color(v);
    if (color) {
      ctx.fillStyle = color;
      ctx.fillRect(x0 + (start - f0) * ppf, y, Math.max(1, (f - start) * ppf), h);
    }
    start = f; v = nv;
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
  // Ticks: every 1, 5 or 10 seconds depending on the zoom.
  const step = (S.zoom <= 10 ? 1 : S.zoom <= 60 ? 5 : 30) * S.fps;
  ctx.fillStyle = COLOR.muted;
  ctx.font = '10px system-ui';
  ctx.textAlign = 'center';
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

function render() {
  if (!S.meta) return;
  const f = S.frame;
  const near = nearestByTeam(f);
  drawOverlay(f, near);
  drawPitch(f, near);
  drawInfo(f, near);
  drawOverview(f);
  drawDetail(f);
  const t = f / S.fps;
  $('clock').textContent = `frame ${f} · video ${fmt(t)}.${String(f % S.fps).padStart(2, '0')} · ${matchClock(f)}`;
  $('zoom-label').textContent = `±${S.zoom} s`;
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
$('rate').addEventListener('change', (e) => { video.playbackRate = Number(e.target.value); });
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
$('detail').addEventListener('click', (e) => {
  const w = $('detail').clientWidth;
  const span = S.zoom * S.fps;
  const x = e.offsetX - LANE_W;
  if (x < 0) return;
  video.pause(); seek(S.frame - span + (x / (w - LANE_W)) * (2 * span + 1));
});

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const step = e.altKey ? 250 : e.shiftKey ? 25 : 1;
  const v1 = S.lanes && S.lanes.find((l) => l.name === 'v1');
  if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); video.pause(); seek(S.frame + step); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); video.pause(); seek(S.frame - step); }
  else if (e.key === '[') zoom(0.5);
  else if (e.key === ']') zoom(2);
  else if (e.key === 'n' && v1) { video.pause(); seek(nextChange(v1, S.frame)); }
  else if (e.key === 'p' && v1) { video.pause(); seek(previousChange(v1, S.frame)); }
  else if (e.key === 'j') { e.preventDefault(); $('jump').focus(); }
});
window.addEventListener('resize', resize);

load().catch((error) => { $('loading').textContent = `Could not load the viewer data: ${error.message}`; });
