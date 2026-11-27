// RL Reels — swipeable micro-lessons with a memory of what you've watched.
'use strict';
(() => {
const VERSION = 14;
const TOPICS = window.RLTopics, BY = Object.fromEntries(TOPICS.map(t => [t.id, t]));
const RLA = window.RLAnim || { A: {}, CAP: {}, SPEED: {}, TOUCH: new Set() }, { A } = RLA;
const STORY = window.RLStory, SL = STORY.lib;
const KEY = 'rlreels.v1', DAY = 864e5;
const MIN_SEC_PER_SLIDE = 1.5;         // reaching the end faster than this per content slide counts as "skimmed"
const REVIEW_DAYS = [3, 8, 21, 60];    // spaced re-showing of watched reels
const $ = s => document.querySelector(s);

// sanity check on content
for (const t of TOPICS) {
  t.story = true;   // every topic is a hand-drawn story reel (story/<id>.js), loaded when first shown
  for (const p of t.pre) if (!BY[p]) console.warn('unknown prereq', t.id, p);
}
// ---------- story reels: load on demand, turn steps into slides ----------
const storyLoading = {};
function loadStory(id) {
  if (STORY.reels[id]) return Promise.resolve();
  return storyLoading[id] ||= new Promise((res, rej) => {
    const s = document.createElement('script'); s.src = 'story/' + id + '.js?v=' + VERSION;
    s.onload = () => STORY.reels[id] ? res() : rej(); s.onerror = () => { delete storyLoading[id]; rej(); };
    document.body.appendChild(s);
  });
}
function prepStory(t) {
  const R = STORY.reels[t.id]; if (!R || t._prep) return; t._prep = true;
  t.slides = R.steps.map((s, k) => ({ get h() { return s.h; }, get t() { return (s.p || []).join(' '); }, a: '__story', p: { k }, step: s }));
}
const storyText = s => (s.step.p || []).map(p => `<p>${md(p)}</p>`).join('');
const ctlFmt = (c, i) => { const v = c.values[i]; return c.fmt ? c.fmt(v) : String(v); };

// ---------- memory ----------
let M = load();
function load() { try { const m = JSON.parse(localStorage.getItem(KEY)); if (m && m.t) return m; } catch (e) { } return { t: {}, seq: 0 }; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(M)); } catch (e) { } updateCount(); }
const rec = id => M.t[id] || (M.t[id] = { views: 0 });
const st = id => M.t[id]?.status;          // undefined | 'partial' | 'skimmed' | 'done'
const isDone = id => st(id) === 'done';
const unlocked = t => t.pre.every(isDone);
const pendingAgain = r => r && (r.status === 'partial' || r.status === 'skimmed' || r.again);

// ---------- scheduling ----------
// A reel you leave early (even on its cover) "cools down": it is not offered again until at least
// SKIP_REELS other reels AND SKIP_MIN minutes have passed — unless there is truly nothing else to show.
const SKIP_REELS = 8, SKIP_MIN = 30;
const cooling = (r, now) => r && r.skipSeq != null && (M.seq - r.skipSeq < SKIP_REELS || now - (r.skipAt || 0) < SKIP_MIN * 60e3);
const oldest = (arr, f) => arr.slice().sort((a, b) => f(a) - f(b))[0].id;
function pickNext() {
  const now = Date.now(), R = Math.random();
  const recent = new Set([...hist.slice(-6), hist[pos]].filter(Boolean).map(h => h.id));
  const pool = TOPICS.filter(t => !recent.has(t.id) && unlocked(t));
  const warm = pool.filter(t => !cooling(M.t[t.id], now));
  const fresh = warm.filter(t => !st(t.id));
  const due = warm.filter(t => { const r = M.t[t.id]; return pendingAgain(r) && M.seq >= (r.dueSeq || 0) && now >= (r.dueAt || 0); });
  const review = warm.filter(t => isDone(t.id) && !M.t[t.id].again && now >= (M.t[t.id].dueAt || 0));
  const last = id => M.t[id]?.lastSeen || 0;
  // preview: about 1 reel in 5 is an advanced topic whose prerequisites aren't watched yet (never two close together)
  const locked = TOPICS.filter(t => !recent.has(t.id) && !unlocked(t) && !st(t.id) && !cooling(M.t[t.id], now));
  if (locked.length && (M.seq || 0) >= 2 && R >= .3 && R < .5 && (M.seq || 0) - (M.lastPreview ?? -9) >= 3) {
    const w = locked.map(t => 1 / (1 + t.pre.filter(p => !isDone(p)).length)), s = w.reduce((a, b) => a + b);
    let u = Math.random() * s; for (let i = 0; i < locked.length; i++) { u -= w[i]; if (u <= 0) { M.lastPreview = M.seq || 0; return locked[i].id; } }
  }
  if (fresh.length) {
    if (due.length && R < .3) return oldest(due, t => last(t.id));
    if (review.length && R > .9) return review[Math.floor(Math.random() * review.length)].id;
    // prefer continuing a chain: topics whose prerequisite was just finished
    const justDone = new Set(TOPICS.filter(t => isDone(t.id) && now - (M.t[t.id].doneAt || 0) < 30 * 60e3 && hist.slice(-3).some(h => h.id === t.id)).map(t => t.id));
    const w = fresh.map(t => t.pre.some(p => justDone.has(p)) ? 4 : 1), s = w.reduce((a, b) => a + b);
    let u = Math.random() * s; for (let i = 0; i < fresh.length; i++) { u -= w[i]; if (u <= 0) return fresh[i].id; }
    return fresh[0].id;
  }
  if (due.length) return oldest(due, t => last(t.id));
  if (review.length) return oldest(review, t => last(t.id));
  // nothing unlocked is left: offer a not-yet-unlocked topic (fewest missing prerequisites) rather than repeat a skip
  const early = TOPICS.filter(t => !recent.has(t.id) && !unlocked(t) && !st(t.id) && !cooling(M.t[t.id], now));
  if (early.length) { const miss = t => t.pre.filter(p => !isDone(p)).length, m = Math.min(...early.map(miss)), best = early.filter(t => miss(t) === m); return best[Math.floor(Math.random() * best.length)].id; }
  // then refresh an already-watched reel before re-offering skipped ones
  const watched = pool.filter(t => isDone(t.id) && !cooling(M.t[t.id], now));
  if (watched.length) return oldest(watched, t => last(t.id));
  if (pool.length) return oldest(pool, t => M.t[t.id]?.skipAt || 0);   // skipped longest ago
  return oldest(TOPICS.filter(unlocked), t => last(t.id));
}
function markDone(id) {
  const before = new Set(TOPICS.filter(unlocked).map(t => t.id));
  const r = rec(id), was = r.status === 'done';
  r.level = was ? Math.min((r.level || 0) + 1, REVIEW_DAYS.length - 1) : 0;
  r.status = 'done'; r.doneAt = Date.now(); r.dueAt = Date.now() + REVIEW_DAYS[r.level] * DAY; delete r.again; delete r.dueSeq; delete r.skipSeq; delete r.skipAt;
  save();
  return TOPICS.filter(t => unlocked(t) && !before.has(t.id)).map(t => t.id);
}
function scheduleAgain(id, soon) {
  const r = rec(id);
  r.dueSeq = M.seq + (soon ? 6 : SKIP_REELS); r.dueAt = Date.now() + (soon ? 30 : 60) * 60e3;
}

// ---------- rendering ----------
const esc = s => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const md = s => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`(.+?)`/g, '<code>$1</code>');
const HUE = { 'Foundations': 265, 'Bandits': 28, 'Planning': 195, 'Model-free': 150, 'Deep RL': 315, 'Policy Gradients': 350,
  'Model-based': 175, 'Rewards': 45, 'Learning from people': 285, 'Bigger worlds': 215 };
function badgeFor(t) {
  const r = M.t[t.id];
  if (!unlocked(t)) return '<span class="badge locked">👀 PREVIEW · advanced</span>';
  if (!r || !r.status) return '<span class="badge new">✨ NEW</span>';
  if (r.status === 'done') return r.again ? '<span class="badge again">↻ ROUND 2</span>' : '<span class="badge review">🧠 REFRESH</span>';
  return '<span class="badge again">↻ ROUND 2</span>';
}
function chip(id, cls = '') { const t = BY[id]; return `<button class="chip ${cls || (isDone(id) ? 'done' : 'todo')}" data-open="${id}">${t.emoji} ${esc(t.title)}${isDone(id) ? ' ✓' : ''}</button>`; }
const SLIDE_EMOJI = ['💡', '⚙️', '🎯', '🧩', '🚀', '🔍', '🌟', '🧠'];
function slideDur(t, i) { // seconds a slide auto-plays for
  if (i === 0) return 4.5;
  const s = t.slides[i - 1]; if (!s) return 0;
  if (s.step) return s._dur ??= Math.min(40, Math.max(8, 4 + s.t.length * .07));
  const chars = (s.h || '').length + (s.t || '').length + (s.ex || '').length + (s.g || []).reduce((a, [w, d]) => a + w.length + d.length, 0);
  return Math.min(24, Math.max(6, 3 + chars * .068 + (s.a ? 3.5 : 0)));
}
function buildReel(t, startIdx = 0) {
  const r = M.t[t.id], n = t.slides.length + 2, hue = HUE[t.track] ?? 265;
  const el = document.createElement('div'); el.className = 'reel' + (t._prep ? ' story' : ''); el.style.setProperty('--h', hue);
  let note = '';
  if (r && (r.status === 'partial')) note = `<div class="note">👀 You left at slide ${(r.maxSlide || 0) + 1}/${n} last time</div>`;
  else if (r && r.status === 'skimmed') note = `<div class="note">💨 You speed-ran this one — calmer look?</div>`;
  else if (r && r.status === 'done') note = `<div class="note">🧠 Refresher · watched ${r.views}×</div>`;
  if (!unlocked(t)) note += `<div class="note">Normally after: ${t.pre.filter(p => !isDone(p)).map(p => BY[p].title).join(', ')}</div>`;
  const mins = t._prep ? Math.max(1, Math.round((n - 2) * 22 / 60)) : Math.max(1, Math.round(Array.from({ length: n }, (_, i) => slideDur(t, i)).reduce((a, b) => a + b) / 60));
  const slides = [
    `<section class="slide cover"><div class="halo"><div class="big">${t.emoji}</div></div>${badgeFor(t)}<h1>${esc(t.title)}</h1>
      <div class="hook">${md(t.hook)}</div><div class="meta">${esc(t.track)} · ${n} slides · ~${mins} min</div>${note}
      ${t.pre.length ? `<div class="chips">${t.pre.map(p => chip(p)).join('')}</div>` : ''}
      <div class="hint">${M.autoplay ? 'tap to start · hold to pause' : 'tap to start'}</div></section>`,
    ...t.slides.map((s, i) => s.step ? `<section class="slide sstep"><div class="num"><span>${SLIDE_EMOJI[i % SLIDE_EMOJI.length]}</span>${i + 1}<small>/${t.slides.length}</small></div><h2>${md(s.h)}</h2>
      <div class="cvwrap"><canvas data-i="${i + 1}" data-h="420" class="story${s.step.tap ? ' touch' : ''}"></canvas></div>${s.step.ctl ? `<div class="ctl"><label><span>${esc(s.step.ctl.label)}</span><input type="range" data-sk="1" min="0" max="${s.step.ctl.values.length - 1}" step="1" value="${s.step.ctl.def ?? 0}"><output>${esc(ctlFmt(s.step.ctl, s.step.ctl.def ?? 0))}</output></label></div>` : ''}<div class="stext"></div></section>` : `<section class="slide"><div class="num"><span>${SLIDE_EMOJI[i % SLIDE_EMOJI.length]}</span>${i + 1}<small>/${t.slides.length}</small></div>${s.h ? `<h2>${md(s.h)}</h2>` : ''}
      ${s.a ? `<div class="cvwrap"><canvas data-i="${i + 1}"${RLA.TOUCH.has(s.a) ? ' class="touch"' : ''}></canvas></div>` : ''}${s.ctl ? `<div class="ctl">${s.ctl.map(c => `<label><span>${esc(c.l)}</span><input type="range" data-k="${c.k}" min="${c.min}" max="${c.max}" step="${c.step || (c.max - c.min) / 100}" value="${c.v}"><output>${c.v}</output></label>`).join('')}</div>` : ''}${s.a && RLSpeech.capFor(s) ? `<div class="see"><b>👀</b><div>${md(RLSpeech.capFor(s))}${s.try ? `<em>👆 ${md(s.try)}</em>` : ''}</div></div>` : ''}${s.t ? `<p>${md(s.t)}</p>` : ''}${s.g ? `<div class="gloss">${s.g.map(([w, d], k) => `<div style="--k:${k}"><b>${md(w)}</b><span>${md(d)}</span></div>`).join('')}</div>` : ''}${s.ex ? `<div class="ex"><span>💬</span><div>${md(s.ex)}</div></div>` : ''}</section>`),
    `<section class="slide recap"><div class="trophy">${t.emoji}</div><h2>${t._prep ? 'Reel complete' : 'In a nutshell'}</h2>${t._prep ? '' : `<ul>${t.recap.map((x, k) => `<li style="--k:${k}">${md(x)}</li>`).join('')}</ul>`}
      <div class="end-status"></div><div class="swipe-up"><span>⌃</span>swipe up for the next reel</div></section>`
  ];
  el.innerHTML = `<div class="aurora"><i></i><i></i><i></i></div>
    <div class="bars">${Array.from({ length: n }, () => '<i><u></u></i>').join('')}</div>
    <div class="top"><div class="av"><b>${t.emoji}</b></div><div class="who"><b>${esc(t.title)}</b><small>${esc(t.track)}</small></div>
      <div class="hud"><span class="streak">🔥 ${M.streak || 0}</span><span class="xp">⚡ ${M.xp || 0}</span></div></div>
    <div class="track">${slides.join('')}</div><div class="paused">❚❚</div>
    ${t.pre.length ? `<div class="foot"><span class="lab">↩ builds on</span>${t.pre.map(p => chip(p)).join('')}</div>` : ''}`;
  const reel = { t, el, n, idx: -1, max: startIdx, dwell: new Array(n).fill(0), enteredAt: 0, finalized: false, doneNow: false, canvases: {}, t0: 0, S: {}, pausedAt: 0, pausedFor: 0 };
  el.querySelectorAll('canvas').forEach(cv => reel.canvases[+cv.dataset.i] = cv);
  setSlide(reel, startIdx, true);
  updateLike(reel);
  return reel;
}
function fitStory(sec) { // story step: biggest font + drawing that fit together above the bottom bar; text never hidden
  const cw = sec.querySelector('.cvwrap'), tx = sec.querySelector('.stext'); if (!cw || !tx) return;
  const reel = sec.closest('.reel'), foot = reel && reel.querySelector('.foot'), side = document.getElementById('side');
  const cs = getComputedStyle(sec), gap = parseFloat(cs.rowGap) || 0, padT = parseFloat(cs.paddingTop) || 0;
  const bottomBar = foot ? foot.offsetHeight : 0;
  sec.style.paddingBottom = (bottomBar + 10) + 'px';
  const inner = sec.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const avail = sec.clientHeight - padT - bottomBar - 10;
  const kids = [...sec.children].filter(k => k !== cw && k !== tx);
  const fixed = kids.reduce((a, k) => a + k.offsetHeight, 0) + gap * (sec.children.length - 1);
  // the side buttons sit at the bottom right: a drawing that reaches down to them must leave that column free
  const sideTop = side ? side.getBoundingClientRect().top - sec.getBoundingClientRect().top : 1e9;
  tx.classList.remove('scroll'); let pick = null;
  for (const f of [19, 18, 17, 16, 15, 14]) {
    tx.style.fontSize = f + 'px';
    const th = tx.scrollHeight, ch = avail - fixed - th;
    let w = Math.min(inner, ch * 400 / 420), left = false;
    const top = padT + kids.filter(k => k.compareDocumentPosition(cw) & 4).reduce((a, k) => a + k.offsetHeight + gap, 0);
    if (top + w * 420 / 400 > sideTop + 6) { w = Math.min(w, inner - 44); left = true; }   // keep the button column free: shift left
    const minFrac = f >= 19 ? .99 : f >= 17 ? .88 : f >= 15 ? .76 : .68;
    if (w >= Math.max((inner - 44) * minFrac, Math.min(inner - 44, 230))) { pick = { f, w, left }; break; }   // never shrink the drawing below ~230 px: labels stay readable
  }
  if (!pick) { pick = { f: 14, w: Math.max((inner - 44) * .68, Math.min(inner - 44, 230)), left: true }; tx.style.fontSize = '14px'; tx.classList.add('scroll'); }
  cw.style.width = Math.floor(pick.w) + 'px'; cw.style.alignSelf = pick.left ? 'flex-start' : 'center';
}
function fitSlide(sec) { // shrink text/canvas step by step until a long slide fits the screen
  if (!sec || !sec.clientHeight) return;
  if (sec.classList.contains('sstep')) return fitStory(sec);
  sec.classList.remove('dense', 'dense2');
  for (const c of ['dense', 'dense2']) { if (sec.scrollHeight <= sec.clientHeight + 24) break; sec.classList.add(c); }
}
addEventListener('resize', () => { if (cur) fitSlide(cur.el.querySelectorAll('.slide')[cur.idx]); });
document.fonts && document.fonts.ready.then(() => { if (cur) fitSlide(cur.el.querySelectorAll('.slide')[cur.idx]); });
function setSlide(reel, i, instant) {
  i = Math.max(0, Math.min(reel.n - 1, i));
  if (!instant && reel.idx >= 0 && i !== reel.idx) sfx(i > reel.idx ? 'tick' : 'back');
  const now = performance.now();
  if (reel.idx >= 0) reel.dwell[reel.idx] += Math.min(60, (now - reel.enteredAt) / 1000);
  reel.idx = i; reel.enteredAt = now; reel.t0 = now; reel.S = {}; reel.max = Math.max(reel.max, i); reel.pausedFor = 0; reel.pausedAt = 0; reel.el.classList.remove('is-paused');
  const tr = reel.el.querySelector('.track');
  if (instant) { tr.style.transition = 'none'; requestAnimationFrame(() => tr.style.transition = ''); }
  tr.style.transform = `translateX(${-i * 100}%)`;
  reel.el.querySelectorAll('.slide').forEach((s, k) => s.classList.toggle('active', k === i));
  reel.el.classList.toggle('mid', i > 0 && i < reel.n - 1);   // story steps hide the 'builds on' bar to give the text room
  if (reel.t._prep) for (const k of [i, i + 1]) { const sl = reel.t.slides[k - 1], box = sl && reel.el.querySelectorAll('.slide')[k]?.querySelector('.stext'); if (box && !box.childElementCount) box.innerHTML = storyText(sl); }
  fitSlide(reel.el.querySelectorAll('.slide')[i]);
  reel.el.querySelectorAll('.bars u').forEach((b, k) => b.style.width = k < i ? '100%' : '0%');
  if (i === reel.n - 1) onRecap(reel);
  if (reel === cur) narrate(reel);
  const r = M.t[reel.t.id]; if (r && r.status !== 'done') { r.maxSlide = Math.max(r.maxSlide || 0, i); save(); }
}
// ---------- audio: narration (built-in speech voice) + synthesized sound effects ----------
// M.audio: 2 = voice + effects (default), 1 = effects only, 0 = muted.
const audioMode = () => M.audio ?? 2;
let TTS = 'speechSynthesis' in window ? window.speechSynthesis : null;
try { if (window.parent !== window && window.parent.__mockTTS) TTS = window.parent.__mockTTS; } catch (e) { }  // test hook (same-origin test page only)
let AC = null, audioUnlocked = false, voice = null;
function pickVoice() {
  if (!TTS) return; const vs = TTS.getVoices().filter(v => /^en(-|_|$)/i.test(v.lang)); if (!vs.length) return;
  const score = v => (/natural|neural|enhanced|premium/i.test(v.name) ? 8 : 0) + (/google|samantha|daniel|karen|moira|serena|aria|jenny|guy/i.test(v.name) ? 4 : 0)
    + (/en-(US|GB|IN|AU)/i.test(v.lang) ? 2 : 0) + (v.localService ? 1 : 0) - (/espeak|robot/i.test(v.name) ? 10 : 0);
  voice = vs.sort((a, b) => score(b) - score(a))[0];
}
if (TTS) { pickVoice(); TTS.addEventListener?.('voiceschanged', pickVoice); }
// Phones only grant audio permission on a *completed* tap (pointerup / touchend / click), not on touch-down,
// so unlock on those events, and keep retrying until the audio context is really running.
function unlockAudio() {
  try { if (!AC) AC = new (window.AudioContext || window.webkitAudioContext)(); if (AC.state !== 'running') AC.resume(); } catch (e) { }
  if (audioUnlocked) return; audioUnlocked = true;
  try { const b = AC.createBuffer(1, 1, 22050), src = AC.createBufferSource(); src.buffer = b; src.connect(AC.destination); src.start(0); } catch (e) { }  // iOS
  if (!TTS && audioMode() === 2) setTimeout(() => toast('🔇 This browser has no speech voice — effects only'), 400);
  // Safari only allows speech that starts synchronously inside the tap, so start reading the current slide right here
  if (cur && audioMode() === 2) narrate(cur);
}
['pointerup', 'touchend', 'click', 'keydown'].forEach(ev => document.addEventListener(ev, unlockAudio, true));

// --- sound effects ---
function tone(f, t0, d, type = 'sine', g = .12, f2) {
  const o = AC.createOscillator(), a = AC.createGain(); o.type = type; o.frequency.setValueAtTime(f, t0); if (f2) o.frequency.exponentialRampToValueAtTime(f2, t0 + d);
  a.gain.setValueAtTime(0, t0); a.gain.linearRampToValueAtTime(g, t0 + .008); a.gain.exponentialRampToValueAtTime(.0001, t0 + d);
  o.connect(a).connect(AC.destination); o.start(t0); o.stop(t0 + d + .02);
}
function noise(t0, d, f1, f2, g = .08) {
  const n = AC.createBuffer(1, Math.ceil(AC.sampleRate * d), AC.sampleRate), x = n.getChannelData(0); for (let i = 0; i < x.length; i++) x[i] = Math.random() * 2 - 1;
  const s = AC.createBufferSource(), bp = AC.createBiquadFilter(), a = AC.createGain(); s.buffer = n; bp.type = 'bandpass'; bp.Q.value = 1.2;
  bp.frequency.setValueAtTime(f1, t0); bp.frequency.exponentialRampToValueAtTime(f2, t0 + d);
  a.gain.setValueAtTime(0, t0); a.gain.linearRampToValueAtTime(g, t0 + d * .3); a.gain.exponentialRampToValueAtTime(.0001, t0 + d);
  s.connect(bp).connect(a).connect(AC.destination); s.start(t0);
}
function sfx(name) {
  if (!audioMode() || !AC || AC.state !== 'running') return; const t = AC.currentTime + .01;
  try {
    if (name === 'tick') tone(1250, t, .05, 'sine', .05);
    else if (name === 'back') tone(800, t, .05, 'sine', .05);
    else if (name === 'whoosh') noise(t, .32, 300, 2400, .1);
    else if (name === 'chime') [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => { tone(f, t + i * .085, .5, 'triangle', .1); tone(f * 2, t + i * .085, .25, 'sine', .025); });
    else if (name === 'pop') tone(900, t, .12, 'sine', .12, 380);
    else if (name === 'level') [392, 523.25, 659.25, 783.99, 1046.5, 1318.5].forEach((f, i) => tone(f, t + i * .07, .45, 'square', .045));
    else if (name === 'like') { tone(600, t, .1, 'sine', .1, 900); tone(900, t + .08, .14, 'sine', .08, 1300); }
    else if (name === 'pause') tone(420, t, .06, 'triangle', .07);
    else if (name === 'skim') tone(300, t, .2, 'sawtooth', .04, 200);
  } catch (e) { }
}

// --- narration --- (text helpers live in speech.js: RLSpeech.clean / RLSpeech.narrationText)
const { clean, narrationText } = window.RLSpeech;
let narToken = 0;
// Pre-recorded neural-voice clips (tts/gen.py → audio/index.json); the browser's own voice is only a fallback.
let CLIPS = null;
fetch('audio/index.json?v=' + VERSION).then(r => r.ok ? r.json() : null).then(j => { CLIPS = j; }).catch(() => { });
const player = new Audio(); player.preload = 'auto';
function clipFor(t, i) { const c = CLIPS && CLIPS[t.id + ':' + i]; return c && c.t === clean(narrationText(t, i)) ? c : null; }
function prefetchClip(t, i) { const c = clipFor(t, i); if (c && !prefetchClip.done.has(c.f)) { prefetchClip.done.add(c.f); fetch('audio/' + c.f).catch(() => { }); } }
prefetchClip.done = new Set();
function stopNarration() { narToken++; if (TTS) try { TTS.cancel(); } catch (e) { } try { player.pause(); } catch (e) { } }
function narrate(reel, resume) {
  stopNarration(); reel.nar = null;
  if (audioMode() < 2 || !audioUnlocked || document.hidden || reel.pausedAt) return;
  const c = clipFor(reel.t, reel.idx);
  if (c) return playClip(reel, c, resume);
  speakTTS(reel);
}
function playClip(reel, c, resume) {
  const tok = narToken, nar = reel.nar = { state: 'speaking', est: c.d + .3, idx: reel.idx, endEl: 0, clip: true };
  const finish = ok => { if (tok !== narToken || nar.state !== 'speaking') return; nar.state = ok ? 'done' : 'failed'; nar.endEl = slideElapsed(reel); };
  const fallback = () => { if (tok !== narToken) return; reel.nar = null; speakTTS(reel); };
  const src = 'audio/' + c.f;
  if (!player.src.endsWith(src)) { player.src = src; }
  else if (!resume) player.currentTime = 0;
  player.onended = () => finish(true); player.onerror = fallback;
  setTimeout(() => finish(true), (c.d * 1.6 + 5) * 1000);
  const p = player.play(); if (p && p.catch) p.catch(e => { if (e && e.name === 'NotAllowedError') { if (tok === narToken) reel.nar = null; } else fallback(); });
  prefetchClip(reel.t, reel.idx + 1);
}
function speakTTS(reel) {
  if (!TTS) return;
  const text = clean(narrationText(reel.t, reel.idx)); if (!text) return;
  const tok = narToken, u = new SpeechSynthesisUtterance(text), est = 1.2 + text.length / 14;
  if (voice) u.voice = voice; u.lang = voice?.lang || 'en-US'; u.rate = 0.95; u.pitch = 1;
  const nar = reel.nar = { state: 'speaking', est, idx: reel.idx, endEl: 0 };
  const finish = ok => { if (tok !== narToken || nar.state !== 'speaking') return; nar.state = ok ? 'done' : 'failed'; nar.endEl = slideElapsed(reel); };
  u.onend = () => finish(true); u.onerror = e => finish(e.error === 'interrupted' || e.error === 'canceled');
  nar.safety = setTimeout(() => finish(true), (est * 1.8 + 3) * 1000);   // some browsers never fire onend
  // speak synchronously (Safari needs it inside the tap); Chrome sometimes drops a speak() right after cancel(), so retry once
  let started = false; u.onstart = () => { started = true; };
  try { TTS.resume(); TTS.speak(u); } catch (e) { finish(false); }
  setTimeout(() => { if (tok === narToken && !started && nar.state === 'speaking') { try { TTS.cancel(); TTS.resume(); TTS.speak(u); } catch (e) { } } }, 700);
}
const slideElapsed = reel => ((reel.pausedAt || performance.now()) - reel.enteredAt - reel.pausedFor) / 1000;

function setPaused(reel, on) {
  if (!reel) return; const now = performance.now();
  if (on && !reel.pausedAt) { reel.pausedAt = now; reel.el.classList.add('is-paused'); if (reel === cur && reel.nar?.state === 'speaking') { stopNarration(); reel.nar.state = 'resume'; } sfx('pause'); }
  else if (!on && reel.pausedAt) { reel.pausedFor += now - reel.pausedAt; reel.pausedAt = 0; reel.el.classList.remove('is-paused'); if (reel === cur && reel.nar?.state === 'resume') narrate(reel, true); }
}
function gainXP(n, x, y) {
  const before = Math.floor((M.xp || 0) / 100); M.xp = (M.xp || 0) + n;
  const day = new Date().toDateString(), yest = new Date(Date.now() - DAY).toDateString();
  if (M.lastDay !== day) { M.streak = M.lastDay === yest ? (M.streak || 0) + 1 : 1; M.lastDay = day; }
  save();
  const f = document.createElement('div'); f.className = 'xpfly'; f.textContent = `+${n} ⚡`; f.style.left = x + 'px'; f.style.top = y + 'px';
  $('#app').appendChild(f); setTimeout(() => f.remove(), 1300);
  document.querySelectorAll('.hud .xp').forEach(e => { e.textContent = '⚡ ' + M.xp; e.classList.remove('bump'); void e.offsetWidth; e.classList.add('bump'); });
  document.querySelectorAll('.hud .streak').forEach(e => e.textContent = '🔥 ' + M.streak);
  if (Math.floor(M.xp / 100) > before) setTimeout(() => { toast(`🏆 Level ${Math.floor(M.xp / 100) + 1}!`); confetti(160); sfx('level'); }, 700);
}
function onRecap(reel) {
  const box = reel.el.querySelector('.end-status');
  if (reel.doneNow) return;
  const content = reel.dwell.slice(1, reel.n - 1).reduce((a, b) => a + b, 0), need = MIN_SEC_PER_SLIDE * reel.t.slides.length;
  const r = rec(reel.t.id);
  if (content >= need) {
    reel.doneNow = true;
    const first = r.status !== 'done', un = markDone(reel.t.id);
    box.innerHTML = `<div class="done-msg">✓ Added to your brain</div>` + (un.length ? `<div class="unlocked"><div class="ulab">🔓 UNLOCKED</div><div class="chips">${un.map(u => chip(u, 'unlock')).join('')}</div></div>` : '');
    confetti(un.length ? 140 : 90); sfx('chime'); if (un.length) setTimeout(() => sfx('pop'), 450);
    const rc = $('#app').getBoundingClientRect();
    setTimeout(() => gainXP(first ? 20 + 5 * un.length : 8, rc.width / 2 - 30, rc.height * .45), 250);
    if (navigator.vibrate) navigator.vibrate(30);
  } else {
    if (r.status !== 'done') r.status = 'skimmed'; else r.again = true;
    scheduleAgain(reel.t.id, true); save(); sfx('skim');
    box.innerHTML = `<div class="skim-msg">💨 Speed-run in ${Math.round(content)}s — no XP, and it’ll come back later. Swipe ← to look again.</div>`;
  }
}

// ---------- confetti ----------
const fx = document.createElement('canvas'); fx.id = 'fx'; let parts = [], fxOn = false;
function confetti(n = 100) {
  const app = $('#app'); if (!fx.isConnected) app.appendChild(fx);
  const w = app.clientWidth, h = app.clientHeight, cols = ['#ff5c8a', '#ffc94d', '#3dd6c6', '#7b8cff', '#c084fc', '#4ade80'];
  fx.width = w * 2; fx.height = h * 2;
  for (let i = 0; i < n; i++) parts.push({ x: w / 2, y: h * .42, vx: (Math.random() - .5) * 14, vy: -Math.random() * 13 - 4, r: Math.random() * 6 + 3, c: cols[i % cols.length], a: Math.random() * 6, va: (Math.random() - .5) * .4, life: 1 });
  if (!fxOn) { fxOn = true; requestAnimationFrame(fxStep); }
}
function fxStep() {
  const c = fx.getContext('2d'); c.setTransform(2, 0, 0, 2, 0, 0); c.clearRect(0, 0, fx.width, fx.height);
  parts.forEach(p => { p.vy += .35; p.vx *= .99; p.x += p.vx; p.y += p.vy; p.a += p.va; p.life -= .008;
    c.save(); c.translate(p.x, p.y); c.rotate(p.a); c.globalAlpha = Math.max(0, p.life); c.fillStyle = p.c; c.fillRect(-p.r / 2, -p.r / 4, p.r, p.r / 2); c.restore(); });
  parts = parts.filter(p => p.life > 0 && p.y < fx.height / 2 + 20);
  if (parts.length) requestAnimationFrame(fxStep); else { fxOn = false; c.clearRect(0, 0, fx.width, fx.height); }
}
function updateLike(reel) { const on = !!M.t[reel.t.id]?.liked; $('#btnLike').classList.toggle('on', on); $('#likeIco').textContent = on ? '❤️' : '🤍'; }
function finalize(reel) {
  if (!reel || reel.finalized) return;
  reel.finalized = true;
  reel.dwell[reel.idx] += Math.min(60, (performance.now() - reel.enteredAt) / 1000);
  const id = reel.t.id, r = rec(id);
  r.views = (r.views || 0) + 1; r.lastSeen = Date.now(); M.seq++;
  if (r.status !== 'done') r.maxSlide = reel.max;
  if (!reel.doneNow && reel.max < reel.n - 1) {
    if (r.status === 'done') { if (reel.max >= 1) r.again = true; }   // a refresher left midway: keep "done", show again
    else if (r.status === 'skimmed' || r.status === 'partial') { }   // already pending
    else if (reel.max >= 1) r.status = 'partial';                     // left a new reel midway
    // (swiped away on the cover of a new reel: stays "new", no penalty)
    r.skipSeq = M.seq; r.skipAt = Date.now();                        // cool down before offering it again
    if (pendingAgain(r)) scheduleAgain(id, false);
  }
  save();
}

// ---------- feed / navigation ----------
const stage = $('#stage');
let hist = [], pos = -1, cur = null, busy = false;
function show(entryIdx, dir) {
  if (busy) return;
  const want = BY[hist[entryIdx].id];
  if (!STORY.reels[want.id]) { busy = true; toast('Loading ' + want.title + '…'); loadStory(want.id).then(() => { busy = false; show(entryIdx, dir); }, () => { busy = false; toast('Could not load this reel'); }); return; }
  prepStory(want);
  const old = cur;
  if (old) { finalize(old); hist[pos].idx = old.doneNow ? 0 : old.idx; }
  pos = entryIdx;
  const e = hist[pos]; stopNarration(); cur = buildReel(BY[e.id], e.idx || 0); if (dir) sfx('whoosh'); narrate(cur);
  stage.appendChild(cur.el);
  if (old && dir) {
    busy = true; old.el.classList.add('leave-' + dir); cur.el.classList.add('enter-' + dir);
    setTimeout(() => { old.el.remove(); cur.el.classList.remove('enter-' + dir); busy = false; }, 330);
  } else if (old) old.el.remove();
}
function nextReel() { if (pos < hist.length - 1) show(pos + 1, 'up'); else { hist.push({ id: pickNext(), idx: 0 }); show(hist.length - 1, 'up'); } }
function prevReel() { if (pos > 0) show(pos - 1, 'down'); else toast('This is the first reel'); }
function openTopic(id) {
  if (cur && cur.t.id === id) return;
  hist.splice(pos + 1, 0, { id, idx: 0 }); show(pos + 1, 'up');
  if (hist.length - pos > 1) toast('Swipe down to go back');
}
function nextSlide() { if (!cur) return; if (cur.idx < cur.n - 1) setSlide(cur, cur.idx + 1); else nextReel(); }
function prevSlide() { if (!cur) return; if (cur.idx > 0) setSlide(cur, cur.idx - 1); }

// gestures
let g = null;
function canvasPt(cv, e) { const r = cv.getBoundingClientRect(), h = +(cv.dataset.h || 260); return { x: (e.clientX - r.left) * 400 / r.width, y: (e.clientY - r.top) * h / r.height }; }
let drag = null;
stage.addEventListener('pointerdown', e => {
  if (e.target.closest('.ctl')) { g = null; return; }
  const cv = e.target.closest('canvas.touch');
  if (cv && cur) { g = null; drag = cv; const q = canvasPt(cv, e); (cur.S.taps ||= []).push(q); cur.S.ptr = { ...q, down: true }; try { cv.setPointerCapture(e.pointerId); } catch (_) { } sfx('tick'); return; }
}, true);
stage.addEventListener('pointermove', e => { if (drag && cur) { cur.S.ptr = { ...canvasPt(drag, e), down: true }; } }, true);
addEventListener('pointerup', () => { if (drag && cur && cur.S.ptr) cur.S.ptr.down = false; drag = null; }, true);
stage.addEventListener('input', e => {
  if (e.target.dataset && e.target.dataset.sk && cur) { const sl = cur.t.slides[cur.idx - 1], c = sl && sl.step.ctl; if (!c) return;
    const S = cur.S; S.prevCi = S.ci; S.ci = +e.target.value; S.changed = Math.max(0, performance.now() - cur.t0 - cur.pausedFor) / 1000; c.onChange?.(S);
    e.target.nextElementSibling.textContent = ctlFmt(c, S.ci); return; }
  const k = e.target.dataset && e.target.dataset.k; if (!k || !cur) return; (cur.S.ctl ||= {})[k] = +e.target.value; e.target.nextElementSibling.textContent = e.target.value; });
stage.addEventListener('pointerdown', e => { if (e.target.closest('button') || e.target.closest('.ctl') || e.target.closest('canvas.touch')) return; g = { x: e.clientX, y: e.clientY, t: performance.now(), held: false }; const gg = g; setTimeout(() => { if (g === gg && !gg.moved) { gg.held = true; setPaused(cur, true); } }, 220); });
stage.addEventListener('pointermove', e => {
  if (!g || !cur || busy) return; const dy = e.clientY - g.y, dx = e.clientX - g.x;
  if (Math.abs(dx) > 10 || Math.abs(dy) > 10) g.moved = true;
  if (Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > 10) { cur.el.style.transition = 'none'; cur.el.style.transform = `translateY(${dy * .35}px)`; }
});
stage.addEventListener('pointerup', e => {
  if (!g) return; const dx = e.clientX - g.x, dy = e.clientY - g.y, held = g.held; g = null;
  if (held) { setPaused(cur, false); if (Math.abs(dx) < 40 && Math.abs(dy) < 50) return; }
  if (cur) { cur.el.style.transform = ''; cur.el.style.transition = ''; }
  if (Math.abs(dy) > 50 && Math.abs(dy) > Math.abs(dx)) return dy < 0 ? nextReel() : prevReel();
  if (Math.abs(dx) > 40) return dx < 0 ? nextSlide() : prevSlide();
  if (Math.abs(dx) < 10 && Math.abs(dy) < 10) { const r = stage.getBoundingClientRect(); (e.clientX - r.left < r.width * .3) ? prevSlide() : nextSlide(); }
});
stage.addEventListener('pointercancel', () => { g = null; if (cur) { cur.el.style.transform = ''; setPaused(cur, false); } });
stage.addEventListener('click', e => { const b = e.target.closest('[data-open]'); if (b) { e.stopPropagation(); openTopic(b.dataset.open); } });
let wheelAcc = 0, wheelLock = 0;
stage.addEventListener('wheel', e => {
  e.preventDefault(); const now = performance.now(); if (now < wheelLock) return;
  if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) { wheelAcc += e.deltaX; if (Math.abs(wheelAcc) > 60) { wheelAcc > 0 ? nextSlide() : prevSlide(); wheelAcc = 0; wheelLock = now + 350; } return; }
  wheelAcc += e.deltaY; if (Math.abs(wheelAcc) > 80) { wheelAcc > 0 ? nextReel() : prevReel(); wheelAcc = 0; wheelLock = now + 600; }
}, { passive: false });
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (!$('#lib').classList.contains('hidden')) { if (e.key === 'Escape') closeLib(); return; }
  if (e.key === 'ArrowRight' || e.key === ' ') nextSlide(); else if (e.key === 'ArrowLeft') prevSlide();
  else if (e.key === 'ArrowDown' || e.key === 'j') nextReel(); else if (e.key === 'ArrowUp' || e.key === 'k') prevReel();
  else if (e.key === 'l') openLib(); else if (e.key === 'p') setPaused(cur, !cur.pausedAt); else return; e.preventDefault();
});
document.addEventListener('visibilitychange', () => { if (!cur) return; if (document.hidden) setPaused(cur, true); else if ($('#lib').classList.contains('hidden')) setPaused(cur, false); });
window.addEventListener('pagehide', () => finalize(cur));

// ---------- animation loop ----------
function frame() {
  requestAnimationFrame(frame); const now = performance.now();
  if (!cur || document.hidden) return;
  const libOpen = !$('#lib').classList.contains('hidden');
  if (libOpen && !cur.pausedAt) setPaused(cur, true);
  let dur = slideDur(cur.t, cur.idx); const el = ((cur.pausedAt || now) - cur.enteredAt - cur.pausedFor) / 1000, nar = cur.nar;
  let pct = Math.min(100, 100 * el / dur);
  if (nar && nar.idx === cur.idx) {                          // narration sets the pace
    if (nar.state === 'speaking' || nar.state === 'resume') { dur = Infinity; pct = Math.min(96, 100 * el / nar.est); }
    else if (nar.state === 'done') { dur = Math.max(nar.endEl + 2.2, 3.5); pct = Math.min(100, 96 + 4 * (el - nar.endEl)); }
  }
  const bar = cur.el.querySelectorAll('.bars u')[cur.idx];
  if (bar) bar.style.width = (cur.idx === cur.n - 1 || !M.autoplay ? 100 : pct) + '%';
  if (M.autoplay && dur && el >= dur && cur.idx < cur.n - 1 && !busy && !g) setSlide(cur, cur.idx + 1);
  const cv = cur.canvases[cur.idx]; if (!cv) return;
  const w = cv.clientWidth; if (!w) return;
  const s = cur.t.slides[cur.idx - 1], LH = s.step ? 420 : 260;
  const dpr = Math.min(2, window.devicePixelRatio || 1), W = Math.round(w * dpr), H = Math.round(w * LH / 400 * dpr);
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const c = cv.getContext('2d');
  if (s.step) {
    const S = cur.S, tt = Math.max(0, (cur.pausedAt || now) - cur.t0 - cur.pausedFor) / 1000, R = STORY.reels[cur.t.id];
    if (s.step.ctl && S.ci == null) { const inp = cur.el.querySelectorAll('.slide')[cur.idx].querySelector('input[data-sk]'); S.ci = inp ? +inp.value : (s.step.ctl.def ?? 0); }
    S.taps ||= [];
    c.setTransform(W / 400, 0, 0, H / 420, 0, 0); c.fillStyle = SL.C.bg; c.fillRect(0, 0, 400, 420);
    try { SL.setCtx(c); R.onCtx?.(c); while (S.taps.length) { const q = S.taps.shift(); s.step.tap?.(q.x, q.y, S, tt); } s.step.draw(tt, S); }
    catch (err) { if (!cv.dataset.err) { console.error(cur.t.id, s.p.k, err); cv.dataset.err = 1; } }
    return;
  }
  c.setTransform(W / 400, 0, 0, H / 260, 0, 0); c.fillStyle = '#0b0e1d'; c.fillRect(0, 0, 400, 260);
  if (!cur.S.ctl) { cur.S.ctl = {}; cur.el.querySelectorAll('.slide')[cur.idx].querySelectorAll('.ctl input').forEach(x => cur.S.ctl[x.dataset.k] = +x.value); }
  cur.S.taps ||= [];
  try { A[s.a](c, Math.max(0, (cur.pausedAt || now) - cur.t0 - cur.pausedFor) / 1000 * (RLA.SPEED[s.a] ?? 1), cur.S, s.p || {}); } catch (err) { if (!cv.dataset.err) { console.error(s.a, err); cv.dataset.err = 1; } }
}
requestAnimationFrame(frame);

// ---------- side buttons, toast ----------
let toastT;
function toast(msg) { const el = $('#toast'); el.textContent = msg; el.classList.add('show'); clearTimeout(toastT); toastT = setTimeout(() => el.classList.remove('show'), 1800); }
function updateCount() { $('#memCount').textContent = TOPICS.filter(t => isDone(t.id)).length + '/' + TOPICS.length; }
$('#btnAgain').onclick = () => {
  if (!cur) return; const r = rec(cur.t.id);
  if (r.status === 'done') r.again = true; else r.status = 'partial';
  scheduleAgain(cur.t.id, true); cur.doneNow = true; save(); toast('↻ I’ll show this one again soon');
};
const AUDIO_UI = [['🔇', 'Muted'], ['🎵', 'FX only'], ['🔊', 'Voice']];
function updateAudioBtn() { const [i, l] = AUDIO_UI[audioMode()]; $('#audIco').textContent = i; $('#audLab').textContent = l; }
$('#btnAudio').onclick = () => { M.audio = (audioMode() + 2) % 3; save(); updateAudioBtn(); unlockAudio(); if (audioMode() === 2 && cur) narrate(cur); else stopNarration(); if (cur && audioMode() < 2) cur.nar = null; toast(AUDIO_UI[audioMode()][0] + ' ' + ['Sound off', 'Sound effects only', 'Narration + sound effects'][audioMode()]); };
updateAudioBtn();
function updateAutoBtn() { $('#autoIco').textContent = M.autoplay ? '⏸' : '▶'; $('#autoLab').textContent = M.autoplay ? 'Auto' : 'Manual'; }
$('#btnAuto').onclick = () => { M.autoplay = !M.autoplay; save(); updateAutoBtn(); if (cur) { cur.enteredAt = performance.now(); cur.pausedFor = 0; if (cur.pausedAt) cur.pausedAt = cur.enteredAt; } toast(M.autoplay ? '▶ Autoplay on — hold the screen to pause' : '⏸ Autoplay off — tap to move on'); };
updateAutoBtn();
$('#btnLib').onclick = openLib;
$('#btnLike').onclick = e => {
  if (!cur) return; const r = rec(cur.t.id); r.liked = !r.liked; save(); updateLike(cur);
  if (r.liked) { sfx('like'); const b = e.currentTarget.getBoundingClientRect(), a = $('#app').getBoundingClientRect();
    for (let i = 0; i < 6; i++) { const h = document.createElement('div'); h.className = 'heart'; h.textContent = '❤️'; h.style.left = (b.left - a.left + 12 + (Math.random() - .5) * 30) + 'px'; h.style.top = (b.top - a.top) + 'px'; h.style.animationDelay = i * 60 + 'ms'; $('#app').appendChild(h); setTimeout(() => h.remove(), 1400); } }
};

// ---------- library ----------
function statusLine(t) {
  const r = M.t[t.id];
  if (!unlocked(t)) return ['🔒', 'needs: ' + t.pre.filter(p => !isDone(p)).map(p => BY[p].title).join(', ')];
  if (!r || !r.status) return ['🆕', 'ready to watch'];
  if (r.status === 'done') return [r.again ? '↻' : '✅', `watched · ${r.views || 1} view${r.views > 1 ? 's' : ''}${r.again ? ' · coming back soon' : ''}`];
  if (r.status === 'skimmed') return ['💨', 'skimmed — will come back'];
  return ['◐', `left at slide ${(r.maxSlide || 0) + 1} — will come back`];
}
function openLib() {
  const done = TOPICS.filter(t => isDone(t.id)).length, ready = TOPICS.filter(t => unlocked(t) && !st(t.id)).length, again = TOPICS.filter(t => pendingAgain(M.t[t.id])).length;
  $('#libStats').innerHTML = `<div class="stats"><span>⚡ <b>${M.xp || 0}</b> XP · level ${Math.floor((M.xp || 0) / 100) + 1}</span><span>🔥 <b>${M.streak || 0}</b>-day streak</span></div>${done} of ${TOPICS.length} watched · ${ready} ready · ${again} to revisit<div class="meter"><i style="width:${100 * done / TOPICS.length}%"></i></div>`;
  const groups = {}; TOPICS.forEach(t => (groups[t.track] = groups[t.track] || []).push(t));
  $('#libVer').textContent = 'RL Reels v' + VERSION;
  $('#libBody').innerHTML = Object.entries(groups).map(([g, ts]) => `<div class="grp"><h3>${esc(g)}</h3>${ts.map(t => { const [ic, line] = statusLine(t);
    return `<button class="row ${unlocked(t) ? '' : 'locked'}" data-lib="${t.id}"><span class="e">${t.emoji}</span><span class="tt"><b>${esc(t.title)}${M.t[t.id]?.liked ? ' ❤️' : ''}</b><small>${esc(line)}</small></span><span class="st">${ic}</span></button>`; }).join('')}</div>`).join('');
  $('#lib').classList.remove('hidden');
}
function closeLib() { $('#lib').classList.add('hidden'); setPaused(cur, false); }
$('#libBody').addEventListener('click', e => { const b = e.target.closest('[data-lib]'); if (b) { closeLib(); openTopic(b.dataset.lib); } });
document.querySelectorAll('[data-close]').forEach(b => b.onclick = closeLib);
$('#btnExport').onclick = () => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(M, null, 1)], { type: 'application/json' })); a.download = 'rl-reels-memory.json'; a.click(); };
$('#btnImport').onclick = () => $('#fileImport').click();
$('#fileImport').onchange = async e => { try { const m = JSON.parse(await e.target.files[0].text()); if (!m.t) throw 0; M = m; save(); openLib(); toast('Memory imported'); } catch (err) { toast('Not a valid memory file'); } };
$('#btnReset').onclick = () => { if (confirm('Forget everything you have watched?')) { if (cur) cur.finalized = true; M = { t: {}, seq: 0 }; save(); location.reload(); } };

// ---------- start ----------
function nextReelInitial() { hist.push({ id: pickNext(), idx: 0 }); show(0); }
updateCount(); nextReelInitial();
window.__rlOpen = id => openTopic(id);   // test hook (tools/app_shot.py)
window.__rlDebug = () => ({ id: cur && cur.t.id, locked: cur && !unlocked(cur.t), seq: M.seq, idx: cur && cur.idx, nar: cur && cur.nar && { state: cur.nar.state, clip: !!cur.nar.clip }, src: player.src.split('/').pop(), clips: CLIPS ? Object.keys(CLIPS).length : 0, unlocked: audioUnlocked });
if ('serviceWorker' in navigator && location.protocol.startsWith('http')) {
  const had = !!navigator.serviceWorker.controller; let reloaded = false;
  navigator.serviceWorker.addEventListener('controllerchange', () => { if (had && !reloaded) { reloaded = true; location.reload(); } });
  navigator.serviceWorker.register('sw.js').then(r => r.update()).catch(() => { });
}
})();
