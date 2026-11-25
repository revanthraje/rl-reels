// RL Reels — swipeable micro-lessons with a memory of what you've watched.
'use strict';
(() => {
const TOPICS = window.RLTopics, BY = Object.fromEntries(TOPICS.map(t => [t.id, t]));
const { A } = window.RLAnim;
const KEY = 'rlreels.v1', DAY = 864e5;
const MIN_SEC_PER_SLIDE = 1.5;         // reaching the end faster than this per content slide counts as "skimmed"
const REVIEW_DAYS = [3, 8, 21, 60];    // spaced re-showing of watched reels
const $ = s => document.querySelector(s);

// sanity check on content
for (const t of TOPICS) {
  if (t.slides.length + 2 > 10) console.warn('reel too long', t.id);
  for (const p of t.pre) if (!BY[p]) console.warn('unknown prereq', t.id, p);
  for (const s of t.slides) if (s.a && !A[s.a]) console.warn('unknown anim', t.id, s.a);
}

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
  if (!unlocked(t)) return '<span class="badge locked">🔒 EARLY ACCESS</span>';
  if (!r || !r.status) return '<span class="badge new">✨ NEW</span>';
  if (r.status === 'done') return r.again ? '<span class="badge again">↻ ROUND 2</span>' : '<span class="badge review">🧠 REFRESH</span>';
  return '<span class="badge again">↻ ROUND 2</span>';
}
function chip(id, cls = '') { const t = BY[id]; return `<button class="chip ${cls || (isDone(id) ? 'done' : 'todo')}" data-open="${id}">${t.emoji} ${esc(t.title)}${isDone(id) ? ' ✓' : ''}</button>`; }
const SLIDE_EMOJI = ['💡', '⚙️', '🎯', '🧩', '🚀', '🔍', '🌟', '🧠'];
function slideDur(t, i) { // seconds a slide auto-plays for
  if (i === 0) return 4.5;
  const s = t.slides[i - 1]; if (!s) return 0;
  const chars = (s.h || '').length + (s.t || '').length + (s.ex || '').length;
  return Math.min(16, Math.max(5, 2.5 + chars * .055 + (s.a ? 3.5 : 0)));
}
function buildReel(t, startIdx = 0) {
  const r = M.t[t.id], n = t.slides.length + 2, hue = HUE[t.track] ?? 265;
  const el = document.createElement('div'); el.className = 'reel'; el.style.setProperty('--h', hue);
  let note = '';
  if (r && (r.status === 'partial')) note = `<div class="note">👀 You left at slide ${(r.maxSlide || 0) + 1}/${n} last time</div>`;
  else if (r && r.status === 'skimmed') note = `<div class="note">💨 You speed-ran this one — calmer look?</div>`;
  else if (r && r.status === 'done') note = `<div class="note">🧠 Refresher · watched ${r.views}×</div>`;
  if (!unlocked(t)) note += `<div class="note">Normally after: ${t.pre.filter(p => !isDone(p)).map(p => BY[p].title).join(', ')}</div>`;
  const mins = Math.max(1, Math.round(Array.from({ length: n }, (_, i) => slideDur(t, i)).reduce((a, b) => a + b) / 60));
  const slides = [
    `<section class="slide cover"><div class="halo"><div class="big">${t.emoji}</div></div>${badgeFor(t)}<h1>${esc(t.title)}</h1>
      <div class="hook">${md(t.hook)}</div><div class="meta">${esc(t.track)} · ${n} slides · ~${mins} min</div>${note}
      ${t.pre.length ? `<div class="chips">${t.pre.map(p => chip(p)).join('')}</div>` : ''}
      <div class="hint">tap to start · hold to pause</div></section>`,
    ...t.slides.map((s, i) => `<section class="slide"><div class="num"><span>${SLIDE_EMOJI[i % SLIDE_EMOJI.length]}</span>${i + 1}<small>/${t.slides.length}</small></div>${s.h ? `<h2>${md(s.h)}</h2>` : ''}
      ${s.a ? `<div class="cvwrap"><canvas data-i="${i + 1}"></canvas></div>` : ''}${s.t ? `<p>${md(s.t)}</p>` : ''}${s.ex ? `<div class="ex"><span>💬</span><div>${md(s.ex)}</div></div>` : ''}</section>`),
    `<section class="slide recap"><div class="trophy">${t.emoji}</div><h2>In a nutshell</h2><ul>${t.recap.map((x, k) => `<li style="--k:${k}">${md(x)}</li>`).join('')}</ul>
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
function setSlide(reel, i, instant) {
  i = Math.max(0, Math.min(reel.n - 1, i));
  const now = performance.now();
  if (reel.idx >= 0) reel.dwell[reel.idx] += Math.min(60, (now - reel.enteredAt) / 1000);
  reel.idx = i; reel.enteredAt = now; reel.t0 = now; reel.S = {}; reel.max = Math.max(reel.max, i); reel.pausedFor = 0; reel.pausedAt = 0; reel.el.classList.remove('is-paused');
  const tr = reel.el.querySelector('.track');
  if (instant) { tr.style.transition = 'none'; requestAnimationFrame(() => tr.style.transition = ''); }
  tr.style.transform = `translateX(${-i * 100}%)`;
  reel.el.querySelectorAll('.slide').forEach((s, k) => s.classList.toggle('active', k === i));
  reel.el.querySelectorAll('.bars u').forEach((b, k) => b.style.width = k < i ? '100%' : '0%');
  if (i === reel.n - 1) onRecap(reel);
  const r = M.t[reel.t.id]; if (r && r.status !== 'done') { r.maxSlide = Math.max(r.maxSlide || 0, i); save(); }
}
function setPaused(reel, on) {
  if (!reel) return; const now = performance.now();
  if (on && !reel.pausedAt) { reel.pausedAt = now; reel.el.classList.add('is-paused'); }
  else if (!on && reel.pausedAt) { reel.pausedFor += now - reel.pausedAt; reel.pausedAt = 0; reel.el.classList.remove('is-paused'); }
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
  if (Math.floor(M.xp / 100) > before) setTimeout(() => { toast(`🏆 Level ${Math.floor(M.xp / 100) + 1}!`); confetti(160); }, 700);
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
    confetti(un.length ? 140 : 90);
    const rc = $('#app').getBoundingClientRect();
    setTimeout(() => gainXP(first ? 20 + 5 * un.length : 8, rc.width / 2 - 30, rc.height * .45), 250);
    if (navigator.vibrate) navigator.vibrate(30);
  } else {
    if (r.status !== 'done') r.status = 'skimmed'; else r.again = true;
    scheduleAgain(reel.t.id, true); save();
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
  const old = cur;
  if (old) { finalize(old); hist[pos].idx = old.doneNow ? 0 : old.idx; }
  pos = entryIdx;
  const e = hist[pos]; cur = buildReel(BY[e.id], e.idx || 0);
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
stage.addEventListener('pointerdown', e => { if (e.target.closest('button')) return; g = { x: e.clientX, y: e.clientY, t: performance.now(), held: false }; const gg = g; setTimeout(() => { if (g === gg && !gg.moved) { gg.held = true; setPaused(cur, true); } }, 220); });
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
  if (!$('#lib').classList.contains('hidden')) { if (e.key === 'Escape') closeLib(); return; }
  if (e.key === 'ArrowRight' || e.key === ' ') nextSlide(); else if (e.key === 'ArrowLeft') prevSlide();
  else if (e.key === 'ArrowDown' || e.key === 'j') nextReel(); else if (e.key === 'ArrowUp' || e.key === 'k') prevReel();
  else if (e.key === 'l') openLib(); else if (e.key === 'p') setPaused(cur, !cur.pausedAt); else return; e.preventDefault();
});
document.addEventListener('visibilitychange', () => { if (document.hidden && cur) { cur.dwell[cur.idx] += Math.min(60, (performance.now() - cur.enteredAt) / 1000); cur.enteredAt = performance.now(); } else if (cur) cur.enteredAt = performance.now(); });
window.addEventListener('pagehide', () => finalize(cur));

// ---------- animation loop ----------
function frame() {
  requestAnimationFrame(frame); const now = performance.now();
  if (!cur || document.hidden) return;
  const libOpen = !$('#lib').classList.contains('hidden');
  if (libOpen && !cur.pausedAt) setPaused(cur, true);
  const dur = slideDur(cur.t, cur.idx), el = ((cur.pausedAt || now) - cur.enteredAt - cur.pausedFor) / 1000;
  const bar = cur.el.querySelectorAll('.bars u')[cur.idx];
  if (bar) bar.style.width = (cur.idx === cur.n - 1 ? 100 : Math.min(100, 100 * el / dur)) + '%';
  if (dur && el >= dur && cur.idx < cur.n - 1 && !busy && !g) setSlide(cur, cur.idx + 1);
  const cv = cur.canvases[cur.idx]; if (!cv) return;
  const w = cv.clientWidth; if (!w) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1), W = Math.round(w * dpr), H = Math.round(w * 260 / 400 * dpr);
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const c = cv.getContext('2d'), s = cur.t.slides[cur.idx - 1];
  c.setTransform(W / 400, 0, 0, H / 260, 0, 0); c.fillStyle = '#0b0e1d'; c.fillRect(0, 0, 400, 260);
  try { A[s.a](c, Math.max(0, (cur.pausedAt || now) - cur.t0 - cur.pausedFor) / 1000, cur.S, s.p || {}); } catch (err) { if (!cv.dataset.err) { console.error(s.a, err); cv.dataset.err = 1; } }
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
$('#btnLib').onclick = openLib; $('#btnShare').onclick = openLib;
$('#btnLike').onclick = e => {
  if (!cur) return; const r = rec(cur.t.id); r.liked = !r.liked; save(); updateLike(cur);
  if (r.liked) { const b = e.currentTarget.getBoundingClientRect(), a = $('#app').getBoundingClientRect();
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
if ('serviceWorker' in navigator && location.protocol.startsWith('http')) navigator.serviceWorker.register('sw.js').catch(() => { });
})();
