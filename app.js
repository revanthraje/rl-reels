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
function pickNext() {
  const now = Date.now(), recent = new Set(hist.slice(Math.max(0, pos - 3), pos + 1).map(h => h.id));
  const pool = TOPICS.filter(t => !recent.has(t.id) && unlocked(t));
  const due = pool.filter(t => { const r = M.t[t.id]; return pendingAgain(r) && (M.seq >= (r.dueSeq || 0) || now >= (r.dueAt || 0)); });
  const fresh = pool.filter(t => !st(t.id));
  const review = pool.filter(t => isDone(t.id) && !M.t[t.id].again && now >= (M.t[t.id].dueAt || 0));
  const R = Math.random();
  if (due.length && (R < .45 || !fresh.length)) return due.sort((a, b) => (M.t[a.id].lastSeen || 0) - (M.t[b.id].lastSeen || 0))[0].id;
  if (review.length && (R > .88 || !fresh.length)) return review[Math.floor(Math.random() * review.length)].id;
  if (fresh.length) {
    // prefer continuing a chain: topics whose prerequisite was just finished
    const justDone = new Set(TOPICS.filter(t => isDone(t.id) && now - (M.t[t.id].doneAt || 0) < 30 * 60e3 && hist.slice(-3).some(h => h.id === t.id)).map(t => t.id));
    const w = fresh.map(t => t.pre.some(p => justDone.has(p)) ? 4 : 1), s = w.reduce((a, b) => a + b);
    let u = Math.random() * s; for (let i = 0; i < fresh.length; i++) { u -= w[i]; if (u <= 0) return fresh[i].id; }
    return fresh[0].id;
  }
  // nothing new or due: something not seen recently, least recently seen first
  const any = (pool.length ? pool : TOPICS.filter(unlocked)).slice().sort((a, b) => (M.t[a.id]?.lastSeen || 0) - (M.t[b.id]?.lastSeen || 0));
  return any[0].id;
}
function markDone(id) {
  const before = new Set(TOPICS.filter(unlocked).map(t => t.id));
  const r = rec(id), was = r.status === 'done';
  r.level = was ? Math.min((r.level || 0) + 1, REVIEW_DAYS.length - 1) : 0;
  r.status = 'done'; r.doneAt = Date.now(); r.dueAt = Date.now() + REVIEW_DAYS[r.level] * DAY; delete r.again; delete r.dueSeq;
  save();
  return TOPICS.filter(t => unlocked(t) && !before.has(t.id)).map(t => t.id);
}
function scheduleAgain(id, soon) {
  const r = rec(id);
  r.dueSeq = M.seq + (soon ? 3 : 4); r.dueAt = Date.now() + (soon ? 2 : 6) * 3600e3;
}

// ---------- rendering ----------
const esc = s => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const md = s => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`(.+?)`/g, '<code>$1</code>');
function badgeFor(t) {
  const r = M.t[t.id];
  if (!unlocked(t)) return '<span class="badge locked">🔒 EARLY</span>';
  if (!r || !r.status) return '<span class="badge new">NEW</span>';
  if (r.status === 'done') return r.again ? '<span class="badge again">AGAIN</span>' : '<span class="badge review">REVIEW</span>';
  return '<span class="badge again">AGAIN</span>';
}
function chip(id, cls = '') { const t = BY[id]; return `<button class="chip ${cls || (isDone(id) ? 'done' : 'todo')}" data-open="${id}">${t.emoji} ${esc(t.title)}${isDone(id) ? ' ✓' : ''}</button>`; }

function buildReel(t, startIdx = 0) {
  const r = M.t[t.id], n = t.slides.length + 2;
  const el = document.createElement('div'); el.className = 'reel';
  let note = '';
  if (r && (r.status === 'partial')) note = `<div class="note">You left this one at slide ${(r.maxSlide || 0) + 1}/${n} — here it is again.</div>`;
  else if (r && r.status === 'skimmed') note = `<div class="note">You flew through this last time — one more calm look?</div>`;
  else if (r && r.status === 'done') note = `<div class="note">Quick refresher — you watched this ${r.views} time${r.views > 1 ? 's' : ''}.</div>`;
  const lockNote = unlocked(t) ? '' : `<div class="note">Normally shown after: ${t.pre.filter(p => !isDone(p)).map(p => BY[p].title).join(', ')}</div>`;
  const slides = [
    `<section class="slide cover"><div class="big">${t.emoji}</div><h1>${esc(t.title)}</h1><span class="pill">${esc(t.track)}</span>
      <div class="hook">${md(t.hook)}</div>${note}${lockNote}
      ${t.pre.length ? `<div class="chips">${t.pre.map(p => chip(p)).join('')}</div>` : ''}
      <div class="hint">tap → or swipe ← to start</div></section>`,
    ...t.slides.map((s, i) => `<section class="slide"><div class="num">${i + 1} / ${t.slides.length}</div>${s.h ? `<h2>${md(s.h)}</h2>` : ''}
      ${s.a ? `<canvas data-i="${i + 1}"></canvas>` : ''}${s.t ? `<p>${md(s.t)}</p>` : ''}${s.ex ? `<div class="ex">${md(s.ex)}</div>` : ''}</section>`),
    `<section class="slide recap"><h2>${t.emoji} In a nutshell</h2><ul>${t.recap.map(x => `<li>${md(x)}</li>`).join('')}</ul>
      <div class="end-status"></div><div class="swipe-up">↑ swipe up for the next reel</div></section>`
  ];
  el.innerHTML = `<div class="bars">${Array.from({ length: n }, () => '<i></i>').join('')}</div>
    <div class="top"><div class="av"><b>${t.emoji}</b></div><div class="who"><b>${esc(t.title)}</b><small>${esc(t.track)}</small></div>${badgeFor(t)}</div>
    <div class="track">${slides.join('')}</div>
    ${t.pre.length ? `<div class="foot"><span class="lab">↩ builds on</span>${t.pre.map(p => chip(p)).join('')}</div>` : ''}`;
  const reel = { t, el, n, idx: -1, max: startIdx, dwell: new Array(n).fill(0), enteredAt: 0, finalized: false, doneNow: false, canvases: {}, t0: 0, S: {} };
  el.querySelectorAll('canvas').forEach(cv => reel.canvases[+cv.dataset.i] = cv);
  setSlide(reel, startIdx, true);
  return reel;
}
function setSlide(reel, i, instant) {
  i = Math.max(0, Math.min(reel.n - 1, i));
  const now = performance.now();
  if (reel.idx >= 0) reel.dwell[reel.idx] += Math.min(60, (now - reel.enteredAt) / 1000);
  reel.idx = i; reel.enteredAt = now; reel.t0 = now; reel.S = {}; reel.max = Math.max(reel.max, i);
  const tr = reel.el.querySelector('.track');
  if (instant) { tr.style.transition = 'none'; requestAnimationFrame(() => tr.style.transition = ''); }
  tr.style.transform = `translateX(${-i * 100}%)`;
  reel.el.querySelectorAll('.bars i').forEach((b, k) => b.classList.toggle('on', k <= i));
  if (i === reel.n - 1) onRecap(reel);
  const r = M.t[reel.t.id]; if (r && r.status !== 'done') { r.maxSlide = Math.max(r.maxSlide || 0, i); save(); }
}
function onRecap(reel) {
  const box = reel.el.querySelector('.end-status');
  if (reel.doneNow) return;
  const content = reel.dwell.slice(1, reel.n - 1).reduce((a, b) => a + b, 0), need = MIN_SEC_PER_SLIDE * reel.t.slides.length;
  const r = rec(reel.t.id);
  if (content >= need) {
    reel.doneNow = true;
    const un = markDone(reel.t.id);
    box.innerHTML = `<div class="done-msg">✓ Saved to your memory</div>` + (un.length ? `<div class="num" style="margin-top:8px">Unlocked</div><div class="chips">${un.map(u => chip(u, 'unlock')).join('')}</div>` : '');
    if (un.length) toast(`🔓 ${un.length} new topic${un.length > 1 ? 's' : ''} unlocked`);
  } else {
    if (r.status !== 'done') r.status = 'skimmed'; else r.again = true;
    scheduleAgain(reel.t.id, true); save();
    box.innerHTML = `<div class="skim-msg">That was quick (${Math.round(content)}s) — I’ll bring this one back later. Swipe back ← to look again.</div>`;
  }
}
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
stage.addEventListener('pointerdown', e => { if (e.target.closest('button')) return; g = { x: e.clientX, y: e.clientY, t: performance.now() }; });
stage.addEventListener('pointermove', e => {
  if (!g || !cur || busy) return; const dy = e.clientY - g.y, dx = e.clientX - g.x;
  if (Math.abs(dy) > Math.abs(dx) && Math.abs(dy) > 10) { cur.el.style.transition = 'none'; cur.el.style.transform = `translateY(${dy * .35}px)`; }
});
stage.addEventListener('pointerup', e => {
  if (!g) return; const dx = e.clientX - g.x, dy = e.clientY - g.y; g = null;
  if (cur) { cur.el.style.transform = ''; cur.el.style.transition = ''; }
  if (Math.abs(dy) > 50 && Math.abs(dy) > Math.abs(dx)) return dy < 0 ? nextReel() : prevReel();
  if (Math.abs(dx) > 40) return dx < 0 ? nextSlide() : prevSlide();
  if (Math.abs(dx) < 10 && Math.abs(dy) < 10) { const r = stage.getBoundingClientRect(); (e.clientX - r.left < r.width * .3) ? prevSlide() : nextSlide(); }
});
stage.addEventListener('pointercancel', () => { g = null; if (cur) cur.el.style.transform = ''; });
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
  else if (e.key === 'l') openLib(); else return; e.preventDefault();
});
document.addEventListener('visibilitychange', () => { if (document.hidden && cur) { cur.dwell[cur.idx] += Math.min(60, (performance.now() - cur.enteredAt) / 1000); cur.enteredAt = performance.now(); } else if (cur) cur.enteredAt = performance.now(); });
window.addEventListener('pagehide', () => finalize(cur));

// ---------- animation loop ----------
function frame() {
  requestAnimationFrame(frame); const now = performance.now();
  if (!cur || document.hidden) return;
  const cv = cur.canvases[cur.idx]; if (!cv) return;
  const w = cv.clientWidth; if (!w) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1), W = Math.round(w * dpr), H = Math.round(w * 260 / 400 * dpr);
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const c = cv.getContext('2d'), s = cur.t.slides[cur.idx - 1];
  c.setTransform(W / 400, 0, 0, H / 260, 0, 0); c.fillStyle = '#0b0e1d'; c.fillRect(0, 0, 400, 260);
  try { A[s.a](c, Math.max(0, now - cur.t0) / 1000, cur.S, s.p || {}); } catch (err) { if (!cv.dataset.err) { console.error(s.a, err); cv.dataset.err = 1; } }
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
  $('#libStats').innerHTML = `${done} of ${TOPICS.length} watched · ${ready} ready · ${again} to revisit<div class="meter"><i style="width:${100 * done / TOPICS.length}%"></i></div>`;
  const groups = {}; TOPICS.forEach(t => (groups[t.track] = groups[t.track] || []).push(t));
  $('#libBody').innerHTML = Object.entries(groups).map(([g, ts]) => `<div class="grp"><h3>${esc(g)}</h3>${ts.map(t => { const [ic, line] = statusLine(t);
    return `<button class="row ${unlocked(t) ? '' : 'locked'}" data-lib="${t.id}"><span class="e">${t.emoji}</span><span class="tt"><b>${esc(t.title)}</b><small>${esc(line)}</small></span><span class="st">${ic}</span></button>`; }).join('')}</div>`).join('');
  $('#lib').classList.remove('hidden');
}
function closeLib() { $('#lib').classList.add('hidden'); }
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
