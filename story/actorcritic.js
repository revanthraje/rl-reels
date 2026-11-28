// Actor-critic as a story: Pip keeps its chances (the actor, softmax preferences as in pg/reinforce) and adds a value
// notebook learned by TD (the critic). After every step the critic's TD error grades the move just made, and the actor
// nudges that move's preferences by it. Compared with REINFORCE + baseline on the same dice seeds (pg_world: rng(5000 + k)).
// Every number comes from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
// ===== pg kit (copied from pg.js) =====
const softmax = th => { const m = Math.max(...th), e = th.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
function fans(M, TH, o = {}) {
  for (let r = 0; r < M.rows; r++) for (let c = 0; c < M.cols; c++) {
    if (M.term(c, r) || M.isRock(c, r)) continue; if (o.only && !o.only(c, r)) continue; const k = M.key(c, r), p = softmax(Array.from(TH.slice(k * 4, k * 4 + 4))), best = p.indexOf(Math.max(...p)), x = M.cx(c), y = M.cy(r) + 4;
    p.forEach((q, m) => { const [dx, dy] = M.MOVES[m], len = 3 + q * (M.cs / 2 - 7), big = m === best && q > .3;
      alpha(o.a ?? 1, () => arrow(x + dx * 4, y + dy * 4, x + dx * (4 + len), y + dy * (4 + len), 600 + k * 4 + m, { c: big ? C.goldInk : C.soft, w: big ? 2 : 1.2, head: 4 + q * 4 })); });
  }
}
// ===== end pg kit =====
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, key, maze, trail, pip } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GAM = 0.9, CAP = 100, NW = 300, NRUN = 100, AT = 1, AV = 0.2, BV = 0.1, SK = key(0, 3), BEST = Math.pow(GAM, 6);
const ATS = [0.25, 0.5, 1, 2], AVS = [0.05, 0.1, 0.2, 0.5], NAMES = ['up', 'right', 'down', 'left'];
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−');
const sgn2 = v => (v > .004 ? '+' : '') + f2(v);

// ───── fast world tables: NX = cell after move, RW = reward of the move (+1 cheese, −1 cat, 0 otherwise)
const NX = new Int8Array(80), RW = new Int8Array(80);
for (let s = 0; s < 20; s++) { const c = s % 5, r = (s / 5) | 0; for (let a = 0; a < 4; a++) { const [nc, nr] = G.step(c, r, a); NX[s * 4 + a] = nr * 5 + nc; RW[s * 4 + a] = G.term(nc, nr); } }
const P = new Float64Array(4);
// chances cache: CP holds softmax(θ(s, ·)) for every cell, recomputed only when that cell's preferences changed (DIRTY)
const CP = new Float64Array(80), DIRTY = new Uint8Array(20);
function cpol(th, s) { if (DIRTY[s]) { pol(th, s); for (let a = 0; a < 4; a++) CP[s * 4 + a] = P[a]; DIRTY[s] = 0; } else for (let a = 0; a < 4; a++) P[a] = CP[s * 4 + a]; }
function pol(th, s) { const o = s * 4; let m = th[o]; for (let a = 1; a < 4; a++) if (th[o + a] > m) m = th[o + a]; let z = 0; for (let a = 0; a < 4; a++) { P[a] = Math.exp(th[o + a] - m); z += P[a]; } for (let a = 0; a < 4; a++) P[a] /= z; }
function pick(u) { let c = 0; for (let a = 0; a < 3; a++) { c += P[a]; if (u < c) return a; } return 3; }
// one-step actor-critic. After every step: δ = r + γ V(next) − V(here) (V(next) = 0 if the walk ended);
// critic V(here) += AV δ; actor θ(here, a) += AT δ (1 − chance a), θ(here, b) −= AT δ chance b. log: keep walks + snapshots
function runAC(seed, aT, aV, log) {
  const R = rng(seed), th = new Float64Array(80), V = new Float64Array(20), ret = new Float32Array(NW), walks = log ? [] : null; DIRTY.fill(1);
  for (let e = 0; e < NW; e++) { let s = SK; const tr = log ? [] : null;
    for (let k = 0; k < CAP; k++) { cpol(th, s); const a = pick(R()), ns = NX[s * 4 + a], rew = RW[s * 4 + a], dl = rew + (rew ? 0 : GAM * V[ns]) - V[s];
      if (log) tr.push({ s, a, ns, rew, dl, v: V[s], vn: rew ? 0 : V[ns], p: Array.from(P), c: s % 5, r: (s / 5) | 0, nc: ns % 5, nr: (ns / 5) | 0, m: a, bump: ns === s });
      if (dl !== 0) { V[s] += aV * dl; const o = s * 4; for (let b = 0; b < 4; b++) th[o + b] += aT * dl * ((b === a) - P[b]); DIRTY[s] = 1; } s = ns; if (rew) { ret[e] = rew * Math.pow(GAM, k); break; } }
    if (log) walks.push({ tr, th: Float64Array.from(th), V: Array.from(V) }); }
  return log ? { ret, walks } : ret;
}
// REINFORCE + baseline (reinforce / baseline reels): after the walk, G_t = discounted reward that followed move t;
// θ(s_t, ·) += AT (G_t − yardstick(s_t)) ∇log chance, all from the pre-walk chances; yardstick(s) += AV (G_t − yardstick(s))
const SS = new Int8Array(CAP), AA = new Int8Array(CAP), RR = new Int8Array(CAP), PP = new Float64Array(CAP * 4), GG = new Float64Array(CAP);
function runRB(seed, aT, aV) {
  const R = rng(seed), th = new Float64Array(80), V = new Float64Array(20), ret = new Float32Array(NW), d = new Float64Array(80); DIRTY.fill(1);
  for (let e = 0; e < NW; e++) { let s = SK, n = 0;
    for (let k = 0; k < CAP; k++) { cpol(th, s); const a = pick(R()); SS[n] = s; AA[n] = a; for (let b = 0; b < 4; b++) PP[n * 4 + b] = P[b]; RR[n] = RW[s * 4 + a]; n++; const rew = RW[s * 4 + a]; s = NX[s * 4 + a]; if (rew) break; }
    let g = 0; for (let i = n - 1; i >= 0; i--) { g = RR[i] + GAM * g; GG[i] = g; } ret[e] = RR[n - 1] ? GG[0] : 0;
    d.fill(0); for (let i = 0; i < n; i++) { const A = GG[i] - V[SS[i]], o = SS[i] * 4; for (let b = 0; b < 4; b++) d[o + b] += aT * A * ((b === AA[i]) - PP[i * 4 + b]); }
    for (let i = 0; i < n; i++) V[SS[i]] += aV * (GG[i] - V[SS[i]]);
    for (let i = 0; i < n; i++) DIRTY[SS[i]] = 1; for (let i = 0; i < 80; i++) th[i] += d[i]; }
  return ret;
}
// the race: 100 runs per method, run k uses rng(5000 + k) for both ("same dice"). curve = average return from the start,
// over runs and over a 10-walk window; late = each run's average return over its last 100 walks; stuck = late < 0.4
function summarise(rets) { const cur = new Float64Array(NW); rets.forEach(r => { for (let i = 0; i < NW; i++) cur[i] += r[i] / rets.length; });
  const sm = Array.from(cur, (v, i) => { let s = 0, n = 0; for (let j = Math.max(0, i - 9); j <= i; j++) { s += cur[j]; n++; } return s / n; });
  const late = rets.map(r => { let s = 0; for (let i = NW - 100; i < NW; i++) s += r[i]; return s / 100; });
  const stops = rets.map(r => { let n = 0; for (let i = NW - 100; i < NW; i++) if (r[i] === 0) n++; return n; });   // return exactly 0 = stopped at 100 steps
  return { cur: sm, late, stuck: late.filter(v => v < .4).length, wander: late.filter((v, i) => v < .4 && stops[i] >= 90).length }; }
// results are computed lazily (and warmed up in small chunks after load, so the reel opens instantly)
const RETS = {};
function rets(kind, a, upto = NRUN) { const arr = RETS[kind + a] ||= [];
  while (arr.length < upto) { const k = 5000 + arr.length; arr.push(kind === 'rb' ? runRB(k, a, BV) : kind === 'ac' ? runAC(k, a, AV) : runAC(k, AT, a)); } return arr; }
const RACE = {}, race = aT => RACE[aT] ||= { rb: summarise(rets('rb', aT)), ac: summarise(rets('ac', aT)) };
let FAIR = null; const fairRB = () => FAIR ||= (s => (s.below = s.stuck, s))(summarise(Array.from({ length: NRUN }, (_, k) => runRB(5000 + k, AT, .2))));
const CRIT = {}, crit = aV => aV === AV ? race(AT).ac : CRIT[aV] ||= summarise(rets('cr', aV));
if (typeof setTimeout === 'function') { const Q = [['rb', AT], ['ac', AT], ['cr', .5], ['cr', .05], ['cr', .1]].concat(ATS.filter(a => a !== AT).flatMap(a => [['rb', a], ['ac', a]]));
  const tick = () => { const j = Q.find(([k, a]) => (RETS[k + a] || []).length < NRUN); if (!j) return; rets(j[0], j[1], (RETS[j[0] + j[1]] || []).length + 5); setTimeout(tick, 15); }; setTimeout(tick, 400); }

// ───── the story run: actor-critic, run 0 (rng(5000))
const SR = runAC(5000, AT, AV, true), WS = SR.walks, w1 = WS[0].tr, N1 = w1.length, LAST1 = w1[N1 - 1];
const P1 = softmax(Array.from(WS[0].th.slice(LAST1.s * 4, LAST1.s * 4 + 4)));         // chances in that cell after the update
const ZT = new Float64Array(80), ZV = new Array(20).fill(0);
// a walk where the critic already speaks in the middle: the first walk with ≥ 3 clearly non-zero TD errors before its last step
const MIDW = WS.findIndex(w => w.tr.length <= 30 && w.tr.slice(0, -1).filter(q => Math.abs(q.dl) >= .01).length >= 4), MW = WS[MIDW].tr;
const MIDN = MW.slice(0, -1).filter(q => Math.abs(q.dl) >= .01).length, MIDNEG = MW.slice(0, -1).find(q => q.dl <= -.01), MIDPOS = MW.slice(0, -1).find(q => q.dl >= .01);
const MBI = MW.findIndex(q => q.dl < -.1), MB = MW[MBI], MCH = MW[MW.length - 1];
const VS = WS.map(w => w.V[SK]), V50 = VS[49], V100 = VS[99], VFIN = VS[NW - 1];
const LENFIN = WS.slice(NW - 10).map(w => w.tr.length);
// two signals for one move, with Pip's chances and critic frozen after walk 100: walks that start with "up"
function sigAt(SNAP) { const TH100 = WS[SNAP - 1].th, V100A = WS[SNAP - 1].V;
return (() => { const R = rng(777), out = [];
  while (out.length < 40) { let s = SK, g = 0, first = -1, k; const rs = [];
    for (k = 0; k < CAP; k++) { pol(TH100, s); const a = pick(R()); if (k === 0) first = a; const rew = RW[s * 4 + a]; rs.push(rew); s = NX[s * 4 + a]; if (rew) break; }
    if (first !== 0) continue; for (let i = rs.length - 1; i >= 0; i--) g = rs[i] + GAM * g; out.push({ g, len: rs.length, adv: g - V100A[SK] }); }
  const m = out.reduce((a, q) => a + q.adv, 0) / out.length, sd = Math.sqrt(out.reduce((a, q) => a + (q.adv - m) ** 2, 0) / out.length);
  return { pts: out, mean: m, sd, lo: Math.min(...out.map(q => q.adv)), hi: Math.max(...out.map(q => q.adv)), td: GAM * V100A[key(0, 2)] - V100A[SK], ncat: out.filter(q => q.g < 0).length, p: softmax(Array.from(TH100.slice(SK * 4, SK * 4 + 4))), vs: V100A[SK], vu: V100A[key(0, 2)] }; })(); }
const SNAP = 50, SIG = sigAt(SNAP), TH100 = WS[SNAP - 1].th;
const P100 = softmax(Array.from(TH100.slice(SK * 4, SK * 4 + 4)));

// ───── drawing helpers
function nums(V, a = 1, only) { for (const [c, r] of CELLS) { if (only && !only(c, r)) continue; const v = V[key(c, r)], aa = typeof a === 'function' ? a(c, r) : a; if (aa <= 0) continue;
  txt(f2(v), OX + c * CS + CS - 5, OY + r * CS + 12, { s: 13, al: 'right', a: aa, c: v < -.004 ? C.roseInk : v > .004 ? C.goldInk : C.soft }); } }
function dim(a = .62) { box(0, 0, W, 285, C.bg, a); }
function walkAt(tr, t, dur, from = 0, to = tr.length) {
  const k = Math.floor(Math.max(0, t) / dur);
  if (k >= to - from) { const l = tr[to - 1]; return { x: cx(l.nc), y: cy(l.nr), k: to - from, done: true, face: Math.sign(l.nc - l.c) || 0 }; }
  const s = tr[from + k], tt = Math.max(0, t) % dur, u = ease(tt / dur * 1.4), [dx, dy] = G.MOVES[s.m];
  if (s.bump) { const b = Math.sin(clamp(tt / dur * 1.6) * Math.PI) * 11; return { x: cx(s.c) + dx * b, y: cy(s.r) + dy * b, k, s, face: dx || 0, u }; }
  return { x: lerp(cx(s.c), cx(s.nc), u), y: lerp(cy(s.r), cy(s.nr), u), k, s, face: dx || 0, u };
}
// θ after the first i steps of walk e (story run): replay the updates
function thAt(e, i) { const th = e > 0 ? Float64Array.from(WS[e - 1].th) : new Float64Array(80), V = e > 0 ? WS[e - 1].V.slice() : ZV.slice();
  for (let k = 0; k < i; k++) { const q = WS[e].tr[k]; V[q.s] += AV * q.dl; for (let b = 0; b < 4; b++) th[q.s * 4 + b] += AT * q.dl * ((b === q.m) - q.p[b]); } return { th, V }; }
function coach(x, y, s = 1, t = 0) {          // the critic: a little owl with a clipboard
  ctx.save(); ctx.translate(x, y); ctx.scale(s, s);
  ctx.beginPath(); ctx.ellipse(0, 0, 15, 19, 0, 0, Math.PI * 2); ctx.fillStyle = '#c9b48f'; ctx.fill(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3; ctx.stroke();
  rcircle(-6, -7, 5, 1301, { fill: C.card, j: .4 }); rcircle(6, -7, 5, 1302, { fill: C.card, j: .4 }); dot(-6, -7, 1.8, C.ink); dot(6, -7, 1.8, C.ink);
  ctx.beginPath(); ctx.moveTo(-2, -2); ctx.lineTo(2, -2); ctx.lineTo(0, 2); ctx.closePath(); ctx.fillStyle = C.gold; ctx.fill();
  rrect(6, 2, 14, 17, 1303, { fill: C.card, w: 1 }); rline(9, 8, 17, 8, 1304, { once: true, w: 1, c: C.soft }); rline(9, 12, 17, 12, 1305, { once: true, w: 1, c: C.soft });
  ctx.restore();
}
function moveTag(q) { return NAMES[q.m]; }
// line chart: across = walks 1..NW, up = average return from the start
function raceChart(R, n, o = {}) {
  const x0 = o.x0 ?? 60, x1 = o.x1 ?? 350, y0 = o.y0 ?? 384, y1 = o.y1 ?? 300, lo = -.1, hi = .6, X = i => x0 + i / (NW - 1) * (x1 - x0), Y = v => y0 - (v - lo) / (hi - lo) * (y0 - y1), a = o.ax ?? 1;
  alpha(a, () => { rline(x0, Y(0), x1, Y(0), 960, { c: C.line, w: 1, once: true }); rline(x0, y0, x0, y1, 961, { c: C.line, w: 1, once: true });
    txt('0', x0 - 8, Y(0), { s: 12, c: C.soft, al: 'right' }); txt('0.5', x0 - 8, Y(.5), { s: 12, c: C.soft, al: 'right' });
    txt('walk 1', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NW}`, x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
    if (o.title) txt(o.title, x0 - 30, y1 - 12, { s: 13, c: C.soft, al: 'left' });
    ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.grey; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(x0, Y(BEST)); ctx.lineTo(x1, Y(BEST)); ctx.stroke(); ctx.restore();
    txt(`best ${f2(BEST)}`, x1 + 3, Y(BEST), { s: 12, c: C.soft, al: 'left' }); });
  [[R.rb, C.rose, 'rb'], [R.ac, '#3f8f8b', 'ac']].forEach(([S, col, nm]) => { if (o.only && o.only !== nm) return; const m = Math.min(n, NW); if (m < 2) return;
    ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.beginPath(); for (let i = 0; i < m; i++) { const x = X(i), y = Y(S.cur[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); });
  return { X, Y };
}
function legend(x, y, a = 1) { alpha(a, () => { rline(x, y, x + 22, y, 1900, { c: C.rose, w: 2.5, once: true }); txt('REINFORCE + baseline', x + 28, y, { s: 13, al: 'left' });
  rline(x, y + 18, x + 22, y + 18, 1901, { c: '#3f8f8b', w: 2.5, once: true }); txt('actor-critic', x + 28, y + 18, { s: 13, al: 'left' }); }); }
// dot strip: one dot per run (its average return over the last 100 walks)
function runDots(S, y, col, lab, t, o = {}) {
  const x0 = 60, x1 = 350, lo = -.1, hi = .6, X = v => x0 + (clamp(v, lo, hi) - lo) / (hi - lo) * (x1 - x0);
  txt(lab, x0, y - 16, { s: 13, c: col, al: 'left', a: clamp(t * 2) });
  S.late.forEach((v, i) => { const u = clamp((t - i * .012) * 3); if (u <= 0) return; const jy = ((i * 37) % 11 - 5) * 1.1; if (v < .4) { dot(X(v) + (i % 3) * 4 - 4, y + jy, 4.2, C.ink, u); } else dot(X(v), y + jy, 2.6, col, .75 * u); });
  return X;
}
function runAxis(y, a = 1) { const x0 = 60, x1 = 350, lo = -.1, hi = .6, X = v => x0 + (v - lo) / (hi - lo) * (x1 - x0);
  alpha(a, () => { rline(x0, y, x1, y, 970, { c: C.line, w: 1, once: true }); [0, .4, BEST].forEach(v => { rline(X(v), y - 3, X(v), y + 3, 971 + v * 10, { c: C.soft, w: 1, once: true }); txt(v === BEST ? f2(v) : String(v), X(v), y + 11, { s: 12, c: C.soft }); });
    ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = C.rose; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(.4), y - 160); ctx.lineTo(X(.4), y); ctx.stroke(); ctx.restore(); }); }

const sd = x => { const m = x.reduce((a, b) => a + b, 0) / x.length; return Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0) / x.length); };
let CROSS_ = null; const crossAt = () => CROSS_ ??= (() => { const R = race(AT); let c = 0; for (let i = 0; i < NW; i++) if (R.ac.cur[i] <= R.rb.cur[i]) c = i + 1; return c + 1; })();
// chart of the two signals for the first move "up" (snapshot SNAP): x = signal value, dots = return − yardstick, line = δ
function sigChart(t, o = {}) {
  const x0 = 50, x1 = 360, lo = -.2, hi = .6, X = v => x0 + (v - lo) / (hi - lo) * (x1 - x0), yb = 380;
  box(20, 288, 360, 116, C.card, .9); rrect(20, 288, 360, 116, 1950, { c: C.line, w: 1 });
  rline(x0, yb, x1, yb, 1951, { c: C.line, w: 1, once: true }); [-.2, 0, .2, .4, .6].forEach(v => txt(String(v).replace('-', '−'), X(v), yb + 11, { s: 12, c: C.soft }));
  txt('signal for the first move, up', 30, 300, { s: 13, c: C.soft, al: 'left' });
  SIG.pts.forEach((q, i) => { const u = clamp((t - i * .15) * 3); if (u <= 0) return; dot(X(q.adv), yb - 10 - (i % 5) * 8, 3.2, C.rose, .8 * u); });
  if (t > 6.2) txt('dots: return − yardstick, one per walk', 370, 300, { s: 13, c: C.roseInk, al: 'right', a: clamp(t - 6.2) });
  if (o.mean) { alpha(o.mean, () => { rline(X(SIG.mean), yb - 50, X(SIG.mean), yb, 1952, { c: C.goldInk, w: 2, once: true }); txt(`average ${f2(SIG.mean)}`, X(SIG.mean) + 4, yb - 56, { s: 13, c: C.goldInk, al: 'left' }); }); }
  if (o.line) { alpha(o.line, () => { rline(X(SIG.td), yb - 62, X(SIG.td), yb, 1953, { c: '#3f8f8b', w: 3, once: true }); txt(`δ ${f2(SIG.td)}, every time`, X(SIG.td) + 4, yb - 66, { s: 13, c: '#3f7f7c', al: 'left' }); }); }
}

const STEPS = [
 { h: 'Pip’s chances, again', p: ['Same maze: stepping onto the cheese pays +1, onto the cat −1, and either ends the walk. γ (gamma) is 0.9. A walk stopped after 100 steps scores 0. Pip has no rulebook.', 'In every cell Pip keeps four **preference** numbers, one per move; softmax turns them into chances. All start at 0, so every arrow is a quarter. Arrow length shows the chance.'],
   shot: [3, 8],
   say: [["Welcome back to Pip's maze."], [["Same cheese, same cat, same dry floor.", [40, 16, 320, 256]], ["The arrows are Pip's chances. Every one starts at a quarter.", [40, 16, 320, 256]]], ["And there's still no rulebook. Pip learns only from what happens."]],
   draw(t) { maze({ t, labels: clamp(t - .4) }); fans(G, ZT, { a: clamp(t - 1.2) }); pip(cx(0), cy(3), 1, t);
     if (t > 4) card(['+1 cheese, −1 cat: paid on arrival, walk over', 'γ = 0.9;  stopped after 100 steps: scores 0', { s: 'arrows: Pip’s chances, all 0.25 to begin', c: C.goldInk }], t - 4, { gap: 1.1 }); } },
 { h: 'The long wait', p: [`REINFORCE with a baseline learns only when the walk is over. Here is Pip’s first walk: ${N1} steps, then the cat. Until that last step, nothing was learned.`, 'Then every move is judged by the return that followed it, which includes the luck of every later move. The very first step up gets blamed for a cat met 14 moves later.'],
   shot: [4, 11],
   get say() { return [["REINFORCE has to wait until the walk is over."], [["Watch Pip's first walk.", [40, 16, 320, 256]], `${N1} steps, then the cat. Until that last step, nothing is learned.`], [["Then every move takes the blame.", [40, 16, 320, 256]], ["Even the very first step up, fourteen moves before the cat.", [30, 292, 340, 87]]]]; },
   draw(t) { const D = .45, m = walkAt(w1, t - .4, D); maze({ t }); fans(G, ZT, { a: .35 }); trail(w1.slice(0, Math.min(N1, m.k + 1)).map(q => [q.c, q.r]), .6); pip(m.x, m.y, m.face, t);
     const te = .4 + N1 * D; if (m.done) bubble('−1', cx(3), cy(1) - 18, t - te);
     if (!m.done) caption(`step ${Math.min(N1, m.k + 1)} of ${N1}: learning nothing yet`);
     if (t > te + 1) { const seen = new Set(); w1.forEach(q => { const k = key(q.c, q.r); if (seen.has(k)) return; seen.add(k); box(OX + q.c * CS + 4, OY + q.r * CS + 4, CS - 8, CS - 8, C.rose, .18 * clamp(t - te - 1)); });
       card([`walk over after ${N1} steps: now update`, { s: 'every move takes the blame for the cat,', c: C.roseInk }, { s: '   even the first, 14 moves earlier', c: C.roseInk }], t - te - 1.5, { gap: 1.1 }); } } },
 { h: 'An actor and a critic', p: ['Split Pip in two. The **actor** is the chances: it picks the moves. The **critic** is a value notebook: for each cell, a guess of the return to expect from there with the current chances.', 'Think of an athlete and a coach. The athlete tries a move; the coach says at once, better than I expected, or worse. Both keep learning.'],
   shot: [3, 9],
   say: [["Here's the idea. Split Pip in two."], [["The actor is the arrows. It picks the moves.", [40, 16, 320, 256]], ["The critic is a notebook of numbers, one guess per cell, of the return to expect from there.", [40, 16, 320, 256]]], [["Think of an athlete and a coach.", [30, 292, 340, 96]], ["The athlete tries a move. The coach says at once, better than I expected, or worse.", [30, 292, 340, 96]]]],
   draw(t) { maze({ V: ZV, t }); fans(G, ZT, { a: clamp(t - .5) }); nums(ZV, clamp(t - 2.5)); pip(cx(0), cy(3), 1, t);
     alpha(clamp(t - 4), () => { box(30, 292, 340, 96, C.card); rrect(30, 292, 340, 96, 1310, { c: C.line, w: 1.1 }); L.mouse(72, 330, 1, 1, t); coach(72 + 210, 336, 1, t);
       txt('actor: the arrows', 72, 362, { s: 15, c: C.goldInk }); txt('critic: the numbers', 282, 362, { s: 15, c: '#3f7f7c' });
       txt('picks the moves', 72, 378, { s: 13, c: C.soft }); txt('grades them', 282, 378, { s: 13, c: C.soft }); arrow(240, 330, 120, 330, 1311, { c: C.soft }); txt('“better than I expected!”', 180, 314, { s: 13, c: C.soft, a: clamp(t - 6) }); }); } },
 { h: 'The critic learns by TD', p: ['Same rule as the TD reel. After each step, **target** = reward + 0.9 × guess where Pip landed; if the walk ended, just the reward. Target minus the guess where Pip was is the **TD error δ** (delta).', 'The guess moves a fifth of the way, + 0.2 × δ. It starts blank, all 0, and must keep up as Pip’s chances change.'],
   shot: [3, 9],
   say: [["The critic learns just like in the TD reel."], [["Watch Pip take one step up.", [40, 16, 320, 256]]], [["The target is the reward, plus zero point nine times the guess where Pip landed.", [30, 292, 340, 110]], ["The gap between the target and the old guess is the TD error, delta.", [30, 292, 340, 110]], "The guess moves a fifth of that gap."]],
   draw(t) { maze({ V: ZV, t, hi: t > 1.5 ? [[0, 3, C.teal, .2], [0, 2, C.gold, .22]] : null }); nums(ZV); fans(G, ZT, { a: .3 }); coach(355, 245, .9, t);
     const m = walkAt([{ c: 0, r: 3, nc: 0, nr: 2, m: 0 }], clamp(t - .8, 0, 1), 1); pip(m.x, m.y, 1, t);
     card(['target  =  reward  +  0.9 × guess where Pip landed', 'TD error δ  =  target  −  guess where Pip was', 'critic:  guess  +=  0.2 × δ', { s: 'walk over: target = just the reward', c: C.soft }], t - 1.5, { gap: 1.2, fs: 16, lh: 22 }); } },
 { h: 'δ grades the move', p: ['The critic expected its guess for Pip’s cell. After the move, Pip has one real reward plus the landing cell’s guess. If more, δ is positive: **better than expected**, so make the move more likely. Negative: less likely.', 'The actor pushes as in REINFORCE, δ replacing return minus yardstick: the critic’s guess stands in for the rest of the walk. It happens at once.'],
   shot: [4, 10],
   say: [["Here's the clever part. The critic's TD error also grades the move."], [["REINFORCE waited for the whole return.", [30, 60, 340, 120]], ["Actor-critic uses one real reward, plus the critic's guess for the rest.", [30, 60, 340, 120]]], ["Better than expected, the move becomes more likely. Worse, less likely. And it happens right away."]],
   draw(t) { maze({ t }); fans(G, ZT, { a: .25 }); pip(cx(0), cy(3), 1, t); coach(355, 245, .9, t); dim(.55 * clamp(t - .3));
     card([{ s: 'REINFORCE + baseline, after the walk:', c: C.roseInk }, '   push  =  return that followed  −  yardstick', { s: 'actor-critic, after every step:', c: '#3f7f7c' }, '   push  =  reward + 0.9 × guess there  −  guess here', { s: '   = δ, the critic’s TD error', c: C.goldInk }], t - .6, { gap: 1.3, fs: 16, lh: 21, y: 60 }); } },
 { h: 'One step, in numbers', p: [`Walk 1’s last step: up, onto the cat. Reward −1, guesses 0: δ = −1. Up’s preference changes by 1 × δ × (1 − 0.25), to −0.75; others rise 0.25. Chances: up ${f2(P1[0])}, others ${f2(P1[1])}.`, `That 1 is the actor’s **step size**, the REINFORCE reel’s α: it moves chances clearly, not all at once. Others come later. The critic’s guess there drops to ${f2(WS[0].V[LAST1.s])}.`],
   shot: [3, 11],
   get say() { return [["Let's see it in numbers. Walk one's last step."], [["Watch Pip step up, onto the cat. Minus one.", [40, 16, 320, 256]], "Every guess was zero, so delta is minus one."], [["Up's preference drops. The other three rise a little.", [30, 292, 340, 110]], `Up's chance falls to ${f2(P1[0])}.`]]; },
   draw(t) { const u = ease((t - 5) / 1.2), th = new Float64Array(80); for (let b = 0; b < 4; b++) th[LAST1.s * 4 + b] = (WS[0].th[LAST1.s * 4 + b]) * u;
     const V = ZV.slice(); V[LAST1.s] = WS[0].V[LAST1.s] * u; maze({ V, t, hi: [[LAST1.c, LAST1.r, C.rose, .2]] }); nums(V); fans(G, th);
     const m = walkAt(w1, clamp(t - .5, 0, 1), 1, N1 - 1, N1); pip(m.done ? cx(LAST1.c) : m.x, m.done ? cy(LAST1.r) + 14 : m.y, 1, t); bubble('−1', cx(3), cy(1) - 18, t - 1.3);
     card([`δ  =  −1  +  0  −  0  =  −1`, `up:  0 + 1 × (−1) × (1 − 0.25)  =  −0.75`, `each other move:  0 − 1 × (−1) × 0.25  =  +0.25`, { s: `chances:  up ${f2(P1[0])},  others ${f2(P1[1])}`, c: C.goldInk }], t - 1.8, { gap: 1.1, fs: 16, lh: 22 }); } },
 { h: 'Quiet until the end', p: [`The first ${N1 - 1} steps of walk 1? Each had reward 0 and landed where the guess was 0, so δ = 0: no push.`, 'On walk 1 actor-critic is even quieter than REINFORCE, which pushed every move once the walk ended. A blank critic needs news from the cheese or cat, which spreads back slowly, one walk at a time. Hence a slower start.'],
   shot: [4, 10],
   say: [["What about the steps before that?"], [["Watch Pip wander through walk one.", [40, 16, 320, 256]], ["Every reward is zero, and every guess is zero. So delta is zero. No push at all.", [30, 292, 340, 64]]], ["A blank critic needs news from the cheese or the cat. That news spreads back slowly, so actor-critic starts slow."]],
   draw(t) { const D = .4, m = walkAt(w1, t - .3, D, 0, N1 - 1); maze({ V: ZV, t }); nums(ZV, .6); fans(G, ZT, { a: .3 }); trail(w1.slice(0, Math.min(N1 - 1, m.k + 1)).map(q => [q.c, q.r]), .5); pip(m.x, m.y, m.face, t);
     card(['reward 0  +  0.9 × 0  −  0   =   δ 0', { s: 'no push for the actor, no change for the critic', c: C.soft }], t - 1, { gap: 1.2 }); caption(`walk 1, step ${Math.min(N1 - 1, m.k + 1)}`); } },

 { h: 'Learning mid-walk', p: [`Walk ${MIDW + 1}. The cell right of the cat has guess ${f2(MB.vn)}, from past meetings. Step ${MBI + 1} moves up into it: δ = 0.9 × (${f2(MB.vn)}) − (${f2(MB.v)}) = ${f2(MB.dl)}. A push away, mid-walk.`, `Dots mark δ beyond 0.01 either way, gold better, rose worse; ${MIDN} before the cheese. Leaving that cell again lifted its guess to ${f2(MCH.v)}; the cheese’s δ: 1 − (${f2(MCH.v)}) = ${f2(MCH.dl)}.`],
   shot: [8.6, 12],
   get say() { return [[`Now walk ${MIDW + 1}. The critic has learned a little.`, ["Watch the cell right of the cat. Its guess is negative, from past meetings.", [40, 16, 320, 256]]], [["Pip steps up into it. Delta goes negative, a push away, right there, mid-walk.", [40, 16, 320, 256]]], [["The dots mark each grade. Gold is better than expected, rose is worse.", [40, 16, 320, 256]], "And finally, the cheese."]]; },
   draw(t) { const D = .4, ts = (MBI + 1) * D + .05, t0 = t - .3, tw = t0 < ts ? t0 : t0 < ts + 3 ? ts : t0 - 3, m = walkAt(MW, tw, D), k = Math.min(MW.length, m.k); const S = thAt(MIDW, t0 >= ts && t0 < ts + 1.6 ? k - 1 : k);   // hold: show the guesses before the update first
     maze({ V: S.V, t, hi: [[4, 1, C.rose, .14]] }); nums(S.V, .85); fans(G, S.th, { a: .5 }); pip(m.x, m.y, m.face, t);
     MW.slice(0, k).forEach((q, i) => { if (Math.abs(q.dl) < .01) return; dot(cx(q.c) + ((i * 7) % 15) - 7, cy(q.r) + 20 + ((i * 5) % 7) - 3, 4, q.dl > 0 ? C.gold : C.rose, .9); });
     if (m.done) bubble('+1', cx(4), cy(0) - 18, tw - MW.length * D);
     const last = k > 0 ? MW[k - 1] : null;
     card([`walk ${MIDW + 1}, step ${Math.max(1, k)}`, last ? { s: `δ = ${sgn2(last.dl)}${Math.abs(last.dl) >= .01 ? (last.dl > 0 ? '   better than expected' : '   worse than expected') : '   (too small to matter)'}`, c: Math.abs(last.dl) < .01 ? C.soft : last.dl > 0 ? C.goldInk : C.roseInk } : ''], .5, { gap: 0, y: 300 }); } },
 { h: `After ${NW} walks`, p: [`Let the pair learn for ${NW} walks. The arrows straighten along the short path, and the critic’s numbers fill in behind them.`, `After walk ${NW}, the critic’s guess for the start is ${f2(VFIN)}. The best possible return is 0.9 to the power 6, ${f2(BEST)}: seven steps to the cheese. ${LENFIN.filter(v => v === 7).length} of Pip’s last 10 walks took exactly 7 steps.`],
   shot: [5, 12],
   get say() { return [[`Let the pair learn for ${NW} walks.`, ["Watch the arrows straighten along the short path.", [40, 16, 320, 256]], ["And the critic's numbers fill in behind them.", [40, 16, 320, 256]]], [`At the end, the critic's guess at the start is ${f2(VFIN)}.`, ["The best possible is very close. Most recent walks take just seven steps.", [30, 300, 340, 64]]]]; },
   draw(t) { const n = clamp(Math.floor((t - .8) * 30), 0, NW), S = n ? WS[n - 1] : { th: ZT, V: ZV };
     maze({ V: S.V, t }); nums(S.V, .9); fans(G, S.th); pip(cx(0), cy(3), 1, t);
     caption(n ? `after walk ${n}` : 'blank to begin');
     if (n >= NW) card([`critic, start: ${f2(VFIN)}    best possible: ${f2(BEST)}`, { s: `last 10 walks: ${LENFIN.join(', ')} steps`, c: C.soft }], t - .8 - NW / 30, { gap: 1, y: 300 }); } },
 { h: 'Two signals for one move', p: [`Freeze Pip after walk ${SNAP}: from the start it goes up with chance ${f2(SIG.p[0])}. Take 40 walks starting with up. Each dot is what REINFORCE with a baseline would use for that move: the walk’s return minus a yardstick, here the critic’s start guess, ${f2(SIG.vs)}.`, `The dots spread from ${f2(SIG.lo)} to ${f2(SIG.hi)}: each one carries the luck of the rest of its walk.`],
   shot: [4, 12],
   say: [["Let's compare the two kinds of grade for a single move."], [["Freeze Pip, and send it up from the start, forty times.", [40, 16, 320, 256]]], [["Each dot is the grade REINFORCE with a baseline would give that move.", [20, 288, 360, 116]], ["They're spread wide. Each one carries the luck of the rest of its walk.", [20, 288, 360, 116]]]],
   draw(t) { const T = WS[SNAP - 1]; maze({ V: T.V, t, hi: [[0, 3, C.teal, .2]] }); nums(T.V, .7); fans(G, T.th, { a: .8 }); pip(cx(0), cy(3), 1, t);
     sigChart(t - 1, { line: clamp(t - 11) }); } },
 { h: 'Steady but biased', p: [`The critic’s δ for that move: 0.9 × ${f2(SIG.vu)} − ${f2(SIG.vs)} = ${f2(SIG.td)}. Moves on this dry floor never slip and the critic is frozen, so δ **never changes**: no noise.`, `The dots average ${f2(SIG.mean)}, not ${f2(SIG.td)}. That average estimates the true grade. δ leans on the cell above’s guess, which lags Pip’s progress: **bias**. Returns: unbiased but noisy. δ: steady but can be wrong.`],
   shot: [4, 10],
   get say() { return [[`Now the critic's grade. It's ${f2(SIG.td)}, every single time.`, ["No slips, a frozen critic, so no noise.", [20, 288, 360, 116]]], [["But the dots average somewhere else.", [20, 288, 360, 116]], ["That average is the honest grade. The critic's guess above lags behind Pip's progress.", [40, 16, 320, 256]]], ["Returns are unbiased but noisy. The critic is steady, but can be wrong. That's bias."]]; },
   draw(t) { const T = WS[SNAP - 1]; maze({ V: T.V, t, hi: [[0, 3, C.teal, .2], [0, 2, C.gold, .2]] }); nums(T.V, (c, r) => c === 0 && r >= 2 ? 1 : .35); fans(G, T.th, { a: .4 }); pip(cx(0), cy(3), 1, t);
     sigChart(9, { line: clamp(t - .8), mean: clamp(t - 4) }); } },
 { h: 'A fair race', p: [`A race over ${NW} walks: actor-critic against REINFORCE with a baseline. Both use actor step 1, up from the baseline reel’s 0.2, so it fits. Yardstick step 0.1 and critic step 0.2, as in their own reels. ${NRUN} runs each; run k shares a seed.`, `Across: walks. Up: return from the start, averaged over runs and 10 neighbouring walks. Dashed: the best possible, ${f2(BEST)}.`],
   shot: [4, 11],
   say: [["Time for a fair race. Actor-critic against REINFORCE with a baseline."], [["Same maze, same seeds, a hundred runs each.", [30, 296, 340, 87]]], [["The chart shows the return from the start. Dashed is the best possible.", [40, 50, 320, 230]], ["Watch the first few walks.", [40, 50, 320, 230]]]],
   draw(t) { const R = race(AT); raceChart(R, 0, { x0: 60, x1: 340, y0: 262, y1: 60, ax: clamp(t - .5), title: 'return from the start, average of 100 runs' }); legend(190, 190, clamp(t - 2));
     card(['same maze, same dice seeds, 100 runs each', 'actor step 1;  yardstick step 0.1,  critic 0.2', { s: 'drawn: the first 25 walks; the rest next', c: C.soft }], t - 3, { gap: 1.2, y: 296 }); if (t > 4) raceChart(R, Math.min(25, Math.floor((t - 4) * 8)), { x0: 60, x1: 340, y0: 262, y1: 60, ax: 0 }); } },
 { h: 'Slow start, strong finish', get p() { return [`REINFORCE with a baseline gets going sooner: after 50 walks it averages ${f2(race(AT).rb.cur[49])}, actor-critic ${f2(race(AT).ac.cur[49])}. The blank critic is the reason: its δ stays near 0 until news from the cheese has spread back to the start.`, `From walk ${crossAt()} on, actor-critic leads. After ${NW} walks: ${f2(race(AT).ac.cur[NW - 1])} against ${f2(race(AT).rb.cur[NW - 1])}.`]; },
   shot: [5, 12],
   get say() { return [[["REINFORCE with a baseline gets going sooner.", [40, 50, 320, 230]], "The blank critic is why. Its grades stay near zero, until news from the cheese reaches the start."], [["But watch the lines cross.", [40, 50, 320, 230]], `From walk ${crossAt()} on, actor-critic leads, and it finishes ahead.`]]; },
   draw(t) { const R = race(AT), n = Math.floor(clamp((t - .8) / 6) * NW), g = raceChart(R, n, { x0: 60, x1: 340, y0: 262, y1: 60, title: 'return from the start, average of 100 runs' }); legend(190, 190);
     if (n >= 50) { const X = g.X(49); rline(X, 60, X, 262, 1920, { c: C.line, w: 1, once: true }); txt('walk 50', X + 3, 252, { s: 12, c: C.soft, al: 'left' }); }
     if (n >= NW) card([{ s: `walk 50:  REINFORCE + b. ${f2(R.rb.cur[49])},  actor-critic ${f2(R.ac.cur[49])}`, c: C.ink }, { s: `actor-critic ahead from walk ${crossAt()}`, c: '#3f7f7c' }, { s: `walk ${NW}:  ${f2(R.rb.cur[NW - 1])}  vs  ${f2(R.ac.cur[NW - 1])}`, c: C.ink }], t - 7, { gap: 1.1, y: 296 }); } },
 { h: 'Calmer, too', get p() { return [`Each dot is one run: average return over its last 100 walks. Dashed line: 0.4; big dark dots: runs below it.`, `REINFORCE with a baseline: ${race(AT).rb.stuck} of ${NRUN} runs below 0.4. In ${race(AT).rb.wander}, Pip wanders: 90 or more of its last 100 walks hit the step limit, scoring 0, so return minus yardstick is about 0 and pushes fade. Actor-critic: ${race(AT).ac.stuck} below 0.4, lowest ${f2(Math.min(...race(AT).ac.late))}.`]; },
   shot: [4, 11],
   get say() { const R = race(AT); return [["Each dot is one run. Its average return over the last walks."], [["Top row, REINFORCE with a baseline.", [30, 70, 350, 70]], `${R.rb.stuck} runs fall below the dashed line.`, "In most of them, Pip just wanders. Its walks get stopped, every grade is zero, and pushes fade away."], [["Bottom row, actor-critic.", [30, 160, 350, 70]], R.ac.stuck ? `Only ${R.ac.stuck}.` : 'None at all.']]; },
   draw(t) { const R = race(AT); runAxis(250, clamp(t - .3)); txt('each run’s average return, last 100 walks', 200, 32, { s: 14, c: C.soft, a: clamp(t - .3) });
     runDots(R.rb, 110, C.rose, `REINFORCE + baseline:  ${R.rb.stuck} runs below 0.4`, t - 1.2); runDots(R.ac, 200, '#3f8f8b', `actor-critic:  ${R.ac.stuck} runs below 0.4`, t - 3.5);
     if (t > 6) card([`runs below 0.4:  ${R.rb.stuck}  vs  ${R.ac.stuck}`, { s: `${R.rb.wander} stuck runs: 90+ of the last 100 walks stopped`, c: C.soft }], t - 6, { gap: 1.2, y: 296 }); } },
 { h: 'Fair step sizes?', get p() { const a = race(AT), y2 = fairRB(), c1 = crit(.1);
     return [`The yardstick learned with step 0.1, the critic with 0.2, and step sizes matter. So we swapped the two steps. REINFORCE with a baseline and yardstick step 0.2: ${f2(y2.cur[49])} after 50 walks, ${y2.stuck} runs below 0.4. Actor-critic with critic step 0.1: ${f2(c1.cur[49])} after 50 walks, ${c1.stuck} runs below 0.4.`, 'Same picture either way: REINFORCE with a baseline starts faster and leaves more runs behind.']; },
   shot: [4, 10],
   say: [["Were the step sizes fair? The yardstick and the critic learned at different speeds. So we swapped them."], [["Here's the table.", [20, 26, 360, 200]], ["Same picture either way.", [20, 26, 360, 200]]], [["REINFORCE with a baseline starts faster, but leaves more runs behind.", [30, 270, 340, 64]]]],
   draw(t) { const a = race(AT), y2 = fairRB(), c1 = crit(.1), rows = [[`REINFORCE + b., yardstick 0.1`, a.rb], [`REINFORCE + b., yardstick 0.2`, y2], [`actor-critic, critic 0.1`, c1], [`actor-critic, critic 0.2`, a.ac]];
     txt('after 50 walks · after 300 · runs below 0.4', 370, 40, { s: 13, c: C.soft, al: 'right', a: clamp(t) });
     rows.forEach(([lab, S], i) => { const u = clamp(t - .5 - i * .6), y = 80 + i * 44, col = i < 2 ? C.roseInk : '#3f7f7c'; txt(lab, 24, y, { s: 15, al: 'left', c: col, a: u });
       txt(f2(S.cur[49]), 250, y, { s: 15, a: u, al: 'right' }); txt(f2(S.cur[NW - 1]), 305, y, { s: 15, a: u, al: 'right' }); txt(String(S.stuck ?? S.below), 360, y, { s: 15, a: u, al: 'right' }); });
     if (t > 3.5) card(['matched step sizes: same story', { s: 'REINFORCE + b. faster early, more runs left behind', c: C.soft }], t - 3.5, { gap: 1.2, y: 270 }); } },
 { h: 'Your turn', ctl: { label: 'actor step', values: ATS, def: 2, onChange: st => {} },
   p: ['Drag the **actor step size** and rerun both methods on the same seeds. The chart title counts the runs that end below 0.4.', 'Big steps learn fast but trust each push more. Watch which method copes with step 2, and which one is quicker with small steps.'],
   shot: [3],
   say: [["Your turn. Drag the actor's step size, and both methods rerun on the same seeds."], [["Watch the two curves, and the count of runs left behind.", [40, 20, 320, 250]]], ["Big steps learn fast, but trust every push. Which method copes with a step of two? Which is quicker with small ones?"]],
   draw(t, st) { const a = ATS[st.ci ?? 2], R = race(a), n = Math.floor(clamp((t - (st.changed ?? 0) - .2) / 2) * NW);
     raceChart(R, n, { x0: 60, x1: 340, y0: 262, y1: 70 }); legend(190, 190);
     txt(`actor step ${a}`, 200, 30, { s: 17, c: C.ink }); txt(`runs below 0.4:  REINFORCE + b. ${R.rb.stuck},  actor-critic ${R.ac.stuck}`, 200, 50, { s: 14, c: C.soft });
     card([`walk 50:  ${f2(R.rb.cur[49])}  vs  ${f2(R.ac.cur[49])}`, `walk ${NW}:  ${f2(R.rb.cur[NW - 1])}  vs  ${f2(R.ac.cur[NW - 1])}`, { s: 'REINFORCE + baseline  vs  actor-critic', c: C.soft }], 3, { gap: 0, y: 296 }); } },
 { h: 'A rushed critic', get p() { return [`The critic is only a guess, and how fast it learns matters. With critic step 0.5 it spreads news faster, ${f2(crit(.5).cur[49])} after 50 walks against ${f2(crit(.2).cur[49])} at 0.2.`, `But each walk now jerks the guesses and grades around, and ${crit(.5).stuck} of ${NRUN} runs end below 0.4. At 0.05 the critic is calm but slow: ${f2(crit(.05).cur[49])} after 50 walks. A wrong critic steers the actor wrong.`]; },
   shot: [4, 11],
   get say() { return [["The critic is only a guess. How fast should it learn?"], [["Look at the top bars. A rushed critic spreads news faster.", [40, 30, 320, 140]]], [["But the bottom bars show the cost.", [40, 180, 320, 130]], `With step zero point five, ${crit(.5).stuck} runs fall behind.`, "A wrong critic steers the actor wrong."]]; },
   draw(t) { txt('actor-critic, actor step 1, four critic step sizes', 200, 18, { s: 14, c: C.soft, a: clamp(t) });
     const x0 = 80, gw = 70; [['after 50 walks (average return)', 49, 156, 100], ['runs below 0.4 (of 100)', -1, 300, 90]].forEach(([lab, i, y0, h], g) => {
       txt('critic step', x0 - 4, y0 + 11, { s: 12, c: C.soft, al: 'right', a: clamp(t - 1 - g * 2) });
       txt(lab, 200, y0 - h - 12, { s: 14, c: C.soft, a: clamp(t - .5 - g * 2) }); rline(x0, y0, x0 + 4 * gw, y0, 1930 + g, { c: C.line, w: 1 });
       AVS.forEach((v, j) => { const S = crit(v), val = i >= 0 ? S.cur[i] : S.stuck, mx = i >= 0 ? .4 : 10, u = clamp((t - 1 - g * 2 - j * .3) * 2), bh = clamp(val / mx) * (h - 20) * u, x = x0 + j * gw + 14;
         box(x, y0 - bh, gw - 28, bh, v === .5 ? C.rose : '#3f8f8b', .8); txt(String(v), x + (gw - 28) / 2, y0 + 11, { s: 13, c: C.soft });
         if (u >= 1) txt(i >= 0 ? f2(val) : String(val), x + (gw - 28) / 2, y0 - bh - 9, { s: 13, c: C.ink }); }); });
     if (t > 6) card(['bigger critic step: news spreads faster', { s: 'too big: jumpy grades, more runs left behind', c: C.roseInk }], t - 6, { gap: 1.2, y: 330 }); } },
 { h: 'Where you’ve seen this', p: ['In 1983, Andrew Barto, Richard Sutton and Charles Anderson balanced a pole on a cart with two learning parts: one chose the pushes, an **adaptive critic** graded them. The actor and the critic.', 'Today A2C, A3C, PPO, DDPG, TD3 and SAC are all actor-critic methods, with neural networks in both roles. Next: many workers at once, and better grades than a single step.'],
   shot: [4, 11],
   say: [["The idea is old."], [["In nineteen eighty three, Barto, Sutton and Anderson balanced a pole on a cart.", [40, 60, 320, 180]], ["One part chose the pushes. An adaptive critic graded them.", [40, 60, 320, 180]]], [["Today, A2C, PPO, DDPG and SAC are all actor-critic methods.", [30, 292, 340, 87]]]],
   draw(t) { const a = clamp(t - .3), th = Math.sin(t * 1.3) * .12; alpha(a, () => { rline(60, 200, 340, 200, 1940, { w: 1.8 }); const x = 200 + Math.sin(t * .9) * 30;
       rrect(x - 30, 176, 60, 22, 1941, { fill: '#cfe3e2' }); rcircle(x - 18, 200, 5, 1942, { fill: C.ink }); rcircle(x + 18, 200, 5, 1943, { fill: C.ink });
       rline(x, 176, x + Math.sin(th) * 110, 176 - Math.cos(th) * 110, 1944, { w: 3, c: C.goldInk }); dot(x + Math.sin(th) * 110, 176 - Math.cos(th) * 110, 4, C.goldInk); });
     txt('pole on a cart, 1983', 200, 230, { s: 15, c: C.soft, a: clamp(t - 1) }); coach(320, 120, 1, t); L.mouse(80, 130, 1, 1, t);
     if (t > 3) card(['1983: actor + adaptive critic (Barto, Sutton, Anderson)', { s: 'today: A2C, A3C, PPO, DDPG, TD3, SAC', c: C.goldInk }, { s: 'next: many workers at once; better grades', c: C.soft }], t - 3, { gap: 1.2, fs: 16 }); } },
 { h: 'Recap', p: ['**Actor-critic** splits Pip in two: the actor holds the chances, the critic learns a value notebook by TD. After every step, the critic’s TD error grades the move, and the actor nudges it up or down.', 'No waiting for the walk to end, and steady grades instead of noisy returns. The price: a critic that is wrong, or still blank, grades wrongly. That is bias.'],
   shot: [3, 9],
   say: [["So, actor-critic splits Pip in two. The actor holds the chances. The critic learns values by TD."], [["After every step, the critic's TD error grades the move, and the actor nudges it.", [30, 292, 340, 106]], "No waiting, and steady grades. The price is bias, when the critic is wrong."]],
   draw(t) { const S = WS[NW - 1]; maze({ V: S.V, t }); nums(S.V, .8); fans(G, S.th, { a: .8 }); pip(cx(0), cy(3), 1, t);
     card(['δ  =  reward  +  0.9 × guess there  −  guess here', 'critic:  guess here  +=  0.2 × δ', 'actor:  move made  +=  step × δ × (1 − its chance)', { s: '            other moves  −=  step × δ × their chance', c: C.soft }], t - .3, { gap: 1, fs: 16, lh: 22 }); } },
];

RLStory.register('actorcritic', { title: 'Actor-Critic', track: 'Policy Gradients', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { sigAt, SR, WS, race, crit, SIG, MIDW, MIDN, MIDNEG, MIDPOS, VS, V50, V100, VFIN, LENFIN, P1, P100, LAST1, N1 } });
})();
