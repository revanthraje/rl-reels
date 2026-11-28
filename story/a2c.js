// A2C & A3C as a story: Pip's actor-critic (actorcritic reel) is run by several workers at once, each in its own copy of the
// maze, all sharing one set of chances and one critic. A2C: everyone plays 5 moves, waits, one averaged update. A3C: each
// worker sends its own update the moment it is done, using a copy that may be a few updates old. Toy clock: a move takes
// 1 tick, a quarter of the moves take 2. Every number comes from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
// ===== pg kit (copied from pg.js) =====
const softmax = th => { const m = Math.max(...th), e = th.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
function fans(M, TH, o = {}) {
  for (let r = 0; r < M.rows; r++) for (let c = 0; c < M.cols; c++) {
    if (M.term(c, r) || M.isRock(c, r)) continue; const k = M.key(c, r), p = softmax(Array.from(TH.slice(k * 4, k * 4 + 4))), best = p.indexOf(Math.max(...p)), x = M.cx(c), y = M.cy(r) + 4;
    p.forEach((q, m) => { const [dx, dy] = M.MOVES[m], len = 3 + q * (M.cs / 2 - 7), big = m === best && q > .3;
      alpha(o.a ?? 1, () => arrow(x + dx * 4, y + dy * 4, x + dx * (4 + len), y + dy * (4 + len), 600 + k * 4 + m, { c: big ? C.goldInk : C.soft, w: big ? 2 : 1.2, head: 4 + q * 4 })); });
  }
}
// ===== end pg kit =====
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, key, maze, trail, pip } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GAM = 0.9, SK = key(0, 3), BEST = Math.pow(GAM, 6), NRUN = 100, T = 2500, EVERY = 25, NSEG = 5, CAPW = 100, P2 = .25;
const AT = 4, AV = AT / 10, KS = [1, 2, 4, 8], WCOL = ['#4f7fb0', '#c4581f', '#3f8f8b', '#a54444', '#8a5d08', '#7a5ea8', '#5e8a3a', '#b0577f'];
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−');
const NX = new Int8Array(80), RW = new Int8Array(80), TERMC = new Uint8Array(20);
for (let s = 0; s < 20; s++) { const c = s % 5, r = (s / 5) | 0; TERMC[s] = G.term(c, r) ? 1 : 0; for (let a = 0; a < 4; a++) { const [nc, nr] = G.step(c, r, a); NX[s * 4 + a] = nr * 5 + nc; RW[s * 4 + a] = G.term(nc, nr); } }

// ───── simulation
function probsOf(th) { const P = new Float64Array(80); for (let s = 0; s < 20; s++) { const o = s * 4, m = Math.max(th[o], th[o + 1], th[o + 2], th[o + 3]); let z = 0; for (let a = 0; a < 4; a++) { P[o + a] = Math.exp(th[o + a] - m); z += P[o + a]; } for (let a = 0; a < 4; a++) P[o + a] /= z; } return P; }
// how good Pip's chances are: expected return from the start, worked out by US with the rulebook (Pip never sees it).
// Warm-started sweeps (Vw carries the previous answer), 150 sweeps from scratch.
function Jof(th, Vw) { const P = probsOf(th), V = Vw || new Float64Array(20); for (let it = 0; it < (Vw ? 25 : 150); it++) for (let s = 0; s < 20; s++) { if (TERMC[s]) continue; let v = 0;
  for (let a = 0; a < 4; a++) { const r = RW[s * 4 + a]; v += P[s * 4 + a] * (r + (r ? 0 : GAM * V[NX[s * 4 + a]])); } V[s] = v; } return V[SK]; }
function pickP(P, s, u) { let c = 0; for (let a = 0; a < 3; a++) { c += P[s * 4 + a]; if (u < c) return a; } return 3; }
// a worker: its own copy of the maze, position, walk length and dice (run k, worker w: rng(5000 + k + 1000 w)).
// seg(P) plays NSEG moves with chances P; each move takes 1 tick, or 2 with chance P2. Walk over or 100 moves: back to the start.
function worker(seed) { const R = rng(seed); let s = SK, len = 0;
  return { seg(P) { const mv = []; let dur = 0; for (let j = 0; j < NSEG; j++) { const a = pickP(P, s, R()), ns = NX[s * 4 + a], r = RW[s * 4 + a], d = R() < P2 ? 2 : 1; len++; dur += d; mv.push([s, a, ns, r, d]);
      if (r || len >= CAPW) { s = SK; len = 0; } else s = ns; } return { mv, dur }; }, get s() { return s; } }; }
// push of some moves (as in the actorcritic reel): δ = r + γ V(next) − V(here); actor Σ δ (onehot − chance); critic Σ δ
function push(mv, P, V, dT, dV) { for (const [s, a, ns, r] of mv) { const dl = r + (r ? 0 : GAM * V[ns]) - V[s]; dV[s] += dl; for (let b = 0; b < 4; b++) dT[s * 4 + b] += dl * ((b === a) - P[s * 4 + b]); } }
// A2C: all K workers play NSEG moves with the same chances; the round lasts as long as the slowest worker; then ONE update:
// step × (average over workers of each worker's summed push)
function runA2C(run, K, aT, aV, log) {
  const Wk = Array.from({ length: K }, (_, w) => worker(5000 + run + 1000 * w)), th = new Float64Array(80), V = new Float64Array(20), Vj = new Float64Array(20);
  const curve = [], moves = [], rounds = []; let tick = 0, nm = 0, nextE = 0, idle = 0, nup = 0;
  while (tick < T) { const P = probsOf(th), dT = new Float64Array(80), dV = new Float64Array(20); let dmax = 0, dsum = 0; const segs = [];
    for (const w of Wk) { const s0 = w.s, g = w.seg(P); push(g.mv, P, V, dT, dV); dmax = Math.max(dmax, g.dur); dsum += g.dur; nm += g.mv.length; if (log && rounds.length < 8) segs.push({ s0, mv: g.mv, dur: g.dur }); }
    if (log && rounds.length < 8) rounds.push({ t0: tick, dmax, segs });
    idle += K * dmax - dsum; tick += dmax; while (nextE <= tick && nextE <= T) { curve.push(Jof(th, Vj)); moves.push(nm); nextE += EVERY; }
    for (let i = 0; i < 80; i++) th[i] += aT * dT[i] / K; for (let i = 0; i < 20; i++) V[i] += aV * dV[i] / K; nup++; }
  return { curve, moves, idle: idle / (K * tick), rounds, th, V, nup };
}
// A3C: each worker copies the shared numbers, plays NSEG moves, then applies its own push at once (step / K, so a full round
// of K pushes adds up to one A2C update) and copies again. stale = updates by other workers between its copy and its push.
function runA3C(run, K, aT, aV, log) {
  const Wk = Array.from({ length: K }, (_, w) => worker(5000 + run + 1000 * w)), th = new Float64Array(80), V = new Float64Array(20), Vj = new Float64Array(20);
  const curve = [], moves = [], segs = []; let nup = 0, nm = 0, nextE = 0, stale = 0;
  const st = Wk.map(w => { const P = probsOf(th), g = w.seg(P); return { P, Vc: Float64Array.from(V), mv: g.mv, t0: 0, done: g.dur, copy: 0 }; });
  for (;;) { let i = 0; for (let k = 1; k < K; k++) if (st[k].done < st[i].done) i = k; const q = st[i], tick = q.done; if (tick > T) break;
    while (nextE <= tick) { curve.push(Jof(th, Vj)); moves.push(nm); nextE += EVERY; }
    const dT = new Float64Array(80), dV = new Float64Array(20); push(q.mv, q.P, q.Vc, dT, dV); nm += q.mv.length;
    for (let j = 0; j < 80; j++) th[j] += aT * dT[j] / K; for (let j = 0; j < 20; j++) V[j] += aV * dV[j] / K;
    if (log && q.done <= 40) segs.push({ w: i, t0: q.t0, t1: q.done, stale: nup - q.copy }); stale += nup - q.copy; nup++;
    q.P = probsOf(th); q.Vc = Float64Array.from(V); q.copy = nup; const g = Wk[i].seg(q.P); q.mv = g.mv; q.t0 = tick; q.done = tick + g.dur; }
  while (nextE <= T) { curve.push(Jof(th, Vj)); moves.push(nm); nextE += EVERY; }
  return { curve, moves, stale: stale / nup, segs, th, V, nup };
}
// many runs, computed lazily and warmed up in small chunks after load
function summ(rs) { const n = rs[0].curve.length, cur = new Float64Array(n), mv = new Float64Array(n); rs.forEach(r => { for (let i = 0; i < n; i++) { cur[i] += r.curve[i] / rs.length; mv[i] += r.moves[i] / rs.length; } });
  const fin = rs.map(r => r.curve[n - 1]); return { cur, mv, below: fin.filter(v => v < .4).length, idle: rs.reduce((a, r) => a + (r.idle || 0), 0) / rs.length, stale: rs.reduce((a, r) => a + (r.stale || 0), 0) / rs.length }; }
const RR = {};
function runs(kind, K, aT, upto = NRUN) { const id = kind + K + '_' + aT, arr = RR[id] ||= []; while (arr.length < upto) { const k = arr.length; arr.push(kind === 'a2c' ? runA2C(k, K, aT, aT / 10) : runA3C(k, K, aT, aT / 10));   /* critic step = actor step / 10 */ } return arr; }
const SM = {}, res = (kind, K, aT = AT) => SM[kind + K + '_' + aT] ||= summ(runs(kind, K, aT));
if (typeof setTimeout === 'function') { const Q = [['a2c', 1, AT], ['a2c', 8, AT], ['a2c', 1, 1], ['a3c', 8, AT], ['a2c', 2, AT], ['a2c', 4, AT], ['a3c', 4, AT]];
  const tick = () => { const j = Q.find(([k, K, a]) => (RR[k + K + '_' + a] || []).length < NRUN); if (!j) return; runs(j[0], j[1], j[2], (RR[j[0] + j[1] + '_' + j[2]] || []).length + 4); setTimeout(tick, 15); }; setTimeout(tick, 400); }

// ───── story runs (cheap): A2C with 8 workers, run 0 (rounds logged); A2C and A3C with 4 workers for the timelines
const S8 = runA2C(0, 8, AT, AV, true), S4 = runA2C(0, 4, AT, AV, true), S4A = runA3C(0, 4, AT, AV, true);
// look-alike moments: with Pip's starting chances (all 0.25), 8 workers' cells move by move
const DEC = (() => { const P = probsOf(new Float64Array(80)), Wk = Array.from({ length: 8 }, (_, w) => worker(5000 + 1000 * w)), cells = Wk.map(() => []), path0 = [];
  for (let t = 0; t < 420; t++) Wk.forEach((w, i) => { const g = w.seg(P); g.mv.forEach(m => { cells[i].push(m[0]); if (i === 0) path0.push(m); }); });
  const N = cells[0].length, dist = a => new Set(a).size; let one = 0, many = 0, n = 0;
  for (let t = 100; t + 8 <= N; t++) { one += dist(cells[0].slice(t, t + 8)); many += dist(cells.map(c => c[t])); n++; }
  // an example stretch of 8 moves of worker 0 with exactly the rounded average number of cells, early on
  const want = Math.round(one / n); let ex = 0; for (let t = 0; t + 8 <= 200; t++) if (dist(cells[0].slice(t, t + 8)) === want && t > 4) { ex = t; break; }
  const T0 = 200; return { one: one / n, many: many / n, ex, path: path0.slice(ex, ex + 8), snap: cells.map(c => c[T0]), snapN: dist(cells.map(c => c[T0])), want }; })();

// ───── drawing helpers
const MINI = Array.from({ length: 8 }, (_, i) => L.gridWorld({ cs: 16, ox: 22 + (i % 4) * 92, oy: 44 + Math.floor(i / 4) * 112 }));
function miniDot(M, s, col, a = 1, r = 5) { dot(M.cx(s % 5), M.cy((s / 5) | 0), r, col, a); }
function minis(t, o = {}) { MINI.forEach((M, i) => { if (o.n != null && i >= o.n) return; const a = o.a ? o.a(i) : 1; if (a <= 0) return; alpha(a, () => { M.maze({ t, pop: { cheese: 1, cat: 1 } }); txt(`worker ${i + 1}`, M.ox + 40, M.oy - 12, { s: 13, c: WCOL[i] }); }); }); }
// position of worker i inside round rd at time u ticks into the round (moves of 1 or 2 ticks)
function wpos(seg, u) { let t = 0, s = seg.s0; for (const m of seg.mv) { if (u < t + m[4]) { const f = ease((u - t) / m[4]); return { s, ns: m[2], f, moving: true }; } t += m[4]; s = m[3] ? SK : m[2]; } return { s, ns: s, f: 0, moving: false, waiting: true }; }
function miniPos(M, p) { const x0 = M.cx(p.s % 5), y0 = M.cy((p.s / 5) | 0), x1 = M.cx(p.ns % 5), y1 = M.cy((p.ns / 5) | 0); return [lerp(x0, x1, p.f), lerp(y0, y1, p.f)]; }
// line chart: series [{ x: [...], y: [...], col, w }], axes from o
function chart(series, o) {
  const { x0, x1, y0, y1 } = o, xm = o.xmax, lo = -.1, hi = .6, X = v => x0 + clamp(v / xm) * (x1 - x0), Y = v => y0 - (v - lo) / (hi - lo) * (y0 - y1), a = o.ax ?? 1;
  alpha(a, () => { rline(x0, Y(0), x1, Y(0), 960, { c: C.line, w: 1, once: true }); rline(x0, y0, x0, y1, 961, { c: C.line, w: 1, once: true });
    txt('0', x0 - 8, Y(0), { s: 12, c: C.soft, al: 'right' }); txt('0.5', x0 - 8, Y(.5), { s: 12, c: C.soft, al: 'right' });
    txt(o.xl0 || '0', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(o.xl1, x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
    if (o.title) txt(o.title, x0 - 30, y1 - 12, { s: 13, c: C.soft, al: 'left' });
    ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.grey; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(x0, Y(BEST)); ctx.lineTo(x1, Y(BEST)); ctx.stroke(); ctx.restore();
    txt(`best ${f2(BEST)}`, x1 + 3, Y(BEST), { s: 12, c: C.soft, al: 'left' }); });
  series.forEach(S => { const n = Math.min(S.y.length, S.n ?? 1e9); ctx.save(); ctx.strokeStyle = S.col; ctx.lineWidth = S.w || 2; ctx.globalAlpha = S.a ?? 1; ctx.lineJoin = 'round'; if (S.dash) ctx.setLineDash(S.dash); ctx.beginPath(); let started = false;
    for (let i = 0; i < n; i++) { if (S.x[i] > xm) break; const x = X(S.x[i]), y = Y(S.y[i]); started ? ctx.lineTo(x, y) : ctx.moveTo(x, y); started = true; } ctx.stroke(); ctx.restore(); });
  return { X, Y };
}
const TICKS = Array.from({ length: T / EVERY + 1 }, (_, i) => i * EVERY);
function legend(items, x, y, a = 1) { box(x - 6, y - 10, 150, items.length * 18 + 2, C.bg, .85 * a); alpha(a, () => items.forEach(([col, lab, dash], i) => { ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2.5; if (dash) ctx.setLineDash(dash); ctx.beginPath(); ctx.moveTo(x, y + i * 18); ctx.lineTo(x + 22, y + i * 18); ctx.stroke(); ctx.restore(); txt(lab, x + 28, y + i * 18, { s: 13, al: 'left' }); })); }
// timelines: lanes of worker bars over ticks 0..TL
const TL = 30;
function lanes(kind, t, y0, o = {}) {
  const x0 = 70, x1 = 370, X = v => x0 + v / TL * (x1 - x0), lh = 16, now = t * 4;      // 4 ticks per second
  txt(kind === 'a2c' ? 'A2C' : 'A3C', 20, y0 + 1.5 * lh + 6, { s: 17, al: 'left', c: kind === 'a2c' ? '#3f7f7c' : C.roseInk });
  for (let w = 0; w < 4; w++) txt(`w${w + 1}`, x0 - 6, y0 + w * lh + 6, { s: 12, c: WCOL[w], al: 'right' });
  if (kind === 'a2c') S4.rounds.forEach(rd => { if (rd.t0 >= TL) return; rd.segs.forEach((sg, w) => { const a = rd.t0, b = Math.min(TL, rd.t0 + sg.dur), c = Math.min(TL, rd.t0 + rd.dmax), y = y0 + w * lh;
      if (now > a) box(X(a) + 1, y, (X(Math.min(b, now)) - X(a)) - 2, lh - 4, WCOL[w], .75);
      if (now > b && c > b) { const e = Math.min(c, now); ctx.save(); ctx.setLineDash([2, 3]); ctx.strokeStyle = C.soft; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(b) + 2, y + lh / 2 - 2); ctx.lineTo(X(e) - 2, y + lh / 2 - 2); ctx.stroke(); ctx.restore(); } });
    const tu = rd.t0 + rd.dmax; if (tu <= TL && now >= tu) { rline(X(tu), y0 - 4, X(tu), y0 + 4 * lh, 1990 + rd.t0, { c: C.ink, w: 1.6, once: true }); } });
  else S4A.segs.forEach(sg => { if (sg.t0 >= TL) return; const a = sg.t0, b = Math.min(TL, sg.t1), y = y0 + sg.w * lh; if (now > a) box(X(a) + 1, y, (X(Math.min(b, now)) - X(a)) - 2, lh - 4, WCOL[sg.w], .75);
      if (sg.t1 <= TL && now >= sg.t1) { dot(X(sg.t1), y + lh / 2 - 2, 3.4, C.ink); if (o.stale) txt(String(sg.stale), X(sg.t1) - 7, y + lh / 2 - 2, { s: 11, c: '#fff' }); } });
  rline(x0, y0 + 4 * lh + 2, x1, y0 + 4 * lh + 2, 1980, { c: C.line, w: 1, once: true });
  if (o.axis) { txt('tick 0', x0, y0 + 4 * lh + 12, { s: 12, c: C.soft, al: 'left' }); txt(`tick ${TL}`, x1, y0 + 4 * lh + 12, { s: 12, c: C.soft, al: 'right' }); }
}
const A2CIDLE = () => res('a2c', 8).idle, MPT8 = () => S8.moves[S8.moves.length - 1] / T;

const STEPS = [
 { h: 'One Pip, one maze', p: ['Same maze as the actor-critic reel: +1 for the cheese, −1 for the cat, either ends the walk, γ (gamma) 0.9, and a walk is stopped after 100 moves.', 'Pip is an **actor**, its chances, and a **critic**, its notebook. After every move the critic’s TD error grades that move. Everything Pip learns comes from one walk at a time, one move after another.'],
   shot: [3, 9],
   say: [[["Back in the actor critic maze.", [40, 16, 320, 256]], ["Pip is an actor, the arrows, and a critic, the notebook.", [40, 16, 320, 256]]], [["After every move, the critic's TD error grades that move.", [30, 292, 340, 87]], "Everything Pip learns comes from one walk, one move after another."]],
   draw(t) { maze({ t, labels: clamp(t - .4) }); fans(G, new Float64Array(80), { a: clamp(t - 1) }); pip(cx(0), cy(3), 1, t);
     if (t > 4) card(['actor: the chances (arrows)', 'critic: the notebook, learned by TD', { s: 'one Pip, one maze, one move at a time', c: C.goldInk }], t - 4, { gap: 1.1 }); } },
 { h: 'Look-alike moments', p: [`Consecutive moves look alike: Pip is still in one corner. In 8 moves in a row it visits only ${DEC.one.toFixed(1)} different cells on average, chances still all a quarter.`, 'So successive updates push the same few cells. In Pip’s table, with separate numbers per cell, that is mild. In a neural network, where all situations share numbers, look-alike updates can drag it one way.'],
   shot: [4, 10],
   say: [[["Here's the trouble with one walk.", [40, 16, 320, 256]], ["Watch the teal cells as Pip makes eight moves in a row.", [40, 16, 320, 256]], ["It barely leaves its corner.", [40, 16, 320, 256]]], [["Moves in a row look alike, so updates keep pushing the same few cells.", [30, 296, 340, 64]], "In a neural network, where everything shares numbers, that can drag it one way."]],
   draw(t) { maze({ t }); const D = .55, k = clamp(Math.floor((t - .5) / D) + 1, 0, 8), P = DEC.path; const seen = new Set();
     P.slice(0, k).forEach((m, i) => { const s = m[0]; if (!seen.has(s)) { seen.add(s); box(OX + (s % 5) * CS + 4, OY + ((s / 5) | 0) * CS + 4, CS - 8, CS - 8, C.teal, .22); } });
     const cur = k > 0 ? P[k - 1] : P[0], s = k > 0 ? (cur[3] ? SK : cur[2]) : P[0][0]; pip(cx(s % 5), cy((s / 5) | 0), 1, t);
     card([`8 moves in a row: ${seen.size} different cells here`, { s: `on average: ${DEC.one.toFixed(1)} cells in 8 moves`, c: C.soft }], t - .5 - 8 * D, { gap: 1.2 });
     caption(k < 8 ? `move ${k} of 8` : ''); } },
 { h: 'Copies of the maze', p: ['The fix: run **many copies** of the maze at once. Each has its own Pip, a **worker**, with its own dice: here 8.', 'But all workers share **one** set of chances and **one** critic. Whatever one worker’s moves teach, every worker uses from then on. A worker whose walk ends just starts anew and plays on.'],
   shot: [4, 10],
   say: [[["The fix. Run many copies of the maze at once.", [16, 38, 368, 190]], ["Each copy has its own Pip, a worker, with its own dice. Here, eight.", [16, 38, 368, 190]]], [["But they all share one set of chances, and one critic.", [30, 300, 340, 64]], "Whatever one worker learns, every worker uses."]],
   draw(t) { minis(t, { a: i => clamp(t - .4 - i * .25) }); MINI.forEach((M, i) => { if (t > .6 + i * .25) miniDot(M, SK, WCOL[i]); });
     if (t > 3.5) card(['8 workers, 8 copies of the maze, 8 dice', { s: 'one shared set of chances, one shared critic', c: C.goldInk }], t - 3.5, { gap: 1.3, y: 300 }); } },
 { h: 'A varied batch', p: [`Now take one move from each worker at the same moment. The workers drift apart, so their cells differ: on average ${DEC.many.toFixed(1)} different cells among 8 moves, against ${DEC.one.toFixed(1)} for one worker’s 8 moves in a row.`, 'Each update then mixes many situations at once. That is the first reason for workers: **varied** data.'],
   shot: [4, 10],
   say: [[["Now take one move from each worker, at the same moment.", [16, 38, 368, 190]], ["The workers have drifted apart, so their cells differ.", [16, 38, 368, 190]]], [["Compare the bars.", [20, 290, 360, 105]], ["Eight workers cover far more cells than one worker's eight moves in a row.", [20, 290, 360, 105]], "Varied data. That's the first reason for workers."]],
   draw(t) { minis(t); MINI.forEach((M, i) => { const s = DEC.snap[i]; box(M.ox + (s % 5) * 16 + 1, M.oy + ((s / 5) | 0) * 16 + 1, 14, 14, WCOL[i], .25 * clamp(t - 1)); miniDot(M, s, WCOL[i], clamp(t - .5)); });
     if (t > 3) { const u = clamp((t - 3) / 1.5); txt('different cells among 8 moves, on average', 30, 300, { s: 13, c: C.soft, al: 'left', a: u });
       [[DEC.one, 'one worker, 8 in a row', C.grey], [DEC.many, '8 workers, one move each', '#3f8f8b']].forEach(([v, lab, col], i) => { const y = 330 + i * 38, w = v / 8 * 150 * u;
         txt(lab, 30, y, { s: 14, al: 'left', a: u }); box(200, y - 10, w, 20, col, .85); txt(v.toFixed(1), 206 + w, y, { s: 14, al: 'left', a: u }); });
       rline(200, 316, 200, 376, 2050, { c: C.line, w: 1, once: true }); txt('0', 200, 386, { s: 11, c: C.soft, a: u }); txt('8', 350, 386, { s: 11, c: C.soft, a: u }); } } },
 { h: 'A2C: play, wait, update', p: [`In **A2C**, each worker plays 5 moves with the shared chances and waits; the 40 moves are pooled. Each move’s push uses its TD error, with the critic from the round’s start. One update averages the workers’ summed pushes: chances move by step ${AT}, the critic by ${AV}.`, 'A2C means advantage actor-critic: the TD error plays the baseline reel’s advantage. This lock-step version is **synchronous**.'],
   shot: [5, 12],
   say: [[["Here's A2C.", [16, 38, 368, 190]], ["Each worker plays five moves, then waits for the others.", [16, 38, 368, 190]], ["Then all forty moves make one shared update.", [16, 38, 368, 190]]], [["Each move's push uses its TD error, just as before.", [30, 290, 340, 84]], "A2C stands for advantage actor critic. Everyone moves in lock step, so it's synchronous."]],
   draw(t) { minis(t); const tps = .45; let tt = t / tps, ri = 0; while (ri < 2 && tt >= S8.rounds[ri].dmax + 2) { tt -= S8.rounds[ri].dmax + 2; ri++; } const rd = S8.rounds[ri];
     rd.segs.forEach((sg, i) => { const p = wpos(sg, Math.min(tt, sg.dur)), [x, y] = miniPos(MINI[i], p); dot(x, y, 5, WCOL[i]); if (tt >= sg.dur && tt < rd.dmax) txt('waiting', MINI[i].ox + 40, MINI[i].oy + 74, { s: 12, c: C.soft }); });
     const upd = tt >= rd.dmax; if (upd) { box(0, 20, W, 230, C.gold, .12 * (1 - clamp((tt - rd.dmax) / 2))); txt('update: 40 moves → one change of the shared chances', W / 2, 258, { s: 14, c: C.goldInk }); }
     caption(`round ${ri + 1}: ${upd ? 'update' : `tick ${Math.min(rd.dmax, Math.floor(tt))} of ${rd.dmax}`}`);
     card(['each worker: 5 moves, then wait for the others', `update: step ${AT} × average over workers of their pushes`, { s: 'each move’s push: its TD error, as before', c: C.soft }], t - 1, { gap: 1.3, y: 290, fs: 16, lh: 22 }); } },
 { h: 'Why average?', p: [`One worker’s 5 moves carry one walk’s luck; averaging 8 workers cancels much of it, steadying each update’s direction.`, `So steps can be big: actor ${AT}, critic ${AV}, pushes summed over each worker’s 5 moves. Alone, one worker is the actor-critic reel’s method, updated every 5 moves at about four times its step: too big, as the next charts show. With eight it works; one worker with smaller steps is steadier but slow.`],
   shot: [4, 10],
   say: [[["Why average across workers?", [40, 16, 320, 256]], "One worker's five moves carry one walk's luck. Eight workers cancel much of it."], [["So each update can take a bigger step, and stay safe.", [30, 292, 340, 87]], "A single worker with that step would overshoot. It would need smaller, slower steps."]],
   draw(t) { maze({ t }); fans(G, S8.th, { a: clamp(t - .5) }); pip(cx(0), cy(3), 1, t); txt(`8 workers: shared chances after ${T} ticks`, W / 2, 284, { s: 13, c: C.soft, a: clamp(t - .5) });
     card(['average of 8 pushes: less luck per update', `→ a bigger step (${AT}) stays safe`, { s: 'one worker would need a smaller, slower step', c: C.soft }], t - 2, { gap: 1.3 }); } },
 { h: 'Two clocks', p: [`For a clock: a move takes one **tick**, but one move in four takes two: 1.25 ticks on average. Eight workers on eight computers: at most ${(8 / 1.25).toFixed(1)} moves per tick. In A2C the fast also wait for the slowest, ${Math.round(S8.idle * 100)} percent of the time, leaving about ${MPT8().toFixed(1)}.`, 'Top: a 4-worker run’s first 30 ticks; bars moving, dots waiting. Chart: across, ticks; up, the start’s expected return, computed with the rulebook Pip never sees.'],
   shot: [4, 10],
   say: [["Now add a clock. A move takes one tick, but some moves take two.", [20, 15, 360, 110]], [["In A2C, the fast workers wait for the slowest.", [20, 15, 360, 110]], ["Watch the dots. That's waiting time.", [20, 15, 360, 110]]], ["The chart below will track how good Pip's habit is, tick by tick.", [30, 296, 340, 87]]],
   draw(t) { lanes('a2c', t * 1.5, 20, { axis: true }); chart([], { x0: 60, x1: 340, y0: 262, y1: 140, xmax: T, xl1: `tick ${T}`, ax: clamp(t - 3), title: 'expected return from the start (rulebook)' });
     card(['1 move = 1 tick (a quarter of moves: 2 ticks)', `8 workers: at most 6.4, here ${MPT8().toFixed(1)} moves per tick`, { s: `A2C waits for the slowest: ${Math.round(S8.idle * 100)}% idle`, c: C.roseInk }], t - 2, { gap: 1.2, y: 296 }); } },
 { h: 'One worker against eight', get p() { const a = res('a2c', 1), b = res('a2c', 8), c = res('a2c', 1, 1); const i = T / EVERY;
     return [`${NRUN} runs each, same dice. With step ${AT}, one worker (grey) reaches ${f2(a.cur[i])} on average after ${T} ticks, and ${a.below} of ${NRUN} runs are still below 0.4. Eight workers (teal): ${f2(b.cur[i])}, ${b.below} runs below 0.4.`, `Dashed: one worker with both steps four times smaller. It is steadier but slow: ${f2(c.cur[i])} after ${T} ticks. On this clock, eight workers end far higher, and steadier.`]; },
   shot: [5, 12],
   say: [[["Here's the race, a hundred runs each.", [40, 50, 320, 230]], ["Grey is one worker. Teal is eight workers, same step size.", [40, 50, 320, 230]]], [["Watch teal climb faster, and end far higher.", [40, 50, 320, 230]], ["The dashed line is one worker with smaller steps. Steadier, but slow.", [40, 50, 320, 230]]]],
   draw(t) { const a = res('a2c', 1), b = res('a2c', 8), c = res('a2c', 1, 1), n = Math.floor(clamp((t - .8) / 5) * TICKS.length);
     chart([{ x: TICKS, y: a.cur, col: C.grey, n }, { x: TICKS, y: c.cur, col: C.soft, dash: [5, 4], w: 1.6, n }, { x: TICKS, y: b.cur, col: '#3f8f8b', n }], { x0: 60, x1: 340, y0: 262, y1: 60, xmax: T, xl1: `tick ${T}`, title: `expected return from the start, average of ${NRUN} runs` });
     legend([['#3f8f8b', `8 workers, step ${AT}`], [C.grey, `1 worker, step ${AT}`], [C.soft, '1 worker, steps ÷ 4', [5, 4]]], 105, 104);
     if (t > 6.5) { const i = T / EVERY; card([`after ${T} ticks:  8 workers ${f2(b.cur[i])},  1 worker ${f2(a.cur[i])}`, { s: `runs below 0.4:  ${b.below}  vs  ${a.below}`, c: C.soft }], t - 6.5, { gap: 1.2, y: 296 }); } } },
 { h: 'Count moves, not ticks', get p() { const a = res('a2c', 1), b = res('a2c', 8), c = res('a2c', 1, 1), i8 = b.mv.findIndex(v => v >= 2000);
     return [`Same runs, but across is moves gathered by all workers together, up to 2,000. By then one worker with step ${AT} is at ${f2(a.cur[a.mv.findIndex(v => v >= 2000)])} on average; eight workers have only reached ${f2(b.cur[i8])}.`, 'So workers do not make Pip learn from fewer moves. They buy clock speed, moving in parallel, and steadier updates, but need many computers or cores.']; },
   shot: [5, 11],
   say: [[["Same runs, but now count moves instead of ticks.", [40, 50, 320, 230]], ["Across is every move gathered by all workers together.", [40, 50, 320, 230]]], [["Now one worker is ahead.", [40, 50, 320, 230]], ["So workers don't help Pip learn from fewer moves. They save time on the clock, and steady the updates.", [30, 296, 340, 64]]]],
   draw(t) { const a = res('a2c', 1), b = res('a2c', 8), c = res('a2c', 1, 1), n = Math.floor(clamp((t - .8) / 4) * TICKS.length);
     chart([{ x: a.mv, y: a.cur, col: C.grey, n }, { x: c.mv, y: c.cur, col: C.soft, dash: [5, 4], w: 1.6, n }, { x: b.mv, y: b.cur, col: '#3f8f8b', n }], { x0: 60, x1: 340, y0: 262, y1: 60, xmax: 2000, xl1: '2,000 moves', title: 'expected return from the start, by moves gathered' });
     legend([['#3f8f8b', `8 workers, step ${AT}`], [C.grey, `1 worker, step ${AT}`], [C.soft, '1 worker, steps ÷ 4', [5, 4]]], 105, 104);
     if (t > 5.5) card(['same runs, counted in moves', { s: 'workers save time, not moves', c: C.goldInk }], t - 5.5, { gap: 1.2, y: 296 }); } },
 { h: 'Your turn', ctl: { label: 'workers', values: KS, def: 3, onChange: st => {} },
   get p() { return [`Drag the **workers**: 1, 2, 4 or 8, all A2C with step ${AT}, ${NRUN} runs each, on the tick clock.`, `Watch the curve and the runs below 0.4. 1 to 2 workers helps a lot, 4 more still; 8 gains nothing here: 4 workers end at ${f2(res('a2c', 4).cur[T / EVERY])} with ${Math.round(res('a2c', 4).idle * 100)}% waiting, 8 at ${f2(res('a2c', 8).cur[T / EVERY])} with ${Math.round(res('a2c', 8).idle * 100)}%. Averaging 20 moves is already steady enough, and 8 workers make each round a little longer.`]; },
   shot: [3],
   say: [["Your turn. Drag the number of workers, from one up to eight.", [20, 15, 360, 270]], [["Watch the curve, and how many runs stay stuck below zero point four.", [40, 50, 320, 230]], "Going from one worker to two helps a lot. Four helps more. Eight gains nothing here."]],
   draw(t, st) { const K = KS[st.ci ?? 3], R = res('a2c', K), n = Math.floor(clamp((t - (st.changed ?? 0) - .2) / 2) * TICKS.length), i = T / EVERY;
     chart([{ x: TICKS, y: R.cur, col: WCOL[2], n }], { x0: 60, x1: 340, y0: 262, y1: 70, xmax: T, xl1: `tick ${T}` });
     txt(`${K} worker${K > 1 ? 's' : ''}, A2C, step ${AT}`, 200, 30, { s: 17 }); txt(`after ${T} ticks: ${f2(R.cur[i])}    runs below 0.4: ${R.below}`, 200, 50, { s: 14, c: C.soft });
     card([`after 500 ticks: ${f2(R.cur[500 / EVERY])}`, `after ${T} ticks: ${f2(R.cur[i])}`, { s: `time spent waiting: ${Math.round(R.idle * 100)}%`, c: C.soft }], 3, { gap: 0, y: 296 }); } },
 { h: 'A3C: no waiting', p: ['**A3C**, asynchronous advantage actor-critic (Mnih and colleagues, 2016), never waits. Each worker copies the shared numbers, plays its 5 moves, and sends its update **the moment it is done**, then copies the latest numbers and goes on.', 'Top: A2C, 4 workers: bars are a worker’s 5 moves, dotted lines waiting, black lines shared updates. Bottom: A3C, each black dot one worker’s update.'],
   shot: [4, 10],
   say: [[["A3C never waits.", [20, 15, 360, 280]], ["Each worker copies the shared numbers, plays its moves, and sends its update the moment it's done.", [20, 15, 360, 280]]], [["Compare the lanes. Top is A2C, with waiting.", [20, 15, 360, 135]], ["Bottom is A3C. Every black dot is one worker's update, arriving on its own.", [20, 145, 360, 140]]]],
   draw(t) { txt('the first 30 ticks, 4 workers', W / 2, 26, { s: 14, c: C.soft }); lanes('a2c', t, 50); lanes('a3c', t - .5, 160, { axis: true });
     if (t > 8.5) card(['A2C: wait, then one shared update', { s: 'A3C: each worker updates as soon as it is done', c: C.roseInk }], t - 8.5, { gap: 1.2, y: 300 }); } },
 { h: 'Stale copies', get p() { const r = res('a3c', 8), r4 = res('a3c', 4), b = res('a2c', 8), i = T / EVERY;
     return [`The catch: while a worker plays, others update, so its copy goes stale. White numbers count missed updates: ${r4.stale.toFixed(1)} on average with 4 workers, ${r.stale.toFixed(1)} with 8.`, `Why does A3C work, when one worker’s push was too big? Each push uses step ${AT} divided by 8 workers: ${AT / 8}. Eight small pushes make about one A2C update, arriving one by one. With 8 workers: A3C ends at ${r.cur[i].toFixed(3)}, A2C at ${b.cur[i].toFixed(3)}, best ${BEST.toFixed(3)}; at 1,000 ticks, ${f2(r.cur[1000 / EVERY])} against ${f2(b.cur[1000 / EVERY])}.`]; },
   shot: [4, 11],
   say: [[["The catch. While a worker plays, others update, so its copy goes stale.", [20, 15, 360, 130]], ["The white numbers count the missed updates.", [20, 15, 360, 130]]], [["Why does it still work? Each push is eight times smaller, so eight of them add up to about one A2C update.", [40, 140, 320, 140]], ["Watch the two lines finish close together.", [40, 140, 320, 140]]]],
   draw(t) { lanes('a3c', 9, 30, { stale: true, axis: true }); const r = res('a3c', 8), b = res('a2c', 8);
     if (t > 2.5) { const n = Math.floor(clamp((t - 2.5) / 4) * TICKS.length); chart([{ x: TICKS, y: b.cur, col: '#3f8f8b', n }, { x: TICKS, y: r.cur, col: C.rose, n, w: 1.8 }], { x0: 60, x1: 340, y0: 270, y1: 150, xmax: T, xl1: `tick ${T}`, ax: clamp(t - 2.5) });
       legend([['#3f8f8b', 'A2C, 8 workers'], [C.rose, 'A3C, 8 workers']], 190, 215, clamp(t - 3)); }
     if (t > 7.5) card([`missed updates per push: ${r.stale.toFixed(1)} (8 workers)`, { s: `after 1,000 ticks: A3C ${f2(r.cur[1000 / EVERY])}, A2C ${f2(b.cur[1000 / EVERY])}`, c: C.soft }], t - 7.5, { gap: 1.2, y: 300 }); } },
 { h: 'A2C or A3C?', p: ['A3C ran its workers on one ordinary computer’s cores, no graphics card, and on Atari beat earlier graphics-card agents in about half their training time. A year later OpenAI found synchronous A2C did as well, its one big batch per update suiting graphics cards.', 'Our toy has none, so it shows only the costs: A2C’s waiting, A3C’s stale copies. Real speed depends on hardware.'],
   shot: [4, 10],
   say: [[["A3C ran on an ordinary computer's cores, with no graphics card.", [50, 30, 140, 135]], "On Atari, it beat earlier agents in about half the time."], [["A year later, OpenAI found that A2C did just as well. Its one big batch suits graphics cards.", [210, 30, 140, 135]], ["Real speed depends on the hardware.", [30, 292, 340, 87]]]],
   draw(t) { const a = clamp(t - .3); alpha(a, () => { rrect(60, 60, 120, 80, 2001, { fill: '#e9dcc6' }); for (let i = 0; i < 4; i++) for (let j = 0; j < 2; j++) rrect(72 + i * 26, 74 + j * 30, 20, 22, 2010 + i * 2 + j, { fill: '#cfe3e2' }); txt('CPU cores', 120, 156, { s: 14, c: C.soft });
       rrect(220, 60, 120, 80, 2002, { fill: '#e9dcc6' }); for (let i = 0; i < 10; i++) for (let j = 0; j < 4; j++) box(230 + i * 10.5, 70 + j * 16, 8, 12, '#f2b8b8', .9); txt('graphics card', 280, 156, { s: 14, c: C.soft }); });
     txt('A3C (2016)', 120, 44, { s: 16, c: C.roseInk, a: clamp(t - 1) }); txt('A2C (OpenAI, 2017)', 280, 44, { s: 16, c: '#3f7f7c', a: clamp(t - 1.5) });
     if (t > 3) card(['A3C: workers on CPU cores, no waiting', 'A2C: one big batch, suits graphics cards', { s: 'OpenAI: A2C did as well as A3C', c: C.soft }], t - 3, { gap: 1.2, y: 220 }); } },
 { h: 'Where it went', p: ['Running **many copies of the world in parallel** became standard. PPO, the next big method in this track, is usually trained exactly this way, with anything from 8 copies on a laptop to thousands in large labs.', 'The asynchronous style lives on in large distributed systems such as IMPALA (2018), which corrects for the stale chances its workers act with.'],
   shot: [4, 10],
   say: [["Running many copies of the world in parallel became standard practice.", [30, 40, 340, 200]], ["PPO, coming next, is usually trained this way. From eight copies on a laptop, to thousands in big labs.", [30, 260, 340, 64]], "The asynchronous style lives on in systems like IMPALA."],
   draw(t) { for (let i = 0; i < 24; i++) { const x = 40 + (i % 8) * 42, y = 50 + Math.floor(i / 8) * 62, a = clamp(t - .3 - i * .08); if (a <= 0) continue;
       alpha(a, () => { const M = L.gridWorld({ cs: 6.4, ox: x, oy: y }); rrect(x - 2, y - 2, 36, 29, 2100 + i, { c: C.line, w: 1 }); box(x + 25.6, y, 6.4, 6.4, C.gold, .8); box(x + 19.2, y + 6.4, 6.4, 6.4, C.rose, .5); dot(x + 3.2 + ((i * 3) % 5) * 6.4, y + 3.2 + ((i * 7) % 4) * 6.4, 2.2, WCOL[i % 8]); }); }
     if (t > 3) card(['many copies of the world, one learner', { s: 'standard practice, PPO included', c: C.goldInk }], t - 3, { gap: 1.2, y: 260 }); } },
 { h: 'Recap', p: ['**Workers** run copies of the world in parallel and share one actor and one critic. Their moves are more varied, and averaging their pushes steadies each update.', '**A2C** waits for all workers and makes one update per round. **A3C** lets each worker update on its own, with slightly stale copies. Workers save time on the clock, not moves.'],
   shot: [3, 9],
   say: [["So that's A2C and A3C. Workers run copies of the world, and share one actor and one critic.", [16, 38, 368, 190]], ["Their moves are more varied, and averaging steadies each update.", [30, 290, 340, 106]], ["A2C waits for everyone. A3C doesn't. Either way, workers save time, not moves.", [30, 290, 340, 106]]],
   draw(t) { minis(t); MINI.forEach((M, i) => miniDot(M, DEC.snap[i], WCOL[i]));
     card(['workers: copies of the world, one shared learner', 'more varied moves, steadier averaged updates', 'A2C: wait, then one update  ·  A3C: no waiting', { s: 'faster on the clock, not per move', c: C.soft }], t - .3, { gap: 1, y: 290, fs: 16, lh: 22 }); } },
];

RLStory.register('a2c', { title: 'A2C & A3C', track: 'Policy Gradients', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { S8, S4, S4A, DEC, res, runA2C, runA3C, Jof } });
})();
