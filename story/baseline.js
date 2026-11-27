// Baselines & advantage as a story: Pip learns to judge each outcome against a yardstick (what usually happens), so only
// better-than-usual outcomes push chances up. Door room (average-reward yardstick) then the maze (a yardstick per cell,
// V(s), learned from returns). Same dice as the REINFORCE reel. All numbers come from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
// ===== pg kit (shared helpers for the Policy Gradients track; copy, do not import) =====
const softmax = th => { const m = Math.max(...th), e = th.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
const DCOL = ['#8fb3d9', '#ef8f5a', '#6fb7b4'], DINK = ['#4f7fb0', '#c4581f', '#3f7f7c'], DNAME = ['A', 'B', 'C'], DX = [90, 200, 310], DP = [.2, .5, .8];
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−'), fz = v => Math.abs(v) < .005 ? '0' : f2(v), sg = v => (v > .005 ? '+' : '') + fz(v);
function pie(x, y, r, pr, o = {}) {
  let a0 = -Math.PI / 2; pr.forEach((p, i) => { const a1 = a0 + p * Math.PI * 2; ctx.save(); ctx.globalAlpha = o.a ?? 1; ctx.fillStyle = (o.cols || DCOL)[i];
    ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, r, a0, a1); ctx.closePath(); ctx.fill(); ctx.restore();
    if (o.labels && p > .06) { const am = (a0 + a1) / 2; txt((o.names || DNAME)[i], x + Math.cos(am) * r * .62, y + Math.sin(am) * r * .62, { s: 13, a: o.a ?? 1 }); } a0 = a1; });
  rcircle(x, y, r, 880, { j: .6 });
}
function door(i, o = {}) {
  const x = DX[i], y0 = 50, w = 62, h = 96;
  box(x - w / 2, y0, w, h, DCOL[i], o.glow ? .6 : .3); rrect(x - w / 2, y0, w, h, 820 + i * 5, { w: o.glow ? 2.4 : 1.6 }); dot(x + w / 2 - 10, y0 + h / 2 + 4, 2.6, C.ink);
  txt(DNAME[i], x, y0 - 14, { s: 20, c: DINK[i] }); box(x - 18, y0 + 14, 36, 20, C.card); rrect(x - 18, y0 + 14, 36, 20, 840 + i, { c: C.line, w: 1 });
  txt(o.secret ? String(DP[i]) : '?', x, y0 + 24, { s: 15, c: o.secret ? C.goldInk : C.soft });
}
function room(t, o = {}) {
  rline(20, 146, 380, 146, 801, { w: 1.6 }); for (let i = 0; i < 3; i++) door(i, { secret: o.secret, glow: o.glow === i });
  const th = o.th || [0, 0, 0], p = softmax(th);
  if (o.rows !== false) for (let i = 0; i < 3; i++) { txt(`pref ${f2(th[i])}`, DX[i], 162, { s: 14, c: C.soft }); txt(`chance ${f2(p[i])}`, DX[i], 180, { s: 15, c: DINK[i] }); }
  if (o.pie !== false) pie(o.px ?? 300, o.py ?? 238, 34, p, { labels: true });
  if (o.pip !== false) L.mouse(o.mx ?? 110, o.my ?? 244, o.face ?? 1, 1, t);
}
function fans(M, TH, o = {}) {
  for (let r = 0; r < M.rows; r++) for (let c = 0; c < M.cols; c++) {
    if (M.term(c, r) || M.isRock(c, r) || (o.skip && o.skip(c, r))) continue; const k = M.key(c, r), p = softmax(Array.from(TH.slice(k * 4, k * 4 + 4))), best = p.indexOf(Math.max(...p)), x = M.cx(c), y = M.cy(r) + 4;
    p.forEach((q, m) => { const [dx, dy] = M.MOVES[m], len = 3 + q * (M.cs / 2 - 7), big = m === best && q > .4;
      alpha(o.a ?? 1, () => arrow(x + dx * 4, y + dy * 4, x + dx * (4 + len), y + dy * (4 + len), 600 + k * 4 + m, { c: big ? C.goldInk : C.soft, w: big ? 2 : 1.2, head: 4 + q * 4 })); });
  }
}
const GAM = .9, CAP = 100;
function soft4(th, s) { const o = s * 4, m = Math.max(th[o], th[o + 1], th[o + 2], th[o + 3]), a = Math.exp(th[o] - m), b = Math.exp(th[o + 1] - m), c = Math.exp(th[o + 2] - m), d = Math.exp(th[o + 3] - m), z = a + b + c + d; return [a / z, b / z, c / z, d / z]; }
function walkMaze(G, th, R) { let c = G.start[0], r = G.start[1]; const tr = [];
  for (let k = 0; k < CAP; k++) { const s = G.key(c, r), p = soft4(th, s); let u = R(), m = 0; while (m < 3 && u >= p[m]) { u -= p[m]; m++; }
    const [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr); tr.push({ c, r, nc, nr, s, m, rew, p }); c = nc; r = nr; if (rew) break; }
  let g = 0; for (let i = tr.length - 1; i >= 0; i--) { g = tr[i].rew + GAM * g; tr[i].G = g; } return tr; }
function runMaze(G, seed, alphaStep, nep, opts = {}) {
  const R = rng(seed), th = new Float64Array(80), V = new Float64Array(20), rets = new Float64Array(nep), eps = [];
  for (let e = 0; e < nep; e++) {
    const tr = walkMaze(G, th, R);
    for (const q of tr) { q.b = opts.base === 'v' ? V[q.s] : 0; const A = q.G - q.b; for (let a = 0; a < 4; a++) th[q.s * 4 + a] += alphaStep * A * ((a === q.m) - q.p[a]); }
    if (opts.base === 'v') for (const q of tr) V[q.s] += (opts.ab ?? .1) * (q.G - V[q.s]);
    rets[e] = tr[0].G; if (opts.keep) eps.push({ tr, ret: tr[0].G, end: tr[tr.length - 1].rew, th: Float64Array.from(th), V: Float64Array.from(V) });
  }
  return { rets, eps };
}
// ===== end pg kit =====
const G = L.gridWorld({}), { cx, cy, key } = G;
// ── door room with a yardstick. Same dice: picks from rng(5000 + k); the n-th crumb draw of door a in run k is the n-th draw of rng(100000 (a + 1) + k)
const AD = .5, NP = 200, NR = 500;
function runDoor(k, base, keep) {           // base: a number (fixed yardstick) or 'avg' (average reward so far, 0 before the first pick)
  const R = rng(5000 + k), tape = [0, 1, 2].map(a => rng(100000 * (a + 1) + k)), th = [0, 0, 0]; let sum = 0; const pc = new Float64Array(NP), log = [];
  for (let i = 0; i < NP; i++) { const p = softmax(th); let u = R(), a = 0; while (a < 2 && u >= p[a]) { u -= p[a]; a++; } const r = tape[a]() < DP[a] ? 1 : 0;
    const b = base === 'avg' ? (i ? sum / i : 0) : base, A = r - b; if (keep) log.push({ a, r, b, A, p, th: th.slice() });
    for (let j = 0; j < 3; j++) th[j] += AD * A * ((j === a) - p[j]); sum += r; pc[i] = softmax(th)[2]; }
  return keep ? { pc, log, th } : pc;
}
const dmemo = {};
function doorStats(base) { const key_ = String(base); if (dmemo[key_]) return dmemo[key_];
  const runs = []; for (let k = 0; k < NR; k++) runs.push(runDoor(k, base));
  const mean = new Float64Array(NP), lo = new Float64Array(NP), hi = new Float64Array(NP);
  for (let i = 0; i < NP; i++) { const col = runs.map(r => r[i]).sort((x, y) => x - y); mean[i] = col.reduce((x, y) => x + y, 0) / NR; lo[i] = col[Math.floor(NR * .1)]; hi[i] = col[Math.floor(NR * .9)]; }
  const bad = runs.filter(r => r[NP - 1] < .5).length;
  return (dmemo[key_] = { mean, lo, hi, bad }); }
const D0 = doorStats(0), DA = doorStats('avg');
// story run 0, plain REINFORCE: the first pick of A that pays a crumb
const R0 = runDoor(0, 0, true), AHIT = R0.log.findIndex(q => q.a === 0 && q.r === 1), AH = R0.log[AHIT];
const RA = runDoor(0, 'avg', true);
// exact spread of a single try's nudge at the uniform policy (sum over the three preferences of the variance), yardstick b
function spread(b) { let m = [0, 0, 0], m2 = [0, 0, 0];
  for (let a = 0; a < 3; a++) for (const r of [0, 1]) { const pr = (1 / 3) * (r ? DP[a] : 1 - DP[a]); for (let j = 0; j < 3; j++) { const v = (r - b) * ((j === a) - 1 / 3); m[j] += pr * v; m2[j] += pr * v * v; } }
  return { mean: m, sd: Math.sqrt(m2.reduce((s, x, j) => s + x - m[j] ** 2, 0)) }; }
const SP0 = spread(0), SP5 = spread(.5);
// single-try C-nudges at the uniform policy, both ways (exact outcomes and chances)
const CN = (b) => [[2, 1], [2, 0], [0, 1], [0, 0], [1, 1], [1, 0]].map(([a, r]) => ({ a, r, v: (r - b) * ((a === 2) - 1 / 3), pr: (1 / 3) * (r ? DP[a] : 1 - DP[a]) }));
const BS = [0, .25, .5, .75, 1, 'avg'];
// ── maze: same seed and α as the REINFORCE reel, with and without a yardstick per cell V(s) (moved 0.1 of the way toward each return)
const A0 = .2, NEP = 300, NRUN = 100, AB = .1;
const SB = runMaze(G, 5000, A0, NEP, { keep: true, base: 'v', ab: AB }), EB = SB.eps;
const mm = {};
function many(base) { if (mm[base]) return mm[base]; const runs = []; for (let k = 0; k < NRUN; k++) runs.push(runMaze(G, 5000 + k, A0, NEP, { base, ab: AB }).rets);
  const blk = (rr, i) => { let s = 0; for (let j = i; j < i + 25; j++) s += rr[j]; return s / 25; };
  const mean = new Float64Array(NEP); runs.forEach(rr => rr.forEach((v, i) => mean[i] += v / NRUN));
  const ends = runs.map(rr => blk(rr, NEP - 25)), mu = ends.reduce((a, b) => a + b, 0) / NRUN, sd = Math.sqrt(ends.reduce((a, b) => a + (b - mu) ** 2, 0) / NRUN);
  return (mm[base] = { runs, mean, sm: smooth(mean, 10), end: mu, sd, at: i => blk(mean, i) }); }
function smooth(v, w) { const o = new Float64Array(v.length); for (let i = 0; i < v.length; i++) { let s = 0, n = 0; for (let j = Math.max(0, i - w + 1); j <= i; j++) { s += v[j]; n++; } o[i] = s / n; } return o; }
const MP = many('none'), MV = many('v');
// variance of one walk's total nudge at a frozen policy (from the yardstick run after walk w), without and with that run's V(s)
const VW = [50, 150, 300], NVAR = 2000;
const VAR = VW.map(w => { const th = EB[w - 1].th, V = EB[w - 1].V, R = rng(99), s = [new Float64Array(80), new Float64Array(80)], s2 = [new Float64Array(80), new Float64Array(80)];
  for (let i = 0; i < NVAR; i++) { const tr = walkMaze(G, th, R); [0, 1].forEach(b => { const gr = new Float64Array(80); for (const q of tr) { const A = q.G - (b ? V[q.s] : 0); for (let a = 0; a < 4; a++) gr[q.s * 4 + a] += A * ((a === q.m) - q.p[a]); }
    for (let j = 0; j < 80; j++) { s[b][j] += gr[j]; s2[b][j] += gr[j] * gr[j]; } }); }
  const tv = b => { let x = 0; for (let j = 0; j < 80; j++) x += s2[b][j] / NVAR - (s[b][j] / NVAR) ** 2; return x; };
  return { w, plain: tv(0), base: tv(1), ratio: tv(0) / tv(1), Vs: V[key(0, 3)], Vc: V[key(3, 0)] }; });
// a late walk of the yardstick run with a step whose return fell short of the yardstick, beside the cheese
const LATE = (() => { for (let e = 200; e < NEP; e++) { const tr = EB[e].tr; const i = tr.findIndex(q => q.c === 3 && q.r === 0 && q.G - q.b < -.05); if (i >= 0) return { e, q: tr[i], good: tr.find(q => q.c === 3 && q.r === 0 && q.G - q.b > 0) || null }; } return null; })();
const LATEG = (() => { for (let e = 200; e < NEP; e++) { const q = EB[e].tr.find(q => q.c === 3 && q.r === 0 && q.G - q.b > 0); if (q) return { e, q }; } return null; })();
const VFIN = EB[NEP - 1].V;

// ───── drawing helpers
function band(o) {
  const x0 = o.x0 ?? 56, x1 = o.x1 ?? 350, y0 = o.y0 ?? 390, y1 = o.y1 ?? 300, n = o.n ?? NP, X = i => x0 + i / (NP - 1) * (x1 - x0), Y = v => y0 - v * (y0 - y1);
  rline(x0 - 4, y0, x1 + 4, y0, 960, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1 - 2, 961, { c: C.line, w: 1, once: true });
  txt('1', x0 - 14, Y(1), { s: 12, c: C.soft }); txt('0', x0 - 14, Y(0), { s: 12, c: C.soft }); txt('pick 1', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt(`pick ${NP}`, x1, y0 + 11, { s: 12, c: C.soft, al: 'right' });
  if (o.title) txt(o.title, x0 - 10, y1 - 12, { s: 13, c: C.soft, al: 'left' });
  (o.series || []).forEach(s => { const m = Math.min(n, NP);
    if (s.lo) { ctx.save(); ctx.globalAlpha = s.ba ?? .22; ctx.fillStyle = s.col; ctx.beginPath(); for (let i = 0; i < m; i++) ctx.lineTo(X(i), Y(s.hi[i])); for (let i = m - 1; i >= 0; i--) ctx.lineTo(X(i), Y(s.lo[i])); ctx.closePath(); ctx.fill(); ctx.restore(); }
    ctx.save(); ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 2; ctx.beginPath(); for (let i = 0; i < m; i++) { const x = X(i), y = Y(s.mean[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); });
  return { X, Y };
}
function chartM(o) {
  const x0 = o.x0 ?? 56, x1 = o.x1 ?? 350, y0 = o.y0 ?? 388, y1 = o.y1 ?? 304, lo = o.lo ?? -.2, hi = o.hi ?? .6, n = o.n ?? NEP;
  const X = i => x0 + i / (NEP - 1) * (x1 - x0), Y = v => y0 - (clamp(v, lo, hi) - lo) / (hi - lo) * (y0 - y1);
  rline(x0 - 4, Y(0), x1 + 4, Y(0), 962, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1 - 2, 963, { c: C.line, w: 1, once: true });
  [[0, '0'], [hi, f2(hi)], [lo, f2(lo)]].forEach(([v, s]) => txt(s, x0 - 18, Y(v), { s: 12, c: C.soft }));
  txt('walk 1', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NEP}`, x1, y0 + 11, { s: 12, c: C.soft, al: 'right' }); if (o.title) txt(o.title, x0 - 10, y1 - 12, { s: 13, c: C.soft, al: 'left' });
  (o.series || []).forEach(s => { ctx.save(); ctx.globalAlpha = s.a ?? 1; ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 1.6; ctx.beginPath(); for (let i = 0; i < Math.min(n, NEP); i++) { const x = X(i), y = Y(s.v[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); });
}
// a measuring stick next to the doors: the yardstick value b
function stick(b, a = 1) { const x = 372, y0 = 140, y1 = 50, Y = v => y0 - v * (y0 - y1); alpha(a, () => { rline(x, y0, x, y1, 990, { w: 1.4 }); [0, .5, 1].forEach(v => { rline(x - 4, Y(v), x + 4, Y(v), 991 + v * 4, { w: 1, once: true }); txt(String(v), x + 14, Y(v), { s: 11, c: C.soft }); });
  rline(x - 10, Y(b), x + 10, Y(b), 995, { c: C.goldInk, w: 2.4 }); txt('yardstick', x, y1 - 12, { s: 12, c: C.goldInk }); }); }
// strip of the possible single-try C-nudges (dot size ∝ chance)
function nudgeStrip(y, list, col, label, a = 1) { const x0 = 70, x1 = 340, X = v => lerp(x0, x1, (v + .7) / 1.4);
  alpha(a, () => { rline(x0, y, x1, y, 997 + y, { c: C.line, w: 1, once: true }); txt(label, x0 - 6, y - 16, { s: 13, c: col, al: 'left' });
    [-.5, 0, .5].forEach(v => txt(String(v).replace('-', '−'), X(v), y + 13, { s: 11, c: C.soft })); list.forEach(q => dot(X(q.v), y, 3 + 14 * Math.sqrt(q.pr), col, .55)); }); }

const STEPS = [
 { h: 'A crumb behind the worst door', p: [`Back in the door room with plain REINFORCE, ${AHIT} picks into a run. B’s crumbs have made B the favourite, though C is better. Now Pip opens door A, the worst door, and a crumb appears: reward 1.`, `So A becomes more likely, though it pays only 2 times in 10. An empty door gives reward 0: no change. A picked door is pushed **up** or left alone; it loses chance only slowly, when other doors pay.`],
   shot: [3, 10],
   draw(t) { const th = AH.th, after = th.map((v, j) => v + AD * AH.A * ((j === 0) - AH.p[j])), u = ease((t - 3) / 1.2); room(t, { th: th.map((v, j) => lerp(v, after[j], u)), glow: 0, secret: t > 6 });
     if (t > .8 && t < 4) txt('crumb! +1', DX[0] + 80, 214, { s: 18, c: '#b07d12', a: clamp((t - .8) * 3) * (1 - clamp((t - 3.4) * 2)) });
     if (t > 4) card([{ s: `A: ${f2(AH.p[0])} → ${f2(softmax(after)[0])}, pushed up by a lucky crumb`, c: C.roseInk }, { s: 'reward 0: no push at all', c: C.soft }, { s: 'a picked door: pushed up or left alone', c: C.soft }], t - 4, { gap: 1.3 }); } },
 { h: 'Better than usual?', p: ['Think of grading on a curve. A score of 75 is great if the class usually gets 60, and poor if it usually gets 90. What matters is the score compared with what is usual.', 'So Pip keeps a **baseline**, a yardstick for a usual reward, and nudges by reward minus yardstick instead of the reward itself. Better than usual: push up. Worse than usual: push down.'],
   shot: [3, 10],
   draw(t) { const xs = [100, 300]; [[60, 'class usually 60', '#3f7f7c', 'great'], [90, 'class usually 90', C.roseInk, 'poor']].forEach(([m, s, col, w], j) => { const a = clamp(t - .5 - j * 1.5); if (a <= 0) return;
       alpha(a, () => { const x = xs[j], Y = v => 220 - (v - 40) * 2.4; rline(x - 50, 220, x + 50, 220, 980 + j, { c: C.line }); box(x - 20, Y(75), 40, 220 - Y(75), C.yellow); txt('75', x, Y(75) - 10, { s: 15 });
         ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = C.ink; ctx.beginPath(); ctx.moveTo(x - 46, Y(m)); ctx.lineTo(x + 46, Y(m)); ctx.stroke(); ctx.restore(); txt(s, x, 246, { s: 14, c: C.soft }); txt(w, x, 268, { s: 17, c: col }); }); });
     if (t > 4.5) card(['nudge by  (reward − yardstick)', { s: 'above the yardstick: push up', c: '#3f7f7c' }, { s: 'below the yardstick: push down', c: C.roseInk }], t - 4.5, { gap: 1.2 }); } },
 { h: 'The advantage', p: ['Reward minus yardstick has a name: the **advantage**. It says how much better this try went than usual.', 'A simple yardstick is the average of all of Pip’s rewards so far in this run. At first a third each, Pip gets a crumb about half the time, so the yardstick is near 0.5. A crumb from A then gives advantage 1 − 0.5 = +0.5; an empty door gives 0 − 0.5 = −0.5, a push down.'],
   shot: [3, 10],
   draw(t) { room(t, { secret: 1 }); stick(.5, clamp(t - .5));
     card(['advantage = reward − yardstick', { s: 'crumb:    1 − 0.5 = +0.5   push up', c: '#3f7f7c' }, { s: 'nothing:  0 − 0.5 = −0.5   push down', c: C.roseInk }], t - 2, { gap: 1.4 }); } },
 { h: 'Same direction, on average', p: ['Does subtracting a yardstick bend the direction Pip learns in? No, because the yardstick does not depend on which door Pip picks. Take door A, yardstick 0.5. Picked, a third of the time, A’s preference loses 0.5 × 0.67 = 0.33. Not picked, two thirds of the time, it gains 0.5 × 0.33 = 0.17.', 'On average: a third of 0.33 minus two thirds of 0.17, which is 0. The same holds for every door and any yardstick.'],
   shot: [3, 10],
   draw(t) { room(t, { secret: 1 }); stick(.5);
     card(['the yardstick’s part of A’s nudge (−0.5 × factor):', '   A picked (chance 1/3):  −0.5 × 0.67 = −0.33', '   A not picked (2/3):   +0.5 × 0.33 = +0.17', { s: '   average: 1/3 × (−0.33) + 2/3 × 0.17 = 0', c: C.goldInk }], t - 1, { gap: 1.4, fs: 16 }); } },
 { h: 'Smaller spread', p: ['So the average nudge stays −0.10, 0, +0.10, the exact gradient. What changes is the **spread**: how far a single nudge typically lands from that average, measured in preference units. Here each dot is one possible single-try nudge to C’s preference; its size shows how often it happens.', `Combining all three doors, the spread is ${f2(SP0.sd)} plain and ${f2(SP5.sd)} with the yardstick.`],
   shot: [3, 10],
   draw(t) { room(t, { secret: 1, rows: false, pie: false, pip: false }); const C0 = CN(0), C5 = CN(.5);
     nudgeStrip(222, C0, '#8a8178', 'nudge to C’s preference, one try: plain', clamp(t - 1));
     nudgeStrip(290, C5, DINK[2], 'with yardstick 0.5', clamp(t - 3));
     txt('dot size = how often that outcome happens', W / 2, 330, { s: 13, c: C.soft, a: clamp(t - 4) });
     if (t > 5) card([`average nudge to C: +0.10 both ways`, { s: `spread, all doors: ${f2(SP0.sd)} plain, ${f2(SP5.sd)} with yardstick`, c: C.goldInk }, { s: '(typical distance from the average nudge)', c: C.soft }], t - 5, { gap: 1.3, y: 340, fs: 15, lh: 19 }); } },
 { h: 'Less spread, faster learning', p: [`Now ${NR} runs of ${NP} picks each, with and without the average-reward yardstick, α = ${AD}. Both use the same dice. Along the bottom: picks. Up the side: Pip’s chance of picking C. Lines: the average over runs. Shaded: where the middle 80 percent of runs lie.`, `After 50 picks the chance of C averages ${f2(D0.mean[49])} plain and ${f2(DA.mean[49])} with the yardstick, and the shaded band is much narrower.`],
   shot: [4, 12],
   draw(t) { const n = clamp(Math.floor((t - 1) * 40), 0, NP); room(t, { secret: 1, rows: false, pie: false, pip: false });
     band({ y0: 384, y1: 196, n, series: [{ mean: D0.mean, lo: D0.lo, hi: D0.hi, col: C.grey, w: 2, ba: .3 }].concat(t > 3 ? [{ mean: DA.mean, lo: DA.lo, hi: DA.hi, col: DINK[2], w: 2.4, ba: .4 }] : []), title: 'chance of picking C' });
     txt('grey: plain', 345, 350, { s: 14, c: '#8a8178', al: 'right' }); if (t > 3) txt('teal: with yardstick', 345, 368, { s: 14, c: DINK[2], al: 'right' }); } },
 { h: 'Fewer stuck runs', p: [`Look at single runs at pick ${NP}. Plain REINFORCE leaves ${D0.bad} of ${NR} runs with C’s chance still below one half: a few lucky crumbs from A or B pushed them the wrong way, and empty doors change nothing; only C’s crumbs slowly pull them back.`, `With the yardstick, ${DA.bad} of ${NR}. Empty doors now push down, so a wrong lead gets corrected.`],
   shot: [3, 9],
   draw(t) { room(t, { secret: 1, rows: false, pie: false, pip: false }); const u = clamp(t - 1);
     [[D0, 'plain', C.grey, 110], [DA, 'with yardstick', DINK[2], 290]].forEach(([D, s, col, x], j) => { const a = clamp(t - 1 - j * 1.5); if (a <= 0) return;
       alpha(a, () => { const h = D.bad / NR * 1500; box(x - 30, 330 - h, 60, h, col); rline(x - 50, 330, x + 50, 330, 970 + j, { c: C.line }); txt(`${D.bad} of ${NR}`, x, 320 - h, { s: 17 }); txt(s, x, 346, { s: 15, c: col === C.grey ? '#8a8178' : col }); }); });
     txt(`runs with chance of C below 0.5 at pick ${NP}`, W / 2, 196, { s: 15, c: C.soft, a: u }); } },
 { h: 'Your turn', ctl: { label: 'yardstick', values: BS, def: 5, fmt: v => v === 'avg' ? 'average' : String(v), onChange: st => {} },
   p: [`Drag the **yardstick**: a fixed number from 0 to 1, or Pip’s running average reward. Each setting reruns the same ${NR} runs.`, `Any yardstick from 0.25 to 1 beats none here. A yardstick of 1 is the mirror of plain REINFORCE: crumbs change nothing, empty doors push down. It helps less: ${f2(doorStats(1).mean[49])} after 50 picks, against ${f2(doorStats(.75).mean[49])} with 0.75. The running average is not the best, but needs no tuning.`],
   shot: [3, 9],
   draw(t, st) { const b = BS[st.ci ?? 5], D = doorStats(b), n = clamp(Math.floor((t - (st.changed ?? 0) - .2) * 80), 0, NP);
     room(t, { secret: 1, rows: false, pie: false, pip: false }); if (b !== 'avg') stick(b);
     band({ y0: 368, y1: 196, n, series: [{ mean: D0.mean, col: C.grey, w: 1.4, lo: null }, { mean: D.mean, lo: D.lo, hi: D.hi, col: DINK[2], w: 2.4 }], title: 'chance of picking C (grey line: plain)' });
     txt(`after 50 picks ${f2(D.mean[49])} · below 0.5 at pick ${NP}: ${D.bad} of ${NR} runs`, W / 2, 404, { s: 14, c: C.soft }); } },
 { h: 'One yardstick is not enough', p: [`Back to the maze, with REINFORCE and the same α = ${A0} as before. Here one average won’t do: late in training, Pip’s return from the start is usually around ${f2(VFIN[key(0, 3)])}, from the cell left of the cheese ${f2(VFIN[key(3, 0)])}. The numbers are Pip’s yardsticks after ${NEP} walks, learned as the next step shows.`, 'So Pip keeps a yardstick per cell: its **value** V(s), the usual return from that cell under its current chances, as in the values reel.'],
   shot: [3, 10],
   draw(t) { G.maze({ t, V: VFIN }); for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.term(c, r) || G.isRock(c, r)) continue; const v = VFIN[key(c, r)]; txt(f2(v), G.ox + c * 64 + 57, G.oy + r * 64 + 12, { s: 14, al: 'right', c: v < -.005 ? C.roseInk : C.goldInk, a: clamp(t - .5 - (c + r) * .15) }); }
     if (t > 4) card(['a yardstick per cell: V(s)', { s: 'the usual return from that cell, under Pip’s chances', c: C.soft }], t - 4, { gap: 1.3 }); } },
 { h: 'Learning the yardsticks', p: [`Pip learns V(s) like the Monte Carlo reel, from returns, with no rulebook: after each walk, each visited cell’s yardstick moves a tenth of the way toward the return that followed.`, 'Each move is nudged by its advantage: return minus its cell’s yardstick, using the yardsticks from before the walk; only then are they updated. Preferences and yardsticks learn side by side.'],
   shot: [3, 10],
   draw(t) { const n = clamp(Math.floor((t - 1) * 25), 0, NEP), e = EB[Math.max(0, n - 1)]; G.maze({ t, V: n ? e.V : new Float64Array(20) });
     for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.term(c, r) || G.isRock(c, r)) continue; const v = n ? e.V[key(c, r)] : 0; txt(fz(v), G.ox + c * 64 + 57, G.oy + r * 64 + 12, { s: 14, al: 'right', c: v < -.005 ? C.roseInk : v > .005 ? C.goldInk : C.soft }); }
     card(['after each walk, in every visited cell:', '   V(s) += 0.1 × (return − V(s))', { s: '   advantage = return − V(s)  → nudge', c: C.goldInk }], t - 2, { gap: 1.3 }); caption(n ? `after walk ${n}` : ''); } },
 { h: 'Better or worse than usual', p: LATE && LATEG ? [`Walk ${LATEG.e + 1}: Pip steps from the cell left of the cheese straight onto it. Return 1, yardstick there ${f2(LATEG.q.b)}: advantage ${sg(LATEG.q.G - LATEG.q.b)}, a gentle push up.`, `Walk ${LATE.e + 1}: in that same cell Pip’s dice pick ${['up', 'right', 'down', 'left'][LATE.q.m]}, then right, right: the cheese comes on the third move, return ${f2(LATE.q.G)}. Advantage ${f2(LATE.q.G - LATE.q.b)}: that move gets pushed down. Plain REINFORCE would have pushed it **up**, since its return was positive.`] : ['missing'],
   shot: [3, 10],
   draw(t) { const e = EB[LATE.e]; G.maze({ t, hi: [[3, 0, C.gold, .25]] }); fans(G, e.th, { a: .4 }); G.pip(cx(3), cy(0), 1, t);
     card([{ s: `straight onto the cheese: 1 − ${f2(LATEG.q.b)} = ${sg(LATEG.q.G - LATEG.q.b)}  up`, c: '#3f7f7c' }, { s: `${['up', 'right', 'down', 'left'][LATE.q.m]} first: ${f2(LATE.q.G)} − ${f2(LATE.q.b)} = ${f2(LATE.q.G - LATE.q.b)}  down`, c: C.roseInk }, { s: 'plain REINFORCE: both pushed up', c: C.soft }], t - 1, { gap: 1.8 }); } },
 { h: 'How much calmer?', p: [`Freeze Pip’s chances and yardsticks after walk 50, 150 and 300; run ${NVAR.toLocaleString('en-US')} walks from each. Measure the **variance** of one walk’s nudges: the average squared distance from their mean, added up over all 64 preferences, with and without yardsticks.`, `After walk 50 the young yardsticks are poor guesses and make it **worse**: ${VAR[0].plain.toFixed(2)} becomes ${VAR[0].base.toFixed(2)}. After 150 they cut it by a factor of ${VAR[1].ratio.toFixed(1)}, after 300 by ${VAR[2].ratio.toFixed(1)}. The average direction never changes; a poor yardstick only adds noise.`],
   shot: [4, 11],
   draw(t) { const x0 = 70, base = 250, mx = Math.max(...VAR.map(v => Math.max(v.plain, v.base)));
     txt('variance of one walk’s nudges (preference units squared)', W / 2, 40, { s: 15, c: C.soft });
     VAR.forEach((v, i) => { const a = clamp(t - .8 - i * 1.4); if (a <= 0) return; const x = x0 + i * 110, h0 = v.plain / mx * 170, h1 = v.base / mx * 170;
       alpha(a, () => { box(x, base - h0, 30, h0, C.grey); box(x + 36, base - h1, 30, h1, DINK[2]); txt(v.plain.toFixed(2), x + 15, base - h0 - 9, { s: 13 }); txt(v.base.toFixed(2), x + 51, base - h1 - 9, { s: 13, c: DINK[2] });
         txt(`after walk ${v.w}`, x + 33, base + 14, { s: 14, c: C.soft }); txt(v.ratio >= 1 ? `÷ ${v.ratio.toFixed(1)}` : `× ${(1 / v.ratio).toFixed(1)} worse`, x + 33, base + 32, { s: 15, c: v.ratio >= 1 ? C.goldInk : C.roseInk }); }); });
     rline(50, base, 370, base, 975, { c: C.line });
     if (t > 5.5) card([{ s: 'grey: plain REINFORCE', c: '#8a8178' }, { s: 'teal: with a yardstick per cell', c: DINK[2] }], t - 5.5, { gap: 1, y: 300 }); } },
 { h: 'An honest result', p: [`Over ${NRUN} runs in this maze the average curves almost overlap: last 25 walks, ${f2(MP.end)} plain and ${f2(MV.end)} with yardsticks. The spread between runs, the standard deviation of each run’s last-25 average return, shrinks a little: ${MP.sd.toFixed(3)} to ${MV.sd.toFixed(3)}.`, 'Why so little? Most learning happens early, while returns are near 0 and the yardsticks are still poor. The door room, where the usual reward sat far above 0, showed the big effect. Baselines help most when returns share a large common level.'],
   shot: [4, 11],
   draw(t) { const n = clamp(Math.floor((t - .8) * 60), 0, NEP);
     chartM({ y0: 280, y1: 140, n, series: [{ v: MP.sm, col: C.grey, w: 2.4 }, { v: MV.sm, col: DINK[2], w: 2 }], title: `average return of ${NRUN} runs, same dice` });
     txt('grey: plain', 90, 96, { s: 14, c: '#8a8178', al: 'left' }); txt('teal: with yardsticks', 90, 114, { s: 14, c: DINK[2], al: 'left' });
     if (t > 5) card([`last 25 walks: ${f2(MP.end)} plain, ${f2(MV.end)} with yardsticks`, { s: `spread between runs: ${MP.sd.toFixed(3)} → ${MV.sd.toFixed(3)}`, c: C.soft }], t - 5, { gap: 1.3, y: 306 }); } },
 { h: 'Where it is used', p: ['Nearly every modern policy gradient method subtracts a baseline. Actor-critic methods, A2C and PPO among them, learn V(s) with a second network and nudge by an advantage: return minus V(s), or a refinement of it.', 'For language models, methods like RLOO and GRPO sample several answers to the same question and use their average reward as the yardstick: the door-room idea, almost exactly.'],
   shot: [3, 9],
   draw(t) { alpha(clamp(t - .5), () => { L.robot(100, 130, 2, t); txt('actor + value: advantage', 100, 196, { s: 15 }); });
     alpha(clamp(t - 2), () => { ['answer 1  ★ 0.9', 'answer 2  ★ 0.4', 'answer 3  ★ 0.5'].forEach((s, i) => txt(s, 290, 92 + i * 24, { s: 14, c: i === 0 ? '#3f7f7c' : C.roseInk })); rrect(225, 74, 130, 82, 977); txt('yardstick: average 0.6', 290, 174, { s: 14, c: C.goldInk }); txt('language models', 290, 196, { s: 15 }); });
     if (t > 4) card(['baseline = usual outcome', { s: 'advantage = outcome − baseline', c: C.goldInk }], t - 4, { gap: 1.2 }); } },
 { h: 'Recap', p: ['A **baseline** is a yardstick for the usual return. Nudging by return minus baseline, the **advantage**, pushes up only what went better than usual and pushes down what went worse.', 'It does not change the average direction, only the noise. It helps most when returns share a big common level; a common yardstick is the state’s value V(s), which the next reel learns step by step.'],
   shot: [3, 9],
   draw(t) { room(t, { secret: 1, th: RA.th }); stick(RA.log[NP - 1].b, 1);
     card(['advantage = return − baseline', 'baseline: average return, or V(s) per state', 'same average direction, less spread', { s: 'door room: big gain · maze: small, later', c: C.soft }], t - .3, { gap: 1, fs: 16 }); } },
];

RLStory.register('baseline', { title: 'Baselines & Advantage', track: 'Policy Gradients', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { AHIT, AH: { p: AH.p, A: AH.A }, SP0: SP0.sd, SP5: SP5.sd, SPmean: [SP0.mean, SP5.mean], D0: [D0.mean[49], D0.mean[NP - 1], D0.bad], DA: [DA.mean[49], DA.mean[NP - 1], DA.bad],
    bs: () => BS.map(b => [b, doorStats(b).mean[49], doorStats(b).bad]), VFIN: [VFIN[key(0, 3)], VFIN[key(3, 0)]], LATE: LATE && { e: LATE.e, G: LATE.q.G, b: LATE.q.b, m: LATE.q.m }, LATEG: LATEG && { e: LATEG.e, G: LATEG.q.G, b: LATEG.q.b },
    VAR, MP: [MP.end, MP.sd], MV: [MV.end, MV.sd], raEnd: softmax(RA.th), raB: RA.log[NP - 1].b } });
})();
