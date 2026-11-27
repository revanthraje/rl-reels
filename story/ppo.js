// PPO as a story: Pip starts from a blank policy (all chances 1/4). To squeeze more out of each batch of walks it takes 10 passes over it.
// Without brakes, one batch drives Pip into never finishing a walk, and it is stuck for good (every return 0, nothing to learn).
// Fix: the clipped ratio objective. Every number comes from the seeded simulation below. World: pg_world.md World 2 (dry floor).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({});
const { cx, cy, maze } = G;

// ───── simulation: Pip's maze, tabular softmax, batches of 10 walks, advantage = return that followed − exact critic ─────
const GAM = 0.9, CAP = 100, NB = 10, MV = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const key = (c, r) => r * 5 + c, SK = key(0, 3);
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);
const NXT = new Int8Array(80), RW = new Int8Array(80);
for (let k = 0; k < 20; k++) { const c = k % 5, r = (k / 5) | 0; for (let m = 0; m < 4; m++) { const [nc, nr] = G.step(c, r, m); NXT[k * 4 + m] = key(nc, nr); RW[k * 4 + m] = G.term(nc, nr); } }
const IDX = new Array(20).fill(-1); CELLS.forEach(([c, r], i) => { IDX[key(c, r)] = i; });
function probs(T) { const P = new Float64Array(80); for (let k = 0; k < 20; k++) { const o = k * 4, mx = Math.max(T[o], T[o + 1], T[o + 2], T[o + 3]); let s = 0;
  for (let m = 0; m < 4; m++) { P[o + m] = Math.exp(T[o + m] - mx); s += P[o + m]; } for (let m = 0; m < 4; m++) P[o + m] /= s; } return P; }
// exact values of a policy (solve the 16 Bellman equations with the rulebook): the critic, and the score. Pip cannot do this.
function evalP(P) { const n = 16, M = [], b = new Float64Array(n);
  CELLS.forEach(([c, r], i) => { const k = key(c, r), row = new Float64Array(n); row[i] = 1;
    for (let m = 0; m < 4; m++) { const p = P[k * 4 + m], rw = RW[k * 4 + m]; b[i] += p * rw; if (!rw) row[IDX[NXT[k * 4 + m]]] -= GAM * p; } M.push(row); });
  for (let i = 0; i < n; i++) { let pv = i; for (let j = i + 1; j < n; j++) if (Math.abs(M[j][i]) > Math.abs(M[pv][i])) pv = j;
    let t = M[i]; M[i] = M[pv]; M[pv] = t; t = b[i]; b[i] = b[pv]; b[pv] = t;
    for (let j = 0; j < n; j++) { if (j === i) continue; const f = M[j][i] / M[i][i]; if (!f) continue; for (let q = i; q < n; q++) M[j][q] -= f * M[i][q]; b[j] -= f * b[i]; } }
  const V = new Float64Array(20); CELLS.forEach(([c, r], i) => { V[key(c, r)] = b[i] / M[i][i]; }); return V; }
const score = T => evalP(probs(T))[SK];
function walk(P, R) { let k = SK; const tr = [];
  for (let i = 0; i < CAP; i++) { const u = R(); let m = 0, acc = P[k * 4]; while (u > acc && m < 3) { m++; acc += P[k * 4 + m]; }
    const nk = NXT[k * 4 + m], rew = RW[k * 4 + m]; tr.push({ k, m, nk, rew, p: P[k * 4 + m] }); k = nk; if (rew) break; }
  let Gr = 0; for (let i = tr.length - 1; i >= 0; i--) { Gr = tr[i].rew + GAM * Gr; tr[i].G = Gr; } return tr; }
function batch(T, R) { const P = probs(T), V = evalP(P), S = [], Wk = [];
  for (let i = 0; i < NB; i++) { const tr = walk(P, R); Wk.push(tr); for (const s of tr) { s.A = s.G - V[s.k]; S.push(s); } } return { S, W: Wk, V, P }; }
// one pass: gradient of the (clipped) objective  (1/10) Σ min(ratio × A, clip(ratio, 1 − ε, 1 + ε) × A);  ε = Infinity → no clip
function pass(T, S, eps, lr = LR) { const P = probs(T), g = new Float64Array(80);
  for (const s of S) { const o = s.k * 4, r = P[o + s.m] / s.p, A = s.A, cl = r < 1 - eps ? 1 - eps : r > 1 + eps ? 1 + eps : r;
    if (r * A > cl * A) continue;                                   // the flat part: this move stops pushing
    for (let b = 0; b < 4; b++) g[o + b] += A * r * ((b === s.m ? 1 : 0) - P[o + b]) / NB; }
  return T.map((v, i) => v + lr * g[i]); }
const LR = 3, K = 10;
function update(T, S, eps, k = K, lr = LR) { let Tn = Float64Array.from(T); const hist = [Tn]; for (let e = 0; e < k; e++) { Tn = pass(Tn, S, eps, lr); hist.push(Tn); } return { T: Tn, hist }; }
function run(seed, eps, k = K, iters = 40, lr = LR) { const R = rng(5000 + seed); let T = new Float64Array(80); const sc = [score(T)];
  for (let it = 0; it < iters; it++) { const B = batch(T, R); T = update(T, B.S, eps, k, lr).T; sc.push(score(T)); } return { sc, T }; }
// test walks: how 1000 walks end under a policy (rng(1), so every policy faces the same dice stream)
function ends(T) { const P = probs(T), R = rng(1); let ch = 0, ca = 0; for (let i = 0; i < 1000; i++) { const w = walk(P, R), e = w[w.length - 1].rew; if (e > 0) ch++; else if (e < 0) ca++; } return { ch, ca, no: 1000 - ch - ca }; }
const klAvg = (Pn, Po, S) => { let s = 0; for (const x of S) for (let m = 0; m < 4; m++) { const o = x.k * 4 + m; s += Pn[o] * Math.log(Pn[o] / Po[o]); } return s / S.length; };

// the story run (run 14): the very first batch from a blank policy
const T0 = new Float64Array(80), P0 = probs(T0), S0 = score(T0), E0 = ends(T0);
const B1 = batch(T0, rng(5000 + 14)), ENDW = B1.W.map(w => w[w.length - 1].rew), NCAT = ENDW.filter(e => e < 0).length, NCH = ENDW.filter(e => e > 0).length, NSTOP = ENDW.filter(e => !e).length;
const STOPS = B1.W.filter(w => !w[w.length - 1].rew).flat(), STOPPOS = STOPS.filter(s => s.A > 0).length, STOPAVG = STOPS.reduce((a, s) => a + s.A, 0) / STOPS.length;
const VMIN = Math.min(...CELLS.map(([c, r]) => B1.V[key(c, r)]));
const CH = B1.S.find(s => s.rew > 0);                                   // the step onto the cheese
const U1 = update(T0, B1.S, Infinity, 1), UN = update(T0, B1.S, Infinity), UC = update(T0, B1.S, .2);
const SC1 = score(U1.T), SCN = score(UN.T), SCC = score(UC.T), EN = ends(UN.T), EC = ends(UC.T);
const ratioOf = (T, s) => probs(T)[s.k * 4 + s.m] / s.p;
const RN = UN.hist.map(T => ratioOf(T, CH)), RC = UC.hist.map(T => ratioOf(T, CH));
const RALL = probs(UC.T), RATIOS = B1.S.map(s => RALL[s.k * 4 + s.m] / s.p), ROUT = RATIOS.filter(r => r > 1.2 || r < .8).length, RMAX = Math.max(...RATIOS), RMIN = Math.min(...RATIOS);
const KLN = klAvg(probs(UN.T), P0, B1.S), KLC = klAvg(probs(UC.T), P0, B1.S);
const RUNN = run(14, Infinity), RUNC = run(14, .2);
// after the collapse: next batch (same dice stream as the run) — every walk stopped, largest advantage
const AFTER = (() => { const R = rng(5014); const Bb = batch(T0, R); const T1 = update(T0, Bb.S, Infinity).T; const B2 = batch(T1, R);
  return { stopped: B2.W.filter(w => !w[w.length - 1].rew).length, maxA: Math.max(...B2.S.map(s => Math.abs(s.A))) }; })();
// averaged over 50 runs (runs 0–49), progressive and memoised
const NRUN = 50, AV = {};
function avg(id, fn) { let a = AV[id]; if (!a) a = AV[id] = { k: 0, sum: new Float64Array(41), stuck: 0 };
  const t0 = Date.now(); while (a.k < NRUN && Date.now() - t0 < 14) { const sc = fn(a.k); sc.forEach((v, i) => { a.sum[i] += v; }); if (sc[40] < .1) a.stuck++; a.k++; }
  return { k: a.k, done: a.k >= NRUN, stuck: a.stuck, v: Array.from(a.sum, v => v / Math.max(1, a.k)) }; }
const avgAll = (id, fn) => { let a; do a = avg(id, fn); while (!a.done); return a; };
const EPSV = [0.05, 0.1, 0.2, 0.3, 0.5, Infinity], fmtE = e => e === Infinity ? 'no clip' : String(e);
const fnE = e => s => run(s, e).sc, fn1 = s => run(s, Infinity, 1).sc, fn30 = s => run(s, Infinity, 1, 40, 30).sc, fnSmall = s => run(s, Infinity, K, 40, .3).sc;

// ───── formatting ─────
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−');
const f3 = v => (Math.abs(v) < .0005 ? 0 : v).toFixed(3).replace('-', '−');

// ───── drawing helpers ─────
function fans(P, o = {}) {
  const a = o.a ?? 1;
  for (const [c, r] of CELLS) { const k = key(c, r); const aa = typeof a === 'function' ? a(k) : a; if (aa <= 0) continue;
    const ps = [0, 1, 2, 3].map(m => P[k * 4 + m]), mx = Math.max(...ps), x = cx(c), y = cy(r) + 2;
    for (let m = 0; m < 4; m++) { const p = ps[m]; if (p < .015) continue; const [dx, dy] = MV[m], big = p === mx && mx > .3, r0 = 4, len = r0 + p * 25;
      alpha(aa, () => arrow(x + dx * r0, y + dy * r0, x + dx * len, y + dy * len, 300 + k * 4 + m, { c: big ? C.goldInk : '#a89c8f', w: big ? 1.7 : 1.1, head: 4 + 3 * p })); } }
}
const mixP = (A, B, u) => A.map((v, i) => lerp(v, B[i], u));
function pipAt(k, t) { L.mouse(cx(k % 5) - 18, cy((k / 5) | 0) + 18, 1, .55, t); }
function trailW(w, a = .45, n) { const pts = [SK].concat(w.slice(0, n ?? w.length).map(s => s.nk)), jx = i => ((i * 7) % 11) - 5, jy = i => ((i * 5) % 11) - 5;
  ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = C.teal; ctx.lineWidth = 2.2; ctx.lineJoin = 'round'; ctx.beginPath();
  pts.forEach((k, i) => { const x = cx(k % 5) + jx(i), y = cy((k / 5) | 0) + jy(i); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.restore(); }
function upChart(o) {
  const lo = -.05, hi = .6, N = 40, X = i => o.x0 + i / N * (o.x1 - o.x0), Y = v => o.y0 - (v - lo) / (hi - lo) * (o.y0 - o.y1);
  rline(o.x0 - 4, Y(0), o.x1 + 4, Y(0), 960, { c: C.line, w: 1, once: true }); rline(o.x0 - 4, o.y0, o.x0 - 4, o.y1, 961, { c: C.line, w: 1, once: true });
  txt('0', o.x0 - 13, Y(0), { s: 12, c: C.soft }); txt('0.5', o.x0 - 17, Y(.5), { s: 12, c: C.soft });
  txt('update 0', o.x0, o.y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt('update 40', o.x1, o.y0 + 10, { s: 12, c: C.soft, al: 'right' });
  ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.grey; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(o.x0, Y(.531)); ctx.lineTo(o.x1, Y(.531)); ctx.stroke(); ctx.restore();
  txt('best possible 0.53', o.x0 + 4, Y(.531) - 9, { s: 12, c: C.soft, al: 'left' });
  if (o.title) txt(o.title, o.x0 - 8, o.y1 - 10, { s: 13, c: C.soft, al: 'left' });
  (o.series || []).forEach(s => { const n = Math.min(s.n ?? o.n ?? N, s.v.length - 1); ctx.save(); ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 1.8; ctx.lineJoin = 'round'; ctx.beginPath();
    for (let i = 0; i <= n; i++) { const x = X(i), y = Y(s.v[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore();
    if (s.label && n >= 40) txt(s.label, o.x1 + 3, Y(s.v[40]) + (s.dy || 0), { s: 12, c: s.col, al: 'left' }); });
}
// the clipped objective as a hand-drawn curve. sign = +1 (advantage > 0) or −1; u = how much is drawn (0..1); parts: which pieces to show
function clipPlot(x0, y0, w, h, sign, t, o = {}) {
  const eps = .2, X = r => x0 + r / 2 * w, Y = v => y0 - h / 2 - v * h / 4.4, A = sign;
  rline(x0, Y(0), x0 + w, Y(0), 1100 + sign, { c: C.line, w: 1, once: true }); rline(x0, y0, x0, y0 - h, 1102 + sign, { c: C.line, w: 1, once: true });
  txt('ratio', x0 + w + 4, Y(0), { s: 13, c: C.soft, al: 'left' }); txt('objective', x0 - 6, y0 - h - 10, { s: 13, c: C.soft, al: 'left' }); [0, 1, 2].forEach(r => txt(String(r), X(r), Y(0) + (sign > 0 ? 12 : -12) * (r ? 1 : 0) + (r ? 0 : 12), { s: 12, c: C.soft }));
  // the band 0.8 … 1.2
  if (o.band) box(X(1 - eps), y0 - h, X(1 + eps) - X(1 - eps), h, C.yellow, .35 * o.band);
  // the unclipped line ratio × A (dashed), then the clipped objective (thick)
  const plain = clamp(o.plain ?? 1), clipU = clamp(o.clip ?? 0);
  if (plain > 0) { ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = C.soft; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(2 * plain), Y(2 * plain * A)); ctx.stroke(); ctx.restore(); }
  if (clipU > 0) { ctx.save(); ctx.strokeStyle = sign > 0 ? C.goldInk : C.roseInk; ctx.lineWidth = 2.6; ctx.lineJoin = 'round'; ctx.beginPath();
    for (let i = 0; i <= 80 * clipU; i++) { const r = i / 40, cl = clamp(r, 1 - eps, 1 + eps), v = Math.min(r * A, cl * A); i ? ctx.lineTo(X(r), Y(v)) : ctx.moveTo(X(r), Y(v)); } ctx.stroke(); ctx.restore(); }
}

const PX = r => 70 + r * 130, PY = v => 260 - 100 - v * 200 / 4.4;
const GREEN = '#3f7f7c', BLUE = '#4f7fb0', ROSEL = '#c0605f';

const STEPS = [
 { h: 'Pip starts from scratch', p: ['Same maze, dry floor: +1 for the step onto the cheese, −1 onto the cat, both end the walk, γ (gamma) is 0.9. A walk still going after 100 steps is stopped and scores 0.', `Pip’s four chances per cell, drawn as arrows, start at a quarter each. Of 1000 test walks, ${E0.ch} reach the cheese, ${E0.ca} the cat, ${E0.no} are stopped. Its **score**, the return it can expect from the start, worked out exactly with the rulebook, is ${f2(S0)}; the best possible is 0.53.`],
   shot: [3, 9],
   draw(t) { maze({ t }); fans(P0, { a: k => clamp((t - .4 - IDX[k] * .06) * 3) }); pipAt(SK, t);
     if (t > 3.5) card(['arrow length = chance of that move', `1000 test walks: ${E0.ch} cheese, ${E0.ca} cat, ${E0.no} stopped`, { s: `score ${f2(S0)}  ·  best possible 0.53`, c: C.soft }], t - 3.5, { gap: 1.2, fs: 16 }); } },
 { h: 'One update', p: [`Each update, Pip walks ${NB} times: a **batch**. Each move gets an **advantage**: the return that followed it minus the critic’s value of the cell. The critic here is exact, so we can watch the actor alone.`, `Then the move made gains α (alpha) × advantage × (1 − its chance), the others lose α × advantage × their chance; divide by ${NB}, the number of walks. α = ${LR}.`],
   shot: [6, 12.5],
   draw(t) { const D = .9, i = clamp(Math.floor((t - .3) / D), 0, NB - 1), w = B1.W[i]; maze({ t }); fans(P0, { a: .3 });
     for (let j = 0; j < i; j++) trailW(B1.W[j], .08); trailW(w, .7, Math.ceil(clamp(((t - .3) % D) / (D * .8)) * w.length) + (t > .3 + NB * D ? w.length : 0));
     const tally = B1.W.slice(0, i + 1).map(w => w[w.length - 1].rew), nc = tally.filter(e => e < 0).length, nh = tally.filter(e => e > 0).length, ns = tally.filter(e => !e).length;
     txt(`walk ${i + 1} of ${NB}`, 40, 298, { s: 16, al: 'left' }); txt(`cheese ${nh}   cat ${nc}   stopped ${ns}`, 360, 298, { s: 16, al: 'right', c: C.soft });
     if (t > 9.5) card(['advantage = return after the move − critic’s value', { s: `the update’s step size: α = ${LR}`, c: C.goldInk }], t - 9.5, { y: 318 }); } },
 { h: 'Walks are precious', p: ['This batch cost ten walks. The update uses it once, then throws it away. In a real game or on a real robot, collecting walks is the slow, expensive part; computing is cheap.', `So learn more from each batch: go over the same walks ${K} times, ${K} **passes**, each one an update with α = ${LR}. Careful: ten such passes can push up to ten times as far as one, a very big step, unless something stops them.`],
   shot: [3, 9],
   draw(t) { maze({ t }); fans(P0, { a: .3 }); B1.W.forEach((w, j) => trailW(w, .12)); box(0, 0, W, 285, C.bg, .35);
     for (let k = 0; k < K; k++) { const u = clamp(t - 3 - k * .35); if (u <= 0) continue; rrect(40 + k * 32, 300, 26, 34, 1200 + k, { fill: k < 1 ? C.yellow : '#efe6d8', a: u }); txt(String(k + 1), 53 + k * 32, 317, { s: 14, a: u }); }
     if (t > 3) txt(`… or ${K} passes over the same 10 walks?`, W / 2, 356, { s: 16, c: C.goldInk, a: clamp(t - 6.5) }); caption('pass 1 is the ordinary update', clamp(t - 3.5)); } },
 { h: 'The ratio', p: ['After the first pass the chances have changed, but the walks came from the old ones. So PPO weighs each move by the **ratio**: new chance ÷ old chance of that move in that cell.', `Example: the step onto the cheese, from the cell left of it. Its old chance was 0.25. After one pass it is ${f3(.25 * RN[1])}, so the ratio is ${f3(RN[1])}: new Pip would make that move about ${f2(RN[1])} times as often.`],
   shot: [4, 11],
   draw(t) { const u = ease((t - 2) / 1.5), P = mixP(P0, probs(UN.hist[1]), u); maze({ t }); box(G.ox + 3 * 64 + 3, G.oy + 3, 58, 58, C.gold, .25); fans(P, { a: k => k === key(3, 0) ? 1 : .35 });
     const pn = P[key(3, 0) * 4 + 1]; txt(`right: ${f3(pn)}`, cx(3), cy(0) + 24, { s: 13, c: C.goldInk });
     card(['ratio  =  new chance ÷ old chance', `cheese step:  ${f3(.25 * RN[1])} ÷ 0.25  =  ${f3(RN[1])}`, { s: 'ratio 1: the move’s chance has not changed', c: C.soft }], t - 3.5, { gap: 1.2 }); } },
 { h: 'What each pass climbs', p: ['Each pass makes an **objective** bigger: the average of **ratio × advantage** over the batch’s moves. Why the ratio? The walks came from old Pip. Weighting each move by new chance ÷ old chance turns an average over old Pip’s moves into an estimate for new Pip: moves new Pip makes more often count more.', 'At ratio 1 the slope of new ÷ old is the slope of log(new chance), so the first pass is exactly the ordinary update.'],
   shot: [4, 11],
   draw(t) { clipPlot(70, 260, 260, 200, 1, t, { plain: clamp((t - .8) / 2), clip: 0 }); txt('advantage > 0', 250, 40, { s: 15, al: 'left', c: C.goldInk });
     txt('ratio × advantage', PX(1.55), PY(1.55) - 22, { s: 14, c: C.soft, a: clamp(t - 2.5) }); if (t > 4) { dot(PX(1), PY(1), 4, C.ink); txt('ratio 1: start of pass 1', PX(1) + 8, PY(1) + 16, { s: 13, al: 'left' }); }
     card(['objective = average of  ratio × advantage', { s: 'advantage > 0: bigger ratio, bigger objective', c: C.goldInk }], t - 5.5, { y: 300 }); } },
 { h: 'Ten passes, no brakes', p: [`Pip takes ${K} passes over its first batch, pushing ratio × advantage each time, with no brake. Of 1000 test walks, ${EN.no === 1000 ? 'not one' : 1000 - EN.no} now ends: Pip wanders and bumps until it is stopped, meeting neither cat nor cheese.`, `So the score goes from ${f2(S0)} to ${f2(SCN)}. That looks harmless. It is not: watch what happens next.`],
   shot: [5, 12],
   draw(t) { const k = clamp(Math.floor((t - 1) / .6), 0, K), u = clamp(((t - 1) % .6) / .5), Pa = probs(UN.hist[Math.max(0, k - 1)]), Pb = probs(UN.hist[k]);
     maze({ t }); fans(k === 0 ? P0 : mixP(Pa, Pb, k >= K ? 1 : u)); txt(`pass ${k} of ${K}`, W / 2, 298, { s: 17 });
     if (t > 8) card([`score  ${f2(S0)}  →  ${f2(SCN)}`, { s: `1000 test walks: ${EN.ch} cheese, ${EN.ca} cat, ${EN.no} stopped`, c: C.roseInk }], t - 8, { y: 316, gap: 1 }); } },
 { h: 'Why never finish?', p: [`The batch: ${NCAT} walks hit the cat, ${NCH} the cheese, ${NSTOP} were stopped at 100 steps with return 0. The critic expected less than 0 everywhere, down to ${f2(VMIN)} beside the cat. So all ${STOPS.length} moves of the stopped walks had a positive advantage, ${f2(STOPAVG)} on average.`, `Pushed ten times over, “keep wandering” beat “reach the cheese”. One ordinary pass sees the same advantages but pushes once: the score barely moves, to ${f2(SC1)}, and later batches can correct it.`],
   shot: [4, 11],
   draw(t) { maze({ t }); fans(P0, { a: .25 }); B1.W.forEach((w, j) => { const e = w[w.length - 1].rew; if (!e) trailW(w, .35 * clamp(t - 1)); });
     const x0 = 40, y0 = 316; ENDW.forEach((e, j) => { const u = clamp(t - .5 - j * .15); if (u <= 0) return; const col = e > 0 ? C.gold : e < 0 ? C.rose : '#9fc9c7';
       box(x0 + j * 32, y0 - 14, 26, 28, col, .8 * u); txt(e > 0 ? '+' : e < 0 ? '−' : '0', x0 + j * 32 + 13, y0, { s: 16, a: u }); });
     txt('walk returns: + cheese, − cat, 0 stopped', x0, y0 + 26, { s: 13, al: 'left', c: C.soft, a: clamp(t - 2) });
     if (t > 4) txt(`stopped walks: return 0, critic said below 0  →  advantage > 0`, W / 2, 374, { s: 14, c: GREEN, a: clamp(t - 4) }); } },
 { h: 'Stuck for good', p: [`Now every walk is stopped. In the next batch, ${AFTER.stopped} of ${NB} walks stop at 100 steps, every return is 0, and the critic, exact for a Pip who never finishes, says about 0 everywhere. The largest advantage is ${AFTER.maxA.toFixed(4)}.`, `With no surprises there is nothing to learn from. The chart shows this run’s score after each update: forty updates later it is still ${f2(RUNN.sc[40])}.`],
   shot: [4, 11],
   draw(t) { const n = clamp(Math.floor((t - 1) * 5), 0, 40); upChart({ x0: 56, x1: 330, y0: 250, y1: 50, n, series: [{ v: RUNN.sc, col: ROSEL, label: f2(RUNN.sc[40]) }], title: `this run: score after each update, ${K} passes, no clip` });
     card(['every walk stopped  →  every return 0', { s: 'advantage ≈ 0  →  no push  →  no change', c: C.roseInk }], t - 2, { y: 300, gap: 1.3 }); } },
 { h: 'Clip the ratio', p: ['**PPO**, Proximal Policy Optimization, adds a brake. Clip the ratio to a band around 1: from 1 − ε to 1 + ε, with ε (epsilon, not the one from ε-greedy) usually 0.2. So 0.8 to 1.2.', 'For a move with positive advantage, follow ratio × advantage up to 1.2; beyond that the objective stays flat, so there is no reward for pushing further. Below 0.8 the line continues: making a good move rarer is still penalised.'],
   shot: [4, 11],
   draw(t) { clipPlot(70, 260, 260, 200, 1, t, { band: clamp(t - 1), plain: 1, clip: clamp((t - 3) / 3) }); txt('advantage > 0', 250, 40, { s: 15, al: 'left', c: C.goldInk });
     txt('0.8', 70 + .8 * 130, 268, { s: 12, c: C.goldInk, a: clamp(t - 1) }); txt('1.2', 70 + 1.2 * 130, 268, { s: 12, c: C.goldInk, a: clamp(t - 1) });
     if (t > 6.5) txt('flat: stop pushing', PX(1.6), PY(1.2) + 16, { s: 14, c: C.goldInk, a: clamp(t - 6.5) });
     card(['band: ratio from 1 − ε to 1 + ε', { s: 'positive advantage: no reward beyond 1 + ε', c: C.goldInk }], t - 7.5, { y: 300 }); } },
 { h: 'When the advantage is negative', p: ['For a move with negative advantage, it mirrors: lowering the ratio helps only down to 0.8, then the objective goes flat.', 'But raising that bad move’s chance still costs, with no cap. PPO keeps whichever of the two numbers, clipped or not, is lower. It removes the reward for going too far, never the penalty.'],
   shot: [4, 11],
   draw(t) { clipPlot(70, 260, 260, 200, -1, t, { band: 1, plain: 1, clip: clamp((t - 1) / 3) }); txt('advantage < 0', 250, 40, { s: 15, al: 'left', c: C.roseInk });
     if (t > 4) { txt('flat below 0.8', PX(.4), PY(-.8) + 16, { s: 14, c: C.roseInk, a: clamp(t - 4) }); txt('still falls: bad move made likelier', PX(1.1), PY(-1.9) + 4, { s: 13, c: C.roseInk, a: clamp(t - 5) }); }
     card(['objective = the lower of', '    ratio × advantage   and   clipped ratio × advantage'], t - 6.5, { y: 300, gap: 1.2 }); } },
 { h: 'Flat means stop', p: [`On a flat stretch the objective has no slope, so that move stops pushing. The cheese step again, over ${K} passes. Without the clip its ratio climbs to ${f2(RN[K])}. With ε = 0.2 it overshoots on the first pass, to ${f2(RC[1])}, then stays there.`, 'The first pass starts at ratio 1, inside the band, so the clip cannot hold it back yet. Nothing else in that cell moves it afterwards.'],
   shot: [5, 12],
   draw(t) { const x0 = 70, x1 = 340, y0 = 250, y1 = 60, X = k => x0 + k / K * (x1 - x0), Y = r => y0 - (r - .6) / 2.2 * (y0 - y1), n = clamp(Math.floor((t - 1) / .5), 0, K);
     box(x0, Y(1.2), x1 - x0, Y(.8) - Y(1.2), C.yellow, .4); txt('band 0.8 to 1.2', x1, Y(.8) + 10, { s: 12, c: C.goldInk, al: 'right' });
     rline(x0, y0, x1, y0, 1300, { c: C.line, w: 1, once: true }); rline(x0, y0, x0, y1, 1301, { c: C.line, w: 1, once: true });
     [1, 2].forEach(r => txt(String(r), x0 - 12, Y(r), { s: 12, c: C.soft })); txt('ratio of the cheese step', x0 - 8, y1 - 14, { s: 13, c: C.soft, al: 'left' });
     txt('pass 0', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt(`pass ${K}`, x1, y0 + 11, { s: 12, c: C.soft, al: 'right' });
     [[RN, ROSEL, 'no clip'], [RC, BLUE, 'ε = 0.2']].forEach(([R, col, lab]) => { ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); for (let k = 0; k <= n; k++) { k ? ctx.lineTo(X(k), Y(R[k])) : ctx.moveTo(X(k), Y(R[k])); } ctx.stroke(); ctx.restore();
       for (let k = 0; k <= n; k++) dot(X(k), Y(R[k]), 2.6, col); if (n >= K) txt(`${lab}: ${f2(R[K])}`, x1 + 2, Y(R[K]) - 10, { s: 13, c: col, al: 'right' }); });
     card([{ s: `no clip: 1 → ${f2(RN[1])} → … → ${f2(RN[K])}`, c: ROSEL }, { s: `ε = 0.2: 1 → ${f2(RC[1])}, then flat`, c: BLUE }], t - 6.5, { y: 300 }); } },
 { h: 'The same batch, with the brake', p: [`Same first batch, same ${K} passes, now clipped at 0.2. Score ${f2(SCC)}; ${EC.ch + EC.ca} of 1000 test walks still end, fewer than the blank policy’s ${E0.ch + E0.ca}. The brake limits how far the bad advice goes; it does not undo it. Average KL divergence, how different old and new chances are: ${f3(KLC)}, against ${f3(KLN)} unclipped.`, `Later batches correct it. The chart adds this clipped run, blue: ${f2(RUNC.sc[10])} after 10 updates, ${f2(RUNC.sc[40])} after 40.`],
   shot: [4, 12],
   draw(t) { const n = clamp(Math.floor((t - 1.5) * 5), 0, 40);
     upChart({ x0: 56, x1: 330, y0: 250, y1: 50, n, series: [{ v: RUNN.sc, col: ROSEL, label: f2(RUNN.sc[40]) }, { v: RUNC.sc, col: BLUE, label: f2(RUNC.sc[40]) }], title: `this run, ${K} passes per batch: score after each update` });
     card([{ s: 'rose: no clip', c: ROSEL }, { s: 'blue: clipped, ε = 0.2', c: BLUE }], t - .8, { y: 300, gap: .8 }); } },
 { h: 'Fifty runs', p: [`Now 50 runs with different dice, 40 updates each, averaged. Teal: the ordinary update, one pass per batch. Rose: ${K} passes, no clip. Blue: ${K} passes, clipped at 0.2.`, '{AVG}'],
   shot: [4, 13],
   draw(t) { const n = clamp(Math.floor((t - 1) * 6), 0, 40), a1 = avgAll('one', fn1), a2 = avgAll('e' + Infinity, fnE(Infinity)), a3 = avgAll('e0.2', fnE(.2));
     upChart({ x0: 56, x1: 330, y0: 250, y1: 50, n, series: [{ v: a1.v, col: GREEN, label: f2(a1.v[40]), dy: 9 }, { v: a2.v, col: ROSEL, label: f2(a2.v[40]) }, { v: a3.v, col: BLUE, label: f2(a3.v[40]), dy: -6 }], title: 'average score after each update (50 runs)' });
     card([{ s: `one pass: ${a1.stuck} of 50 runs stuck near 0`, c: GREEN }, { s: `${K} passes, no clip: ${a2.stuck} stuck`, c: ROSEL }, { s: `${K} passes, ε = 0.2: ${a3.stuck} stuck`, c: BLUE }], t - 7.5, { gap: .9 }); } },
 { h: 'Ten passes, one big step?', p: [`Each pass uses α = ${LR}, so ten unclipped passes can push about as far as one pass at α = 30. Rose: one pass at α = 30. Teal: ten passes at α = 0.3, a tenth of the push each.`, '{BIG}'],
   shot: [4, 13],
   draw(t) { const n = clamp(Math.floor((t - 1) * 6), 0, 40), a1 = avgAll('one30', fn30), a2 = avgAll('small', fnSmall), a3 = avgAll('e0.2', fnE(.2));
     upChart({ x0: 56, x1: 330, y0: 250, y1: 50, n, series: [{ v: a1.v, col: ROSEL, label: f2(a1.v[40]) }, { v: a2.v, col: GREEN, label: f2(a2.v[40]), dy: 9 }, { v: a3.v, col: BLUE, label: f2(a3.v[40]), dy: -6 }], title: 'average score after each update (50 runs)' });
     card([{ s: `one pass, α = 30: ${a1.stuck} of 50 stuck`, c: ROSEL }, { s: `${K} passes, α = 0.3: ${a2.stuck} stuck`, c: GREEN }, { s: `${K} passes, α = ${LR}, ε = 0.2: ${a3.stuck} stuck`, c: BLUE }], t - 7.5, { gap: .9 }); } },
 { h: 'Your turn', ctl: { label: 'clip ε', values: EPSV, def: 2, fmt: fmtE },
   p: [`Drag **ε**. Each setting reruns the same 50 runs: ${K} passes per batch, α = ${LR}. The line is the average score; the card counts runs stuck near 0.`, '{SLIDE}'],
   shot: [4, 10],
   draw(t, st) { const e = EPSV[st.ci ?? 2], a = avg('e' + e, fnE(e)), ref = avgAll('e0.2', fnE(.2));
     upChart({ x0: 56, x1: 330, y0: 250, y1: 50, series: [{ v: ref.v, col: '#b9cde3', w: 1.2 }, { v: a.v, col: BLUE, n: a.done ? 40 : 0, label: a.done ? f2(a.v[40]) : '' }], title: `${K} passes, ${e === Infinity ? 'no clip' : 'ε = ' + e}: average score (50 runs)` });
     txt('pale blue line: ε = 0.2, for comparison', 56, 278, { s: 12, c: C.soft, al: 'left' });
     card(a.done ? [e === Infinity ? 'no clip' : `ε = ${e}`, `after 10 updates ${f2(a.v[10])},  after 40 ${f2(a.v[40])}`, { s: `${a.stuck} of 50 runs stuck near 0`, c: a.stuck ? C.roseInk : GREEN }] : [`running … ${a.k} of 50`], 1, { gap: 0 }); } },
 { h: 'A brake, not a wall', p: [`The clip only stops a move from pushing itself further. Other moves in the same cell still shift its chance. After the clipped update, ${ROUT} of the batch’s ${B1.S.length} ratios lie outside 0.8 to 1.2, from ${f2(RMIN)} to ${f2(RMAX)}.`, 'So PPO is a soft limit, not a guarantee. Practical versions often also stop the passes early once the average KL passes a small threshold, such as 0.015.'],
   shot: [4, 10],
   draw(t) { const x0 = 50, x1 = 350, y0 = 250, y1 = 70, lo = 0, hi = 1.6, nb = 32, X = r => x0 + (r - lo) / (hi - lo) * (x1 - x0), cnt = new Array(nb).fill(0);
     RATIOS.forEach(r => { cnt[clamp(Math.floor((r - lo) / (hi - lo) * nb), 0, nb - 1)]++; }); const mx = Math.max(...cnt), u = ease((t - .8) / 2);
     box(X(.8), y1, X(1.2) - X(.8), y0 - y1, C.yellow, .4);
     cnt.forEach((c, i) => { const h = c / mx * (y0 - y1 - 10) * u, out = (i + .5) / nb * (hi - lo) + lo; box(X(lo + i / nb * (hi - lo)) + 1, y0 - h, (x1 - x0) / nb - 2, h, out > 1.2 || out < .8 ? ROSEL : BLUE, .75); });
     rline(x0, y0, x1, y0, 1400, { c: C.line, w: 1, once: true }); [0, .5, 1, 1.5].forEach(r => txt(String(r), X(r), y0 + 12, { s: 12, c: C.soft }));
     txt('ratios after the clipped update (one bar = how many moves)', x0 - 4, y1 - 14, { s: 13, c: C.soft, al: 'left' });
     card([`${ROUT} of ${B1.S.length} outside the band`, { s: 'the clip removes the push, not the change', c: C.soft }], t - 4, { y: 300 }); } },
 { h: 'Where it is used', p: ['Schulman and colleagues at OpenAI introduced PPO in 2017. Simple to code and forgiving to tune, it became a common default.', 'A scaled-up PPO trained **OpenAI Five**, which played Dota 2 from 2018 and beat the world champions in 2019. In 2022, InstructGPT used PPO to steer a language model toward answers people preferred, the reinforcement step of **RLHF**. There, the cell is the text so far and the move is the next word.'],
   shot: [3, 9],
   draw(t) { rline(20, 170, 380, 170, 1500, { c: C.line });
     alpha(clamp(t - .5), () => { rrect(50, 80, 110, 74, 1501, { fill: '#e8e0d4' }); for (let k = 0; k < 5; k++) dot(70 + k * 18, 104 + Math.sin(t * 2 + k) * 6, 4, C.blue); for (let k = 0; k < 5; k++) dot(70 + k * 18, 132 + Math.cos(t * 2 + k) * 6, 4, C.rose); txt('Dota 2, 5 vs 5', 105, 192, { s: 15 }); });
     alpha(clamp(t - 2), () => { rrect(230, 80, 130, 74, 1502, { fill: C.card }); txt('Q: …', 244, 100, { s: 14, al: 'left' }); txt('A: preferred ✓', 244, 124, { s: 14, al: 'left', c: GREEN }); txt('A: less liked ✗', 244, 144, { s: 14, al: 'left', c: C.roseInk }); txt('RLHF, 2022', 295, 192, { s: 15 }); });
     card(['2017: PPO (Schulman and colleagues, OpenAI)', '2018–2019: OpenAI Five, Dota 2', '2022: InstructGPT, RLHF with PPO'], t - 3.5, { gap: 1.1 }); } },
 { h: 'Recap', p: ['Reusing a batch for several passes squeezes more learning out of each walk, but unchecked passes can push the policy far past what the batch supports, and a wrecked policy may never recover.', 'PPO clips the **ratio**, new chance ÷ old chance, to 1 ± ε, so a move stops earning credit once it has moved enough.'],
   shot: [3, 9],
   draw(t) { maze({ t }); fans(probs(RUNC.T)); pipAt(SK, t);
     card(['ratio = new chance ÷ old chance', 'objective: lower of ratio × advantage, clipped or not', 'clip band: 1 − ε to 1 + ε, ε about 0.2', { s: 'several passes per batch, with a soft brake', c: C.goldInk }], t - .3, { gap: 1 }); } },
];
// texts quoting 50-run averages are built when their step is shown; earlier steps warm the averages up a little per frame
const lazyP = (h, fn) => { const s = STEPS.find(x => x.h === h), p0 = s.p; Object.defineProperty(s, 'p', { get: () => fn(p0) }); };
lazyP('Fifty runs', p0 => { const a1 = avgAll('one', fn1), a2 = avgAll('e' + Infinity, fnE(Infinity)), a3 = avgAll('e0.2', fnE(.2));
  return [p0[0], `No clip: ${a2.stuck} of 50 runs got stuck. One pass: ${f2(a1.v[10])} after 10 updates, ${f2(a1.v[40])} after 40. Clipped passes: ${f2(a3.v[10])} and ${f2(a3.v[40])}, ${a3.stuck ? a3.stuck : 'none'} stuck. Even counting the crashes, unclipped passes lead early. All compared per walk collected; ten passes cost ten times the computing.`]; });
lazyP('Ten passes, one big step?', p0 => { const a1 = avgAll('one30', fn30), a2 = avgAll('small', fnSmall), a3 = avgAll('e0.2', fnE(.2)), a4 = avgAll('one', fn1);
  return [p0[0], `One pass at α = 30: ${a1.stuck} of 50 stuck, the too-big step again. Ten passes at α = 0.3 are as safe and as slow as one ordinary pass: ${f2(a2.v[10])} against ${f2(a4.v[10])} after 10 updates. The clip is what lets ten full passes through: ${f2(a3.v[10])}, none stuck.`]; });
lazyP('Your turn', p0 => { const a = avgAll('e' + Infinity, fnE(Infinity));
  return [p0[0], `A tight band is safe but learns slowly. A wide band lets more through. With no clip, ${a.stuck} of 50 runs get stuck.`]; });
const WARM = [['one', fn1], ['e' + Infinity, fnE(Infinity)], ['e0.2', fnE(.2)], ['one30', fn30], ['small', fnSmall]];
function warm() { for (const [id, fn] of WARM) { const a = avg(id, fn); if (!a.done) return; } }
['Stuck for good', 'Clip the ratio', 'When the advantage is negative', 'Flat means stop', 'The same batch, with the brake'].forEach(h => { const s = STEPS.find(x => x.h === h), d = s.draw; s.draw = (t, st) => { d(t, st); warm(); }; });

RLStory.register('ppo', { title: 'PPO', track: 'Policy Gradients', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { S0, E0, B1, NCAT, NCH, NSTOP, STOPS, STOPPOS, STOPAVG, VMIN, SC1, SCN, SCC, EN, EC, RN, RC, ROUT, RMAX, RMIN, KLN, KLC, RUNN, RUNC, AFTER, avgAll, fnE, fn1, EPSV } });
})();
