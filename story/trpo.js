// TRPO as a story: Pip's policy (chances per cell, softmax of preferences) is doing fine; one update with a step 33 times too big
// wrecks it (the start cell learns to bump into the wall), and it takes many updates to climb back. Fix: limit how much the
// chances may change per update, measured by KL divergence (old ‖ new, the TRPO paper's direction), and halve the step until it fits.
// World: the shared Pip maze, dry floor (pg_world.md, World 2). Every number comes from the seeded simulation below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({});
const { cx, cy, maze, pip } = G;

// ───── simulation: Pip's maze, tabular softmax policy, batches of 10 walks, advantage = return that followed − exact critic ─────
const GAM = 0.9, CAP = 100, NB = 10, MV = [[0, -1], [1, 0], [0, 1], [-1, 0]], MN = ['up', 'right', 'down', 'left'];
const key = (c, r) => r * 5 + c, SK = key(0, 3);
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);
const NXT = new Int8Array(80), RW = new Int8Array(80);
for (let k = 0; k < 20; k++) { const c = k % 5, r = (k / 5) | 0; for (let m = 0; m < 4; m++) { const [nc, nr] = G.step(c, r, m); NXT[k * 4 + m] = key(nc, nr); RW[k * 4 + m] = G.term(nc, nr); } }
const IDX = new Array(20).fill(-1); CELLS.forEach(([c, r], i) => { IDX[key(c, r)] = i; });
// softmax per cell: preferences T (80 numbers) -> chances P
function probs(T) { const P = new Float64Array(80); for (let k = 0; k < 20; k++) { const o = k * 4, mx = Math.max(T[o], T[o + 1], T[o + 2], T[o + 3]); let s = 0;
  for (let m = 0; m < 4; m++) { P[o + m] = Math.exp(T[o + m] - mx); s += P[o + m]; } for (let m = 0; m < 4; m++) P[o + m] /= s; } return P; }
// exact values of a policy (the rulebook: solve the 16 Bellman equations). Used for the critic and for the score; Pip cannot do this.
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
// direction of the update: for every move in the batch, advantage × (1 − chance) for the move made, −advantage × chance for the others; averaged per walk
function grad(T, S) { const P = probs(T), g = new Float64Array(80);
  for (const s of S) { const o = s.k * 4, r = P[o + s.m] / s.p; for (let b = 0; b < 4; b++) g[o + b] += s.A * r * ((b === s.m ? 1 : 0) - P[o + b]) / NB; } return g; }
const addT = (T, g, a) => T.map((v, i) => v + a * g[i]);
// KL(old ‖ new) for one cell: average over old Pip's own choices of log(old chance ÷ new chance); then averaged over every cell visit in the batch
function klCell(Pn, Po, k) { let s = 0; for (let m = 0; m < 4; m++) if (Po[k * 4 + m] > 0) s += Po[k * 4 + m] * Math.log(Po[k * 4 + m] / Pn[k * 4 + m]); return s; }
function klAvg(Pn, Po, S) { let s = 0; for (const x of S) s += klCell(Pn, Po, x.k); return s / S.length; }
function surL(T, S) { const P = probs(T); let v = 0; for (const s of S) v += P[s.k * 4 + s.m] / s.p * s.A / NB; return v; }
// toy TRPO step: start from the big step α = 100, halve until the average KL is within the limit and the batch still predicts a gain
// d = Infinity: no KL check, only the gain check (to see which part does the work)
function trpoStep(T, B, d, b0 = 100, wantScore = false) { const g = grad(T, B.S), L0 = surL(T, B.S), tries = []; let beta = b0;
  for (let i = 0; i < 14; i++) { const Tn = addT(T, g, beta), kl = d === Infinity ? 0 : klAvg(probs(Tn), B.P, B.S), dL = surL(Tn, B.S) - L0; tries.push({ beta, kl, dL, sc: wantScore ? score(Tn) : null });
    if (kl <= d && dL > 0) return { T: Tn, beta, tries }; beta /= 2; }
  return { T, beta: 0, tries }; }
function runPlain(seed, a, T0, iters = 40, R = rng(5000 + seed)) { let T = T0 ? Float64Array.from(T0) : new Float64Array(80); const sc = [score(T)], lens = [];
  for (let it = 0; it < iters; it++) { const B = batch(T, R); if (!it) lens.push(...B.W.map(w => w.length)); T = addT(T, grad(T, B.S), a); sc.push(score(T)); } return { sc, lens, T }; }
function runTRPO(seed, d, iters = 40) { const R = rng(5000 + seed); let T = new Float64Array(80); const sc = [score(T)], betas = [];
  for (let it = 0; it < iters; it++) { const B = batch(T, R), st = trpoStep(T, B, d); T = st.T; betas.push(st.beta); sc.push(score(T)); } return { sc, betas }; }

// the story run (run 37): 10 ordinary updates with α = 3, then the batch of update 11
const A0 = 3, ABIG = 100, LIM = 0.01;
const ST = (() => { const R = rng(5000 + 37); let T = new Float64Array(80); const hist = [score(T)];
  for (let it = 0; it < 10; it++) { const B = batch(T, R); T = addT(T, grad(T, B.S), A0); hist.push(score(T)); }
  const B = batch(T, R); return { T0: T, hist, B, g: grad(T, B.S) }; })();
const T0 = ST.T0, B11 = ST.B, P0 = B11.P, T3 = addT(T0, ST.g, A0), T100 = addT(T0, ST.g, ABIG), P3 = probs(T3), P100 = probs(T100);
const S0 = score(T0), S3 = score(T3), S100 = score(T100), KL3 = klAvg(P3, P0, B11.S), KL100 = klAvg(P100, P0, B11.S);
const START = B11.S.filter(s => s.k === SK), BYM = [0, 1, 2, 3].map(m => START.filter(s => s.m === m)), SUMA = BYM.map(l => l.reduce((a, s) => a + s.A, 0));
const UPCAT = BYM[0].filter(s => s.A < -.3).map(s => s.A);
const CATW = B11.W.filter(w => w[w.length - 1].rew < 0).length, CHW = B11.W.filter(w => w[w.length - 1].rew > 0).length;
// after the update: both policies keep learning with the ordinary α = 3 for 40 more updates (same dice, rng(6000))
const C3 = runPlain(0, A0, T3, 40, rng(6000)), C100 = runPlain(0, A0, T100, 40, rng(6000));
const CAPPED = C100.lens.filter(l => l >= CAP).length, CAPPED3 = C3.lens.filter(l => l >= CAP).length;
const FIRSTUP = C100.sc.findIndex(v => v > .17), LOW = C100.sc.slice(1).findIndex(v => v >= .1);
const BUMPWALK = (() => { const R = rng(6000), P = probs(T100); for (let i = 0; i < 10; i++) { const w = walk(P, R); if (w.length >= CAP) return w; } return null; })();
// score along the same direction for step sizes 0 … 100
const SSX = Array.from({ length: 41 }, (_, i) => i * 2.5), SSY = SSX.map(a => score(addT(T0, ST.g, a)));
const PEAK = SSX[SSY.indexOf(Math.max(...SSY))];
// the halving on the same batch
const HALF = trpoStep(T0, B11, LIM, 100, true), PT = probs(HALF.T), ST_T = score(HALF.T);
// a TRPO run from scratch: which step sizes were accepted
const RUN0 = runTRPO(0, LIM);
// preference example: same +2 change, very different behaviour
const sm = (a) => Math.exp(a) / (Math.exp(a) + 3);
// averaged curves over 50 runs (runs 0–49), computed a few runs per frame and memoised
const NRUN = 50, AV = {};
let GFULL = 0;   // runs whose first update (no KL check) took the full α = 100
function avg(id, fn) { let a = AV[id]; if (!a) a = AV[id] = { k: 0, sum: new Float64Array(41), stuck: 0, min: 9 };
  const t0 = Date.now(); while (a.k < NRUN && Date.now() - t0 < 14) { const sc = fn(a.k); sc.forEach((v, i) => { a.sum[i] += v; }); if (sc[40] < .1) a.stuck++; a.min = Math.min(a.min, sc[40]); a.k++; }
  return { k: a.k, done: a.k >= NRUN, stuck: a.stuck, min: a.min, v: Array.from(a.sum, v => v / Math.max(1, a.k)) }; }
const avgAll = (id, fn) => { let a; do a = avg(id, fn); while (!a.done); return a; };
const LIMS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
const fnT = d => s => runTRPO(s, d).sc, fnG = s => { const r = runTRPO(s, Infinity); if (r.betas[0] === ABIG) GFULL++; return r.sc; }, fnP = a => s => runPlain(s, a).sc;

// ───── formatting ─────
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−');
const f3 = v => (Math.abs(v) < .0005 ? 0 : v).toFixed(3).replace('-', '−');
const fa = b => String(+b.toFixed(2));
const pc = (P, m, k = SK) => f2(P[k * 4 + m]);
const LUP = P0[SK * 4] * Math.log(P0[SK * 4] / P100[SK * 4]);   // up's share of the start-cell KL

// ───── drawing helpers ─────
const ZERO = new Float64Array(80), PU = probs(ZERO);
// arrow fans: in each cell four arrows from the centre, length ∝ chance (1 reaches the cell edge); the biggest in dark gold
function fans(P, o = {}) {
  const Gx = o.G || G, a = o.a ?? 1, s = Gx.cs / 64;
  for (const [c, r] of CELLS) { const k = key(c, r); if (o.only && !o.only(k)) continue; const aa = typeof a === 'function' ? a(k) : a; if (aa <= 0) continue;
    const ps = [0, 1, 2, 3].map(m => P[k * 4 + m]), mx = Math.max(...ps), x = Gx.cx(c), y = Gx.cy(r) + 2 * s;
    for (let m = 0; m < 4; m++) { const p = ps[m]; if (p < .015) continue; const [dx, dy] = MV[m], big = p === mx, r0 = 4 * s, len = r0 + p * 25 * s;
      alpha(aa, () => arrow(x + dx * r0, y + dy * r0, x + dx * len, y + dy * len, 300 + k * 4 + m, { c: big ? C.goldInk : '#a89c8f', w: (big ? 1.7 : 1.1) * Math.max(.8, s), head: (4 + 3 * p) * s })); } }
}
const mixP = (A, B, u) => A.map((v, i) => lerp(v, B[i], u));
function dim(a = .7, y0 = 0, y1 = 285) { box(0, y0, W, y1 - y0, C.bg, a); }
function pipAt(k, t, face = 1) { const c = k % 5, r = (k / 5) | 0; mouse(cx(c) - 18, cy(r) + 18, face, .55, t); }
const mouse = L.mouse;
function hiCell(k, col = C.teal, a = .22, Gx = G) { const c = k % 5, r = (k / 5) | 0; box(Gx.ox + c * Gx.cs + 3, Gx.oy + r * Gx.cs + 3, Gx.cs - 6, Gx.cs - 6, col, a); }
function trailW(w, a = .45, n) { const pts = [SK].concat(w.slice(0, n ?? w.length).map(s => s.nk)), jx = i => ((i * 7) % 11) - 5, jy = i => ((i * 5) % 11) - 5;
  ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = C.teal; ctx.lineWidth = 2.4; ctx.lineJoin = 'round'; ctx.beginPath();
  pts.forEach((k, i) => { const x = cx(k % 5) + jx(i), y = cy((k / 5) | 0) + jy(i); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.restore(); }
// score chart over updates. o = { x0, x1, y0 (bottom), y1 (top), n: updates shown (0..40), series: [{ v, col, w, label }], title }
function upChart(o) {
  const lo = -.05, hi = .6, N = 40, X = i => o.x0 + i / N * (o.x1 - o.x0), Y = v => o.y0 - (v - lo) / (hi - lo) * (o.y0 - o.y1);
  rline(o.x0 - 4, Y(0), o.x1 + 4, Y(0), 960, { c: C.line, w: 1, once: true }); rline(o.x0 - 4, o.y0, o.x0 - 4, o.y1, 961, { c: C.line, w: 1, once: true });
  txt('0', o.x0 - 13, Y(0), { s: 12, c: C.soft }); txt('0.5', o.x0 - 17, Y(.5), { s: 12, c: C.soft });
  txt(o.xl0 || 'update 0', o.x0, o.y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(o.xl1 || 'update 40', o.x1, o.y0 + 10, { s: 12, c: C.soft, al: 'right' });
  ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.grey; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(o.x0, Y(.531)); ctx.lineTo(o.x1, Y(.531)); ctx.stroke(); ctx.restore();
  txt('best possible 0.53', o.x0 + 4, Y(.531) - 9, { s: 12, c: C.soft, al: 'left' });
  if (o.title) txt(o.title, o.x0 - 8, o.y1 - 10, { s: 13, c: C.soft, al: 'left' });
  (o.series || []).forEach(s => { const n = Math.min(s.n ?? o.n ?? N, s.v.length - 1); ctx.save(); ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 1.8; ctx.lineJoin = 'round'; ctx.beginPath();
    for (let i = 0; i <= n; i++) { const x = X(i), y = Y(s.v[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore();
    if (s.label && n >= 40) txt(s.label, o.x1 + 3, Y(s.v[40]) + (s.dy || 0), { s: 12, c: s.col, al: 'left' }); });
  return { X, Y };
}
// a big single cell with its four arrows and chances (for the KL step)
function bigCell(x, y, P, k, title, col, a = 1) {
  alpha(a, () => { rrect(x - 50, y - 50, 100, 100, 820 + x, { c: C.ink, w: 1.4 }); txt(title, x, y - 88, { s: 15, c: col });
    for (let m = 0; m < 4; m++) { const p = P[k * 4 + m], [dx, dy] = MV[m]; if (p >= .015) arrow(x + dx * 6, y + dy * 6, x + dx * (6 + p * 38), y + dy * (6 + p * 38), 840 + m + x, { c: col, w: 1.6, head: 6 });
      txt(f2(p), x + dx * 70, y + dy * 62 + (dy ? 0 : -9), { s: 14, c: col }); } });
}

const GREEN = '#3f7f7c', BLUE = '#4f7fb0', ROSEL = '#c0605f';

const STEPS = [
 { h: 'Pip’s policy, as chances', p: ['Same maze, dry floor: +1 onto the cheese, −1 onto the cat, both end the walk, γ (gamma) 0.9. Walks still going after 100 steps stop and score 0.', `Pip holds four chances per cell, the arrows. After 10 ordinary updates from blank, its **score**, the expected return from the start, from the rulebook, is ${f2(S0)}. The best: 0.53, seven steps straight to the cheese.`],
   shot: [3, 9],
   draw(t) { maze({ t }); fans(P0, { a: k => clamp((t - .4 - IDX[k] * .06) * 3) }); pipAt(SK, t);
     if (t > 3.5) card(['arrow length = chance of that move', `score now ${f2(S0)}   ·   best possible 0.53`, { s: 'scores use the rulebook: we can, Pip cannot', c: C.soft }], t - 3.5, { gap: 1.2 }); } },
 { h: 'One update', p: [`Each update, Pip walks ${NB} times with its current chances: a **batch**, where REINFORCE used one walk. For each move it asks: did things go better than usual from here?`, 'That is the move’s **advantage**: the return after the move minus the critic’s value of the cell, the simplest grade. The critic here is exact, so we watch the actor alone.'],
   shot: [5, 12],
   draw(t) { const D = .9, i = clamp(Math.floor((t - .3) / D), 0, NB - 1), w = B11.W[i]; maze({ t }); fans(P0, { a: .35 });
     for (let j = 0; j < i; j++) trailW(B11.W[j], .1); const done = (t - .3) % D > D * .8 || t > .3 + NB * D; trailW(w, .75, Math.ceil(clamp(((t - .3) % D) / (D * .8)) * w.length) + (t > .3 + NB * D ? w.length : 0));
     const end = w[w.length - 1].rew; if (done) txt(`walk ${i + 1} of ${NB}:  ${end > 0 ? 'cheese' : end < 0 ? 'cat' : 'stopped'}, ${w.length} steps`, W / 2, 298, { s: 16, c: end > 0 ? C.goldInk : end < 0 ? C.roseInk : C.soft });
     if (t > 3) card(['advantage = return after the move', '                    − critic’s value of the cell', { s: '> 0: went better than usual from here', c: C.goldInk }], t - 3, { y: 312, gap: 1, lh: 21, fs: 16 }); } },
 { h: 'What the batch says at the start', p: [`At the start Pip moved up ${BYM[0].length} times. Twice, that walk later met the cat: advantages ${f2(UPCAT[0])} and ${f2(UPCAT[1])}. Down and left both bump a wall, wasting a step, yet after ${BYM[2].filter(s => s.A > 0).length} of ${BYM[2].length} down bumps the walk went better than usual by luck.`, `So: less up, more of two useless moves. Summed: up ${f2(SUMA[0])}, down ${f2(SUMA[2])}, left ${f2(SUMA[3])}. Right was never tried.`],
   shot: [4, 11],
   draw(t) { maze({ t }); hiCell(SK); fans(P0, { a: k => k === SK ? 1 : .3 });
     const x0 = 40, y0 = 300; [0, 2, 3].forEach((m, row) => { const l = BYM[m], u = clamp(t - .6 - row * 1.6); if (u <= 0) return;
       txt(`${MN[m]}:`, x0, y0 + row * 30, { s: 16, al: 'left', a: u });
       l.forEach((s, j) => { const uu = clamp((t - .6 - row * 1.6 - j * .12) * 4); if (uu <= 0) return; const v = s.A, h = Math.min(13, Math.abs(v) * 16) + 1.5, x = x0 + 52 + j * 16;
         box(x, v >= 0 ? y0 + row * 30 - h : y0 + row * 30, 10, h, v >= 0 ? C.gold : C.rose, uu); });
       if (t > 1.6 + row * 1.6) txt(`sum ${f2(SUMA[m])}`, 360, y0 + row * 30, { s: 15, al: 'right', c: SUMA[m] >= 0 ? C.goldInk : C.roseInk, a: clamp(t - 1.6 - row * 1.6) }); });
     caption('one bar per time Pip made that move at the start: its advantage', clamp(t - 1)); } },
 { h: 'A normal step', p: [`For each batch move, the move made gains α (alpha) × advantage × (1 − its chance) in preference; each other move loses α × advantage × its chance. Divide by the ${NB} walks. Usual step: α = ${A0}.`, `At the start, up drops from ${pc(P0, 0)} to ${pc(P3, 0)}, down rises from ${pc(P0, 2)} to ${pc(P3, 2)}, and other batch moves make the top row surer of right. Score: ${f2(S0)} to ${f2(S3)}.`],
   shot: [3, 9],
   draw(t) { const u = ease((t - 1.5) / 2); maze({ t }); hiCell(SK); fans(mixP(P0, P3, u)); pipAt(SK, t);
     card([`step size α = ${A0}`, `start: up ${pc(P0, 0)} → ${pc(P3, 0)},  down ${pc(P0, 2)} → ${pc(P3, 2)}`, { s: `score ${f2(S0)} → ${f2(S3)}`, c: C.goldInk }], t - 2.5, { gap: 1.3 }); } },
 { h: 'Why not a bigger step?', p: [`If a small step helps, a bigger one should help more and learn faster. Same batch, same direction, but α = ${ABIG}: about 33 times bigger.`, `At the start, up falls from ${pc(P0, 0)} to ${pc(P100, 0)} and down climbs to ${pc(P100, 2)}. Pip now mostly walks into the wall. The score crashes from ${f2(S0)} to ${f2(S100)}.`],
   shot: [4, 11],
   draw(t) { const u = ease((t - 1.5) / 2.5); maze({ t }); hiCell(SK, u > .5 ? C.rose : C.teal, .25); fans(mixP(P0, P100, u)); pipAt(SK, t);
     card([`step size α = ${ABIG}`, `start: up ${pc(P0, 0)} → ${pc(P100, 0)},  down ${pc(P0, 2)} → ${pc(P100, 2)}`, { s: `score ${f2(S0)} → ${f2(S100)}`, c: C.roseInk }], t - 3, { gap: 1.3 }); } },
 { h: 'Bump, bump, bump', p: [`With the new chances, Pip bumps into the bottom wall again and again. In its next batch, ${CAPPED} of ${NB} walks hit the 100-step limit and score 0. After the normal step, ${CAPPED3 ? CAPPED3 : 'none'} did.`, 'The wrecked policy now collects poor walks, and poor walks are what it must learn from next.'],
   shot: [5, 11],
   draw(t) { maze({ t }); hiCell(SK, C.rose, .2); fans(P100, { a: .4 }); const w = BUMPWALK, n = Math.min(w.length, Math.floor(clamp(t - .5, 0, 99) * 14));
     const s = w[Math.max(0, n - 1)], bump = n > 0 && s.nk === s.k, k = n ? s.nk : SK;
     L.mouse(cx(k % 5) + (bump ? Math.sin(t * 30) * 2 : 0), cy((k / 5) | 0) + 5, 1, .85, t); if (bump && s.m === 2) txt('bump!', cx(0) + 4, cy(3) - 22, { s: 14, c: C.roseInk, a: .5 + .5 * Math.sin(t * 9) });
     trailW(w, .5, n); txt(`step ${n} of ${CAP}`, W / 2, 300, { s: 17 });
     if (n >= w.length) card([`next batch: ${CAPPED} of ${NB} walks stopped at 100 steps`, { s: 'stopped walks score 0', c: C.soft }], t - .5 - w.length / 14, { y: 318 }); } },
 { h: 'Climbing back is slow', p: ['Both policies now keep learning with the usual α = 3. The chart shows the score after each update: teal continues from the normal step, rose from the big one. The dashed line is the best possible, 0.53.', `The teal line reaches ${f2(C3.sc[40])} after 40 updates. The rose line stays below 0.1 for ${LOW} more updates, and is still only at ${f2(C100.sc[40])} after 40.`],
   shot: [3, 12],
   draw(t) { const n = clamp(Math.floor((t - 1.5) * 5), 0, 40);      upChart({ x0: 56, x1: 330, y0: 250, y1: 50, n, xl0: 'right after the step', xl1: '40 updates later', series: [{ v: C3.sc, col: GREEN, label: f2(C3.sc[40]), dy: 8 }, { v: C100.sc, col: ROSEL, label: f2(C100.sc[40]) }], title: 'score after each update (both with α = 3)' });
     card([{ s: 'teal: after the normal step', c: GREEN }, { s: 'rose: after the step 33 times bigger', c: ROSEL }, { s: 'Pip learns from its own walks: bad walks, slow news', c: C.soft }], t - 1, { gap: .8 }); } },
 { h: 'The batch is a local map', p: ['Why did the batch mislead? Its advice is a slope: which way to nudge each preference, measured at the old policy. A slope is reliable only near where it was measured, and all its walks came from the old policy.', `The chart steps along that direction by different sizes, scored exactly. Small steps help, up to about α = ${PEAK}; further out the same advice hurts: at ${ABIG} the score is ${f2(S100)}.`],
   shot: [4, 12],
   draw(t) { const x0 = 60, x1 = 350, y0 = 250, y1 = 60, X = a => x0 + a / 100 * (x1 - x0), Y = v => y0 - v / .45 * (y0 - y1), n = clamp(Math.floor((t - 1.5) * 6), 0, SSX.length);
     rline(x0 - 4, y0, x1 + 4, y0, 970, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1, 971, { c: C.line, w: 1, once: true });
     txt('0', x0 - 14, y0, { s: 12, c: C.soft }); txt('0.4', x0 - 18, Y(.4), { s: 12, c: C.soft }); txt('score', x0 - 8, y1 - 14, { s: 13, c: C.soft, al: 'left' });
     [0, 25, 50, 75, 100].forEach(a => txt(String(a), X(a), y0 + 12, { s: 12, c: C.soft })); txt('step size α', (x0 + x1) / 2, y0 + 28, { s: 13, c: C.soft });
     ctx.save(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8; ctx.beginPath(); for (let i = 0; i < n; i++) { const x = X(SSX[i]), y = Y(SSY[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore();
     if (n > 0) { dot(X(0), Y(S0), 4, C.ink); txt(`α = 0 (no change): ${f2(S0)}`, X(0) + 6, Y(S0) + 16, { s: 13, c: C.ink, al: 'left' }); }
     if (n > 2) { dot(X(A0), Y(S3), 4, GREEN); txt(`α = ${A0}: ${f2(S3)}`, X(A0) + 8, Y(S3) - 14, { s: 13, c: GREEN, al: 'left' }); }
     if (n >= SSX.length) { dot(X(ABIG), Y(S100), 4, ROSEL); txt(`α = ${ABIG}: ${f2(S100)}`, X(ABIG) - 4, Y(S100) - 14, { s: 13, c: ROSEL, al: 'right' }); }
     if (t > 9) card(['the advice holds near the old policy', { s: 'trust it only within a small region', c: C.goldInk }], t - 9, { y: 300 }); } },
 { h: 'How far is too far?', p: [`So limit how far one update may move. But measure it in behaviour, not in preference numbers. In a cell with all preferences 0, adding 2 to up lifts its chance from 0.25 to ${f2(sm(2))}. If up’s preference is already 10, the same +2 leaves its chance at ${f2(sm(12))}.`, 'What matters is how different the chances become.'],
   shot: [4, 11],
   draw(t) { const GA = L.gridWorld({ cs: 90, ox: 40, oy: 70 }), xA = 110, xB = 290, y = 150, u = ease((t - 2) / 1.5);
     const cell = (x, base, lab) => { rrect(x - 55, y - 55, 110, 110, 870 + x, { c: C.ink, w: 1.4 }); const e = [Math.exp(base + 2 * u), 1, 1, 1], s = e[0] + 3;
       for (let m = 0; m < 4; m++) { const p = e[m] / s, [dx, dy] = MV[m]; if (p >= .01) arrow(x + dx * 6, y + dy * 6, x + dx * (6 + p * 42), y + dy * (6 + p * 42), 880 + m + x, { c: m ? '#a89c8f' : C.goldInk, w: 1.6, head: 6 }); }
       txt(lab, x, y - 72, { s: 15 }); txt(`up: ${f2(e[0] / s)}`, x, y + 74, { s: 16, c: C.goldInk }); };
     cell(xA, 0, 'preferences 0, 0, 0, 0'); cell(xB, 10, 'up already at 10');
     if (t > 2) txt('+2 to up', W / 2, y, { s: 15, c: C.soft, a: clamp(t - 2) });
     if (t > 5) card([`same +2:  0.25 → ${f2(sm(2))}   vs   ${f2(sm(10))} → ${f2(sm(12))}`, { s: 'measure the change in chances, not in numbers', c: C.goldInk }], t - 5, { gap: 1.2 }); } },
 { h: 'Surprise', p: ['First, a way to score one choice. The **surprise** at a choice is minus the log of its chance. A sure thing, chance 1, surprises 0. Chance one half gives 0.69, one quarter 1.39, one tenth 2.30.', 'Rarer choices surprise more, without limit as the chance shrinks to 0. The curve shows surprise against chance.'],
   shot: [4, 10],
   draw(t) { const x0 = 70, x1 = 350, y0 = 250, y1 = 50, X = p => x0 + p * (x1 - x0), Y = s => y0 - s / 4 * (y0 - y1);
     rline(x0, y0, x1 + 4, y0, 975, { c: C.line, w: 1, once: true }); rline(x0, y0, x0, y1, 976, { c: C.line, w: 1, once: true });
     txt('chance', x1, y0 + 26, { s: 13, c: C.soft, al: 'right' }); txt('surprise', x0 - 6, y1 - 12, { s: 13, c: C.soft, al: 'left' });
     [0, .5, 1].forEach(p => txt(String(p), X(p), y0 + 12, { s: 12, c: C.soft })); [0, 2, 4].forEach(s => txt(String(s), x0 - 12, Y(s), { s: 12, c: C.soft }));
     const u = clamp((t - .8) / 2.5); ctx.save(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.8; ctx.beginPath(); let first = true;
     for (let i = 100; i >= 1; i--) { const p = i / 100; if (p < 1 - u) break; const s = -Math.log(p); if (s > 4) break; const x = X(p), y = Y(s); first ? ctx.moveTo(x, y) : ctx.lineTo(x, y); first = false; } ctx.stroke(); ctx.restore();
     [[1, '1 → 0'], [.5, '0.5 → 0.69'], [.25, '0.25 → 1.39'], [.1, '0.1 → 2.30']].forEach(([p, lab], i) => { const a = clamp(t - 3.5 - i * .9); if (a <= 0) return; dot(X(p), Y(-Math.log(p)), 4, C.orange, a); txt(lab, X(p) + 8, Y(-Math.log(p)) - 12, { s: 13, c: '#c4581f', al: 'left', a }); });
     card(['surprise = − log(chance)', { s: 'rare choices surprise more', c: C.soft }], t - 7, { y: 300 }); } },
 { h: 'KL divergence', p: ['The batch came from old Pip. How much less likely is new Pip to make old Pip’s moves? Per move, log of old chance over new; averaged by old Pip’s chances, that is the **KL divergence**, 0 for identical policies.', `At the start, new Pip almost never goes up, which old Pip did ${pc(P0, 0)} of the time: that term alone is ${f2(LUP)}. The others add ${f2(klCell(P100, P0, SK) - LUP)}; single terms can be negative, never the total. KL ${f2(klCell(P100, P0, SK))}; after the normal step, ${f3(klCell(P3, P0, SK))}.`],
   shot: [4, 12],
   draw(t) { bigCell(110, 130, P0, SK, 'old Pip, start cell', C.ink, clamp(t - .3)); bigCell(290, 130, P100, SK, 'new Pip (α = 100)', ROSEL, clamp(t - 1.2));
     if (t > 3) card([`up: old ${f3(P0[SK * 4])}, new ${P100[SK * 4].toFixed(4)}: log(old ÷ new) = ${f2(Math.log(P0[SK * 4] / P100[SK * 4]))}`, `KL = sum of old chance × log(old ÷ new)`, { s: 'log(old ÷ new) = new Pip’s surprise − old Pip’s', c: C.soft }, { s: `start cell:  big step ${f2(klCell(P100, P0, SK))},  normal step ${f3(klCell(P3, P0, SK))}`, c: C.goldInk }], t - 3, { y: 256, gap: 1.4, fs: 16, lh: 22 }); } },
 { h: 'The trust region rule', p: [`**TRPO**, Trust Region Policy Optimization (Schulman and colleagues, 2015), turns this into a rule: improve as the batch suggests, but keep the KL under a limit, commonly ${LIM}. The KL is computed in each visited cell, then averaged over visits.`, `Here the big step’s average KL is ${f2(KL100)}. Even the normal step, ${f3(KL3)}, is over, though harmless: ${LIM} is a large-network habit, stricter than this maze needs.`],
   shot: [4, 11],
   draw(t) { const x0 = 70, x1 = 340, y = [120, 170], X = v => x0 + Math.min(1, v / .45) * (x1 - x0);
     txt('average KL over the batch', x0, 70, { s: 15, al: 'left', c: C.soft }); const lx = X(LIM);
     [[`α = ${ABIG}`, KL100, ROSEL], [`α = ${A0}`, KL3, GREEN]].forEach(([lab, v, col], i) => { const u = ease((t - .8 - i * 1.2) / 1.2); if (u <= 0) return;
       txt(lab, x0 - 6, y[i], { s: 15, al: 'right' }); box(x0, y[i] - 8, (X(v) - x0) * u, 16, col, .8); txt(i ? f3(v) : f2(v), x0 + (X(v) - x0) * u + 6, y[i], { s: 14, al: 'left', c: col }); });
     rline(lx, 95, lx, 195, 977, { c: C.ink, w: 1.6 }); txt(`limit ${LIM}`, lx + 4, 210, { s: 14, al: 'left' });
     card(['find a better policy, as the batch suggests', { s: `while  average KL ≤ ${LIM}`, c: C.goldInk }, { s: 'trust region: where the batch can be believed', c: C.soft }], t - 3.2, { gap: 1.2 }); } },
 { h: 'Halve until it fits', p: [`Real TRPO computes a step that just fits, with heavier maths. Our toy halves the big step until the KL fits and the batch still predicts a gain: the sum of each move’s advantage times new chance ÷ old chance must beat the old policy’s.`, `Here: six halvings, from ${ABIG} to ${fa(HALF.beta)}, KL ${f3(HALF.tries[HALF.tries.length - 1].kl)}. Score ${f2(S0)} → ${f2(ST_T)}, a bit below the plain α = ${A0} step: safety costs some speed.`],
   shot: [4, 13],
   draw(t) { const rows = HALF.tries, x0 = 44, y0 = 40, dy = 32, X = v => 200 + Math.min(1, v / .45) * 105, lx = X(LIM);
     txt('step α', x0, y0 - 18, { s: 13, c: C.soft, al: 'left' }); txt('average KL', 200, y0 - 18, { s: 13, c: C.soft, al: 'left' });
     rows.forEach((r, i) => { const u = clamp((t - .6 - i * 1.2) * 2); if (u <= 0) return; const ok = r.kl <= LIM && r.dL > 0, y = y0 + i * dy;
       txt(`α = ${fa(r.beta)}`, x0, y, { s: 16, al: 'left', a: u }); box(200, y - 7, (X(r.kl) - 200) * u, 14, ok ? GREEN : ROSEL, .75);
       if (r.kl > .45) txt('»', 307, y, { s: 14, c: C.roseInk, a: u }); txt(`${f3(r.kl)} ${ok ? '✓' : '✗'}`, Math.max(X(r.kl), lx) + (r.kl > .45 ? 14 : 8), y, { s: 14, al: 'left', c: ok ? GREEN : C.roseInk, a: u }); });
     rline(lx, y0 - 10, lx, y0 + rows.length * dy - 16, 978, { c: C.ink, w: 1.4 }); txt(`limit ${LIM}`, lx, y0 + rows.length * dy - 4, { s: 12, c: C.soft }); txt('» bar cut off', 350, y0 + rows.length * dy - 4, { s: 12, c: C.soft, al: 'right' });
     const tt = .6 + rows.length * 1.2; if (t > tt) card([`accepted: α = ${fa(HALF.beta)}`, `start: up ${pc(P0, 0)} → ${pc(PT, 0)},  down ${pc(P0, 2)} → ${pc(PT, 2)}`, { s: `score ${f2(S0)} → ${f2(ST_T)}`, c: C.goldInk }], t - tt, { gap: 1, y: 292 }); } },
 { h: 'Forty updates, fifty runs', p: [`From a blank policy, 40 updates, averaged over 50 runs with different dice. Rose: plain α = ${ABIG}. Teal: plain α = ${A0}. Blue: TRPO, limit ${LIM}, halving from ${ABIG}.`, '{STUCK}'],
   shot: [4, 13],
   draw(t) { const n = clamp(Math.floor((t - 1) * 6), 0, 40), a1 = avgAll('p100', fnP(ABIG)), a2 = avgAll('p3', fnP(A0)), a3 = avgAll('t' + LIM, fnT(LIM));
     upChart({ x0: 56, x1: 330, y0: 250, y1: 50, n, series: [{ v: a1.v, col: ROSEL, label: f2(a1.v[40]) }, { v: a2.v, col: GREEN, label: f2(a2.v[40]), dy: 9 }, { v: a3.v, col: BLUE, label: f2(a3.v[40]), dy: -7 }], title: 'average score after each update (50 runs)' });
     card([{ s: `α = ${ABIG}: ${a1.stuck} of 50 runs stuck near 0`, c: ROSEL }, { s: `α = ${A0}: ${a2.stuck} stuck`, c: GREEN }, { s: `TRPO: ${a3.stuck} stuck`, c: BLUE }], t - 7.5, { gap: .9 }); } },
 { h: 'Which part does the work?', p: ['TRPO here halves the step and checks the KL. Which prevents the crashes? Rose keeps the halving and gain check but drops the KL check.', '{GAIN}'],
   shot: [4, 12],
   draw(t) { const n = clamp(Math.floor((t - 1) * 6), 0, 40), a1 = avgAll('gain', fnG), a3 = avgAll('t' + LIM, fnT(LIM));
     upChart({ x0: 56, x1: 330, y0: 250, y1: 50, n, series: [{ v: a1.v, col: ROSEL, label: f2(a1.v[40]) }, { v: a3.v, col: BLUE, label: f2(a3.v[40]) }], title: 'average score after each update (50 runs)' });
     card([{ s: `halving + gain check, no KL: ${a1.stuck} of 50 stuck`, c: ROSEL }, { s: `full toy TRPO, limit ${LIM}: ${a3.stuck} stuck`, c: BLUE }], t - 7.5, { gap: .9 }); } },
 { h: 'Your turn', ctl: { label: 'KL limit', values: LIMS, def: 2, fmt: v => v },
   p: ['Drag the **KL limit**. Each setting reruns the same 50 runs of TRPO from a blank policy, always halving from α = 100. The line is the average score; the card counts runs stuck near 0.', '{SLIDE}'],
   shot: [4, 10],
   draw(t, st) { const d = LIMS[st.ci ?? 2], a = avg('t' + d, fnT(d));
     const ref = avgAll('t' + LIM, fnT(LIM));
     upChart({ x0: 56, x1: 330, y0: 250, y1: 50, series: [{ v: ref.v, col: '#b9cde3', w: 1.2 }, { v: a.v, col: BLUE, n: a.done ? 40 : 0, label: a.done ? f2(a.v[40]) : '' }], title: `TRPO, KL limit ${d}: average score (50 runs)` });
     card(a.done ? [`KL limit ${d}`, `after 10 updates ${f2(a.v[10])},  after 40 ${f2(a.v[40])}`, { s: `${a.stuck} of 50 runs stuck near 0`, c: a.stuck ? C.roseInk : GREEN }] : [`running … ${a.k} of 50`], 1, { gap: 0 });
     txt('pale blue line: limit 0.01, for comparison', 56, 278, { s: 12, c: C.soft, al: 'left' }); } },
 { h: 'A limit that adapts, at a price', p: [`A fixed small α also worked here, but the safe α changes as Pip learns. In run 0 the first 10 updates fit only steps of ${fa(Math.min(...RUN0.betas.slice(0, 10)))} to ${fa(Math.max(...RUN0.betas.slice(0, 10)))}; later the full ${ABIG} fit, as chances near 0 or 1 make the (1 − chance) factors and advantages small.`, 'The cost: at the old policy the KL is 0 and flat, so its curvature sets its growth. Real TRPO computes that curvature with second-order maths, slow and fiddly for big networks.'],
   shot: [4, 11],
   draw(t) { const b = RUN0.betas, x0 = 50, x1 = 350, y0 = 250, y1 = 60, X = i => x0 + (i + .5) / 40 * (x1 - x0), Y = v => y0 - Math.log2(v / .5) / Math.log2(200) * (y0 - y1);
     rline(x0, y0, x1, y0, 979, { c: C.line, w: 1, once: true }); txt('accepted step α, run 0, each update', x0 - 6, y1 - 20, { s: 13, c: C.soft, al: 'left' });
     [1, 10, 100].forEach(v => { txt(String(v), x0 - 12, Y(v), { s: 12, c: C.soft }); }); txt('update 1', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt('update 40', x1, y0 + 11, { s: 12, c: C.soft, al: 'right' });
     const n = clamp(Math.floor((t - .8) * 8), 0, 40); for (let i = 0; i < n; i++) { const v = b[i]; box(X(i) - 3, Y(v), 6, y0 - Y(v), BLUE, .7); }
     if (t > 7) card(['the limit is on behaviour, so the step adapts', { s: 'cost: second-order maths, slow for big networks', c: C.roseInk }], t - 7, { y: 300, gap: 1.2 }); } },
 { h: 'Where it was used', p: ['In 2015 TRPO trained simulated robots to swim, hop and walk, and played several Atari games straight from the screen pixels. Its steady improvement made it a standard baseline.', 'In 2017 Schulman and colleagues, now at OpenAI, introduced **PPO**, which keeps the “don’t move too far” idea with a much simpler trick. PPO then largely replaced TRPO in practice.'],
   shot: [3, 9],
   draw(t) { rline(20, 170, 380, 170, 990, { c: C.line });
     [['swim', 70], ['hop', 160], ['walk', 250]].forEach(([s, x], i) => { const u = clamp(t - .5 - i * .6); if (u <= 0) return; alpha(u, () => {
       const y = i === 1 ? 140 - Math.abs(Math.sin(t * 3)) * 18 : 140; if (i === 0) for (let k = 0; k < 3; k++) rline(x - 30, 160 + k * 6 - 6, x + 30, 160 + k * 6 - 6, 991 + k, { c: C.blue, w: 1.2 });
       L.robot(x + (i === 2 ? Math.sin(t) * 6 : 0), y, 1.4, t); txt(s, x, 192, { s: 16 }); }); });
     if (t > 2.3) alpha(clamp(t - 2.3), () => { rrect(300, 92, 70, 54, 995, { fill: '#e8e0d4' }); dot(320 + Math.sin(t * 2) * 12, 112, 4, C.rose); rline(345, 134, 360, 134, 996, { w: 2 }); txt('Atari', 335, 192, { s: 16 }); });
     card(['2015: TRPO, Schulman and colleagues', '2017: PPO, the simpler successor (OpenAI)'], t - 3, { gap: 1.2 }); } },
 { h: 'Recap', p: ['A step that is too big can wreck a good policy, and the wrecked policy then collects poor walks to learn from. A batch’s advice is a local map: trust it only near the old policy.', 'TRPO measures how far a step goes by the **KL divergence** between old and new chances, and only accepts steps within a limit.'],
   shot: [3, 9],
   draw(t) { maze({ t }); fans(PT); pipAt(SK, t);
     card(['too big a step can wreck the policy', 'measure the step by how much the chances change', 'KL: how much less likely new Pip makes old Pip’s moves', { s: 'TRPO: improve, but keep average KL ≤ limit', c: C.goldInk }], t - .3, { gap: 1 }); } },
];
// the 'forty updates' text needs the averaged numbers: compute them once (about a second) when that step's text is built
// texts that quote 50-run averages are built when their step is shown (the averages are warmed up a little per frame in earlier steps)
const lazyP = (h, fn) => { const s = STEPS.find(x => x.h === h), p0 = s.p; Object.defineProperty(s, 'p', { get: () => fn(p0) }); };
lazyP('Forty updates, fifty runs', p0 => { const a1 = avgAll('p100', fnP(ABIG)), a2 = avgAll('p3', fnP(A0)), a3 = avgAll('t' + LIM, fnT(LIM));
  return [p0[0], `Big plain steps: ${a1.stuck} of 50 runs stuck near 0. Small ones: ${f2(a2.v[10])} after 10 updates, ${f2(a2.v[40])} after 40. TRPO starts slower, ${f2(a3.v[10])} after 10, as early on only steps of about 1 to 3 fit. It ends at ${f2(a3.v[40])}, every run within ${f2(.531 - a3.min)} of the best, ${a3.stuck ? a3.stuck : 'none'} stuck.`]; });
lazyP('Which part does the work?', p0 => { const g = avgAll('gain', fnG);
  return [p0[0], `Without the KL check, ${g.stuck} of 50 runs get stuck, as bad as plain big steps. The batch predicts a gain even for huge steps: on the first update, ${GFULL} of 50 took the full α = ${ABIG}. The KL limit does the work.`]; });
lazyP('Your turn', p0 => { const a9 = avgAll('t1', fnT(1));
  return [p0[0], `Tiny limits are safe but crawl. Big limits let the big steps through: at limit 1, ${a9.stuck} of 50 runs get stuck.`]; });
const WARM = [['p100', fnP(ABIG)], ['p3', fnP(A0)], ['t' + LIM, fnT(LIM)], ['gain', fnG], ['t1', fnT(1)]];
function warm() { for (const [id, fn] of WARM) { const a = avg(id, fn); if (!a.done) return; } }
["The batch is a local map", 'How far is too far?', 'Surprise', 'KL divergence', 'The trust region rule', 'Halve until it fits'].forEach(h => { const s = STEPS.find(x => x.h === h), d = s.draw; s.draw = (t, st) => { d(t, st); warm(); }; });

RLStory.register('trpo', { title: 'TRPO: Trust Regions', track: 'Policy Gradients', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { S0, S3, S100, KL3, KL100, P0, P3, P100, SUMA, BYM, UPCAT, CAPPED, CAPPED3, C3, C100, FIRSTUP, SSX, SSY, PEAK, HALF, ST_T, RUN0, ST, CATW, CHW, avgAll, fnT, fnP, klCell, LIMS, BUMPWALK } });
})();
