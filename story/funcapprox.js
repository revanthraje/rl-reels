// ===== deeprl_kit.js (copy) =====
// Shared Deep RL kit: Pip's meadow (continuous x, y; cheese, cat, 4 wobbly moves), features, tiny tanh networks
// trained by SGD, a DQN trainer with switches (replay, target network, double), and drawing helpers.
// See briefs/notes/deeprl_world.md. Safe to load more than once.
(function () {
'use strict';
if (window.RLStory && RLStory.deeprl) return;
const L = RLStory.lib, { C, lerp, clamp, rng, gauss, rline, rrect, rcircle, txt, dot, box, alpha } = L;
const X = () => L.ctx;
// ───── the meadow ─────
const CHEESE = [0.85, 0.85], CR = 0.10, CAT = [0.60, 0.60], KR = 0.12, STEP = 0.15, WOB = 0.02, TMAX = 60, GAM = 0.9, NH = 8;
const START = [0.10, 0.10], MOVES = [[0, 1], [1, 0], [0, -1], [-1, 0]], MN = ['up', 'right', 'down', 'left'];
const dist = (x, y, p) => Math.hypot(x - p[0], y - p[1]);
const zone = (x, y) => dist(x, y, CHEESE) < CR ? 1 : dist(x, y, CAT) < KR ? -1 : 0;   // +1 cheese, −1 cat, 0 grass
// the four measurements, each scaled to roughly −1 … 1 (networks learn faster from centred inputs)
const feats = (x, y) => [2 * x - 1, 2 * y - 1, dist(x, y, CHEESE) / .6 - 1, dist(x, y, CAT) / .6 - 1];
const FNAMES = ['x', 'y', 'to cheese', 'to cat'];
// one move: 0.10 in direction m plus N(0, 0.02²) wobble per coordinate, clipped to the field. R = meadow rng.
function envStep(x, y, m, R) {
  const nx = clamp(x + MOVES[m][0] * STEP + gauss(R) * WOB), ny = clamp(y + MOVES[m][1] * STEP + gauss(R) * WOB), r = zone(nx, ny);
  return { x: nx, y: ny, r, done: r !== 0 };
}
function randStart(R) { for (;;) { const x = R(), y = R(); if (!zone(x, y)) return [x, y]; } }

// ───── tiny networks: nIn → nH (tanh) → nOut (linear). All knobs in one flat array w:
// W1 (nH × nIn, row j = weights into hidden unit j), b1 (nH), W2 (nOut × nH), b2 (nOut). Accessors: w1(n, j, i) etc.
function newNet(nIn, nH, nOut, seed) {
  const R = rng(seed), a = 1 / Math.sqrt(nIn), b = 1 / Math.sqrt(nH), n = shape(nIn, nH, nOut), w = n.w;
  for (let q = 0; q < n.oW2; q++) w[q] = (R() * 2 - 1) * a;
  for (let q = n.oW2; q < w.length; q++) w[q] = (R() * 2 - 1) * b;
  return n;
}
function shape(nIn, nH, nOut) { const oB1 = nH * nIn, oW2 = oB1 + nH, oB2 = oW2 + nOut * nH, N = oB2 + nOut;
  return { nIn, nH, nOut, oB1, oW2, oB2, N, w: new Float64Array(N), h: new Float64Array(nH), out: new Float64Array(nOut), g: new Float64Array(N), adam: null }; }
const copyNet = n => { const c = shape(n.nIn, n.nH, n.nOut); c.w.set(n.w); return c; };
const w1 = (n, j, i) => n.w[j * n.nIn + i], b1 = (n, j) => n.w[n.oB1 + j], w2 = (n, k, j) => n.w[n.oW2 + k * n.nH + j], b2 = (n, k) => n.w[n.oB2 + k];
// forward pass; returns { h, out } (fresh arrays, safe to keep)
function fwd(n, f) {
  const w = n.w, nIn = n.nIn, nH = n.nH, h = n.h, out = n.out;
  for (let j = 0; j < nH; j++) { let s = w[n.oB1 + j]; const o = j * nIn; for (let i = 0; i < nIn; i++) s += w[o + i] * f[i]; h[j] = Math.tanh(s); }
  for (let k = 0; k < n.nOut; k++) { let s = w[n.oB2 + k]; const o = n.oW2 + k * nH; for (let j = 0; j < nH; j++) s += w[o + j] * h[j]; out[k] = s; }
  return out;
}
function forward(n, f) { const out = Array.from(fwd(n, f)); return { h: Array.from(n.h), out }; }
// add the gradient of ½(target − out[k])² (sign flipped: direction that raises out[k] when err > 0) times err to g
function accGrad(n, f, k, target, g) {
  const out = fwd(n, f), err = target - out[k], w = n.w, h = n.h, nIn = n.nIn, nH = n.nH, o2 = n.oW2 + k * nH;
  for (let j = 0; j < nH; j++) { const gh = err * w[o2 + j] * (1 - h[j] * h[j]); g[o2 + j] += err * h[j]; g[n.oB1 + j] += gh; const o = j * nIn; for (let i = 0; i < nIn; i++) g[o + i] += gh * f[i]; }
  g[n.oB2 + k] += err; return err;
}
// one plain gradient step on a single example: every knob moves lr × (its share of the error). Returns the error before the step.
function sgd(n, f, k, target, lr) { n.g.fill(0); const err = accGrad(n, f, k, target, n.g); const w = n.w, g = n.g; for (let q = 0; q < n.N; q++) w[q] += lr * g[q]; return err; }
// minibatch step: average the gradient over items [{ f, k, target }], then 'sgd' (knobs += lr × average)
// or 'adam' (Adam: per-knob step sizes, state kept on the net). Returns the mean squared error before the step.
function batchStep(n, items, lr, opt = 'sgd') {
  const g = n.g, w = n.w, m = items.length; g.fill(0); let mse = 0;
  for (const it of items) { const e = accGrad(n, it.f, it.k, it.target, g); mse += e * e; }
  if (opt === 'adam') {
    const A = n.adam || (n.adam = { t: 0, m: new Float64Array(n.N), v: new Float64Array(n.N) }); A.t++;
    const c1 = 1 - .9 ** A.t, c2 = 1 - .999 ** A.t;
    for (let q = 0; q < n.N; q++) { const gg = g[q] / m; A.m[q] = .9 * A.m[q] + .1 * gg; A.v[q] = .999 * A.v[q] + .001 * gg * gg; w[q] += lr * (A.m[q] / c1) / (Math.sqrt(A.v[q] / c2) + 1e-8); }
  } else for (let q = 0; q < n.N; q++) w[q] += lr * g[q] / m;
  return mse / m;
}
const qOf = (n, x, y) => Array.from(fwd(n, feats(x, y)));
function argmax(v, R) { let b = -1e9, c = []; for (let i = 0; i < v.length; i++) { if (v[i] > b + 1e-12) { b = v[i]; c = [i]; } else if (v[i] > b - 1e-12) c.push(i); } return c.length === 1 || !R ? c[0] : c[Math.floor(R() * c.length)]; }

// ───── evaluation: greedy walks from fixed test starts (own seed, same for every net) ─────
const TESTS = (() => { const R = rng(4242), out = []; for (let i = 0; i < 30; i++) out.push(randStart(R)); return out; })();
function greedyWalk(n, x, y, seed, maxT = TMAX) {
  const R = rng(seed), P = [[x, y]]; let ret = 0, end = 0;
  for (let t = 0; t < maxT; t++) { const m = argmax(qOf(n, x, y)), s = envStep(x, y, m, R); x = s.x; y = s.y; P.push([x, y]); if (s.done) { end = s.r; ret = s.r * GAM ** t; break; } }
  return { P, ret, end };
}
function evalNet(n) { let ret = 0, ch = 0, ct = 0; TESTS.forEach(([x, y], i) => { const w = greedyWalk(n, x, y, 9000 + i); ret += w.ret; ch += w.end > 0; ct += w.end < 0; }); return { ret: ret / TESTS.length, cheese: ch, cat: ct, n: TESTS.length }; }

// ───── DQN trainer ─────
// o = { k: run id (seeds 5000+k meadow, 6000+k agent, 7000+k init), steps: env moves, lr, eps: number or fn(step),
//       replay: buffer size (0 = learn from the latest move only, in order), batch, targetEvery: copy period in updates
//       (0 = no target network: targets use the live net), double: bool, warm: moves before learning starts,
//       snapEvery: record a snapshot every that many moves, probe: [x, y] whose Q we log every move,
//       onMove(info, phase): optional callback, phase 'pre' (before the update) and 'post' (after); info = { s, x, y, m, rnd, nx, ny, r, done, net, tgt } }
function dqn(o = {}) {
  const k = o.k ?? 0, steps = o.steps ?? 8000, lr = o.lr ?? 0.003, batch = o.batch ?? 32, cap = o.replay ?? 2000, tEv = o.targetEvery ?? 100;
  const warm = o.warm ?? 100, snapEvery = o.snapEvery ?? 250, epsF = typeof o.eps === 'function' ? o.eps : () => (o.eps ?? 0.2);
  const Rm = rng(5000 + k), Ra = rng(6000 + k), net = newNet(4, NH, 4, 7000 + k); let tgt = tEv ? copyNet(net) : null;
  const buf = [], snaps = [], probeQ = [], walks = []; let [x, y] = randStart(Rm), t = 0, upd = 0, walk = [[x, y]], wret = 0;
  for (let s = 0; s < steps; s++) {
    const f = feats(x, y), rnd = Ra() < epsF(s), m = rnd ? Math.floor(Ra() * 4) : argmax(Array.from(fwd(net, f)), Ra), e = envStep(x, y, m, Rm);
    const info = o.onMove ? { s, x, y, m, rnd, nx: e.x, ny: e.y, r: e.r, done: e.done, net, tgt } : null; if (info) o.onMove(info, 'pre');
    const tr = { f, m, r: e.r, f2: feats(e.x, e.y), done: e.done };
    if (cap) { if (buf.length < cap) buf.push(tr); else buf[s % cap] = tr; }
    if (s >= warm) {
      const B = cap ? Array.from({ length: batch }, () => buf[Math.floor(Ra() * buf.length)]) : [tr];
      const items = B.map(b => { let next = 0;
        if (!b.done) { const qn = Array.from(fwd(tgt || net, b.f2)); next = o.double ? qn[argmax(Array.from(fwd(net, b.f2)))] : Math.max(...qn); }
        return { f: b.f, k: b.m, target: b.r + GAM * next }; });
      batchStep(net, items, lr, o.opt || 'adam');
      upd++; if (tEv && upd % tEv === 0) tgt = copyNet(net);
    }
    if (info) o.onMove(info, 'post');
    walk.push([e.x, e.y]); t++;
    if (e.done || t >= TMAX) { walks.push({ P: walk, end: e.done ? e.r : 0, s }); [x, y] = randStart(Rm); t = 0; walk = [[x, y]]; } else { x = e.x; y = e.y; }
    if (o.probe) probeQ.push(qOf(net, o.probe[0], o.probe[1]));
    if ((s + 1) % snapEvery === 0) snaps.push({ s: s + 1, net: copyNet(net), ev: evalNet(net) });
  }
  return { net, snaps, probeQ, walks };
}

// ───── drawing ─────
const FX = 72, FY = 14, FS = 256, px = x => FX + FS * x, py = y => FY + FS * (1 - y);
const TUFTS = (() => { const R = rng(31337), out = []; for (let i = 0; i < 46; i++) out.push([R(), R(), R()]); return out; })();
function dashCircle(x, y, r, col, a = 1) { const c = X(); c.save(); c.globalAlpha = a; c.setLineDash([4, 4]); c.strokeStyle = col; c.lineWidth = 1.3; c.beginPath(); c.arc(x, y, r, 0, 7); c.stroke(); c.restore(); }
// heat colour of a value: gold positive, rose negative, stronger with |v| (full at |v| ≥ o.vmax, default 0.8)
function heatCell(x0, y0, w, h, v, vmax = .8, a = 1) { if (Math.abs(v) < .004) return; box(x0, y0, w, h, v > 0 ? C.gold : C.rose, a * lerp(.1, .7, clamp(Math.abs(v) / vmax))); }
// meadow({ t, heat: (x, y) => value | null, n: heat squares per side (16), vmax, draw: 0..1 build-in, cat: alpha, cheese: alpha, labels: alpha, tufts: alpha })
function meadow(o = {}) {
  const t = o.t || 0, p = o.draw ?? 1, n = o.n || 16;
  box(FX, FY, FS, FS, '#eef3dc', .55 * p);
  if (o.heat) { const d = FS / n; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const x = (i + .5) / n, y = (j + .5) / n; if (zone(x, y)) continue; const v = o.heat(x, y); if (v != null) heatCell(FX + i * d, FY + FS - (j + 1) * d, d + .3, d + .3, v, o.vmax, o.heatA ?? 1); } }
  alpha((o.tufts ?? 1) * p, () => TUFTS.forEach(([a, b, c], i) => { const x = px(.03 + a * .94), y = py(.03 + b * .94); rline(x - 3, y + 3, x - 1, y - 3 - c * 2, 4000 + i, { c: '#9cb77a', w: 1, once: true, j: .4 }); rline(x + 1, y + 3, x + 3, y - 2 - c * 2, 4100 + i, { c: '#9cb77a', w: 1, once: true, j: .4 }); }));
  rrect(FX, FY, FS, FS, 4200, { w: 1.8, p: p >= 1 ? 1 : clamp(p * 1.5) });
  if ((o.cheese ?? 1) > 0) { alpha(o.cheese ?? 1, () => { dashCircle(px(CHEESE[0]), py(CHEESE[1]), CR * FS, '#c99a3a'); L.cheese(px(CHEESE[0]), py(CHEESE[1]) + 2, .8); }); }
  if ((o.cat ?? 1) > 0) { alpha(o.cat ?? 1, () => { dashCircle(px(CAT[0]), py(CAT[1]), KR * FS, C.rose); L.cat(px(CAT[0]), py(CAT[1]) + 3, t, .85); }); }
  if (o.labels) { const a = o.labels; txt('+1 on arrival', px(CHEESE[0]) - 4, py(CHEESE[1]) - CR * FS - 8, { s: 13, c: '#b07d12', a }); txt('−1', px(CAT[0]) - KR * FS - 10, py(CAT[1]) - KR * FS + 2, { s: 15, c: '#b54b4b', a }); }
}
// Pip at meadow position (x, y); face +1 right, −1 left
function pipAt(x, y, face = 1, t = 0, s = .72) { L.mouse(px(x), py(y) + 2, face, s, t); }
// path of positions [[x, y], ...] drawn as fading teal dots joined by a thin line; n = how many points to show
function path(P, n = P.length, o = {}) { const c = X(), m = Math.min(n, P.length); if (m < 1) return;
  c.save(); c.globalAlpha = (o.a ?? 1) * .5; c.strokeStyle = o.c || C.teal; c.lineWidth = 1.2; c.beginPath(); for (let i = 0; i < m; i++) { const x = px(P[i][0]), y = py(P[i][1]); i ? c.lineTo(x, y) : c.moveTo(x, y); } c.stroke(); c.restore();
  for (let i = 0; i < m; i++) dot(px(P[i][0]), py(P[i][1]), o.r || 2.3, o.c || C.teal, (o.a ?? 1) * (.35 + .65 * i / m)); }
// a network as dots and lines. b = { x, y, w, h }. o = { inL: labels, outL: labels, f: input vector (shows activations
// and output numbers), outFmt, a: alpha, hl: output index to highlight, wa: line alpha factor, build: 0..1 (lines appear) }
function drawNet(n, b, o = {}) {
  const a = o.a ?? 1; if (a <= 0) return; const c = X(); c.save(); c.globalAlpha = a;
  const col = (k, m) => m === 1 ? b.y + b.h / 2 : b.y + 10 + k * (b.h - 20) / (m - 1);
  const xs = [b.x, b.x + b.w / 2, b.x + b.w], Y = [n.nIn, n.nH, n.nOut].map(m => Array.from({ length: m }, (_, k) => col(k, m)));
  const act = o.f ? forward(n, o.f) : null, build = o.build ?? 1;
  const edge = (x1, y1, x2, y2, v, q) => { if (q > build) return; const m = clamp(Math.abs(v) / 1.5); c.save(); c.globalAlpha = a * (o.wa ?? 1) * lerp(.15, .8, m); c.strokeStyle = v >= 0 ? '#d8a23a' : C.rose; c.lineWidth = lerp(.5, 3, m); c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke(); c.restore(); };
  let q = 0; const NE = n.nH * (n.nIn + n.nOut);
  for (let j = 0; j < n.nH; j++) for (let i = 0; i < n.nIn; i++) edge(xs[0] + 7, Y[0][i], xs[1] - 6, Y[1][j], w1(n, j, i), q++ / NE);
  for (let k = 0; k < n.nOut; k++) for (let j = 0; j < n.nH; j++) edge(xs[1] + 6, Y[1][j], xs[2] - 7, Y[2][k], w2(n, k, j), q++ / NE);
  Y[0].forEach((y, i) => { rcircle(xs[0], y, 7, 4300 + i, { fill: C.card, j: .5 }); if (o.inL) txt(o.inL[i], xs[0] - 12, y, { s: 13, al: 'right', c: C.soft }); });
  Y[1].forEach((y, j) => { const h = act ? act.h[j] : 0; rcircle(xs[1], y, 5.5, 4320 + j, { fill: act ? (h >= 0 ? C.gold : C.rose) : C.card, j: .4 }); if (act) { c.save(); c.globalAlpha = a * (1 - Math.abs(h)); c.fillStyle = C.card; c.beginPath(); c.arc(xs[1], y, 4.5, 0, 7); c.fill(); c.restore(); } });
  Y[2].forEach((y, k) => { if (o.hl === k) dot(xs[2], y, 11, C.yellow, .8); rcircle(xs[2], y, 7, 4340 + k, { fill: C.card, j: .5 });
    const lab = (o.outL ? o.outL[k] : '') + (act ? (o.outL ? '  ' : '') + (o.outFmt || L.num)(act.out[k]) : ''); if (lab) txt(lab, xs[2] + 12, y, { s: 13, al: 'left', c: act && act.out[k] < 0 ? C.roseInk : C.goldInk }); });
  c.restore();
}

RLStory.deeprl = { CHEESE, CR, CAT, KR, STEP, WOB, TMAX, GAM, START, MOVES, MN, FNAMES, dist, zone, feats, envStep, randStart,
  newNet, copyNet, w1, b1, w2, b2, fwd, forward, accGrad, sgd, batchStep, qOf, argmax, TESTS, greedyWalk, evalNet, dqn,
  FX, FY, FS, px, py, meadow, heatCell, dashCircle, pipAt, path, drawNet };
})();
// ===== end deeprl_kit =====
// Function approximation as a story: Pip leaves the maze for an open meadow where no spot is ever visited twice.
// It learns the value of its habit with a five-knob linear recipe, then with a tiny neural network, by nudging knobs
// toward TD targets. Every number comes from the seeded simulation below (see briefs/notes/funcapprox.md).
(function () {
'use strict';
const L = RLStory.lib, D = RLStory.deeprl, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const { px, py, FX, FY, FS, CHEESE, CAT, GAM } = D;
const SEED = 5116, NW = 10000, SNAP = 100, NG = 16, NMC = 100, A_LIN = 0.05, A_NET = 0.1, EPS = 0.2;
const decay = k => Math.max(0.05, 1 - k / NW);                       // step size multiplier for walk k (0-based)
// ───── Pip's habit: below 0.7 high go up; then step toward the cheese along the bigger gap. 1 move in 5 random.
const habit = (x, y) => { if (y < 0.7) return 0; const dx = CHEESE[0] - x, dy = CHEESE[1] - y; return Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 1 : 3) : (dy > 0 ? 0 : 2); };
const raw = (x, y) => [x, y, D.dist(x, y, CHEESE), D.dist(x, y, CAT)];
// ───── the walks: walk 1 starts at the start (0.10, 0.10), the rest at random spots. rng(5116) for everything.
const WALKS = (() => { const R = rng(SEED), out = [];
  for (let w = 0; w < NW; w++) { let [x, y] = w === 0 ? D.START : D.randStart(R); const tr = [];
    for (let t = 0; t < D.TMAX; t++) { const rnd = R() < EPS, m = rnd ? Math.floor(R() * 4) : habit(x, y), s = D.envStep(x, y, m, R);
      tr.push({ x, y, m, rnd, nx: s.x, ny: s.y, r: s.r, done: s.done }); x = s.x; y = s.y; if (s.done) break; }
    out.push(tr); }
  return out; })();
const W1 = WALKS[0], W1P = [[W1[0].x, W1[0].y]].concat(W1.map(s => [s.nx, s.ny])), W1R = W1.findIndex(s => s.rnd);
const NSPOT = WALKS.reduce((s, w) => s + w.length, 0) + NW;           // landing spots + start spots
const DUP = (() =>  // repeats of an exact spot, not counting the four fence corners (Pip pushed into a corner)
 { const seen = new Set(); let d = 0; WALKS.forEach(w => { [[w[0].x, w[0].y]].concat(w.map(s => [s.nx, s.ny])).forEach(([x, y]) => { const k = x + ',' + y, corner = (x === 0 || x === 1) && (y === 0 || y === 1); if (seen.has(k)) { if (!corner) d++; } else seen.add(k); }); }); return d; })();
// demo of three 'right' moves from the start (rng 5300)
const DEMO = (() => { const R = rng(5300), P = [D.START.slice()]; let [x, y] = D.START; for (let i = 0; i < 3; i++) { const s = D.envStep(x, y, 1, R); x = s.x; y = s.y; P.push([x, y]); } return P; })();
// ───── the true values of the habit (only the narrator can do this): 100 walks from the centre of each of 16 × 16 squares
const memo = {}, M = (k, f) => (k in memo ? memo[k] : (memo[k] = f()));
const SQ = []; for (let i = 0; i < NG; i++) for (let j = 0; j < NG; j++) { const x = (i + .5) / NG, y = (j + .5) / NG; if (!D.zone(x, y)) SQ.push([x, y, i, j]); }
const TRUE = () => M('true', () => { const R = rng(5200), T = {}; SQ.forEach(([x0, y0, i, j]) => { let s = 0;
  for (let n = 0; n < NMC; n++) { let x = x0, y = y0, g = 1; for (let t = 0; t < 300; t++) { const m = R() < EPS ? Math.floor(R() * 4) : habit(x, y), e = D.envStep(x, y, m, R); if (e.done) { s += e.r * g; break; } g *= GAM; x = e.x; y = e.y; } }
  T[i * NG + j] = s / NMC; }); return T; });
const trueAt = (x, y) => TRUE()[Math.min(NG - 1, Math.floor(x * NG)) * NG + Math.min(NG - 1, Math.floor(y * NG))];
const gap = V => { const T = TRUE(); let s = 0; SQ.forEach(([x, y, i, j]) => { s += Math.abs(V(x, y) - T[i * NG + j]); }); return s / SQ.length; };
// ───── the linear recipe: guess = b + w1·x + w2·y + w3·(to cheese) + w4·(to cat); semi-gradient TD(0)
const linV = w => (x, y) => { const f = raw(x, y); return w[0] + w[1] * f[0] + w[2] * f[1] + w[3] * f[2] + w[4] * f[3]; };
const LIN = () => M('lin', () => { const w = [0, 0, 0, 0, 0], V = linV(w), snaps = [w.slice()]; let first = null;
  WALKS.forEach((tr, k) => { const a = A_LIN * decay(k); tr.forEach(s => { const f = [1].concat(raw(s.x, s.y)), g = V(s.x, s.y), tg = s.r + (s.done ? 0 : GAM * V(s.nx, s.ny)), e = tg - g;
      if (!first && e !== 0) first = { s, f, before: w.slice(), g, tg, e, a, k }; for (let i = 0; i < 5; i++) w[i] += a * e * f[i]; if (first && !first.after) first.after = w.slice(); });
    if ((k + 1) % SNAP === 0) snaps.push(w.slice()); });
  return { w, snaps, first, gaps: snaps.map(q => gap(linV(q))) }; });
// ───── the tiny network: 4 measurements (scaled) → 8 tanh → 1; same TD target, plain gradient steps
const netV = n => (x, y) => D.fwd(n, D.feats(x, y))[0];
const NET = () => M('net', () => { const n = D.newNet(4, 8, 1, 7100), V = netV(n), snaps = [D.copyNet(n)]; let first = null;
  WALKS.forEach((tr, k) => { const a = A_NET * decay(k); tr.forEach(s => { const tg = s.r + (s.done ? 0 : GAM * V(s.nx, s.ny)), g = V(s.x, s.y);
      if (!first && s.r !== 0) first = { s, g, tg, a, k, pre: D.copyNet(n) }; const e = D.sgd(n, D.feats(s.x, s.y), 0, tg, a); if (first && first.after == null) { first.e = e; first.after = V(s.x, s.y); first.net = D.copyNet(n); } });
    if ((k + 1) % SNAP === 0) snaps.push(D.copyNet(n));
    if (k === RIP.k) { RIP.after = D.copyNet(n); } if (k === RIP.k - 1) RIP.before = D.copyNet(n); });
  return { n, snaps, first, gaps: snaps.map(q => gap(netV(q))) }; });
// the ripple walk: the first walk from walk 51 on that is a single move into the cheese (one nudge at one spot)
const RIP = { k: WALKS.findIndex((w, k) => k >= 50 && w.length === 1 && w[0].r > 0) };
// ───── numbers the story quotes
const f2 = v => L.num(v, 2).replace('-', '−'), f3 = v => (Math.abs(v) < .0005 ? 0 : v).toFixed(3).replace('-', '−'), fx = v => v.toFixed(2);
const WIDE = [0.40625, 0.34375], BELOW = [0.60, 0.30], NEAR = [0.85, 0.55];                       // a spot below the cat; a spot below the cheese, right of the cat
const S0 = D.START, SF = raw(S0[0], S0[1]);
const sg = v => (v >= 0 ? '+' : '') + f3(v);
const ripD = () => { NET(); const A = netV(RIP.before), B = netV(RIP.after), s = WALKS[RIP.k][0]; return { start: B(...S0) - A(...S0), below: B(...BELOW) - A(...BELOW), here: B(s.x, s.y) - A(s.x, s.y) }; };
// your turn: nudge a copy of the trained model toward +1 at the tapped spot, 5 times with step size 0.05
function nudged(kind, x, y) { const key = kind + x.toFixed(3) + y.toFixed(3); return M(key, () => {
  if (kind === 0) { const w = LIN().w.slice(), V = linV(w); for (let i = 0; i < 5; i++) { const f = [1].concat(raw(x, y)), e = 1 - V(x, y); for (let q = 0; q < 5; q++) w[q] += .05 * e * f[q]; } return V; }
  const n = D.copyNet(NET().n); for (let i = 0; i < 5; i++) D.sgd(n, D.feats(x, y), 0, 1, .05); return netV(n); }); }

// ───── drawing helpers
function mini(ox, oy, s, o) { ctx.save(); ctx.translate(ox, oy); ctx.scale(s, s); ctx.translate(-FX, -FY); D.meadow(o); if (o.after) o.after(); ctx.restore(); }
function axes(a = 1) { txt('x →', px(.5), FY + FS + 10, { s: 13, c: C.soft, a }); txt('y', FX - 14, py(.52), { s: 13, c: C.soft, a }); txt('↑', FX - 14, py(.6), { s: 13, c: C.soft, a }); txt('0', FX - 8, FY + FS + 8, { s: 12, c: C.soft, a }); txt('1', FX + FS, FY + FS + 9, { s: 12, c: C.soft, a }); txt('1', FX - 9, FY + 4, { s: 12, c: C.soft, a }); }
function habitArrows(a) { if (a <= 0) return; for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) { const x = (i + .5) / 8, y = (j + .5) / 8; if (D.zone(x, y)) continue; const m = habit(x, y), [dx, dy] = D.MOVES[m];
  alpha(a * .55, () => arrow(px(x) - dx * 6, py(y) + dy * 6, px(x) + dx * 7, py(y) - dy * 7, 600 + i * 8 + j, { c: C.goldInk, w: 1.1, head: 4 })); } }
// Pip walking along positions P (one move per dur seconds), with fading dots
function walkAlong(P, t, dur, o = {}) { const k = Math.floor(t / dur), u = ease((t % dur) / dur * 1.4), n = P.length - 1;
  if (k >= n) { D.path(P, P.length, o); const l = P[n], p = P[n - 1]; D.pipAt(l[0], l[1], Math.sign(l[0] - p[0]) || 1, t); return { k: n, done: true }; }
  D.path(P, k + 1, o); const a = P[k], b = P[k + 1], x = lerp(a[0], b[0], u), y = lerp(a[1], b[1], u); D.pipAt(x, y, Math.sign(b[0] - a[0]) || 1, t); return { k, u }; }
function ring(x, y, col, a = 1, r = 9) { alpha(a, () => rcircle(px(x), py(y), r, 700, { c: col, w: 2 })); }
function tag(s, x, y, a = 1, col, o = {}) { const X = px(x) + (o.dx ?? 0), Y = py(y) + (o.dy ?? -14); box(X - (o.w ?? 34), Y - 9, 2 * (o.w ?? 34), 18, C.card, .9 * a); txt(s, X, Y, { s: o.s || 13, c: col || C.ink, a }); }
// line chart of the average gap over walks. series [{ v (one per snapshot), col, w }], n = snapshots shown
function gapChart(series, n, o = {}) {
  const x0 = 70, x1 = 330, y0 = 386, y1 = 306, hi = 0.5, X = i => x0 + i / (NW / SNAP) * (x1 - x0), Y = v => y0 - v / hi * (y0 - y1);
  rline(x0 - 4, y0, x1 + 4, y0, 960, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1 - 4, 961, { c: C.line, w: 1, once: true });
  txt('0', x0 - 14, y0, { s: 12, c: C.soft }); txt('0.5', x0 - 18, y1, { s: 12, c: C.soft });
  txt('walk 1', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt('walk ' + NW.toLocaleString('en-US'), x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
  txt(o.title || 'average gap: guess vs true value', x0 - 10, y1 - 13, { s: 13, c: C.soft, al: 'left' });
  series.forEach(s => { ctx.save(); ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 1.6; ctx.lineJoin = 'round'; ctx.beginPath(); const m = Math.min(s.n ?? n, s.v.length);
    for (let i = 0; i < m; i++) { const x = X(i), y = Y(Math.min(hi, s.v[i])); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore();
    if (m === s.v.length && s.label) txt(s.label, x1 + 6, Y(s.v[m - 1]) + (s.dy || 0), { s: 13, c: s.col, al: 'left' }); });
}
function legend(y, a = 1) { alpha(a, () => { for (let i = 0; i < 6; i++) { box(124 + i * 14, y - 6, 14, 12, C.rose, lerp(.7, .1, i / 5)); box(206 + i * 14, y - 6, 14, 12, C.gold, lerp(.1, .7, i / 5)); }
  txt('−1', 112, y, { s: 12, c: C.roseInk }); txt('0', 203, y, { s: 12, c: C.soft }); txt('+1', 300, y, { s: 12, c: C.goldInk }); }); }
const INL = ['x', 'y', 'to cheese', 'to cat'];

const STEPS = [
 { h: 'Out of the maze', get p() { return ['Pip has left the maze for an open **meadow**: no cells, no walls inside. Its position is just two numbers, **x** from left to right and **y** from bottom to top, each between 0 and 1.', 'The rules stay: +1 on the step landing inside the cheese circle, −1 inside the cat’s circle, either ending the walk. γ (gamma) is still 0.9.']; },
   shot: [3, 9],
   draw(t) { D.meadow({ t, draw: clamp(t / 1.2), cheese: clamp(t - 1.4), cat: clamp(t - 2), labels: clamp(t - 2.6) }); axes(clamp(t - 1)); D.pipAt(S0[0], S0[1], 1, t);
     if (t > 3.4) card(['position: two numbers, x and y, from 0 to 1', '+1 into the cheese circle, −1 into the cat’s', { s: 'either one ends the walk · γ = 0.9', c: C.soft }], t - 3.4); } },
 { h: 'Moves on bumpy grass', get p() { return [`Pip still has four moves: up, right, down, left. A move carries it 0.15 that way, plus a small random **wobble** from the uneven grass. Three moves right from the start (0.10, 0.10) landed at (${fx(DEMO[1][0])}, ${fx(DEMO[1][1])}), then (${fx(DEMO[2][0])}, ${fx(DEMO[2][1])}), then (${fx(DEMO[3][0])}, ${fx(DEMO[3][1])}).`, 'The fence round the meadow stops Pip at the edge.']; },
   shot: [2, 7],
   draw(t) { D.meadow({ t }); const m = walkAlong(DEMO, Math.max(0, t - .8), 1.3, { c: C.teal });
     DEMO.slice(1, m.k + 1).forEach((p, i) => tag(`${fx(p[0])}, ${fx(p[1])}`, p[0], p[1], clamp((t - .8 - (i + 1) * 1.3) * 2), C.ink, { dy: i % 2 ? 18 : -18, w: 32 }));
     card(['a move: 0.15 in its direction', '+ a small random wobble', { s: 'the fence stops Pip at the edge', c: C.soft }], t - 5.2); } },
 { h: 'Pip’s habit', get p() { return ['As in the values reel, Pip follows a fixed **habit**. Below 0.7 high, it goes up. Higher, it steps toward the cheese along whichever gap is bigger, sideways or up and down. One move in five goes a random direction instead.', `Here is walk 1: ${W1.length} moves; move ${W1R + 1} was random. We want this habit’s **values**: from each spot, the expected discounted return.`]; },
   shot: [3, 10],
   draw(t) { D.meadow({ t }); habitArrows(clamp(t - .3)); const m = walkAlong(W1P, Math.max(0, t - 2), .55);
     if (m.k > W1R && m.k <= W1R + 3) txt('random move', px(W1P[W1R + 1][0]) + 44, py(W1P[W1R + 1][1]), { s: 13, c: C.roseInk });
     if (m.done) bubble('+1', px(CHEESE[0]), py(CHEESE[1]) - 18, t - 2 - W1.length * .55);
     card(['below 0.7 high: go up', 'then: toward the cheese, along the bigger gap', { s: '1 move in 5: a random direction', c: C.soft }], t - 1); } },
 { h: 'A notebook for every spot?', get p() { return ['Each maze cell’s number was visited again and again. Here positions are real numbers and the wobble random, so every landing spot is new.', `Pip took ${NW.toLocaleString('en-US')} walks, from walk 2 on with random starts. It stood on ${NSPOT.toLocaleString('en-US')} spots and, except when squeezed into a fence corner, ${DUP === 0 ? 'never on exactly the same spot twice' : 'on the same spot twice only ' + DUP + ' times'}. A notebook page per spot would never be reread.`]; },
   shot: [3, 10],
   draw(t) { D.meadow({ t }); D.path(W1P, W1P.length, { a: .8 });
     [2, 5, 8].forEach((i, q) => tag(`${f3(W1P[i][0])}, ${f3(W1P[i][1])}`, W1P[i][0], W1P[i][1], clamp(t - .8 - q * .6), C.ink, { dx: q === 1 ? -46 : 46, dy: 0, w: 40 }));
     const n = clamp(Math.floor((t - 3) * 40), 0, 150); for (let w = 1; w <= n; w++) { const s = WALKS[w][0]; dot(px(s.nx), py(s.ny), 1.8, C.blue, .7); }
     if (t > 3) caption('blue: first landing spots of the next 150 walks', clamp(t - 3));
     if (t > 6) card([`spots visited in ${NW.toLocaleString('en-US')} walks: ${NSPOT.toLocaleString('en-US')}`, { s: `exact repeats (fence corners aside): ${DUP}`, c: C.roseInk }], t - 6); } },
 { h: 'Chop it into squares?', get p() { return ['We could cut the meadow into squares and keep one number per square. With 16 by 16 squares that is 256 numbers, and each learns only from its own visits.', 'Worse, two neighbouring squares are almost the same place, yet what one learns teaches the other nothing. Finer squares mean more numbers and fewer visits each; coarser ones blur everything together.']; },
   shot: [3, 10],
   draw(t) { D.meadow({ t }); const a = clamp(t - .5); for (let k = 1; k < 16; k++) { alpha(a * .6, () => { rline(px(k / 16), FY, px(k / 16), FY + FS, 800 + k, { c: C.line, w: .8, once: true, j: .3 }); rline(FX, py(k / 16), FX + FS, py(k / 16), 820 + k, { c: C.line, w: .8, once: true, j: .3 }); }); }
     const u = clamp(t - 2.5); box(px(4 / 16), py(6 / 16), FS / 16, FS / 16, C.teal, .7 * u); box(px(5 / 16), py(6 / 16), FS / 16, FS / 16, C.blue, .7 * u); D.pipAt(4.4 / 16, 3.6 / 16, 1, t);
     if (u > 0) arrow(px(4.9 / 16), py(4.4 / 16), px(5 / 16), py(5.3 / 16), 1600, { c: C.ink, w: 1.3, head: 5 });
     if (t > 3) { txt('two squares, almost the same place,', W / 2, 300, { s: 16, a: clamp(t - 3) }); txt('learning separately', W / 2, 320, { s: 16, a: clamp(t - 3) }); }
     card(['16 × 16 = 256 separate numbers', { s: 'neighbours share nothing they learn', c: C.roseInk }], t - 5, { y: 336 }); } },
 { h: 'Nearby spots, similar values', get p() { return [`The habit’s true values, colour by colour: gold good, rose bad, stronger for bigger. We worked them out with ${NMC} test walks from the middle of each square, something Pip cannot do; it must learn them.`, `Note the dark band **below the cat**: from there the habit climbs straight into it, ${f2(trueAt(...BELOW))} at (0.60, 0.30). Mostly, though, nearby spots have similar values.`]; },
   shot: [4, 11],
   draw(t) { const T = TRUE(); D.meadow({ t, heat: (x, y) => trueAt(x, y), heatA: clamp(t - .5) }); axes(.7);
     legend(300, clamp(t - 1.5)); txt('true value of the habit, square by square', W / 2, 322, { s: 14, c: C.soft, a: clamp(t - 1.5) });
     if (t > 4) { ring(BELOW[0], BELOW[1], C.roseInk, clamp(t - 4)); tag(f2(trueAt(...BELOW)), BELOW[0], BELOW[1], clamp(t - 4), C.roseInk, { dy: 18, w: 22 }); }
     if (t > 6) card([{ s: 'close spots: similar values', c: C.goldInk }, { s: 'so one spot’s lesson should help its neighbours', c: C.soft }], t - 6, { y: 338 }); } },
 { h: 'Four measurements', get p() { return ['So instead of naming the spot, Pip **describes** it with a few measurements, called **features**: its x, its y, its distance to the cheese and its distance to the cat.', `At the start these are ${fx(SF[0])}, ${fx(SF[1])}, ${fx(SF[2])} and ${fx(SF[3])}. Spots that are close have similar measurements, so a rule built on them treats them alike. That is **generalisation**.`]; },
   shot: [3, 9],
   draw(t) { D.meadow({ t, heatA: 0 }); const a = clamp(t - .8), b = clamp(t - 1.6);
     alpha(a, () => { ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = C.goldInk; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(px(S0[0]), py(S0[1])); ctx.lineTo(px(CHEESE[0]), py(CHEESE[1])); ctx.stroke(); ctx.restore(); });
     alpha(b, () => { ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = C.roseInk; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(px(S0[0]), py(S0[1])); ctx.lineTo(px(CAT[0]), py(CAT[1])); ctx.stroke(); ctx.restore(); });
     txt(`to cheese ${fx(SF[2])}`, px(.27), py(.5), { s: 14, c: C.goldInk, a }); txt(`to cat ${fx(SF[3])}`, px(.46), py(.26), { s: 14, c: C.roseInk, a: b }); D.pipAt(S0[0], S0[1], 1, t);
     card([`x = ${fx(SF[0])}      y = ${fx(SF[1])}`, `to cheese = ${fx(SF[2])}      to cat = ${fx(SF[3])}`, { s: 'four measurements describe any spot', c: C.soft }], t - 2.6); } },
 { h: 'A recipe with five knobs', get p() { return ['The simplest rule: multiply each measurement by its own number, add them up, plus a fixed amount b. Those five numbers are **weights**, the knobs Pip can turn. Such a rule is a **linear** recipe.', 'All five knobs start at 0, so the recipe guesses 0 everywhere. One recipe guesses for every spot on the meadow, even spots never visited.']; },
   shot: [3, 9],
   draw(t) { D.meadow({ t, heat: () => 0 }); D.pipAt(S0[0], S0[1], 1, t); txt('every guess 0', px(.3), py(.45), { s: 15, c: C.soft, a: clamp(t - 5) });
     card(['guess = b + w1 × x + w2 × y', '          + w3 × to cheese + w4 × to cat', { s: 'knobs b, w1, w2, w3, w4: all 0 to start', c: C.soft }], t - .6, { gap: 1.4 }); } },
 { h: 'Nudging the knobs', get p() { return [`TD target: reward + 0.9 × the landing guess, or just the reward at the end. Walk 1’s last step, from (${fx(LIN().first.s.x)}, ${fx(LIN().first.s.y)}) into the cheese: target 1, guess 0, error 1.`, `Each knob moves by **step size × error × its measurement**, with 1 for b. At step size 0.05, b becomes ${f3(LIN().first.after[0])}. That is a **gradient step** on (target − guess)², target held fixed: bigger measurements sway the guess more, so move more.`]; },
   shot: [3, 11],
   draw(t) { const F = LIN().first; D.meadow({ t }); D.path(W1P, W1P.length, { a: .5 }); ring(F.s.x, F.s.y, C.goldInk, clamp(t - .5)); D.pipAt(CHEESE[0], CHEESE[1], 1, t); bubble('+1', px(CHEESE[0]), py(CHEESE[1]) - 18, t - .6);
     const nm = ['b', 'w1 (x)', 'w2 (y)', 'w3 (cheese)', 'w4 (cat)'], x0 = 34, dx = 68;
     if (t > 2) { box(18, 290, 364, 108, C.card, .95); rrect(18, 290, 364, 108, 210, { c: C.line, w: 1.1 }); txt('knob += 0.05 × error 1 × measurement', W / 2, 304, { s: 15, c: C.soft }); }
     nm.forEach((s, i) => { const u = clamp(t - 2.6 - i * .7); if (u <= 0) return; const X = x0 + i * dx + 30;
       txt(s, X, 330, { s: 13, c: C.soft, a: u }); txt(`× ${f2(F.f[i])}`, X, 350, { s: 14, a: u }); txt(`0 → ${f3(F.after[i])}`, X, 374, { s: 15, c: C.goldInk, a: u }); }); } },
 { h: 'One nudge moved every spot', get p() { return [`That single nudge changed the guess **everywhere**. The spot Pip left now guesses ${f3(linV(LIN().first.after)(LIN().first.s.x, LIN().first.s.y))}. The start, far away, now guesses ${f3(linV(LIN().first.after)(S0[0], S0[1]))}, and the spot below the cat ${f3(linV(LIN().first.after)(...BELOW))}, though nothing happened there.`, 'This sharing is the whole point, and its danger: every lesson about one spot spills onto all the others.']; },
   shot: [4, 11],
   draw(t) { const F = LIN().first, V = linV(F.after), u = clamp((t - 1) / 1.5); D.meadow({ t, heat: (x, y) => u * V(x, y), vmax: .2 }); txt('colours ×4 brighter here', FX + FS - 4, FY + FS + 10, { s: 12, c: C.soft, al: 'right' });
     [[F.s.x, F.s.y, 'here'], [S0[0], S0[1], 'start'], [BELOW[0], BELOW[1], 'below cat']].forEach(([x, y, s], i) => { const a = clamp(t - 3 - i * 1.2); ring(x, y, C.goldInk, a); tag(`${s} ${f3(u * V(x, y))}`, x, y, a, C.goldInk, { dy: i === 0 ? 18 : -18, dx: i === 1 ? 26 : 0, w: 46 }); });
     card(['one nudge after one step at one spot', { s: 'lifted the guess of every spot on the meadow', c: C.goldInk }], t - 6.5, { y: 310 }); } },
 { h: 'Training the recipe', get p() { return [`Train on all ${NW.toLocaleString('en-US')} walks; random starts cover the whole meadow. Pip nudges after every move, and the step size shrinks from 0.05 to ${f3(A_LIN * .05)}, so late walks only fine-tune.`, `Chart: the **average gap** between guess and true value over the 240 grass squares. It drops from ${f2(LIN().gaps[0])} to ${f2(LIN().gaps[LIN().gaps.length - 1])}, mostly in the first few hundred walks, then wobbles as walks tug the shared knobs.`]; },
   shot: [4, 12],
   draw(t) { const S = LIN(), n = clamp(Math.floor((t - 1) * 12), 0, S.snaps.length - 1); D.meadow({ t, heat: linV(S.snaps[n]) }); D.pipAt(S0[0], S0[1], 1, t);
     gapChart([{ v: S.gaps, col: '#c99a3a', label: f2(S.gaps[S.gaps.length - 1]), dy: -6 }], n + 1); caption(`after walk ${(n * SNAP || 1).toLocaleString('en-US')}`); } },
 { h: 'What a straight recipe can’t draw', get p() { return [`Left: the true values. Right: the trained recipe. It gets the big picture, good near the cheese, poor far away, but misses the band below the cat: at (0.60, 0.30) the truth is ${f2(trueAt(...BELOW))}, the recipe ${f2(linV(LIN().w)(...BELOW))}.`, 'Why? The recipe only adds up measurements. A dip **only** where Pip is near the cat’s x **and** below it needs measurements combined, and adding can’t do “and”.']; },
   shot: [4, 11],
   draw(t) { const V = linV(LIN().w); mini(8, 40, .7, { t, heat: (x, y) => trueAt(x, y), labels: 0, after: () => ring(BELOW[0], BELOW[1], C.roseInk, clamp(t - 3)) });
     mini(212, 40, .7, { t, heat: V, after: () => ring(BELOW[0], BELOW[1], C.roseInk, clamp(t - 3)) }); txt('true values', 98, 26, { s: 15, c: '#3f7f7c' }); txt('linear recipe', 302, 26, { s: 15, c: C.goldInk });
     txt(f2(trueAt(...BELOW)), 98, 240, { s: 15, c: C.roseInk, a: clamp(t - 3) }); txt(f2(V(...BELOW)), 302, 240, { s: 15, c: C.soft, a: clamp(t - 3) }); txt('below the cat:', W / 2, 240, { s: 14, c: C.soft, a: clamp(t - 3) });
     legend(272, .8); card(['the recipe adds measurements, each times its knob', { s: '“near the cat’s x and below it” needs combining', c: C.roseInk }], t - 5, { y: 296 }); } },
 { h: 'A tiny neural network', get p() { return ['Hand-making a measurement for that band needs the answer already. Instead, a tiny **neural network**: the four measurements, scaled to −1 to 1, feed 8 **hidden units**, each summing them with its own weights and squashing the result into −1 to 1. The output sums those with more weights: the guess.', `Lines: weights; gold positive, rose negative, thicker bigger. 49 knobs, starting small and random. From the start’s measurements it guesses ${f2(D.forward(NET().snaps[0], D.feats(...S0)).out[0])}.`]; },
   shot: [3, 10],
   draw(t) { const n = NET().snaps[0], f = D.feats(S0[0], S0[1]); D.drawNet(n, { x: 120, y: 30, w: 170, h: 220 }, { inL: INL, outL: ['guess'], f: t > 6 ? f : null, build: clamp((t - .5) / 3) });
     txt('measurements', 92, 268, { s: 13, c: C.soft }); txt('8 hidden units', 205, 268, { s: 13, c: C.soft }); txt('guess', 300, 268, { s: 13, c: C.soft });
     card(['hidden unit: add up (weight × input), then squash', 'guess: add up (weight × hidden unit)', { s: '4 × 8 + 8 + 8 + 1 = 49 knobs', c: C.soft }], t - 3.5); } },
 { h: 'Same nudge, through the network', get p() { return ['Each knob moves by step size × error × **how much turning that knob would change this guess**. For the recipe that was the measurement. In a network it is computed layer by layer, output first: **backpropagation**.', `At walk 1’s end it guessed ${f3(NET().first.g)} beside the cheese: error 1 − ${f3(NET().first.g)} = ${f3(NET().first.e)}. One nudge at step size 0.1 lifts it to ${f3(NET().first.after)}.`]; },
   shot: [3, 10],
   draw(t) { const F = NET().first, f = D.feats(F.s.x, F.s.y), aft = t > 6.5; D.drawNet(aft ? F.net : F.pre, { x: 120, y: 24, w: 170, h: 200 }, { inL: INL, outL: ['guess'], f, outFmt: f3 });
     const ea = clamp(t - 1.5) * (aft ? 0 : 1); if (ea > 0) { alpha(ea, () => arrow(300, 252, 150, 252, 1700, { c: C.roseInk, w: 1.6 })); txt('the error flows back', 225, 266, { s: 13, c: C.roseInk, a: ea }); }
     txt(aft ? 'after one nudge' : 'before', 330, 180, { s: 14, c: aft ? C.goldInk : C.soft });
     card([`target 1, guess ${f3(F.g)}, error ${f3(F.e)}`, 'knob += 0.1 × error × (effect of knob on guess)', { s: `guess here: ${f3(F.g)} → ${f3(F.after)}`, c: C.goldInk }], t - 3); } },
 { h: 'The network learns the shadow', get p() { const N = netV(NET().n); return [`The same ${NW.toLocaleString('en-US')} walks, now nudging the network, step size 0.1 shrinking to ${f3(A_NET * .05)}. Blue is its average gap, gold the recipe’s. The network ends at ${f2(NET().gaps[NET().gaps.length - 1])} against ${f2(LIN().gaps[LIN().gaps.length - 1])}.`, `The band below the cat appears: at (0.60, 0.30) the network says ${f2(N(...BELOW))}, the truth ${f2(trueAt(...BELOW))}. But it is too wide: at (${fx(WIDE[0])}, ${fx(WIDE[1])}), truly worth ${f2(trueAt(...WIDE))}, it says ${f2(N(...WIDE))}.`]; },
   shot: [5, 12],
   draw(t) { const S = NET(), n = clamp(Math.floor((t - 1) * 12), 0, S.snaps.length - 1); D.meadow({ t, heat: netV(S.snaps[n]) }); D.pipAt(S0[0], S0[1], 1, t);
     if (n === S.snaps.length - 1) { ring(BELOW[0], BELOW[1], C.roseInk, clamp(t - 10)); ring(WIDE[0], WIDE[1], C.ink, clamp(t - 11)); }
     gapChart([{ v: LIN().gaps, col: '#e3c48a', w: 1.2, label: f2(LIN().gaps[LIN().gaps.length - 1]), dy: -7 }, { v: S.gaps, col: '#4f7fb0', n: n + 1, label: f2(S.gaps[S.gaps.length - 1]), dy: 7 }], n + 1);
     caption(`network after walk ${(n * SNAP || 1).toLocaleString('en-US')}`); } },
 { h: 'The catch: shared knobs', get p() { const d = ripD(); return [`Walk ${RIP.k + 1} was one move into the cheese: one nudge, raising that spot by ${f3(d.here)}. The far-off start moved too, by ${sg(d.start)}. TD targets use the same knobs, so nudges move them too.`, 'Shared knobs, guesses from guesses and **different**-policy data can blow learning up together: Sutton and Barto’s **deadly triad**. Here the third was missing: Pip learned about its own habit, and stayed calm.']; },
   shot: [4, 11],
   draw(t) { NET(); const A = netV(RIP.before), B = netV(RIP.after), P = [[WALKS[RIP.k][0].x, WALKS[RIP.k][0].y]].concat(WALKS[RIP.k].map(s => [s.nx, s.ny]));
     D.meadow({ t, heat: (x, y) => (B(x, y) - A(x, y)), vmax: .05, heatA: clamp(t - 1.5) }); D.path(P, P.length, { c: C.ink, a: .9, r: 3 });
     txt('colour = change from this one walk (strongly magnified)', W / 2, 284, { s: 13, c: C.soft, a: clamp(t - 1.5) });
     ring(S0[0], S0[1], C.goldInk, clamp(t - 3)); const d = ripD(); tag(`start ${sg(d.start)}`, S0[0], S0[1], clamp(t - 3), C.ink, { dx: 30, dy: -16, w: 44 }); ring(BELOW[0], BELOW[1], C.ink, clamp(t - 3.6)); tag(`below cat ${sg(d.below)}`, BELOW[0], BELOW[1], clamp(t - 3.6), C.ink, { dy: 18, w: 52 }); tag(`here ${sg(d.here)}`, WALKS[RIP.k][0].x, WALKS[RIP.k][0].y, clamp(t - 2.4), C.ink, { dx: 30, dy: 22, w: 38 });
     card(['shared knobs: one lesson moves every guess', 'TD targets move too: guesses chase guesses', { s: '+ off-policy data = the deadly triad', c: C.roseInk }], t - 5, { y: 296 }); } },
 { h: 'Your turn', ctl: { label: 'recipe ↔ net', values: [0, 1], def: 1, fmt: v => (v ? 'network' : 'linear recipe') },
   get p() { return ['**Tap a spot** on the meadow to teach Pip “this spot is worth 1”: 5 nudges there, step size 0.05, from the trained model. Watch which other spots change with it.', 'Slide to switch between linear recipe and network. The recipe moves near and far spots about equally. The network moves nearby spots more, but far ones too: never only the spot you tapped.']; },
   shot: [3, 9],
   tap(x, y, st) { const wx = (x - FX) / FS, wy = 1 - (y - FY) / FS; if (wx < 0 || wx > 1 || wy < 0 || wy > 1 || D.zone(wx, wy)) return; st.tp = [wx, wy]; },
   draw(t, st) { const kind = st.ci ?? 1, base = kind ? netV(NET().n) : linV(LIN().w), tp = st.tp, V = tp ? nudged(kind, tp[0], tp[1]) : base;
     D.meadow({ t, heat: V }); legend(296, 1);
     if (tp) { ring(tp[0], tp[1], C.ink, 1, 8); const d0 = V(...S0) - base(...S0);
       txt(`tapped spot: ${f2(base(...tp))} → ${f2(V(...tp))}`, W / 2, 322, { s: 16 }); txt(`start: ${f2(base(...S0))} → ${f2(V(...S0))}   ·   below the cat: ${f2(base(...BELOW))} → ${f2(V(...BELOW))}`, W / 2, 346, { s: 15, c: C.soft }); }
     else txt('tap anywhere on the meadow', W / 2, 330, { s: 17, c: C.soft, a: .6 + .4 * Math.sin(t * 3) });
     caption(kind ? 'network (49 knobs)' : 'linear recipe (5 knobs)'); } },
 { h: 'Where you’ve seen this', get p() { return ['In 1992 Gerald Tesauro’s **TD-Gammon** learned backgammon with exactly this mix: a neural network for the values of board positions, trained with TD targets from games against itself. Later versions played close to the best human players.', 'Today almost every large RL system uses a network instead of a table. The next reel, DQN, does it for Q-values and learns Atari games from the screen.']; },
   shot: [3, 10],
   draw(t) { D.meadow({ t, heat: netV(NET().n) }); const P = W1P; walkAlong(P, (t * 1.6) % (P.length * .55 + 1.5), .55, { a: .7 });
     card(['TD-Gammon (1992): network + TD, self-play', 'DQN (2013, 2015): network + Q-learning, Atari', { s: 'tables only for small worlds', c: C.soft }], t - 1.5); } },
 { h: 'Recap', get p() { return ['When states are too many, or never repeat, a table cannot work. **Function approximation** describes a state with features and learns a function with a few knobs that gives a guess for any state.', 'Learning nudges the knobs toward TD targets, a gradient step. Because the knobs are shared, every lesson generalises to similar states, and sometimes spills where it should not.']; },
   shot: [3, 9],
   draw(t) { D.meadow({ t, heat: netV(NET().n) }); D.pipAt(S0[0], S0[1], 1, t);
     card(['describe the spot: features', 'guess = a function of the features, with knobs', 'nudge: knob += step × error × its effect', { s: 'shared knobs: generalise, but spill over', c: C.soft }], t - .3, { gap: 1 }); } },
];

RLStory.register('funcapprox', { title: 'Function Approximation', track: 'Deep RL', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { WALKS, W1, W1R, NSPOT, DUP, DEMO, TRUE, trueAt, LIN, NET, RIP, linV, netV, BELOW, SF, nudged } });
})();
