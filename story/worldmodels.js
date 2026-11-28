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

// World models as a story: in the meadow (deeprl_world.md), Pip learns a small world model from 500 real moves (a move
// predictor and a reward predictor), dreams walks with it, watches dreams drift, trains a Q-network entirely inside the
// dream and tests it in the real meadow. Heavy work runs lazily on first use and is memoised. All numbers from this file.
(function () {
'use strict';
const L = RLStory.lib, D = RLStory.deeprl, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const { px, py, FX, FY, FS } = D, NREAL = 500, H = 15, DREAM = 8000, TEAL = '#3f7f7c', GREY = '#9d9387';
const oneHot = m => [0, 1, 2, 3].map(i => (i === m ? 1 : 0)), mIn = (x, y, m) => D.feats(x, y).concat(oneHot(m));
// ───── real experience: random moves from random starts (meadow rng(seed), move rng(seed + 1)); walks end on cheese, cat or after 60 moves
function collect(N, seed) { const Rm = rng(seed), Ra = rng(seed + 1), data = []; let [x, y] = D.randStart(Rm), t = 0;
  for (let s = 0; s < N; s++) { const m = Math.floor(Ra() * 4), e = D.envStep(x, y, m, Rm); data.push({ x, y, m, nx: e.x, ny: e.y, r: e.r }); t++;
    if (e.done || t >= D.TMAX) { [x, y] = D.randStart(Rm); t = 0; } else { x = e.x; y = e.y; } } return data; }
// move predictor: 4 measurements + 4 move switches → 16 tanh → (Δx, Δy) in units of one move (0.15); Adam, minibatches of 32 moves
function trainMove(data, iters = 3000) { const net = D.newNet(8, 16, 2, 8300), R = rng(8400);
  const items = data.map(d => { const f = mIn(d.x, d.y, d.m); return [{ f, k: 0, target: (d.nx - d.x) / D.STEP }, { f, k: 1, target: (d.ny - d.y) / D.STEP }]; });
  for (let i = 0; i < iters; i++) { const B = []; for (let b = 0; b < 32; b++) B.push(...items[Math.floor(R() * items.length)]); D.batchStep(net, B, .01, 'adam'); } return net; }
// reward predictor: the landing spot's 4 measurements → 8 tanh → reward (target +1, −1 or 0)
function trainRew(data, iters = 8000) { const net = D.newNet(4, 8, 1, 8500), R = rng(8600), items = data.map(d => ({ f: D.feats(d.nx, d.ny), k: 0, target: d.r }));
  for (let i = 0; i < iters; i++) { const B = []; for (let b = 0; b < 32; b++) B.push(items[Math.floor(R() * items.length)]); D.batchStep(net, B, .01, 'adam'); } return net; }
// no clipping here: staying inside the fence must be learned from the data
function predict(mv, x, y, m) { const o = D.fwd(mv, mIn(x, y, m)); return [x + o[0] * D.STEP, y + o[1] * D.STEP]; }
const rewOf = (rw, x, y) => D.fwd(rw, D.feats(x, y))[0];
const dreamR = v => (v > .5 ? 1 : v < -.5 ? -1 : 0);                 // the dream says cheese above 0.5, cat below −0.5
const memo = {};
function base() { if (memo.base) return memo.base; const data = collect(NREAL, 8100), mv = trainMove(data), rw = trainRew(data);
  return (memo.base = { data, mv, rw, nC: data.filter(d => d.r > 0).length, nK: data.filter(d => d.r < 0).length, starts: data.map(d => [d.x, d.y]) }); }
const mvFor = N => memo['mv' + N] ||= (N === NREAL ? base().mv : trainMove(collect(N, 8100)));
// ───── how far dreams drift: 200 random starts, 15 random moves, dream (own predictions chained) vs reality (wobble rng(9200)),
// and a 'perfect average' predictor that knows the exact move but not the wobble
function drift(mv) { const R = rng(9100), Rm = rng(9200), n = 200, d = new Array(H).fill(0), dp = new Array(H).fill(0);
  for (let i = 0; i < n; i++) { let [x, y] = D.randStart(R), ix = x, iy = y, qx = x, qy = y;
    for (let h = 0; h < H; h++) { const m = Math.floor(R() * 4), e = D.envStep(x, y, m, Rm); x = e.x; y = e.y; [ix, iy] = predict(mv, ix, iy, m);
      qx = clamp(qx + D.MOVES[m][0] * D.STEP); qy = clamp(qy + D.MOVES[m][1] * D.STEP); d[h] += Math.hypot(ix - x, iy - y) / n; dp[h] += Math.hypot(qx - x, qy - y) / n; } }
  return { d, dp }; }
const driftFor = N => memo['dr' + N] ||= drift(mvFor(N));
// ───── a fixed move list from the start, played in reality (wobble rng(77)) and in the dream
const PLAN = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0];
function demo(N) { const k = 'demo' + N; if (memo[k]) return memo[k]; const mv = mvFor(N), rw = base().rw, Rm = rng(77);
  let [x, y] = D.START, [ix, iy] = D.START; const P = [[x, y]], Q = [[ix, iy]]; let realEnd = 0, dreamEnd = 0, rEndAt = -1, dEndAt = -1;
  PLAN.forEach((m, i) => { if (!realEnd) { const e = D.envStep(x, y, m, Rm); x = e.x; y = e.y; P.push([x, y]); if (e.r) { realEnd = e.r; rEndAt = i + 1; } }
    if (!dreamEnd) { [ix, iy] = predict(mv, ix, iy, m); Q.push([ix, iy]); const r = dreamR(rewOf(rw, ix, iy)); if (r) { dreamEnd = r; dEndAt = i + 1; } } });
  return (memo[k] = { P, Q, realEnd, dreamEnd, rEndAt, dEndAt }); }
// ───── train a Q-network (4 → 8 → 4, the DQN reel's settings) entirely in the dream. Dream walks start at spots Pip really
// visited, last at most H dreamed moves (cut off, not an ending), end when the reward predictor says cheese or cat.
function dreamDQN(k) { const key = 'dq' + k; if (memo[key]) return memo[key]; const { mv, rw, starts } = base();
  const Ra = rng(6100 + k), net = D.newNet(4, 8, 4, 7000 + k); let tgt = D.copyNet(net); const buf = [], ends = [0, 0, 0], walks = []; let nW = 0, outW = 0, strayed = false;
  let [x, y] = starts[Math.floor(Ra() * starts.length)], t = 0, upd = 0, wk = [[x, y]];
  for (let s = 0; s < DREAM; s++) {
    const f = D.feats(x, y), rnd = Ra() < .2, m = rnd ? Math.floor(Ra() * 4) : D.argmax(Array.from(D.fwd(net, f)), Ra);
    const [nx, ny] = predict(mv, x, y, m), r = dreamR(rewOf(rw, nx, ny)), done = r !== 0, tr = { f, m, r, f2: D.feats(nx, ny), done };
    if (buf.length < 2000) buf.push(tr); else buf[s % 2000] = tr;
    if (s >= 100) { const items = []; for (let b = 0; b < 32; b++) { const q = buf[Math.floor(Ra() * buf.length)]; const nxt = q.done ? 0 : Math.max(...D.fwd(tgt, q.f2)); items.push({ f: q.f, k: q.m, target: q.r + D.GAM * nxt }); }
      D.batchStep(net, items, .003, 'adam'); upd++; if (upd % 100 === 0) tgt = D.copyNet(net); }
    t++; wk.push([nx, ny]); if (nx < 0 || nx > 1 || ny < 0 || ny > 1) strayed = true;
    if (done || t >= H) { ends[r + 1]++; nW++; if (strayed) outW++; if (walks.length < 400) walks.push({ P: wk, end: r, out: strayed }); strayed = false; [x, y] = starts[Math.floor(Ra() * starts.length)]; t = 0; wk = [[x, y]]; } else { x = nx; y = ny; }
  }
  const ev = D.evalNet(net), tests = D.TESTS.map(([a, b], i) => D.greedyWalk(net, a, b, 9000 + i));
  return (memo[key] = { net, ends, walks, ev, tests, nW, outW }); }
const realDQN = k => memo['rq' + k] ||= D.evalNet(D.dqn({ k, steps: NREAL, snapEvery: 1e9 }).net);
// the same 500 real moves fed straight to a Q-network (no model), as many updates as the dream run (7900)
function offline(k) { const key = 'off' + k; if (memo[key]) return memo[key]; const Ra = rng(6200 + k), net = D.newNet(4, 8, 4, 7000 + k); let tgt = D.copyNet(net);
  const buf = base().data.map(d => ({ f: D.feats(d.x, d.y), m: d.m, r: d.r, f2: D.feats(d.nx, d.ny), done: d.r !== 0 }));
  for (let u = 1; u <= DREAM - 100; u++) { const items = []; for (let b = 0; b < 32; b++) { const q = buf[Math.floor(Ra() * buf.length)]; items.push({ f: q.f, k: q.m, target: q.r + (q.done ? 0 : D.GAM * Math.max(...D.fwd(tgt, q.f2))) }); } D.batchStep(net, items, .003, 'adam'); if (u % 100 === 0) tgt = D.copyNet(net); }
  return (memo[key] = D.evalNet(net)); }
// reward-map areas on a 100 × 100 grid: real circles vs where the dream says cheese / cat
function areas() { if (memo.areas) return memo.areas; const rw = base().rw; let tc = 0, pc = 0, tk = 0, pk = 0, fc = 0, fk = 0;
  for (let i = 0; i < 100; i++) for (let j = 0; j < 100; j++) { const x = (i + .5) / 100, y = (j + .5) / 100, z = D.zone(x, y), r = dreamR(rewOf(rw, x, y));
    if (z > 0) tc++; if (z < 0) tk++; if (r > 0) { pc++; if (z <= 0) fc++; } if (r < 0) { pk++; if (z >= 0) fk++; } }
  return (memo.areas = { tc, pc, tk, pk, fc, fk }); }
// one-move error of the learned move predictor on 2000 fresh real moves, and of the perfect average predictor
function oneMove() { if (memo.one) return memo.one; const mv = base().mv, T = collect(2000, 8800); let e = 0, ep = 0;
  T.forEach(d => { const [a, b] = predict(mv, d.x, d.y, d.m); e += Math.hypot(a - d.nx, b - d.ny); ep += Math.hypot(clamp(d.x + D.MOVES[d.m][0] * D.STEP) - d.nx, clamp(d.y + D.MOVES[d.m][1] * D.STEP) - d.ny); });
  return (memo.one = { e: e / T.length, ep: ep / T.length }); }
// catch: run k = 1's test walks that ended on the cat, with the dream's verdict on the landing spot
const KC = 2;
function catFails() { if (memo.cf) return memo.cf; const R1 = dreamDQN(KC), rw = base().rw;
  const f = R1.tests.map((w, i) => ({ i, w })).filter(o => o.w.end < 0).map(o => { const l = o.w.P[o.w.P.length - 1]; return { i: o.i, P: o.w.P, v: rewOf(rw, l[0], l[1]) }; });
  return (memo.cf = f); }

// ───── drawing helpers
const f3 = v => v.toFixed(3), f2 = v => v.toFixed(2);
// where the dream says cheese (gold) or cat (rose): small squares, then the real circles dashed on top
function dreamMap(a = 1, n = 50) { const rw = base().rw, d = FS / n;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const x = (i + .5) / n, y = (j + .5) / n, r = dreamR(rewOf(rw, x, y)); if (r) box(FX + i * d, FY + FS - (j + 1) * d, d + .3, d + .3, r > 0 ? C.gold : C.rose, .55 * a); }
  D.dashCircle(px(D.CHEESE[0]), py(D.CHEESE[1]), D.CR * FS, '#8a5d08', a); D.dashCircle(px(D.CAT[0]), py(D.CAT[1]), D.KR * FS, C.roseInk, a); }
function dashedPath(P, n, col, a = 1) { const c = ctx, m = Math.min(n, P.length); if (m < 1) return; c.save(); c.globalAlpha = a; c.setLineDash([5, 4]); c.strokeStyle = col; c.lineWidth = 1.8; c.beginPath();
  for (let i = 0; i < m; i++) { const x = px(P[i][0]), y = py(P[i][1]); i ? c.lineTo(x, y) : c.moveTo(x, y); } c.stroke(); c.restore(); for (let i = 0; i < m; i++) rcircle(px(P[i][0]), py(P[i][1]), 3, 5000 + i, { c: col, j: .3 }); }
function solidPath(P, n, col, a = 1) { D.path(P, n, { c: col, a, r: 2.8 }); }
// drift chart: x = dreamed steps 1..15, y = average gap 0..0.4
function driftChart(dr, n, o = {}) { const x0 = 70, x1 = 356, y0 = 384, y1 = 306, X = h => x0 + h * (x1 - x0) / (H - 1), YM = o.ymax || .4, Y = v => y0 - Math.min(v, YM) / YM * (y0 - y1), a = o.axA ?? 1;
  alpha(a, () => { rline(x0 - 4, y0, x1 + 4, y0, 960, { c: C.line, w: 1 }); rline(x0 - 4, y0, x0 - 4, y1 - 4, 961, { c: C.line, w: 1 });
    (YM > .4 ? [0, .3, .6] : [0, .15, .3]).forEach((v, i) => { if (v <= YM) txt(String(v), x0 - 9, Y(v), { s: 12, c: C.soft, al: 'right' }); });
    rline(x0 - 4, Y(.15), x1 + 4, Y(.15), 962, { c: '#e6dccd', w: 1, once: true }); txt('one move', x1 + 2, Y(.15) - 8, { s: 12, c: C.soft, al: 'right' });
    txt('1 step', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt('15 dreamed steps', x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
    txt('average gap between dream and reality', x0 - 40, y1 - 14, { s: 14, c: C.soft, al: 'left' }); });
  [[dr.dp, GREY], [dr.d, TEAL]].forEach(([v, col], j) => { if (j === 0 && o.noGrey) return; for (let h = 0; h < Math.min(n, H); h++) { if (h) rline(X(h - 1), Y(v[h - 1]), X(h), Y(v[h]), 970 + h + j * 30, { c: col, w: 1.6, once: true, j: .3 }); dot(X(h), Y(v[h]), 2.8, col); } });
  if (!o.noGrey) alpha(clamp(n / 3), () => { txt('teal: learned model', x0 + 4, y1 + 6, { s: 13, c: TEAL, al: 'left' }); txt('grey: perfect average move', x0 + 4, y1 + 22, { s: 13, c: GREY, al: 'left' }); });
}
const dim = (a = .6) => box(0, 0, W, 285, C.bg, a);
const FN = ['x', 'y', 'to cheese', 'to cat', 'up', 'right', 'down', 'left'];
const NET0 = D.newNet(8, 16, 2, 8300), RNET0 = D.newNet(4, 8, 1, 8500);
function trialsCard(t, y) { const d = [0, 1, 2, 3].map(k => dreamDQN(k).ev.cheese), r = [0, 1, 2, 3].map(realDQN).map(e => e.cheese), o = [0, 1, 2, 3].map(offline).map(e => e.cheese);
  card([`dream-trained, 500 real moves:  ${d.join(', ')}`, `DQN on 500 real moves:  ${r.join(', ')}`, `500 real moves, no model:  ${o.join(', ')}`, { s: 'cheese reached, of 30 test spots, runs 1 to 4', c: C.soft }], t, { y, gap: 1.1, fs: 16, lh: 22 }); }

const STEPS = [
 { h: 'Back in the meadow', p: ['Pip is now in the open meadow. Its position is two numbers, x and y. A move goes 0.15 up, right, down or left, plus a small random wobble. The cheese circle pays +1, the cat circle −1, and either ends the walk. γ (gamma) is 0.9.', 'The DQN reel used 8000 real moves. For a robot, each one costs time and wear.'],
   shot: [4, 10],
   say: [["Welcome back to the open meadow."], [["Pip's position is just two numbers, x and y. Each move goes a little way, plus a random wobble.", [64, 8, 272, 268]], ["Cheese pays plus one, the cat minus one.", [64, 8, 272, 268]]], ["DQN needed eight thousand real moves here. For a robot, every real move costs time and wear."]],
   draw(t) { D.meadow({ t, draw: t / 1.2, labels: clamp(t - 1.5) }); D.pipAt(D.START[0], D.START[1], 1, t);
     if (t > 3.5) card(['move: 0.15 in one direction, plus wobble', 'cheese circle +1 · cat circle −1 · walk ends', { s: 'DQN reel: trained on 8000 real moves', c: C.soft }], t - 3.5, { gap: 1.1 }); } },
 { h: 'Too big for a notebook', p: ['In the model-based reel, Pip noted what each move did in each cell: a learned model, its own copy of the rulebook. Then it practised in its head.', 'Here Pip never stands on the same spot twice, so no notebook can list them all. The fix, as for values in the function approximation reel: a **network** that guesses for spots it has never seen.'],
   shot: [4, 10],
   say: [["In the model-based reel, Pip kept a notebook of what each move did, and practised in its head."], [["But here Pip never stands on the same spot twice. No notebook can list them all.", [64, 8, 272, 268]]], ["The fix is the same as for values. A network, that can guess for spots it has never seen."]],
   draw(t) { const { data } = base(), n = Math.min(data.length, Math.floor(t * 70)); D.meadow({ t });
     for (let i = 0; i < n; i++) dot(px(data[i].x), py(data[i].y), 1.8, TEAL, .6); caption(`${n} spots Pip stood on, no two the same`);
     if (t > 6) card(['a notebook needs the same spot again', { s: 'here: every spot is new', c: C.roseInk }, 'so: a network that generalises'], t - 6, { gap: 1.1 }); } },
 { h: 'A world model', p: ['Pip’s **world model** is two small networks. The move predictor takes Pip’s four measurements and the move, one switch per move, and guesses the change in x and y. The change is nearly the same everywhere, so it is easier to learn than the new position.', 'The reward predictor guesses the reward from the landing spot’s four measurements. Together they are a learned simulator.'],
   shot: [3, 10],
   say: [["Pip's world model is two small networks."], [["The move predictor guesses where a move will land.", [100, 0, 200, 120]], ["The reward predictor guesses what the landing pays.", [100, 182, 200, 110]]], ["Together, they're a learned simulator of the meadow."]],
   draw(t) { D.drawNet(NET0, { x: 120, y: 8, w: 150, h: 150 }, { inL: FN, outL: ['change in x', 'change in y'], a: clamp(t - .3), build: clamp((t - .3) / 2), wa: .6 });
     txt('move predictor', 50, 168, { s: 15, c: TEAL, a: clamp(t - .3), al: 'left' });
     D.drawNet(RNET0, { x: 120, y: 190, w: 150, h: 80 }, { inL: D.FNAMES.map(s => s + '′'), outL: ['reward'], a: clamp(t - 3), build: clamp((t - 3) / 2), wa: .6 });
     txt('reward predictor (landing spot)', 50, 282, { s: 15, c: '#8a5d08', a: clamp(t - 3), al: 'left' });
     if (t > 5.5) card(['spot + move  →  where Pip will land', 'landing spot  →  reward there', { s: 'together: a learned simulator', c: C.soft }], t - 5.5, { y: 300, gap: 1.1 }); } },
 { get p() { const b = base(); return [`First, real experience: Pip wanders with random moves from random spots, ${NREAL} real moves in all. Only ${b.nC} of them landed on the cheese and ${b.nK} on the cat.`, 'Each network is trained as in the function approximation reel: guess, compare with what really happened, nudge every knob a little toward the truth, thousands of times over these same moves.']; },
   h: `Learning from ${NREAL} real moves`, shot: [4, 11],
   get say() { return [[`First, some real experience. Pip wanders at random, ${NREAL} real moves in all.`, ["Watch the dots. Very few land on the cheese or the cat.", [64, 8, 272, 268]]], ["Then each network learns the usual way. Guess, compare with what really happened, and nudge the knobs, thousands of times over."]]; },
   draw(t) { const { data } = base(), n = Math.min(data.length, Math.floor(t * 90)); D.meadow({ t });
     for (let i = 0; i < n; i++) { const d = data[i]; alpha(d.r ? 1 : .45, () => rline(px(d.x), py(d.y), px(d.nx), py(d.ny), 6000 + i, { c: d.r > 0 ? '#8a5d08' : d.r < 0 ? C.roseInk : TEAL, w: d.r ? 2.2 : .8, once: true, j: .2 })); }
     caption(`real moves: ${n} of ${NREAL}`);
     if (t > 6.5) { const b = base(); card([`${NREAL} real moves, random`, `landed on cheese: ${b.nC} · on the cat: ${b.nK}`, { s: 'train both networks on these moves', c: C.soft }], t - 6.5, { gap: 1.1 }); } } },
 { get p() { const o = oneMove(); return [`Teal arrows: the move predictor’s guesses for “right” from a few spots. Grey: the move without wobble. Near the right fence they poke past it: few of the 500 moves showed that bump.`, `On 2000 fresh real moves it is off by ${f3(o.e)} on average, against ${f3(o.ep)} for a perfect predictor of the average move: the wobble cannot be predicted. A move is 0.15.`]; },
   h: 'How good are its moves?', shot: [4, 10],
   say: [["How good are its guesses?"], [["Teal arrows are the predicted moves to the right. Grey is the move without wobble.", [64, 8, 272, 268]], ["Near the right fence, teal pokes through. It rarely saw that bump.", [64, 8, 272, 268]]], ["Elsewhere it's close. The leftover error is mostly the wobble, which nobody can predict."]],
   draw(t) { const mv = base().mv; D.meadow({ t, cat: .4, cheese: .4 }); let i = 0;
     for (const x of [.15, .45, .75, .93]) for (const y of [.15, .4, .9]) { const a = clamp((t - .5 - i++ * .2) * 2); if (a <= 0) continue; const [nx, ny] = predict(mv, x, y, 1), gx = clamp(x + D.STEP);
       alpha(a, () => { rline(px(x), py(y) + 5, px(gx), py(y) + 5, 7000 + i, { c: GREY, w: 1.4, once: true }); arrow(px(x), py(y), px(nx), py(ny), 7100 + i, { c: TEAL, w: 1.8, head: 6 }); dot(px(x), py(y), 3, C.ink); }); }
     if (t > 4.5) { const o = oneMove(); card([`one move, guess vs reality: off by ${f3(o.e)}`, `perfect average move: off by ${f3(o.ep)}`, { s: 'the wobble is the part nobody can predict', c: C.soft }], t - 4.5, { gap: 1.1 }); } } },
 { get p() { const A = areas(), b = base(); return ['Squares show the reward predictor: gold where it says cheese, a guess above 0.5, rose where it says cat, below −0.5.', `Its cheese covers ${Math.round(100 * A.pc / A.tc)} percent of the real circle, its cat only ${Math.round(100 * A.pk / A.tk)} percent. Caveat: “to cheese” and “to cat” are computed from their positions, so the layout comes free. The predictor only learns how near counts, from ${b.nC} cheese and ${b.nK} cat landings.`]; },
   h: 'How good are its rewards?', shot: [3, 10],
   say: [["Now the reward predictor."], [["Gold squares are where it says cheese. Rose, where it says cat.", [64, 8, 272, 268]], ["It covers about two thirds of the cheese circle, but only about a third of the cat.", [64, 8, 272, 268]]], ["It saw very few cat landings to learn from."]],
   draw(t) { D.meadow({ t, cat: 0, cheese: 0 }); dreamMap(clamp(t - .5)); D.pipAt(D.START[0], D.START[1], 1, t);
     if (t > 3.5) { const A = areas(); card([`dream’s cheese: ${Math.round(100 * A.pc / A.tc)}% of the real circle`, `dream’s cat: ${Math.round(100 * A.pk / A.tk)}% of the real circle`, { s: 'squares: the dream · dashed: reality', c: C.soft }], t - 3.5, { gap: 1.1 }); } } },
 { get p() { const d = demo(NREAL); return ['Now Pip can **dream**. From the start it picks a move, asks the move predictor where it lands, then repeats from that guessed spot, with no real step taken.', `Dashed: a dream of ten fixed moves, up four times, then right, then up. Solid: the same moves for real. Reality reaches the cheese after ${d.rEndAt} moves; the dream says after ${d.dEndAt}.`]; },
   h: 'Dreaming', shot: [5, 12],
   say: [["Now Pip can dream. It picks a move, asks the model where it lands, and carries on from there. No real step taken."], [["Dashed is a dream of ten fixed moves. Solid is the same moves, for real.", [64, 8, 272, 268]], ["Watch them drift apart.", [64, 8, 272, 268]]]],
   draw(t) { const d = demo(NREAL), k = clamp(Math.floor((t - .8) / .7) + 1, 1, PLAN.length + 1); D.meadow({ t });
     dashedPath(d.Q, k, TEAL); if (t > 1) solidPath(d.P, k, '#8a5d08', .9);
     const q = d.Q[Math.min(k, d.Q.length) - 1]; alpha(.55, () => L.mouse(px(q[0]), py(q[1]) + 2, 1, .65, t)); const p2 = d.P[Math.min(k, d.P.length) - 1]; D.pipAt(p2[0], p2[1], 1, t);
     txt('dashed: dream', 300, 300, { s: 14, c: TEAL, al: 'left', a: clamp(t - .5) }); txt('solid: reality', 300, 318, { s: 14, c: '#8a5d08', al: 'left', a: clamp(t - 1) });
     card(['moves: up ×4, right ×5, up', `reality: cheese after ${d.rEndAt} moves`, `dream: cheese after ${d.dEndAt} moves`], t - 8, { w: 250, gap: 1.1 }); } },
 { get p() { const r = driftFor(NREAL); return ['From 200 random spots, make 15 random moves, real and dreamed. Teal: the learned model’s average gap. Grey: a perfect predictor of the average move, missing only the wobble.', `After 1 step the gap is ${f3(r.d[0])}; after 15, ${f3(r.d[14])}, against ${f3(r.dp[14])} for grey. Each dream step builds on the last guess, so errors **pile up**. Grey grows too, as the wobble adds up.`]; },
   h: 'Dreams drift', shot: [3, 11],
   say: [["How fast do dreams drift? Make fifteen random moves, real and dreamed, from many spots."], [["Teal is the gap for Pip's model. It grows, step after step.", [50, 296, 320, 110]], ["Each dream step builds on the last guess, so errors pile up.", [50, 296, 320, 110]]], ["Grey, a perfect predictor, grows too, as the wobble adds up."]],
   draw(t) { const r = driftFor(NREAL), n = Math.floor(clamp((t - 2) / 5) * H); D.meadow({ t }); const dm = demo(NREAL); dashedPath(dm.Q, 99, TEAL, .6); solidPath(dm.P, 99, '#8a5d08', .6); dim(.3); driftChart(r, n, { axA: clamp(t) });
     if (n >= H) { const a = clamp(t - 7.5); txt(f3(r.d[14]), 362, 384 - r.d[14] / .4 * 78, { s: 13, c: TEAL, al: 'left', a }); txt(f3(r.dp[14]), 362, 384 - r.dp[14] / .4 * 78 + 6, { s: 13, c: GREY, al: 'left', a }); } } },
 { h: 'Practice in the dream', get p() { return [`Train the DQN reel’s Q-network, same settings, in the dream: ${DREAM} dreamed moves, zero real. Each starts where Pip really was and lasts at most ${H} moves, since long dreams drift.`, `Steps pay 0 until the reward predictor says cheese or cat, ending the dream. Of ${dreamDQN(0).nW} dreams, ${dreamDQN(0).outW} stray past the fence, where real Pip never goes: the move predictor never quite learned it.`]; },
   shot: [4, 11],
   get say() { return [["Now train the DQN network entirely in the dream. Zero real moves."], [`Each dream starts where Pip really was, and lasts at most ${H} moves, since long dreams drift.`, ["Watch the dreams. Some stray past the fence, where real Pip never goes.", [64, 8, 272, 268]]]]; },
   draw(t) { const R0 = dreamDQN(0), n = Math.min(R0.walks.length, Math.floor(t * 4)); D.meadow({ t, cat: .5, cheese: .5 }); dreamMap(.5);
     for (let i = Math.max(0, n - 6); i < n; i++) dashedPath(R0.walks[i].P, 99, R0.walks[i].end > 0 ? '#8a5d08' : R0.walks[i].end < 0 ? C.roseInk : TEAL, i === n - 1 ? 1 : .35);
     caption(`dream ${n}: ${n && R0.walks[n - 1].out ? 'strayed past the fence, ' : ''}${n ? (R0.walks[n - 1].end > 0 ? 'cheese, +1' : R0.walks[n - 1].end < 0 ? 'cat, −1' : 'cut off after 15 moves') : ''}`);
     if (t > 6) card([`${DREAM} dreamed moves, 0 real`, `dreams start at real spots, at most ${H} moves`, { s: 'cut off after 15: not an ending · ε = 0.2', c: C.soft }], t - 6, { gap: 1.1 }); } },
 { get p() { const R0 = dreamDQN(0); return ['Time to wake up. The dream-trained network now steers the real Pip, always taking its best move, from the DQN reel’s 30 test spots.', `It reaches the cheese from ${R0.ev.cheese} of the 30 spots and meets the cat ${R0.ev.cat} times. All it ever saw of the real meadow were those ${NREAL} random moves.`]; },
   h: 'Waking up', shot: [5, 12],
   get say() { return [["Time to wake up. The dream-trained network now steers the real Pip."], [["Watch it set off from the thirty test spots.", [64, 8, 272, 268]], ["Most reach the cheese.", [30, 292, 340, 64]]], [`And all it ever saw of the real meadow were ${NREAL} random moves.`]]; },
   draw(t) { const R0 = dreamDQN(0), n = Math.min(30, Math.floor((t - .5) * 4)); D.meadow({ t });
     for (let i = 0; i < Math.max(0, n); i++) solidPath(R0.tests[i].P, 99, R0.tests[i].end > 0 ? TEAL : C.rose, .55);
     caption(`real test walks: ${Math.max(0, n)} of 30`);
     if (t > 8.5) card([`cheese: ${R0.ev.cheese} of 30 · cat: ${R0.ev.cat}`, `real moves used: ${NREAL}, all random`, { s: 'everything else was learned in the dream', c: C.soft }], t - 8.5, { gap: 1.1 }); } },
 { get p() { const d = [0, 1, 2, 3].map(k => dreamDQN(k).ev.cheese), r = [0, 1, 2, 3].map(k => realDQN(k).cheese), o = [0, 1, 2, 3].map(k => offline(k).cheese); return ['Luck? Four runs, different dice. DQN alone on the 500 real moves, allowing only ' + (NREAL - 100) + ' updates, reaches the cheese from ' + r.join(', ') + ' of the 30 spots. Dream-trained Pip: ' + d.join(', ') + '.', `A Q-network fed those moves directly, with the dream run’s ${DREAM - 100} updates: ${o.join(', ')}. So it is not just more updates: the model fills gaps those few moves left.`]; },
   h: 'A fair test', shot: [4, 10],
   say: [["Was that luck? Four runs, with different dice."], [["Plain DQN, on the same few real moves, does far worse.", [30, 292, 340, 110]], ["So does a network fed those moves directly, with just as many updates.", [30, 292, 340, 110]]], ["It's not just more practice. The model fills the gaps those few moves left."]],
   draw(t) { const R0 = dreamDQN(0); D.meadow({ t }); R0.tests.forEach(w => solidPath(w.P, 99, w.end > 0 ? TEAL : C.rose, .25)); dim(.3); trialsCard(t - .5, 292); } },
 { get p() { const F = catFails(), R1 = dreamDQN(KC), safe = F.filter(f => f.v > -.5).length; const g = F.filter(f => f.v > -.5).map(f => f.v.toFixed(2).replace('-', '−')); return [`The dream’s cat covers only a third of the real one. In run ${KC + 1}, dream-trained Pip met the real cat from ${R1.ev.cat} of 30 spots.`, `At ${safe} of these ${F.length} landings the dream predicted ${g.join(' and ')}: not below −0.5, so grass, and practice never warned against them. At the other ${F.length - safe} it did predict the cat; network mistakes or the wobble did the rest.`]; },
   h: 'The catch: a flawed dream', shot: [4, 11],
   say: [["Here's the catch. The dream's cat is only a third of the real one."], [["Watch where dream-trained Pip meets the real cat.", [64, 8, 272, 268]], ["At some of those spots, the dream said grass. Practice never warned against them.", [64, 8, 272, 268]]], ["The policy inherits the dream's mistakes."]],
   draw(t) { const F = catFails(); D.meadow({ t, cat: 1 }); dreamMap(clamp(t - .5));
     F.forEach((f, j) => { const a = clamp(t - 1.5 - j * .6); if (a > 0) { solidPath(f.P, 99, C.roseInk, a); const l = f.P[f.P.length - 1]; if (f.v > -.5) alpha(a, () => rcircle(px(l[0]), py(l[1]), 6, 7300 + j, { c: C.ink, w: 2.2 })); } });
     if (t > 5) card([`run ${KC + 1}: cat from ${dreamDQN(KC).ev.cat} of 30 spots`, `circled (${catFails().filter(f => f.v > -.5).length}): grass in the dream`, { s: 'rose squares: the dream’s much smaller cat', c: C.soft }], t - 5, { gap: 1.1 }); } },
 { h: 'Your turn', ctl: { label: 'real moves', values: [30, 60, 100, NREAL, 2000], def: 3, fmt: v => String(v) },
   get p() { const a = driftFor(30), b = driftFor(2000); return ['Drag the number of **real moves** the move predictor learns from, and watch its dream of the ten fixed moves and the drift chart. Retraining takes a moment.', `From 30 real moves, the dream is off by ${f2(a.d[14])} after 15 steps, about ${Math.round(a.d[14] / D.STEP)} moves’ length. From 2000, by ${f2(b.d[14])}. More real experience makes the dream more trustworthy, but grey shows a floor no model beats.`]; },
   shot: [3],
   say: [["Your turn. Drag how many real moves the model learns from."], [["Watch the dream of ten moves.", [64, 8, 272, 268]], ["And the drift chart below.", [50, 296, 320, 110]]], ["More real experience makes the dream more trustworthy. But grey shows a floor no model beats."]],
   draw(t, st) { const N = [30, 60, 100, NREAL, 2000][st.ci ?? 3], d = demo(N), r = driftFor(N); D.meadow({ t });
     dashedPath(d.Q, 99, TEAL); solidPath(d.P, 99, '#8a5d08', .9); D.pipAt(d.P[d.P.length - 1][0], d.P[d.P.length - 1][1], 1, t);
     txt(`dream from a model of ${N} real moves`, FX + 6, FY + 12, { s: 13, c: TEAL, al: 'left' }); driftChart(r, H, { ymax: r.d[14] > .38 ? .8 : .4 }); } },
 { h: 'From pictures', p: ['Real agents often see pictures, not positions; predicting every pixel is wasteful. In 2018, David Ha and Jürgen Schmidhuber’s **World Models** squeezed each frame into a short list of numbers, a **latent state**, and learned to predict the next list.', 'In VizDoom, the controller, which picks actions, trained only in that dream yet worked for real. Pip’s four measurements act as its latent state.'],
   shot: [4, 10],
   say: [["Real agents often see pictures, not positions. Predicting every pixel is wasteful."], [["In twenty eighteen, Ha and Schmidhuber squeezed each frame into a short list of numbers, and learned to predict the next list.", [30, 230, 340, 87]]], ["In a Doom game, their controller trained only in the dream, and still worked for real."]],
   draw(t) { const a = clamp(t - .3), b = clamp(t - 1.8), c = clamp(t - 3.3);
     alpha(a, () => { rrect(30, 70, 100, 100, 7500, { fill: '#eef3dc' }); for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) box(34 + i * 16, 74 + j * 16, 14, 14, (i * 7 + j * 3) % 5 ? '#cfd9b5' : '#9cb77a', .8); txt('a frame', 80, 186, { s: 14, c: C.soft }); });
     alpha(b, () => { arrow(140, 120, 182, 120, 7510, { c: C.ink }); [.3, -.8, .1, .6].forEach((v, i) => { rrect(190, 84 + i * 20, 46, 18, 7520 + i, { fill: C.card }); txt(v.toFixed(1), 213, 93 + i * 20, { s: 13 }); }); txt('latent state', 213, 186, { s: 14, c: C.soft }); });
     alpha(c, () => { arrow(244, 120, 282, 120, 7540, { c: TEAL }); [.2, -.7, .3, .5].forEach((v, i) => { rrect(290, 84 + i * 20, 46, 18, 7550 + i, { fill: C.card, c: TEAL }); txt(v.toFixed(1), 313, 93 + i * 20, { s: 13, c: TEAL }); }); txt('predicted next', 313, 186, { s: 14, c: TEAL }); });
     if (t > 5) card(['2018  World Models (Ha, Schmidhuber)', 'frame → short list of numbers → predict next', { s: 'VizDoom: trained in the dream, worked for real', c: C.soft }], t - 5, { y: 230, gap: 1.1 }); } },
 { h: 'Dreamer', p: ['Danijar Hafner and colleagues’ **Dreamer** (2019) learns such a latent model from pixels. It trains an actor, a policy network, and a critic, a value network, on 15-step dreams from real moments, like Pip’s. It keeps collecting real experience and retraining the model, so the dream is checked against reality.', 'DreamerV3 (2023) used one setting for many tasks and collected diamonds in Minecraft from scratch.'],
   shot: [4, 10],
   say: [["Danijar Hafner's Dreamer learns such a model from pixels. Its actor and critic train on short dreams, starting from real moments, like Pip's."], [["It keeps collecting real experience, so the dream is checked against reality.", [30, 292, 340, 87]], ["DreamerV3 even collected diamonds in Minecraft, from scratch.", [30, 292, 340, 87]]]],
   draw(t) { const B = [[200, 50, 'real experience', '#8a5d08'], [320, 200, 'world model', TEAL], [80, 200, 'practice in dreams', C.roseInk]];
     B.forEach(([x, y, s, col], i) => { const a = clamp(t - .3 - i * 1.2); if (a <= 0) return; alpha(a, () => { rrect(x - 66, y - 18, 132, 36, 7600 + i, { fill: C.card, c: col }); txt(s, x, y, { s: 16, c: col }); }); });
     ;[[[245, 72], [300, 178], 'learn'], [[252, 214], [148, 214], 'dream with'], [[100, 178], [155, 72], 'act better']].forEach(([p, q, s], i) => { const a = clamp(t - 1 - i * 1.2); if (a <= 0) return; alpha(a, () => { arrow(p[0], p[1], q[0], q[1], 7620 + i, { c: C.ink, w: 1.4 }); txt(s, (p[0] + q[0]) / 2 + (i === 1 ? 0 : i ? -34 : 34), (p[1] + q[1]) / 2 + (i === 1 ? 14 : 0), { s: 14, c: C.soft }); }); });
     card(['2019  Dreamer: latent model, 15-step dreams', '          collect, retrain, dream again', '2023  DreamerV3: one setting, many tasks', { s: 'Minecraft diamonds from scratch', c: C.soft }], t - 4, { gap: 1.1 }); } },
 { h: 'Recap', p: ['Too big for a notebook? Learn a **world model** with networks: where a move lands and what it pays. Then the policy practises in the dream, almost for free.', 'Keep dreams short, since errors pile up, and check them against reality: the policy inherits the dream’s mistakes. In bigger problems a policy may even exploit the dream’s errors, a known risk not shown here.'],
   shot: [3, 9],
   say: [["So, when the world is too big for a notebook, learn a world model with networks. Where a move lands, and what it pays."], [["Then practise in the dream, almost for free.", [30, 292, 340, 110]], "Keep dreams short, and check them against reality. The policy inherits the dream's mistakes."]],
   draw(t) { const R0 = dreamDQN(0); D.meadow({ t }); R0.tests.slice(0, 10).forEach(w => solidPath(w.P, 99, TEAL, .35)); D.pipAt(D.START[0], D.START[1], 1, t);
     card(['world model: spot + move → landing, reward', 'learn it from a few real moves', 'train the policy in short dreams', { s: 'errors pile up; the policy inherits them', c: C.roseInk }], t - .3, { gap: 1 }); } },
];
RLStory.register('worldmodels', { title: 'World Models & Dreamer', track: 'Model-based', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { base, oneMove, areas, demo, driftFor, dreamDQN, realDQN, offline, catFails } });
})();
