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
// Experience replay as a story: Pip's Q-network in the meadow, first learning from the latest 32 moves in order
// (correlated cards, old lessons overwritten), then from random handfuls drawn out of a shoebox of 2000 cards.
// Every number comes from the seeded runs below (kit DQN settings; frozen copy switched off, see notes/replay.md).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, caption, dot, box, alpha } = L;
const D = RLStory.deeprl;
let ctx;
// ===== drl_story helpers (replay / targetnet / doubledqn) =====
// Background jobs: a training run is a generator; jobs run in small time slices after load (setTimeout), so the page
// stays responsive. need(key) returns the result or null (and moves the job to the front); force(key) finishes it now.
const JOBS = {}, QUEUE = []; let pumping = false;
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
function job(key, make) { if (!JOBS[key]) { JOBS[key] = { make, gen: null, res: null, prog: 0 }; QUEUE.push(key); kick(); } return JOBS[key]; }
function kick() { if (pumping || typeof setTimeout === 'undefined') return; pumping = true; setTimeout(pump, 30); }
function pump() { const t0 = now(); while (QUEUE.length && now() - t0 < 14) { const j = JOBS[QUEUE[0]]; if (!j.gen) j.gen = j.make(); const r = j.gen.next(); if (r.done) { j.res = r.value; QUEUE.shift(); } else j.prog = r.value; }
  if (QUEUE.length) setTimeout(pump, 1); else pumping = false; }
function need(key) { const j = JOBS[key]; if (!j) return null; if (!j.res) { const i = QUEUE.indexOf(key); if (i > 0) { QUEUE.splice(i, 1); QUEUE.unshift(key); } } return j.res; }
function force(key) { const j = JOBS[key]; if (!j.res) { if (!j.gen) j.gen = j.make(); let r; while (!(r = j.gen.next()).done); j.res = r.value; const i = QUEUE.indexOf(key); if (i >= 0) QUEUE.splice(i, 1); } return j.res; }
function progress(keys) { let p = 0; keys.forEach(k => { const j = JOBS[k]; p += j.res ? 1 : j.prog || 0; }); return p / keys.length; }
// DQN on the meadow, exactly as the kit's D.dqn (same seeds, same order of random draws) plus switches and records:
// o.mode 'random' (minibatch drawn from the memory) | 'recent' (the latest B moves, in order); o.cap memory size;
// o.targetEvery (0 = targets from the live network); o.double; o.probe(net, tgt, s) logged every move; o.keep = moves after
// which to keep a copy of the net; o.keepBatch = moves whose minibatch (move numbers) to keep.
function* trainGen(o) {
  const k = o.k ?? 0, steps = o.steps ?? 8000, lr = o.lr ?? .003, B = o.batch ?? 32, cap = o.cap ?? 2000, tEv = o.targetEvery ?? 100, warm = o.warm ?? 100, eps = o.eps ?? .2, mode = o.mode || 'random';
  const Rm = rng(5000 + k), Ra = rng(6000 + k), net = D.newNet(4, 8, 4, 7000 + k); let tgt = tEv ? D.copyNet(net) : null;
  const buf = [], xs = new Float32Array(steps + 1), ys = new Float32Array(steps + 1), lx = new Float32Array(steps), ly = new Float32Array(steps), ms = new Int8Array(steps), rs = new Int8Array(steps), wt = new Uint8Array(steps), use = new Uint16Array(steps);
  const ev = [], snaps = {}, batches = {}, probe = o.probe ? [] : null; let spread = 0, spreadN = 0, upd = 0, copies = [];
  let [x, y] = D.randStart(Rm), t = 0;
  for (let s = 0; s < steps; s++) {
    xs[s] = x; ys[s] = y; wt[s] = t;
    const f = D.feats(x, y), m = Ra() < eps ? Math.floor(Ra() * 4) : D.argmax(Array.from(D.fwd(net, f)), Ra), e = D.envStep(x, y, m, Rm);
    ms[s] = m; rs[s] = e.r; lx[s] = e.x; ly[s] = e.y;
    const tr = { f, m, r: e.r, f2: D.feats(e.x, e.y), done: e.done, s };
    if (buf.length < cap) buf.push(tr); else buf[s % cap] = tr;
    if (s >= warm) {
      let bb; if (mode === 'recent') { bb = []; for (let i = 0; i < B; i++) bb.push(buf[((s - i) % cap + cap) % cap]); } else bb = Array.from({ length: B }, () => buf[Math.floor(Ra() * buf.length)]);
      if (o.keepBatch && o.keepBatch.includes(s)) batches[s] = bb.map(b => b.s);
      if (upd % 25 === 0) { let d = 0, n = 0; for (let i = 0; i < B; i++) for (let j = i + 1; j < B; j++) { d += Math.hypot(xs[bb[i].s] - xs[bb[j].s], ys[bb[i].s] - ys[bb[j].s]); n++; } spread += d / n; spreadN++; }
      for (const b of bb) use[b.s]++;
      const items = bb.map(b => { let next = 0;
        if (!b.done) { const qn = Array.from(D.fwd(tgt || net, b.f2)); next = o.double ? qn[D.argmax(Array.from(D.fwd(net, b.f2)))] : Math.max(...qn); }
        return { f: b.f, k: b.m, target: b.r + D.GAM * next }; });
      D.batchStep(net, items, lr, 'adam'); upd++; if (tEv && upd % tEv === 0) { tgt = D.copyNet(net); copies.push(s); }
    }
    t++; if (e.done || t >= D.TMAX) { [x, y] = D.randStart(Rm); t = 0; } else { x = e.x; y = e.y; }
    if (probe) probe.push(o.probe(net, tgt, s));
    if (o.keep && o.keep.includes(s + 1)) { snaps[s + 1] = D.copyNet(net); if (tgt) snaps['t' + (s + 1)] = D.copyNet(tgt); }
    if ((s + 1) % (o.evEvery || 500) === 0) ev.push(D.evalNet(net).cheese);
    if (s % 40 === 39) yield s / steps;
  }
  xs[steps] = x; ys[steps] = y;
  return { net, xs, ys, lx, ly, ms, rs, wt, use, ev, snaps, batches, probe, copies, spread: spread / spreadN };
}
const maxQ = (n, x, y) => Math.max(...D.qOf(n, x, y));
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−');
const f1 = v => v.toFixed(1);
const fx = v => v.toFixed(2);
// a line chart in a box. o = { x0, x1, y0 (bottom), y1 (top), lo, hi, xmax, yt: [ticks], xl: [left label, right label], title, series: [{ v, col, w, n, a, dx }] }
function chart(o) {
  const X = i => o.x0 + i * (o.x1 - o.x0) / (o.xmax), Y = v => o.y0 - (v - o.lo) / (o.hi - o.lo) * (o.y0 - o.y1);
  rline(o.x0 - 4, o.y0, o.x1 + 4, o.y0, 960, { c: C.line, w: 1, once: true }); rline(o.x0 - 4, o.y0, o.x0 - 4, o.y1 - 2, 961, { c: C.line, w: 1, once: true });
  (o.yt || []).forEach(v => txt(String(v).replace('-', '−'), o.x0 - 9, Y(v), { s: 12, c: C.soft, al: 'right' }));
  if (o.xl) { txt(o.xl[0], o.x0, o.y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(o.xl[1], o.x1, o.y0 + 10, { s: 12, c: C.soft, al: 'right' }); }
  if (o.title) txt(o.title, o.x0 - 8, o.y1 - 11, { s: 13, c: C.soft, al: 'left' });
  (o.series || []).forEach(s => { const n = Math.min(s.n ?? s.v.length, s.v.length); if (n < 1) return; ctx.save(); ctx.globalAlpha = s.a ?? 1; ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 1.4; ctx.lineJoin = 'round'; if (s.dash) ctx.setLineDash(s.dash); ctx.beginPath();
    for (let i = 0; i < n; i++) { const xx = X(i * (s.dx || 1) + (s.x0 || 0)), yy = Y(s.v[i]); if (s.step && i) ctx.lineTo(xx, Y(s.v[i - 1])); i ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); } ctx.stroke(); ctx.restore(); });
  return { X, Y };
}
function busy(p, y = 345) { txt(`Pip is training…  ${Math.round(p * 100)}%`, W / 2, y, { s: 16, c: C.soft }); box(120, y + 14, 160, 5, '#e4d8c7'); box(120, y + 14, 160 * p, 5, C.teal); }
const busyKeys = (keys, y) => { const p = progress(keys); if (p >= 1) return false; busy(p, y); return true; };
function dimScene(a = .62) { box(0, 0, W, 285, C.bg, a); }
// ===== end drl_story helpers =====

const P = [0.85, 0.68], K = [0, 1, 2, 3], W0 = 7074, W1 = 7170, SHOW = 7105, CAPS = [32, 250, 2000, 8000];
const WIN = []; for (let s = W0; s <= W1 + 1; s += 2) WIN.push(s);
const probeP = n => maxQ(n, P[0], P[1]);
// story runs (k = 0): latest 32 in order, and random handfuls from the memory; no frozen copy (targets from the live net)
job('rec0', () => trainGen({ k: 0, mode: 'recent', targetEvery: 0, probe: probeP, keep: WIN.concat([8000]) }));
job('rnd0', () => trainGen({ k: 0, targetEvery: 0, probe: probeP, keep: [8000], keepBatch: [SHOW] }));
job('rec1t', () => trainGen({ k: 1, mode: 'recent', probe: probeP }));   // latest 32 with the frozen copy on (every 100 updates)
K.slice(1).forEach(k => { job('rec' + k, () => trainGen({ k, mode: 'recent', targetEvery: 0 })); job('rnd' + k, () => trainGen({ k, targetEvery: 0 })); });
CAPS.forEach(c => { if (c !== 2000) K.forEach(k => job(`c${c}_${k}`, () => trainGen({ k, cap: c, targetEvery: 0 }))); });
const capKey = (c, k) => c === 2000 ? 'rnd' + k : `c${c}_${k}`;
const RACE = K.map(k => 'rec' + k).concat(K.map(k => 'rnd' + k));
const avgEv = keys => { const rs = keys.map(force); return rs[0].ev.map((_, i) => rs.reduce((a, r) => a + r.ev[i], 0) / rs.length); };
// numbers the story quotes (computed on first use; the background jobs normally have them ready by then)
let NUM = null;
function nums() { if (NUM) return NUM; const a = force('rec0'), b = force('rnd0');
  const lat = []; for (let s = SHOW - 31; s <= SHOW; s++) lat.push(s);
  const md = idx => { let d = 0, n = 0; for (let i = 0; i < idx.length; i++) for (let j = i + 1; j < idx.length; j++) { d += Math.hypot(a.xs[idx[i]] - a.xs[idx[j]], a.ys[idx[i]] - a.ys[idx[j]]); n++; } return d / n; };
  const mdb = idx => { let d = 0, n = 0; for (let i = 0; i < idx.length; i++) for (let j = i + 1; j < idx.length; j++) { d += Math.hypot(b.xs[idx[i]] - b.xs[idx[j]], b.ys[idx[i]] - b.ys[idx[j]]); n++; } return d / n; };
  let far = 9; for (let s = W0; s <= W1; s++) far = Math.min(far, Math.hypot(a.xs[s] - P[0], a.ys[s] - P[1]));
  let xmax = 0; for (let s = W0; s <= W1; s++) xmax = Math.max(xmax, a.xs[s]);
  const sd = arr => { const m = arr.reduce((s, v) => s + v, 0) / arr.length; return [m, Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length)]; };
  const [mA, sA] = sd(a.probe.slice(4000)), [mB, sB] = sd(b.probe.slice(4000));
  let use = 0; for (let s = 2000; s < 6000; s++) use += b.use[s]; use /= 4000;
  const evRec = avgEv(K.map(k => 'rec' + k)), evRnd = avgEv(K.map(k => 'rnd' + k));
  const gw = n => D.greedyWalk(n, D.START[0], D.START[1], 77);
  const gA = gw(a.snaps[8000]), gB = gw(b.snaps[8000]);
  const caps = CAPS.map(c => avgEv(K.map(k => capKey(c, k))));
  let bumps = 0, lefts = 0; const nW = a.snaps[W0]; lat.forEach(s => { if (!a.rs[s] && Math.hypot(a.lx[s] - a.xs[s], a.ly[s] - a.ys[s]) < .07) bumps++; if (D.argmax(D.qOf(nW, a.xs[s], a.ys[s])) === 3) lefts++; });
  const wins = K.filter(k => last(force('rnd' + k).ev) > last(force('rec' + k).ev)).length;
  let ahead = 0; for (let i = evRnd.length - 1; i >= 0; i--) if (evRnd[i] <= evRec[i]) { ahead = (i + 2) * 500; break; } if (!ahead) ahead = 500;
  // with the frozen copy on (run 2): the largest slide of the below-cheese value over 80 moves while Pip stays ≥ 0.45 away
  const c = force('rec1t'); let tb = null; for (let s0 = 2000; s0 + 80 < 8000; s0 += 5) { let far = true; for (let s = s0; s <= s0 + 80; s++) if (Math.hypot(c.xs[s] - P[0], c.ys[s] - P[1]) < .45) { far = false; break; } if (far) { const d = c.probe[s0 + 80] - c.probe[s0]; if (!tb || d < tb.d) tb = { d, s0 }; } }
  return (NUM = { bumps, lefts, wins, ahead, t0: c.probe[tb.s0], t1: c.probe[tb.s0 + 80], tw: 80, a, b, lat, dLat: md(lat), dRnd: mdb(b.batches[SHOW]), spA: a.spread, spB: b.spread, far, xmax, v0: a.probe[W0], v1: a.probe[W1], mA, sA, mB, sB, use, evRec, evRnd, gA, gB, caps }); }
const last = v => v[v.length - 1];
// a card for one experience (move s of run r)
function expCard(r, s, x, y, a = 1, o = {}) { if (a <= 0) return; alpha(a, () => { box(x, y, 84, 58, o.bg || C.card); rrect(x, y, 84, 58, 700 + (s % 37), { c: o.c || C.line, w: 1 });
  txt(`at ${fx(r.xs[s])}, ${fx(r.ys[s])}`, x + 42, y + 11, { s: 13 }); txt(`${D.MN[r.ms[s]]}, reward ${r.rs[s] > 0 ? '+1' : r.rs[s] < 0 ? '−1' : '0'}`, x + 42, y + 28, { s: 13, c: C.soft });
  txt(r.rs[s] ? '→ walk ends' : `→ ${fx(r.lx[s])}, ${fx(r.ly[s])}`, x + 42, y + 45, { s: 13, c: '#3f8582' }); }); }
function heatOf(n) { return (x, y) => maxQ(n, x, y); }
function ring(x, y, a = 1, col = C.teal) { alpha(a, () => { ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(D.px(x), D.py(y), 9, 0, 7); ctx.stroke(); ctx.restore(); }); }
function trailOf(r, s0, s1, o = {}) { const pts = []; for (let s = Math.max(s0, 0); s <= s1; s++) { if (s > s0 && r.wt[s] === 0) pts.length = 0; pts.push([r.xs[s], r.ys[s]]); } D.path(pts, pts.length, o); }
function spots(r, idx, col, a = 1, rad = 3) { idx.forEach(s => dot(D.px(r.xs[s]), D.py(r.ys[s]), rad, col, a)); }
// two small meadows side by side (scaled copies of the kit's meadow)
function mini(tx, ty, s, fn) { ctx.save(); ctx.translate(tx, ty); ctx.scale(s, s); fn(); ctx.restore(); }
function shoebox(x, y, n, t, o = {}) { // box with up to n cards visible as a stack
  rrect(x - 70, y - 30, 140, 60, 820, { fill: '#e9d9bf', w: 1.6 }); rline(x - 78, y - 30, x + 78, y - 30, 821, { w: 1.8 });
  for (let i = 0; i < Math.min(n, 14); i++) { const cx0 = x - 60 + (i % 7) * 17, cy0 = y - 24 + Math.floor(i / 7) * 4; box(cx0, cy0, 14, 22, C.card); rrect(cx0, cy0, 14, 22, 830 + i, { c: C.line, w: .8, once: true }); }
  if (o.label) txt(o.label, x, y + 44, { s: 14, c: C.soft });
}
const BIG = ser => ({ x0: 64, x1: 362, y0: 250, y1: 40, series: ser });
const raceChart = (n, o = {}) => chart(Object.assign({ x0: 60, x1: 352, y0: 388, y1: 318, lo: 0, hi: 30, xmax: 15, yt: [0, 15, 30], xl: ['500 moves', '8000'], title: 'test spots where Pip reaches the cheese (of 30)' }, o));

const STEPS = [
 { h: 'Pip’s network, again', p: ['The DQN reel’s simple version floundered until two fixes came on: a replay memory and a frozen copy for targets. This reel studies the first alone, so the frozen copy stays off: targets come from the network being trained.', 'Same network: four measurements in, one Q-value per move out. Walks start at random spots, last at most 60 moves; one move in five is random.'],
   shot: [3, 9],
   draw(t) { D.meadow({ t, labels: clamp(t - 1) }); D.pipAt(.1, .1, 1, t); const n = D.newNet(4, 8, 4, 7000);
     D.drawNet(n, { x: 130, y: 296, w: 150, h: 92 }, { inL: D.FNAMES, outL: D.MN, a: clamp(t - 2), build: clamp((t - 2) / 2) });
     caption('the network: 4 measurements in, 4 Q-values out', clamp(t - 4)); } },
 { h: 'Learning on the go', p: ['Simplest plan: learn from experience in order. Each move makes one **experience**, a card: where Pip stood, the move, the reward, and where it landed or that the walk ended. That is all a target needs.', 'For fairness later, each update uses the latest 32 cards. After each move the handful slides one card, so each card serves 32 updates in a row.'],
   shot: [4, 11],
   draw(t) { const r = need('rec0'); D.meadow({ t }); if (!r) { busy(progress(['rec0']), 345); return; }
     const s = 7024 + Math.min(Math.floor(t / .8), 18), u = ease((t % .8) / .5);
     trailOf(r, s - 12, s); const x = lerp(r.xs[s - 1], r.xs[s], u), y = lerp(r.ys[s - 1], r.ys[s], u); D.pipAt(x, y, r.xs[s] >= r.xs[s - 1] ? 1 : -1, t);
     for (let i = 0; i < 4; i++) expCard(r, s - 1 - i, 26 + i * 88 + (i === 0 ? 0 : 0), 300, i === 0 ? clamp(u * 2) : 1 - i * .18);
     txt('newest', 68, 370, { s: 13, c: C.soft }); txt('older →', 330, 370, { s: 13, c: C.soft });
     caption(`move ${s} of the run`); } },
 { get p() { const N = nums(), a = N.a, xs = N.lat.map(s => a.xs[s]), ys = N.lat.map(s => a.ys[s]);
     return [`The latest 32 cards, on the meadow: Pip has lingered in the top-left corner, x from ${fx(Math.min(...xs))} to ${fx(Math.max(...xs))}, y from ${fx(Math.min(...ys))} to ${fx(Math.max(...ys))}. ${N.bumps} of the 32 moves bumped the fence; when this stretch began, the network’s favourite move at ${N.lefts === 32 ? 'all 32' : N.lefts + ' of the 32'} spots was left, toward it.`, 'Consecutive experiences are alike, or **correlated**. The network gets nearly the same lesson 32 times.']; },
   h: 'Near-duplicates', shot: [4, 10],
   draw(t) { const r = need('rec0'); D.meadow({ t }); if (!r) { busy(progress(['rec0'])); return; } const N = nums();
     trailOf(r, SHOW - 31, SHOW, { a: .6 }); spots(r, N.lat, C.orange, clamp(t - 1), 3.2); D.pipAt(r.xs[SHOW], r.ys[SHOW], -1, t);
     if (t > 2) txt('the latest 32', D.px(.2), D.py(.45), { s: 15, c: '#c4581f', a: clamp(t - 2), al: 'left' });
     for (let i = 0; i < 4; i++) expCard(r, SHOW - i, 26 + i * 88, 300, clamp(t - 3 - i * .4));
     caption('four of the 32 cards in this handful', clamp(t - 4.5)); } },
 { get p() { const N = nums(); return [`The ringed spot below the cheese is one move from it, worth about 1; the network said ${f2(N.v0)}. Pip never goes near, yet in ${W1 - W0} moves it falls to ${f2(N.v1)}.`, `All spots share 76 knobs, so fitting the corner shifts the cheese’s numbers, uncorrected by any card. Old lessons get **overwritten**. Not mainly the moving target: in run 2, frozen copy on, the same value slid from ${f2(N.t0)} to ${f2(N.t1)} in ${N.tw} moves.`]; },
   h: 'Old lessons get overwritten', shot: [3, 12],
   draw(t) { const r = need('rec0'); if (!r) { D.meadow({ t }); busy(progress(['rec0'])); return; } const N = nums();
     const s = Math.round(lerp(W0, W1, clamp((t - 1.5) / 9))), sn = WIN.reduce((b, v) => v <= s ? v : b, WIN[0]);
     D.meadow({ t, heat: heatOf(r.snaps[sn]) }); trailOf(r, Math.max(W0, s - 14), s, { a: .6 }); D.pipAt(r.xs[s], r.ys[s], 1, t);
     ring(P[0], P[1]); txt(f2(r.probe[s]), D.px(P[0]) + 14, D.py(P[1]) + 2, { s: 15, c: C.goldInk, al: 'left' });
     const c = chart({ x0: 70, x1: 340, y0: 380, y1: 310, lo: 0, hi: 1, xmax: W1 - W0, yt: [0, 0.5, 1], xl: [`move ${W0}`, `move ${W1}`], title: 'best Q-value just below the cheese',
       series: [{ v: Array.from(r.probe.slice(W0, W1 + 1)), col: '#c99a3a', n: s - W0 + 1, w: 1.8 }] });
     txt('gold squares: best Q-value there (for drawing only)', W / 2, 410, { s: 13, c: C.soft, a: clamp(t - 11) }); } },
 { h: 'A shoebox of cards', p: ['The fix is **experience replay**. Every experience card goes into a shoebox, the **replay memory**, and stays there for a while. This box keeps the latest 2000 cards; when it is full, the oldest card is thrown out.', 'So the box holds moves from many walks, all over the meadow.'],
   shot: [4, 10],
   draw(t) { D.meadow({ t }); D.pipAt(.3, .5, 1, t); dimScene(.7);
     const n = Math.min(14, Math.floor(t * 2)); shoebox(W / 2, 170, n, t, { label: 'replay memory: the latest 2000 cards' });
     const k = (t * 1.2) % 1; if (t > .5) { box(W / 2 - 7 + 40 * (1 - k), 60 + k * 90, 14, 22, C.card, 1 - clamp((k - .85) * 6)); rrect(W / 2 - 7 + 40 * (1 - k), 60 + k * 90, 14, 22, 860, { c: C.line, w: .8, once: true }); }
     card(['every move: a new card goes in', 'box full: the oldest card goes out', { s: 'cards stay for 2000 moves', c: C.soft }], t - 2.5, { gap: 1.2 }); } },
 { get p() { const N = nums(); return ['Before each update, Pip draws 32 cards from the box **at random**: a random handful, called a **minibatch** (the DQN reel’s 32 picked at random). One gradient step then fits all 32 at once, on average.',
     `This handful comes from all over the meadow. On average over the run, two cards in a random handful lie ${fx(N.spB)} apart; two of the latest 32 only ${fx(N.spA)}.`]; },
   h: 'A random handful', shot: [3, 9],
   draw(t) { const r = need('rnd0'), q = need('rec0'); D.meadow({ t }); if (!r || !q) { busy(progress(['rec0', 'rnd0'])); return; } const N = nums();
     alpha(.35 * (1 - clamp(t - 1.5)) + .15, () => spots(q, N.lat, C.orange, 1, 2.6));
     const b = r.batches[SHOW]; b.forEach((s, i) => dot(D.px(r.xs[s]), D.py(r.ys[s]), 3.4, C.teal, clamp((t - 1.5) * 3 - i * .08)));
     D.pipAt(r.xs[SHOW], r.ys[SHOW], 1, t);
     card([{ s: 'teal: a random handful of 32 from the box', c: '#3f8582' }, { s: 'faint orange: the latest 32, from before', c: '#c4581f' }, { s: `same moment of training: move ${SHOW}`, c: C.soft }], t - 3, { gap: 1.2 }); } },
 { h: 'A fair race', p: ['Both versions train 8000 moves each, same seeds. Each update uses 32 cards and one gradient step of the same size. Only **which** 32 differs: the latest, or a random handful from the box.', 'Every 500 moves, a test: from 30 fixed spots Pip follows its largest Q, no random moves. The chart counts spots from which it reaches the cheese, out of 30.'],
   shot: [3, 10],
   draw(t) { D.meadow({ t }); dimScene(.3); D.TESTS.forEach(([x, y], i) => dot(D.px(x), D.py(y), 3, C.ink, clamp((t - .5) * 3 - i * .05)));
     if (t > 2.5) txt('the 30 test spots', W / 2, 282, { s: 14, c: C.soft, a: clamp(t - 2.5) });
     alpha(clamp(t - 4), () => raceChart(0)); txt('across: moves of experience', W / 2, 411, { s: 13, c: C.soft, a: clamp(t - 6) }); } },
 { get p() { const N = nums(); return ['Four runs per version: thin lines single runs, thick their average. Teal: random handfuls; orange: the latest 32. Single runs jump a lot.', `After 8000 moves, replay reaches the cheese from ${f1(last(N.evRnd))} of 30 spots on average, the latest 32 from ${f1(last(N.evRec))}. Replay won ${N.wins} of the 4 runs, and its average ${N.ahead === 500 ? 'is ahead at every test' : `stays ahead from move ${N.ahead} on`}. Same work; only the cards differ.`]; },
   h: 'The race', shot: [5, 13],
   draw(t) { if (busyKeys(RACE)) return; const N = nums(), n = clamp(Math.floor((t - .5) * 2.5), 0, 16);
     const ser = []; K.forEach(k => { ser.push({ v: force('rec' + k).ev, col: C.orange, w: 1, a: .35, n }); ser.push({ v: force('rnd' + k).ev, col: C.teal, w: 1, a: .35, n }); });
     ser.push({ v: N.evRec, col: '#c4581f', w: 2.6, n }, { v: N.evRnd, col: '#3f8582', w: 2.6, n }); const c = raceChart(0, BIG(ser));
     if (n >= 16) { txt(`replay: ${f1(last(N.evRnd))}`, 362, c.Y(last(N.evRnd)) - 14, { s: 15, c: '#3f8582', al: 'right', a: clamp(t - 7.2) }); txt(`latest 32: ${f1(last(N.evRec))}`, 362, c.Y(last(N.evRec)) - 30, { s: 15, c: '#c4581f', al: 'right', a: clamp(t - 7.6) }); }
     card([{ s: 'teal: random handfuls from the box', c: '#3f8582' }, { s: 'orange: the latest 32 moves', c: '#c4581f' }, { s: 'thin: single runs · thick: average of 4', c: C.soft }], t - 1, { y: 300, gap: 1.1 }); } },
 { get p() { const N = nums(); return [`A random handful samples the whole meadow fairly, so each step nudges everywhere a little rather than bending toward one corner. Below the cheese, over the last 4000 moves, the replay run swings by about ${f2(N.sB)}, the latest-32 run by ${f2(N.sA)}.`, `Both use each card about ${Math.round(N.use)} times; replay spreads them over 2000 moves, mixed with other cards, instead of 32 updates in a row.`]; },
   h: 'Why the shuffle helps', shot: [4, 12],
   draw(t) { const r = need('rnd0'), q = need('rec0'); D.meadow({ t, heat: r ? heatOf(r.snaps[8000]) : null, heatA: .6 }); if (!r || !q) { busy(progress(['rec0', 'rnd0'])); return; } const N = nums();
     ring(P[0], P[1]); const n = clamp(Math.floor((t - 1) * 600), 0, 4000);
     chart({ x0: 70, x1: 340, y0: 384, y1: 312, lo: -.2, hi: 1.2, xmax: 4000, yt: [0, 1], xl: ['move 4000', '8000'], title: 'best Q-value just below the cheese (run 1)',
       series: [{ v: Array.from(q.probe.slice(4000)), col: C.orange, w: 1, n }, { v: Array.from(r.probe.slice(4000)), col: '#3f8582', w: 1.3, n }] });
     txt('teal: replay', 120, 405, { s: 13, c: '#3f8582' }); txt('orange: latest 32', 280, 405, { s: 13, c: '#c4581f' }); } },
 { get p() { const N = nums(), sa = N.gA, sb = N.gB, say = g => g.end > 0 ? `reaches the cheese in ${g.P.length - 1} moves` : g.end < 0 ? `runs into the cat after ${g.P.length - 1} moves` : 'never gets there in 60 moves';
     return ['Run 1’s two networks at 8000 moves, mapped by largest Q: left replay, right the latest 32. Gold good, rose bad.', `Following the map from the start, the replay Pip ${say(sa)}, along the bottom and up the right side; the left-side route of the DQN reel is about as long. The other one ${say(sb)}.`]; },
   h: 'Two maps', shot: [4, 11],
   draw(t) { const r = need('rnd0'), q = need('rec0'); if (!r || !q) { busy(progress(['rec0', 'rnd0']), 150); return; } const N = nums(), s = .72;
     [[r, N.gA, 12, 'replay', '#3f8582'], [q, N.gB, 204, 'latest 32', '#c4581f']].forEach(([run, g, x0, lab, col], i) => {
       mini(x0 - D.FX * s, 46 - D.FY * s, s, () => { D.meadow({ t, heat: heatOf(run.snaps[8000]), tufts: .5 }); if (t > 3) D.path(g.P, Math.floor((t - 3) * 6), { r: 3 }); D.pipAt(D.START[0], D.START[1], 1, t); });
       txt(lab, x0 + 92, 30, { s: 16, c: col }); });
     card(['maps: the largest Q-value at each spot', 'teal dots: Pip following its largest Q-value', { s: 'run 1, after 8000 moves', c: C.soft }], t - 1, { y: 252, gap: 1.2 }); } },
 { h: 'Old cards, old habits', p: ['The box’s cards came from an **older** Pip, with older Q-values and habits. DQN doesn’t mind: its target, as in Q-learning, asks the worth of the **best** move from the landing spot, whatever habit made the card. Any card gives a fair target.', 'SARSA-like methods, learning about Pip’s current habit, would study a Pip that no longer exists, so they cannot simply reuse old cards.'],
   shot: [3, 10],
   draw(t) { D.meadow({ t }); D.pipAt(.5, .3, 1, t); dimScene(.72); shoebox(W / 2, 150, 14, t, { label: 'cards from moves 6000 … 8000' });
     card(['DQN target: reward + 0.9 × best Q there', { s: '  → any card gives a fair target', c: C.goldInk }, 'SARSA: learns about today’s habit', { s: '  → old cards describe an old Pip', c: C.roseInk }], t - 1.5, { gap: 1.2, fs: 16, lh: 22 }); } },
 { h: 'What about the deadly triad?', p: ['The funcapprox reel warned: a network, plus targets built from guesses, plus data from another habit can blow up. Replay has all three: old cards come from older Pips.', 'Nothing guarantees safety, but here it stayed calm, likely because the box holds only the latest 2000 moves, so its cards come from Pips much like today’s. The next reel’s frozen copy adds more safety.'],
   shot: [3, 9],
   draw(t) { D.meadow({ t }); D.pipAt(.5, .3, 1, t); dimScene(.72);
     card(['network: shared knobs  ✓', 'targets from guesses (bootstrapping)  ✓', 'cards from other habits (off-policy)  ✓', { s: 'here: stable, the box only holds recent moves', c: C.goldInk }], t - .8, { y: 80, gap: 1.2, fs: 16 }); } },
{ get p() { const N = nums(); return ['Drag the **memory size** (four runs each). With 32 cards even a random handful is a near-duplicate; a few hundred already work much better.',
     `Keeping all 8000 was worse here (${f1(last(N.caps[3]))} against ${f1(last(N.caps[2]))}). Every card gives a fair target, but the box decides **which spots** get lessons. Likely: most cards then come from early walks, in places today’s Pip rarely visits.`]; },
   h: 'Your turn', ctl: { label: 'memory', values: CAPS, def: 2, fmt: v => v + ' cards', onChange: st => {} }, shot: [3, 9],
   draw(t, st) { const c = CAPS[st.ci ?? 2], keys = K.map(k => capKey(c, k));
     const ready = keys.every(k => need(k)); if (!ready) { busy(progress(keys), 140); return; }
     const n = clamp(Math.floor((t - (st.changed ?? 0) - .2) * 6), 0, 16), av = avgEv(keys), ser = keys.map(k => ({ v: force(k).ev, col: C.teal, w: 1, a: .35, n }));
     ser.push({ v: av, col: '#3f8582', w: 2.6, n }); raceChart(0, BIG(ser));
     shoebox(W / 2, 318, Math.max(1, Math.round(14 * Math.log(c / 16) / Math.log(500))), t, {});
     txt(`memory: the latest ${c} moves`, W / 2, 364, { s: 15, c: C.soft });
     txt(n >= 16 ? `average after 8000 moves: ${f1(last(av))} of 30` : ' ', W / 2, 392, { s: 16, c: '#3f8582' }); } },
 { h: 'Where it came from', p: ['Long-Ji Lin proposed experience replay in 1992. DQN made it famous in 2015: its memory held the latest one million moves, and each update used a random handful of 32. Without replay, DQN’s Breakout score dropped from about 317 to about 10.', 'One upgrade, **prioritised replay** (Schaul and others, 2016), draws the surprising cards, those with a big TD error, more often.'],
   shot: [3, 9],
   draw(t) { D.meadow({ t }); D.pipAt(.1, .1, 1, t); dimScene(.6);
     card(['1992  Lin: experience replay', '2015  DQN: 1 million cards, handfuls of 32', '2016  prioritised replay: surprises first', { s: 'Breakout: about 317 with replay, 10 without', c: C.soft }], t - .5, { y: 100, gap: 1.2 }); } },
 { h: 'Recap', p: ['Moves that follow each other are near-duplicates, and fitting the latest ones bends the shared knobs toward one corner, overwriting old lessons.', '**Experience replay** keeps the latest moves as cards in a memory and learns from random handfuls: varied, fair samples of the whole meadow, each card reused many times.'],
   shot: [3, 9],
   draw(t) { const r = need('rnd0'); D.meadow({ t, heat: r ? heatOf(r.snaps[8000]) : null, heatA: .5 }); D.pipAt(.1, .1, 1, t);
     card(['every move: a card into the memory', 'every update: 32 cards drawn at random', 'random handful: varied, less forgotten', { s: 'fine for DQN, which learns about the best move', c: C.soft }], t - .3, { gap: 1 }); } },
];
RLStory.register('replay', { title: 'Experience Replay', track: 'Deep RL', onCtx: c => { ctx = c; }, steps: STEPS, _sim: { nums, force, JOBS } });
})();
