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
// Double DQN as a story: a max over noisy guesses leans high (plain numbers first), DQN's target takes exactly such a max,
// and Double DQN lets the live network choose the move and the frozen copy score it. Every number comes from the seeded
// simulations below (plain-number rounds: rng(2024); meadow runs: kit DQN settings, k = 0…7; see notes/doubledqn.md).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, gauss, rline, rrect, txt, arrow, card, caption, dot, box, alpha } = L;
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

// ───── plain numbers: moves all truly worth 0, each guess = 0 + noise (standard deviation SIG)
const SIG = 0.1, NR = 1000, MOVES_N = [2, 4, 8, 16];
function rounds(n, seed = 2024) { const R = rng(seed + n), out = []; for (let i = 0; i < NR; i++) out.push({ a: Array.from({ length: n }, () => SIG * gauss(R)), b: Array.from({ length: n }, () => SIG * gauss(R)) }); return out; }
const RMEMO = {}, RND = n => RMEMO[n] || (RMEMO[n] = (() => { const rs = rounds(n); const am = v => v.indexOf(Math.max(...v));
  let one = 0, mx = 0, dbl = 0, pos = 0; rs.forEach(r => { one += r.a[0]; const m = Math.max(...r.a); mx += m; pos += m > 0; dbl += r.b[am(r.a)]; });
  return { rs, one: one / NR, max: mx / NR, dbl: dbl / NR, pos, am }; })());
const R4 = RND(4);
const sg = v => Math.abs(v) < .005 ? '0.00' : (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2);
// ───── meadow runs: DQN and Double DQN (replay 2000, copy every 100 updates), 8 runs each
const KS = [0, 1, 2, 3, 4, 5, 6, 7], KEEP = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], MID = [2050, 3050, 4050, 5050, 6050, 7050];  // MID: halfway between two refreshes of the copy
KS.forEach(k => { job('q' + k, () => trainGen({ k, keep: KEEP })); job('d' + k, () => trainGen({ k, double: true, keep: k ? KEEP : KEEP.concat(MID) })); });
QUEUE.sort((a, b) => +a.slice(1) - +b.slice(1) || (a < b ? -1 : 1));
// the gap at a snapshot: average over the 30 test spots of (largest Q-value) − (average discounted return of 5 greedy walks from there)
const GAPM = {};
function gapOf(key, s) { const id = key + ':' + s; if (GAPM[id]) return GAPM[id]; const n = force(key).snaps[s]; let est = 0, ret = 0;
  D.TESTS.forEach(([x, y], i) => { est += maxQ(n, x, y); let r = 0; for (let j = 0; j < 5; j++) r += D.greedyWalk(n, x, y, 9000 + i * 10 + j).ret; ret += r / 5; });
  return (GAPM[id] = { est: est / 30, ret: ret / 30, gap: (est - ret) / 30 }); }
const avg = a => a.reduce((s, v) => s + v, 0) / a.length;
let NUM = null;
function nums() { if (NUM) return NUM;
  const gq = KEEP.map(s => avg(KS.map(k => gapOf('q' + k, s).gap))), gd = KEEP.map(s => avg(KS.map(k => gapOf('d' + k, s).gap)));
  const evq = avg(KS.map(k => last(force('q' + k).ev))), evd = avg(KS.map(k => last(force('d' + k).ev)));
  // one real landing spot where the live network and the copy disagree (Double DQN run 1, after 4000 moves)
  // searched over a grid of spots (outside the cheese and cat circles) and snapshots taken 50 updates after a refresh
  const r = force('d0'); let best = null;
  MID.forEach(s => { const live = r.snaps[s], cp = r.snaps['t' + s]; for (let i = 1; i < 20; i++) for (let j = 1; j < 20; j++) { const x = i / 20, y = j / 20; if (D.zone(x, y)) continue;
    const ql = D.qOf(live, x, y), qc = D.qOf(cp, x, y), ml = D.argmax(ql), mc = D.argmax(qc); if (ml === mc) continue; const d = qc[mc] - qc[ml]; if (!best || d > best.d) best = { x, y, s, ql, qc, ml, mc, d }; } });
  // reality check for the card: a hand-made route (up while y < 0.78, then right), 500 walks per first move (wobble rng(70000 + i))
  const route = [0, 2].map(m0 => { let tot = 0; for (let i = 0; i < 500; i++) { const R = rng(70000 + i); let e = D.envStep(best.x, best.y, m0, R), g = 1, ret = 0;
    for (let t = 0; t < D.TMAX; t++) { if (e.done) { ret = e.r * g; break; } g *= D.GAM; e = D.envStep(e.x, e.y, e.y < .78 ? 0 : 1, R); } tot += ret; } return tot / 500; });
  return (NUM = { route, absd: avg(gd.map(Math.abs)), gq, gd, mq: avg(gq), md: avg(gd), evq, evd, card: best }); }
const last = v => v[v.length - 1];
const QK = KS.map(k => 'q' + k), DK = KS.map(k => 'd' + k);

// ───── drawing
const MN = D.MN, NETS = [D.newNet(4, 8, 4, 7000), D.newNet(4, 8, 4, 7001)];
function guessRow(vals, x, y, o = {}) { // four (or n) guesses in a row of small boxes; o.hi = index highlighted, o.col
  const n = vals.length, w = o.w || 56;
  vals.forEach((v, i) => { const bx = x + i * (w + 6); const a = o.a ? o.a(i) : 1; if (a <= 0) return;
    alpha(a, () => { box(bx, y, w, 26, o.hi === i ? (o.hcol || C.yellow) : C.card, o.hi === i ? .8 : 1); rrect(bx, y, w, 26, 900 + i + Math.round(y), { c: C.line, w: 1 });
      if (w >= 30) txt(sg(v), bx + w / 2, y + 13, { s: 15, c: v > .005 ? C.goldInk : v < -.005 ? C.roseInk : C.soft }); }); });
}
function labels(x, y, w = 56, a = 1) { MN.forEach((m, i) => txt(m, x + i * (w + 6) + w / 2, y, { s: 13, c: C.soft, a })); }
function zeroBar(x0, x1, y0, y1, vals, cols, labs, o = {}) { // horizontal bars of averages around a zero line
  const xm = (x0 + x1) / 2, sc = (x1 - x0) / 2 / (o.max || .2), yl = y0 + (vals.length - 1) * (o.gap || 30); rline(xm, y0 - 12, xm, yl + 12, 990, { c: C.line, w: 1 }); txt('0', xm, yl + 22, { s: 12, c: C.soft });
  vals.forEach((v, i) => { const y = y0 + i * (o.gap || 30); const bw = v * sc * (o.u ?? 1); box(Math.min(xm, xm + bw), y - 8, Math.abs(bw), 16, cols[i]); txt(labs[i], x0 - 6, y, { s: 14, al: 'right', c: C.ink }); txt(sg(v * (o.u ?? 1)), xm + bw + (v >= 0 ? 6 : -6), y, { s: 14, al: v >= 0 ? 'left' : 'right', c: v >= 0 ? C.goldInk : C.roseInk }); });
}
function spotScene(t, a = 1) { if (a <= 0) return; D.meadow({ t }); alpha(a, () => { dot(D.px(.3), D.py(.45), 5, C.ink); txt('a spot', D.px(.3), D.py(.45) - 15, { s: 14 }); }); }
const gapChart = (ser, o = {}) => chart(Object.assign({ x0: 64, x1: 356, y0: 250, y1: 40, lo: -.1, hi: .3, xmax: 7, yt: [-.1, 0, .1, .2, .3], xl: ['after 1000 moves', '8000'], series: ser }, o));

const STEPS = [
 { h: 'The max in the target', p: ['Pip now trains the full DQN: replay memory, plus the frozen copy refreshed every 100 updates. A card’s target is the reward plus 0.9 × the **largest** of the copy’s four Q-values at the landing spot.', 'That word “largest” hides a problem. To see it, leave the meadow for a moment and play with plain numbers.'],
   shot: [3, 9],
   draw(t) { D.meadow({ t }); D.pipAt(.1, .1, 1, t); card(['target = reward + 0.9 × largest Q of the copy', { s: '                                   ↑ this max', c: C.roseInk }], t - 2, { gap: 1.4 }); } },
 { h: 'Four moves worth nothing', p: [`Imagine a spot where all four moves are truly worth exactly 0. Pip does not know that; its guesses are 0 plus a bell-shaped random error, typically about ${SIG} either way. One round of guesses: ${R4.rs[0].a.map(sg).join(', ')}.`, `The largest guess is ${sg(Math.max(...R4.rs[0].a))}. If the target uses the largest, it says this spot is worth ${sg(Math.max(...R4.rs[0].a))}, not 0.`],
   shot: [3, 10],
   draw(t) { spotScene(t, 1 - clamp(t - .2)); labels(84, 112, 56, clamp(t - .5)); guessRow(R4.rs[0].a, 84 - 28 + 28, 124, { a: i => clamp(t - 1 - i * .5), hi: t > 4 ? R4.am(R4.rs[0].a) : -1 });
     txt('truly worth: 0, 0, 0, 0', W / 2, 90, { s: 16, c: C.soft, a: clamp(t - .3) });
     if (t > 4.5) txt(`largest guess: ${sg(Math.max(...R4.rs[0].a))}`, W / 2, 180, { s: 18, c: C.goldInk, a: clamp(t - 4.5) });
     card(['true value of every move: 0', `guess = 0 + random error (typically ${SIG})`, { s: 'the max picks whichever guess came out highest', c: C.soft }], t - 6, { gap: 1.2 }); } },
 { h: 'Roll again', p: [`New round, new errors. Each yellow box is that round’s largest guess. A single guess is right on average: over ${NR} rounds, the guesses for “up” average ${sg(R4.one)}.`, `But the largest guess was above 0 in ${R4.pos} of ${NR} rounds, averaging ${sg(R4.max)}. The max keeps picking the lucky errors. An error that always leans one way is a **bias**: here an **overestimation bias**.`],
   shot: [4, 12],
   draw(t) {  labels(84, 30); for (let i = 0; i < 8; i++) { const r = R4.rs[i]; guessRow(r.a, 84, 42 + i * 30, { a: () => clamp(t - .3 - i * .7), hi: R4.am(r.a) }); }
     if (t > 6.5) zeroBar(150, 330, 310, 380, [R4.one, R4.max], [C.teal, C.yellow], ['one guess', 'the largest'], { u: clamp((t - 6.5) / 1.5), max: .15 });
     if (t > 6.5) txt(`averages over ${NR} rounds`, W / 2, 292, { s: 14, c: C.soft, a: clamp(t - 6.5) }); } },
 { h: 'It spreads backwards', p: ['In Q-learning and DQN the target is reward + 0.9 × the largest Q-value where Pip lands. Those Q-values are noisy: the network is still partly wrong, and the wobble adds chance. So the target leans high.', 'It does not stay there. The spot one move earlier learns from that inflated number, the spot before from that one: the optimism travels back along Pip’s routes.'],
   shot: [3, 10],
   draw(t) { D.meadow({ t }); const pts = [[.15, .2], [.3, .2], [.45, .2], [.6, .2], [.75, .2]];
     pts.forEach(([x, y], i) => { dot(D.px(x), D.py(y), 5, C.ink); if (i) arrow(D.px(pts[i - 1][0]) + 7, D.py(y), D.px(x) - 7, D.py(y), 1800 + i, { c: C.soft, w: 1.3 }); });
     for (let i = 4; i >= 0; i--) { const a = clamp(t - 1 - (4 - i) * 1.1); if (a <= 0) continue; txt('too high', D.px(pts[i][0]), D.py(pts[i][1]) - 18 - (i % 2) * 14, { s: 13, c: C.roseInk, a }); }
     card(['noisy guesses at the landing spot', '→ max leans high → target too high', { s: '→ the spot before learns it, and so on back', c: C.roseInk }], t - 1.5, { gap: 1.3 }); } },
 { h: 'Two opinions', p: ['The fix needs two independent sets of guesses, A and B. Let A **choose** the move: the one with A’s largest guess. Then let B **score** that move, with B’s guess for it.', 'B’s error has nothing to do with A’s, so B’s number is just 0 plus a fresh random error: no longer lucky on purpose.'],
   shot: [4, 11],
   draw(t) {  labels(84, 40); const r = R4.rs[0], m = R4.am(r.a);
     txt('A', 60, 66, { s: 18, c: C.goldInk }); guessRow(r.a, 84, 53, { hi: t > 2 ? m : -1 }); if (t > 2) txt('A chooses: ' + MN[m], W / 2, 102, { s: 16, c: C.goldInk, a: clamp(t - 2) });
     if (t > 3.5) { txt('B', 60, 138, { s: 18, c: '#3f8582', a: clamp(t - 3.5) }); guessRow(r.b, 84, 125, { a: () => clamp(t - 3.5), hi: t > 5 ? m : -1, hcol: '#bfe0de' }); }
     if (t > 5.5) txt(`B scores ${MN[m]}: ${sg(r.b[m])}`, W / 2, 174, { s: 18, c: '#3f8582', a: clamp(t - 5.5) });
     card([`largest of A: ${sg(r.a[m])}  (lucky)`, `B’s guess for the same move: ${sg(r.b[m])}`, { s: 'choose with one set, score with the other', c: C.soft }], t - 6.5, { gap: 1.2 }); } },
 { h: 'Back to zero', p: [`Over the same ${NR} rounds, the largest guess averaged ${sg(R4.max)}. Choosing with A and scoring with B averaged ${sg(R4.dbl)}: the bias is gone.`, 'This trick is called **Double Q-learning**. Hado van Hasselt proposed it in 2010 for tables of Q-values, keeping two tables and letting each pick moves for the other.'],
   shot: [3, 9],
   draw(t) {  txt(`averages over ${NR} rounds, all moves truly worth 0`, W / 2, 70, { s: 15, c: C.soft });
     zeroBar(150, 330, 120, 190, [R4.one, R4.max, R4.dbl], [C.teal, C.yellow, '#9fc9c7'], ['one guess', 'the largest', 'A picks, B scores'], { u: clamp((t - .5) / 1.5), max: .15, gap: 40 });
     card(['the max of noisy guesses: too high', 'pick with one, score with the other: fair', { s: 'Double Q-learning, van Hasselt 2010', c: C.soft }], t - 3, { gap: 1.2 }); } },
 { h: 'DQN already has two', p: ['DQN already has two networks, live and frozen copy. **Double DQN**, by van Hasselt, Guez and Silver, 2015, uses them as A and B: at the landing spot the live one chooses the move, the copy scores it. Nothing else changes.', 'The copy lags the live one by up to 100 updates, so their errors are only **partly** independent. The bias shrinks, maybe not fully.'],
   shot: [3, 10],
   draw(t) { D.meadow({ t }); dimScene(.65);
     card(['DQN:', '   target = reward + 0.9 × copy’s largest Q', 'Double DQN:', { s: '   live network chooses the move m', c: C.goldInk }, { s: '   target = reward + 0.9 × copy’s Q for m', c: '#3f8582' }], t - .5, { y: 60, gap: 1.2 });
     const a = clamp(t - 6); if (a > 0) { D.drawNet(NETS[0], { x: 50, y: 306, w: 110, h: 76 }, { a, wa: .8 }); txt('live: chooses', 105, 296, { s: 14, c: C.goldInk, a });
       D.drawNet(NETS[1], { x: 240, y: 306, w: 110, h: 76 }, { a: a * .8, wa: .5 }); txt('copy: scores', 295, 296, { s: 14, c: '#3f8582', a }); } } },
 { get h() { return 'One real card'; }, get p() { const N = nums(), c = N.card; return [`At the corner (${fx(c.x)}, ${fx(c.y)}), Double DQN run, ${c.s} moves in, the copy favours “${MN[c.mc]}”, ${f2(c.qc[c.mc])}; the live network prefers “${MN[c.ml]}”, bumping the fence. DQN would use ${f2(c.qc[c.mc])}; Double DQN, the copy’s “${MN[c.ml]}”, ${f2(c.qc[c.ml])}.`,
     `On a simple route, up the left and along the top, “up” first is worth about ${f2(N.route[0])}, “down” first ${f2(N.route[1])}. DQN was about right; Double DQN **underestimates**: it can err low too.`]; },
   shot: [4, 11],
   draw(t) { if (busyKeys(['d0'])) return; const N = nums(), c = N.card; D.meadow({ t }); dot(D.px(c.x), D.py(c.y), 5, C.ink); txt('landed here', D.px(c.x), D.py(c.y) - 16, { s: 14 });
     labels(84, 296); txt('live', 50, 321, { s: 15, c: C.goldInk, al: 'right' }); guessRow(c.ql, 84, 308, { hi: t > 1.5 ? c.ml : -1 });
     txt('copy', 50, 357, { s: 15, c: '#3f8582', al: 'right' }); guessRow(c.qc, 84, 344, { a: () => clamp(t - 2.5), hi: t > 4 ? c.ml : -1, hcol: '#bfe0de' });
     if (t > 5) txt(`DQN: ${f2(c.qc[c.mc])}   ·   Double DQN: ${f2(c.qc[c.ml])}`, W / 2, 390, { s: 16, a: clamp(t - 5) });
     if (t > 7) txt(`simple route: up first ${f2(N.route[0])}, down first ${f2(N.route[1])}`, W / 2, 410, { s: 14, c: C.soft, a: clamp(t - 7) }); } },
 { h: 'Estimates against reality', p: ['Does it matter in the meadow? At each of the 30 test spots, compare the network’s largest Q-value, its **estimate**, with what Pip really gets: the average discounted return of 5 walks from there following its largest Q-values.', 'The chart shows estimate minus reality, averaged over the spots and 8 runs, so the chance in single walks mostly averages out. Above 0: too optimistic.'],
   shot: [3, 10],
   draw(t) { D.meadow({ t }); dimScene(.3); D.TESTS.forEach(([x, y], i) => dot(D.px(x), D.py(y), 3, C.ink, clamp((t - .5) * 3 - i * .05)));
     card(['estimate: largest Q-value at the spot', 'reality: average return of 5 real walks', { s: 'gap = estimate − reality   (> 0: optimistic)', c: C.goldInk }], t - 2.5, { gap: 1.3 }); } },
 { get p() { const N = nums(); return [`Eight runs each, averaged. Orange, DQN: ${f2(N.gq[0])} too high after 1000 moves, ${f2(N.mq)} averaged over eight checks. Teal, Double DQN: ${f2(N.gd[0])} at first, then near 0, sometimes below: ${f2(N.md)} on average.`,
     `The gap is mostly early: by 8000 moves both are ${f2(N.gq[7])} too high. Counting misses either way, Double DQN is off by ${f2(N.absd)}, DQN by ${f2(N.mq)}. Cheese reached: ${f1(N.evq)} and ${f1(N.evd)} of 30 spots.`]; },
   h: 'The bias shrinks', shot: [5, 12],
   draw(t) { if (busyKeys(QK.concat(DK))) return; const N = nums(), n = clamp(Math.floor((t - .5) * 1.5), 0, 8);
     const c = gapChart([{ v: N.gq, col: '#c4581f', w: 2.4, n }, { v: N.gd, col: '#3f8582', w: 2.4, n }], { title: 'estimate − reality, average of 8 runs' });
     ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.grey; ctx.beginPath(); ctx.moveTo(64, c.Y(0)); ctx.lineTo(356, c.Y(0)); ctx.stroke(); ctx.restore();
     card([{ s: `DQN (orange): ${f2(N.mq)} too high on average`, c: '#c4581f' }, { s: `Double DQN (teal): ${f2(N.md)}`, c: '#3f8582' }, { s: 'measured after 1000, 2000 … 8000 moves', c: C.soft }], t - 6, { y: 296, gap: 1.2 }); } },
 { get p() { const N = nums(), r = MOVES_N.map(RND); return ['Back to the plain numbers. Drag the number of **moves**, all truly worth 0, and see a thousand rounds.',
     `More moves give the max more chances to catch a lucky error: with ${MOVES_N[0]} moves the largest guess averages ${sg(r[0].max)}, with ${MOVES_N[3]} it averages ${sg(r[3].max)}. Choosing with A and scoring with B stays near 0 every time.`]; },
   h: 'Your turn', ctl: { label: 'moves', values: MOVES_N, def: 1, fmt: v => v, onChange: st => {} }, shot: [3, 9],
   draw(t, st) { const n = MOVES_N[st.ci ?? 1], r = RND(n), u = clamp((t - (st.changed ?? 0)) / 1.2), w = n <= 4 ? 56 : n <= 8 ? 36 : 17;
      for (let i = 0; i < 5; i++) { const rr = r.rs[i]; guessRow(rr.a, W / 2 - (n * (w + 6) - 6) / 2, 30 + i * 32, { w, hi: r.am(rr.a), a: () => clamp((t - (st.changed ?? 0)) * 3 - i * .4) }); }
     if (n > 8) txt('yellow: each round’s largest guess', W / 2, 205, { s: 13, c: C.soft });
     txt(`${n} moves, all worth 0: averages over ${NR} rounds`, W / 2, 230, { s: 15, c: C.soft });
     zeroBar(150, 330, 290, 360, [r.max, r.dbl], [C.yellow, '#9fc9c7'], ['the largest', 'A picks, B scores'], { u, max: .25, gap: 40 }); } },
 { h: 'Where it’s used', p: ['Van Hasselt, Guez and Silver found that DQN’s value estimates on several Atari games were far above the scores it really earned. Double DQN brought them down and played better on many games.', 'The same worry shaped later methods. TD3 (2018) keeps two critics and trusts the **smaller** of their two guesses. Rainbow (2017) bundles Double DQN with other upgrades.'],
   shot: [3, 9],
   draw(t) { D.meadow({ t }); D.pipAt(.1, .1, 1, t); dimScene(.6);
     card(['2010  Double Q-learning (van Hasselt)', '2015  Double DQN: live picks, copy scores', '2017  Rainbow: Double DQN plus other upgrades', '2018  TD3: trust the smaller of two critics'], t - .5, { y: 100, gap: 1.2 }); } },
 { h: 'Recap', p: ['The largest of several noisy guesses is too high on average. DQN’s target takes exactly such a max, so its numbers lean optimistic, and the optimism spreads backwards.', '**Double DQN** lets the live network choose the move and the frozen copy score it. Their errors are only partly shared, so most lucky high guesses stop getting in for free.'],
   shot: [3, 9],
   draw(t) { D.meadow({ t }); D.pipAt(.1, .1, 1, t); dimScene(.5);
     card(['max of noisy guesses: biased upward', 'DQN target = r + 0.9 × copy’s max', 'Double DQN: live chooses, copy scores', { s: 'same networks, one line changed', c: C.soft }], t - .3, { y: 100, gap: 1 }); } },
];
RLStory.register('doubledqn', { title: 'Double DQN', track: 'Deep RL', onCtx: c => { ctx = c; }, steps: STEPS, _sim: { nums, force, JOBS, RND } });
})();
