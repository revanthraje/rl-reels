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
// Target networks as a story: Pip's Q-network (with replay) computes its own targets, so every update moves the target
// it is chasing. A frozen copy of the network, refreshed every N updates, holds the targets still.
// Every number comes from the seeded runs below (kit DQN settings; see notes/targetnet.md).
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

const K = [0, 1, 2, 3], NS = [0, 10, 100, 1000], A = [0.50, 0.20], B = [0.65, 0.20], MA = 1, LR = 0.02, NSTEP = 200, REF = 50, S0 = 3000;
const probeS = n => maxQ(n, D.START[0], D.START[1]);
NS.forEach(N => K.forEach(k => job(`n${N}_${k}`, () => trainGen({ k, targetEvery: N, probe: probeS, keep: N === 0 && k === 0 ? [S0] : null }))));
// order: the story run first, then the comparison, then the slider
QUEUE.sort((a, b) => rank(a) - rank(b)); function rank(key) { const [n, k] = key.slice(1).split('_').map(Number); return (n === 0 || n === 100 ? 0 : 10) + k + (n === 100 ? .5 : 0); }
const keysN = N => K.map(k => `n${N}_${k}`);
const avgEv = keys => { const rs = keys.map(force); return rs[0].ev.map((_, i) => rs.reduce((a, r) => a + r.ev[i], 0) / rs.length); };
const jit = r => { let j = 0; for (let i = 4100; i < 8000; i += 100) j += Math.abs(r.probe[i] - r.probe[i - 100]); return j / 39; };
// the one-card experiment: from the live network after 3000 moves (run 1, no copy), 200 plain gradient steps on one card
function experiment(net0, refresh) { const n = D.copyNet(net0), fa = D.feats(A[0], A[1]); let tg = D.copyNet(n); const q = [], g = [];
  for (let i = 0; i <= NSTEP; i++) { if (refresh === 1 || (refresh > 1 && i % refresh === 0)) tg = D.copyNet(n);
    const qa = D.qOf(n, A[0], A[1])[MA], tq = D.GAM * Math.max(...D.qOf(tg, B[0], B[1])); q.push(qa); g.push(tq); if (i < NSTEP) D.sgd(n, fa, MA, tq, LR); }
  return { q, g }; }
let NUM = null;
function nums() { if (NUM) return NUM; const r0 = force('n0_0'), net = r0.snaps[S0];
  const live = experiment(net, 1), frozen = experiment(net, 0), stairs = experiment(net, REF);
  const per = NS.map(N => { const ev = avgEv(keysN(N)), js = keysN(N).map(k => jit(force(k))); return { ev, jit: js.reduce((a, b) => a + b, 0) / js.length, fin: keysN(N).map(k => last(force(k).ev)) }; });
  return (NUM = { live, frozen, stairs, per, r0, r100: force('n100_0') }); }
const last = v => v[v.length - 1];
function spotDot(x, y, col, lab, a = 1, dy = -14) { if (a <= 0) return; dot(D.px(x), D.py(y), 4.5, col, a); if (lab) txt(lab, D.px(x), D.py(y) + dy, { s: 14, c: col, a }); }
function cardScene(t, a = 1) { D.meadow({ t }); alpha(a, () => { spotDot(A[0], A[1], '#c99a3a', 'here', 1, 24); spotDot(B[0], B[1], gcol, 'landed', 1, 24); arrow(D.px(A[0]) + 7, D.py(A[1]), D.px(B[0]) - 7, D.py(B[1]), 1700, { c: C.ink, w: 1.5 }); }); }
const expChart = (o) => chart(Object.assign({ x0: 64, x1: 356, y0: 380, y1: 314, lo: -.08, hi: .2, xmax: NSTEP, yt: [0, .1, .2], xl: ['step 0', `step ${NSTEP}`] }, o));
const qcol = '#c99a3a', gcol = '#3f8582';
function twoNets(t, o = {}) { const net = need('n0_0') ? force('n0_0').snaps[S0] : D.newNet(4, 8, 4, 7000);
  D.drawNet(net, { x: 50, y: 300, w: 110, h: 80 }, { a: clamp(t - .3), wa: .8 }); txt('live network', 105, 296, { s: 14, c: C.goldInk, a: clamp(t - .3) });
  if (o.copy) { alpha(clamp(t - o.copy) * .75, () => D.drawNet(o.net || net, { x: 240, y: 300, w: 110, h: 80 }, { wa: .45 })); txt(o.copyLab || 'frozen copy', 295, 296, { s: 14, c: C.soft, a: clamp(t - o.copy) });
    if (o.arrowA) alpha(clamp(t - o.arrowA), () => { arrow(176, 340, 222, 340, 1710, { c: C.soft, w: 1.4 }); txt(o.arrowLab || 'copy', 199, 328, { s: 13, c: C.soft }); }); }
}

const STEPS = [
 { h: 'Where replay left off', p: ['The DQN reel spotted the problem: the target is computed by the same network being nudged, so it moves with the guess. Pip keeps the replay memory from the last reel; now we look at that chase up close.', 'A card’s target is the reward plus 0.9 × the largest Q-value where Pip landed. Who supplies that Q-value? The very network being trained.'],
   shot: [3, 9],
   say: [["Pip keeps its replay memory from last time. Now we zoom in on the other problem."], [["Here's Pip's network.", [40, 284, 135, 104]], ["To build a target, it needs the best value where Pip landed.", [30, 110, 340, 64]], ["And who supplies that value? The very network being trained.", [30, 110, 340, 64]]]],
   draw(t) { D.meadow({ t }); D.pipAt(.1, .1, 1, t); twoNets(t);
     if (t > 3) dimScene(.6 * clamp(t - 3)); card(['target = reward + 0.9 × largest Q where it landed', { s: '          ↑ read off the same network', c: C.roseInk }], t - 3, { y: 110, gap: 1.6, fs: 16 }); } },
 { h: 'One card, many steps', get p() { const N = nums(); return [`An experiment: take Pip’s network after 3000 moves of training, and one card. From (0.50, 0.20) Pip moved right to (0.65, 0.20), reward 0.`,
     `The network values this move at ${f2(N.live.q[0])}; its target is 0 + 0.9 × the best Q-value at the landing spot = ${f2(N.live.g[0])}. Now take ${NSTEP} small gradient steps on this one card and watch both.`]; },
   shot: [3, 9],
   get say() { const N = nums(); return [["Let's run a small experiment, with one single card."], [["Watch Pip step right, from here to the landed spot. No reward.", [64, 8, 272, 268]]], [`The network values that move at ${f2(N.live.q[0])}.`, ["Its target is zero point nine times the best value where Pip landed.", [30, 292, 340, 87]], "Now we train on this card alone, and watch both numbers."]]; },
   draw(t) { cardScene(t, clamp(t - .5)); D.pipAt(lerp(A[0], B[0], ease((t - 1) / 1.2)), A[1] + .05, 1, t); if (!need('n0_0')) { busy(progress(['n0_0'])); return; } const N = nums();
     card([`this move’s Q-value:  ${f2(N.live.q[0])}`, `target:  0 + 0.9 × ${f2(N.live.g[0] / D.GAM)}  =  ${f2(N.live.g[0])}`, { s: 'experiment: learn from this one card only', c: C.soft }], t - 2, { gap: 1.3 }); } },
 { h: 'Chasing your own shadow', get p() { const N = nums(); return [`Each step pulls gold toward teal, but teal moves too. After ${NSTEP} steps gold is ${f2(last(N.live.q))} and the target ${f2(last(N.live.g))}, levelling off. No new information came in; whatever the true value, this climb came from the chase alone.`,
     'Why? The nearby landing spot shares the same 76 knobs. Raising the value here raises it there, so the target runs ahead: **chasing your own shadow**.']; },
   shot: [4, 12],
   get say() { const N = nums(); return [[["Watch the two lines.", [40, 300, 330, 115]], ["Gold is the move's value. Teal is its target.", [40, 300, 330, 115]], ["Gold climbs toward teal, but teal climbs too.", [40, 300, 330, 115]]], [`Gold ends near ${f2(last(N.live.q))}, and nothing new was learned at all.`, "The landing spot is close by, and shares the same knobs. Raise one, and the other rises too. Pip is chasing its own shadow."]]; },
   draw(t) { cardScene(t, .8); if (!need('n0_0')) { busy(progress(['n0_0'])); return; } const N = nums(), n = clamp(Math.floor((t - .8) * 25), 0, NSTEP + 1);
     const c = expChart({ title: 'one card, step by step', series: [{ v: N.live.g, col: gcol, w: 2, n }, { v: N.live.q, col: qcol, w: 2, n }] });
     if (n > 1) { const i = n - 1; txt(f2(N.live.q[i]), c.X(i) + 4, c.Y(N.live.q[i]) + 10, { s: 13, c: C.goldInk, al: 'left' }); txt(f2(N.live.g[i]), c.X(i) + 4, c.Y(N.live.g[i]) - 10, { s: 13, c: gcol, al: 'left' }); }
     txt('gold: Q-value of the move', 120, 411, { s: 13, c: C.goldInk }); txt('teal: its target', 290, 411, { s: 13, c: gcol }); } },
 { h: 'Freeze a copy', p: ['The fix: keep a second network, a **frozen copy** of Pip’s network, used only to compute targets. It is called the **target network**. Training changes the live network; the copy stays as it was.', 'Now the target is reward + 0.9 × the largest Q-value **the copy** gives at the landing spot.'],
   shot: [3, 9],
   say: [["The fix is simple. Keep a second network, a frozen copy, just for targets."], [["The live network on the left learns and acts.", [40, 284, 135, 104]], ["The copy on the right stays still.", [230, 284, 135, 104]]], [["Now the target reads the copy's best value instead.", [30, 100, 340, 87]]]],
   draw(t) { D.meadow({ t }); D.pipAt(.1, .1, 1, t); if (t > 3.2) dimScene(.6 * clamp(t - 3.2)); twoNets(t, { copy: 1.2, arrowA: 2 });
     card(['live network: what Pip trains and acts with', 'frozen copy: only supplies the targets', { s: 'target = reward + 0.9 × copy’s best Q there', c: C.goldInk }], t - 3.2, { y: 100, gap: 1.3, fs: 16 }); } },
 { h: 'The chase stops', get p() { const N = nums(); return [`Same card, same ${NSTEP} steps, but the target comes from the frozen copy. Teal stays at ${f2(N.frozen.g[0])}, and gold reaches it, to within 0.01, after ${N.frozen.q.findIndex((v, i) => Math.abs(v - N.frozen.g[i]) < .01)} steps, then stays there.`,
     'With the target held still, each update solves an ordinary fitting problem: hit a fixed number.']; },
   shot: [4, 11],
   get say() { const N = nums(); return [[["Same card, same steps, but now the target comes from the frozen copy.", [40, 300, 330, 115]], ["Watch teal. It stays flat.", [40, 300, 330, 115]], `Gold reaches it in ${N.frozen.q.findIndex((v, i) => Math.abs(v - N.frozen.g[i]) < .01)} steps, and stops.`], ["With the target held still, it's just ordinary fitting. Hit a fixed number."]]; },
   draw(t) { cardScene(t, .8); if (!need('n0_0')) { busy(progress(['n0_0'])); return; } const N = nums(), n = clamp(Math.floor((t - .8) * 25), 0, NSTEP + 1);
     expChart({ title: 'one card, target from the frozen copy', series: [{ v: N.live.g, col: gcol, w: 1, a: .25 }, { v: N.live.q, col: qcol, w: 1, a: .25 }, { v: N.frozen.g, col: gcol, w: 2, n }, { v: N.frozen.q, col: qcol, w: 2, n }] });
     txt('faint: the chase from before', W / 2, 411, { s: 13, c: C.soft, a: clamp(t - 2) }); } },
 { h: 'Refresh every N updates', get p() { const N = nums(); return [`A copy frozen forever would never learn, so every **N** updates the live knobs are copied into it. Here N = ${REF}: the target holds, gold catches up, the target jumps.`,
     `One card anchors nothing, so the drift remains, slower: ${f2(last(N.stairs.q))} after ${NSTEP} steps instead of ${f2(last(N.live.q))}. The copy slows the loop but does not remove it. In real training, other cards bring real information.`]; },
   shot: [4, 11],
   get say() { const N = nums(); return ["A copy frozen forever would never learn. So every so often, we refresh it with the live knobs.", [["Watch the staircase.", [40, 300, 330, 115]], ["The target holds, gold catches up, then the target jumps.", [40, 300, 330, 115]]], [`The drift is still there, just slower. Gold ends at ${f2(last(N.stairs.q))}.`, "In real training, other cards bring real information to anchor it."]]; },
   draw(t) { cardScene(t, .8); if (!need('n0_0')) { busy(progress(['n0_0'])); return; } const N = nums(), n = clamp(Math.floor((t - .8) * 25), 0, NSTEP + 1);
     const c = expChart({ title: `one card, copy refreshed every ${REF} steps`, series: [{ v: N.stairs.g, col: gcol, w: 2, n, step: 1 }, { v: N.stairs.q, col: qcol, w: 2, n }] });
     for (let i = REF; i < Math.min(n, NSTEP); i += REF) txt('copy', c.X(i), 411, { s: 12, c: C.soft });
     if (n < 2) return; } },
 { h: 'Back to real training', p: ['Now full training: 8000 moves with replay, once with live targets and once with a copy refreshed every 100 updates. Same seeds.', 'The chart tracks the largest Q-value at Pip’s start spot, bottom left, over 8000 moves of experience. Orange: live targets. Teal: with the copy. At move 0 the number comes from random knobs.'],
   shot: [4, 12],
   say: [["Now back to full training, with replay, and the same seeds. Once with live targets, once with a copy."], [["We'll track one number, the best value at Pip's start spot.", [64, 8, 272, 268]]], [["Watch the lines draw.", [40, 300, 330, 100]], ["Orange uses live targets. Teal uses the copy.", [40, 300, 330, 100]]]],
   draw(t) { D.meadow({ t }); D.pipAt(.1, .1, 1, t); spotDot(.1, .1, C.ink, 'start', clamp(t - 1), -22); if (busyKeys(['n0_0', 'n100_0'])) return; const N = nums(), n = clamp(Math.floor((t - 3) * 1100), 0, 8000), samp = r => Array.from(r.probe).filter((_, i) => i % 50 === 0);
     chart({ x0: 64, x1: 356, y0: 388, y1: 316, lo: -.2, hi: .8, xmax: 160, yt: [0, .4, .8], xl: ['move 0', '8000'], title: 'largest Q-value at the start (run 1)',
       series: [{ v: samp(N.r0), col: C.orange, w: 1.3, n: n / 50 }, { v: samp(N.r100), col: '#3f8582', w: 1.6, n: n / 50 }] }); } },
 { h: 'Calmer numbers', get p() { const N = nums(), a = N.per[0], b = N.per[2]; return [`Live targets jitter. Measure the **jumps**: the average size of the change between readings 100 moves apart, over moves 4000 to 8000, four runs each: ${f2(a.jit)} with live targets, ${f2(b.jit)} with the copy.`,
     `Cheese from 30 test spots after 8000 moves: ${f1(last(a.ev))} against ${f1(last(b.ev))}. So in this meadow replay did most of the work; the copy mainly steadies the numbers and adds a little.`]; },
   shot: [4, 12],
   get say() { const N = nums(), a = N.per[0], b = N.per[2]; return [[["Here's the second half of training, four runs each.", [40, 28, 330, 236]], ["Orange jitters. Teal is calmer.", [40, 28, 330, 236]]], [`On average, live targets jump about ${f2(a.jit)} between readings, the copy only ${f2(b.jit)}.`, ["Reaching the cheese, they're close. Replay did most of the work here. The copy mainly steadies things.", [30, 296, 340, 87]]]]; },
   draw(t) { if (busyKeys(keysN(0).concat(keysN(100)))) return; const N = nums(), samp = r => Array.from(r.probe).filter((_, i) => i % 50 === 0).slice(80), n = clamp(Math.floor((t - .5) * 30), 0, 80);
     const ser = []; K.forEach(k => { ser.push({ v: samp(force('n0_' + k)), col: C.orange, w: 1.1, n, a: .7 }); ser.push({ v: samp(force('n100_' + k)), col: '#3f8582', w: 1.4, n, a: .9 }); });
     chart({ x0: 64, x1: 356, y0: 250, y1: 40, lo: -.1, hi: .8, xmax: 80, yt: [0, .4, .8], xl: ['move 4000', '8000'], title: 'largest Q-value at the start, 4 runs each', series: ser });
     card([{ s: `live targets: jumps ${f2(N.per[0].jit)} per 100 moves`, c: '#c4581f' }, { s: `copy every 100: jumps ${f2(N.per[2].jit)}`, c: '#3f8582' }, { s: `cheese from 30 spots: ${f1(last(N.per[0].ev))} vs ${f1(last(N.per[2].ev))}`, c: C.soft }], t - 4, { y: 296, gap: 1.2 }); } },
 { h: 'Your turn', get p() { const N = nums(); return ['Drag **N**, the number of updates between refreshes of the copy (four runs each; “none” means live targets).',
     `Too often, and the chase is back. Too rarely, and the targets go stale: N = 1000 gives only ${force('n1000_0').copies.length} refreshes in the run, the calmest numbers (jumps ${f2(N.per[3].jit)}), but only ${f1(last(N.per[3].ev))} of 30 spots reach the cheese. N = 100 balances the two here.`]; },
   ctl: { label: 'N', values: NS, def: 2, fmt: v => v ? v : 'none', onChange: st => {} }, shot: [3, 9],
   get say() { const N = nums(); return ["Your turn. Drag N, how many updates pass before the copy is refreshed.", [["Watch the start value as you change it.", [40, 28, 330, 236]], "Refresh too often, and the chase comes back."], [["Refresh too rarely, and the targets go stale.", [30, 292, 340, 106]], `At a thousand, the numbers are calmest, but only ${f1(last(N.per[3].ev))} spots out of thirty reach the cheese.`]]; },
   draw(t, st) { const N0 = NS[st.ci ?? 2], keys = keysN(N0); if (busyKeys(keys)) return; const r = force(keys[0]), ev = avgEv(keys), n = clamp(Math.floor((t - (st.changed ?? 0) - .2) * 40), 0, 160), js = keys.map(k => jit(force(k)));
     const samp = Array.from(r.probe).filter((_, i) => i % 50 === 0);
     chart({ x0: 64, x1: 356, y0: 250, y1: 40, lo: -.2, hi: .8, xmax: 160, yt: [0, .4, .8], xl: ['move 0', '8000'], title: 'largest Q-value at the start (run 1)', series: [{ v: samp, col: '#3f8582', w: 1.5, n }] });
     card([N0 ? `copy refreshed every ${N0} updates` : 'no copy: targets from the live network', `jumps per 100 moves: ${f2(js.reduce((a, b) => a + b, 0) / 4)}`, `cheese from 30 spots after 8000 moves: ${f1(last(ev))}`, { s: 'averages over 4 runs', c: C.soft }], 10, { y: 292, lh: 22, fs: 16 }); } },
 { h: 'Where it’s used', p: ['DQN (2015) used a target network refreshed every 10,000 updates. Without it, still with replay, its Breakout score fell from about 317 to about 241.', 'Many later methods, like DDPG and SAC, use **soft updates** instead: after every update the copy moves a tiny fraction, such as half a percent, toward the live network, so the target drifts smoothly.'],
   shot: [3, 9],
   say: [[["DQN, in twenty fifteen, refreshed its copy every ten thousand updates.", [30, 100, 340, 110]], ["Without it, its Breakout score fell from about three hundred and seventeen to about two hundred and forty one.", [30, 100, 340, 110]]], [["Later methods, like DDPG and SAC, blend instead. The copy slides a tiny bit toward the live network after every update.", [30, 100, 340, 110]]]],
   draw(t) { D.meadow({ t }); D.pipAt(.1, .1, 1, t); dimScene(.6);
     card(['2015  DQN: copy every 10,000 updates', '        Breakout: about 317 with it, 241 without', 'soft updates: copy = 99.5% copy + 0.5% live', { s: '        every update; used by DDPG and SAC', c: C.soft }], t - .5, { y: 100, gap: 1.2, fs: 16 }); } },
 { h: 'Recap', p: ['When a network computes its own targets, every update moves the target it chases: the numbers can drift and wobble.', 'A **target network**, a frozen copy refreshed every N updates, holds the targets still, so each stretch of training is a fixed fitting problem. Refresh too rarely and learning slows.'],
   shot: [3, 9],
   say: [["When a network builds its own targets, every update moves the thing it's chasing."], [["A frozen copy, refreshed every so often, holds the target still.", [30, 90, 340, 110]], "Refresh too often, and the chase returns. Too rarely, and learning goes stale."]],
   draw(t) { D.meadow({ t }); D.pipAt(.1, .1, 1, t); dimScene(.6); twoNets(t, { copy: .3, arrowA: .5, arrowLab: 'every N' });
     card(['target = r + 0.9 × best Q of the frozen copy', 'every N updates: copy ← live network', 'N too small: chase · N too big: stale', { s: 'soft version: blend a little every update', c: C.soft }], t - .3, { y: 90, gap: 1 }); } },
];
RLStory.register('targetnet', { title: 'Target Networks', track: 'Deep RL', onCtx: c => { ctx = c; }, steps: STEPS, _sim: { nums, force, JOBS } });
})();
