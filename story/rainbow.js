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
// Rainbow as a story: Pip's DQN in the meadow gets two new upgrades looked at closely, dueling heads (value of the spot +
// advantage of each move) and distributional learning (the whole spread of returns, not just the average), then the
// six upgrades of Rainbow in one card. Every number comes from the seeded runs below (kit DQN settings, k = 0…7; see
// notes/rainbow.md). Training runs happen in background slices after load, as in replay.js.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, caption, dot, box, alpha } = L;
const D = RLStory.deeprl;
let ctx;
// ───── background jobs (as in replay / targetnet / doubledqn)
const JOBS = {}, QUEUE = []; let pumping = false;
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
function job(key, make) { if (!JOBS[key]) { JOBS[key] = { make, gen: null, res: null, prog: 0 }; QUEUE.push(key); kick(); } return JOBS[key]; }
function kick() { if (pumping || typeof setTimeout === 'undefined') return; pumping = true; setTimeout(pump, 30); }
function pump() { const t0 = now(); while (QUEUE.length && now() - t0 < 14) { const j = JOBS[QUEUE[0]]; if (!j.gen) j.gen = j.make(); const r = j.gen.next(); if (r.done) { j.res = r.value; QUEUE.shift(); } else j.prog = r.value; }
  if (QUEUE.length) setTimeout(pump, 1); else pumping = false; }
function need(key) { const j = JOBS[key]; if (!j) return null; if (!j.res) { const i = QUEUE.indexOf(key); if (i > 0) { QUEUE.splice(i, 1); QUEUE.unshift(key); } } return j.res; }
function force(key) { const j = JOBS[key]; if (!j.res) { if (!j.gen) j.gen = j.make(); let r; while (!(r = j.gen.next()).done); j.res = r.value; const i = QUEUE.indexOf(key); if (i >= 0) QUEUE.splice(i, 1); } return j.res; }
function progress(keys) { let p = 0; keys.forEach(k => { const j = JOBS[k]; p += j.res ? 1 : j.prog || 0; }); return p / keys.length; }
function busy(p, y = 345) { txt(`Pip is training…  ${Math.round(p * 100)}%`, W / 2, y, { s: 16, c: C.soft }); box(120, y + 14, 160, 5, '#e4d8c7'); box(120, y + 14, 160 * p, 5, C.teal); }
const busyKeys = (keys, y) => { const p = progress(keys); if (p >= 1) return false; busy(p, y); return true; };
const dimScene = (a = .62) => box(0, 0, W, 285, C.bg, a);

// ───── dueling network: the kit's net with 5 outputs, 0 = V (the spot), 1…4 = A (each move); Q = V + A − average A
function qDuel(n, f) { const o = D.fwd(n, f), m = (o[1] + o[2] + o[3] + o[4]) / 4; return [o[0] + o[1] - m, o[0] + o[2] - m, o[0] + o[3] - m, o[0] + o[4] - m]; }
// gradient of ½ (target − Q)² for any output weighting gout (same sign convention as the kit's accGrad)
function accVec(n, f, gout, g) { D.fwd(n, f); const w = n.w, h = n.h, nIn = n.nIn, nH = n.nH;
  for (let j = 0; j < nH; j++) { let gh = 0; for (let k = 0; k < n.nOut; k++) { if (!gout[k]) continue; const o2 = n.oW2 + k * nH; gh += gout[k] * w[o2 + j]; g[o2 + j] += gout[k] * h[j]; }
    gh *= 1 - h[j] * h[j]; g[n.oB1 + j] += gh; const o = j * nIn; for (let i = 0; i < nIn; i++) g[o + i] += gh * f[i]; }
  for (let k = 0; k < n.nOut; k++) g[n.oB2 + k] += gout[k]; }
function adam(n, m, lr) { const g = n.g, w = n.w, A = n.adam || (n.adam = { t: 0, m: new Float64Array(n.N), v: new Float64Array(n.N) }); A.t++;
  const c1 = 1 - .9 ** A.t, c2 = 1 - .999 ** A.t; for (let q = 0; q < n.N; q++) { const gg = g[q] / m; A.m[q] = .9 * A.m[q] + .1 * gg; A.v[q] = .999 * A.v[q] + .001 * gg * gg; w[q] += lr * (A.m[q] / c1) / (Math.sqrt(A.v[q] / c2) + 1e-8); } }
// DQN exactly as the kit's D.dqn (same seeds and order of random draws; replay 2000, minibatch 32, Adam 0.003, copy every
// 100 updates, 1 move in 5 random), with a switch for the dueling heads. Tests every 500 moves (cheese from the 30 test spots).
function* trainGen(o) {
  const k = o.k ?? 0, steps = 8000, lr = .003, B = 32, cap = 2000, tEv = 100, warm = 100, eps = .2, du = !!o.duel;
  const Rm = rng(5000 + k), Ra = rng(6000 + k), net = D.newNet(4, 8, du ? 5 : 4, 7000 + k); let tgt = D.copyNet(net);
  const Q = (n, f) => du ? qDuel(n, f) : Array.from(D.fwd(n, f)), buf = [], ev = [], gout = new Float64Array(5); let upd = 0, [x, y] = D.randStart(Rm), t = 0;
  for (let s = 0; s < steps; s++) {
    const f = D.feats(x, y), m = Ra() < eps ? Math.floor(Ra() * 4) : D.argmax(Q(net, f), Ra), e = D.envStep(x, y, m, Rm);
    const tr = { f, m, r: e.r, f2: D.feats(e.x, e.y), done: e.done }; if (buf.length < cap) buf.push(tr); else buf[s % cap] = tr;
    if (s >= warm) { const bb = Array.from({ length: B }, () => buf[Math.floor(Ra() * buf.length)]); net.g.fill(0);
      for (const b of bb) { const target = b.r + (b.done ? 0 : D.GAM * Math.max(...Q(tgt, b.f2))), err = target - Q(net, b.f)[b.m];
        if (du) { gout[0] = err; for (let j = 0; j < 4; j++) gout[1 + j] = err * ((j === b.m ? 1 : 0) - .25); } else { gout.fill(0); gout[b.m] = err; } accVec(net, b.f, gout, net.g); }
      adam(net, B, lr); upd++; if (upd % tEv === 0) tgt = D.copyNet(net); }
    t++; if (e.done || t >= D.TMAX) { [x, y] = D.randStart(Rm); t = 0; } else { x = e.x; y = e.y; }
    if ((s + 1) % 500 === 0) ev.push(testQ(f2 => Q(net, f2)));
    if (s % 40 === 39) yield s / steps;
  }
  return { net, ev, duel: du, q: (x2, y2) => Q(net, D.feats(x2, y2)) };
}
function testQ(Q) { let ch = 0; D.TESTS.forEach(([a, b], i) => { let x = a, y = b; const R = rng(9000 + i); for (let t = 0; t < D.TMAX; t++) { const s = D.envStep(x, y, D.argmax(Q(D.feats(x, y))), R); x = s.x; y = s.y; if (s.done) { ch += s.r > 0; break; } } }); return ch; }
const KS = [0, 1, 2, 3, 4, 5, 6, 7];
KS.forEach(k => { job('p' + k, () => trainGen({ k })); job('d' + k, () => trainGen({ k, duel: true })); });
QUEUE.sort((a, b) => +a.slice(1) - +b.slice(1) || (a > b ? -1 : 1));   // p0 first: the distributional steps use it
const PK = KS.map(k => 'p' + k), DK = KS.map(k => 'd' + k);

// ───── returns from one spot: Pip follows its trained network (run 1, plain DQN) with 1 move in 5 random, from spot (x, y)
// first = forced first move (−1: none). Return of a walk = ±1 × 0.9^(moves − 1) at the cheese or cat, 0 if 60 moves pass.
const SPOT = [0.75, 0.45], NW = 1000, NB = 20;
function walksFrom(net, x0, y0, first, n, seed, keepP = 0) { const R = rng(seed), out = [];
  for (let i = 0; i < n; i++) { let x = x0, y = y0, G = 0, end = 0; const P = i < keepP ? [[x, y]] : null;
    for (let t = 0; t < D.TMAX; t++) { const m = t === 0 && first >= 0 ? first : (R() < .2 ? Math.floor(R() * 4) : D.argmax(D.qOf(net, x, y))); const s = D.envStep(x, y, m, R); x = s.x; y = s.y; if (P) P.push([x, y]); if (s.done) { end = s.r; G = s.r * D.GAM ** t; break; } }
    out.push({ G, end, P }); } return out; }
function summary(w) { const h = new Array(NB).fill(0); let m = 0, cat = 0, ch = 0; w.forEach(q => { h[Math.min(NB - 1, Math.floor((q.G + 1) * NB / 2))]++; m += q.G; cat += q.end < 0; ch += q.end > 0; });
  m /= w.length; const near = w.filter(q => Math.abs(q.G - m) < .1).length; return { h, m, cat, ch, n: w.length, near, to: w.length - cat - ch }; }
// numbers the story quotes, in three lazy parts so a step only waits for the runs it needs
const MEMO = {};
const part = (k, fn) => () => MEMO[k] || (MEMO[k] = fn());
const nA = part('a', () => { const net = force('p0').net;
  let lo = 9, hi = -9, sp = 0; D.TESTS.forEach(([x, y]) => { const q = D.qOf(net, x, y), m = Math.max(...q); lo = Math.min(lo, m); hi = Math.max(hi, m); sp += m - Math.min(...q); });
  const qStart = D.qOf(net, D.START[0], D.START[1]);
  const W0 = walksFrom(net, SPOT[0], SPOT[1], -1, NW, 21, 200), S0 = summary(W0);
  // the eight walks drawn: the first six that reached the cheese and the first two that met the cat
  const SHOW = W0.filter(w => w.end > 0).slice(0, 6).concat(W0.filter(w => w.end < 0).slice(0, 2)), sh = [0, 1, 6, 2, 3, 7, 4, 5].map(i => SHOW[i]);
  const Su = summary(walksFrom(net, SPOT[0], SPOT[1], 0, NW, 22)), Sd = summary(walksFrom(net, SPOT[0], SPOT[1], 2, NW, 23));
  // one real move to the right from the spot, and the returns from where it landed
  const land = D.envStep(SPOT[0], SPOT[1], 1, rng(31)), Sl = summary(walksFrom(net, land.x, land.y, -1, NW, 24));
  return { net, lo, hi, sp: sp / 30, qStart, W0, sh, S0, Su, Sd, land, Sl }; });
const nD = part('d', () => { const Dn = force('d0'), oS = Array.from(D.fwd(Dn.net, D.feats(D.START[0], D.START[1]))), aS = oS.slice(1);
  return { Dn, oS, aS, aM: aS.reduce((a, b) => a + b, 0) / 4, qS: qDuel(Dn.net, D.feats(D.START[0], D.START[1])) }; });
const nR = part('r', () => { const evP = KS.map(k => force('p' + k).ev), evD = KS.map(k => force('d' + k).ev), avg = a => a.reduce((s, v) => s + v, 0) / a.length;
  return { evP, evD, mP: evP[0].map((_, i) => avg(evP.map(e => e[i]))), mD: evD[0].map((_, i) => avg(evD.map(e => e[i]))) }; });
const nums = () => Object.assign({}, nA(), nD(), nR());

// ───── drawing helpers
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−'), f1 = v => v.toFixed(1), sg = v => (v >= .005 ? '+' : '') + f2(v);
const pc = (a, n) => Math.round(100 * a / n) + '%';
const MN = D.MN;
// histogram of returns: bins over −1 … +1, gold for returns above 0, rose below. o = { x0, x1, y0 (bottom), hh (height), n: bins shown, mean, max, sq (squeeze factor), a }
function histo(S, o) { const x0 = o.x0 ?? 50, x1 = o.x1 ?? 350, y0 = o.y0 ?? 380, hh = o.hh ?? 70, mx = o.max ?? Math.max(...S.h), a = o.a ?? 1, q = o.sq ?? 1;
  const X = v => x0 + (v * q + 1) / 2 * (x1 - x0), bw = (x1 - x0) / NB * q; if (a <= 0) return X;
  alpha(a, () => { rline(x0 - 4, y0, x1 + 4, y0, 970, { c: C.line, w: 1, once: true });
    [-1, 0, 1].forEach(v => { const xx = x0 + (v + 1) / 2 * (x1 - x0); rline(xx, y0, xx, y0 + 4, 971 + v, { c: C.line, w: 1, once: true }); txt(v > 0 ? '+1' : v < 0 ? '−1' : '0', xx, y0 + 12, { s: 12, c: C.soft }); });
    for (let i = 0; i < Math.min(NB, o.n ?? NB); i++) { const v = S.h[i]; if (!v) continue; const lo = -1 + i * 2 / NB, bh = Math.max(1.5, v / mx * hh); box(X(lo) + bw * .08, y0 - bh, bw * .84, bh, lo + .05 < 0 ? C.rose : C.gold, o.ba ?? .85); }
    if (o.mean != null) { const xm = X(o.mean); ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(xm, y0); ctx.lineTo(xm, y0 - hh - 6); ctx.stroke(); ctx.restore();
      txt(o.ml ?? `average ${f2(o.mean)}`, xm, y0 - hh - 14, { s: 14, c: C.ink }); } });
  return X; }
function chart(o) {
  const X = i => o.x0 + i * (o.x1 - o.x0) / (o.xmax), Y = v => o.y0 - (v - o.lo) / (o.hi - o.lo) * (o.y0 - o.y1);
  rline(o.x0 - 4, o.y0, o.x1 + 4, o.y0, 960, { c: C.line, w: 1, once: true }); rline(o.x0 - 4, o.y0, o.x0 - 4, o.y1 - 2, 961, { c: C.line, w: 1, once: true });
  (o.yt || []).forEach(v => txt(String(v), o.x0 - 9, Y(v), { s: 12, c: C.soft, al: 'right' }));
  if (o.xl) { txt(o.xl[0], o.x0, o.y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(o.xl[1], o.x1, o.y0 + 10, { s: 12, c: C.soft, al: 'right' }); }
  if (o.title) txt(o.title, o.x0 - 8, o.y1 - 11, { s: 13, c: C.soft, al: 'left' });
  (o.series || []).forEach(s => { const n = Math.min(s.n ?? s.v.length, s.v.length); if (n < 1) return; ctx.save(); ctx.globalAlpha = s.a ?? 1; ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 1.4; ctx.lineJoin = 'round'; ctx.beginPath();
    for (let i = 0; i < n; i++) { const xx = X(i), yy = Y(s.v[i]); i ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); } ctx.stroke(); ctx.restore(); });
  return { X, Y };
}
function mini(tx, ty, s, fn) { ctx.save(); ctx.translate(tx, ty); ctx.scale(s, s); fn(); ctx.restore(); }
// four Q-values in a row of boxes with move names above
function qRow(q, x, y, o = {}) { const w = o.w || 58; q.forEach((v, i) => { const bx = x + i * (w + 6), a = o.a ? o.a(i) : 1; if (a <= 0) return;
  alpha(a, () => { box(bx, y, w, 26, o.hi === i ? C.yellow : C.card, o.hi === i ? .8 : 1); rrect(bx, y, w, 26, 900 + i + Math.round(y), { c: C.line, w: 1 });
    txt((o.fmt || f2)(v), bx + w / 2, y + 13, { s: 15, c: v < -.005 ? C.roseInk : C.goldInk }); if (!o.nolab) txt(MN[i], bx + w / 2, y - 9, { s: 13, c: C.soft }); }); }); }
function spotMark(x, y, lab, a = 1, col = C.ink) { alpha(a, () => { dot(D.px(x), D.py(y), 4.5, col); if (lab) txt(lab, D.px(x), D.py(y) - 14, { s: 14, c: col }); }); }
const ORG = '#c4581f', TEAL = '#3f8582';
const raceChart = (ser, o = {}) => chart(Object.assign({ x0: 64, x1: 356, y0: 250, y1: 40, lo: 0, hi: 30, xmax: 15, yt: [0, 15, 30], xl: ['500 moves', '8000'], title: 'test spots where Pip reaches the cheese (of 30)', series: ser }, o));
// map of a dueling net's V (the spot) or of its move spread (largest A − smallest A)
const vMap = n => (x, y) => D.fwd(n, D.feats(x, y))[0];
const aMap = n => (x, y) => { const o = D.fwd(n, D.feats(x, y)); return Math.max(o[1], o[2], o[3], o[4]) - Math.min(o[1], o[2], o[3], o[4]); };
// squeeze a histogram on the fixed bins (centres −0.95 … 0.95) by g and split each bar between the two nearest centres
function proj(h, g) { const out = new Array(NB).fill(0); h.forEach((v, i) => { const c = (-1 + (i + .5) * 2 / NB) * g, p = (c + 1) / (2 / NB) - .5, j = Math.floor(p), f = p - j;
  if (j >= 0) out[j] += v * (1 - f); if (j + 1 < NB) out[j + 1] += v * f; }); return out; }
// tap-your-own-spot state lives in st
function tapSummary(st) { if (!st.tap) return null; if (!st.sum) st.sum = summary(walksFrom(nA().net, st.tap[0], st.tap[1], -1, 400, 77)); return st.sum; }

const STEPS = [
 { h: 'Six upgrades', p: ['Pip’s DQN has a replay memory, a frozen copy for the targets and, from the last reel, the Double DQN trick. Each came from a different paper. In 2017 Matteo Hessel and colleagues at DeepMind stacked **six** such upgrades into one agent: **Rainbow**.', 'This reel opens up two new ones in the meadow, **dueling heads** and **distributional** learning, then lists all six.'],
   shot: [3, 10],
   say: [["Back to the meadow. Pip's DQN already has a few upgrades, each from a different paper.", [72, 14, 256, 256]], [["In 2017, a DeepMind team stacked six of them into one agent, and called it Rainbow.", [30, 292, 340, 110]], ["This reel opens up two new ones, then lists all six.", [30, 292, 340, 110]]]],
   draw(t) { D.meadow({ t, labels: clamp(t - .5) }); D.pipAt(.1, .1, 1, t);
     card(['so far: replay, frozen copy, Double DQN', { s: 'new here: dueling heads', c: C.goldInk }, { s: 'new here: distributional learning', c: C.goldInk }, { s: 'Rainbow (2017): six upgrades in one agent', c: C.soft }], t - 2, { gap: 1.2 }); } },
 { get p() { const N = nA(); return [`Pip’s DQN-reel network, run 1, after 8000 moves, no Double trick: at the bottom-left start its Q-values are ${N.qStart.map(f2).join(', ')}. Close: from out here, every move leaves a long way to go.`,
     `Over the 30 test spots the best Q-value spans ${f2(N.lo)} to ${f2(N.hi)}, but best and worst moves typically differ by only ${f2(N.sp)}. **Where** Pip stands often matters more than **which** move.`]; },
   h: 'Does the move matter?', shot: [4, 11],
   say: [["Start with a question. Does the move really matter?", [72, 14, 256, 256]], [["Here are Pip's four Q-values at the start.", [30, 295, 340, 40]], ["They're very close. From way out here, every move leaves a long way to go.", [30, 295, 340, 40]]], ["Now watch the gold squares. Where Pip stands changes the value far more than which move it picks.", [72, 14, 256, 256]]],
   draw(t) { if (busyKeys(['p0'], 330)) { D.meadow({ t }); return; } const N = nA(); D.meadow({ t, heat: (x, y) => Math.max(...D.qOf(N.net, x, y)), heatA: clamp(t - 5) * .8 });
     D.pipAt(.1, .1, 1, t); spotMark(.1, .1, '', 1);
     qRow(N.qStart, 76, 312, { a: i => clamp(t - .8 - i * .3), hi: D.argmax(N.qStart) }); txt('start', 40, 325, { s: 15, c: C.soft });
     if (t > 5) caption('gold squares: best Q-value at each spot (for drawing only)', clamp(t - 5)); } },
 { get p() { return ['Split each Q-value into a spot part, **V**, and a move part, its **advantage** **A**: Q = V + A. Shifting 0.1 between V and the A’s changes no Q, so the split needs a rule.',
     'The rule: subtract the four A’s average. V then equals the four Q’s plain average, the value of a random first move; not the values reel’s V, which weights moves by Pip’s choices.']; },
   h: 'Spot plus move', shot: [4, 12],
   say: [[["So split each Q-value in two.", [20, 50, 360, 110]], ["A part for the spot, called V. And a part for the move, its advantage, A.", [20, 50, 360, 110]]], [["But you could shift a little from V to every A, and no Q would change.", [30, 292, 340, 87]], ["So there's a rule. Subtract the average advantage. Then V is just the average of the four Q-values.", [30, 292, 340, 87]]]],
   draw(t) { D.meadow({ t }); dimScene(.75);
     txt('Q (spot, move)  =  V (spot)  +  A (spot, move)', W / 2, 70, { s: 19, a: clamp(t - .5) });
     txt('V: the spot’s part (here: average of the 4 Q’s)', W / 2, 115, { s: 17, c: C.goldInk, a: clamp(t - 2) }); txt('A: how far this move is above or below that', W / 2, 145, { s: 17, c: TEAL, a: clamp(t - 3.2) });
     card(['V + 0.1 and every A − 0.1: same Q-values', { s: 'so: subtract the average of the four A’s', c: C.goldInk }, { s: 'V = average Q, not the value under Pip’s habit', c: C.soft }], t - 5.5, { gap: 1.3 }); } },
 { get p() { const N = nD(), q = N.qS; return ['A **dueling network** builds this into its shape (Ziyu Wang and colleagues, 2016). One hidden layer feeds two heads: one output for V, four for the advantages. Then Q = V + A − average A.',
     `Trained like Pip’s DQN, same seeds, it starts at V = ${f2(N.oS[0])}; the four A’s average ${f2(N.aM)}. The Q-values come out ${q.map(f2).join(', ')}, averaging ${f2(N.oS[0])}: the V.`]; },
   h: 'Two heads', shot: [4, 12],
   say: [[["A dueling network builds this split into its shape.", [100, 20, 280, 230]], ["One hidden layer feeds two heads. One output for V, and four for the advantages.", [100, 20, 280, 230]]], [["At the start, the four advantages are averaged and subtracted.", [30, 280, 340, 90]], ["What's left is Q equals V plus A, minus the average A.", [30, 280, 340, 90]]]],
   draw(t) { if (busyKeys(['d0'], 200)) return; const N = nD(), f = D.feats(D.START[0], D.START[1]), q = qDuel(N.Dn.net, f);
     D.drawNet(N.Dn.net, { x: 110, y: 30, w: 150, h: 200 }, { inL: D.FNAMES, outL: ['V', 'A up', 'A right', 'A down', 'A left'], f, a: clamp(t - .3), build: clamp((t - .3) / 2), outFmt: f2 });
     if (t > 3) { rrect(250, 22, 116, 38, 1201, { c: C.goldInk, w: 1 }); txt('spot', 372, 41, { s: 13, c: C.goldInk, al: 'left', a: clamp(t - 3) }); rrect(250, 66, 116, 170, 1202, { c: TEAL, w: 1 }); }
     txt('at the start spot', 110, 250, { s: 14, c: C.soft, a: clamp(t - 3) });
     if (t > 5) { txt(`average of the four A’s: ${f2(N.aM)}, subtracted`, W / 2, 292, { s: 15, a: clamp(t - 5) }); txt('Q = V + A − average A', W / 2, 316, { s: 15, c: C.soft, a: clamp(t - 6) });
       qRow(q, 76, 350, { a: i => clamp(t - 7 - i * .3), hi: D.argmax(q) }); } } },
 { h: 'Why bother?', p: ['Each replay card is about one move: where Pip stood, which move, what followed. In a plain Q-network, it teaches that move’s output; shared hidden knobs nudge the other three only indirectly.', 'In a dueling network, V sits inside every Q-value, so every card trains V directly, whatever the move. The advantage head only learns the differences, small in most of the meadow.'],
   shot: [4, 12],
   say: ["Why bother? Think about one replay card, about one move.", [["In a plain network, it mostly teaches that one move's output.", [40, 30, 160, 190]], ["In a dueling network, V sits inside every Q-value. So every card trains V, whatever the move.", [200, 30, 170, 190]]], ["The advantage head only has to learn the differences.", [30, 292, 340, 87]]],
   draw(t) { D.meadow({ t }); dimScene(.7); const n = D.newNet(4, 8, 4, 7000), m = D.newNet(4, 8, 5, 7000); dimScene(.25);
     txt('plain', 105, 40, { s: 17, c: ORG, a: clamp(t - .5) }); D.drawNet(n, { x: 50, y: 56, w: 100, h: 130 }, { a: clamp(t - .5), outL: MN, hl: t > 2 ? 0 : -1, wa: .5 });
     txt('dueling', 285, 40, { s: 17, c: TEAL, a: clamp(t - 3) }); D.drawNet(m, { x: 225, y: 56, w: 100, h: 130 }, { a: clamp(t - 3), outL: ['V', 'up', 'right', 'down', 'left'], hl: t > 5 ? 0 : -1, wa: .5 });
     if (t > 2) txt('a card about “up”: one output learns', 105, 210, { s: 14, c: ORG, a: clamp(t - 2) });
     if (t > 5) txt('the same card also teaches V', 285, 210, { s: 14, c: TEAL, a: clamp(t - 5) });
     card(['V hears from every card', 'A learns only the differences', { s: 'helps most where the move barely matters', c: C.soft }], t - 7, { gap: 1.2 }); } },
 { get p() { const N = nA(); return ['Dueling network, run 1, 8000 moves. Left: **V** at every spot, rising toward the cheese, lowest just below and left of the cat, where every route goes the long way round.',
     'Right: how much the move matters, largest advantage minus smallest. Strongest around the cat, where one wrong step costs a lot; palest bottom left and in a band below the cat, where moves are about equally good. The heads split the question as hoped.']; },
   h: 'Two maps', shot: [4, 11],
   say: [[["Here's what the two heads learned.", [8, 22, 386, 210]], ["On the left, V. It rises toward the cheese, and dips behind the cat.", [8, 22, 190, 210]]], [["On the right, how much the move matters.", [204, 22, 190, 210]], ["Strongest around the cat, where one wrong step costs a lot. Pale where moves are about equally good.", [204, 22, 190, 210]]]],
   draw(t) { if (busyKeys(['d0'], 150)) return; const N = nD(), s = .7;
     mini(12 - D.FX * s, 46 - D.FY * s, s, () => D.meadow({ t, heat: (x, y) => vMap(N.Dn.net)(x, y) - .2, vmax: .5, heatA: clamp(t - .5), tufts: .4 }));
     mini(208 - D.FX * s, 46 - D.FY * s, s, () => D.meadow({ t, heat: aMap(N.Dn.net), heatA: clamp(t - 2.5), vmax: .45, tufts: .4 }));
     txt('V: value of the spot', 102, 30, { s: 16, c: C.goldInk, a: clamp(t - .5) }); txt('how much the move matters', 298, 30, { s: 16, c: TEAL, a: clamp(t - 2.5) });
     card(['left: V, the dueling net’s spot value', 'right: largest A − smallest A', { s: 'deeper colour = bigger; squares only for drawing', c: C.soft }], t - 4, { y: 252, gap: 1.2 }); } },
 { get p() { const N = nR(), last = v => v[v.length - 1]; return ['Does it learn faster? Eight runs each, same seeds for starts, wobble and random moves, same 30 test spots, tested every 500 moves.',
     `After 8000 moves: ${f1(last(N.mP))} of 30 for plain, ${f1(last(N.mD))} for dueling. Single runs spread widely, so this gap is within the noise: about equal with four moves. Wang and colleagues saw clearer gains on Atari, with up to 18 moves.`]; },
   h: 'A fair race', shot: [5, 13],
   say: [["Does it learn faster? Eight runs each, with matching seeds.", [40, 25, 330, 250]], [["Watch the thick lines, the averages.", [40, 25, 330, 250]], ["Orange is plain DQN. Teal is dueling.", [30, 296, 340, 87]]], ["They end up about equal. Single runs spread widely, so the gap is within the noise.", "On Atari, with more moves to choose from, the gains were clearer."]],
   draw(t) { if (busyKeys(PK.concat(DK))) return; const N = nR(), n = clamp(Math.floor((t - .5) * 2.5), 0, 16), ser = [];
     KS.forEach(k => { ser.push({ v: N.evP[k], col: C.orange, w: 1, a: .3, n }); ser.push({ v: N.evD[k], col: C.teal, w: 1, a: .3, n }); });
     ser.push({ v: N.mP, col: ORG, w: 2.6, n }, { v: N.mD, col: TEAL, w: 2.6, n }); raceChart(ser);
     card([{ s: 'orange: plain DQN', c: ORG }, { s: 'teal: dueling DQN', c: TEAL }, { s: 'thin: single runs · thick: average of 8', c: C.soft }], t - 1, { y: 296, gap: 1.1 }); } },
 { get p() { const N = nA(); return [`Now the second upgrade. From one spot, below and right of the cat, Pip walks ${NW} times, following its network, one move in five random. Each walk ends on the cheese, the cat, or after 60 moves.`,
     'A walk’s **return** is the final +1 or −1, discounted by 0.9 per move before it: cheese after 3 moves gives 0.81. Drawn: six cheese walks, two cat walks. Same spot, same habit, different luck.']; },
   h: 'One spot, many futures', shot: [5, 12],
   say: [[["Now the second upgrade.", [72, 14, 256, 256]], ["From one spot, below and right of the cat, Pip walks a thousand times, with some random moves.", [72, 14, 256, 256]]], [["Watch the paths. Most reach the cheese. Some meet the cat.", [72, 14, 256, 256]], ["Each walk's return depends on how it ended, and how long it took.", [30, 288, 340, 40]]], "Same spot, same habit, different luck."],
   draw(t) { if (busyKeys(['p0'], 330)) { D.meadow({ t }); return; } const N = nA(); D.meadow({ t });
     N.sh.forEach((w, i) => { const u = t - 1 - i * 1; if (u <= 0) return; D.path(w.P, Math.floor(u * 8) + 1, { c: w.end > 0 ? '#c99a3a' : w.end < 0 ? C.rose : C.grey, a: i === Math.min(7, Math.floor(t - 1)) ? 1 : .45 }); });
     spotMark(SPOT[0], SPOT[1], 'the spot');
     N.sh.slice(0, clamp(Math.floor(t - 1.5), 0, 8)).forEach((w, i) => txt(w.end > 0 ? '+' + w.G.toFixed(2) : w.end < 0 ? '−' + Math.abs(w.G).toFixed(2) : '0', 46 + i * 44, 300, { s: 16, c: w.end < 0 ? C.roseInk : C.goldInk }));
     if (t > 2) txt('returns of eight of the walks', W / 2, 322, { s: 13, c: C.soft });
     card(['cheese after n moves: +0.9 to the power n − 1', 'cat after n moves: −0.9 to the power n − 1'], t - 9, { y: 340, gap: 1 }); } },
 { get p() { const N = nA(), S = N.S0; return [`Now sort all ${NW} returns into a **histogram**. Across: the return, −1 to +1, in slices 0.1 wide. Height: how many walks landed in that slice.`,
     `Gold bars are walks that reached the cheese: ${S.ch} of them, spread by how many moves they took. Rose bars on the far left are walks that met the cat: ${S.cat}. ${S.to ? `${S.to} ran out of time.` : 'None ran out of time.'}`]; },
   h: 'A histogram of returns', shot: [4, 12],
   say: [[["Now sort all thousand returns into a histogram.", [40, 285, 320, 130]], ["Across is the return, from minus one to plus one. Height is how many walks landed there.", [40, 285, 320, 130]]], [["The gold bars on the right reached the cheese, spread by how long they took.", [40, 285, 320, 130]], ["The rose bars on the far left met the cat.", [40, 285, 320, 130]]]],
   draw(t) { if (busyKeys(['p0'], 330)) return; const N = nA(); D.meadow({ t }); dimScene(.5); spotMark(SPOT[0], SPOT[1], 'the spot');
     histo(N.S0, { x0: 50, x1: 350, y0: 380, hh: 74, n: clamp(Math.floor((t - 1) * 3), 0, NB) });
     if (t > 2) txt('how many walks', 52, 296, { s: 13, c: C.soft, al: 'left', a: clamp(t - 2) }); txt('return of the walk', W / 2, 408, { s: 13, c: C.soft, a: clamp(t - 1) });
     if (t > 7) { txt(`cat: ${N.S0.cat}`, 64, 340, { s: 15, c: C.roseInk, a: clamp(t - 7) }); txt(`cheese: ${N.S0.ch}`, 330, 296, { s: 15, c: C.goldInk, a: clamp(t - 8), al: 'right' }); } } },
 { get p() { const N = nA(), S = N.S0; return [`A Q-value keeps just one number: the average return, here ${f2(S.m)}. Only ${S.near} of the ${NW} walks came within 0.1 of it; the rest scattered, the cat walks far off on the left.`,
     `And it hides the risk: ${pc(S.cat, NW)} of walks from here end on the cat. A single number cannot say whether ${f2(S.m)} is a sure thing or a gamble.`]; },
   h: 'The average hides it', shot: [4, 11],
   say: [[["A Q-value keeps just one number, the average return.", [40, 285, 320, 130]], ["Watch the teal band. Only a small share of walks land anywhere near that average.", [40, 285, 320, 130]]], [["And the average hides the risk.", [40, 285, 320, 130]], "It can't tell a sure thing from a gamble."]],
   draw(t) { if (busyKeys(['p0'], 330)) return; const N = nA(); D.meadow({ t }); dimScene(.5); spotMark(SPOT[0], SPOT[1], 'the spot');
     histo(N.S0, { x0: 50, x1: 350, y0: 380, hh: 74, mean: t > 1.5 ? N.S0.m : null, ba: t > 4 ? .5 : .85 });
     if (t > 4) { const X = v => 50 + (v + 1) / 2 * 300; box(X(N.S0.m - .1), 300, X(N.S0.m + .1) - X(N.S0.m - .1), 80, C.teal, .18); txt(`${N.S0.near} walks within 0.1`, X(N.S0.m - .1) - 6, 318, { s: 14, c: TEAL, al: 'right', a: clamp(t - 4) }); }
     if (t > 6.5) txt(`cat: ${pc(N.S0.cat, NW)} of walks`, 60, 348, { s: 15, c: C.roseInk, al: 'left', a: clamp(t - 6.5) }); } },
 { get p() { const N = nA(); return [`Force the first move from the same spot, then carry on. Up first, past the cat’s edge: average ${f2(N.Su.m)}. Down first, away from it: ${f2(N.Sd.m)}. Almost the same.`,
     `But up meets the cat in ${pc(N.Su.cat, NW)} of walks, down in only ${pc(N.Sd.cat, NW)}. Nearly equal averages, very different risks. Unforced, the network goes right here; down averages slightly lower because its way round is longer.`]; },
   h: 'Same average, different risk', shot: [4, 12],
   say: [[["Now force the first move.", [40, 25, 320, 135]], ["Up first, past the cat's edge.", [40, 25, 320, 135]], ["Then down first, away from it.", [40, 190, 320, 135]]], [["The two averages are almost the same.", [40, 25, 320, 300]], ["But up meets the cat far more often. Nearly equal averages, very different risks.", [40, 25, 320, 300]]]],
   draw(t) { if (busyKeys(['p0'], 200)) return; const N = nA(), mx = Math.max(...N.Su.h, ...N.Sd.h);
     txt('first move up', 50, 40, { s: 16, al: 'left', c: C.ink }); histo(N.Su, { x0: 50, x1: 350, y0: 150, hh: 70, max: mx, mean: t > 2 ? N.Su.m : null, a: clamp(t - .3) });
     txt('first move down', 50, 205, { s: 16, al: 'left', c: C.ink, a: clamp(t - 3) }); histo(N.Sd, { x0: 50, x1: 350, y0: 315, hh: 70, max: mx, mean: t > 4.5 ? N.Sd.m : null, a: clamp(t - 3) });
     if (t > 6) { txt(`cat ${pc(N.Su.cat, NW)}`, 60, 120, { s: 15, c: C.roseInk, al: 'left', a: clamp(t - 6) }); txt(`cat ${pc(N.Sd.cat, NW)}`, 60, 285, { s: 15, c: C.roseInk, al: 'left', a: clamp(t - 6.5) }); }
     caption(`from the spot (${SPOT[0]}, ${SPOT[1]}), ${NW} walks each`, clamp(t - 1)); } },
 { p: ['**Distributional** RL learns the whole histogram for every move (Marc Bellemare, Will Dabney and Rémi Munos, 2017). Their **C51** network gives each move the chances of 51 fixed return values.',
     'Target: Pip moved right, reward 0; take the landing spot’s histogram, squeeze it toward 0 by 0.9, add the reward. Ours come from walks; real C51 learns them, using the best move where it landed.'],
   h: 'Learning a histogram', shot: [4, 12],
   say: ["Distributional RL learns the whole histogram, for every move. The C51 network gives each move the chances of fifty one return values.", [["Here's a target. Pip moved right, and landed here.", [72, 14, 256, 256]], ["Take the histogram from where it landed.", [40, 285, 320, 130]], ["Squeeze it toward zero by zero point nine, and add the reward.", [40, 285, 320, 130]]]],
   draw(t) { if (busyKeys(['p0'], 200)) return; const N = nA(), mx = Math.max(...N.Sl.h), u = ease((t - 4) / 2), q = lerp(1, .9, u);
     D.meadow({ t }); dimScene(.55); spotMark(SPOT[0], SPOT[1], ''); spotMark(N.land.x, N.land.y, 'landed', 1, TEAL);
     box(84, 226, 232, 38, C.bg, .85); txt('histograms: measured from walks, 1 move in 5 random', W / 2, 236, { s: 13, c: C.ink }); txt('real C51: learned, target from the best move', W / 2, 254, { s: 13, c: C.soft });
     arrow(D.px(SPOT[0]) + 6, D.py(SPOT[1]), D.px(N.land.x) - 6, D.py(N.land.y), 1300, { c: C.ink, w: 1.4 });
     if (t > 4) histo(N.Sl, { x0: 50, x1: 350, y0: 380, hh: 70, max: mx, ba: .2 }); histo(N.Sl, { x0: 50, x1: 350, y0: 380, hh: 70, max: mx, sq: q, a: clamp(t - 1.5) });
     txt(t < 4 ? 'returns from the landing spot' : 'squeezed by 0.9, plus reward 0: the target', W / 2, 296, { s: 15, c: t < 4 ? TEAL : C.goldInk, a: clamp(t - 1.5) });
     } },
  { h: 'Back onto the bins', p: ['The squeezed bars no longer sit on the fixed return values. So each is split between its two nearest fixed values, more to the nearer. Now the target is again one chance per fixed value.',
     'The network’s histogram for that move is then nudged toward the target. The loss, cross-entropy, measures how far apart the histograms are, as squared error did for two numbers.'],
   shot: [4, 11],
   say: [[["Now the squeezed bars sit between the fixed return values.", [40, 45, 320, 165]], ["So each one is split between its two neighbours, more to the nearer one.", [40, 45, 320, 165]]], [["Then the network's histogram is nudged toward that target.", [30, 260, 340, 87]], ["The loss measures how far apart the two histograms are.", [30, 260, 340, 87]]]],
   draw(t) { if (busyKeys(['p0'], 200)) return; const N = nA(), P = proj(N.Sl.h, .9), mx = Math.max(...N.Sl.h, ...P), u = clamp((t - 3) / 1.5);
     txt('squeezed bars (faint) split back onto the fixed bins', W / 2, 60, { s: 15, c: C.goldInk });
     histo(N.Sl, { x0: 50, x1: 350, y0: 200, hh: 100, max: mx, sq: .9, ba: lerp(.85, .25, u) });
     if (t > 3) histo({ h: P }, { x0: 50, x1: 350, y0: 200, hh: 100, max: mx, ba: .85 * u });
     card(['squeezed bar between two fixed values:', { s: '   split it, more to the nearer one', c: C.goldInk }, { s: 'loss: how far the histograms are apart', c: C.soft }], t - 5, { y: 260, gap: 1.2 }); } },
{ h: 'Why it helps', p: ['Pip still picks moves by the average, which the histogram contains. Yet on Atari, learning the whole shape scored clearly better than the average alone. Why is still debated; perhaps the histogram is a much richer lesson per card.', 'It also opens the door to **risk-aware** choices, like preferring the move that rarely meets the cat, though C51 itself did not do that.'],
   shot: [3, 10],
   say: [["Pip still picks moves by the average. The histogram contains it.", [30, 60, 340, 110]], [["Yet on Atari, learning the whole shape scored clearly better. Why is still debated.", [30, 60, 340, 110]], "Perhaps each card simply teaches more."], ["It also opens the door to risk aware choices.", [40, 285, 320, 130]]],
   draw(t) { if (busyKeys(['p0'], 330)) return; const N = nA(), mx = Math.max(...N.Su.h, ...N.Sd.h); D.meadow({ t }); dimScene(.6);
     card(['one number per move: the average', { s: 'C51: 51 chances per move, the whole shape', c: C.goldInk }, 'choose moves: still by the average', { s: 'why it helps: still debated', c: C.soft }], t - .5, { y: 60, gap: 1.2 });
     histo(N.S0, { x0: 50, x1: 350, y0: 380, hh: 60, mean: N.S0.m, a: clamp(t - 5) }); } },
 { h: 'Your turn', p: ['**Tap any spot** on the meadow. Pip walks from there 400 times, following its network with one move in five random, and the histogram of returns appears below, with its average.', 'Try spots in open grass, then spots touching the cat’s circle, then spots right next to the cheese. Where is the average a fair summary, and where does it hide a cat?'],
   shot: [3, 8],
   tap(x, y, st) { const wx = (x - D.FX) / D.FS, wy = 1 - (y - D.FY) / D.FS; if (wx < 0 || wx > 1 || wy < 0 || wy > 1 || D.zone(wx, wy)) return; st.tap = [wx, wy]; st.sum = null; st.tt = null; },
   say: [[["Your turn. Tap any spot on the meadow.", [72, 14, 256, 256]], ["Pip walks from there four hundred times, and the histogram appears below.", [40, 285, 320, 130]]], [["Try open grass, then spots touching the cat's circle.", [72, 14, 256, 256]], "Where is the average a fair summary, and where does it hide a cat?"]],
   draw(t, st) { if (busyKeys(['p0'], 330)) { D.meadow({ t }); return; } const S = tapSummary(st); if (S && st.tt == null) st.tt = t; D.meadow({ t });
     if (!S) { txt('tap a spot on the meadow', W / 2, 330, { s: 18, c: C.soft, a: .5 + .5 * Math.sin(t * 3) ** 2 }); return; }
     const u = t - st.tt; spotMark(st.tap[0], st.tap[1], `(${st.tap[0].toFixed(2)}, ${st.tap[1].toFixed(2)})`);
     histo(S, { x0: 50, x1: 350, y0: 380, hh: 62, n: clamp(Math.floor(u * 12), 0, NB), mean: u > 1.6 ? S.m : null });
     txt(`cheese ${S.ch} · cat ${S.cat} · out of time ${S.to}`, W / 2, 408, { s: 13, c: C.soft, a: clamp(u - 1.6) }); } },
 { h: 'All six', p: ['You have seen four of the six: Double DQN, dueling heads, distributional learning, and prioritised replay, mentioned in the replay reel.', 'The other two: **multi-step** targets use three real rewards, discounted, then 0.9 cubed × a guess. **Noisy networks** add random wiggles to the knobs, learn how big they should be, and pick greedily on the wiggled Q-values, instead of one random move in five.'],
   shot: [4, 12],
   say: [[["Here are all six upgrades.", [30, 60, 340, 198]], ["You've now seen four. Double DQN, prioritised replay, dueling heads, and distributional learning.", [30, 60, 340, 198]]], [["The other two. Multi step targets use a few real rewards before trusting a guess.", [30, 60, 340, 198]], ["And noisy networks explore with learned wiggles, instead of random moves.", [30, 60, 340, 198]]]],
   draw(t) { D.meadow({ t }); D.pipAt(.1, .1, 1, t); dimScene(.72);
     card(['Double DQN: live picks, copy scores', 'prioritised replay: surprising cards more often', 'dueling: value of the spot + advantage', 'multi-step: 3 real rewards, then a guess', 'distributional: the spread, not just the average', 'noisy nets: explore with learned wiggles'], t - .5, { y: 60, gap: 1.2, fs: 16, lh: 30 }); } },
 { h: 'What mattered', p: ['On 57 Atari games, Rainbow beat every one of its parts alone, and needed much less experience than DQN to reach DQN’s final scores.', 'Removing one part at a time, prioritised replay and multi-step targets hurt most, distributional learning next. Dueling and Double DQN changed the median game little (scores scaled to a human player’s), though some games cared. Small, compatible upgrades can add up.'],
   shot: [3, 10],
   say: [[["On fifty seven Atari games, Rainbow beat every one of its parts alone.", [30, 70, 340, 156]], "It also needed far less experience than DQN."], [["Remove one part at a time, and prioritised replay and multi step targets hurt most.", [30, 70, 340, 156]], "Small, compatible upgrades can add up."]],
   draw(t) { D.meadow({ t }); D.pipAt(.1, .1, 1, t); dimScene(.65);
     card(['Hessel and colleagues, 2017: 57 Atari games', 'all six together beat each alone', { s: 'remove one, biggest losses:', c: C.soft }, { s: '   prioritised replay, multi-step', c: C.roseInk }, { s: '   then distributional', c: C.roseInk }, { s: '   dueling, Double: small on the median', c: C.soft }], t - .5, { y: 70, gap: 1.2 }); } },
 { h: 'Recap', p: ['A **dueling network** splits each Q-value into the value of the spot, V, and the advantage of each move, A, with the average advantage subtracted so the split is unique. V learns from every card.', '**Distributional** RL learns the whole histogram of returns, which an average hides. **Rainbow** combined six upgrades to DQN, and together they worked better than any one alone.'],
   shot: [3, 9],
   say: ["So that's Rainbow.", ["A dueling network splits each Q-value into the spot's value and each move's advantage. V learns from every card.", [30, 100, 340, 110]], ["Distributional RL learns the whole spread of returns. And six upgrades, stacked together, beat any one alone.", [30, 100, 340, 110]]],
   draw(t) { D.meadow({ t }); D.pipAt(.1, .1, 1, t); dimScene(.5);
     card(['Q = V (spot) + A (move) − average A', 'V hears from every card', 'distributional: histogram of returns', { s: 'Rainbow: six upgrades, stacked', c: C.soft }], t - .3, { y: 100, gap: 1 }); } },
];
RLStory.register('rainbow', { title: 'Rainbow: Dueling & Distributional DQN', track: 'Deep RL', onCtx: c => { ctx = c; }, steps: STEPS, _sim: { nums, force, JOBS, summary, walksFrom } });
})();
