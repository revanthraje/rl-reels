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
// Goal-conditioned RL and Hindsight Experience Replay as a story: the cheese and cat leave the meadow, a flag marks a new
// goal every walk, and the reward is +1 only for landing right next to it. Without hindsight, Pip's memory is nearly all
// zeros; with HER, every failed walk is stored a second time as a success for the spot Pip actually reached.
// Every number comes from the seeded runs below (see notes/her.md). Training runs in background slices, as in replay.js.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, gauss, rline, rrect, txt, arrow, card, caption, dot, box, alpha } = L;
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

// ───── the goal meadow: same field, moves and wobble as the kit; no cheese, no cat. A flag at the goal (gx, gy).
// +1 on the step that lands within GR of the flag (ends the walk); otherwise 0. A walk is stopped after TM moves (not an ending).
const GR = 0.04, TM = 30, NH = 16, STEPS_N = 10000, CAP = 4000, KS = [0, 1, 2, 3];
const gfe = (x, y, gx, gy) => [2 * x - 1, 2 * y - 1, 2 * gx - 1, 2 * gy - 1, 2 * (gx - x), 2 * (gy - y)];
const GNAMES = ['x', 'y', 'goal x', 'goal y', 'gap x', 'gap y'];
const mv = (x, y, m, R) => [clamp(x + D.MOVES[m][0] * D.STEP + gauss(R) * D.WOB), clamp(y + D.MOVES[m][1] * D.STEP + gauss(R) * D.WOB)];
const hit = (x, y, gx, gy) => Math.hypot(x - gx, y - gy) < GR;
// a start and a goal, uniform on the field, at least 0.3 apart
function pair(R) { for (;;) { const a = [R(), R()], g = [R(), R()]; if (Math.hypot(a[0] - g[0], a[1] - g[1]) > .3) return [a, g]; } }
const TP = (() => { const R = rng(4343), o = []; for (let i = 0; i < 30; i++) o.push(pair(R)); return o; })();   // 30 fixed test pairs
const qG = (net, x, y, g) => Array.from(D.fwd(net, gfe(x, y, g[0], g[1])));
function greedyGo(net, a, g, seed) { const R = rng(seed), P = [a.slice()]; let [x, y] = a, ok = false;
  for (let t = 0; t < TM; t++) { [x, y] = mv(x, y, D.argmax(qG(net, x, y, g)), R); P.push([x, y]); if (hit(x, y, g[0], g[1])) { ok = true; break; } } return { P, ok }; }
const testN = net => TP.reduce((s, [a, g], i) => s + greedyGo(net, a, g, 9100 + i).ok, 0);
// DQN with the goal as input (replay CAP, minibatch 32, Adam 0.003, copy every 100 updates, 1 move in 5 random, learning
// from move 100). her: when a walk fails, every card of it is stored again with the goal set to where the walk ended.
function* trainGen(o) {
  const k = o.k ?? 0, Rm = rng(5100 + k), Ra = rng(6100 + k), net = D.newNet(6, NH, 4, 7100 + k); let tgt = D.copyNet(net);
  const buf = [], walks = [], ev = [], mem = {}; let ins = 0, upd = 0, nw = 0, ok = 0;
  const push = c => { if (buf.length < CAP) buf.push(c); else buf[ins % CAP] = c; ins++; };
  let [[x, y], g] = pair(Rm), t = 0, walk = [[x, y]], cards = [];
  for (let s = 0; s < STEPS_N; s++) {
    const f = gfe(x, y, g[0], g[1]), m = Ra() < .2 ? Math.floor(Ra() * 4) : D.argmax(Array.from(D.fwd(net, f)), Ra), [nx, ny] = mv(x, y, m, Rm), r = hit(nx, ny, g[0], g[1]) ? 1 : 0;
    cards.push([x, y, m, nx, ny]); walk.push([nx, ny]); push({ f, m, r, f2: gfe(nx, ny, g[0], g[1]), done: r > 0, h: 0 });
    if (s >= 100) { const items = []; for (let i = 0; i < 32; i++) { const b = buf[Math.floor(Ra() * buf.length)], next = b.done ? 0 : Math.max(...D.fwd(tgt, b.f2)); items.push({ f: b.f, k: b.m, target: b.r + D.GAM * next }); }
      D.batchStep(net, items, .003, 'adam'); upd++; if (upd % 100 === 0) tgt = D.copyNet(net); }
    t++;
    if (r || t >= TM) { nw++; ok += r; if (walks.length < 120) walks.push({ P: walk, M: cards.map(c => c[2]), g, ok: !!r, s });
      if (o.her && !r) { const e = [nx, ny]; cards.forEach(([a, b, mm, c, d]) => { const rr = hit(c, d, e[0], e[1]) ? 1 : 0; push({ f: gfe(a, b, e[0], e[1]), m: mm, r: rr, f2: gfe(c, d, e[0], e[1]), done: rr > 0, h: 1, fe: Math.min(e[0], e[1], 1 - e[0], 1 - e[1]) < .02 }); }); }
      [[x, y], g] = pair(Rm); t = 0; walk = [[x, y]]; cards = []; } else { x = nx; y = ny; }
    if (s + 1 === 2000) mem.n = buf.length, mem.win = buf.filter(c => c.r > 0).length, mem.hwin = buf.filter(c => c.r > 0 && c.h).length, mem.h = buf.filter(c => c.h).length, mem.hfence = buf.filter(c => c.r > 0 && c.h && c.fe).length, mem.flags = buf.map(c => c.r > 0 ? 1 : 0);
    if ((s + 1) % 500 === 0) ev.push(testN(net));
    if (s % 40 === 39) yield s / STEPS_N;
  }
  return { net, walks, ev, mem, nw, ok };
}
KS.forEach(k => { job('h' + k, () => trainGen({ k, her: true })); job('n' + k, () => trainGen({ k })); });
QUEUE.sort((a, b) => +a.slice(1) - +b.slice(1) || (a < b ? -1 : 1));
const HK = KS.map(k => 'h' + k), NK = KS.map(k => 'n' + k);
const avg = a => a.reduce((s, v) => s + v, 0) / a.length;
const MEMO = {}, part = (k, fn) => () => MEMO[k] || (MEMO[k] = fn());
// early walks of the plain run 1 (the first 100 walks of each plain run are counted)
const nE = part('e', () => { const r = force('n0'), early = KS.map(k => force('n' + k).walks.slice(0, 100).filter(w => w.ok).length);
  return { r, early, earlyAvg: avg(early), mem: force('n0').mem }; });
// the hindsight example: the first failed walk of HER run 1 that ended away from the fence, at least 0.3 from its start and
// its flag, passed no earlier spot within GR of its end (so in the copy only the last move pays +1), and rarely bumped the fence
const nH = part('h', () => { const r = force('h0'), w = r.walks.find(w => { const e = w.P[w.P.length - 1]; return !w.ok && e[0] > .05 && e[0] < .95 && e[1] > .05 && e[1] < .95 &&
  Math.hypot(e[0] - w.P[0][0], e[1] - w.P[0][1]) > .3 && Math.hypot(e[0] - w.g[0], e[1] - w.g[1]) > .3 && !w.P.slice(1, -1).some(p => hit(p[0], p[1], e[0], e[1])) &&
  w.P.slice(1).filter((p, i) => Math.hypot(p[0] - w.P[i][0], p[1] - w.P[i][1]) > .1).length >= 12; });
  return { r, w, end: w.P[w.P.length - 1], mem: r.mem }; });
const nR = part('r', () => { const evH = KS.map(k => force('h' + k).ev), evN = KS.map(k => force('n' + k).ev), mean = E => E[0].map((_, i) => avg(E.map(e => e[i])));
  const net = force('h0').net, demo = [0, 3, 7, 12, 18, 25].map(i => ({ a: TP[i][0], g: TP[i][1], ...greedyGo(net, TP[i][0], TP[i][1], 9100 + i) }));
  return { evH, evN, mH: mean(evH), mN: mean(evN), fH: evH.map(e => e[e.length - 1]), fN: evN.map(e => e[e.length - 1]), demo }; });

// ───── drawing helpers
const f1 = v => v.toFixed(1), fx = v => v.toFixed(2), ORG = '#c4581f', TEAL = '#3f8582', FLAG = '#d0533f';
function flag(x, y, a = 1, col = FLAG, ghost = false) { const X = D.px(x), Y = D.py(y); alpha(a, () => { D.dashCircle(X, Y, GR * D.FS, col, ghost ? .6 : 1);
  rline(X, Y, X, Y - 22, 1400 + Math.round(X), { c: C.ink, w: 1.5, once: true }); ctx.save(); ctx.globalAlpha = a * (ghost ? .45 : 1); ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(X, Y - 22); ctx.lineTo(X + 13, Y - 18); ctx.lineTo(X, Y - 13); ctx.closePath(); ctx.fill(); ctx.restore(); }); }
const field = (t, o = {}) => D.meadow(Object.assign({ t, cat: 0, cheese: 0 }, o));
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
const raceChart = (ser, o = {}) => chart(Object.assign({ x0: 64, x1: 356, y0: 250, y1: 40, lo: 0, hi: 30, xmax: 19, yt: [0, 15, 30], xl: ['500 moves', '10000'], title: 'test flags Pip reaches (of 30)', series: ser }, o));
// an experience card: goal, move, reward
function gCard(x, y, goal, m, r, a = 1, o = {}) { if (a <= 0) return; alpha(a, () => { box(x, y, 160, 46, o.bg || C.card); rrect(x, y, 160, 46, 1500 + Math.round(x + y), { c: o.c || C.line, w: 1.1 });
  txt(`goal (${fx(goal[0])}, ${fx(goal[1])})`, x + 80, y + 13, { s: 14, c: o.gc || C.ink }); txt(`move ${D.MN[m]}   reward ${r ? '+1' : '0'}`, x + 80, y + 32, { s: 14, c: r ? C.goldInk : C.soft }); }); }
// the replay memory as a grid, one box per 5 cards (gold: boxes standing for cards with reward +1)
// the replay memory as a grid, one box per 5 cards in the order they were stored; gold: the box holds at least one +1 card
function memGrid(M, t) { const n = Math.ceil(M.n / 5), cols = 40;
  for (let i = 0; i < n; i++) { const a = clamp((t - .5) * 200 - i); if (a <= 0) continue; const g = M.flags.slice(i * 5, i * 5 + 5).some(v => v); box(56 + (i % cols) * 7.2, 30 + Math.floor(i / cols) * 9, 5.6, 7, g ? C.gold : '#e4d8c7', a); }
  txt('each box: 5 cards, in order · gold: holds a +1 card', W / 2, 264, { s: 13, c: C.soft, a: clamp(t - 3) }); }
function walkAnim(P, t, rate = 6, o = {}) { const n = clamp(Math.floor(t * rate) + 1, 1, P.length); D.path(P, n, o); const p = P[n - 1], q = P[Math.max(0, n - 2)]; if (!o.nopip) D.pipAt(p[0], p[1], p[0] >= q[0] ? 1 : -1, t); return n >= P.length; }

const STEPS = [
 { h: 'Go to the flag', p: [`New job. The cheese and the cat leave the meadow. Instead, each walk a **flag** is planted at a random spot, the **goal**, and Pip should go there. Pip starts at a random spot at least 0.3 away.`, `Same moves and wobble as before. Landing within ${GR} of the flag pays +1 and ends the walk; every other move pays 0. A walk is stopped after ${TM} moves; that is not an ending, so the last move’s target still adds 0.9 × a guess. γ is still 0.9.`],
   shot: [3, 10],
   draw(t) { field(t); const [a, g] = TP[0]; flag(g[0], g[1], clamp(t - .8)); D.pipAt(a[0], a[1], 1, t);
     if (t > 2) txt('goal', D.px(g[0]) + 16, D.py(g[1]) - 20, { s: 14, c: FLAG, al: 'left', a: clamp(t - 2) });
     card([`+1: land within ${GR} of the flag (walk ends)`, 'every other move: 0', { s: `stopped after ${TM} moves: not an ending`, c: C.soft }], t - 3, { gap: 1.2 }); } },
 { h: 'The goal is an input', p: ['A new flag every walk means one fixed set of Q-values cannot work: “up” is good for a flag above Pip and bad for one below. So the goal becomes part of what the network sees.', `Six inputs: where Pip is, where the flag is, and the gap between them, across and up. The network could work out the gap itself; handing it over makes this small network’s job easier. ${NH} hidden units, four Q-values out. One network for every goal: a **goal-conditioned** network.`],
   shot: [4, 11],
   draw(t) { field(t); dimScene(.8); const n = D.newNet(6, NH, 4, 7100);
     D.drawNet(n, { x: 150, y: 30, w: 130, h: 220 }, { inL: GNAMES, outL: D.MN, a: clamp(t - .5), build: clamp((t - .5) / 2), wa: .5 });
     card(['in: Pip’s spot, the flag’s spot, the gap', 'out: one Q-value per move', { s: 'same network, any flag', c: C.goldInk }], t - 4, { gap: 1.2 }); } },
 { get p() { const N = nE(); return [`At first Pip’s network knows nothing, so its moves are aimless. The flag’s circle is tiny, only ${GR} from its centre to its edge, so aimless walks almost never land in it. Here are the first walks of a training run, one after another.`,
     `In the first 100 walks, Pip reached the flag ${N.early[0]} times in this run; averaged over four runs, ${f1(N.earlyAvg)} times. Rewards this rare are called **sparse**.`]; },
   h: 'Almost never there', shot: [5, 13],
   draw(t) { if (busyKeys(['n0'], 330)) { field(t); return; } const N = nE(), i = clamp(Math.floor(t / 2.2), 0, 5), w = N.r.walks[i], u = t - i * 2.2; field(t);
     flag(w.g[0], w.g[1]); walkAnim(w.P, u, 14, { c: w.ok ? C.gold : C.teal }); const e = w.P[w.P.length - 1];
     if (u > w.P.length / 14 + .2) txt(w.ok ? 'reached!' : 'missed', D.px(e[0]), D.py(e[1]) + 18, { s: 15, c: w.ok ? C.goldInk : C.roseInk });
     txt(`walk ${i + 1}` + (w.ok ? ': one of the lucky ones' : ''), 40, 300, { s: 16, al: 'left' });
     card([`reached in the first 100 walks: ${N.early[0]}`, { s: `four runs: ${N.early.join(', ')}`, c: C.soft }], t - 9, { y: 316, gap: 1.2 }); } },
 { get p() { const N = nE(); return [`Look inside Pip’s replay memory after 2000 moves: ${N.mem.n} cards, and only ${N.mem.win} of them carry a reward of +1. Every other target is 0 + 0.9 × a guess, and the guesses start near 0.`,
     `So nearly every lesson just copies one guess from another, with no real reward behind it. The ${N.mem.win} successes are drowned out, and almost nothing tells the network which way leads to a flag.`]; },
   h: 'A memory full of zeros', shot: [4, 11],
   draw(t) { if (busyKeys(['n0'], 200)) return; const N = nE(); field(t); dimScene(.82);
     const M = N.mem;
     memGrid(M, t);
     card([`${N.mem.n} cards after 2000 moves`, { s: `reward +1: ${N.mem.win}`, c: C.goldInk }, { s: 'all the rest: target 0 + 0.9 × guess', c: C.soft }], t - 4, { gap: 1.2 }); } },
 { get p() { const N = nR(); return [`Four training runs, ${STEPS_N} moves each. Every 500 moves, Pip is tested on 30 fixed start-and-flag pairs, following its largest Q-value. The chart counts the flags it reaches, out of 30.`,
     `After ${STEPS_N} moves the four runs reach ${N.fN.join(', ')} flags. One run broke through late; the other three stayed low.`]; },
   h: 'Training on zeros', shot: [5, 12],
   draw(t) { if (busyKeys(NK)) return; const N = nR(), n = clamp(Math.floor((t - .5) * 3), 0, 20), ser = KS.map(k => ({ v: N.evN[k], col: ORG, w: 1.6, a: .75, n }));
     raceChart(ser); card([{ s: 'orange: four training runs', c: ORG }, { s: 'reward only at the flag', c: C.soft }], t - 1, { y: 300, gap: 1.1 }); } },
 { get p() { const N = nH(); return [`Here is one failed walk from training. The flag was at (${fx(N.w.g[0])}, ${fx(N.w.g[1])}); after ${N.w.P.length - 1} moves Pip was stopped at (${fx(N.end[0])}, ${fx(N.end[1])}), well outside the flag’s circle.`,
     'Now ask a question in **hindsight**: what if the flag had been where Pip ended? Then the very same walk would have been a success. Nothing about the moves changes; only the goal written on the cards.']; },
   h: 'A miss, in hindsight', shot: [5, 13],
   draw(t) { if (busyKeys(['h0'], 330)) { field(t); return; } const N = nH(); field(t);
     flag(N.w.g[0], N.w.g[1], t > 7 ? .35 : 1); const done = walkAnim(N.w.P, t - .5, 7, { c: C.teal });
     if (t > 7) { flag(N.end[0], N.end[1], clamp(t - 7), C.gold, true); txt('hindsight goal', D.px(N.end[0]), D.py(N.end[1]) - 34, { s: 15, c: C.goldInk, a: clamp(t - 7.5), al: N.end[0] < .3 ? 'left' : 'center' }); }
     card([{ s: 'real goal: missed, every reward 0', c: C.roseInk }, { s: 'goal = where Pip ended: a success', c: C.goldInk }], t - 8.5, { y: 300, gap: 1.3 }); } },
 { get p() { const N = nH(); return ['This is **Hindsight Experience Replay**, or HER (Marcin Andrychowicz and colleagues, OpenAI, 2017). Each card of a failed walk is stored twice: with the real goal and reward 0, and with the goal set to where the walk ended.',
     `In the copy, every move landing within ${GR} of the new goal pays +1 and ends it: always the last move, plus any earlier one that came that close (here, none). The rest pay 0 but bootstrap from the +1, as in TD.`]; },
   h: 'Store it twice', shot: [4, 12],
   draw(t) { if (busyKeys(['h0'], 200)) return; const N = nH(), w = N.w, n = w.P.length - 1, last = n - 1; field(t); dimScene(.45);
     D.path(w.P, w.P.length, { c: C.teal, a: .7 }); flag(w.g[0], w.g[1], .5); flag(N.end[0], N.end[1], 1, C.gold, true);
     const mOf = i => w.M[i];
     txt('last move, as stored:', 36, 296, { s: 14, c: C.soft, al: 'left', a: clamp(t - 1) });
     gCard(30, 306, w.g, mOf(last), 0, clamp(t - 1.5)); gCard(210, 306, N.end, mOf(last), 1, clamp(t - 3.5), { c: C.gold, gc: C.goldInk });
     txt('real goal', 110, 364, { s: 13, c: C.roseInk, a: clamp(t - 1.5) }); txt('hindsight copy', 290, 364, { s: 13, c: C.goldInk, a: clamp(t - 3.5) });
     caption(`and so on for all ${n} moves of the walk`, clamp(t - 6)); } },
 { h: 'Why that is not cheating', p: ['The copied cards tell the truth: from those spots, those moves really did lead to that place. So Pip learns, for real, how to reach that spot.', 'Early walks often end against the fence, so many hindsight goals sit there, gold below. That still helps: the network reads the **gap** to the goal, so practice on any goal teaches the skill the real flags need, shrinking the gap.'],
   shot: [4, 11],
   draw(t) { if (busyKeys(['h0'], 330)) { field(t); return; } const N = nH(); field(t); const ws = N.r.walks.filter(w => !w.ok).slice(0, 24);
     ws.forEach((w, i) => { const a = clamp(t - .5 - i * .2); if (a <= 0) return; const e = w.P[w.P.length - 1]; dot(D.px(e[0]), D.py(e[1]), 4, C.gold, a); });
     txt('gold: where 24 early failed walks ended', W / 2, 296, { s: 14, c: C.goldInk, a: clamp(t - 3) });
     card(['the moves really reached those spots', { s: 'skill learned: shrink the gap to a goal', c: C.goldInk }], t - 5, { y: 316, gap: 1.2 }); } },
 { get p() { const N = nH(); return [`With HER, after 2000 moves: ${N.mem.n} cards. Only failed walks are copied: ${N.mem.h / TM} walks × ${TM} moves = ${N.mem.h} copies. ${N.mem.win} cards carry +1 (without HER: ${nE().mem.win}). ${N.mem.hwin} are copies, about ${Math.round(N.mem.hwin / (N.mem.h / TM))} per copied walk, because Pip often lingered near its end spot: ${N.mem.hfence} of them come from walks that ended pinned against the fence.`,
     'Real successes stay rare at first, but the network now gets plenty of lessons that end in +1, and they teach it how to steer toward a goal.']; },
   h: 'A memory with successes', shot: [4, 11],
   draw(t) { if (busyKeys(['h0', 'n0'], 200)) return; const N = nH(), M = N.mem; field(t); dimScene(.82);
     memGrid(M, t);
     card([`${M.n} cards, ${M.h} of them hindsight copies`, { s: `reward +1: ${M.win}  (without HER: ${nE().mem.win})`, c: C.goldInk }], t - 4, { gap: 1.2 }); } },
 { get p() { const N = nR(); return [`Now the race: four runs with HER in teal, the same four runs without it in orange, same seeds, same number of updates. Each minibatch draws 32 cards at random from the whole memory, real cards and copies alike. Thin lines: single runs; thick: averages.`,
     `After ${STEPS_N} moves, with HER the runs reach ${N.fH.join(', ')} of 30 flags, ${f1(avg(N.fH))} on average. Without it, ${f1(avg(N.fN))}.`]; },
   h: 'The race', shot: [5, 13],
   draw(t) { if (busyKeys(NK.concat(HK))) return; const N = nR(), n = clamp(Math.floor((t - .5) * 3), 0, 20), ser = [];
     KS.forEach(k => { ser.push({ v: N.evN[k], col: C.orange, w: 1, a: .35, n }); ser.push({ v: N.evH[k], col: C.teal, w: 1, a: .35, n }); });
     ser.push({ v: N.mN, col: ORG, w: 2.6, n }, { v: N.mH, col: TEAL, w: 2.6, n }); raceChart(ser);
     card([{ s: 'teal: with HER', c: TEAL }, { s: 'orange: without', c: ORG }, { s: 'thin: single runs · thick: average of 4', c: C.soft }], t - 1, { y: 296, gap: 1.1 }); } },
 { get p() { const N = nR(), k = N.demo.filter(d => d.ok).length; return ['Here is the HER network of run 1 after training, sent to six of the test flags in turn. It never trained on these exact pairs; it reads the gap and steers.',
     `It reaches ${k} of these 6. Near a flag it sometimes overshoots: one move covers 0.15, while the flag’s circle reaches only ${GR} from its centre, so the last step has to land just right.`]; },
   h: 'Any flag', shot: [6, 14],
   draw(t) { if (busyKeys(['h0'], 330)) { field(t); return; } const N = nR(), i = clamp(Math.floor(t / 2.6), 0, 5), d = N.demo[i], u = t - i * 2.6; field(t);
     N.demo.slice(0, i).forEach(e => D.path(e.P, e.P.length, { c: e.ok ? C.gold : C.grey, a: .25 }));
     flag(d.g[0], d.g[1]); walkAnim(d.P, u, 8, { c: TEAL }); if (u > d.P.length / 8 + .2) { const e = d.P[d.P.length - 1]; txt(d.ok ? 'reached' : 'not in 30 moves', D.px(e[0]), D.py(e[1]) + 18, { s: 15, c: d.ok ? C.goldInk : C.roseInk }); }
     txt(`test flag ${i + 1} of 6`, 40, 300, { s: 16, al: 'left' }); } },
 { h: 'Your turn', p: ['**Tap the meadow** to plant a flag. Pip starts bottom left and follows the HER network’s largest Q-value, no random moves.', 'Try flags far away, near the edges, and close by. The network was never told about your flag; it only reads where it is.'],
   shot: [3, 8],
   tap(x, y, st, t) { const wx = (x - D.FX) / D.FS, wy = 1 - (y - D.FY) / D.FS; if (wx < 0 || wx > 1 || wy < 0 || wy > 1 || Math.hypot(wx - .1, wy - .1) < .1) return; st.g = [wx, wy]; st.go = null; st.tt = t; st.n = (st.n || 0) + 1; },
   draw(t, st) { if (busyKeys(['h0'], 330)) { field(t); return; } field(t); const net = force('h0').net;
     if (!st.g) { D.pipAt(.1, .1, 1, t); txt('tap the meadow to plant a flag', W / 2, 320, { s: 18, c: C.soft, a: .5 + .5 * Math.sin(t * 3) ** 2 }); return; }
     if (!st.go) st.go = greedyGo(net, [.1, .1], st.g, 500 + st.n); flag(st.g[0], st.g[1]); const done = walkAnim(st.go.P, t - st.tt, 7, { c: TEAL });
     if (done) txt(st.go.ok ? `reached in ${st.go.P.length - 1} moves` : `not reached in ${TM} moves`, W / 2, 320, { s: 18, c: st.go.ok ? C.goldInk : C.roseInk }); } },
 { h: 'Where it’s used', p: ['Andrychowicz and colleagues used HER with DDPG, an off-policy learner for continuous moves, where an actor network proposes the move (see the DDPG reel), on a simulated robot arm: pushing a block, sliding a puck, picking up and placing an object, with reward only on success. Without HER, plain DDPG failed to learn these tasks; with it, the arm learned all three.', 'A policy trained in simulation for picking and placing also worked on a real robot arm. Feeding the goal in as an input goes back to Tom Schaul and colleagues, 2015.'],
   shot: [3, 10],
   draw(t) { field(t); dimScene(.6);
     card(['2015  goal as an input (Schaul and others)', '2017  HER (Andrychowicz and others, OpenAI)', '      robot arm: push, slide, pick and place', { s: '      reward only on success; DDPG + HER', c: C.soft }], t - .5, { y: 90, gap: 1.2 }); } },
 { h: 'Limits', p: ['HER needs goals that are places you can reach, so that wherever a walk ends can be written down as a goal. For a goal like “win the game”, an accidental ending is rarely useful.', 'It also needs an **off-policy** learner like DQN or DDPG, since the copied cards were made while chasing another goal. Here each copy uses the walk’s final spot; the paper did even better relabelling each card with spots the same walk reached after that card’s move, its “future” strategy.'],
   shot: [3, 10],
   draw(t) { field(t); dimScene(.6);
     card([{ s: 'needs: any reached state can be a goal', c: C.ink }, { s: 'needs: off-policy learning (DQN, DDPG)', c: C.ink }, { s: 'here: goal = final spot of the walk', c: C.soft }, { s: 'paper’s best: spots reached after each move', c: C.soft }], t - .5, { y: 90, gap: 1.2 }); } },
 { h: 'Recap', p: ['A **goal-conditioned** network gets the goal as input, so one network serves every goal. With a reward only at the goal, early walks almost never succeed, and the memory fills with zeros.', '**Hindsight Experience Replay** stores each failed walk again as a success for the spot it actually reached. The moves really got there, so the lessons are true, and they teach the network to steer.'],
   shot: [3, 9],
   draw(t) { field(t); dimScene(.5);
     card(['goal = part of the input', 'sparse reward: nearly every card says 0', 'HER: relabel a miss with where it ended', { s: 'needs reachable goals, off-policy learning', c: C.soft }], t - .3, { y: 100, gap: 1 }); } },
];
RLStory.register('her', { title: 'Goal-Conditioned RL & HER', track: 'Deep RL', onCtx: c => { ctx = c; }, steps: STEPS, _sim: { nE, nH, nR, force, JOBS, TP } });
})();
