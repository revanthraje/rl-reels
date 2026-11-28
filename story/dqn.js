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
// DQN as a story: Pip on the meadow wants the best moves, not just its habit's values. Q-learning needs one number
// per spot and move: impossible here, so a network outputs four Q-values. Naive online learning flounders; DQN's two
// fixes (replay memory, target network) make the same network learn. Numbers: seeded kit runs (k = 0), see notes/dqn.md.
(function () {
'use strict';
const L = RLStory.lib, D = RLStory.deeprl, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const { px, py, FX, FY, FS, CHEESE, CAT, GAM } = D;
const K = 0, STEPS_N = 8000, SNAP = 250, MN = ['up', 'right', 'down', 'left'], MCOL = [C.teal, C.gold, C.blue, C.orange];
const memo = {}, M = (k, f) => (k in memo ? memo[k] : (memo[k] = f()));
// ───── the two training runs (same meadow and agent dice, k = 0)
// online: no memory, no target copy; one nudge per move from that move only, learning from move 1
const ONLINE = () => M('on', () => { let pre = null; const R = { first: null, ex: null, ratios: [], pos: [] };
  R.run = D.dqn({ k: K, replay: 0, targetEvery: 0, warm: 0, probe: null, onMove(i, ph) {
    if (ph === 'pre') { pre = { q: D.qOf(i.net, i.x, i.y), qn: D.qOf(i.net, i.nx, i.ny) }; R.pos.push([i.x, i.y]); return; }
    const q = D.qOf(i.net, i.x, i.y), qn = D.qOf(i.net, i.nx, i.ny);
    if (!R.first && i.r !== 0) R.first = { x: i.x, y: i.y, m: i.m, rnd: i.rnd, nx: i.nx, ny: i.ny, r: i.r, s: i.s, q0: pre.q, q1: q };
    if (i.done) return; const dq = q[i.m] - pre.q[i.m], dt = GAM * (Math.max(...qn) - Math.max(...pre.qn));
    if (Math.abs(dq) > 1e-4) R.ratios.push(dt / dq);
    if (!R.ex && i.s >= 1000 && dq > .004 && dt / dq > .5 && dt / dq < 1 && pre.q[i.m] < i.r + GAM * Math.max(...pre.qn) - .02 && i.x > .15 && i.x < .85 && i.y > .15 && i.y < .85 && i.nx > .15 && i.nx < .85 && i.ny > .15 && i.ny < .85)
      R.ex = { s: i.s, x: i.x, y: i.y, nx: i.nx, ny: i.ny, m: i.m, q0: pre.q[i.m], q1: q[i.m], t0: GAM * Math.max(...pre.qn), t1: GAM * Math.max(...qn) }; } });
  const r = R.ratios.slice().sort((a, b) => a - b); R.med = r[r.length >> 1]; R.same = R.ratios.filter(v => v > 0).length / R.ratios.length; R.n = R.ratios.length;
  R.ev = R.run.snaps.map(s => s.ev.cheese); return R; });
// DQN: replay memory 2000, minibatch 32, target copy every 100 updates, learning from move 101 (kit defaults)
const FULL = () => M('full', () => { const pos = []; const r = D.dqn({ k: K, onMove(i, ph) { if (ph === 'pre') pos.push([i.x, i.y]); } }); r.pos = pos; r.ev = r.snaps.map(s => s.ev.cheese); r.ret = r.snaps.map(s => s.ev.ret); return r; });
const NET0 = D.newNet(4, 8, 4, 7000 + K);                              // the untrained network (both runs start here)
const med = v => v.slice().sort((a, b) => a - b)[v.length >> 1];
// describe the end of the simple run's test record honestly (it differs by seed)
const lastJump = ev => { const L = ev[ev.length - 1], mx = Math.max(...ev.slice(0, -1)); return L > mx ? `Only the very last test, after ${n0(STEPS_N)} moves, scores ${L}: one good test after ${ev.length - 1} poor ones is nothing to rely on.` : `Its best, ${mx} of 30, does not last: ${L} at the end.`; };
const BELOWUP = () => M('bu', () => { let up = 0, n = 0; for (let i = 0; i < 20; i++) for (let j = 0; j < 20; j++) { const x = (i + .5) / 20, y = (j + .5) / 20; if (x > .5 && x < .7 && y < .48) { n++; if (D.argmax(D.qOf(FULL().net, x, y)) === 0) up++; } } return { up, n }; });
const AVG12 = () => M('a12', () => { let s = 0; for (let k = 0; k < 200; k++) { const w = D.greedyWalk(FULL().net, S0[0], S0[1], 20000 + k); s += w.end > 0 ? w.P.length - 1 : D.TMAX; } return s / 200; });
const firstAt = (ev, n) => ev.findIndex(v => v >= n);                  // first snapshot with at least n cheeses
// ───── numbers
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−'), f3 = v => (Math.abs(v) < .0005 ? 0 : v).toFixed(3).replace('-', '−'), fx = v => v.toFixed(2), n0 = n => n.toLocaleString('en-US');
const S0 = D.START, Q0 = D.qOf(NET0, S0[0], S0[1]);

// ───── drawing helpers
function axes(a = 1) { txt('x →', px(.5), FY + FS + 10, { s: 13, c: C.soft, a }); txt('y ↑', FX - 16, py(.52), { s: 13, c: C.soft, a }); }
function ring(x, y, col, a = 1, r = 9) { alpha(a, () => rcircle(px(x), py(y), r, 700, { c: col, w: 2 })); }
function tag(s, x, y, a = 1, col, o = {}) { const X = px(x) + (o.dx ?? 0), Y = py(y) + (o.dy ?? -14); box(X - (o.w ?? 34), Y - 9, 2 * (o.w ?? 34), 18, C.card, .9 * a); txt(s, X, Y, { s: o.s || 13, c: col || C.ink, a }); }
// best-move arrows of a network on an n × n grid, and its best Q as the heat colour
const maxQ = n => (x, y) => Math.max(...D.qOf(n, x, y));
function qArrows(n, a = 1, g = 9) { if (a <= 0) return; for (let i = 0; i < g; i++) for (let j = 0; j < g; j++) { const x = (i + .5) / g, y = (j + .5) / g; if (D.zone(x, y)) continue;
  const m = D.argmax(D.qOf(n, x, y)), [dx, dy] = D.MOVES[m]; alpha(a * .8, () => arrow(px(x) - dx * 6, py(y) + dy * 6, px(x) + dx * 8, py(y) - dy * 8, 600 + i * g + j, { c: C.ink, w: 1.2, head: 4 })); } }
// four Q bars (up, right, down, left) in a box; hl = index highlighted; lo..hi value range
function qBars(q, x, y, o = {}) { const w = o.w ?? 200, h = o.h ?? 64, lo = o.lo ?? -1, hi = o.hi ?? 1, bw = w / 4, Y = v => y + h - (v - lo) / (hi - lo) * h, z = Y(0);
  rline(x - 4, z, x + w + 4, z, 930, { c: C.line, w: 1, once: true });
  q.forEach((v, k) => { const top = Math.min(Y(v), z), hh = Math.abs(Y(v) - z); box(x + k * bw + bw * .18, top, bw * .64, Math.max(1, hh), v >= 0 ? C.gold : C.rose, o.hl === k ? .95 : .55);
    if (o.hl === k) rrect(x + k * bw + bw * .18, top, bw * .64, Math.max(1, hh), 940 + k, { c: C.ink, w: 1.4 });
    txt(MN[k], x + k * bw + bw / 2, y + h + 22, { s: 13, c: C.soft }); txt((Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−'), x + k * bw + bw / 2, v >= 0 ? top - 9 : top + hh + 9, { s: 13, c: v < 0 ? C.roseInk : C.goldInk }); }); }
// chart: cheese reached from the 30 test spots, after every 250 moves of training
function evChart(series, n, o = {}) {
  const x0 = 64, x1 = 330, y0 = 388, y1 = 312, NS = STEPS_N / SNAP, X = i => x0 + (i + 1) / NS * (x1 - x0), Y = v => y0 - v / 30 * (y0 - y1);
  rline(x0 - 4, y0, x1 + 4, y0, 960, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1 - 4, 961, { c: C.line, w: 1, once: true });
  txt('0', x0 - 12, y0, { s: 12, c: C.soft }); txt('30', x0 - 16, y1, { s: 12, c: C.soft }); txt('move 0', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(n0(STEPS_N), x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
  txt(o.title || 'tests reaching the cheese (of 30)', x0 - 10, y1 - 13, { s: 13, c: C.soft, al: 'left' }); if (o.now) txt(o.now, (x0 + x1) / 2, y0 + 10, { s: 13, c: C.ink });
  series.forEach(s => { const m = Math.min(s.n ?? n, s.v.length); ctx.save(); ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 1.6; ctx.lineJoin = 'round'; ctx.beginPath();
    for (let i = 0; i < m; i++) { const x = X(i), y = Y(s.v[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore();
    for (let i = 0; i < m; i++) dot(X(i), Y(s.v[i]), 2, s.col); if (m === s.v.length && s.label) txt(s.label, x1 + 6, Y(s.v[m - 1]) + (s.dy || 0), { s: 13, c: s.col, al: 'left' }); });
}
// the 30 test spots and greedy walks from them
function testWalks(n, t, a = 1, upto = 30) { D.TESTS.slice(0, upto).forEach(([x, y], i) => { const w = M('gw' + n.__id + '_' + i, () => D.greedyWalk(n, x, y, 9000 + i)); const k = clamp(Math.floor(t * 3), 0, w.P.length);
  D.path(w.P, k, { c: w.end > 0 ? C.teal : w.end < 0 ? C.rose : C.grey, a: a * .8, r: 1.6 }); dot(px(x), py(y), 2.6, C.ink, a); }); }
const DEF_TP = [0.15, 0.45], A0W = 300; let NID = 0; const tagNet = n => (n.__id ??= ++NID, n);
// Atari sketch: a TV with bricks, paddle and ball, four stacked frames
function tv(x, y, s, t, k = 0) { const w = 92 * s, h = 70 * s; rrect(x, y, w, h, 1100 + k, { fill: '#2f3a44', c: C.ink, w: 1.4 });
  const cols = ['#e07a7a', '#ef8f5a', '#f6d58c', '#6fb7b4']; for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++) box(x + 6 * s + c * 10 * s, y + 8 * s + r * 5 * s, 9 * s, 4 * s, cols[r]);
  const bx = x + w / 2 + Math.sin(t * 1.3 + k * .4) * 30 * s, by = y + 42 * s - Math.abs(Math.cos(t * 1.3 + k * .4)) * 12 * s; dot(bx, by, 2.2 * s, '#fff');
  box(x + w / 2 - 12 * s + Math.sin(t * 1.3 + k * .4) * 24 * s, y + h - 9 * s, 24 * s, 3 * s, '#cfe3e2'); }

const STEPS = [
 { h: 'From values to choices', get p() { return ['In the last reel a tiny network learned the **values** of Pip’s meadow habit. But a fixed habit’s values don’t say how to do better.', 'The maze had **Q-learning**: a number Q per spot and move, “how good is this move here, if I play well afterwards”. On the meadow, where spots never repeat, a Q table is as impossible as a value table.']; },
   shot: [3, 9],
   say: [[["Back on the meadow.", [60, 8, 280, 270]], ["Last reel, a network learned how good each spot was, under Pip's habit.", [60, 8, 280, 270]]], [["But to do better, Pip needs a number for each move. Watch the four arrows.", [60, 8, 280, 270]], ["That's Q-learning. And spots never repeat, so a table won't work.", [30, 292, 340, 87]]]],
   draw(t) { D.meadow({ t, labels: 1 }); axes(.7); D.pipAt(S0[0], S0[1], 1, t);
     [0, 1, 2, 3].forEach(m => { const u = clamp(t - 1.5 - m * .4); if (u <= 0) return; const [dx, dy] = D.MOVES[m]; alpha(u, () => arrow(px(S0[0]) + dx * 10, py(S0[1]) - dy * 10, px(S0[0]) + dx * 30, py(S0[1]) - dy * 30, 1500 + m, { c: C.goldInk, w: 1.6, head: 5 })); });
     card(['values of a habit: how good is this spot', { s: 'Q-values: how good is each move from here', c: C.goldInk }, { s: 'a table? no spot ever repeats', c: C.roseInk }], t - 3, { gap: 1.3 }); } },
 { h: 'Q-learning, remembered', get p() { return ['After each move, Q-learning nudges that move’s number toward a **target**: the reward + 0.9 × the **largest** Q at the landing spot, or just the reward if the walk ended. Taking the largest means: assume Pip plays its best from there.', 'And Pip acts **ε-greedy**, said epsilon-greedy: one move in five at random, to keep exploring; otherwise the largest-Q move.']; },
   shot: [3, 9],
   say: [[["Quick reminder of Q-learning.", [30, 292, 340, 110]], ["After each move, nudge its number toward the reward, plus zero point nine times the best Q where Pip landed.", [30, 292, 340, 110]]], [["And Pip acts epsilon greedy. One move in five at random.", [30, 292, 340, 110]]]],
   draw(t) { D.meadow({ t, heatA: 0 }); D.pipAt(S0[0], S0[1], 1, t);
     card(['target = reward + 0.9 × largest Q at landing', '               (just the reward if the walk ended)', 'nudge Q(spot, move) toward the target', { s: 'act: 1 in 5 random, else the largest Q', c: C.soft }], t - .5, { gap: 1.3, y: 290 }); } },
 { h: 'One output per move', get p() { return ['The fix, as in the last reel: a network. This one has **four outputs**, one Q-value per move: up, right, down, left. Feed in Pip’s four measurements, and one pass gives all four. That is a **Deep Q-Network**, or DQN.', `Its 76 knobs start random. At the start spot it says: up ${f2(Q0[0])}, right ${f2(Q0[1])}, down ${f2(Q0[2])}, left ${f2(Q0[3])}. Pure noise, for now.`]; },
   shot: [3, 10],
   say: [[["The fix is a network, as in the last reel. Watch it build.", [90, 20, 230, 260]], ["This one has four outputs, one Q value per move.", [90, 20, 230, 260]]], [["One pass gives all four. That's a Deep Q Network, DQN.", [30, 292, 340, 87]], "Its knobs start random, so for now its numbers are pure noise."]],
   draw(t) { D.drawNet(NET0, { x: 110, y: 30, w: 170, h: 220 }, { inL: ['x', 'y', 'to cheese', 'to cat'], outL: MN, outFmt: f2, f: t > 2.5 ? D.feats(S0[0], S0[1]) : null, build: clamp((t - .3) / 2.5), hl: t > 5 ? D.argmax(Q0) : null });
     txt('measurements', 82, 268, { s: 13, c: C.soft }); txt('8 hidden units', 195, 268, { s: 13, c: C.soft }); txt('Q per move', 300, 268, { s: 13, c: C.soft });
     card(['4 inputs → 8 hidden → 4 outputs', `4 × 8 + 8 + 8 × 4 + 4 = 76 knobs`, { s: 'highest output = the move Pip would pick', c: C.soft }], t - 4); } },
 { h: 'The simple way: learn as you go', get p() { const F = ONLINE().first, q = F.q0, m = F.m, o = q.map((v, k) => k).filter(k => k !== m).sort((a, b) => q[b] - q[a])[0]; return [`First try: after each move, one nudge on its output. Move ${F.s + 1} was ${F.rnd ? 'a random step ' + MN[m] : 'a step ' + MN[m] + ', its untrained numbers’ favourite, by a hair: ' + f3(q[m]) + ' against ' + f3(q[o]) + ' for ' + MN[o]}. It went from (${fx(F.x)}, ${fx(F.y)}) into the cat: target −1.`, `One nudge took “${MN[m]}” from ${f3(q[m])} to ${f3(F.q1[m])}. Every knob moves only about 0.003 per nudge, so one bad moment can’t wipe out everything; the other outputs, sharing knobs, moved a little too.`]; },
   shot: [3, 10],
   say: [[["First, the simple way. One nudge after each move.", [60, 8, 280, 270]], ["Watch move thirty seven. A step left, straight into the cat.", [60, 8, 280, 270]]], [["One nudge lowers left a little.", [70, 290, 260, 108]], ["Each nudge is small, so one bad moment can't wipe out everything.", [70, 290, 260, 108]]]],
   draw(t) { const F = ONLINE().first, aft = t > 6; D.meadow({ t }); ring(F.x, F.y, C.ink, clamp(t - .5)); const u = ease(clamp((t - 1.5) / 1)); D.pipAt(lerp(F.x, F.nx, u), lerp(F.y, F.ny, u), 1, t);
     if (t > 2.4) bubble('−1', px(CAT[0]), py(CAT[1]) - 22, t - 2.4, C.roseInk);
     if (t > 3) { box(70, 290, 260, 108, C.card, .95); rrect(70, 290, 260, 108, 211, { c: C.line, w: 1.1 }); qBars(aft ? F.q1 : F.q0, 96, 318, { w: 210, h: 48, lo: -.7, hi: .7, hl: F.m }); txt(aft ? 'after one nudge' : `Q-values at (${fx(F.x)}, ${fx(F.y)}), before`, 200, 300, { s: 13, c: C.soft }); } } },
 { h: 'It flounders', get p() { return [`Pip learns this way for ${n0(STEPS_N)} moves, from random starts. Black dots: 30 fixed test spots, tried every 250 moves by largest Q, no random moves. The chart counts walks reaching the cheese.`, `It hardly learns: ${ONLINE().ev.filter(v => v <= 1).length} of ${ONLINE().ev.length} tests reach the cheese at most once. ${lastJump(ONLINE().ev)}`]; },
   shot: [4, 12],
   say: [[["Pip learns this way for eight thousand moves.", [60, 8, 280, 270]], ["Every so often, thirty test walks check how it's doing.", [60, 8, 280, 270]]], [["The chart counts walks that reach the cheese.", [20, 285, 360, 115]], ["It hardly learns. Mostly near zero.", [20, 285, 360, 115]]], "Only the very last test scores well. Nothing to rely on."],
   draw(t) { const R = ONLINE(), n = clamp(Math.floor((t - 1) * 3), 0, R.ev.length), net = tagNet(R.run.snaps[Math.max(0, n - 1)].net);
     D.meadow({ t, heat: maxQ(net), heatA: .7 }); if (t > 1) testWalks(net, 3, .9);
     evChart([{ v: R.ev, col: C.rose, n, label: String(R.ev[R.ev.length - 1]) }], n, { now: n ? `after ${n0(n * SNAP)} moves` : '' });
     txt('walks: teal = cheese, rose = cat, grey = neither', W / 2, 280, { s: 13, c: C.soft, a: clamp(t - 1) }); } },
 { h: 'Problem 1: a target that runs away', get p() { const E = ONLINE().ex; return ['Unlike a table, the target comes from the **same network**, at a spot only 0.15 away: nudge the guess up and the target rises too.', `Here the target moved a median ${f2(ONLINE().med)} times as far as the guess, the same way ${Math.round(ONLINE().same * 100)} percent of the time, so a nudge closes only a sliver of the gap: ${f3(E.t0 - E.q0)} to ${f3(E.t1 - E.q1)}. Frozen targets, the **target network** reel, mostly steady the numbers here; the next problem matters more.`]; },
   shot: [4, 11],
   say: [[["Problem one. The target comes from the same network.", [60, 8, 280, 270]], ["At a spot only a step away.", [60, 8, 280, 270]]], [["Watch the bars. Nudge the guess up, and the target rises too.", [40, 290, 320, 110]], ["So the gap barely closes.", [40, 290, 320, 110]]]],
   draw(t) { const E = ONLINE().ex, u = clamp((t - 3) / 1.2); D.meadow({ t }); D.pipAt(E.x, E.y, D.MOVES[E.m][0] || 1, t); ring(E.nx, E.ny, C.goldInk, clamp(t - 1));
     const [dx, dy] = D.MOVES[E.m]; alpha(clamp(t - .5), () => arrow(px(E.x) + dx * 12, py(E.y) - dy * 12, px(E.nx) - dx * 10, py(E.ny) + dy * 10, 1700, { c: C.goldInk, w: 1.6, head: 5 }));
     box(40, 290, 320, 110, C.card, .95); rrect(40, 290, 320, 110, 212, { c: C.line, w: 1.1 });
     const Yv = v => 384 - v / .3 * 60; rline(60, 384, 340, 384, 950, { c: C.line, w: 1, once: true });
     const g = lerp(E.q0, E.q1, u), tg = lerp(E.t0, E.t1, u);
     txt(`move ${n0(E.s + 1)}: Q(${fx(E.x)}, ${fx(E.y)}, ${MN[E.m]})`, 200, 300, { s: 13, c: C.soft });
     box(110, Yv(g), 50, 384 - Yv(g), C.gold, .7); txt(`guess ${f3(g)}`, 135, Yv(g) - 10, { s: 14, c: C.goldInk });
     ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = C.roseInk; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(220, Yv(tg)); ctx.lineTo(300, Yv(tg)); ctx.stroke(); ctx.restore(); txt(`target ${f3(tg)}`, 260, Yv(tg) - 10, { s: 14, c: C.roseInk });
     txt(`before: guess ${f3(E.q0)}, target ${f3(E.t0)}`, 200, 318, { s: 13, c: C.soft }); if (t > 4.4) txt(`one nudge: both went up · gap ${f3(E.t0 - E.q0)} → ${f3(E.t1 - E.q1)}`, 200, 394, { s: 13, c: C.soft }); } },
 { h: 'Problem 2: one moment at a time', get p() { return ['Second problem: moves arrive in order. The rose dots: 60 consecutive moves early in the next run, a few walks each hugging one patch or fence. The network bends toward each patch in turn, drifting elsewhere.', 'Random old moves from a memory, the blue dots, would mix the whole meadow into each lesson. That is **experience replay**, its own reel.']; },
   shot: [4, 11],
   say: [[["Problem two. Moves arrive in order.", [60, 8, 280, 270]], ["Watch the rose dots. Sixty moves in a row, hugging one patch.", [60, 8, 280, 270]]], [["The network bends toward each patch in turn.", [30, 312, 340, 64]], ["Random old moves, the blue dots, would mix the whole meadow into each lesson.", [60, 8, 280, 270]]]],
   draw(t) { const P = FULL().pos, a0 = A0W, n = clamp(Math.floor((t - .5) * 12), 0, 60); D.meadow({ t }); D.path(P.slice(a0, a0 + n), n, { c: C.rose, r: 2.6 });
     if (n > 0) caption(`moves ${a0 + 1} to ${a0 + 60}, in order`, clamp(t - .5));
     if (t > 6.5) { const R = rng(77), u = clamp(Math.floor((t - 6.5) * 10), 0, 32); for (let i = 0; i < u; i++) { const p = P[Math.floor(R() * (a0 + 60))]; dot(px(p[0]), py(p[1]), 3, C.blue, .9); } }
     card([{ s: 'in order: one patch at a time', c: C.roseInk }, { s: '32 random picks from all moves so far', c: '#4f74a0' }], t - 3, { y: 312, gap: 3.5 }); } },
 { h: 'DQN’s two fixes', get p() { return ['DeepMind’s DQN adds two things. A **replay memory** keeps the last 2000 moves; each move, Pip draws 32 at random, a **batch** that may include the newest, and makes one nudge from their average error.', 'A **target network**, a frozen copy refreshed every 100 nudges, so every 100 moves, computes targets. Each fix has its own reel; here both are on.']; },
   shot: [3, 10],
   say: [[["DQN adds two fixes.", [20, 30, 360, 130]], ["First, a replay memory. Pip learns from thirty two random old moves at a time.", [30, 40, 160, 110]]], [["Second, a target network. A frozen copy, refreshed now and then, sets the targets.", [210, 40, 160, 110]]], [["Same network, same knobs, same meadow.", [30, 300, 340, 64]]]],
   draw(t) { D.meadow({ t, heatA: 0, tufts: .5 }); box(0, 0, W, 285, C.bg, .55);
     const a = clamp(t - .6), b = clamp(t - 3);
     alpha(a, () => { rrect(30, 40, 160, 110, 1201, { fill: C.card }); txt('replay memory', 110, 56, { s: 16, c: '#4f74a0' }); for (let i = 0; i < 40; i++) box(42 + (i % 10) * 14, 70 + Math.floor(i / 10) * 16, 11, 12, C.blue, .25 + .5 * ((i * 37) % 11) / 11); txt('nudge with 32 random picks', 110, 140, { s: 13, c: C.soft }); });
     alpha(b, () => { rrect(210, 40, 160, 110, 1202, { fill: C.card }); txt('target network', 290, 56, { s: 16, c: C.roseInk }); txt('frozen copy for targets', 290, 92, { s: 14, c: C.soft }); txt('refreshed every 100 nudges', 290, 116, { s: 14, c: C.soft }); });
     card(['memory 2000 · batch 32 · copy every 100 moves', { s: 'same network, same 76 knobs, same meadow', c: C.soft }], t - 5.5, { y: 300 }); } },
 { h: 'Now it learns', get p() { return [`Same ${n0(STEPS_N)} moves, same dice: blue DQN, rose the simple version. DQN first scores 28 of 30 or better after ${n0((firstAt(FULL().ev, 28) + 1) * SNAP)} moves, and after ${n0(STEPS_N)} gets all ${FULL().ev[FULL().ev.length - 1]}.`, 'Effort was unequal: each DQN nudge averages 32 moves, a simple nudge one. The replay reel’s fair race still has memory winning clearly. Over 12 seeds DQN usually ended near 30, the simple one mostly far lower.']; },
   shot: [5, 12],
   say: [[["Now run it again, with both fixes. Blue is DQN.", [20, 285, 360, 115]], ["Watch the test walks turn teal.", [60, 8, 280, 270]]], [["By the end, all thirty reach the cheese.", [20, 285, 360, 115]], ["The faint rose line is the simple version, far behind.", [20, 285, 360, 115]]]],
   draw(t) { const R = FULL(), n = clamp(Math.floor((t - 1) * 3), 0, R.ev.length), net = tagNet(R.snaps[Math.max(0, n - 1)].net);
     D.meadow({ t, heat: maxQ(net), heatA: .7 }); if (t > 1) testWalks(net, 3, .9);
     evChart([{ v: ONLINE().ev, col: '#e7a9a9', w: 1.1, label: String(ONLINE().ev[ONLINE().ev.length - 1]), dy: 6 }, { v: R.ev, col: '#4f7fb0', n, label: String(R.ev[R.ev.length - 1]), dy: -6 }], n, { now: n ? `DQN after ${n0(n * SNAP)} moves` : '' });
     txt('walks: teal = cheese, rose = cat, grey = neither', W / 2, 280, { s: 13, c: C.soft, a: clamp(t - 1) }); } },
 { h: 'What the network learned', get p() { return [`Best move per grid spot; colour is the largest Q. From the bottom left: up the side, along the top; from the right: straight up. Mostly it rounds the cat, but ${BELOWUP().up} of ${BELOWUP().n} spots right below it still point up, into it.`, `From the start: ${M('gwS', () => D.greedyWalk(FULL().net, S0[0], S0[1], 9100)).P.length - 1} moves; over 200 wobbles, ${AVG12().toFixed(1)} on average. Five up and five right take about 10: good, not best. It learned this from moves and rewards alone.`]; },
   shot: [4, 11],
   say: [[["Here's what the network learned.", [60, 8, 280, 270]], ["Each arrow is its best move. Colour is how good that move looks.", [60, 8, 280, 270]]], [["Mostly it rounds the cat, though a few spots right below still point into it.", [60, 8, 280, 270]], ["Watch the teal walk from the start. Twelve moves. Good, not perfect.", [60, 8, 280, 270]]]],
   draw(t) { const n = FULL().net; D.meadow({ t, heat: maxQ(n), heatA: clamp(t - .3) }); qArrows(n, clamp(t - 1.5));
     const w = M('gwS', () => D.greedyWalk(n, S0[0], S0[1], 9100)); if (t > 4) D.path(w.P, clamp(Math.floor((t - 4) * 3), 0, w.P.length), { c: C.teal, r: 2.6 }); D.pipAt(S0[0], S0[1], 1, t);
     card(['arrow: the move with the largest Q', 'colour: that largest Q (gold = good)', { s: 'teal: Pip’s walk from the start, no random moves', c: C.soft }], t - 2, { y: 300 }); } },
 { h: 'Inside the network', get p() { const q = D.qOf(FULL().net, S0[0], S0[1]), n = M('gwS', () => D.greedyWalk(FULL().net, S0[0], S0[1], 9100)).P.length - 1; return [`From the start spot, the outputs read up ${f2(q[0])}, right ${f2(q[1])}, down ${f2(q[2])}, left ${f2(q[3])}. Pip picks ${MN[D.argmax(q)]}.`, `The ${n}-move walk it takes is worth 0.9 to the power ${n - 1}, about ${f2(GAM ** (n - 1))}, less than its ${f2(Math.max(...q))}. Left, a wasted fence bump, scores above right, a good first move. It ranks the best move right here, but the numbers are only rough.`]; },
   shot: [3, 10],
   say: [[["Inside the trained network, with Pip at the start.", [90, 20, 230, 260]], ["Up scores highest, so Pip goes up.", [70, 296, 260, 100]]], [["But the numbers are rough. Left, a wasted bump into the fence, scores above right.", [70, 296, 260, 100]], "It ranks the best move here, but the values are only approximate."]],
   draw(t) { const n = FULL().net, q = D.qOf(n, S0[0], S0[1]); D.drawNet(n, { x: 110, y: 30, w: 170, h: 220 }, { inL: ['x', 'y', 'to cheese', 'to cat'], outL: MN, outFmt: f2, f: D.feats(S0[0], S0[1]), hl: t > 3 ? D.argmax(q) : null });
     txt('trained network, Pip at the start', W / 2, 272, { s: 14, c: C.soft });
     if (t > 4) { box(70, 296, 260, 100, C.card, .95); rrect(70, 296, 260, 100, 213, { c: C.line, w: 1.1 }); qBars(q, 96, 314, { w: 210, h: 54, lo: 0, hi: .6, hl: D.argmax(q) }); } } },
 { h: 'Your turn', ctl: { label: 'trained for', values: [1, 4, 8, 16, 32], def: 4, fmt: v => n0(v * SNAP) + ' moves' },
   get p() { return ['**Tap a spot** on the meadow to drop Pip there. It walks by its network’s largest Q, no random moves. Teal ends at the cheese, rose at the cat, grey runs out of moves.', 'Slide to pick how long DQN trained. Early networks wander, bump the fence or walk into the cat; later ones take Pip round the cat to the cheese from almost anywhere.']; },
   shot: [3, 9],
   tap(x, y, st) { const wx = (x - FX) / FS, wy = 1 - (y - FY) / FS; if (wx < 0 || wx > 1 || wy < 0 || wy > 1 || D.zone(wx, wy)) return; st.tp = [wx, wy]; st.tt = null; },
   say: [[["Your turn. Tap a spot to drop Pip there.", [60, 8, 280, 270]], ["It walks by its biggest Q, no random moves.", [60, 8, 280, 270]]], [["Slide to pick how long DQN trained.", [40, 290, 320, 50]], ["Early networks wander. Later ones round the cat, from almost anywhere.", [60, 8, 280, 270]]]],
   draw(t, st) { const k = this.ctl.values[st.ci ?? 4] - 1, n = tagNet(FULL().snaps[k].net), tp = st.tp || DEF_TP;
     if (st.tt == null || st.changed > st.tt || st.lastTp !== tp) { st.tt = t; st.lastTp = tp; }
     D.meadow({ t, heat: maxQ(n), heatA: .6 }); qArrows(n, .35);
     const w = M(`yt${n.__id}_${tp[0].toFixed(3)}_${tp[1].toFixed(3)}`, () => D.greedyWalk(n, tp[0], tp[1], 9200)), kk = clamp(Math.floor((t - st.tt) * 3), 0, w.P.length);
     D.path(w.P, kk, { c: w.end > 0 ? C.teal : w.end < 0 ? C.rose : C.grey, r: 2.6 }); const p = w.P[Math.max(0, kk - 1)]; D.pipAt(p[0], p[1], 1, t);
     const ev = FULL().snaps[k].ev; txt(`after ${n0((k + 1) * SNAP)} moves: ${ev.cheese} of 30 test walks reach the cheese`, W / 2, 300, { s: 15 });
     txt(st.tp ? (kk >= w.P.length ? (w.end > 0 ? `cheese in ${w.P.length - 1} moves` : w.end < 0 ? 'caught by the cat' : 'no ending in 60 moves') : '') : 'tap anywhere on the meadow', W / 2, 326, { s: 15, c: st.tp ? C.soft : C.goldInk, a: st.tp ? 1 : .6 + .4 * Math.sin(t * 3) }); } },
 { h: 'Atari, from pixels', get p() { return ['DQN made its name on Atari video games. In 2013 Volodymyr Mnih and colleagues at DeepMind trained it on seven games. In 2015, in Nature, one DQN with the **same network and settings** played **49 games**, learning from screen pixels and score alone.', 'Input: the last **four frames**, not four measurements, so motion was visible; outputs: one Q-value per joystick action.']; },
   shot: [3, 10],
   say: [[["DQN made its name on Atari games.", [30, 40, 350, 160]], ["The input was the last four frames of the screen, so motion showed.", [30, 40, 350, 160]], ["The outputs, one Q per joystick action.", [30, 40, 350, 160]]], [["In twenty fifteen, one network with the same settings played forty nine games.", [30, 292, 340, 87]]]],
   draw(t) { [3, 2, 1, 0].forEach(k => alpha(clamp(t - .4 - (3 - k) * .4), () => tv(40 + k * 10, 50 + k * 10, 1.1, t - k * .12, k)));
     alpha(clamp(t - 2.5), () => { txt('last 4 frames', 110, 182, { s: 14, c: C.soft }); arrow(170, 95, 222, 95, 1300, { w: 1.6 });
       rrect(228, 60, 70, 70, 1301, { fill: C.card }); txt('network', 263, 95, { s: 15 }); arrow(304, 95, 330, 95, 1302, { w: 1.6 });
       ['◀', '▶', 'fire', '…'].forEach((s, i) => txt(s, 352, 66 + i * 20, { s: 14, c: C.goldInk })); txt('one Q per action', 300, 150, { s: 13, c: C.soft }); });
     card(['2013: 7 games · 2015 (Nature): 49 games', 'one network, same settings for every game', { s: 'input: pixels + score · output: Q per action', c: C.soft }], t - 4); } },
 { h: 'What it did, and didn’t', get p() { return ['DQN reached at least three quarters of a professional human tester’s score on 29 of the 49 games, and beat earlier learning methods on most.', 'It failed where rewards are rare, like **Montezuma’s Revenge**, where random moves almost never score. It needed tens of millions of frames per game, far more practice than a person. Later reels show replay, target networks and their successors.']; },
   shot: [3, 9],
   say: [[["Forty nine games, one recipe.", [130, 30, 140, 130]]], [["It reached at least three quarters of a human tester's score on twenty nine of them.", [30, 292, 340, 87]], ["It failed where rewards are rare, like Montezuma's Revenge.", [30, 292, 340, 87]]], "And it needed far more practice than a person."],
   draw(t) { tv(150, 40, 1.1, t); txt('49 games, one recipe', W / 2, 140, { s: 17, c: C.ink, a: clamp(t - .5) });
     card([{ s: '≥ 75 % of a human tester: 29 of 49 games', c: C.goldInk }, { s: 'rare rewards (Montezuma’s Revenge): failed', c: C.roseInk }, { s: 'tens of millions of frames per game', c: C.soft }], t - 1.5, { y: 180, gap: 1.5 }); } },
 { h: 'Recap', get p() { return ['**DQN** is Q-learning with a neural network: Pip’s measurements in, one Q-value per move out, ε-greedy to act, and nudges toward reward + 0.9 × the largest Q at the landing spot.', 'Done naively, the targets run away and the lessons come one corner at a time. Replay memory and a target network made it learn, here on the meadow and on 49 Atari games.']; },
   shot: [3, 9],
   say: [[["So, DQN is Q-learning with a neural network.", [30, 292, 340, 110]], ["Measurements in, one Q per move out.", [30, 292, 340, 110]]], ["Done naively, it flounders. Replay memory and a target network make it learn, on the meadow and on Atari."]],
   draw(t) { const n = FULL().net; D.meadow({ t, heat: maxQ(n), heatA: .6 }); qArrows(n, .5); D.pipAt(S0[0], S0[1], 1, t);
     card(['network: measurements in, a Q per move out', 'target: reward + 0.9 × largest Q at landing', 'replay memory: learn from random old moves', { s: 'target network: a frozen copy sets the targets', c: C.soft }], t - .3, { gap: 1 }); } },
];

RLStory.register('dqn', { title: 'Deep Q-Network (DQN)', track: 'Deep RL', onCtx: c => { ctx = c; }, steps: STEPS, sim: { ONLINE, FULL, NET0, Q0 } });
})();
