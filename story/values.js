// Value functions as a story: Pip measures how good each cell is by walking from it many times and averaging.
// All numbers come from the seeded walks below (no learning involved: the policy is fixed, values are measured).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, key, maze, trail, walkPos, pip } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GAM = 0.9, EPS = 0.2, NW = 1000, START = [0, 3], SK = key(0, 3);
// ───── Pip's arrows policy (0 up, 1 right, 2 down, 3 left). Shortest routes; at the start both routes tie, Pip goes up.
const ARR = new Array(20).fill(-1);
[[0, 3, 0], [0, 2, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [2, 1, 0], [1, 2, 3], [1, 3, 1], [2, 3, 1], [3, 3, 1], [4, 3, 0], [4, 2, 0], [4, 1, 0], [3, 2, 1]]
  .forEach(([c, r, m]) => { ARR[key(c, r)] = m; });
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);
// one walk: with probability eps a random direction (any of 4, a bump keeps Pip in place but still counts as a step), else the arrow
function walk(c, r, eps, R, first, keepTr = true) {
  const tr = []; let n = 0;
  for (let k = 0; k < 5000; k++) {
    let m, rnd = false;
    if (k === 0 && first != null) m = first; else if (R() < eps) { rnd = true; m = Math.floor(R() * 4); } else m = ARR[key(c, r)];
    const [nc, nr] = G.step(c, r, m); if (keepTr) tr.push({ c, r, nc, nr, m, rnd }); n++; c = nc; r = nr;
    const rw = G.term(c, r); if (rw) return { tr, n, rew: rw, ret: rw * GAM ** (n - 1) };
  }
  return { tr, n, rew: 0, ret: 0 };
}
const memo = {}, M = (k, f) => (k in memo ? memo[k] : (memo[k] = f()));
const BASE = { 0: 11000, 0.2: 87000, 0.5: 5000, 1: 23000 };
const walksFrom = (c, r, eps) => M(`w${eps}_${c}_${r}`, () => { const R = rng(BASE[eps] + key(c, r)), out = []; for (let i = 0; i < NW; i++) out.push(walk(c, r, eps, R, null, i < 6)); return out; });
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
const notebook = eps => M('nb' + eps, () => { const V = new Array(20).fill(0); CELLS.forEach(([c, r]) => { V[key(c, r)] = mean(walksFrom(c, r, eps).map(w => w.ret)); }); return V; });
const stepsUsed = eps => M('st' + eps, () => CELLS.reduce((s, [c, r]) => s + walksFrom(c, r, eps).reduce((a, w) => a + w.n, 0), 0));
const qWalks = (c, r, m) => M(`q${c}_${r}_${m}`, () => { const R = rng(31000 + key(c, r) * 4 + m), out = []; for (let i = 0; i < NW; i++) out.push(walk(c, r, EPS, R, m, i < 3)); return out; });
const Qmc = (c, r, m) => mean(qWalks(c, r, m).map(w => w.ret));
// exact values (for the notes / checks only): repeated sweeps of the averaging rule
function exactV(eps) { const V = new Array(20).fill(0); for (let it = 0; it < 3000; it++) CELLS.forEach(([c, r]) => { const k = key(c, r); let v = 0; for (let m = 0; m < 4; m++) { const p = eps / 4 + (m === ARR[k] ? 1 - eps : 0), [nc, nr] = G.step(c, r, m), rw = G.term(nc, nr); v += p * (rw + (rw ? 0 : GAM * V[key(nc, nr)])); } V[k] = v; }); return V; }

// ───── the numbers the story quotes
const SW = walksFrom(0, 3, EPS), RET = SW.map(w => w.ret), AVG = []; { let s = 0; RET.forEach((x, i) => { s += x; AVG.push(s / (i + 1)); }); }
const f2 = v => { const s = (Math.abs(v) < .005 ? 0 : v).toFixed(2); return s.replace('-', '−'); };
const NBA = notebook(EPS), NBW = notebook(1);
const WW = walksFrom(0, 3, 1)[0];                         // the Wanderer's first walk from the start
const Q30 = [0, 1, 2, 3].map(m => Qmc(3, 0, m));          // up, right, down, left beside the cheese

// ───── drawing helpers
const ZERO = () => new Array(20).fill(0);
function nums(V, a = 1, Gx = G, s = 14, only) {
  for (const [c, r] of CELLS) { const k = key(c, r); if (only && !only(c, r)) continue; const aa = typeof a === 'function' ? a(c, r) : a; if (aa <= 0) continue; const v = V[k];
    txt(f2(v), Gx.ox + c * Gx.cs + Gx.cs - 5, Gx.oy + r * Gx.cs + (s >= 14 ? 12 : 10), { s, al: 'right', a: aa, c: v < -.004 ? C.roseInk : v > .004 ? C.goldInk : C.soft }); }
}
function arrows(a = 1, t0 = 0, t = 99, o = {}) { CELLS.forEach(([c, r], i) => { const u = typeof a === 'function' ? a(c, r) : a * clamp((t - t0 - i * (o.gap ?? 0)) * 3); if (u <= 0) return; alpha(u, () => G.arrowIn(c, r, ARR[key(c, r)], 400 + key(c, r), { c: o.c || C.goldInk, w: o.w || 1.5 })); }); }
function ghost(c, r, s, a = 1, o = {}) { if (o.bg) box(cx(c) - 29, cy(r) + (o.dy ?? 20) - 9, 58, 18, C.card, .9 * a); txt(s, cx(c), cy(r) + (o.dy ?? 20), { s: o.s || 14, c: o.c || '#9a8f84', a }); }
function dim(a = .62) { box(0, 0, W, 285, C.bg, a); }
// arrow from near a cell's centre to its edge, in direction m
function edgeArrow(c, r, m, seed, col, r0 = 12) { const [dx, dy] = G.MOVES[m]; arrow(cx(c) + dx * r0, cy(r) + dy * r0, cx(c) + dx * 29, cy(r) + dy * 29, seed, { c: col, w: 1.6, head: 6 }); }
// animate one recorded walk; '?' marks a random (distracted) move
function animWalk(w, t, dur, o = {}) {
  const tr = w.tr, m = walkPos(tr, 0, tr.length, Math.max(0, t), dur); trail(tr.slice(0, Math.min(tr.length, m.k + 1)).map(s => [s.c, s.r]), o.ta ?? .55);
  if (!o.noPip) pip(m.x, m.y, m.face, t);
  if (!m.done && m.s && m.s.rnd && o.q !== false) txt('?', m.x + 14, m.y - 24, { s: 20, c: C.roseInk });
  return m;
}
// chart of returns: across = walk number (1..xmax), up = return (−1..+1); dots gold = cheese, rose = cat; dark line = running average
const CH = { x0: 66, x1: 366, yt: 300, yb: 388 }, CY = v => (CH.yt + CH.yb) / 2 - v * (CH.yb - CH.yt) / 2;
function chart(n, xmax, o = {}) {
  const X = i => CH.x0 + (i + .5) / xmax * (CH.x1 - CH.x0), ax = o.ax ?? 1;
  alpha(ax, () => { rline(CH.x0, CY(0), CH.x1, CY(0), 960, { c: C.line, w: 1 }); rline(CH.x0, CH.yt, CH.x0, CH.yb, 961, { c: C.line, w: 1 });
    rline(CH.x0, CY(1), CH.x1, CY(1), 962, { c: '#ece2d4', w: 1, once: true }); rline(CH.x0, CY(-1), CH.x1, CY(-1), 963, { c: '#ece2d4', w: 1, once: true });
    txt('+1', CH.x0 - 14, CY(1), { s: 12, c: C.soft }); txt('0', CH.x0 - 14, CY(0), { s: 12, c: C.soft }); txt('−1', CH.x0 - 14, CY(-1), { s: 12, c: C.soft });
    txt('return', CH.x0 - 30, CY(.5), { s: 12, c: C.soft }); txt(`walk number (1 to ${xmax})`, CH.x1, CH.yb + 12, { s: 12, c: C.soft, al: 'right' }); });
  const big = xmax <= 10, r = big ? 4.5 : xmax <= 100 ? 2.6 : 1.7;
  for (let i = 0; i < n; i++) dot(X(i), CY(RET[i]), r, SW[i].rew > 0 ? C.gold : C.rose, big ? 1 : .7);
  if (big && o.vals) for (let i = 0; i < n; i++) txt(f2(RET[i]), X(i), CY(RET[i]) + (RET[i] > 0 ? -12 : 12), { s: 12, c: SW[i].rew > 0 ? C.goldInk : C.roseInk });
  if (o.avg && n > 0) { ctx.save(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.6; ctx.beginPath(); const step = Math.max(1, Math.floor(n / 300)); for (let i = 0; i < n; i += step) { i ? ctx.lineTo(X(i), CY(AVG[i])) : ctx.moveTo(X(i), CY(AVG[i])); } ctx.lineTo(X(n - 1), CY(AVG[n - 1])); ctx.stroke(); ctx.restore();
    if (big) txt(`average ${f2(AVG[n - 1])}`, X(n - 1) + 14, CY(AVG[n - 1]) + 4, { s: 14, al: 'left', c: C.ink }); else txt(`average ${f2(AVG[n - 1])}`, CH.x1, CY(AVG[n - 1]) - 14, { s: 14, al: 'right', c: C.ink }); }
}
function counted(V, cells, t, t0, per) { // cells' numbers count up as their walks are averaged
  const out = ZERO(), a = []; cells.forEach(([c, r], i) => { const u = clamp((t - t0 - i * per) / (per * 1.4)); if (u <= 0) return; const ws = V(c, r), n = Math.max(1, Math.ceil(ease(u) * NW)); out[key(c, r)] = mean(ws.slice(0, n).map(w => w.ret)); a.push([c, r, u]); }); return { V: out, a };
}

// the Wanderer's notebook and the arrows notebook side by side
const GA = L.gridWorld({ cs: 38, ox: 8, oy: 44 }), GB = L.gridWorld({ cs: 38, ox: 202, oy: 44 });
function goBoard(x, y, s) { for (let i = 0; i < 7; i++) { rline(x, y + i * s, x + 6 * s, y + i * s, 1200 + i, { c: C.line, w: 1 }); rline(x + i * s, y, x + i * s, y + 6 * s, 1210 + i, { c: C.line, w: 1 }); }
  [[1, 1, 0], [2, 1, 1], [2, 2, 0], [3, 2, 1], [4, 3, 0], [3, 4, 1], [2, 4, 0], [4, 4, 1], [1, 3, 1], [5, 2, 0]].forEach(([a, b, wht], i) => L.rcircle(x + a * s, y + b * s, s * .42, 1230 + i, { fill: wht ? '#fffaf1' : '#4a4540', j: .4 })); }

const SI = { w2: SW[1], w3: SW[2], w4: SW[3] };
const PCAT = Math.round(100 * walksFrom(0, 3, 1).filter(w => w.rew < 0).length / NW);
const R2 = SW[1].tr.filter(s => s.rnd).length;
const RSE = { 10: AVG[9], 100: AVG[99], 1000: AVG[999] };
const QFMT = Q30.map(f2), MIX = .85 * Q30[1] + .05 * (Q30[0] + Q30[2] + Q30[3]);
const SLV = [0, .2, .5, 1], SLN = ['never', '1 in 5', '1 in 2', 'always'];
function tapCell(x, y, st, t) {
  const c = Math.floor((x - OX) / CS), r = Math.floor((y - OY) / CS);
  if (c < 0 || c >= 5 || r < 0 || r >= 4 || G.isRock(c, r) || G.term(c, r)) return;
  st.taps = (st.taps || 0) + 1; const eps = SLV[st.ci ?? 1]; st.w = walk(c, r, eps, rng(7000 + st.taps * 31 + key(c, r))); st.wt = t; st.wc = [c, r]; st.weps = eps;
}

const STEPS = [
 { h: 'Back in the maze', p: ['Stepping **onto** the cheese pays +1, onto the cat −1, and either ends the walk; other steps pay 0. The **return** adds the rewards, each step later multiplied by one more **γ (gamma)**, here 0.9.', 'The floor is dry again: every move goes where Pip intends. All randomness now is Pip’s own habit, so numbers differ from the policy reel’s on purpose.'],
   shot: [5, 10],
   draw(t) { maze({ draw: t / 1.6, pop: { cheese: ease((t - 1.8) * 3), cat: ease((t - 2.3) * 3) }, labels: clamp(t - 3), t }); if (t > 2.8) pip(cx(0), cy(3), 1, t);
     if (t > 5) card(['cheese: +1 on the step onto it, walk ends', 'cat: −1 on the step onto it, walk ends', { s: 'any other step: 0  ·  γ = 0.9  ·  dry floor', c: C.soft }], t - 5, { gap: .9 }); } },
 { h: 'Pip’s policy', p: ['Pip uses **policy A** from the policy reel, the rule card with an arrow in every cell, with the ε (epsilon) dial at 0.2: one move in five is a random direction. So the arrow wins 85 times in 100, each other move 5 times.', 'Bumping into a wall or rock leaves Pip in place, and still counts as a step.'],
   shot: [4, 11],
   draw(t) { maze({ t }); arrows(1, .5, t, { gap: .18 }); pip(cx(0), cy(3), 1, t);
     if (t > 4) card(['policy A arrows, ε = 0.2: arrow 85 in 100', '1 move in 5: a random direction  (shown as ?)', { s: 'bump into a wall or rock: stay put, still a step', c: C.soft }], t - 4, { gap: 1.1 }); } },
 { h: 'One walk', p: [`How good is it to start in the bottom-left corner with this policy? Let’s try. Walk 1: Pip follows the arrows up and along the top and reaches the cheese in ${SW[0].n} steps.`, `The first step’s reward counts fully, and each later one is multiplied by one more 0.9. The +1 came on step ${SW[0].n}, so the return is 0.9 multiplied ${SW[0].n - 1} times: ${f2(RET[0])}.`],
   shot: [3, 11],
   draw(t) { maze({ t }); arrows(.35); const w = SW[0], m = animWalk(w, t - .6, .6);
     if (m.done) bubble('+1', cx(4), cy(0) - 18, t - .6 - w.n * .6);
     if (!m.done) caption(`step ${Math.min(w.n, m.k + 1)}`);
     if (t > 5.4) card([`walk 1: cheese on step ${w.n}`, `return = 0.9 × 0.9 × 0.9 × 0.9 × 0.9 × 0.9 × 1 = ${f2(RET[0])}`, { s: 'reward on step 7 is discounted 6 times', c: C.soft }], t - 5.4, { gap: 1.1 }); } },
 { h: 'Same start, different walks', p: [`Walk 2: Pip gets distracted twice, once stepping back down, once bumping the top wall. Cheese on step ${SI.w2.n}, return ${f2(RET[1])}.`, `Walk 3: beside the cheese, a random move points down, onto the cat. Return ${f2(RET[2])} after ${SI.w3.n} steps. Walk 4’s only random move, at the start, picks up, the arrow’s own direction, so nothing changes: ${f2(RET[3])}. Same cell, same policy, different returns.`],
   shot: [4, 13],
   draw(t) { maze({ t }); arrows(.3); const D = .32, ts = [0, SI.w2.n * D + 1.2]; ts.push(ts[1] + SI.w3.n * D + 1.2); const ends = ts.map((s, i) => s + [SI.w2, SI.w3, SI.w4][i].n * D);
     const i = t < ts[1] ? 0 : t < ts[2] ? 1 : 2, w = [SI.w2, SI.w3, SI.w4][i], m = animWalk(w, t - ts[i], D);
     if (m.done) bubble(w.rew > 0 ? '+1' : '−1', cx(w.rew > 0 ? 4 : 3), cy(w.rew > 0 ? 0 : 1) - 18, t - ends[i], w.rew > 0 ? null : C.roseInk);
     const lines = [{ s: `walk 1:  cheese, ${SW[0].n} steps  →  ${f2(RET[0])}`, c: C.goldInk }];
     [SI.w2, SI.w3, SI.w4].forEach((q, j) => { if (t > ends[j] + .3) lines.push({ s: `walk ${j + 2}:  ${q.rew > 0 ? 'cheese' : 'cat'}, ${q.n} steps  →  ${f2(q.ret)}`, c: q.rew > 0 ? C.goldInk : C.roseInk }); });
     card(lines, 10, { lh: 22, fs: 16 }); } },
 { h: 'Keeping score', p: ['Let’s put the returns on a chart. Across is the walk number, up is the return, from −1 to +1. Each dot is one walk: gold means the walk reached the cheese, rose means the cat.', `The dark line is the **average** of all returns so far. After four walks: (0.53 + 0.39 − 0.48 + 0.53) ÷ 4 = ${f2(AVG[3])}.`],
   shot: [3, 10],
   draw(t) { maze({ t }); arrows(.3); pip(cx(0), cy(3), 1, t); const ax = clamp(t - .3), n = clamp(Math.floor((t - 2.5) / .9) + 1, 0, 4);
     chart(n, 10, { ax, vals: 1, avg: t > 7 });
     if (t < 2.5) txt('across: walk number   ·   up: its return', W / 2, 292, { s: 14, c: C.soft, a: clamp(t - .6) }); } },
 { h: 'Many walks', p: [`Pip keeps walking. Early on the average jumps around: after 10 walks it is ${f2(RSE[10])}, after 100 walks ${f2(RSE[100])}.`, `Each new walk moves the average less and less. After ${NW} walks it has settled at ${f2(RSE[1000])}, give or take about 0.015 of leftover wobble. That settled number is what we were after.`],
   shot: [3, 7, 13],
   draw(t) { let n, xm; if (t < 3) { xm = 10; n = 4 + Math.floor(clamp(t / 2.5) * 6); } else if (t < 7) { xm = 100; n = 10 + Math.floor(clamp((t - 3) / 3) * 90); } else { xm = 1000; n = 100 + Math.floor(clamp((t - 7) / 4) * 900); }
     maze({ t }); arrows(.3); const w = SW[n - 1]; if (w.tr.length) trail(w.tr.map(s => [s.c, s.r]), .35); pip(cx(w.rew > 0 ? 4 : 3), cy(w.rew > 0 ? 0 : 1), 1, t);
     chart(n, xm, { avg: 1, vals: n <= 10 }); txt(`walk ${n} of ${NW}`, CH.x1, 291, { s: 14, al: 'right', c: C.soft }); } },
 { h: 'The value of a cell', p: ['The average return of many walks that start in a cell and follow the policy is that cell’s **value**. In probability terms it is the **expected** discounted return: the number the average closes in on as walks pile up.', `We write it **V(s)**, said “V of s”, where s is the state, here a cell. So V(start) is about ${f2(RSE[1000])}.`],
   shot: [3, 9],
   draw(t) { const V = ZERO(); V[SK] = RSE[1000]; maze({ V, t }); nums(V, clamp(t - .6), G, 14, (c, r) => c === 0 && r === 3); arrows(.25); pip(cx(0), cy(3), 1, t);
     alpha(clamp(t - 1.2), () => { box(cx(1) + 4, cy(3) - 13, 128, 26, C.card, .95); txt(`V(start) ≈ ${f2(RSE[1000])}`, cx(1) + 68, cy(3), { s: 18, c: C.goldInk }); arrow(cx(1) + 2, cy(3), cx(0) + 28, cy(3), 1320, { c: C.goldInk, w: 1.4 }); });
     card(['V(s) = average return of walks that', '          start in s and follow the policy', { s: `V(start) ≈ ${f2(RSE[1000])}   (from ${NW} walks)`, c: C.goldInk }], t - 2, { gap: 1.1 }); } },
 { h: 'A whole notebook', p: [`Now do the same from every cell: ${NW} walks each, and write the average in the cell’s corner. That is Pip’s **notebook** of values for this policy. Stronger gold shading means a higher value.`, 'The cheese and cat cells get no walks. Standing on them, the walk is already over, so their value is 0 and they stay blank.'],
   shot: [5, 13],
   draw(t) { const { V, a } = counted((c, r) => walksFrom(c, r, EPS), CELLS, t, .5, .55); maze({ V, t }); arrows(.25); nums(V, (c, r) => a.some(q => q[0] === c && q[1] === r) ? 1 : 0);
     const cur = a.filter(q => q[2] < 1).pop(); if (cur) box(OX + cur[0] * CS + 3, OY + cur[1] * CS + 3, CS - 6, CS - 6, C.teal, .18);
     if (t > 10) card(['notebook: one value per cell, for this policy', { s: 'cheese and cat: walk already over, value 0', c: C.soft }], t - 10, { gap: .9 }); } },
 { h: 'Reading the notebook', p: [`Beside the cheese the value is ${f2(NBA[key(3, 0)])}, not 1. The arrow points at the cheese, but one move in five is random, and one in twenty goes down, onto the cat.`, `Two cells three steps from the cheese: on the top row ${f2(NBA[key(1, 0)])}, beside the cat only ${f2(NBA[key(2, 1)])}. From there, a random move to the right lands on the cat.`],
   shot: [3, 10],
   draw(t) { const a1 = clamp(t - .5), a2 = clamp(t - 5); maze({ V: NBA, t, hi: [[3, 0, C.gold, .3 * a1 * (1 - a2)], [1, 0, C.teal, .3 * a2], [2, 1, C.rose, .3 * a2]] }); arrows(.25); nums(NBA);
     if (t < 5.2) { ghost(3, 0, '1 in 20 ↓', a1, { dy: 22, s: 14, c: C.roseInk, bg: 1 }); }
     else { ghost(1, 0, '3 steps', a2, { dy: 22, s: 14, c: '#3f7f7c', bg: 1 }); ghost(2, 1, '3 steps', a2, { dy: 22, s: 14, c: C.roseInk, bg: 1 }); if (t > 6.5) alpha(clamp(t - 6.5), () => arrow(cx(2) + 10, cy(1) - 6, cx(3) - 16, cy(1) - 6, 1300, { c: C.roseInk, w: 1.5 })); }
     card([`beside the cheese: ${f2(NBA[key(3, 0)])}, not 1`, { s: `3 steps away: top row ${f2(NBA[key(1, 0)])},  beside the cat ${f2(NBA[key(2, 1)])}`, c: C.soft }], t - 1.2, { gap: 3.8 }); } },
 { h: 'A different policy', p: ['Meet the **Wanderer**: a policy with no arrows at all. Every single move is random.', `Same maze, same rewards, same way of computing returns. The Wanderer’s first walk from the start wanders back and forth and does find the cheese, but only on step ${WW.n}. By then the +1 has shrunk to a return of ${f2(WW.ret)}.`],
   shot: [4, 12],
   draw(t) { maze({ t }); const D = Math.min(.25, 9 / WW.n), m = animWalk(WW, t - .8, D, { q: false, ta: .4 }); if (!m.done) caption(`Wanderer, step ${Math.min(WW.n, m.k + 1)}`);
     else { bubble(WW.rew > 0 ? '+1' : '−1', cx(WW.rew > 0 ? 4 : 3), cy(WW.rew > 0 ? 0 : 1) - 18, t - .8 - WW.n * D, WW.rew > 0 ? null : C.roseInk); caption(`${WW.n} steps, return ${f2(WW.ret)}`); }
     if (t > 1) card(['Wanderer: up, right, down or left,', '                 1 in 4 each, every move', { s: m.done ? `this walk: 0.9 multiplied ${WW.n - 1} times = ${f2(WW.ret)}` : ' ', c: C.goldInk }], t - 1, { gap: .9 }); } },
 { h: 'The Wanderer’s notebook', p: [`${NW} Wanderer walks from every cell give a new notebook. Every value is **negative**: the start is worth ${f2(NBW[SK])}, the cell left of the cat ${f2(NBW[key(2, 1)])}.`, `Why? A random walker usually meets the cat first: from the start, ${PCAT} walks in 100 end there, about three in four. The cat has four entry cells, the cheese two. Long walks shrink any reward to almost nothing.`],
   shot: [6, 13],
   draw(t) { const { V, a } = counted((c, r) => walksFrom(c, r, 1), CELLS, t, .4, .35); maze({ V, t }); nums(V, (c, r) => a.some(q => q[0] === c && q[1] === r) ? 1 : 0);
     if (t > 7) { const u = clamp(t - 7); [[3, 0], [2, 1], [4, 1], [3, 2]].forEach(([c, r], i) => alpha(u, () => edgeArrow(c, r, [2, 1, 3, 0][i], 1400 + i, C.roseInk))); [[3, 0], [4, 1]].forEach(([c, r], i) => alpha(clamp(t - 8.5), () => edgeArrow(c, r, [1, 0][i], 1410 + i, C.goldInk)));
       card([{ s: 'ways onto the cat: 4 cells', c: C.roseInk }, { s: 'ways onto the cheese: 2 cells', c: C.goldInk }], t - 7, { gap: 1.5 }); } } },
 { h: 'Values belong to a policy', p: [`Same cell, two numbers: the start is worth ${f2(NBA[SK])} under the arrows, ${f2(NBW[SK])} under the Wanderer. A value means something only **for a given policy**, so we may write V with a small π (pi).`, 'The arrows win in every cell, so they are the better policy wherever Pip starts. If each won somewhere, neither would be better everywhere; it would depend on the start.'],
   shot: [4, 11],
   draw(t) { const a = clamp(t - .3), b = clamp(t - 1.8);
     alpha(a, () => { GA.maze({ V: NBA, t }); nums(NBA, 1, GA, 12); txt('arrows (1 in 5 random)', GA.ox + 95, 28, { s: 15, c: C.goldInk }); });
     alpha(b, () => { GB.maze({ V: NBW, t }); nums(NBW, 1, GB, 12); txt('Wanderer (all random)', GB.ox + 95, 28, { s: 15, c: C.roseInk }); });
     if (t > 3.5) { const u = clamp(t - 3.5); [GA, GB].forEach(g => box(g.ox + 3, g.oy + 3 * g.cs + 3, g.cs - 6, g.cs - 6, C.teal, .3 * u)); }
     if (t > 4.5) card([`start:  ${f2(NBA[SK])} with arrows,  ${f2(NBW[SK])} with the Wanderer`, 'V depends on the policy:  Vπ(s)', { s: `arrows ≥ Wanderer in all ${CELLS.length} cells`, c: C.soft }], t - 4.5, { y: 226, gap: 1.2 }); } },
 { h: 'How good is a move?', p: ['The notebook says how good a cell is. To choose a move, Pip wants to know how good each **move** is.', 'The **action value**, written **Q(s, a)**, said “Q of s and a”, is the average return when Pip takes move a first, then follows the policy as usual. It is measured the same way: many walks, each forced to start with that move.'],
   shot: [3, 10],
   draw(t) { maze({ t, hi: [[3, 0, C.teal, .2]] }); arrows(.2); const qa = clamp(t - 1); [0, 1, 2, 3].forEach(m => alpha(qa, () => edgeArrow(3, 0, m, 1500 + m, C.ink, 17)));
     if (t > 4) { const qs = qWalks(3, 0, 3), w = qs.find(q => q.tr.length && q.n === 3) || qs[0]; const m = animWalk(w, t - 4, .5); if (m.done) caption(`forced left, then the arrows: ${w.n} steps, return ${f2(w.ret)}`); else caption('first move forced: left, then the policy'); } else pip(cx(3), cy(0), 1, t);
     card(['Q(s, a) = average return when Pip', '               takes move a first, then follows the policy', { s: 'measured with walks forced to start with a', c: C.soft }], t - 1.5, { gap: 1.1, fs: 16 }); } },
 { h: 'The four moves beside the cheese', p: [`Right pays +1 at once: Q = 1. Down is the cat: −1. Left gives ${QFMT[3]}; up bumps the wall, wasting a step: ${QFMT[0]}. The best move has the biggest Q.`, `Weighted by the policy’s chances, π(a | s), 0.85 for the arrow, 0.05 per other move: 0.85 × 1 + 0.05 × (${QFMT[0]} − 1 + ${QFMT[3]}) = ${f2(MIX)}, the notebook value.`],
   shot: [4, 12],
   draw(t) { maze({ t, hi: [[3, 0, C.teal, .2]] }); arrows(.2); pip(cx(3), cy(0), 1, t);
     const pos = [[cx(3), 7], [cx(4), cy(0) + 22], [cx(3), cy(1) + 25], [cx(2), cy(0) + 20]], order = [1, 2, 3, 0], lab = ['Q = ' + QFMT[0] + '  (bump)', 'Q = 1', 'Q = −1', 'Q = ' + QFMT[3]];
     order.forEach((m, i) => { const u = clamp(t - .5 - i * .9); if (u <= 0) return; alpha(u, () => edgeArrow(3, 0, m, 1500 + m, C.ink, 17)); const [x, y] = pos[m], v = Q30[m], wd = lab[m].length * 6.4 + 10;
       const best = m === 1 && t > 4.5; box(x - wd / 2, y - 8, wd, 17, best ? C.yellow : C.card, .92 * u); txt(lab[m], x, y, { s: 14, c: v < 0 ? C.roseInk : C.goldInk, a: u }); });
     if (t > 5) card(['V = 0.85 × Q(right) + 0.05 × (Q(up) + Q(down) + Q(left))', `   = 0.85 × 1 + 0.05 × (${QFMT[0]} − 1 + ${QFMT[3]}) = ${f2(MIX)}`, { s: `notebook value of this cell: ${f2(NBA[key(3, 0)])}`, c: C.goldInk }], t - 5, { gap: 1.3, fs: 15 }); } },
 { h: 'Why values are worth having', p: ['V answers “how good is it here?”: compare cells, and policies cell by cell.', 'Q answers “how good is this move?”: the biggest Q is the best next move if Pip follows the policy afterwards. If a move’s Q beats the habit’s value, switching to it only helps: every visit to that cell gains, and gains add up. Much of RL builds on this.'],
   shot: [3, 12],
   draw(t) { const a1 = clamp(t - .5), a2 = clamp(t - 4.5); maze({ V: NBA, t, hi: [[0, 3, C.teal, .25 * a1 * (1 - a2)], [3, 0, C.gold, .3 * a2]] }); nums(NBA, .9); arrows(.3);
     card([{ s: 'V(s): how good is this cell?  compare cells, policies', c: C.goldInk }, { s: 'Q(s, a): how good is this move?  pick the biggest', c: C.ink }, { s: 'a move with bigger Q than the policy’s → better policy', c: C.soft }], t - .5, { gap: 3.8, fs: 16 }); } },
 { h: 'The catch', p: [`Measuring by walking is slow. This notebook took ${(CELLS.length * NW).toLocaleString('en-US')} walks and ${stepsUsed(EPS).toLocaleString('en-US')} steps. The Wanderer’s took ${stepsUsed(1).toLocaleString('en-US')} steps. And every number is still a little noisy.`, `Yet neighbouring numbers are clearly linked: ${f2(NBA[key(3, 0)])} beside the cheese, ${f2(NBA[key(2, 0)])} one step back, a little less than 0.9 times as much. The next reel turns that link into a shortcut, the **Bellman equation**.`],
   shot: [4, 11],
   draw(t) { maze({ V: NBA, t, hi: t > 5 ? [[3, 0, C.gold, .3], [2, 0, C.gold, .2]] : null }); nums(NBA); arrows(.25); const u = clamp((t - .5) / 3.5);
     card([`arrows notebook: ${Math.round(ease(u) * stepsUsed(EPS)).toLocaleString('en-US')} steps walked`, `Wanderer notebook: ${Math.round(ease(u) * stepsUsed(1)).toLocaleString('en-US')} steps walked`, { s: 'neighbours are linked: is there a shortcut?', c: C.soft }], t - .3, { gap: 2.2 }); } },
 { h: 'Your turn', ctl: { label: 'random moves', values: SLV, def: 1, fmt: v => SLN[SLV.indexOf(v)], onChange: st => { st.w = null; } }, tap: tapCell,
   p: ['Drag the slider to change how often Pip moves at random; the notebook is remeasured with 1000 walks per cell. With no randomness every walk from a cell is identical, so one walk gives the value. At “always”, Pip is the Wanderer.', 'Tap a cell to send Pip on one walk from there and compare its return with the notebook.'],
   shot: [3],
   draw(t, st) { const eps = SLV[st.ci ?? 1], V1 = notebook(eps), prev = st.prevCi != null ? notebook(SLV[st.prevCi]) : V1, u = ease((t - (st.changed ?? -9)) / .9), V = V1.map((v, i) => lerp(prev[i], v, u));
     maze({ V, t }); nums(V); if (eps < 1) arrows(lerp(.45, .15, eps));
     if (st.w && st.weps === eps) { const w = st.w, D = Math.min(.3, 8 / w.n), m = animWalk(w, t - st.wt, D, { q: eps > 0 && eps < 1 });
       caption(m.done ? `this walk: ${w.n} steps, return ${f2(w.ret)}  ·  notebook: ${f2(V1[key(...st.wc)])}` : `step ${m.k + 1}`); }
     else { pip(cx(0), cy(3), 1, t); caption('tap a cell to send Pip on a walk'); }
     txt(eps === 0 ? 'no randomness: every walk is the shortest route' : eps === 1 ? 'the Wanderer: every value negative' : `1 move in ${Math.round(1 / eps)} random`, W / 2, 300, { s: 17, c: C.soft }); } },
 { h: 'Where values show up', p: ['AlphaGo, in 2016, had a **value network**: shown a Go position, it estimated the game’s expected result for the player to move, +1 for a win, −1 for a loss.', 'Chess programs score positions in a similar spirit. Many modern learners, including those used to fine-tune chatbots, train a value estimate called a **critic** to judge how well things are going.'],
   shot: [4, 10],
   draw(t) { goBoard(70, 40, 30); const a = clamp(t - 1.5); alpha(a, () => { rrect(270, 70, 110, 70, 1250, { fill: C.card, c: C.line }); txt('value network', 325, 90, { s: 15 }); txt('expected result', 325, 112, { s: 14, c: C.soft }); txt('≈ +0.3', 325, 130, { s: 15, c: C.goldInk }); arrow(256, 130, 270, 110, 1260, { c: C.soft }); });
     txt('a position  →  a number', 160, 248, { s: 16, c: C.soft, a: clamp(t - 3) });
     txt('(illustration, not a real AlphaGo output)', 200, 272, { s: 12, c: C.soft, a: clamp(t - 3) });
     if (t > 4) card(['Go, chess: how good is this position?', 'modern learners: a critic = a value estimate', { s: 'same idea as Pip’s notebook, learned at scale', c: C.soft }], t - 4, { gap: 1.1 }); } },
 { h: 'Recap', p: ['A cell’s **value** is the expected discounted return when Pip starts there and follows its policy: the average over many walks.', 'Change the policy and the values change. **Q** scores a move: take it first, then follow the policy. Pick the move with the biggest Q.'],
   shot: [3, 9],
   draw(t) { maze({ V: NBA, t }); nums(NBA); arrows(.3); pip(cx(0), cy(3), 1, t);
     card(['V(s) = expected discounted return from s', '         = average return of many walks', 'new policy → new notebook:  Vπ', 'Q(s, a): move a first, then the policy'], t - .3, { gap: 1 }); } },
];

RLStory.register('values', { title: 'Value Functions', track: 'Foundations', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { G, ARR, CELLS, walk, walksFrom, notebook, exactV, qWalks, Qmc, SW, RET, AVG, NBA, NBW, WW, Q30, MIX, stepsUsed, R2, PCAT, f2 } });
})();
