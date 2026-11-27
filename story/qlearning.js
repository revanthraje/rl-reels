// Q-learning as a story: Pip loses the map, keeps four numbers per cell (one per move) and learns them from experience.
// All numbers come from the seeded runs below (Q-learning seed 54; the TD comparison reuses td.js's seed 35).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, txt, arrow, card, bubble, caption, dot, box, num } = L;
let ctx;
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, key, maze, trail, pip } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GAM = 0.9, ALPHA = 0.2, SEED = 54, NEP = 300, N = ['up', 'right', 'down', 'left'];
const qk = (c, r, m) => key(c, r) * 4 + m, DEMO = [{ c: 2, r: 0, nc: 3, nr: 0 }];
// ───── Q-learning: eps-greedy over all 4 moves (bumps are moves too), ties broken at random; update on landing, then pick the next move
function runQ(eps, seed = SEED, episodes = NEP) {
  const R = rng(seed), Q = new Float64Array(80), E = [];
  const best = (c, r) => G.term(c, r) ? 0 : Math.max(Q[qk(c, r, 0)], Q[qk(c, r, 1)], Q[qk(c, r, 2)], Q[qk(c, r, 3)]);
  const act = (c, r) => { if (R() < eps) return [Math.floor(R() * 4), true]; let b = -1e9, ms = []; for (let m = 0; m < 4; m++) { const q = Q[qk(c, r, m)]; if (q > b + 1e-9) { b = q; ms = [m]; } else if (q > b - 1e-9) ms.push(m); } return [ms[Math.floor(R() * ms.length)], false]; };
  for (let e = 0; e < episodes; e++) {
    let [c, r] = G.start; const tr = [];
    for (let k = 0; k < 2000; k++) {
      const [m, rnd] = act(c, r), [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr), done = rew !== 0;
      const old = Q[qk(c, r, m)], next = done ? 0 : best(nc, nr), target = rew + GAM * next, nw = old + ALPHA * (target - old);
      Q[qk(c, r, m)] = nw; tr.push({ c, r, m, rnd, nc, nr, rew, old, next, target, nw }); c = nc; r = nr; if (done) break;
    }
    const l = tr[tr.length - 1]; E.push({ tr, end: l.rew, ret: l.rew * GAM ** (tr.length - 1), Q: Array.from(Q) });
  }
  return E;
}
const runs = {}, RUN = e => runs[e] ||= runQ(e);
const E = RUN(0.2), ZQ = new Array(80).fill(0);
// Q after walk e's first i transitions
function Qat(e, i) { const Q = e > 0 ? E[e - 1].Q.slice() : ZQ.slice(); for (let k = 0; k < i; k++) { const t = E[e].tr[k]; Q[qk(t.c, t.r, t.m)] = t.nw; } return Q; }
const maxQ = (Q, c, r) => Math.max(Q[qk(c, r, 0)], Q[qk(c, r, 1)], Q[qk(c, r, 2)], Q[qk(c, r, 3)]);
const argQ = (Q, c, r) => { let b = -1e9, bm = 0; for (let m = 0; m < 4; m++) if (Q[qk(c, r, m)] > b + 1e-12) { b = Q[qk(c, r, m)]; bm = m; } return bm; };
const flat = (Q, c, r) => { const v = [0, 1, 2, 3].map(m => Q[qk(c, r, m)]); return Math.max(...v) - Math.min(...v) < 1e-9; };
// ───── story moments, found in the run (not hard-coded)
const e0 = E[0].tr, L0 = e0[e0.length - 1];                                  // walk 1 ends with a random 'up' from below the cheese
const W2 = E[1].tr, L2 = W2[W2.length - 1];                                  // walk 2 ends with 'right' from beside the cheese
const W3I = 2, W3K = E[W3I].tr.findIndex(t => !t.rew && t.next > 0), S3 = E[W3I].tr[W3K]; // first time a guess passes one cell back
const CATW = E.findIndex((e, i) => i > 2 && e.end < 0 && e.tr[e.tr.length - 1].c === 3 && e.tr[e.tr.length - 1].r === 0);
const CT = E[CATW].tr, CL = CT[CT.length - 1], CP = CT[CT.length - 2];       // CP: right into (3,0) using the max; CL: random down onto the cat
const Q30 = E.map(e => [0, 1, 2, 3].map(m => e.Q[qk(3, 0, m)]));
const FIN = E[NEP - 1].Q, PATH = (() => { let [c, r] = G.start; const P = [[c, r]]; for (let k = 0; k < 30 && !G.term(c, r); k++) { [c, r] = G.step(c, r, argQ(FIN, c, r)); P.push([c, r]); } return P; })();
const IDEAL = {}; for (let i = 0; i < PATH.length - 1; i++) IDEAL[key(...PATH[i])] = GAM ** (PATH.length - 2 - i);
const PATHERR = Math.max(...PATH.slice(0, -1).map(([c, r]) => Math.abs(maxQ(FIN, c, r) - IDEAL[key(c, r)])));
const lens = E.map(e => e.tr.length), ends = E.map(e => e.end);
const LAST = E.slice(NEP - 100), AVGRET = LAST.reduce((s, e) => s + e.ret, 0) / 100, CATS = LAST.filter(e => e.end < 0).length;
const AVGLEN = LAST.reduce((s, e) => s + e.tr.length, 0) / 100;
const UNTRIED = (() => { const seen = new Set(); E.forEach(e => e.tr.forEach(t => seen.add(qk(t.c, t.r, t.m)))); let n = 0; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) for (let m = 0; m < 4; m++) if (!seen.has(qk(c, r, m))) n++; return n; })();
// TD comparison (the td reel's run): the cell beside the cheese before and after its cat scare
// Pip's habit from the policy/values/TD reels: follow the arrows, 1 move in 5 any direction at random. HV = its exact values (repeated sweeps).
const ARR = new Array(20).fill(-1);
[[0, 3, 0], [0, 2, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [2, 1, 0], [1, 2, 3], [1, 3, 1], [2, 3, 1], [3, 3, 1], [4, 3, 0], [4, 2, 0], [4, 1, 0], [3, 2, 1]].forEach(([c, r, m]) => { ARR[key(c, r)] = m; });
const HV = (() => { let V = new Array(20).fill(0); for (let it = 0; it < 400; it++) { const N2 = V.slice(); for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.term(c, r) || G.isRock(c, r)) continue; let v = 0;
  for (let m = 0; m < 4; m++) { const p = (m === ARR[key(c, r)] ? .8 : 0) + .05, [nc, nr] = G.step(c, r, m), rw = G.term(nc, nr); v += p * (rw + (rw ? 0 : GAM * V[key(nc, nr)])); } N2[key(c, r)] = v; } V = N2; } return V; })();
const f2 = v => num(v), f3 = v => num(v, 3);
// your-turn summaries per eps
// exact best-play values Q* (sweeps of the optimality rule; used only to check which arrows are best moves)
const VST = (() => { let V = new Array(20).fill(0); for (let it = 0; it < 300; it++) { const N2 = V.slice(); for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.term(c, r) || G.isRock(c, r)) continue; let b = -9; for (let m = 0; m < 4; m++) { const [nc, nr] = G.step(c, r, m), rw = G.term(nc, nr); b = Math.max(b, rw + (rw ? 0 : GAM * V[key(nc, nr)])); } N2[key(c, r)] = b; } V = N2; } return V; })();
const QST = (c, r, m) => { const [nc, nr] = G.step(c, r, m), rw = G.term(nc, nr); return rw + (rw ? 0 : GAM * VST[key(nc, nr)]); };
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);
const bestArrows = Q => CELLS.filter(([c, r]) => !flat(Q, c, r) && QST(c, r, argQ(Q, c, r)) > Math.max(...[0, 1, 2, 3].map(m => QST(c, r, m))) - 1e-9).length;
const VIS = new Array(20).fill(0); E.forEach(e => e.tr.forEach(t => VIS[key(t.c, t.r)]++));
const BADC = CELLS.filter(([c, r]) => !flat(FIN, c, r) && QST(c, r, argQ(FIN, c, r)) < Math.max(...[0, 1, 2, 3].map(m => QST(c, r, m))) - 1e-9), QMARK = CELLS.filter(([c, r]) => flat(FIN, c, r)).length;
const EPSV = [0, .1, .2, .5, 1];
let QBEST; function summary(eps) { const R = RUN(eps), Q = R[NEP - 1].Q, last = R.slice(NEP - 100); return { Q, best: bestArrows(Q), start: maxQ(Q, 0, 3), ret: last.reduce((s, e) => s + e.ret, 0) / 100, len: last.reduce((s, e) => s + e.tr.length, 0) / 100 }; }

QBEST = summary(1).Q;  // eps = 1 run: every cell visited, all numbers within 0.006 of Q*, used to illustrate Q*
// ───── drawing
function corners(c, r) { const x0 = OX + c * CS, y0 = OY + r * CS, x1 = x0 + CS, y1 = y0 + CS; return { x0, y0, x1, y1, xm: (x0 + x1) / 2, ym: (y0 + y1) / 2 }; }
function tri(xm, ym, x0, y0, x1, y1, m, col, a) { const P = [[[x0, y0], [x1, y0]], [[x1, y0], [x1, y1]], [[x1, y1], [x0, y1]], [[x0, y1], [x0, y0]]][m];
  ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(xm, ym); ctx.lineTo(...P[0]); ctx.lineTo(...P[1]); ctx.closePath(); ctx.fill(); ctx.restore(); }
function wedges(Q, c, r, x0, y0, x1, y1, hl) {
  const xm = (x0 + x1) / 2, ym = (y0 + y1) / 2;
  for (let m = 0; m < 4; m++) { const v = Q[qk(c, r, m)]; if (Math.abs(v) >= .004) tri(xm, ym, x0 + 2, y0 + 2, x1 - 2, y1 - 2, m, v > 0 ? C.gold : C.rose, lerp(.14, .62, clamp(Math.abs(v) / .9))); }
  if (hl) tri(xm, ym, x0 + 2, y0 + 2, x1 - 2, y1 - 2, hl[0], hl[1] || C.teal, hl[2] ?? .35);
  ctx.save(); ctx.globalAlpha = .55; rline(x0 + 3, y0 + 3, x1 - 3, y1 - 3, 300 + c * 9 + r, { c: C.line, w: .9, once: true, j: .6 }); rline(x1 - 3, y0 + 3, x0 + 3, y1 - 3, 330 + c * 9 + r, { c: C.line, w: .9, once: true, j: .6 }); ctx.restore();
}
function qnums(Q, c, r, x0, y0, x1, y1, s, a = 1, only, d3) {
  const xm = (x0 + x1) / 2, ym = (y0 + y1) / 2, d = (x1 - x0), P = [[xm, y0 + d * .17], [x1 - d * .2, ym + 1], [xm, y1 - d * .15], [x0 + d * .2, ym + 1]];
  for (let m = 0; m < 4; m++) { if (only != null && only !== m) continue; const v = Q[qk(c, r, m)]; txt(d3 ? num(v, 3) : num(v), P[m][0], P[m][1], { s, a, c: v < -.004 ? C.roseInk : v > .004 ? C.goldInk : C.soft }); }
}
// whole maze with four triangles per cell. o = { t, nums(c, r) -> alpha, hl: { [key]: [m, col, a] }, pop, hi }
function qmaze(Q, o = {}) {
  for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.term(c, r) || G.isRock(c, r)) continue; const k = corners(c, r); wedges(Q, c, r, k.x0, k.y0, k.x1, k.y1, o.hl && o.hl[key(c, r)]); }
  maze({ t: o.t || 0, pop: o.pop, hi: o.hi, draw: o.draw, labels: o.labels });
  if (o.nums) for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.term(c, r) || G.isRock(c, r)) continue; const a = typeof o.nums === 'function' ? o.nums(c, r) : 1; if (a > 0) { const k = corners(c, r); qnums(Q, c, r, k.x0, k.y0, k.x1, k.y1, 12, a); } }
}
// a big copy of one cell (in the card area)
function zoom(Q, c, r, x, y, s, o = {}) {
  const a = o.a ?? 1; if (a <= 0) return; ctx.save(); ctx.globalAlpha = a;
  box(x, y, s, s, C.card); wedges(Q, c, r, x, y, x + s, y + s, o.hl); L.rrect(x, y, s, s, 640 + c * 5 + r, { w: 1.4 });
  qnums(Q, c, r, x, y, x + s, y + s, o.d3 ? 14 : 15, 1, null, o.d3);
  if (o.label) txt(o.label, x + s / 2, y + s + 11, { s: 13, c: C.soft }); ctx.restore();
}
function dim(a = .62) { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, 285); ctx.restore(); }
function arrowsQ(Q, t0, t, o = {}) { let n = 0; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.term(c, r) || G.isRock(c, r)) continue;
  if (flat(Q, c, r)) { txt('?', cx(c), cy(r) + 4, { s: 18, c: '#c2b6a6', a: clamp(t - t0 - 2.6) }); continue; }
  const a = clamp((t - t0 - n++ * .15) * 3) * (o.a ?? 1); if (a <= 0) continue; G.arrowIn(c, r, argQ(Q, c, r), 400 + key(c, r), { c: '#8a5d08', w: 1.5 }); } }
function peekArrows(t, a) { if (a <= 0) return; [[1, 3], [3, 3]].forEach(([c, r], i) => { const dx = c - 2, dy = r - 3; ctx.save(); ctx.globalAlpha = a; arrow(cx(2) + dx * 14, cy(3) + dy * 14, cx(c) - dx * 14, cy(r) - dy * 14, 300 + i, { c: C.soft }); ctx.restore(); }); }
// small chart under the maze: steps per walk (teal = cheese, rose = cat)
function strip(n, label) {
  const x0 = 40, x1 = 360, y0 = 384, y1 = 300, w = (x1 - x0) / NEP, max = 80;
  rline(x0, y0, x1, y0, 900, { c: C.line, w: 1 }); txt(label, x0, y1 - 2, { s: 14, al: 'left', c: C.soft });
  for (let i = 0; i < n; i++) { const h = Math.min(max, lens[i]) / max * (y0 - y1 - 14); box(x0 + i * w, y0 - h, Math.max(1, w - .2), h, ends[i] > 0 ? C.teal : C.rose); }
  txt('walk 1', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NEP}`, x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
  txt('80+', x0 - 4, y1 + 12, { s: 12, c: C.soft, al: 'right' });
}
// chart of the four numbers of the cell beside the cheese, walk by walk
function q30Chart(n, o = {}) {
  const x0 = 64, x1 = 336, y0 = 392, y1 = 304, NW = 20, X = i => x0 + i * (x1 - x0) / (NW - 1), Y = v => lerp(y0, y1, (v + .6) / 1.6);
  rline(x0 - 6, Y(0), x1 + 4, Y(0), 960, { c: C.line, w: 1 }); rline(x0 - 6, Y(1), x1 + 4, Y(1), 961, { c: '#e4d8c7', w: 1, once: true });
  txt('1', x0 - 16, Y(1), { s: 12, c: C.soft }); txt('0', x0 - 16, Y(0), { s: 12, c: C.soft }); txt('−0.5', x0 - 20, Y(-.5), { s: 12, c: C.soft });
  txt('beside the cheese, after each walk:', x0 - 30, y1 - 13, { s: 14, al: 'left', c: C.soft });
  txt('right', 290, y1 - 13, { s: 14, al: 'left', c: C.goldInk }); txt('down', 330, y1 - 13, { s: 14, al: 'left', c: C.roseInk });
  [[1, '#c99a3a'], [2, C.rose]].forEach(([m, col]) => { for (let i = 0; i < Math.min(n, NW); i++) { if (i) rline(X(i - 1), Y(Q30[i - 1][m]), X(i), Y(Q30[i][m]), 970 + i + m * 30, { c: col, w: 1.4, once: true }); dot(X(i), Y(Q30[i][m]), 3.2, col); } });
  if (n > CATW) { txt('cat', X(CATW), Y(-.5) + 2, { s: 13, c: C.roseInk }); }
  if (o.labels) o.labels.forEach(i => { if (i < n) txt(f2(Q30[i][1]), i === NW - 1 ? X(i) + 20 : X(i), i === NW - 1 ? Y(Q30[i][1]) + 1 : Y(Q30[i][1]) - 11, { s: 13, c: C.goldInk }); });
}
function crossMap(x, y, a) { if (a <= 0) return; ctx.save(); ctx.globalAlpha = a;
  L.rrect(x - 22, y - 16, 44, 32, 820, { fill: C.card }); [-7, 0, 7].forEach((d, i) => rline(x - 15, y + d, x + 15, y + d, 821 + i, { c: C.line, w: 1, once: true }));
  rline(x - 26, y - 20, x + 26, y + 20, 830, { c: C.rose, w: 2.4 }); rline(x + 26, y - 20, x - 26, y + 20, 831, { c: C.rose, w: 2.4 }); ctx.restore(); }

const STEPS = [
 { h: 'Where TD left off', p: ['In the TD reel, Pip followed a fixed habit: the arrows, with one move in five random. TD learning worked out, from experience, how good each cell is **under that habit**. Here is that notebook, with the true values of the habit, which TD’s guesses head toward.', 'Now Pip wants to do better than its habit. Can the notebook tell it which move is best?'],
   shot: [3, 9],
   draw(t) { maze({ V: HV, nums: 1, t }); for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (ARR[key(c, r)] < 0) continue; const a = clamp((t - .6 - key(c, r) * .08) * 3); if (a > 0) { ctx.save(); ctx.globalAlpha = a * .8; G.arrowIn(c, r, ARR[key(c, r)], 500 + key(c, r)); ctx.restore(); } }
     pip(cx(0), cy(3), 1, t);
     if (t > 3) card(['value of a cell = expected return from there', 'following the arrows, 1 move in 5 random', { s: 'how good each cell is, not each move', c: C.soft }], t - 3, { gap: 1.2 }); } },
 { h: 'A notebook is not enough', p: [`From the bottom row’s middle, the cell to its right is worth ${f2(HV[key(3, 3)])}, to its left ${f2(HV[key(1, 3)])}. Right looks better, if Pip knows “right” leads there. Pip doesn’t know the rules: only after a move does it see the reward and where it landed.`, 'It could learn the rules too, a later topic. **Q-learning** skips them and learns a value for each move directly.'],
   draw(t) { maze({ V: HV, nums: (c, r) => ((c === 3 || c === 1) && r === 3 ? 1 : 0), t }); pip(cx(2), cy(3), 1, t); peekArrows(t, clamp(t - .6));
     [[1, 3], [3, 3]].forEach(([c, r], i) => txt('?', cx(c), cy(r) + 4, { s: 26, c: C.roseInk, a: clamp(t - 1.2 - i * .4) }));
     crossMap(344, 336, clamp(t - 2.6)); txt('no map', 344, 372, { s: 14, c: C.roseInk, a: clamp(t - 2.6) });
     if (t > 2) card(['Pip knows: the cell it is in', 'after a move: the reward and where it landed', { s: 'Pip does not know: where a move will lead', c: C.roseInk }], t - 2, { x: 24, w: 290, gap: 1.1, fs: 16 }); } },
 { h: 'A number for each move', p: ['Pip now keeps **four** numbers per cell, one per move, drawn as triangles, each pointing its way. Gold is positive, rose negative.', 'Each is an **action value**, Q(s, a). In the values reel it meant: make move a, then follow the habit. Here: make move a, then act **as well as possible**: the best policy’s Q, **Q***, said “Q star”. All start at 0.'],
   shot: [3, 10],
   draw(t) { const Qd = ZQ.slice(); qmaze(Qd, { t }); const k = corners(1, 3);
     zoom(QBEST, 0, 3, 40, 296, 92, { a: clamp(t - 1), label: 'the start cell, made big' });
     if (t > 1) { ctx.save(); ctx.globalAlpha = clamp(t - 1); arrow(cx(0) + 10, OY + 4 * CS - 2, 86, 293, 610, { c: C.soft }); ctx.restore(); }
     card(['up: number for “up”', 'right, down, left: the same', { s: 'shown: the Q* numbers', c: C.soft }], t - 2.5, { x: 150, w: 220, gap: 1.1, fs: 16 });
     pip(cx(0), cy(3), 1, t); } },
 { h: 'Choosing needs no map', p: ['To move, Pip looks only at its own cell and takes the move with the **biggest** number, ties broken at random. Bumping a wall is a move too, with its own number.', 'One move in five is still random, any direction, so every move gets tried. This habit is **ε-greedy**, with ε (epsilon) = 0.2: greedy means taking the biggest number.'],
   draw(t) { const Q = QBEST; qmaze(Q, { t, nums: (c, r) => (c === 0 && r === 2 ? clamp(t - 1) : 0) }); pip(cx(0), cy(3), 1, t);
     if (t > 2.2) { ctx.save(); ctx.globalAlpha = clamp(t - 2.2); G.arrowIn(0, 2, argQ(Q, 0, 2), 620, { c: C.teal, w: 2.2 }); ctx.restore(); }
     card([`in this cell: biggest is “${N[argQ(Q, 0, 2)]}”, ${f2(maxQ(Q, 0, 2))}  →  go ${N[argQ(Q, 0, 2)]}`, '4 moves in 5: the biggest number', { s: '1 move in 5: any direction at random', c: C.soft }], t - 3, { gap: 1.1 });
     caption('numbers shown: the Q* values Pip is after', clamp(t - 1)); } },
 { h: 'The update, in words', p: ['After each move, Pip updates that move’s number. The target: that move’s reward, plus 0.9 times the **biggest** of the four numbers where it landed.', 'Then it nudges the old number a fifth of the way toward the target. That fifth is **α (alpha)**, the learning rate, as in TD learning. On the cheese or cat, the walk ends and nothing is added.'],
   draw(t) { const Q = QBEST, mv = G.walkPos(DEMO, 0, 1, clamp(t - .4, 0, 1), 1); qmaze(Q, { t, hl: { [key(2, 0)]: [1, C.teal, .45 * clamp(t - 1.6)] } }); pip(mv.x, mv.y, 1, t);
     ghost(2, 0, 'move made', clamp(t - 1.6), 24);
     zoom(Q, 3, 0, 36, 296, 92, { label: 'landed here', a: clamp(t - 2.2), hl: [argQ(Q, 3, 0), C.gold, .3] });
     card(['target  =  reward for the move', '   + 0.9 × biggest in landed cell', 'new = old + 0.2 × (target − old)'], t - 3, { x: 145, w: 228, gap: 1.3, fs: 16 }); } },
 { h: 'In symbols', p: ['With s the cell Pip left, a its move, r the reward, and s′, said “s prime”, where it landed:', 'Q(s, a) becomes Q(s, a) + α × (r + γ × max Q(s′, ·) − Q(s, a)). “max Q(s′, ·)” is the biggest of the four numbers in s′. γ (gamma) = 0.9 and α = 0.2, the same knobs as in TD learning.'],
   draw(t) { qmaze(ZQ, { t }); pip(cx(0), cy(3), 1, t); dim(.7);
     const y = 70; txt('Q(s, a)  ←  Q(s, a) + α ( r + γ max Q(s′, ·) − Q(s, a) )', W / 2, y, { s: 19, a: clamp(t * 2) });
     const parts = [['s, a', 'cell left, move made', 92], ['r', 'reward for that move', 170], ['max Q(s′, ·)', 'best number where it landed', 245], ['r + γ max Q(s′, ·)', 'the target', 205]];
     [['s: cell Pip left', 1], ['a: move it made', 1.6], ['r: reward for that move', 2.2], ['s′: cell where it landed', 2.8], ['max Q(s′, ·): biggest of its four numbers', 3.4], ['target − old = the TD error, as before', 4.2]].forEach(([s, d], i) => txt(s, 60, 112 + i * 27, { s: 17, al: 'left', a: clamp((t - d) * 2), c: i === 5 ? C.goldInk : C.ink }));
     card(['γ = 0.9: later rewards count a bit less', 'α = 0.2: move one fifth of the way', { s: 'walk over (cheese or cat): no max term', c: C.soft }], t - 5.2, { gap: 1 }); } },
 { h: 'Walk 1', p: [`All numbers are 0, so every choice is a tie, a coin toss. For ${e0.length - 1} steps every update says “0 stays 0”.`, `On step ${e0.length}, from the cell below the cheese, one of the one-in-five random moves points up, onto the cheese: reward +1, walk over. Target 1, so that one number, “up” in that cell, goes from 0 to 0.2. Nothing else changes.`],
   shot: [5, 13],
   draw(t) { const dur = .2, n = e0.length, tEnd = (n - 1) * dur; let Q = ZQ.slice(), m;
     if (t < tEnd) { m = G.walkPos(e0, 0, n - 1, t, dur); trail(e0.slice(0, m.k + 1).map(s => [s.c, s.r]), .5); }
     else { m = G.walkPos(e0, n - 1, n, Math.min(1, t - tEnd), 1); trail(e0.map(s => [s.c, s.r]), .3); if (t > tEnd + 6) Q[qk(L0.c, L0.r, L0.m)] = L0.nw * ease((t - tEnd - 6) / 1); }
     qmaze(Q, { t, hl: t > tEnd + 1.5 ? { [key(L0.c, L0.r)]: [L0.m, C.teal, .3] } : null }); pip(m.x, m.y, m.face || 1, t);
     if (t < tEnd) caption(`walk 1, step ${Math.min(n - 1, m.k + 1)} (sped up)`);
     else { bubble('+1', cx(4), cy(0) - 18, t - tEnd - .8);
       zoom(Q, L0.c, L0.r, 36, 296, 92, { hl: [L0.m, C.teal, .3], label: 'the cell below the cheese', a: clamp(t - tEnd - 1.5) });
       card(['move: up (a random one)', 'target = 1   (walk over)', `new = 0 + 0.2 × (1 − 0) = ${f2(L0.nw)}`], t - tEnd - 2.5, { x: 150, w: 222, gap: 1.2, fs: 16 }); } } },
 { h: 'The news travels back', p: ['Walk 2 ends stepping right onto the cheese, so “right” there becomes 0.2. Walk 3: Pip steps right into that cell, whose biggest number is 0.2.', `Target = 0 + 0.9 × 0.2 = 0.18, and “right” in the cell it left becomes 0 + 0.2 × 0.18 = ${f2(S3.nw)}. As in TD, a guess learned from the next guess, but stored on a move.`],
   draw(t) { const Qb = Qat(W3I, W3K), Q = Qb.slice(); if (t > 6.5) Q[qk(S3.c, S3.r, S3.m)] = S3.nw * ease((t - 6.5) / 1);
     const mv = G.walkPos(E[W3I].tr, W3K, W3K + 1, clamp(t - .5, 0, 1), 1);
     qmaze(Q, { t, hl: { [key(S3.c, S3.r)]: [S3.m, C.teal, .35 * clamp(t - 1.6)] } }); pip(mv.x, mv.y, mv.face || 1, t);
     zoom(Q, S3.nc, S3.nr, 36, 296, 92, { label: 'landed: beside the cheese', a: clamp(t - 2), hl: [1, C.gold, .25 * clamp(t - 3)] });
     card([`biggest here: right, ${f2(S3.next)}`, `target = 0 + 0.9 × ${f2(S3.next)} = ${f2(S3.target)}`, `new = 0 + 0.2 × ${f2(S3.target)} = ${f2(S3.nw)}`], t - 3, { x: 150, w: 222, gap: 1.3, fs: 16 });
     caption(`walk 3, step ${W3K + 1}`, clamp(t - .2)); } },
 { h: 'The max ignores what comes next', p: [`Walk ${CATW + 1}. Pip steps right into the cell beside the cheese and updates at once, before choosing its next move. The biggest number there is ${f3(CP.next)}, for “right”: target 0.9 × ${f3(CP.next)} = ${f3(CP.target)}, and the old ${f3(CP.old)} becomes ${f3(CP.nw)}.`, 'The target asks: what if Pip makes the **best** move from here? Whatever Pip actually does next plays no part.'],
   shot: [4, 11],
   draw(t) { const Qb = Qat(CATW, CT.length - 2), Q = Qb.slice(); if (t > 6) Q[qk(CP.c, CP.r, CP.m)] = lerp(CP.old, CP.nw, ease((t - 6) / 1));
     const mv = G.walkPos(CT, CT.length - 2, CT.length - 1, clamp(t - .5, 0, 1), 1);
     qmaze(Q, { t, hl: { [key(CP.c, CP.r)]: [CP.m, C.teal, .35 * clamp(t - 1.6)] } }); pip(mv.x, mv.y, mv.face || 1, t);
     zoom(Q, 3, 0, 36, 296, 92, { label: 'landed: beside the cheese', a: clamp(t - 2), hl: [1, C.gold, .3 * clamp(t - 3)], d3: 1 });
     card([`biggest here: right, ${f3(CP.next)}`, `target = 0 + 0.9 × ${f3(CP.next)} = ${f3(CP.target)}`, `new = ${f3(CP.old)} + 0.2 × ${f3(CP.target - CP.old)} = ${f3(CP.nw)}`], t - 3, { x: 145, w: 228, gap: 1.3, fs: 16 }); } },
 { h: 'Bad news stays with the bad move', p: ['Then Pip’s next move is a random one: down, onto the cat. Reward −1, walk over. Only that cell’s “down” number changes, from 0 to −0.2. “Right” keeps its ' + f3(Q30[CATW][1]) + '.', 'With one number per cell, as in the TD reel, this −1 would have pulled down the whole cell’s value. Four numbers keep a bad move apart from a good one.'],
   draw(t) { const Qb = Qat(CATW, CT.length - 1), Q = Qb.slice(); if (t > 2.4) Q[qk(3, 0, 2)] = CL.nw * ease((t - 2.4) / .8);
     const mv = G.walkPos(CT, CT.length - 1, CT.length, clamp(t - .6, 0, 1), 1);
     qmaze(Q, { t, hl: { [key(3, 0)]: [2, C.rose, .3 * clamp(t - 1.6)] } }); pip(mv.x, mv.y, mv.face || 1, t); if (t > 1.2) bubble('−1', cx(3), cy(1) - 18, t - 1.4);
     zoom(Q, 3, 0, 36, 296, 92, { label: 'beside the cheese', a: clamp(t - 2), hl: [2, C.rose, .25], d3: 1 });
     card(['move: down (random)  →  cat', `down: 0 + 0.2 × (−1 − 0) = ${f3(CL.nw)}`, `right: still ${f3(Q30[CATW][1])}`], t - 2.6, { x: 150, w: 222, gap: 1.3, fs: 16 }); } },
 { h: 'Walk by walk', p: ['Two of the four numbers beside the cheese. Across: walk number, 1 to 20; up: the number after that walk. Gold is “right”, onto the cheese; rose is “down”, onto the cat.', `“Right” climbs: ${f2(Q30[1][1])}, ${f2(Q30[2][1])}, ${f2(Q30[3][1])} … ${f2(Q30[19][1])} after 20 walks. Cat scares only push “down” lower. “Right” never drops: it lands on the cheese, so its target is always exactly 1.`],
   draw(t) { const n = clamp(Math.floor(t / .55) + 1, 1, 20); qmaze(E[n - 1].Q, { t, nums: (c, r) => (c === 3 && r === 0 ? 1 : 0) }); pip(cx(ends[n - 1] > 0 ? 4 : 3), cy(ends[n - 1] > 0 ? 0 : 1), 1, t);
     q30Chart(n, { labels: [1, 2, 3, 19] }); caption(`after walk ${n}`, 0); } },
 { h: `${NEP} walks later`, p: [`Repeat for ${NEP} walks. Bars show each walk’s steps; teal cheese, rose cat. Early walks wander: ${lens.slice(0, 4).join(', ')} steps. Last-100 average: ${AVGLEN.toFixed(1)}.`, 'Arrows show Pip’s best guess at the best move; small numbers count visits. Rarely visited cells off the route can be wrong: the bottom row’s two left arrows are not best moves. A ? means all four numbers are still equal.'],
   draw(t) { let w, k; if (t < 5) { w = 1 + Math.floor(t / .8); k = (t % .8) / .8; } else { w = 7 + Math.floor((t - 5) / .03); k = ((t - 5) % .03) / .03; }
     const done = w >= NEP; if (done) { w = NEP - 1; k = 1; }
     const tr = E[w].tr, i = Math.min(tr.length - 1, Math.floor(k * tr.length)), q = tr[i];
     const Q = done ? FIN : Qat(w, i); qmaze(Q, { t }); if (done) { arrowsQ(FIN, 0, 10, { a: clamp(t - 14.5) }); CELLS.forEach(([c, r]) => txt(String(VIS[key(c, r)]), OX + c * CS + 5, OY + r * CS + CS - 9, { s: 12, al: 'left', c: C.soft, a: clamp(t - 15.5) })); }
     pip(done ? cx(4) : cx(q.c), done ? cy(0) : cy(q.r), 1, t); strip(done ? NEP : w, 'steps per walk'); caption(done ? `walk ${NEP} of ${NEP}` : `walk ${w + 1} of ${NEP}`); },
   shot: [3, 18] },
 { h: 'The best route’s numbers', p: [`Along the arrows, each cell’s biggest number: ${f2(maxQ(FIN, ...PATH[PATH.length - 2]))} beside the cheese, then ${f2(maxQ(FIN, ...PATH[PATH.length - 3]))}, ${f2(maxQ(FIN, ...PATH[PATH.length - 4]))}, down to ${f2(maxQ(FIN, 0, 3))} at the start. These are the values of always taking the shortest route: the Bellman reel’s V star, learned from walks alone.`, `Under the old habit that cell is worth ${f2(HV[key(3, 0)])}. Q-learning learned 1, despite Pip’s random moves: the max makes every target assume the **best next move**.`],
   draw(t) { qmaze(FIN, { t }); arrowsQ(FIN, -5, t + 10, { a: .5 });
     PATH.slice(0, -1).reverse().forEach(([c, r], i) => { const a = clamp((t - .6 - i * .5) * 2); if (a <= 0) return; box(cx(c) - 21, cy(r) - 11, 42, 22, C.card, .9 * a); txt(f2(maxQ(FIN, c, r)), cx(c), cy(r), { s: 18, c: C.goldInk, a }); });
     card([`biggest numbers: ${PATH.slice(0, -1).reverse().slice(0, 4).map(([c, r]) => f2(maxQ(FIN, c, r))).join(', ')} … ${f2(maxQ(FIN, 0, 3))}`, `perfect:              1, 0.9, 0.81, 0.73 … 0.53`, { s: `old habit, same cell beside the cheese: ${f2(HV[key(3, 0)])}`, c: C.soft }], t - 4.5, { gap: 1.1 }); } },
 { h: 'Why they sit still', p: ['In the TD reel, a fixed α of 0.2 kept the guesses wobbling. Here they settle exactly. Why?', 'The floor is dry, and the max ignores Pip’s random next move. Once the next cell’s numbers settle, every target for a move is the same number, walk after walk: nothing left to wobble. TD’s targets kept changing because random moves changed where Pip went next.'],
   draw(t) { qmaze(FIN, { t }); arrowsQ(FIN, -5, t + 10, { a: .35 }); dim(.25);
     zoom(FIN, 3, 0, 36, 296, 92, { label: 'beside the cheese', a: clamp(t - .5) });
     card(['“right”: target = 1, every single time', `cell to its left: 0.9 × ${f2(maxQ(FIN, 3, 0))} = ${f2(GAM * maxQ(FIN, 3, 0))}, every time`, { s: 'same target each time: no wobble', c: C.soft }], t - 1.5, { x: 140, w: 236, gap: 1.4, fs: 15 }); } },
 { h: 'But Pip still stumbles', p: [`Those numbers describe a Pip that never moves at random. The real Pip still does, one time in five. Each dot: one of its last 100 walks, at its return; gold cheese, rose cat. ${CATS} met the cat; the average is ${f2(AVGRET)}, not ${f2(maxQ(FIN, 0, 3))}.`, 'Q-learning learns the best way to act, not how Pip acts while learning. The next reel learns the real walker’s numbers.'],
   draw(t) { qmaze(FIN, { t }); arrowsQ(FIN, -5, t + 10, { a: .45 });
     const n = Math.min(100, Math.floor(t * 9)); const pts = LAST.slice(0, n);
     const x0 = 50, x1 = 360, y0 = 392, y1 = 304, X = i => x0 + i * (x1 - x0) / 99, Y = v => lerp(y0, y1, (v + 1) / 2);
     rline(x0 - 6, Y(0), x1 + 4, Y(0), 960, { c: C.line, w: 1 }); txt('+1', x0 - 22, Y(1), { s: 12, c: C.soft }); txt('0', x0 - 18, Y(0), { s: 12, c: C.soft }); txt('−1', x0 - 22, Y(-1), { s: 12, c: C.soft });
     txt(`return of each of the last 100 walks, from the start`, x0 - 20, y1 - 12, { s: 14, al: 'left', c: C.soft });
     pts.forEach((e, i) => dot(X(i), Y(e.ret), 2.6, e.end > 0 ? C.gold : C.rose)); txt(`walk ${NEP - 99}`, x0, y0 + 9, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NEP}`, x1, y0 + 9, { s: 12, c: C.soft, al: 'right' });
     if (n >= 100) { const a = clamp(t - 11.5); rline(x0, Y(maxQ(FIN, 0, 3)), x1, Y(maxQ(FIN, 0, 3)), 980, { c: C.goldInk, w: 1.2, once: true, p: a }); rline(x0, Y(AVGRET), x1, Y(AVGRET), 981, { c: C.ink, w: 1.8, once: true, p: clamp(t - 12.3) });
       box(x1 - 118, Y(maxQ(FIN, 0, 3)) - 25, 120, 16, C.bg, .9 * a); txt(`biggest Q at start ${f2(maxQ(FIN, 0, 3))}`, x1, Y(maxQ(FIN, 0, 3)) - 17, { s: 13, al: 'right', c: C.goldInk, a });
       box(x1 - 70, Y(AVGRET) + 5, 72, 16, C.bg, .9 * clamp(t - 12.3)); txt(`average ${f2(AVGRET)}`, x1, Y(AVGRET) + 13, { s: 13, al: 'right', a: clamp(t - 12.3) }); }
     pip(cx(0), cy(3), 1, t); },
   shot: [5, 14] },
 { h: 'Your turn', ctl: { label: 'ε', values: EPSV, def: 2, fmt: v => v, onChange: st => { st.path = null; } }, tap: tapCell,
   p: [`Drag **ε**, how often Pip moves at random, to rerun all ${NEP} walks. At every setting the start’s number reaches 0.53. More randomness means more cells visited: at 1, all ${summary(1).best} arrows are best moves, at 0.2 only ${summary(.2).best}. But a more random Pip earns less along the way.`, 'Tap a cell to drop Pip there and follow the arrows.'],
   draw(t, st) { const e = EPSV[st.ci ?? 2], S = summary(e), P0 = st.prevCi != null ? summary(EPSV[st.prevCi]).Q : S.Q, u = ease((t - (st.changed ?? -9)) / .9);
     const Q = S.Q.map((v, i) => lerp(P0[i], v, u)); qmaze(Q, { t }); arrowsQ(S.Q, -5, t + 10, { a: .55 });
     const P = st.path; if (P) { const t2 = t - st.pt, kk = clamp(Math.floor(t2 / .45), 0, P.length - 1), uu = ease((t2 % .45) / .45 * 1.3), a = P[kk], b = P[Math.min(P.length - 1, kk + 1)];
       trail(P.slice(0, kk + 1), .7); pip(lerp(cx(a[0]), cx(b[0]), uu), lerp(cy(a[1]), cy(b[1]), uu), b[0] - a[0], t);
       caption(kk >= P.length - 1 ? (st.why ? st.why : `reached the ${G.term(...P[P.length - 1]) > 0 ? 'cheese' : 'cat'} in ${P.length - 1} steps`) : `step ${kk}`);
     } else { pip(cx(0), cy(3), 1, t); caption('tap a cell to drop Pip there'); }
     card([`ε = ${e}: random move ${e === 0 ? 'never' : e === 1 ? 'every time' : (e === .1 ? '1 time in 10' : e === .2 ? '1 time in 5' : '1 time in 2')}`, `biggest Q at the start: ${f2(S.start)}   (perfect 0.53)`, `real return, last 100 walks: ${f2(S.ret)}`, { s: `arrows that are best moves: ${S.best} of ${CELLS.length}`, c: C.soft }], 10, { lh: 21, fs: 16, y: 290 }); } },
 { h: 'Where it came from', p: ['Chris Watkins introduced Q-learning in his 1989 PhD thesis. In 1992 he and Peter Dayan proved the table reaches the best values, if every move everywhere keeps being tried and steps shrink suitably. Our run broke both, yet found the route; its rarely tried cells stayed wrong.', 'In 2015 DeepMind’s **DQN** swapped the table for a network reading Atari screens: one Q per joystick move.'],
   draw(t) { qmaze(FIN, { t }); arrowsQ(FIN, -5, t + 10, { a: .5 }); pip(cx(0), cy(3), 1, t); dim(.35);
     card(['1989  Watkins: Q-learning', '1992  Watkins and Dayan: it converges (table)', '2015  DQN: a network instead of a table', { s: 'screen pixels in, one Q per joystick move out', c: C.soft }], t - .5, { gap: 1.1 }); } },
 { h: 'Recap', p: ['Without a map, keep a number for every move: **Q(s, a)**. Pick the biggest, explore now and then.', 'After each move, nudge its number toward the reward plus γ times the **best** number where you landed. Because of that max, the numbers head for the best way to act, even while Pip acts at random.'],
   draw(t) { qmaze(FIN, { t }); arrowsQ(FIN, -5, t + 10); pip(cx(0), cy(3), 1, t);
     card(['one number per cell and move: Q(s, a)', 'act: biggest number, random 1 time in 5 (ε)', 'target = r + γ × biggest Q where you land', 'new = old + α × (target − old)'], t - .3, { gap: 1 }); } },
];
function ghost(c, r, s, a, dy) { if (a <= 0) return; box(cx(c) - 34, cy(r) + dy - 9, 68, 18, C.card, .85 * a); txt(s, cx(c), cy(r) + dy, { s: 13, c: '#6d645b', a }); }
// Pip follows its arrows from a cell; stops if all four numbers are equal (no clue) or it loops
function follow(Q, from) {
  let [c, r] = from; const P = [[c, r]], seen = new Set([key(c, r)]);
  while (!G.term(c, r) && P.length < 30) {
    if (flat(Q, c, r)) return { P, why: (P.length > 1 ? `after ${P.length - 1} steps: ` : '') + 'no clue here, all four numbers equal' };
    const [nc, nr] = G.step(c, r, argQ(Q, c, r)); if (nc === c && nr === r) return { P, why: 'its best move bumps a wall' }; c = nc; r = nr; P.push([c, r]);
    if (seen.has(key(c, r))) return { P, why: 'going in circles' }; seen.add(key(c, r));
  }
  return { P, why: null };
}
function tapCell(x, y, st, t) {
  const c = Math.floor((x - OX) / CS), r = Math.floor((y - OY) / CS);
  if (c < 0 || c >= 5 || r < 0 || r >= 4 || G.isRock(c, r) || G.term(c, r)) return;
  const f = follow(summary(EPSV[st.ci ?? 2]).Q, [c, r]); st.path = f.P; st.why = f.why; st.pt = t; st.changed = -9;
}
RLStory.register('qlearning', { title: 'Q-Learning', track: 'Model-free', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { VIS, BADC, QMARK, QBEST, bestArrows, E, e0, L0, W2, L2, S3, W3K, CATW, CP, CL, Q30, FIN, PATH, PATHERR, AVGRET, CATS, AVGLEN, UNTRIED, HV, lens, summary, EPSV, maxQ, argQ } });
})();
