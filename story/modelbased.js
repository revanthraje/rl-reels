// Model-based RL as a story: Pip writes down what its moves did (a learned model), plans with it (value iteration on
// the notebook), then uses it for imagined practice (Dyna-Q), races plain Q-learning on the same seeds, and is fooled
// when the cat moves. Every number comes from the seeded runs below (main seed 54, the Q-learning reel's seed).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, hachure, txt, arrow, card, bubble, caption, dot, box, alpha, num } = L;
let ctx;
const G = L.gridWorld({}), G1 = L.gridWorld({ cat: [3, 3] });          // G: the usual maze; G1: the cat has moved to the bottom row
const { cx, cy, key, trail, pip, MOVES } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GAM = 0.9, AL = 0.2, EPS = 0.2, NIM = 10, SEED = 54, NAMES = ['up', 'right', 'down', 'left'];
const qk = (c, r, m) => key(c, r) * 4 + m;
const CELLS1 = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G1.term(c, r) && !G1.isRock(c, r)) CELLS1.push([c, r]);
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);

// ───── Dyna-Q (n = 0 is plain Q-learning, identical to the qlearning reel's runQ). The notebook (model) keeps, for each
// tried (cell, move), the latest landing and reward. Acting rng = rng(seed), imagination rng = rng(seed + 1000).
function runDyna(Gw, n, seed, walks, init, rec) {
  const R = rng(seed), RP = rng(seed + 1000), Q = init ? Float64Array.from(init.Q) : new Float64Array(80);
  const M = init ? new Map(init.M) : new Map(), seen = init ? init.seen.slice() : [], Wk = [], Qs = [], Ms = [];
  const best = (c, r) => Math.max(Q[qk(c, r, 0)], Q[qk(c, r, 1)], Q[qk(c, r, 2)], Q[qk(c, r, 3)]);
  const upd = (c, r, m, rew, nc, nr) => { const k = qk(c, r, m), old = Q[k], tg = rew + (rew ? 0 : GAM * best(nc, nr)); Q[k] = old + AL * (tg - old); return { c, r, m, nc, nr, rew, old, tg, nw: Q[k] }; };
  let ups = 0;
  for (let e = 0; e < walks; e++) {
    let [c, r] = Gw.start; const tr = [];
    for (let s = 0; s < 3000; s++) {
      let m, rnd = false;
      if (R() < EPS) { m = Math.floor(R() * 4); rnd = true; }
      else { let b = -1e9, ms = []; for (let a = 0; a < 4; a++) { const q = Q[qk(c, r, a)]; if (q > b + 1e-9) { b = q; ms = [a]; } else if (q > b - 1e-9) ms.push(a); } m = ms[Math.floor(R() * ms.length)]; }
      const [nc, nr] = Gw.step(c, r, m), rew = Gw.term(nc, nr), u = upd(c, r, m, rew, nc, nr); u.rnd = rnd; ups++;
      const k = qk(c, r, m); if (!M.has(k)) seen.push(k); M.set(k, [nc, nr, rew]);
      const im = rec ? [] : null;
      for (let i = 0; i < n; i++) { const kk = seen[Math.floor(RP() * seen.length)], [a, b, rw] = M.get(kk), cell = kk >> 2; const v = upd(cell % 5, Math.floor(cell / 5), kk & 3, rw, a, b); ups++; if (rec) im.push(v); }
      if (rec) u.im = im; tr.push(u); c = nc; r = nr; if (rew) break;
    }
    Wk.push({ tr, len: tr.length, end: tr[tr.length - 1].rew });
    if (rec) { Qs.push(Array.from(Q)); Ms.push(M.size); }
  }
  return { W: Wk, Q: Array.from(Q), M, seen, ups, Qs, Ms };
}
const QL = runDyna(G, 0, SEED, 30, null, true), DY = runDyna(G, NIM, SEED, 50, null, true);
const W1 = QL.W[0].tr, LAST1 = W1[W1.length - 1];
// notebook after walk 1, built step by step (same walk for both methods: all numbers are 0 until the cheese)
const NB = []; { const M = new Map(); W1.forEach(s => { M.set(qk(s.c, s.r, s.m), [s.nc, s.nr, s.rew]); NB.push(new Map(M)); }); }
const M1 = NB[NB.length - 1], NE1 = M1.size, NC1 = new Set([...M1.keys()].map(k => k >> 2)).size;
const SAME1 = DY.W[0].len === QL.W[0].len && DY.W[0].tr.every((s, i) => s.m === W1[i].m);

// ───── value iteration on a notebook: value = best over NOTED moves of [reward + γ × value of landing]; no entry → 0
function viNote(M) {
  const H = [new Array(20).fill(0)];
  for (let it = 0; it < 60; it++) {
    const V = H[H.length - 1], bv = new Array(20).fill(-1e9);
    for (const [k, [nc, nr, rw]] of M) { const cell = k >> 2, v = rw + (rw ? 0 : GAM * V[key(nc, nr)]); if (v > bv[cell]) bv[cell] = v; }
    const N = bv.map(v => (v > -1e8 ? v : 0)); H.push(N); if (N.every((v, i) => Math.abs(v - V[i]) < 1e-12)) break;
  }
  return H;
}
function noteBest(M, V, c, r) { let b = -1e9, bm = -1; for (let m = 0; m < 4; m++) { const e = M.get(qk(c, r, m)); if (!e) continue; const v = e[2] + (e[2] ? 0 : GAM * V[key(e[0], e[1])]); if (v > b + 1e-12) { b = v; bm = m; } } return bm; }
const VH = viNote(M1), V1 = VH[VH.length - 1], VS = VH.length - 2;   // VS sweeps that changed something, then one with no change
const VSW = VH.length - 1;                                             // total sweeps run, including the final check
const ROUTE = (() => { let [c, r] = G.start; const P = [[c, r]]; while (!G.term(c, r) && P.length < 30) { const m = noteBest(M1, V1, c, r); if (m < 0) break; [c, r] = G.step(c, r, m); P.push([c, r]); } return P; })();

// ───── wet-floor counting (the mdp reel's run: rng(2), Pip tries 'right' at the start 1000 times, slip 0.2)
const WETC = (() => { const R = rng(2), cum = [[0, 0, 0]]; for (let i = 0; i < 1000; i++) { const u = R(); let mm = 1; if (u >= .8) mm = u < .9 ? 0 : 2; const [nc, nr] = G.step(0, 3, mm), k = nc === 1 ? 0 : nr === 2 ? 1 : 2; const a = cum[cum.length - 1].slice(); a[k]++; cum.push(a); } return cum; })();

// ───── Q-learning vs Dyna-Q after walk 2; walk 3 lengths
const nz = Q => Q.filter(v => Math.abs(v) > 1e-9).length;
const NZQ2 = nz(QL.Qs[1]), NZD2 = nz(DY.Qs[1]);
// the real step shown for Dyna: first step of walk 3 where at least 3 of its 10 imagined updates change a number
const DW3 = DY.W[2].tr, DSI = DW3.findIndex(s => s.nw - s.old > .05 && s.im.filter(v => Math.abs(v.nw - v.old) > 1e-9).length >= 3), DS = DW3[DSI];
const Qbefore = (Rn, e, i) => { const Q = (e > 0 ? Rn.Qs[e - 1] : new Array(80).fill(0)).slice(); for (let k = 0; k < i; k++) { const s = Rn.W[e].tr[k]; Q[qk(s.c, s.r, s.m)] = s.nw; s.im && s.im.forEach(v => { Q[qk(v.c, v.r, v.m)] = v.nw; }); } return Q; };
const QDS0 = Qbefore(DY, 2, DSI);                                      // numbers just before that real step
const MDS = (() => { const M = new Map(); for (let e = 0; e < 2; e++) DY.W[e].tr.forEach(s => M.set(qk(s.c, s.r, s.m), [s.nc, s.nr, s.rew])); DW3.slice(0, DSI + 1).forEach(s => M.set(qk(s.c, s.r, s.m), [s.nc, s.nr, s.rew])); return M; })();

// ───── the race: 20 runs (seeds 1 to 20), 30 walks, the same seeds for every n. Lazy + memoised.
const RACE = {}, NV = [0, 1, 5, 10, 20, 50];
function race(n) { if (RACE[n]) return RACE[n]; const cur = new Array(30).fill(0); let t10 = 0, ups = 0, real = 0;
  for (let s = 1; s <= 20; s++) { const D = runDyna(G, n, s, 30); D.W.forEach((w, i) => { cur[i] += w.len / 20; real += w.len; if (i < 10) t10 += w.len / 20; }); ups += D.ups; }
  return (RACE[n] = { cur, t10, late: cur.slice(20).reduce((a, b) => a + b, 0) / 10, perReal: ups / real, ups: ups / 20 }); }

// ───── the catch: after Dyna-Q's 50 walks the cat moves to the bottom row (3,3); the old cat cell (3,1) is now free
const argQ = (Q, c, r) => { let b = -1e9, bm = 0; for (let m = 0; m < 4; m++) if (Q[qk(c, r, m)] > b + 1e-12) { b = Q[qk(c, r, m)]; bm = m; } return bm; };
const maxQ = (Q, c, r) => Math.max(Q[qk(c, r, 0)], Q[qk(c, r, 1)], Q[qk(c, r, 2)], Q[qk(c, r, 3)]);
function qroute(Gw, Q) { let [c, r] = Gw.start; const P = [[c, r]]; for (let k = 0; k < 30 && !Gw.term(c, r); k++) { [c, r] = Gw.step(c, r, argQ(Q, c, r)); P.push([c, r]); } return P; }
const DQ50 = DY.Q, R50 = qroute(G, DQ50), Q50S = maxQ(DQ50, 0, 3), CSEED = 154;
const CH = runDyna(G1, NIM, CSEED, 10, { Q: DY.Q, M: DY.M, seen: DY.seen }, true);
const CW1 = CH.W[0].tr, CW1RND = CW1.some(s => s.rnd);
const FIXW = CH.Qs.findIndex(Q => { const P = qroute(G1, Q); return G1.term(...P[P.length - 1]) > 0; }) + 1;  // first walk after which the arrows reach the cheese
const RFIX = qroute(G1, CH.Qs[FIXW - 1]);
const STALE = [...CH.M.entries()].filter(([k, e]) => e[0] === 3 && e[1] === 1 && e[2] < 0).length;      // notebook entries still saying 'onto the old cat cell: −1'
const BQ0 = [0, 1, 2, 3].map(m => CH.Qs[0][qk(2, 3, m)]), NBUMP = CH.W[1].tr.filter(q => q.c === 2 && q.r === 3 && q.nc === 2 && q.nr === 3).length;
const BQ1 = (() => { const Q = CH.Qs[0].slice(); CH.W[1].tr.slice(0, -1).forEach(q => { Q[qk(q.c, q.r, q.m)] = q.nw; q.im.forEach(v => { Q[qk(v.c, v.r, v.m)] = v.nw; }); }); return [0, 1, 2, 3].map(m => Q[qk(2, 3, m)]); })();
const STALEK = [...CH.M.entries()].filter(([k, e]) => e[0] === 3 && e[1] === 1 && e[2] < 0).map(([k]) => k);

// ───── drawing helpers
const f2 = v => num(v), f3 = v => num(v, 3), hund = v => String(Math.round(v / 100) * 100);
function corners(Gx, c, r) { const x0 = Gx.ox + c * Gx.cs, y0 = Gx.oy + r * Gx.cs; return { x0, y0, x1: x0 + Gx.cs, y1: y0 + Gx.cs }; }
function tri(xm, ym, x0, y0, x1, y1, m, col, a) { const P = [[[x0, y0], [x1, y0]], [[x1, y0], [x1, y1]], [[x1, y1], [x0, y1]], [[x0, y1], [x0, y0]]][m];
  ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(xm, ym); ctx.lineTo(...P[0]); ctx.lineTo(...P[1]); ctx.closePath(); ctx.fill(); ctx.restore(); }
function wedges(Q, c, r, x0, y0, x1, y1, hl) {
  const xm = (x0 + x1) / 2, ym = (y0 + y1) / 2;
  for (let m = 0; m < 4; m++) { const v = Q[qk(c, r, m)]; if (Math.abs(v) >= .004) tri(xm, ym, x0 + 2, y0 + 2, x1 - 2, y1 - 2, m, v > 0 ? C.gold : C.rose, lerp(.16, .65, clamp(Math.abs(v) / .9))); }
  if (hl) tri(xm, ym, x0 + 2, y0 + 2, x1 - 2, y1 - 2, hl[0], hl[1] || C.teal, hl[2] ?? .35);
  alpha(.5, () => { rline(x0 + 3, y0 + 3, x1 - 3, y1 - 3, 300 + c * 9 + r, { c: C.line, w: .9, once: true, j: .6 }); rline(x1 - 3, y0 + 3, x0 + 3, y1 - 3, 330 + c * 9 + r, { c: C.line, w: .9, once: true, j: .6 }); });
}
// whole maze with four triangles per cell (gold +, rose −). o = { t, hl: { key: [m, col, a] }, Gx, pop, nums }
function qmaze(Q, o = {}) { const Gx = o.Gx || G, cells = Gx === G1 ? CELLS1 : CELLS;
  for (const [c, r] of cells) { const k = corners(Gx, c, r); wedges(Q, c, r, k.x0, k.y0, k.x1, k.y1, o.hl && o.hl[key(c, r)]); }
  Gx.maze({ t: o.t || 0, pop: o.pop, labels: o.labels });
  if (o.nums) for (const [c, r] of cells) { const a = o.nums(c, r); if (a <= 0) continue; const k = corners(Gx, c, r), d = Gx.cs, xm = (k.x0 + k.x1) / 2, ym = (k.y0 + k.y1) / 2;
    const P = [[xm, k.y0 + d * .17], [k.x1 - d * .2, ym + 1], [xm, k.y1 - d * .15], [k.x0 + d * .2, ym + 1]];
    for (let m = 0; m < 4; m++) { const v = Q[qk(c, r, m)]; txt(num(v), P[m][0], P[m][1], { s: 12, a, c: v < -.004 ? C.roseInk : v > .004 ? C.goldInk : C.soft }); } }
}
// one notebook entry drawn in the maze: an arrow toward the landing cell (or a U-turn for a bump); gold = +1, rose = −1
function entry(c, r, m, e, a = 1, o = {}) { if (a <= 0) return; const [nc, nr, rw] = e, [dx, dy] = MOVES[m], col = o.c || (rw > 0 ? C.goldInk : rw < 0 ? C.roseInk : '#3f7f7c');
  const px = -dy * 6, py = dx * 6;                                     // shift sideways so opposite arrows on one edge do not collide
  alpha(a, () => {
    if (nc === c && nr === r) { const x = cx(c) + dx * 20 + px * .6, y = cy(r) + dy * 20 + py * .6, a0 = Math.atan2(dy, dx); ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = o.w || 1.5; ctx.beginPath(); ctx.arc(x, y, 6, a0 - Math.PI / 2 - .3, a0 + Math.PI / 2 + .3); ctx.stroke(); ctx.restore(); }
    else arrow(cx(c) + dx * 6 + px, cy(r) + dy * 6 + py, cx(c) + dx * 27 + px, cy(r) + dy * 27 + py, 700 + qk(c, r, m), { c: col, w: o.w || 1.5, head: 6 });
  });
}
function notebook(M, a = 1, o = {}) { for (const [k, e] of M) { const cell = k >> 2; entry(cell % 5, Math.floor(cell / 5), k & 3, e, typeof a === 'function' ? a(k) : a, o); } }
function wetFloor(a = 1) { for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.isRock(c, r)) continue; hachure(OX + c * CS, OY + r * CS, CS, CS, C.blue, 11, 300 + c * 5 + r, .25 * a); } }
function dim(a = .6) { box(0, 0, W, 285, C.bg, a); }
function routeArrows(P, a, col = C.goldInk) { for (let i = 0; i < P.length - 1; i++) { const [c, r] = P[i], [nc, nr] = P[i + 1], m = MOVES.findIndex(([dx, dy]) => dx === nc - c && dy === nr - r); if (m >= 0) alpha(a, () => G.arrowIn(c, r, m, 900 + i, { c: col, w: 2 })); } }
function walkPath(P, t, dur) { const k = clamp(Math.floor(Math.max(0, t) / dur), 0, P.length - 1), u = ease((Math.max(0, t) % dur) / dur * 1.3), a = P[k], b = P[Math.min(P.length - 1, k + 1)];
  trail(P.slice(0, k + 1), .7); pip(lerp(cx(a[0]), cx(b[0]), u), lerp(cy(a[1]), cy(b[1]), u), b[0] - a[0], t); return { k, done: k >= P.length - 1 }; }
// replay recorded transitions tr at dur s per step; returns index of the current step
function replay(tr, t, dur, trailA = .5) { const n = tr.length, k = clamp(Math.floor(Math.max(0, t) / dur), 0, n), s = tr[Math.min(k, n - 1)];
  const u = k >= n ? 1 : ease((Math.max(0, t) % dur) / dur * 1.3);
  trail(tr.slice(0, Math.min(n, k + 1)).map(q => [q.c, q.r]), trailA);
  const bump = s.nc === s.c && s.nr === s.r, [dx, dy] = MOVES[s.m], kk = Math.sin(Math.PI * u) * CS * .25;
  const x = bump && k < n ? cx(s.c) + dx * kk : lerp(cx(s.c), cx(s.nc), u), y = bump && k < n ? cy(s.r) + dy * kk : lerp(cy(s.r), cy(s.nr), u);
  pip(x, y, Math.sign(s.nc - s.c) || 0, t); return k; }
// race chart: x = walk 1..30, y = average steps (0..100). curves = [{ v, col, label }]
function raceChart(curves, n, o = {}) {
  const x0 = 62, x1 = 356, y0 = 384, y1 = 304, X = i => x0 + i * (x1 - x0) / 29, Y = v => y0 - Math.min(v, 100) / 100 * (y0 - y1), a = o.axA ?? 1;
  alpha(a, () => { rline(x0 - 4, y0, x1 + 4, y0, 960, { c: C.line, w: 1 }); rline(x0 - 4, y0, x0 - 4, y1 - 4, 961, { c: C.line, w: 1 });
    txt('100', x0 - 9, Y(100), { s: 12, c: C.soft, al: 'right' }); txt('50', x0 - 9, Y(50), { s: 12, c: C.soft, al: 'right' }); txt('0', x0 - 9, y0, { s: 12, c: C.soft, al: 'right' });
    txt('walk 1', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt('walk 30', x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
    txt(o.title || 'steps per walk, average of 20 runs', x0 - 30, y1 - 13, { s: 14, c: C.soft, al: 'left' }); });
  curves.forEach((cv, j) => { const col = cv.col; for (let i = 0; i < Math.min(n, 30); i++) { if (i) rline(X(i - 1), Y(cv.v[i - 1]), X(i), Y(cv.v[i]), 970 + i + j * 40, { c: col, w: 1.6, once: true, j: .3 }); dot(X(i), Y(cv.v[i]), 2.6, col); }
    if (cv.label) txt(cv.label, x1, y1 - 13 + j * 0, { s: 14, c: col, al: 'right', a: clamp(n / 3) }); });
}
const GREY = '#9d9387', TEAL = '#3f7f7c';
function legend(lab, a) { txt('grey: Q-learning', 352, 318, { s: 14, c: GREY, al: 'right', a }); txt(`teal: ${lab}`, 352, 336, { s: 14, c: TEAL, al: 'right', a }); }
const GA = L.gridWorld({ cs: 38, ox: 8, oy: 50 }), GB = L.gridWorld({ cs: 38, ox: 202, oy: 50 });

// a cat that walks from its old cell to its new one (drawn instead of the maze's own cat)
function movingCat(u, t) { const x = lerp(cx(3), cx(3), u), y = lerp(cy(1), cy(3), ease(u)) + 4; L.cat(x, y, t, 1); }

const STEPS = [
 { h: 'No rulebook', p: ['Pip’s maze again. Stepping **onto** the cheese pays +1, onto the cat −1, and either ends the walk. Other steps pay 0, a bump leaves Pip in place, and γ (gamma) is 0.9.', `Pip has no rulebook: it cannot tell where a move will land. In the Q-learning reel it learned from experience alone; its first four walks took ${QL.W.slice(0, 4).map(w => w.len).join(', ').replace(/, (\d+)$/, ' and $1')} steps.`],
   shot: [4, 10],
   say: [["Welcome back to Pip's maze. Cheese, cat, and a dry floor."], [["But Pip still has no rulebook. It can't tell where a move will land.", [40, 16, 320, 256]]], ["In the Q-learning reel, it learned from experience alone. Its first walks were long, very long."]],
   draw(t) { G.maze({ draw: t / 1.4, pop: { cheese: ease((t - 1.6) * 3), cat: ease((t - 2) * 3) }, labels: clamp(t - 2.6), t }); if (t > 2.4) pip(cx(0), cy(3), 1, t);
     if (t > 4) card(['onto cheese +1, onto cat −1: the walk ends', 'other steps 0 · bump = stay put · γ = 0.9', { s: 'Pip cannot tell where a move will land', c: C.roseInk }], t - 4, { gap: 1.1 }); } },
 { h: 'One walk, one number', p: [`That first walk again, sped up. Pip keeps four numbers per cell, one per move, all starting at 0, so each update nudges 0 toward 0 + 0.9 × 0 and nothing moves.`, `Only the last step, onto the cheese, changes a number: “up” below the cheese, from 0 to ${f2(LAST1.nw)}. Pip saw ${W1.length} steps of the maze; the other ${W1.length - 1} left no trace.`],
   shot: [4, 11],
   get say() { return [["Here's that first walk again, sped up."], [["Watch the numbers in each cell. Nothing moves.", [40, 16, 320, 256]], "Every update nudges zero toward zero."], [["Only the very last step, onto the cheese, changes one number.", [40, 16, 320, 256]], `Pip saw ${W1.length} steps of the maze, and almost all of them left no trace.`]]; },
   draw(t) { const dur = .25, tEnd = W1.length * dur, Q = new Array(80).fill(0); if (t > tEnd + .8) Q[qk(LAST1.c, LAST1.r, LAST1.m)] = LAST1.nw * ease(t - tEnd - .8);
     qmaze(Q, { t, hl: t > tEnd + .8 ? { [key(LAST1.c, LAST1.r)]: [LAST1.m, C.teal, .3] } : null }); const k = replay(W1, t, dur);
     if (k < W1.length) caption(`walk 1, step ${k + 1} of ${W1.length} (sped up)`); else bubble('+1', cx(4), cy(0) - 18, t - tEnd);
     if (t > tEnd + 1) card([`${W1.length} steps seen`, 'numbers changed: 1', { s: `“up” below the cheese: 0 → ${f2(LAST1.nw)}`, c: C.goldInk }], t - tEnd - 1, { gap: 1.1 }); } },
 { h: 'Write it down', p: ['What if Pip also noted what each move did? In this cell, this move landed there and paid this much. Bumps count too: pushed up, stayed put.', `After walk 1 the notebook holds ${NE1} entries, from the ${NC1} cells Pip visited, out of 64 cell and move pairs; some moves repeated. The notebook is a **learned model**: Pip’s copy of part of the rulebook.`],
   shot: [5, 12],
   say: [["What if Pip also wrote down what each move did?"], [["In this cell, this move landed there, and paid this much.", [40, 16, 320, 256]], ["Watch the notes fill the maze. Bumps count too.", [40, 16, 320, 256]]], ["This notebook is a learned model. Pip's own copy of part of the rulebook."]],
   draw(t) { const dur = .25, k = clamp(Math.floor(t / dur), 0, W1.length); G.maze({ t }); notebook(k > 0 ? NB[Math.min(k, W1.length) - 1] : new Map(), 1); replay(W1, t, dur, .25);
     caption(k < W1.length ? `walk 1, step ${k + 1}: entries so far ${NB[Math.max(0, k - 1)].size * (k > 0 ? 1 : 0)}` : `walk 1 done: ${NE1} entries`);
     if (t > W1.length * dur + .6) card(['entry: cell + move  →  where Pip landed, what it paid', { s: 'arrow = landed next door · U-turn = bumped, stayed', c: C.soft }, `after walk 1: ${NE1} of 64 pairs, in ${NC1} cells`], t - W1.length * dur - .6, { gap: 1.1, fs: 16 }); } },
 { h: 'Counting', p: [`Pip’s floor stays dry: one look per move is enough. On the MDP reel’s wet floor, “right” from the start was tried 1000 times: ${WETC[1000][0]} moved right, ${WETC[1000][1]} slid up, ${WETC[1000][2]} stayed.`, `Count divided by tries guesses each chance: ${(WETC[1000][0] / 1000).toFixed(3)}, ${(WETC[1000][1] / 1000).toFixed(3)}, ${(WETC[1000][2] / 1000).toFixed(3)}, near the true 0.8, 0.1, 0.1. Planning would weight landings by these chances; an imagined step would draw one with them.`],
   shot: [3, 12],
   say: [["Pip's floor is dry, so one look per move is enough. But on a wet floor, moves can slip."], [["Remember the wet floor? Try right from the start, again and again.", [40, 16, 320, 256]], ["Count where Pip lands.", [40, 16, 320, 256]]], [["The counts, divided by the tries, land close to the true chances.", [30, 292, 340, 87]]]],
   draw(t) { wetFloor(); G.maze({ t }); pip(cx(0), cy(3), 1, t); const n = t < 4 ? 10 : Math.round(lerp(10, 1000, ease((t - 4) / 4))), cnt = WETC[n];
     alpha(clamp(t - .5), () => { arrow(cx(0) + 22, cy(3) + 4, cx(1) - 18, cy(3) + 4, 801, { c: C.goldInk, w: 2 }); arrow(cx(0), cy(3) - 18, cx(0), cy(2) + 14, 802, { c: TEAL, w: 1.6 });
       txt(String(cnt[0]), cx(1), cy(3) + 20, { s: 17, c: C.goldInk }); txt(String(cnt[1]), cx(0) + 22, cy(2) + 2, { s: 17, c: TEAL }); txt(`${cnt[2]} stayed`, cx(0), cy(3) + 25, { s: 14, c: '#4f74a0' }); });
     caption(`tries of “right” at the start: ${n}`);
     if (t > 8.5) card([`chance guess = count ÷ tries`, `right ${(cnt[0] / n).toFixed(3)} · slid up ${(cnt[1] / n).toFixed(3)} · stayed ${(cnt[2] / n).toFixed(3)}`, { s: 'true chances: 0.8, 0.1, 0.1', c: C.soft }], t - 8.5, { gap: 1.1 }); } },
 { h: 'Planning in the notebook', p: ['Plan with it, as in the value iteration reel: value of a cell = the best, over moves, of reward + 0.9 × value where that move lands. Start from zeros and sweep.', `One difference: Pip uses only moves in its notebook, since it has no idea where untried ones lead. Here ${VS} sweeps change something; sweep ${VSW} changes nothing, so the planning stops.`],
   shot: [4, 11],
   get say() { return [["Now Pip can plan with its notebook, just like value iteration."], [["Each cell's value is the best noted move, its reward plus zero point nine times the value where it lands.", [40, 16, 320, 256]], "Watch the values spread out, sweep by sweep."], [["Pip only uses moves it has noted. Untried moves are a mystery.", [30, 292, 340, 87]]]]; },
   draw(t) { const D = 1.1, k = clamp(Math.floor((t - 1) / D) + 1, 0, VSW), V = VH[k]; G.maze({ V, t, nums: (c, r) => ([0, 1, 2, 3].some(m => M1.has(qk(c, r, m))) ? 1 : 0) }); notebook(M1, .35);
     pip(cx(0), cy(3), 1, t); caption(k ? `after sweep ${k}${k === VSW ? ': no change, stop' : ''}` : 'all values 0');
     if (t > 2) card(['value = best over NOTED moves of', '          reward + 0.9 × value where it lands', { s: 'cells with no entries stay 0', c: C.soft }], t - 2, { gap: 1.1 }); } },
 { h: 'A route after one walk', p: [`Taking the best noted move in each cell, Pip walks from the start to the cheese in ${ROUTE.length - 1} steps, the shortest possible. Its real experience so far: one walk of ${W1.length} steps.`, 'After that same walk, plain Q-learning knew one move, the one below the cheese. Writing things down got far more out of the same experience: better **sample efficiency**, learning per real step.'],
   shot: [4, 11],
   get say() { return [[["Now follow the best noted move from each cell.", [40, 16, 320, 256]], `Pip reaches the cheese in ${ROUTE.length - 1} steps, the shortest way.`, "After just one real walk!"], [["Plain Q-learning, after the same walk, knew one single move.", [30, 292, 340, 87]], "Learning more from each real step is called sample efficiency."]]; },
   draw(t) { G.maze({ V: V1, t, nums: (c, r) => ([0, 1, 2, 3].some(m => M1.has(qk(c, r, m))) ? .8 : 0) }); routeArrows(ROUTE, .6); const w = walkPath(ROUTE, t - 1, .6);
     if (w.done) bubble('+1', cx(4), cy(0) - 18, t - 1 - (ROUTE.length - 1) * .6);
     if (t > 5.5) card([`model + planning: a ${ROUTE.length - 1}-step route`, 'Q-learning alone: 1 number, no route', { s: `both from the same ${W1.length} real steps`, c: C.soft }], t - 5.5, { gap: 1.2 }); } },
 { h: 'Planning costs thinking', p: [`Planning is not free. That plan took ${VSW} sweeps over ${NE1} entries: ${VSW * NE1} updates. Each new walk adds entries, and a fresh plan would sweep them all again.`, 'Here that is cheap. In a world with millions of states, sweeping everything after every step is not. Richard Sutton’s **Dyna** spreads the planning thin instead: a little after each real step.'],
   shot: [4, 10],
   say: [["But planning isn't free. Each plan sweeps the whole notebook, again and again."], [["Here that's cheap.", [30, 292, 340, 87]], ["In a world with millions of states, it's not.", [30, 292, 340, 87]]], ["So Richard Sutton's Dyna spreads the planning thin. A little after each real step."]],
   draw(t) { G.maze({ V: V1, t }); notebook(M1, .7); pip(cx(0), cy(3), 1, t);
     card([`one plan: ${VSW} sweeps × ${NE1} entries = ${VSW * NE1} updates`, 'millions of states: too much after every step', { s: 'Dyna: a few imagined updates per real step', c: TEAL }], t - .8, { gap: 1.3 }); } },
 { h: 'Dyna-Q, part one: a real step', p: ['Dyna-Q runs Q-learning as usual. After each real move, the target is the reward plus 0.9 times the biggest number in the landing cell, and the move’s number goes a fifth of the way toward it: α (alpha) = 0.2.', 'Pip still takes the biggest number, with one move in five random: ε (epsilon) = 0.2. New: it also notes the move in its notebook.'],
   shot: [4, 10],
   say: [["Here's how Dyna-Q works. Part one is plain Q-learning."], [["Pip makes a real move.", [40, 16, 320, 256]], ["Its number moves a fifth of the way toward the reward, plus zero point nine times the best number where it landed.", [30, 292, 340, 87]]], ["And one new thing. Pip writes the move into its notebook."]],
   draw(t) { const Q = QDS0.slice(); if (t > 3) Q[qk(DS.c, DS.r, DS.m)] = lerp(DS.old, DS.nw, ease(t - 3)); qmaze(Q, { t, hl: { [key(DS.c, DS.r)]: [DS.m, C.teal, .35 * clamp(t - 2)] } });
     replay([DS], t - .5, 1, 0); if (t > 4) entry(DS.c, DS.r, DS.m, [DS.nc, DS.nr, DS.rew], clamp(t - 4), { w: 2.4 });
     caption(`Dyna-Q, walk 3, real step ${DSI + 1}`);
     card([`target = ${DS.rew} + 0.9 × ${f3((DS.tg - DS.rew) / GAM)} = ${f3(DS.tg)}`, `new = ${f3(DS.old)} + 0.2 × (${f3(DS.tg)} − ${f3(DS.old)}) = ${f3(DS.nw)}`, { s: 'and: write “this move landed there, paid 0”', c: TEAL }], t - 2, { gap: 1.3 }); } },
 { h: 'Part two: imagined steps', p: [`Before its next real move, Pip practises in its head ${NIM} times: pick a random entry, read its landing and reward, and apply the same update as if it were real.`, 'Imagination takes no real step and replays only tried moves. Unlike a replay memory of every move made, the notebook keeps one entry per cell and move: a rulebook Pip could also plan with.'],
   shot: [3, 10],
   get say() { return [[`Part two. Before its next real move, Pip practises in its head, ${NIM} times.`], [["It picks a random note, reads where the move landed and what it paid, and updates, as if it were real.", [30, 292, 340, 87]]], ["Imagination takes no real step. And it only replays moves Pip has really tried."]]; },
   draw(t) { const D = .95, k = clamp(Math.floor((t - .8) / D), -1, NIM - 1), Q = QDS0.slice(); Q[qk(DS.c, DS.r, DS.m)] = DS.nw;
     for (let i = 0; i <= k; i++) { const v = DS.im[i]; Q[qk(v.c, v.r, v.m)] = v.nw; }
     const v = k >= 0 ? DS.im[k] : null; qmaze(Q, { t, hl: v ? { [key(v.c, v.r)]: [v.m, C.teal, .35] } : null }); notebook(MDS, .3);
     pip(cx(DS.nc), cy(DS.nr), 1, t);
     if (v) { alpha(.5, () => L.mouse(cx(v.c), cy(v.r) + 3, 1, .8, t)); entry(v.c, v.r, v.m, [v.nc, v.nr, v.rew], 1, { w: 2.4, c: C.ink }); rrect(OX + v.c * CS + 2, OY + v.r * CS + 2, CS - 4, CS - 4, 1300 + k, { c: '#6d645b', w: 1, j: 2 }); }
     const ch = DS.im.slice(0, k + 1).filter(q => Math.abs(q.nw - q.old) > 1e-9).length;
     caption(v ? `imagined step ${k + 1} of ${NIM}: ${NAMES[v.m]} · ${Math.abs(v.nw - v.old) > 1e-9 ? `${f2(v.old)} → ${f2(v.nw)}` : 'no change'}` : 'Pip stands still and imagines');
     if (t > 1) card(['pick a random notebook entry', 'landing + reward from the notebook', `same update as a real step  (changed so far: ${ch})`], t - 1, { gap: 1.1 }); } },
 { h: 'The first walks', p: [`Walk 1 is the same ${W1.length} steps for both methods: every number is 0, so imagining changes nothing until the cheese is found. Walk 2 also takes ${DY.W[1].len} steps for both.`, `But by its end Q-learning has ${NZQ2} nonzero numbers and Dyna-Q has ${NZD2}. On walk 3, Q-learning wanders ${QL.W[2].len} steps; Dyna-Q walks ${DY.W[2].len}.`],
   shot: [4, 10],
   get say() { return [["How do the first walks compare?"], [["Walk one is identical for both. Every number is zero, so imagining changes nothing.", [8, 30, 384, 180]]], [["But by the end of walk two, Dyna-Q has far more nonzero numbers.", [8, 30, 384, 180]], `On walk three, Q-learning wanders ${QL.W[2].len} steps. Dyna-Q walks ${DY.W[2].len}.`]]; },
   draw(t) { const a = clamp(t - .3), b = clamp(t - 1.5);
     alpha(a, () => { qmaze(QL.Qs[1], { Gx: GA, t }); txt('Q-learning', GA.ox + 95, 30, { s: 16, c: GREY }); });
     alpha(b, () => { qmaze(DY.Qs[1], { Gx: GB, t }); txt(`Dyna-Q, ${NIM} imagined per step`, GB.ox + 95, 30, { s: 16, c: TEAL }); });
     txt('after walk 2', W / 2, 220, { s: 15, c: C.soft, a: b });
     if (t > 3) card([`nonzero numbers:  Q-learning ${NZQ2},  Dyna-Q ${NZD2}`, `walk 3:  Q-learning ${QL.W[2].len} steps,  Dyna-Q ${DY.W[2].len}`, { s: `walks 1 and 2: same lengths (${W1.length}, ${DY.W[1].len})`, c: C.soft }], t - 3, { y: 250, gap: 1.3 }); } },
 { h: 'The race', p: ['One run can be lucky, so race on 20 runs with different dice, seeds 1 to 20; both methods get the same 20.', 'In the chart, across is the walk number, 1 to 30. Up is the number of steps that walk took, averaged over the 20 runs. Grey is Q-learning, teal is Dyna-Q with 10 imagined steps per real one.'],
   shot: [3, 12],
   say: [["One run could be lucky. So let's race twenty runs, with the same dice for both."], [["Across is the walk number. Up is how many steps it took.", [40, 290, 330, 110]], ["Grey is Q-learning. Teal is Dyna-Q.", [40, 290, 330, 110]], ["Watch which line falls faster.", [40, 290, 330, 110]]]],
   draw(t) { const A = race(0), B = race(NIM), n = Math.floor(clamp((t - 2.5) / 5) * 30);
     G.maze({ t }); pip(cx(0), cy(3), 1, t);
     raceChart([{ v: A.cur, col: GREY }, { v: B.cur, col: TEAL }], n, { axA: clamp(t) });
     legend('Dyna-Q', clamp(t - 1.5)); } },
 { h: 'Real steps saved', p: (() => { const A = race(0), B = race(NIM); return [`Over the first 10 walks, Q-learning used ${Math.round(A.t10)} real steps on average and Dyna-Q ${Math.round(B.t10)}, about half. In walks 21 to 30 both average about ${A.late.toFixed(0)} steps a walk, not 7, because one move in five is still random.`, `The price is thinking: Dyna-Q made ${NIM} imagined updates for every real one, about ${hund(B.ups)} updates per run against about ${hund(A.ups)}.`]; })(),
   shot: [3, 10],
   say: [[["Over the first ten walks, Dyna-Q used about half the real steps.", [30, 150, 340, 87]]], [["Later, both settle at the same level, above the shortest route, because one move in five is still random.", [40, 290, 330, 110]]], ["The price is thinking. Ten imagined updates for every real one."]],
   draw(t) { const A = race(0), B = race(NIM); G.maze({ t }); dim(.7); raceChart([{ v: A.cur, col: GREY }, { v: B.cur, col: TEAL }], 30);
     legend('Dyna-Q', 1);
     card([`first 10 walks: ${Math.round(A.t10)} vs ${Math.round(B.t10)} real steps`, `walks 21 to 30: ${A.late.toFixed(1)} vs ${B.late.toFixed(1)} steps a walk`, { s: `updates per run: about ${hund(A.ups)} vs ${hund(B.ups)}`, c: C.soft }], t - 1, { y: 150, gap: 1.2 }); } },
 { h: 'The catch: the world changes', p: [`After 50 walks with Dyna-Q, Pip’s arrows take the bottom route and the start’s best number is ${f2(Q50S)}. Then the cat wakes up and moves to the bottom row, right onto that route.`, `The notebook does not know. It still says the bottom row is safe, and the numbers built from it agree. Pip follows its confident plan and meets the cat after ${CW1.length} steps.`],
   shot: [4, 11],
   get say() { return [["Now the catch. What if the world changes?"], [["After fifty walks, Pip's arrows take the bottom route.", [40, 16, 320, 256]], ["Then the cat wakes up, and moves right onto it.", [40, 16, 320, 256]]], [["The notebook doesn't know. Pip follows its confident plan, and meets the cat.", [30, 318, 340, 64]]]]; },
   draw(t) { const u = clamp((t - 1.5) / 1.5); qmaze(DQ50, { t, pop: { cat: 0 } }); routeArrows(R50, .7 * (1 - clamp(t - 4))); movingCat(u, t);
     if (t < 4) pip(cx(0), cy(3), 1, t); else { const k = replay(CW1, t - 4, .7); if (k >= CW1.length) bubble('−1', cx(3), cy(3) - 18, t - 4 - CW1.length * .7); }
     txt(`start: best number ${f2(Q50S)}`, W / 2, 304, { s: 17, c: C.goldInk, a: clamp(t - .5) });
     if (t > 4 + CW1.length * .7 + .5) card([`the notebook: bottom row safe (it was)`, `reality: cat after ${CW1.length} steps, −1`, { s: 'confident plan, wrong model', c: C.roseInk }], t - 4.5 - CW1.length * .7, { y: 318, gap: 1.1 }); } },
 { h: 'Real steps fix it, slowly', p: [`Its entry is overwritten: “right here lands on the cat, −1”. On walk 2, Pip bumps up and down in that cell ${NBUMP} times: their numbers, ${f2(BQ0[0])} and ${f2(BQ0[2])}, date from the old cheese route.`, `A bump’s target is 0.9 times the cell’s best number, so they shrink slowly. Once below “left”, a random move hits the cat again. After walk ${FIXW}, the arrows go over the top.`],
   shot: [3, 12],
   get say() { return [["The note for that move gets overwritten. Right here, lands on the cat."], [["On walk two, Pip bumps up and down in that cell, again and again.", [40, 16, 320, 256]], ["Those numbers come from the old route, and shrink only slowly.", [40, 16, 320, 256]]], [`Only after walk ${FIXW} do the arrows go over the top.`]]; },
   draw(t) { const T2 = CH.W[1].len * .45 + 1, k = replay(CH.W[1].tr, Math.min(t, T2) - .3, .45), i = Math.min(k, CH.W[1].len), Q = Qbefore(CH, 1, i);
     qmaze(Q, { t, Gx: G1, nums: (c, r) => (c === 2 && r === 3 ? 1 : 0) });
     caption(k < CH.W[1].len ? `walk 2, step ${k + 1}${CH.W[1].tr[k].rnd ? ' (random move)' : ''}` : 'walk 2: the cat again');
     if (t > T2 + .5) card([`up and down (bumps): ${f2(BQ0[0])}, ${f2(BQ0[2])} → ${f2(BQ1[0])}, ${f2(BQ1[2])}`, `left: ${f2(BQ1[3])} · right (cat): ${f2(BQ1[1])}`, { s: 'bump target = 0.9 × this cell’s best number', c: C.soft }], t - T2 - .5, { gap: 1.2, fs: 16 }); } },
 { h: 'A belief nobody checks', p: [`${STALE} notebook entries, the rose −1 tags, still say stepping into the old cat cell costs −1. The plan keeps away, so only a random move could check, and imagination keeps replaying those entries, reinforcing the mistake.`, 'A known fix, Dyna-Q+ in Sutton and Barto’s textbook, adds a small bonus to moves not tried for a long time, so Pip goes back to check.'],
   shot: [3, 9],
   say: [[["Look at the rose tags. These notes still say the old cat cell costs minus one.", [40, 16, 320, 256]]], ["The plan stays away, so nothing checks them. And imagination keeps replaying them, making the mistake stronger."], ["One fix, Dyna-Q plus, adds a small bonus for moves not tried in a while. So Pip goes back to check."]],
   draw(t) { const Q = CH.Qs[FIXW - 1]; qmaze(Q, { t, Gx: G1 }); routeArrows(RFIX, clamp(t - .3)); pip(cx(0), cy(3), 1, t);
     if (t > 1.5) { const a = clamp(t - 1.5); STALEK.forEach(k => { const cell = k >> 2, c = cell % 5, r = Math.floor(cell / 5), x = (cx(c) + cx(3)) / 2, y = (cy(r) + cy(1)) / 2 + 4; box(x - 13, y - 9, 26, 18, C.card, .95 * a); rrect(x - 13, y - 9, 26, 18, 7800 + k, { c: C.rose, w: 1.4 }); txt('−1', x, y, { s: 14, c: C.roseInk, a }); }); txt('?', cx(3), cy(1) + 4, { s: 22, c: C.roseInk, a: clamp(t - 2) }); }
     if (t > 3) card([{ s: 'gold arrows: the new plan, over the top', c: C.goldInk }, { s: `rose −1 tags: ${STALE} stale notebook entries`, c: C.roseInk }, { s: 'fix: bonus for long-untried moves (Dyna-Q+)', c: C.soft }], t - 3, { gap: 1.2 }); } },
 { h: 'Your turn', ctl: { label: 'imagined', values: NV, def: 3, fmt: v => String(v) },
   p: ['Drag the number of **imagined steps** per real step. 0 is plain Q-learning. The chart reruns the same 20 runs and compares with Q-learning in grey.', 'Watch the first walks shrink as Pip imagines more, then stop shrinking: once the notebook is mostly learned, extra imagining adds little, and the random moves remain.'],
   shot: [3],
   say: [["Your turn. Drag how many imagined steps Pip takes per real step. Zero is plain Q-learning."], [["Watch the teal line against grey.", [40, 290, 330, 110]], ["The first walks shrink as Pip imagines more, then stop shrinking.", [40, 290, 330, 110]]], ["Once the notebook is mostly learned, extra imagining adds little."]],
   draw(t, st) { const n = NV[st.ci ?? 3], A = race(0), B = race(n); G.maze({ t }); dim(.7);
     raceChart([{ v: A.cur, col: GREY }, { v: B.cur, col: TEAL }], 30);
     legend(`${n} imagined`, 1);
     card([`${n} imagined per real step`, `real steps, first 10 walks: ${Math.round(B.t10)}`, { s: `Q-learning: ${Math.round(A.t10)} · updates per run: about ${hund(B.ups)}`, c: C.soft }], 10, { y: 150, gap: 0, fs: 17 }); } },
 { h: 'Where it came from', p: ['Richard Sutton introduced **Dyna** in 1990: learn a model, and mix real updates with imagined ones. Learned models matter most where real experience is slow or costly, like robots. PILCO, in 2011, learned to swing up a real cart-pole within a handful of trials.', 'The next reels search ahead with a model (tree search, AlphaZero, MuZero) or learn a model from pictures (World Models, Dreamer).'],
   shot: [4, 10],
   say: [["Richard Sutton introduced Dyna in nineteen ninety."], [["Learned models matter most where real experience is slow or costly, like robots.", [30, 292, 340, 87]], ["In twenty eleven, PILCO learned to swing up a real cart-pole in a handful of trials.", [30, 292, 340, 87]]], ["Next, searching ahead with a model, and learning models from pictures."]],
   draw(t) { qmaze(DY.Q, { t }); notebook(DY.M, .25); pip(cx(0), cy(3), 1, t); dim(.3);
     card(['1990  Sutton: Dyna', '2011  PILCO: a real robot in a few trials', { s: 'next: tree search, AlphaZero, MuZero, Dreamer', c: C.soft }], t - .5, { gap: 1.2 }); } },
 { h: 'Recap', p: ['A **model** says, for each state and action, where you land and what it pays. If nobody hands you one, learn it by writing down, and counting, what really happened.', 'Then plan with it, or use it for imagined practice between real steps, as Dyna does. In Pip’s maze that halved the real steps. But a wrong model gives confident wrong plans.'],
   shot: [3, 9],
   say: [["So, a model says where each move lands, and what it pays. If no one hands you one, write down what really happened."], [["Then plan with it, or practise in your head between real steps.", [30, 292, 340, 110]], "That halved Pip's real steps. But a wrong model gives confident, wrong plans."]],
   draw(t) { qmaze(DY.Q, { t }); routeArrows(R50, .8); pip(cx(0), cy(3), 1, t);
     card(['model: state + action → landing, reward', 'learn it from experience: write down, count', 'plan with it, or imagine practice (Dyna)', { s: 'wrong model → confident wrong plans', c: C.roseInk }], t - .3, { gap: 1 }); } },
];

RLStory.register('modelbased', { title: 'Model-Based RL', track: 'Model-based', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { QL, DY, W1, LAST1, NE1, NC1, SAME1, VH, VS, VSW, ROUTE, WETC, NZQ2, NZD2, DSI, DS, race, Q50S, R50, CH, CW1, CW1RND, FIXW, RFIX, STALE } });
})();
