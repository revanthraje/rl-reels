// Hierarchical RL as a story: a four-room maze, Pip plans with big steps (options: "go to door B") and small steps (moves).
// Options are hand-made by us from the map of doors; learning is SMDP Q-learning vs flat Q-learning on the same 20 seeds.
// Every number comes from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha, hachure } = L;
let ctx;
// ───── the four-room maze: 11 × 11 cells, walls along column 5 and row 5 except four doors
const N = 11, MV = [[0, -1], [1, 0], [0, 1], [-1, 0]], GAM = 0.9, EPS = 0.2, ALPHA = 0.2;
const DOORS = [[5, 2], [2, 5], [8, 5], [5, 8]], DN = ['A', 'B', 'C', 'D'];       // A: rooms 1–2, B: 1–3, C: 2–4, D: 3–4
const K = (x, y) => y * N + x;
const doorAt = (x, y) => DOORS.findIndex(d => d[0] === x && d[1] === y);
const wall = (x, y) => x < 0 || y < 0 || x >= N || y >= N || ((x === 5 || y === 5) && doorAt(x, y) < 0);
const room = (x, y) => (doorAt(x, y) >= 0 || wall(x, y)) ? -1 : (x < 5 ? 0 : 1) + (y < 5 ? 0 : 2);   // 0..3 = rooms 1..4
const RD = [[0, 1], [0, 2], [1, 3], [2, 3]];                                       // the two doors of each room
function step(x, y, m) { const nx = x + MV[m][0], ny = y + MV[m][1]; return wall(nx, ny) ? [x, y] : [nx, ny]; }
// ───── options: for each room and each of its doors, "go to that door": shortest-path arrows inside the room (+ its other door)
const inRD = (x, y, r) => room(x, y) === r || RD[r].includes(doorAt(x, y));
const OPTS = [];
for (let r = 0; r < 4; r++) for (const d of RD[r]) {
  const [tx, ty] = DOORS[d], dist = new Map([[K(tx, ty), 0]]), q = [[tx, ty]];
  while (q.length) { const [x, y] = q.shift(); for (let m = 0; m < 4; m++) { const nx = x + MV[m][0], ny = y + MV[m][1]; if (wall(nx, ny) || !inRD(nx, ny, r) || dist.has(K(nx, ny))) continue; dist.set(K(nx, ny), dist.get(K(x, y)) + 1); q.push([nx, ny]); } }
  const pol = new Int8Array(N * N).fill(-1);
  for (const [k, dd] of dist) { if (!dd) continue; const x = k % N, y = Math.floor(k / N); for (const m of [1, 2, 0, 3]) { const [nx, ny] = step(x, y, m); if (dist.get(K(nx, ny)) === dd - 1) { pol[k] = m; break; } } }
  OPTS.push({ r, d, pol, name: 'go to door ' + DN[d] });
}
const NA = 4 + OPTS.length;
const AV = [false, true].map(u => { const a = []; for (let k = 0; k < N * N; k++) { const l = [0, 1, 2, 3]; if (u) OPTS.forEach((o, i) => { if (o.pol[k] >= 0) l.push(4 + i); }); a.push(l); } return a; });
const START = [0, 0];
// ───── one learner. useOpt: menu = 4 moves (+ options available here). SMDP Q-learning: after a choice that took k steps,
// target = rewards (discounted inside the choice) + γ^k × best number where it ended. Walk ends on the cheese (+1) or at cap steps.
function run(useOpt, seed, nW, goal, cap = 20000, keep = false) {
  const R = rng(seed), Q = new Float64Array(N * N * NA), A = AV[useOpt ? 1 : 0], G = K(goal[0], goal[1]), out = [];
  const best = k => { if (k === G) return 0; let b = -1e9; for (const a of A[k]) if (Q[k * NA + a] > b) b = Q[k * NA + a]; return b; };
  for (let e = 0; e < nW; e++) {
    let x = START[0], y = START[1], steps = 0, dec = 0; const tr = keep ? [] : null, D = keep ? [] : null; let firstNews = -1;
    while (steps < cap) {
      const k = K(x, y), Ak = A[k]; let a;
      if (R() < EPS) a = Ak[Math.floor(R() * Ak.length)]; else { let b = -1e9, ms = []; for (const aa of Ak) { const v = Q[k * NA + aa]; if (v > b + 1e-12) { b = v; ms = [aa]; } else if (v > b - 1e-12) ms.push(aa); } a = ms[Math.floor(R() * ms.length)]; }
      dec++; let cx = x, cy = y, Rw = 0, n = 0, done = false; const t0 = steps;
      if (a < 4) { [cx, cy] = step(x, y, a); n = 1; steps++; if (keep) tr.push([cx, cy]); if (K(cx, cy) === G) { Rw = 1; done = true; } }
      else { const o = OPTS[a - 4]; while (o.pol[K(cx, cy)] >= 0 && steps < cap) { [cx, cy] = step(cx, cy, o.pol[K(cx, cy)]); steps++; n++; if (keep) tr.push([cx, cy]); if (K(cx, cy) === G) { Rw = GAM ** (n - 1); done = true; break; } } }
      const nb = done ? 0 : best(K(cx, cy)), old = Q[k * NA + a], tg = Rw + GAM ** n * nb, nw = old + ALPHA * (tg - old);
      Q[k * NA + a] = nw; if (keep) D.push({ x, y, a, n, Rw, nb, old, tg, nw, t0, done });
      x = cx; y = cy; if (done) break;
    }
    out.push({ steps, dec, reached: steps < cap || (x === goal[0] && y === goal[1]), tr, D, startKnows: best(K(START[0], START[1])) > 0 });
  }
  return { out, Q };
}
const greedyWalk = (Q, useOpt, goal) => { let x = START[0], y = START[1]; const D = [], P = [[x, y]], A = AV[useOpt ? 1 : 0], G = K(goal[0], goal[1]);
  for (let i = 0; i < 40 && K(x, y) !== G; i++) { const k = K(x, y); let b = -1e9, ba = 0; for (const a of A[k]) if (Q[k * NA + a] > b + 1e-12) { b = Q[k * NA + a]; ba = a; }
    D.push({ x, y, a: ba }); if (ba < 4) { [x, y] = step(x, y, ba); P.push([x, y]); } else { const o = OPTS[ba - 4]; while (o.pol[K(x, y)] >= 0 && K(x, y) !== G) { [x, y] = step(x, y, o.pol[K(x, y)]); P.push([x, y]); } } }
  return { D, P }; };
// ───── the main experiment: cheese in door D, 20 runs (seeds 101..120), 60 walks each
const GOAL = DOORS[3], NRUN = 20, NW = 60, SEEDS = Array.from({ length: NRUN }, (_, i) => 101 + i);
const EXP = (() => { const r = {}; for (const u of [0, 1]) { const steps = new Array(NW).fill(0), dec = new Array(NW).fill(0); let knows = 0;
  SEEDS.forEach(s => { const o = run(!!u, s, NW, GOAL).out; o.forEach((w, i) => { steps[i] += w.steps / NRUN; dec[i] += w.dec / NRUN; }); knows += (o.findIndex(w => w.startKnows) + 1) / NRUN; });
  r[u ? 'opt' : 'flat'] = { steps, dec, knows }; } return r; })();
const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
const LAST10 = { flat: mean(EXP.flat.steps.slice(-10)), opt: mean(EXP.opt.steps.slice(-10)), fdec: mean(EXP.flat.dec.slice(-10)), odec: mean(EXP.opt.dec.slice(-10)) };
const TOT = { flat: EXP.flat.steps.reduce((s, v) => s + v, 0), opt: EXP.opt.steps.reduce((s, v) => s + v, 0) };
// story run: seed 101 with traces
const SF = run(false, 101, NW, GOAL, 20000, true), SO = run(true, 101, NW, GOAL, 20000, true);
const F1 = SF.out[0], O1 = SO.out[0], OLAST = O1.D[O1.D.length - 1];
const FIRSTBOOT = (() => { for (let e = 1; e < NW; e++) { const d = SO.out[e].D.find(d => d.a >= 4 && !d.done && d.nb > 0); if (d) return Object.assign({ e }, d); } return null; })();
const GW_O = greedyWalk(SO.Q, true, GOAL), GW_F = greedyWalk(SF.Q, false, GOAL);
const SHORT = Math.abs(GOAL[0] - START[0]) + Math.abs(GOAL[1] - START[1]);
// ───── the pitfall: cheese in the far corner of room 4; walk 1 only, same 20 seeds, cap 5000 steps
const CORNER = [10, 10], CAP = 5000;
const PIT = (() => { const f = SEEDS.map(s => run(false, s, 1, CORNER, CAP).out[0]), o = SEEDS.map(s => run(true, s, 1, CORNER, CAP).out[0]);
  return { fMean: mean(f.map(w => w.steps)), fCap: f.filter(w => !w.reached).length, oCap: o.filter(w => !w.reached).length, oMean: mean(o.map(w => w.steps)) }; })();
let P60m = null; const PIT60 = () => P60m || (P60m = [0, 1].map(u => { const st = new Array(60).fill(0); SEEDS.forEach(s => run(!!u, s, 60, CORNER, CAP).out.forEach((w, j) => { st[j] += w.steps / NRUN; })); return st; }));   // does options-Pip recover? 60 walks, same seeds and cap (lazy)
const PITW = run(true, 101, 1, CORNER, 400, true).out[0];          // the first 400 steps of one such walk, for the picture
const DOORVIS = PITW.tr.filter(([x, y]) => doorAt(x, y) >= 0).length;
// ───── your turn: where is the cheese? (computed when chosen; 10 runs, 30 walks, cap 3000)
const SPOTS = [[5, 8], [8, 5], [8, 8], [10, 10]], SPN = ['door D', 'door C', 'middle of room 4', 'far corner'];
const YT = {}; function yt(i) { if (YT[i]) return YT[i]; const g = SPOTS[i], r = {};
  for (const u of [0, 1]) { const steps = new Array(30).fill(0); let capped = 0; for (let s = 1; s <= 10; s++) run(!!u, 300 + s, 30, g, 3000).out.forEach((w, j) => { steps[j] += w.steps / 10; if (!w.reached) capped++; }); r[u ? 'opt' : 'flat'] = { steps, capped }; }
  return (YT[i] = r); }
// ───── formatting
const f0 = v => Math.round(v).toLocaleString('en-US'), f1 = v => v.toFixed(1), f2 = v => v.toFixed(2), f3 = v => v.toFixed(3);

// ───── drawing
let CS = 24, OX = 68, OY = 8;
const cx = x => OX + x * CS + CS / 2, cy = y => OY + y * CS + CS / 2;
function withGeom(cs, ox, oy, fn) { const o = [CS, OX, OY]; CS = cs; OX = ox; OY = oy; try { fn(); } finally { [CS, OX, OY] = o; } }
function maze(o = {}) {
  const p = o.draw ?? 1;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const X = OX + x * CS, Y = OY + y * CS;
    if (wall(x, y)) { if (p > .3) { box(X, Y, CS, CS, '#d8cfc3', clamp((p - .3) * 2)); hachure(X, Y, CS, CS, '#a99e90', 6, 200 + x * 13 + y, .6 * clamp((p - .3) * 2)); } continue; }
    if (o.shade) { const v = o.shade(x, y); if (v) box(X + 1, Y + 1, CS - 2, CS - 2, v[0], v[1]); } }
  ctx.save(); ctx.globalAlpha = .55 * clamp(p * 2); ctx.strokeStyle = C.line; ctx.lineWidth = .6;
  for (let k = 1; k < N; k++) { ctx.beginPath(); ctx.moveTo(OX + k * CS, OY); ctx.lineTo(OX + k * CS, OY + N * CS); ctx.moveTo(OX, OY + k * CS); ctx.lineTo(OX + N * CS, OY + k * CS); ctx.stroke(); }
  ctx.restore(); rrect(OX, OY, N * CS, N * CS, 90, { w: 2, p: clamp(p * 1.5) });
  const g = o.goal ?? GOAL;
  if (o.doors !== 0) DOORS.forEach(([x, y], i) => { const a = (o.doors ?? 1) * clamp(p * 2 - 1); box(OX + x * CS + 2, OY + y * CS + 2, CS - 4, CS - 4, '#f6e7c8', a);
    if (o.cheese !== 0 && g[0] === x && g[1] === y) { const lx = x === 5 ? cx(x) + CS : cx(x), ly = x === 5 ? cy(y) : cy(y) - CS; box(lx - 8, ly - 9, 16, 18, C.card, a); txt(DN[i], lx, ly + 1, { s: CS < 20 ? 11 : 14, c: C.goldInk, a }); }
    else txt(DN[i], cx(x), cy(y) + 1, { s: CS < 20 ? 11 : 14, c: C.goldInk, a }); });
  if (o.rooms) [[2, 2], [8, 2], [2, 8], [8, 8]].forEach(([x, y], i) => txt('room ' + (i + 1), cx(x), cy(y), { s: 15, c: C.soft, a: o.rooms }));
  if (o.cheese !== 0) L.cheese(cx(g[0]), cy(g[1]) + 2, CS / 48 * (o.cheese ?? 1));
}
function pip(x, y, face, t, s = .52) { L.mouse(x, y + 2, face || 1, s, t); }
function trail(P, n, col = C.teal, a = .5) { for (let i = 0; i < Math.min(n, P.length); i++) { const [x, y] = P[i]; dot(cx(x) + ((i * 7) % 11) - 5, cy(y) + ((i * 5) % 11) - 5, 1.6, col, a); } }
function optArrows(oi, a = 1, col = C.goldInk) { const o = OPTS[oi]; for (let k = 0; k < N * N; k++) { const m = o.pol[k]; if (m < 0) continue; const x = k % N, y = Math.floor(k / N), [dx, dy] = MV[m];
  alpha(a, () => arrow(cx(x) - dx * 5, cy(y) - dy * 5, cx(x) + dx * 7, cy(y) + dy * 7, 700 + k, { c: col, w: 1.3, head: 4 })); } }
const OI = (r, d) => OPTS.findIndex(o => o.r === r && o.d === d);
function pathLine(P, n, col, w = 2.4, seed = 800) { for (let i = 1; i < Math.min(n, P.length); i++) rline(cx(P[i - 1][0]), cy(P[i - 1][1]), cx(P[i][0]), cy(P[i][1]), seed + i, { c: col, w, j: .5, once: true }); }
// steps-per-walk chart; o = { lines: [{ v, col, n, label }], max, nW }
function chart(o) {
  const x0 = 62, x1 = 350, y0 = o.y0 ?? 386, y1 = o.y1 ?? 308, nW = o.nW ?? NW, max = o.max, X = i => x0 + i / (nW - 1) * (x1 - x0), Y = v => y0 - Math.min(v, max) / max * (y0 - y1);
  rline(x0 - 4, y0, x1 + 4, y0, 960, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1 - 4, 961, { c: C.line, w: 1, once: true });
  txt('0', x0 - 12, y0, { s: 12, c: C.soft }); txt(f0(max), x0 - 8, y1, { s: 12, c: C.soft, al: 'right' }); txt('walk 1', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt('walk ' + nW, x1, y0 + 11, { s: 12, c: C.soft, al: 'right' });
  if (o.title) txt(o.title, x0 - 10, y1 - 14, { s: 13, c: C.soft, al: 'left' });
  if (o.short) { ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.grey; ctx.beginPath(); ctx.moveTo(x0, Y(o.short)); ctx.lineTo(x1, Y(o.short)); ctx.stroke(); ctx.restore(); }
  if (o.legend) o.legend.forEach(([s, col], i) => { box(x1 - 120, y1 + 4 + i * 18, 14, 4, col); txt(s, x1 - 100, y1 + 6 + i * 18, { s: 13, c: col, al: 'left' }); });
  if (o.short && o.shortLabel) txt(o.shortLabel, x1, Y(o.short) - 8, { s: 12, c: C.soft, al: 'right' });
  (o.lines || []).forEach(s => { const n = Math.min(nW, s.n ?? nW); if (n < 1) return; ctx.save(); ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 2; ctx.lineJoin = 'round'; ctx.beginPath();
    for (let i = 0; i < n; i++) { const x = X(i), y = Y(s.v[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); if (n === 1) dot(X(0), Y(s.v[0]), 3, s.col);
    if (s.label && n >= nW) txt(s.label, x1, Y(s.v[nW - 1]) - (s.dy ?? 9), { s: 12, c: s.lc || s.col, al: 'right' }); });
  return { X, Y };
}
const FCOL = '#c99a3a', OCOL = '#3f9a96';
const CMAX = Math.ceil(Math.max(...EXP.flat.steps, ...EXP.opt.steps) / 100) * 100;

const STEPS = [
 { h: 'A bigger maze', p: [`Pip moves into a house: four rooms of 5 by 5 cells, joined by four doors, A to D. Walls are shaded. Pip starts in room 1’s top-left corner; the cheese sits in door D, between rooms 3 and 4.`, `Rules as before: +1 for stepping onto the cheese, which ends the walk, γ (gamma) 0.9. No cat. The shortest route takes ${SHORT} steps.`],
   shot: [3, 9],
   say: [[["Pip moves into a bigger house. Four rooms, joined by four doors.", [60, 0, 280, 280]], ["Pip starts in the top left corner. The cheese sits in the bottom door, between rooms three and four.", [60, 0, 280, 280]]], [["No cat this time. The shortest route, in teal, takes thirteen steps.", [60, 0, 280, 280]]]],
   draw(t) { maze({ draw: t / 1.5, rooms: clamp(t - 2), cheese: ease((t - 2.5) * 2) }); pip(cx(0), cy(0), 1, t);
     if (t > 4.5) pathLine(GW_F.P, Math.floor((t - 4.5) * 6) + 1, C.teal, 2.2);
     if (t > 5) card(['4 rooms, 4 doors (A to D), 104 open cells', { s: 'cheese in door D: +1, walk over;  γ = 0.9', c: C.goldInk }, { s: `shortest route: ${SHORT} steps (teal)`, c: '#3f7f7c' }], t - 5, { gap: 1.2 }); } },
 { h: 'Tiny steps, a long way', p: [`First, plain Q-learning, as in its reel: a number for each move in each cell, one move in five random, α (alpha) 0.2. All start at 0, so walk 1 is pure wandering.`, `Here walk 1 took ${f0(F1.steps)} steps to stumble onto the cheese; over 20 runs with different seeds, ${f0(EXP.flat.steps[0])} on average. Dots mark the cells Pip passed.`],
   shot: [4, 11],
   say: ["First, plain Q-learning, just as before. Every number starts at zero, so the first walk is pure wandering.", [["Watch the dots pile up.", [60, 0, 280, 280]], ["This walk took eight hundred and forty steps to stumble onto the cheese.", [60, 0, 280, 280]]], [["The shortest route is only thirteen.", [30, 292, 340, 87]]]],
   draw(t) { const n = Math.floor(clamp((t - .8) / 6) * F1.steps); maze({}); trail(F1.tr, n, C.teal, .35); const p = n ? F1.tr[n - 1] : START; pip(cx(p[0]), cy(p[1]), 1, t);
     caption(n >= F1.steps ? `walk 1: ${f0(F1.steps)} steps` : `walk 1, step ${f0(n)}`);
     if (t > 7.5) card([`plain Q-learning, walk 1: ${f0(F1.steps)} steps (this run)`, { s: `average over 20 runs: ${f0(EXP.flat.steps[0])}`, c: C.soft }, { s: `shortest route: ${SHORT}`, c: C.soft }], t - 7.5, { gap: 1.2 }); } },
 { h: 'Think in big steps', p: ['You don’t plan a trip to the kitchen one footstep at a time. You think: leave this room, go down the hall, into the kitchen. Each big step hides many small ones.', 'For Pip: from room 1, go to door B; from door B, go to door D. Two big steps instead of thirteen small ones.'],
   shot: [3, 9],
   say: ["You don't plan a trip to the kitchen one footstep at a time. You think, leave this room, go down the hall, into the kitchen.", [["Pip can do the same. First, go to door B.", [60, 0, 280, 280]], ["Then, go to door D.", [60, 0, 280, 280]], ["Two big steps instead of thirteen small ones.", [60, 0, 280, 280]]]],
   draw(t) { maze({ rooms: .6 }); pip(cx(0), cy(0), 1, t);
     const u1 = clamp((t - 1.5) / 1.2), u2 = clamp((t - 3.5) / 1.2), B = DOORS[1], D = DOORS[3];
     if (u1 > 0) alpha(1, () => arrow(cx(0) + 6, cy(0) + 8, lerp(cx(0) + 6, cx(B[0]), u1), lerp(cy(0) + 8, cy(B[1]) - 10, u1), 650, { c: C.goldInk, w: 3, head: 9 }));
     if (u2 > 0) alpha(1, () => arrow(cx(B[0]) + 6, cy(B[1]) + 6, lerp(cx(B[0]) + 6, cx(D[0]) - 10, u2), lerp(cy(B[1]) + 6, cy(D[1]), u2), 660, { c: C.goldInk, w: 3, head: 9 }));
     if (t > 2.4) { box(146, 116, 118, 22, C.bg, .85 * clamp(t - 2.4)); txt('1. go to door B', 150, 127, { s: 16, c: C.goldInk, a: clamp(t - 2.4), al: 'left' }); } if (t > 4.4) txt('2. go to door D', 120, 230, { s: 16, c: C.goldInk, a: clamp(t - 4.4) });
     if (t > 6) card(['big steps: which door next', { s: 'small steps: the moves to get there', c: C.soft }], t - 6, { gap: 1.2 }); } },
 { h: 'An option', p: ['A big step needs its own little policy. Here is “go to door B”: an arrow in every cell of room 1 and door A, all leading to door B. It can **start** anywhere there, and **stops** at door B.', 'A mini-policy with a start zone and a stopping rule is an **option**. We hand-drew 8, one per door of each room.'],
   shot: [3, 10],
   say: [[["A big step needs its own little policy.", [60, 0, 280, 280]], ["Here's go to door B. Every arrow leads to that door.", [60, 0, 280, 280]]], [["It can start anywhere in the shaded zone, and it stops at door B.", [60, 0, 280, 280]], ["A mini policy, a start zone and a stopping rule. That's an option.", [30, 292, 340, 64]]], ["We drew eight of them by hand."]],
   draw(t) { maze({ shade: (x, y) => inRD(x, y, 0) ? [C.teal, .15 * clamp(t - 3.5)] : null }); optArrows(OI(0, 1), clamp(t - .8)); pip(cx(0), cy(0), 1, t);
     if (t > 3.5) { txt('start zone:', OX + 6 * CS + 2, 84, { s: 14, c: '#3f7f7c', a: clamp(t - 3.5), al: 'left' }); txt('room 1 + door A', OX + 6 * CS + 2, 102, { s: 14, c: '#3f7f7c', a: clamp(t - 3.5), al: 'left' }); }
     if (t > 4.5) { box(cx(2) + 22, cy(5) - 9, 64, 18, C.card, clamp(t - 4.5)); txt('stops here', cx(2) + 54, cy(5) + 1, { s: 13, c: C.goldInk, a: clamp(t - 4.5) }); }
     card([{ s: 'option = policy + where it can start + when it stops', c: C.goldInk }, { s: '8 options: each room, each of its 2 doors', c: C.soft }], t - 6, { gap: 1.2 }); } },
 { h: 'A bigger menu', p: ['The menu: 4 moves plus options that can start here: 2 inside a room, and in a doorway one per room it joins, like “go to door A” or “go to door D” in door B. Picking one is **one decision**: Pip follows its arrows until it stops.', 'Random picks happen only at decisions, and nobody says which choice is good: options are just extra choices.'],
   shot: [3, 10],
   say: [["Now Pip's menu grows.", ["Four moves, plus any option that can start here.", [30, 280, 340, 70]]], [["Watch Pip pick go to door B.", [60, 0, 280, 280]], ["One decision, and it follows the arrows all the way to the door.", [60, 0, 280, 280]]], ["Nobody says which choice is good. Options are just extra choices."]],
   draw(t) { maze({ rooms: .3 }); const P = []; let x = 0, y = 0; const o = OPTS[OI(0, 1)]; while (o.pol[K(x, y)] >= 0) { [x, y] = step(x, y, o.pol[K(x, y)]); P.push([x, y]); }
     const u = clamp((t - 4) / 2.5), n = Math.floor(u * P.length), p = n ? P[n - 1] : START; pathLine([START, ...P], n + 1, C.goldInk, 2); pip(cx(p[0]), cy(p[1]), 1, t);
     const items = ['up', 'right', 'down', 'left', 'go to door A', 'go to door B'];
     if (t < 4) { txt('Pip’s menu in its corner:', 40, 298, { s: 14, c: C.soft, al: 'left', a: clamp(t - .4) }); items.forEach((s, i) => { const a = clamp(t - .6 - i * .3), X = 40 + (i % 3) * 110 + (i > 3 ? (i - 4) * 55 : 0), Y = 318 + Math.floor(i / 3) * 0 + (i > 3 ? 32 : 0), xx = i > 3 ? 60 + (i - 4) * 150 : 40 + i * 82; box(xx, Y - 10, i > 3 ? 130 : 72, 22, i === 5 ? '#f6e7c8' : C.card, a); rrect(xx, Y - 10, i > 3 ? 130 : 72, 22, 880 + i, { c: C.line, w: 1 }); txt(s, xx + (i > 3 ? 65 : 36), Y + 1, { s: 15, a, c: i > 3 ? C.goldInk : C.ink }); }); }
     if (t >= 4) caption(n >= P.length ? `one decision, ${P.length} steps` : `following “go to door B”: step ${n}`);
     if (t > 7) card(['menu = 4 moves + options that can start here', { s: 'an option: one decision, many steps', c: C.goldInk }], t - 7, { gap: 1.2 }); } },
 { h: 'Learning numbers for options', p: [`Stopping after k steps, its target is the **discounted** reward en route, plus 0.9 to the k times the best move or option number at the stop. Only the chosen option’s number changes.`, `Walk 1 ends with “go to door D”: its +1 comes on step ${OLAST.n}, worth 0.9 to the ${OLAST.n - 1}: ${f3(OLAST.Rw)}. α 0.2 moves the number a fifth of the way: ${f3(OLAST.nw)}.`],
   shot: [4, 11],
   say: ["How does Pip learn a number for an option?", [["Watch the last option of walk one, go to door D.", [60, 0, 280, 280]], ["The cheese comes on its fourth step, so the reward is discounted three times, to zero point seven two nine.", [30, 292, 340, 87]]], [["Alpha moves the number a fifth of the way, to zero point one four six.", [30, 292, 340, 87]]]],
   draw(t) { maze({}); const tr = O1.tr, a = OLAST.t0, P = tr.slice(a), n = Math.floor(clamp((t - .8) / 2.4) * P.length); trail(tr.slice(0, a), a, C.teal, .25);
     const start = a ? tr[a - 1] : START; pathLine([start, ...P], n + 1, C.goldInk, 2.2); const p = n ? P[n - 1] : start; pip(cx(p[0]), cy(p[1]), 1, t); if (n >= P.length) bubble('+1', cx(GOAL[0]), cy(GOAL[1]) - 12, t - 3.4);
     if (t > 4) card([`target = reward on the way + 0.9^k × best at the stop`, `“go to door D”: k = ${OLAST.n}, reward 0.9^${OLAST.n - 1} = ${f3(OLAST.Rw)}, walk over`, { s: `new number = 0 + 0.2 × (${f3(OLAST.tg)} − 0) = ${f3(OLAST.nw)}`, c: C.goldInk }], t - 4, { gap: 1.3, fs: 16 }); } },
 { h: 'Walk 1, side by side', p: [`Same seed, same rules, walk 1. Plain Q-learning, gold, wanders for ${f0(F1.steps)} steps. With options, teal, Pip needs ${f0(O1.steps)} steps and ${O1.dec} decisions.`, `Why so much faster? A random option carries Pip a whole room at a time, so even pure wandering crosses the house quickly. Over 20 runs, walk 1 takes ${f0(EXP.opt.steps[0])} steps with options on average, against ${f0(EXP.flat.steps[0])}.`],
   shot: [5, 12],
   say: [[["Same seed, same rules, walk one, side by side.", [60, 0, 280, 280]], ["Gold is plain Q-learning. Teal uses options.", [60, 0, 280, 280]]], [["Watch teal reach the cheese in thirty nine steps, while gold keeps wandering.", [60, 0, 280, 280]]], ["Why so fast? A random option carries Pip a whole room at a time."]],
   draw(t) { const u = clamp((t - .8) / 6), nf = Math.floor(u * F1.steps), no = Math.floor(clamp((t - .8) / 2.5) * O1.steps); maze({}); trail(F1.tr, nf, FCOL, .3); trail(O1.tr, no, OCOL, .8);
     const pf = nf ? F1.tr[nf - 1] : START, po = no ? O1.tr[no - 1] : START; pip(cx(pf[0]), cy(pf[1]), 1, t); alpha(1, () => pip(cx(po[0]) + 3, cy(po[1]) - 3, 1, t + .3));
     card([{ s: `plain: ${f0(nf)} steps` + (nf >= F1.steps ? ' (cheese)' : ''), c: '#9a6f14' }, { s: `with options: ${f0(no)} steps, ${no >= O1.steps ? O1.dec : '…'} decisions` + (no >= O1.steps ? ' (cheese)' : ''), c: '#2f6f6c' }, { s: `20 runs, walk 1: ${f0(EXP.flat.steps[0])} vs ${f0(EXP.opt.steps[0])} steps`, c: C.soft }], t - 1, { gap: 1.2 }); } },
 { h: 'Sixty walks', p: [`Now ${NW} walks per run. Across is the walk number; up is the steps that walk took, averaged over the same 20 seeds for both learners. The dashed line is the shortest route, ${SHORT}.`, `Gold, plain Q-learning, needs many walks to come down. Teal, with options, is near the bottom almost at once. Over all ${NW} walks: ${f0(TOT.flat)} steps against ${f0(TOT.opt)}.`],
   shot: [4, 12],
   say: [[["Now sixty walks per run, averaged over twenty runs.", [20, 30, 370, 380]], ["Up is how many steps each walk took. The dashed line is the shortest route.", [20, 30, 370, 380]]], [["Gold needs many walks to come down.", [20, 30, 370, 380]], ["Teal is near the bottom almost at once.", [20, 30, 370, 380]]]],
   draw(t) { const n1 = clamp(Math.floor((t - 2) * 15), 0, NW), n2 = clamp(Math.floor((t - 5) * 15), 0, NW);
     chart({ y0: 380, y1: 50, max: CMAX, short: SHORT, shortLabel: t > 1 ? `shortest route, ${SHORT}` : '', title: 'steps per walk (mean of 20 runs)', legend: [['plain Q-learning', FCOL], ['with options', OCOL]].slice(0, t > 5 ? 2 : t > 2 ? 1 : 0), lines: [{ v: EXP.flat.steps, col: FCOL, n: n1 }, { v: EXP.opt.steps, col: OCOL, n: n2, w: 2.6 }] });
     if (t > 9) txt(`all ${NW} walks: ${f0(TOT.flat)} steps vs ${f0(TOT.opt)}`, W / 2, 404, { s: 15, c: C.soft, a: clamp(t - 9) }); } },
 { h: 'Manager and worker', p: [`After ${NW} walks, follow the best choices. Plain: ${GW_F.D.length} decisions, one per step. With options: ${GW_O.D.length} decisions, “${OPTS[GW_O.D[0].a - 4].name}” then “${OPTS[GW_O.D[GW_O.D.length - 1].a - 4].name}”, over the same ${GW_O.P.length - 1} steps.`, 'Think of a **manager** that picks the next big step, and a **worker**, the option’s arrows, that takes the small steps. The manager’s problem is tiny: a few decisions per walk.'],
   shot: [3, 10],
   say: [[["After sixty walks, follow the best choices.", [60, 0, 280, 280]], ["With options, Pip makes just two decisions, the gold dots. Door B, then door D.", [60, 0, 280, 280]]], [["Think of a manager who picks the next big step, and a worker who takes the small ones.", [30, 292, 340, 87]], ["Plain Q-learning makes thirteen decisions for the same route.", [30, 292, 340, 87]]]],
   draw(t) { maze({}); const P = GW_O.P, n = Math.floor(clamp((t - .8) / 4) * P.length); pathLine(P, n, OCOL, 2.4);
     const decAt = []; let idx = 0; GW_O.D.forEach(d => { decAt.push([d.x, d.y]); });
     decAt.forEach(([x, y], i) => { const k = P.findIndex(p => p[0] === x && p[1] === y); if (k < n || (k === 0 && t > .8)) { dot(cx(x), cy(y), 7, C.goldInk, .9); txt(String(i + 1), cx(x), cy(y) + 1, { s: 12, c: '#fff' }); } });
     const p = n ? P[n - 1] : START; pip(cx(p[0]), cy(p[1]), 1, t);
     if (t > 5.5) card([{ s: `manager: ${GW_O.D.length} decisions (gold dots)`, c: C.goldInk }, { s: `worker: the option’s arrows, ${GW_O.P.length - 1} steps`, c: '#2f6f6c' }, { s: `plain Q-learning: ${GW_F.D.length} decisions for ${GW_F.P.length - 1} steps`, c: C.soft }], t - 5.5, { gap: 1.2 }); } },
 { h: 'Why news travels faster', p: [`In plain Q-learning, the cheese’s +1 creeps back about one cell per walk, so the start learns of it only once a chain of ${SHORT} numbers fills in. Over the 20 runs, the start’s best number first rose above 0 after about ${Math.round(EXP.flat.knows)} walks.`, `With options the chain has two links: start to door B, door B to cheese. Average: about ${Math.round(EXP.opt.knows)} walks.`],
   shot: [4, 11],
   say: [[["Why does news travel faster?", [60, 0, 280, 280]], ["In plain Q-learning, the cheese's reward creeps back about one cell per walk. A chain of thirteen links.", [60, 0, 280, 280]]], [["With options, the chain has just two links.", [60, 0, 280, 280]], ["The start learns of the cheese after about five walks, instead of about twenty three.", [30, 292, 340, 87]]]],
   draw(t) { maze({}); pathLine(GW_F.P, Math.floor(clamp((t - .6) / 3) * GW_F.P.length), FCOL, 2);
     GW_F.P.forEach(([x, y], i) => { if (t > .6 + i * .22) dot(cx(x), cy(y), 3.5, FCOL, .9); });
     if (t > 4) { const B = DOORS[1]; alpha(clamp(t - 4), () => { arrow(cx(0) + 4, cy(0) + 4, cx(B[0]) - 2, cy(B[1]) - 10, 671, { c: OCOL, w: 3, head: 8 }); arrow(cx(B[0]) + 8, cy(B[1]) + 4, cx(GOAL[0]) - 10, cy(GOAL[1]) - 2, 672, { c: OCOL, w: 3, head: 8 }); }); }
     if (t > 5.5) card([{ s: `plain: ${SHORT} links, start knows after ~${Math.round(EXP.flat.knows)} walks`, c: '#9a6f14' }, { s: `options: 2 links, start knows after ~${Math.round(EXP.opt.knows)} walks`, c: '#2f6f6c' }, { s: 'average over the 20 runs', c: C.soft }], t - 5.5, { gap: 1.2 }); } },
 { h: 'When options get in the way', get p() { const [pf, po] = PIT60(); return [`Move the cheese to room 4’s far corner, seven steps from any door. Same options and seeds; walk 1 capped at ${f0(CAP)}. Plain Q-learning averages ${f0(PIT.fMean)} steps; with options, ${PIT.oCap} of 20 runs hit the cap.`, `One choice in three is an option, and each ends at a door, yanking Pip back. Finding the cheese, it slowly recovers. With options versus plain: walk 20, ${f0(po[19])} to ${f0(pf[19])}; walk 60, ${f0(po[59])} to ${f0(pf[59])}.`]; },
   shot: [4, 11],
   say: [[["But options can get in the way.", [60, 0, 280, 280]], ["Move the cheese to the far corner of room four, well away from any door.", [60, 0, 280, 280]]], [["Watch Pip. Every option ends on a door, so it keeps getting yanked back.", [60, 0, 280, 280]]], [["Nineteen of twenty runs hit the step limit.", [30, 292, 340, 87]], "Plain Q-learning does far better here."]],
   draw(t) { const n = Math.floor(clamp((t - .8) / 5) * PITW.tr.length); maze({ goal: CORNER, shade: (x, y) => doorAt(x, y) >= 0 ? [C.rose, .25] : null }); trail(PITW.tr, n, OCOL, .6);
     const p = n ? PITW.tr[n - 1] : START; pip(cx(p[0]), cy(p[1]), 1, t); caption(`with options, first ${f0(n)} steps: ${PITW.tr.slice(0, n).filter(([x, y]) => doorAt(x, y) >= 0).length} of them on a door`);
     if (t > 6.5) card([{ s: `cheese in the corner, walk 1, limit ${f0(CAP)} steps:`, c: C.ink }, { s: `plain: ${f0(PIT.fMean)} steps on average, ${PIT.fCap} of 20 hit the limit`, c: '#9a6f14' }, { s: `options: ${PIT.oCap} of 20 runs hit the limit`, c: C.roseInk }], t - 6.5, { gap: 1.2 }); } },
 { h: 'The hard part', p: ['Options help only when their stopping points suit the task. We drew ours knowing where the doors were. Making an agent **discover** good options itself is the hard, still open problem of hierarchical RL.', 'Some methods seek bottlenecks like doorways, which many good routes pass through. Others, like option-critic (2017), learn options and stopping rules from reward alone.'],
   shot: [3, 9],
   say: [[["Options only help when their stopping points suit the task.", [60, 0, 280, 280]], ["We drew ours knowing where the doors were.", [60, 0, 280, 280]]], [["Getting an agent to discover good options by itself is the hard, still open problem.", [30, 292, 340, 87]]], ["Some methods look for bottlenecks, like doorways. Others learn options from reward alone."]],
   draw(t) { maze({ shade: (x, y) => doorAt(x, y) >= 0 ? [C.gold, .35 + .2 * Math.sin(t * 3)] : null, cheese: 0 });
     card(['good options = good stopping points', { s: 'here: hand-made from the map of doors', c: C.soft }, { s: 'discovering them is the hard part', c: C.roseInk }], t - 1, { gap: 1.3 }); } },
 { h: 'Other ways to split the work', p: ['Options come from Sutton, Precup and Singh, 1999. Others split the work differently.', 'In **feudal** designs, dating to Dayan and Hinton, 1993, a manager hands out goals and is rewarded by the task, while the worker is rewarded only for reaching them. In **goal-conditioned** versions, the manager picks a target state, like “be at door B”, and one worker policy learns to reach any target.'],
   shot: [3, 9],
   say: ["Options came from Sutton, Precup and Singh, in 1999. Others split the work differently.", [["In feudal designs, a manager hands out goals, and the worker is rewarded just for reaching them.", [30, 120, 340, 110]]], [["In goal conditioned versions, one worker learns to reach any target the manager picks.", [30, 120, 340, 110]]]],
   draw(t) { maze({ rooms: .3, cheese: 1 }); dim(.5);
     card(['options: Sutton, Precup, Singh (1999)', { s: 'feudal: manager gives goals, worker earns', c: C.goldInk }, { s: '   reward for reaching them (Dayan, Hinton 1993)', c: C.goldInk }, { s: 'goal-conditioned: one worker for any target', c: '#2f6f6c' }], t - .6, { y: 120, gap: 1.4 }); } },
 { h: 'Your turn', ctl: { label: 'cheese', values: [0, 1, 2, 3], def: 0, fmt: v => SPN[v], onChange: () => {} },
   p: ['Slide to move the cheese: door D, door C, the middle of room 4, or its far corner. Both learners rerun, 10 seeds, 30 walks each, walks stopped at 3,000 steps.', 'Where do the options shine, and where do they hurt? Look at how close the cheese is to a door.'],
   shot: [3, 9],
   say: ["Your turn. Slide to move the cheese, from door D to the far corner. Both learners rerun.", [["Watch the two lines.", [20, 205, 370, 205]], ["Where do options shine, and where do they hurt?", [20, 205, 370, 205]]], ["Look at how close the cheese is to a door."]],
   draw(t, st) { const i = st.ci ?? 0, R = yt(i), mx = Math.ceil(Math.max(...R.flat.steps, ...R.opt.steps) / 100) * 100, n = t < (st.changed ?? 0) ? 30 : clamp(Math.floor((t - (st.changed ?? 0) - .3) * 40), 0, 30);
     withGeom(16, 112, 6, () => { maze({ goal: SPOTS[i] }); pip(cx(0), cy(0), 1, t, .36); });
     chart({ y0: 376, y1: 222, nW: 30, max: mx, title: `steps per walk, mean of 10 runs: cheese at ${SPN[i]}`, legend: [['plain', FCOL], ['options', OCOL]], lines: [{ v: R.flat.steps, col: FCOL, n }, { v: R.opt.steps, col: OCOL, n, w: 2.6 }] });
     txt(`walks 1 to 10: plain ${f0(mean(R.flat.steps.slice(0, 10)))}, options ${f0(mean(R.opt.steps.slice(0, 10)))} steps on average`, W / 2, 404, { s: 14, c: C.soft }); } },
 { h: 'Where it is used', p: ['Robots often combine skills like reach, grasp and place, hand-written or learned one at a time, with a higher-level policy picking the next.', 'DeepMind’s FeUdal Networks, in 2017, used a manager setting goals for a worker in Atari games like Montezuma’s Revenge. It helps most when tasks are long, rewards rare, and sub-goals fit the task.'],
   shot: [3, 9],
   say: [["Robots often chain skills like reach, grasp and place.", ["A higher level policy picks the next skill, and the skill does the moves.", [30, 120, 340, 87]]], ["DeepMind's FeUdal Networks used a manager and a worker on hard Atari games.", ["It helps most when tasks are long and rewards are rare.", [30, 120, 340, 87]]]],
   draw(t) { maze({ rooms: .3 }); pathLine(GW_O.P, GW_O.P.length, OCOL, 2.2); dim(.45);
     card(['robot: reach → grasp → place', 'manager picks the skill, skills do the moves', { s: 'best for long tasks with rare rewards', c: C.soft }], t - .6, { y: 120, gap: 1.3 }); } },
 { h: 'Recap', p: ['**Hierarchical RL** plans with big steps and small ones. An **option** is a mini-policy with a start zone and a stopping rule; picking one is a single decision that may take many steps.', 'With options that fit the task, Pip explored and learned far faster. With ill-fitting options it did worse. Finding good options automatically is the hard part.'],
   shot: [3, 9],
   say: ["So that's hierarchical RL. Plan with big steps and small ones.", [["An option is a mini policy with a start zone and a stopping rule. Picking one is a single decision.", [30, 120, 340, 110]]], [["Options that fit the task help a lot. Bad ones hurt. Finding good ones automatically is the hard part.", [30, 120, 340, 110]]]],
   draw(t) { maze({ rooms: .3 }); pathLine(GW_O.P, GW_O.P.length, OCOL, 2.2); dim(.4);
     card(['option = policy + start zone + stopping rule', 'manager picks options, options take the moves', 'target = reward on the way + γ^k × best at the stop', { s: 'fitting options help, bad ones hurt', c: C.roseInk }], t - .3, { y: 120, gap: 1 }); } },
];
function dim(a = .7, y0 = 0, y1 = 285) { box(0, y0, W, y1 - y0, C.bg, a); }

RLStory.register('hrl', { title: 'Hierarchical RL', track: 'Bigger worlds', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { EXP, LAST10, TOT, F1, O1, OLAST, FIRSTBOOT, GW_O, GW_F, PIT, DOORVIS, SHORT, yt, OPTS } });
})();
