// TD learning as a story: Pip keeps its usual habit (the arrows, 1 move in 5 random, dry floor, γ = 0.9) and learns that
// habit's notebook from experience alone, one step at a time: no rulebook, no waiting for the walk to end.
// Every number comes from the seeded walks below; the exact notebook comes from Bellman sweeps (as in bellman.js).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, key, maze, trail, walkPos, pip } = G;
const GAM = 0.9, EPS = 0.2, NWALK = 200, A0 = 0.2, ALPHAS = [0.05, 0.1, 0.2, 0.5];
const K30 = key(3, 0), K20 = key(2, 0), SK = key(0, 3), K13 = key(1, 3);
// ───── Pip's habit: the arrows from the values / bellman reels (0 up, 1 right, 2 down, 3 left)
const ARR = new Array(20).fill(-1);
[[0, 3, 0], [0, 2, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [2, 1, 0], [1, 2, 3], [1, 3, 1], [2, 3, 1], [3, 3, 1], [4, 3, 0], [4, 2, 0], [4, 1, 0], [3, 2, 1]]
  .forEach(([c, r, m]) => { ARR[key(c, r)] = m; });
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);
// ───── the walks: the habit never looks at the notebook, so the same walks serve every α
// the same start walks as the values, mc and traces reels: rng(87000 + start key); walk 1 is the 7-step cheese walk
const WALKS = (() => { const R = rng(87000 + SK), out = [];
  for (let i = 0; i < NWALK; i++) { let c = 0, r = 3; const tr = [];
    for (let k = 0; k < 400; k++) { let m, rnd = false; if (R() < EPS) { rnd = true; m = Math.floor(R() * 4); } else m = ARR[key(c, r)];
      const [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr); tr.push({ c, r, nc, nr, m, rnd, rew }); c = nc; r = nr; if (rew) break; }
    out.push({ tr, end: tr[tr.length - 1].rew }); }
  return out; })();
// ───── TD(0): after every step, nudge the guess of the cell just left toward reward + γ × guess where Pip landed
const memo = {};
function runTD(a) { if (memo[a]) return memo[a]; const V = new Array(20).fill(0), eps = [];
  WALKS.forEach(w => { const tr = w.tr.map(s => { const k = key(s.c, s.r), old = V[k], next = s.rew ? 0 : V[key(s.nc, s.nr)], target = s.rew + GAM * next, nw = old + a * (target - old);
    V[k] = nw; return Object.assign({ old, next, target, err: target - old, nw }, s); }); eps.push({ tr, end: w.end, V: V.slice() }); });
  return (memo[a] = eps); }
const E = runTD(A0);
// the exact notebook of the habit: Bellman sweeps until nothing changes (bellman reel)
const EXACT = (() => { let V = new Array(20).fill(0); for (let i = 0; i < 300; i++) { const N = V.slice(); CELLS.forEach(([c, r]) => { const k = key(c, r); let v = 0;
  for (let m = 0; m < 4; m++) { const p = EPS / 4 + (m === ARR[k] ? 1 - EPS : 0), [nc, nr] = G.step(c, r, m), rw = G.term(nc, nr); v += p * (rw + (rw ? 0 : GAM * V[key(nc, nr)])); } N[k] = v; }); V = N; } return V; })();
// V after walk e's first i steps (for any α)
function Vat(Ea, e, i) { const V = e > 0 ? Ea[e - 1].V.slice() : new Array(20).fill(0); for (let k = 0; k < i; k++) { const s = Ea[e].tr[k]; V[key(s.c, s.r)] = s.nw; } return V; }
// ───── numbers the story quotes
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−');
const fv = v => Math.abs(v) < .0005 ? '0' : Math.abs(v) < .1 ? v.toFixed(3).replace('-', '−') : f2(v);   // as on the canvas
const w1 = E[0].tr, w2 = E[1].tr, LAST1 = w1[w1.length - 1];
const LEAK = w2.findIndex(s => s.c === 2 && s.r === 0 && s.nc === 3), S2 = w2[LEAK], LAST2 = w2[w2.length - 1];
const V30 = E.map(e => e.V[K30]), VS = E.map(e => e.V[SK]);
const CAT1 = WALKS.findIndex(w => w.end < 0), CATS = WALKS.map((w, i) => w.end < 0 ? i : -1).filter(i => i >= 0);
const LC = CATS[CATS.length - 1], PC = CATS[CATS.length - 2];          // last cat walk before the end, and the one before it
const lastOf = e => E[e].tr[E[e].tr.length - 1];
const SUR = E.map((e, i) => lastOf(i).err);                            // TD error of the walk's final step
const EXP = E.map((e, i) => lastOf(i).old);                            // what Pip expected just before that step
const REST2 = w2.slice(LEAK + 1);                                       // walk 2 after the news step (a bump, then the cheese)
const VISITS = new Array(20).fill(0); WALKS.forEach(w => w.tr.forEach(s => VISITS[key(s.c, s.r)]++));
const range = (arr, a, b) => { const s = arr.slice(a, b); return [Math.min(...s), Math.max(...s)]; };
const ES = runTD(0.05), V30S = ES.map(e => e.V[K30]), VSS = ES.map(e => e.V[SK]);
const RG2 = range(V30, 100, 200), RG05 = range(V30S, 100, 200);
// Monte Carlo on the same walks (mc reel's rule: first-visit returns, plain running average)
const MC = (() => { const V = new Array(20).fill(0), N = new Array(20).fill(0); WALKS.forEach(w => { const G = []; let g = 0; for (let i = w.tr.length - 1; i >= 0; i--) { g = w.tr[i].rew + GAM * g; G[i] = g; }
  const seen = new Set(); w.tr.forEach((s, i) => { const k = key(s.c, s.r); if (seen.has(k)) return; seen.add(k); N[k]++; V[k] += (G[i] - V[k]) / N[k]; }); }); return { V, N }; })();
// TD variants on the same walks: landing cells' true values plugged in (no bootstrapping from bad guesses); α = 1/n per cell
function tdVar(o) { const V = new Array(20).fill(0), N = new Array(20).fill(0); WALKS.forEach(w => w.tr.forEach(s => { const k = key(s.c, s.r); N[k]++;
  const nx = s.rew ? 0 : (o.oracle ? EXACT : V)[key(s.nc, s.nr)], a = o.inv ? 1 / N[k] : A0; V[k] += a * (s.rew + GAM * nx - V[k]); })); return V; }
const TDOR = tdVar({ oracle: 1 }), TDINV = tdVar({ inv: 1 });
const NEXT13 = Math.max(...E.flatMap(e => e.tr.filter(s => s.c === 1 && s.r === 3).map(s => s.next)));   // biggest landing guess seen from the cell right of the start
const ROUTE = [[0, 3], [0, 2], [0, 1], [0, 0], [1, 0], [2, 0], [3, 0]], OFF = CELLS.filter(([c, r]) => !ROUTE.some(([a, b]) => a === c && b === r) && VISITS[key(c, r)] > 0);
const merr = (V, cs) => cs.reduce((x, [c, r]) => x + Math.abs(V[key(c, r)] - EXACT[key(c, r)]), 0) / cs.length;
const maxe = (V, cs) => Math.max(...cs.map(([c, r]) => Math.abs(V[key(c, r)] - EXACT[key(c, r)])));
const ERR = { mcRmax: maxe(MC.V, ROUTE), tdRmax: maxe(E[NWALK - 1].V, ROUTE), mcR: merr(MC.V, ROUTE), tdR: merr(E[NWALK - 1].V, ROUTE), mcO: merr(MC.V, OFF), tdO: merr(E[NWALK - 1].V, OFF) };
const FIN = E[NWALK - 1].V, MEAN2 = V30.slice(NWALK - 100).reduce((x, y) => x + y, 0) / 100;   // average guess left of the cheese over the last 100 walks

// ───── drawing helpers
const ZERO = () => new Array(20).fill(0);
function nums(V, a = 1, Gx = G, s = 14, only) {
  for (const [c, r] of CELLS) { if (only && !only(c, r)) continue; const aa = typeof a === 'function' ? a(c, r) : a; if (aa <= 0) continue; const v = V[key(c, r)];
    txt(fv(v), Gx.ox + c * Gx.cs + Gx.cs - 5, Gx.oy + r * Gx.cs + (s >= 14 ? 12 : 10), { s, al: 'right', a: aa, c: v < -.0005 ? C.roseInk : v > .0005 ? C.goldInk : C.soft }); }
}
function arrows(a = 1, Gx = G) { CELLS.forEach(([c, r]) => alpha(a, () => Gx.arrowIn(c, r, ARR[key(c, r)], 400 + key(c, r), { c: C.goldInk, w: 1.4 }))); }
function ghost(c, r, s, a = 1, o = {}) { if (o.bg) box(cx(c) - 29, cy(r) + (o.dy ?? 20) - 9, 58, 18, C.card, .9 * a); txt(s, cx(c), cy(r) + (o.dy ?? 20), { s: o.s || 14, c: o.c || '#9a8f84', a }); }
function dim(a = .62) { box(0, 0, W, 285, C.bg, a); }
function edgeArrow(c, r, m, seed, col, r0 = 12) { const [dx, dy] = G.MOVES[m]; arrow(cx(c) + dx * r0, cy(r) + dy * r0, cx(c) + dx * 29, cy(r) + dy * 29, seed, { c: col, w: 1.6, head: 6 }); }
function car(x, y) { rrect(x - 13, y - 9, 26, 11, 777, { w: 1.2 }); box(x - 12, y - 8, 24, 9, '#9fc9c7'); rcircle(x - 7, y + 3, 3, 778, { fill: C.ink, j: .3 }); rcircle(x + 7, y + 3, 3, 779, { fill: C.ink, j: .3 }); }
function commute(t) {
  const y = 362, xs = [52, 150, 252, 350], names = ['office', '10 min in', 'traffic jam!', 'home'];
  rline(40, y, 362, y, 880, { w: 2 }); xs.forEach((x, i) => { rline(x, y - 5, x, y + 5, 881 + i, { w: 1.5 }); txt(names[i], x, y + 18, { s: 14, c: i === 2 ? C.roseInk : C.soft }); });
  let x = xs[0]; if (t > 1.5) x = lerp(xs[0], xs[1], ease((t - 1.5) / 1.5)); if (t > 4) x = lerp(xs[1], xs[2], ease((t - 4) / 1.5));
  car(x, y - 12);
  const say = (s, a, col) => txt(s, W / 2, 300, { s: 18, a, c: col });
  say('at the office: guess 30 min in total', clamp((t - .3) * 2) * (1 - clamp((t - 3) * 2)));
  if (t > 3.2) say('10 gone + 20 to go = still 30', clamp((t - 3.2) * 2) * (1 - clamp((t - 5.6) * 2)), C.soft);
  if (t > 5.8) say('10 gone + 30 to go = 40!  update now', clamp((t - 5.8) * 2), C.roseInk);
  if (t > 7.5) txt('seen so far  +  guess for the rest', W / 2, 324, { s: 15, c: C.goldInk, a: clamp(t - 7.5) });
}
// line chart of guesses over walks. o = { x0, x1, y0 (bottom), y1 (top), lo, hi, n, series: [{ v, col, w }], exact: [{ v, col, label }], cats, title }
function lineChart(o) {
  const n = o.n ?? NWALK, lo = o.lo ?? 0, hi = o.hi ?? 1, X = i => o.x0 + i / (NWALK - 1) * (o.x1 - o.x0), Y = v => o.y0 - (v - lo) / (hi - lo) * (o.y0 - o.y1);
  rline(o.x0 - 4, Y(0), o.x1 + 4, Y(0), 960, { c: C.line, w: 1, once: true }); rline(o.x0 - 4, o.y0, o.x0 - 4, o.y1 - 2, 961, { c: C.line, w: 1, once: true });
  txt('0', o.x0 - 14, Y(0), { s: 12, c: C.soft }); txt('1', o.x0 - 14, Y(1), { s: 12, c: C.soft }); if (lo < 0) txt(String(lo).replace('-', '−'), o.x0 - 18, o.y0, { s: 12, c: C.soft });
  txt('walk 1', o.x0, o.y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt('walk ' + NWALK, o.x1, o.y0 + 11, { s: 12, c: C.soft, al: 'right' });
  if (o.title) txt(o.title, o.x0 - 8, o.y1 - 12, { s: 13, c: C.soft, al: 'left' });
  (o.exact || []).forEach(e => { ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = e.col || C.grey; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(o.x0, Y(e.v)); ctx.lineTo(o.x1, Y(e.v)); ctx.stroke(); ctx.restore();
    if (e.label) txt(e.label, o.x1 + 4, Y(e.v) + (e.dy || 0), { s: 12, c: e.lc || C.soft, al: 'left' }); });
  if (o.cats) CATS.forEach(i => { if (i < n) rline(X(i), o.y0, X(i), o.y0 - 7, 990 + i, { c: C.rose, w: 1.4, once: true, j: .2 }); });
  (o.series || []).forEach(s => { ctx.save(); ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 1.4; ctx.lineJoin = 'round'; ctx.beginPath();
    for (let i = 0; i < Math.min(s.n ?? n, NWALK); i++) { const x = X(i), y = Y(s.v[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); });
  return { X, Y };
}
// small chart under the maze (surprise at the cheese, walk by walk)
function strip(n) {
  const x0 = 50, x1 = 360, y0 = 384, y1 = 304, w = (x1 - x0) / NWALK, Y = v => y0 - v * (y0 - y1);
  rline(x0, y0, x1, y0, 900, { c: C.line, w: 1 }); txt('surprise when Pip reached the cheese', x0 - 6, y1 - 10, { s: 14, al: 'left', c: C.soft });
  txt('1', x0 - 12, y1, { s: 12, c: C.soft }); txt('0', x0 - 12, y0, { s: 12, c: C.soft });
  for (let i = 0; i < n; i++) { const x = x0 + i * w + w / 2;
    if (E[i].end < 0) { rline(x, y0 + 2, x, y1 + 6, 950 + i, { c: C.rose, w: 1.2, once: true, j: .4 }); continue; }
    dot(x, Y(SUR[i]), 2.1, C.orange); }
}
// the two notebooks side by side
const GA = L.gridWorld({ cs: 38, ox: 8, oy: 44 }), GB = L.gridWorld({ cs: 38, ox: 202, oy: 44 });

const STEPS = [
 { h: 'Back in the maze', p: ['Same maze and rules as the values and Bellman reels, on a **dry** floor: moves go where Pip points. Stepping onto the cheese pays +1, onto the cat −1, and either ends the walk. γ (gamma) is 0.9.', `Pip keeps its **habit**: arrows, but one move in five random. The sweeps computed its notebook: ${f2(EXACT[K30])} left of the cheese, ${f2(EXACT[SK])} at the start.`],
   say: [[["Welcome back to Pip's maze.", [40, 16, 320, 256]], ["Same cheese, same cat, and a dry floor, so every move goes where Pip points.", [40, 16, 320, 256]]], [["The faint arrows are Pip's habit.", [40, 16, 320, 256]], ["The numbers are what that habit is worth, worked out by the sweeps.", [40, 16, 320, 256]], ["Zero point eight seven beside the cheese.", [40, 16, 320, 256]], ["Zero point four at the start.", [40, 16, 320, 256]], ["Pip must now learn them the hard way.", [40, 16, 320, 256]]]],
   shot: [3, 9],
   draw(t) { maze({ V: EXACT, t }); nums(EXACT, (c, r) => clamp((t - .4 - key(c, r) * .05) * 3)); arrows(.35 * clamp(t - 1.5)); pip(cx(0), cy(3), 1, t);
     if (t > 4) card(['dry floor: each move goes where Pip points', 'habit: the arrows, 1 move in 5 random', { s: 'numbers: the habit’s values, from the sweeps', c: C.soft }], t - 4, { gap: 1.1 }); } },
 { h: 'No rulebook, no waiting', p: ['The sweeps needed the **rulebook**: where every move leads and what it pays. Monte Carlo learning needs no rulebook: it averages returns of whole walks. But a return is known only when the walk ends.', 'Now Pip gets neither help. Its notebook starts blank, every guess 0, and it must learn the habit’s numbers while walking, from what it actually sees.'],
   say: [["The sweeps had a cheat sheet.", ["They knew where every move leads.", [30, 292, 340, 87]]], [["Monte Carlo didn't need that, but it had to wait until each walk was over.", [30, 292, 340, 87]]], [["Now Pip gets neither shortcut.", [30, 292, 340, 87]], ["Every guess starts at zero.", [40, 16, 320, 256]], ["And it has to learn while it walks.", [40, 16, 320, 256]]]],
   shot: [3, 9],
   draw(t) { maze({ V: ZERO(), t }); nums(ZERO(), clamp(t - .5)); arrows(.2); pip(cx(0), cy(3), 1, t);
     card([{ s: 'sweeps: need the rulebook   ✗', c: C.roseInk }, { s: 'Monte Carlo: wait for the end   ✗', c: C.roseInk }, { s: 'now: learn after every step, no rulebook', c: C.goldInk }], t - 1.2, { gap: 1.4 }); } },
 { h: 'Learn now, not later', p: ['Think of driving home. You guess 30 minutes. Ten minutes in, you hit a traffic jam, so you revise: 40 minutes. You learned before you got home.', 'Your new guess was what you had seen so far plus a guess for the rest. That idea is **temporal-difference learning**, or TD learning, and Pip’s target will have exactly this shape.'],
   say: [[["Picture driving home from the office.", [20, 286, 360, 110]], ["You expect the trip to take half an hour.", [20, 286, 360, 110]]], [["Ten minutes in, you're still on track.", [20, 286, 360, 110]], ["Then, a traffic jam.", [20, 286, 360, 110]], ["You don't wait until you're home to feel late. You update right there, to forty minutes.", [20, 286, 360, 110]]], [["Time so far, plus a fresh guess for the rest.", [20, 286, 360, 110]], "That's the whole trick behind TD learning."]],
   shot: [4, 9.5],
   draw(t) { maze({ V: ZERO(), t }); pip(cx(0), cy(3), 1, t); dim(); commute(t); } },
 { h: 'The Bellman rule, one real step', p: ['The Bellman rule averages, over all moves, reward + 0.9 × value where the move lands; that needs the rulebook.', 'Each step gives one real example: a reward and a landing. Pip’s **target**, reward + 0.9 × its guess there, is that rule with one real step for the average. Over many steps, targets average to the Bellman sum, if the guesses are right.'],
   say: [[["Look at the start cell.", [40, 16, 320, 256]], ["The arrows show every move Pip might make, and how likely each one is.", [40, 16, 320, 256]], ["Mostly up.", [40, 16, 320, 256]]], [["The Bellman rule averages over all of them. That needs the rulebook.", [30, 292, 340, 106]]], [["Now watch Pip take one real step.", [40, 16, 320, 256]], ["TD uses just that: the reward, plus zero point nine times its guess where it landed.", [30, 292, 340, 106]]]],
   shot: [3, 10],
   draw(t) { const keep = clamp((t - 4.2) * 1.5); maze({ t, hi: t > 5 ? [[0, 3, C.teal, .22], [0, 2, C.gold, .25]] : null });
     const lab = [['up 0.85', cx(0) + 30, cy(2) + 24], ['right 0.05', cx(1) + 8, cy(3) + 22], ['down, left: bump, 0.05 each', 42, 282], null];
     [0, 1, 2, 3].forEach((m, i) => { const u = clamp(t - .5 - i * .5) * (m === 0 ? 1 : 1 - keep * .8); if (u <= 0) return;
       alpha(u, () => edgeArrow(0, 3, m, 1500 + m, C.ink, 17)); if (m !== 3) txt(lab[m][0], lab[m][1], lab[m][2], { s: 13, c: C.ink, al: m === 2 ? 'left' : 'center', a: u * (m === 0 ? 1 - keep : 1) }); });
     const m = walkPos([{ c: 0, r: 3, nc: 0, nr: 2 }], 0, 1, clamp(t - 4.6, 0, 1), 1); pip(m.x, m.y, 1, t);
     if (t > 5.6) { ghost(0, 3, 'from', clamp(t - 5.6), { dy: 22, s: 13 }); ghost(0, 2, 'landed', clamp(t - 5.9), { dy: -20, s: 13 }); }
     card(['Bellman: average over all moves of', '      reward + 0.9 × value where it lands', { s: 'TD: only the move Pip really made', c: C.goldInk }, { s: '      target = reward + 0.9 × guess where it landed', c: C.goldInk }], t - 1.5, { gap: [1, 1, 3.2, 3.2][0] * 1.4, fs: 16, lh: 22 }); } },
 { h: 'Walk 1: no news yet', p: ['Pip walks its habit. After each step it updates the guess of the cell it just **left**. Step 1: reward 0, and the landing cell’s guess is 0, so the target is 0 + 0.9 × 0 = 0. The old guess was 0 too: no change.', `Same for each of the first ${w1.length - 1} steps: no reward yet, so no news to learn from.`],
   say: [[["Here comes Pip's first walk.", [40, 16, 320, 256]], ["Watch the highlighted cell.", [40, 16, 320, 256]], ["After every step, Pip updates the cell it just left.", [40, 16, 320, 256]]], ["But nothing has happened yet.", ["No reward, and every guess is still zero.", [30, 292, 340, 64]], ["So the target is zero, and nothing changes.", [30, 292, 340, 64]]], [["Six steps in a row, and the notebook stays blank.", [40, 16, 320, 256]]]],
   shot: [2.5, 7],
   draw(t) { const n = w1.length - 1, m = walkPos(w1, 0, n, Math.max(0, t - .4), .9), k = Math.min(n, m.k + (m.done ? 0 : 0));
     trail(w1.slice(0, Math.min(n, m.k + 1)).map(s => [s.c, s.r])); const s = w1[Math.max(0, Math.min(n - 1, m.done ? n - 1 : m.k - (m.u < .6 ? 1 : 0)))];
     maze({ V: ZERO(), t, hi: t > 1.2 ? [[s.c, s.r, C.teal, .22]] : null }); nums(ZERO()); pip(m.x, m.y, m.face, t);
     if (t > 1.2) card(['reward 0  +  0.9 × 0   =   target 0', 'old guess 0   →   stays 0'], t - 1.2, { gap: 1 });
     caption(`walk 1, step ${clamp(m.done ? n : m.k + (m.u > .6 ? 1 : 0), 1, n)} of ${w1.length}`); } },
 { h: 'Cheese!', p: [`On step ${w1.length}, Pip steps from the cell left of the cheese onto it: reward +1, and the walk ends, so there is no landing cell to add. The target is simply 1. The old guess was 0.`, 'The gap, target minus old guess, is the **TD error**: how surprised Pip is. Here the surprise is 1.'],
   say: [[["Watch the last step.", [40, 16, 320, 256]], ["Pip moves from the cell beside the cheese, onto it.", [40, 16, 320, 256]], ["Plus one!", [40, 16, 320, 256]]], [["The walk is over, so there's nothing to add. The target is just one.", [30, 292, 340, 87]], ["Pip's old guess was zero.", [30, 292, 340, 87]]], [["That gap, between what happened and what Pip expected, is the TD error. Think of it as surprise. Here, the surprise is a full one.", [30, 292, 340, 87]]]],
   shot: [3, 8],
   draw(t) { trail(w1.map(s => [s.c, s.r]), .3); maze({ V: ZERO(), t, hi: t > 2 ? [[LAST1.c, LAST1.r, C.gold, .3]] : null }); nums(ZERO());
     const m = walkPos(w1, w1.length - 1, w1.length, Math.max(0, t - .6), 1); pip(m.x, m.y, 1, t); bubble('+1', cx(4), cy(0) - 18, t - 1.4);
     if (t > 2.2) card(['target  =  1   (walk over: nothing to add)', 'old guess  =  0', { s: `TD error  =  1 − 0  =  ${f2(LAST1.err).replace('.00', '')}     surprise!`, c: C.goldInk }], t - 2.2, { gap: 1.3 }); } },
 { h: 'A fifth of the way', p: [`Pip does not jump to 1. It moves its guess a fifth of the way: 0 + 0.2 × 1 = ${f2(LAST1.nw)}. That 0.2 is **α (alpha)**, the **learning rate**. It is unrelated to the one random move in five; the numbers just match.`, 'Why so cautious? The next walk might end on the cat. Small steps let many walks average out.'],
   say: [[["Watch the cell beside the cheese. Pip doesn't jump to one.", [40, 16, 320, 256]]], [["It moves just a fifth of the way, to zero point two.", [40, 16, 320, 256]], ["That fraction is alpha, the learning rate.", [30, 292, 340, 87]], "The match with one move in five is a coincidence."], [["Why so cautious? The next walk might end on the cat.", [40, 16, 320, 256]], "Small steps let the good and bad walks average out."]],
   shot: [3, 9],
   draw(t) { const V = ZERO(); V[K30] = LAST1.nw * ease((t - 4.6) / 1.2); maze({ V, t, hi: [[3, 0, C.gold, .3]] }); nums(V); pip(cx(4), cy(0), 1, t);
     card(['new guess  =  old  +  α × TD error', `                =  0  +  0.2 × 1  =  ${f2(LAST1.nw)}`, { s: 'α = 0.2: move one fifth of the way', c: C.soft }], t - .5, { gap: 1.5 }); } },
 { h: 'The news travels back', p: [`Walk 2: Pip enters the cell left of the cheese, worth ${f2(S2.next)}. Target: 0 + 0.9 × ${f2(S2.next)} = ${fv(S2.target)}, so its old cell moves a fifth of the way: ${fv(S2.nw)}. A guess learned from a guess: **bootstrapping**, TD’s heart.`, `Then a random wall bump: target 0.9 × ${f2(REST2[0].old)} = ${f2(REST2[0].target)}, so the guess dips to ${REST2[0].nw.toFixed(3)}; the cheese lifts it to ${f2(LAST2.nw)}.`],
   say: [[["Walk two, sped up.", [40, 16, 320, 256]], ["Watch Pip step beside the cheese.", [40, 16, 320, 256]]], [["That cell already reads zero point two.", [40, 16, 320, 256]], ["So the cell Pip just left gets a little credit too.", [40, 16, 320, 256]], ["A guess learned from a guess. That's bootstrapping, the heart of TD.", [30, 292, 340, 110]]], [["Then a bump into the wall, and a small dip.", [30, 292, 340, 110]], ["Then the cheese lifts the guess to zero point three six.", [30, 292, 340, 110]]]],
   shot: [5, 14],
   draw(t) { const fast = .25, t1 = LEAK * fast, t2 = t1 + 6, D = 2.2; let m, V = Vat(E, 1, 0), hi = null;
     if (t < t1) m = walkPos(w2, 0, LEAK, t, fast);
     else if (t < t2) { m = walkPos(w2, LEAK, LEAK + 1, Math.min(1, t - t1), 1); V = Vat(E, 1, LEAK); V[K20] = lerp(S2.old, S2.nw, ease((t - t1 - 4.4) / 1)); hi = t > t1 + 1 ? [[2, 0, C.teal, .22], [3, 0, C.gold, .25]] : null; }
     else { const k = clamp(Math.floor((t - t2) / D), 0, REST2.length - 1), q = REST2[k], u = t - t2 - k * D; m = walkPos(w2, LEAK + 1 + k, LEAK + 2 + k, Math.min(1, u * 1.4), 1);
       V = Vat(E, 1, LEAK + 1 + k); V[K30] = lerp(q.old, q.nw, ease((u - .9) / .8)); hi = [[3, 0, C.gold, .3]]; if (q.rew > 0) bubble('+1', cx(4), cy(0) - 18, u - .7); else if (u < 1.2) txt('bump!', cx(3), cy(0) - 24, { s: 15, c: C.roseInk, a: clamp(u * 2) }); }
     trail(w2.slice(0, t < t1 ? m.k + 1 : LEAK + 1).map(q => [q.c, q.r]), .4);
     maze({ V, t, hi }); nums(V); pip(m.x, m.y, m.face, t);
     if (t < t1) { caption('walk 2 (sped up)'); return; }
     const lines = [`reward 0  +  0.9 × ${f2(S2.next)}   =   target ${fv(S2.target)}`, `new  =  0 + 0.2 × (${fv(S2.target)} − 0)  =  ${fv(S2.nw)}`];
     REST2.forEach((q, k) => { if (t > t2 + k * D + .9) lines.push({ s: q.rew > 0 ? `cheese:  ${q.old.toFixed(3)} + 0.2 × (1 − ${q.old.toFixed(3)})  =  ${f2(q.nw)}` : `bump:  target 0.9 × ${f2(q.old)} = ${f2(q.target)},  guess → ${q.nw.toFixed(3)}`, c: C.goldInk }); });
     card(lines, t - t1 - 1.1, { gap: 1.1 }); } },
 { h: 'How surprised is Pip?', p: [`At each cheese, split the +1: what Pip **expected**, its guess left of the cheese, and the surprise, the TD error. Walk 1: expected 0, surprise 1. Walk 2: expected ${fv(EXP[1])}, surprise ${f2(SUR[1])}.`, `On walk ${CAT1 + 1}, a random move there points into the cat: TD error ${f2(SUR[CAT1])}; the guess drops to ${fv(V30[CAT1])}. On walk ${CAT1 + 2} the cheese surprises again: ${f2(SUR[CAT1 + 1])}. Walk ${CATS[1] + 1} brings another cat.`],
   say: [[["At each cheese, split the plus one in two.", [40, 286, 190, 114]], ["Yellow is what Pip expected.", [40, 286, 190, 114]], ["Orange is the surprise.", [40, 286, 190, 114]]], [["Watch the orange shrink.", [236, 300, 144, 96]], ["Then, on walk three, a random move sends Pip into the cat.", [40, 16, 320, 256]], ["A big negative surprise, and the guess drops.", [236, 300, 144, 96]]], [["So the next cheese surprises Pip all over again.", [236, 300, 144, 96]]]],
   shot: [5, 13.5],
   draw(t) { const D = 1.1, w = clamp(Math.floor(t / D), 0, 11), cat = E[w].end < 0, last = lastOf(w);
     maze({ V: Vat(E, w, E[w].tr.length - 1), t, hi: [[3, 0, C.gold, .25]] }); nums(Vat(E, w, E[w].tr.length - 1)); pip(cx(last.c), cy(last.r), 1, t); ghost(3, 0, 'expected', 1, { dy: 20, s: 12, c: C.goldInk });
     if (cat) { bubble('−1', cx(3), cy(1) - 18, (t % D)); txt('random move: down', cx(3), cy(0) + 36, { s: 13, c: C.roseInk }); } else bubble('+1', cx(4), cy(0) - 18, (t % D));
     txt(`walk ${w + 1}`, 40, 296, { s: 16, al: 'left' });
     const bx = 44, by = 392, bh = 76; rrect(bx, by - bh, 34, bh, 990, { c: C.line, w: 1 });
     if (!cat) { const sp = SUR[w], ex = EXP[w]; box(bx + 2, by - ex * bh, 30, ex * bh - 1, C.yellow); box(bx + 2, by - bh + 1, 30, sp * bh - 1, C.orange);
       txt(`surprise ${f2(sp)}`, bx + 42, by - bh + 10, { s: 15, al: 'left', c: '#c4581f' }); txt(`expected ${fv(ex)}`, bx + 42, by - 10, { s: 15, al: 'left', c: C.goldInk }); txt('the +1', bx + 17, by + 10, { s: 12, c: C.soft }); }
     else { txt('cat! no cheese', bx + 42, by - bh + 10, { s: 15, al: 'left', c: C.roseInk }); txt(`TD error ${f2(SUR[w])}`, bx + 42, by - 10, { s: 15, al: 'left', c: C.roseInk }); }
     const x0 = 246, x1 = 372, y0 = 392, y1 = 318, X = i => x0 + i * (x1 - x0) / 11, Y = v => y0 - v * (y0 - y1);
     rline(x0 - 6, y0, x1 + 4, y0, 991, { c: C.line, w: 1 }); txt('1', x0 - 14, y1, { s: 12, c: C.soft }); txt('0', x0 - 14, y0, { s: 12, c: C.soft }); txt('surprise, walk by walk', x0 - 10, y1 - 14, { s: 13, al: 'left', c: C.soft });
     for (let i = 0; i <= w; i++) { if (E[i].end < 0) { rline(X(i), y0, X(i), y1, 992 + i, { c: C.rose, w: 1.2, once: true, j: .5 }); continue; } dot(X(i), Y(SUR[i]), 3.4, C.orange); } } },
 { h: `${NWALK} walks`, p: [`Pip walks ${NWALK} times. The chart: the guess left of the cheese after each walk, 1 to ${NWALK}, left to right. Dashed: the true value from the sweeps, ${f2(EXACT[K30])}. Rose ticks: walks ending on the cat.`, `The guess climbs fast, then hovers around the dashed line, dropping after each cat scare. After 50 walks the start already reads ${f2(VS[49])}; its true value is ${f2(EXACT[SK])}.`],
   say: [[["Now Pip walks two hundred times.", [40, 286, 330, 112]], ["Look at the start cell around walk fifty. It already reads zero point three five, and the truth is zero point four.", [40, 16, 320, 256]]], [["Gold is the guess beside the cheese.", [40, 286, 330, 112]], ["Dashed is the true value, zero point eight seven.", [40, 286, 330, 112]]], [["The gold line hovers, dipping at every rose tick, every cat.", [40, 286, 330, 112]]]],
   shot: [5, 14],
   draw(t) { const n = t < 4 ? clamp(Math.floor((t - 1.5) * 20), 0, 50) : t < 5.5 ? 50 : clamp(50 + Math.floor((t - 5.5) * 20), 50, NWALK); maze({ V: n ? E[n - 1].V : ZERO(), t, hi: [[3, 0, C.gold, .2]] }); nums(n ? E[n - 1].V : ZERO()); pip(cx(0), cy(3), 1, t);
     lineChart({ x0: 56, x1: 340, y0: 388, y1: 306, n, series: [{ v: V30, col: '#c99a3a' }], exact: [{ v: EXACT[K30], label: f2(EXACT[K30]), dy: -6 }], cats: true, title: 'guess left of the cheese, after each walk' });
     if (n === 50 && t < 5.5) box(cx(0) - 30, cy(3) - 30, 60, 60, C.teal, .25);
     caption(n ? (n === 50 && t < 5.5 ? `after walk 50: the start reads ${f2(VS[49])}` : `after walk ${n}`) : '', 1); } },
 { h: 'Close, but never still', p: [`Left of the cheese after walk ${NWALK}: TD ${f2(FIN[K30])}, sweeps ${f2(EXACT[K30])}. Why higher? Since the last cat scare, walk ${LC + 1}, ${NWALK - LC - 1} good walks pulled it toward 1. Rare cat scares hit hard: a cluster once dragged it to ${f2(RG2[0])}. Its last-100-walk average: ${f2(MEAN2)}.`, 'A fixed α keeps guesses wobbling around the true values. The right half is off too, for another reason, next.'],
   say: [["Here are the two notebooks side by side.", ["TD on the left.", [8, 20, 190, 176]], ["The exact sweeps on the right.", [202, 20, 190, 176]]], [["Beside the cheese, TD reads zero point nine eight.", [8, 20, 190, 176]], "Why so high?", ["The last cat scare was fifteen walks ago, and every good walk since nudged it up.", [30, 226, 340, 87]]], [["A fixed alpha never lets the guesses settle. They keep wobbling around the truth.", [30, 226, 340, 87]]]],
   shot: [5, 12],
   draw(t) { const a = clamp(t - .3), b = clamp(t - 1.4), hl = Gx => [[3, 0, C.gold, .3]].map(([c, r, col, aa]) => box(Gx.ox + c * Gx.cs + 2, Gx.oy + r * Gx.cs + 2, Gx.cs - 4, Gx.cs - 4, col, aa));
     alpha(a, () => { hl(GA); GA.maze({ V: FIN, t }); nums(FIN, 1, GA, 12); txt(`TD, after walk ${NWALK}`, GA.ox + 95, 28, { s: 15, c: C.goldInk }); });
     alpha(b, () => { hl(GB); GB.maze({ V: EXACT, t }); nums(EXACT, 1, GB, 12); txt('sweeps (exact)', GB.ox + 95, 28, { s: 15, c: '#3f7f7c' }); });
     if (t > 2.5) card([`left of the cheese, walk ${LC + 1}: cat  →  ${f2(V30[LC])}`, `15 good walks later  →  ${f2(FIN[K30])}`.replace('15', NWALK - LC - 1), { s: `true ${f2(EXACT[K30])};  last 100 walks average ${f2(MEAN2)}`, c: C.soft }], t - 2.5, { y: 226, gap: 1.6 }); } },
 { h: 'Monte Carlo, same walks', p: [`Monte Carlo’s walks began with these same ${NWALK}. Averaging real returns ends closer almost everywhere: on Pip’s route within ${ERR.mcRmax < .01 ? '0.01' : f2(ERR.mcRmax)} of the sweeps, against ${f2(ERR.tdRmax)} for TD; in rarely visited cells ${f2(ERR.mcO)} off on average, against ${f2(ERR.tdO)}.`, 'Every walk here ends within a few steps, so waiting costs little and Monte Carlo wins. TD pays off when walks are very long or never end.'],
   say: [[["Now Monte Carlo, on the left, trained on these very same walks.", [8, 20, 190, 176]]], [["Monte Carlo ends closer almost everywhere.", [30, 226, 340, 87]], ["Along Pip's usual route, it's within a hundredth of the truth.", [30, 226, 340, 87]], ["In rarely visited cells, TD is far off.", [30, 226, 340, 87]]], ["Why? These walks are short, so waiting for the end costs almost nothing. TD shines when walks are long, or never end."]],
   shot: [4, 11],
   draw(t) { const a = clamp(t - .3), b = clamp(t - 1.4);
     alpha(a, () => { GA.maze({ V: MC.V, t }); nums(MC.V, 1, GA, 12); txt(`Monte Carlo, walk ${NWALK}`, GA.ox + 95, 28, { s: 15, c: C.roseInk }); });
     alpha(b, () => { GB.maze({ V: FIN, t }); nums(FIN, 1, GB, 12); txt(`TD, walk ${NWALK}`, GB.ox + 95, 28, { s: 15, c: C.goldInk }); });
     if (t > 2.5) card(['distance from the sweeps:', `usual route:  MC within ${ERR.mcRmax < .01 ? '0.01' : f2(ERR.mcRmax)},  TD up to ${f2(ERR.tdRmax)}`, `rarely visited:  MC ${f2(ERR.mcO)},  TD ${f2(ERR.tdO)}  (average)`], t - 2.5, { y: 226, gap: 1.4 }); } },
 { h: 'Why TD lags off the route', p: [`Not for lack of visits: right of the start, Monte Carlo had ${MC.N[K13]} returns, TD ${VISITS[K13]} updates. Every TD target there leaned on the next cell’s guess, never above ${fv(NEXT13)}: the cheese’s news hadn’t reached it.`, `Plug in true landing values: the same ${VISITS[K13]} updates give ${f2(TDOR[K13])}. Bootstrapping is fast on busy paths, slow through rarely visited cells. Small numbers: visits in ${NWALK} walks.`],
   say: [[["Look right of the start.", [40, 16, 320, 256]], ["The small numbers are visits.", [40, 16, 320, 256]], ["TD updated this cell nine times, more than Monte Carlo's eight returns.", [40, 16, 320, 256]]], [["But each update leaned on the next cell's guess, and that was still almost zero.", [40, 16, 320, 256]], ["The cheese's news never reached this far.", [30, 292, 340, 87]]], [["Give it the true values, and the same nine updates land on zero point four.", [30, 292, 340, 87]]]],
   shot: [4, 11],
   draw(t) { maze({ V: FIN, t, hi: [[1, 3, C.teal, .25], [2, 3, C.gold, .2 * clamp(t - 1.5)]] }); nums(FIN); pip(cx(1), cy(3), 1, t);
     if (t > 1.5) alpha(clamp(t - 1.5), () => edgeArrow(1, 3, 1, 1600, C.ink, 17));
     CELLS.forEach(([c, r], i) => { const u = clamp(t - 2.5 - i * .1); if (u > 0) txt(`${VISITS[key(c, r)]} visits`, cx(c), cy(r) + 20, { s: 12, c: VISITS[key(c, r)] < 20 ? C.roseInk : C.soft, a: u }); });
     if (t > 4.5) card([`right of the start: ${VISITS[K13]} TD updates,  next guess ≤ ${fv(NEXT13)}`, `→ TD ${fv(FIN[K13])}    (true ${f2(EXACT[K13])})`, { s: `true values plugged in: ${f2(TDOR[K13])}   ·   MC, ${MC.N[K13]} returns: ${f2(MC.V[K13])}`, c: C.soft }], t - 4.5, { gap: 1.3, fs: 16 }); } },
 { h: 'A smaller step', p: [`Same ${NWALK} walks at α = 0.05; only α differs, since the habit ignores the notebook. Blue, last 100 walks: ${f2(RG05[0])} to ${f2(RG05[1])}, against ${f2(RG2[0])} to ${f2(RG2[1])}. At walk 100 the start: only ${f2(VSS[99])}, against ${f2(VS[99])}.`, `Monte Carlo’s step, 1 ÷ n, shrinks too fast for TD: early zero-based targets keep their weight; after ${NWALK} walks the start reads only ${f2(TDINV[SK])}; it gets there, slowly.`],
   say: [["Same walks, smaller step.", ["Alpha is now zero point zero five.", [40, 286, 330, 112]], ["Watch the blue line grow over the pale gold one.", [40, 286, 330, 112]]], [["Blue is much calmer.", [40, 286, 330, 112]], ["But it's slow.", [40, 16, 320, 256]], "At walk one hundred, the start reads only zero point zero eight."], ["Monte Carlo's one over n step is slower still. It shrinks so fast that the early zeros linger."]],
   shot: [4, 11],
   draw(t) { const n = clamp(Math.floor((t - .8) * 30), 0, NWALK), u = clamp(t - 2.5);
     maze({ V: n ? ES[n - 1].V : ZERO(), t, hi: [[3, 0, C.blue, .2]] }); nums(n ? ES[n - 1].V : ZERO()); pip(cx(0), cy(3), 1, t);
     lineChart({ x0: 56, x1: 340, y0: 388, y1: 306, n: NWALK, series: [{ v: V30, col: '#e3c48a', w: 1.1 }, { v: V30S, col: '#4f7fb0', w: 1.8, n }], exact: [{ v: EXACT[K30], label: f2(EXACT[K30]), dy: -6 }], title: 'guess left of the cheese' });
     // the blue line is drawn progressively on top of the pale α = 0.2 line
     txt('α = 0.2', 300, 296, { s: 13, c: '#c99a3a', al: 'right' }); txt('α = 0.05', 360, 296, { s: 13, c: '#4f7fb0', al: 'right' });
     caption(n < NWALK ? `α = 0.05: after walk ${n}` : `α = 0.05 notebook after walk ${NWALK}`, u); } },
 { h: 'Your turn', ctl: { label: 'α', values: ALPHAS, def: 2, onChange: st => {} },
   p: [`Drag **α** to rerun the same ${NWALK} walks with another learning rate. Gold is the guess left of the cheese, teal the start; dashed lines are their true values.`, 'Watch how wildly the lines jump around the dashed ones, and how long teal takes to climb toward its dashed line. Big α learns fast but nervously; small α is calm but slow.'],
   say: [["Your turn. Drag the alpha slider, and the same two hundred walks replay with that learning rate."], [["Gold is beside the cheese.", [40, 16, 320, 256]], ["Teal is the start.", [40, 16, 320, 256]], ["Watch how far the lines jump around their dashed targets, and how long teal takes to climb.", [40, 286, 330, 112]]], ["Big alpha is quick but jumpy. Small alpha is calm, but slow."]],
   shot: [3, 9],
   draw(t, st) { const a = ALPHAS[st.ci ?? 2], Ea = runTD(a), prev = st.prevCi != null ? runTD(ALPHAS[st.prevCi]) : Ea, u = t < (st.changed ?? -9) ? 1 : ease((t - (st.changed ?? -9)) / .9);
     const V = Ea[NWALK - 1].V.map((v, i) => lerp(prev[NWALK - 1].V[i], v, u)); maze({ V, t }); nums(V); pip(cx(0), cy(3), 1, t);
     const tc = st.changed ?? 0, n = t < tc ? NWALK : clamp(Math.floor((t - tc - .3) * 60), 0, NWALK);
     lineChart({ x0: 56, x1: 340, y0: 390, y1: 308, lo: -.3, n, series: [{ v: Ea.map(e => e.V[K30]), col: '#c99a3a' }, { v: Ea.map(e => e.V[SK]), col: '#3f9a96', w: 1.6 }],
       exact: [{ v: EXACT[K30], label: f2(EXACT[K30]), dy: -6 }, { v: EXACT[SK], col: '#9fc9c7', label: f2(EXACT[SK]), lc: '#3f7f7c' }], title: `α = ${a}: after walk ${NWALK}, left of cheese ${f2(Ea[NWALK - 1].V[K30])}, start ${f2(Ea[NWALK - 1].V[SK])}` }); } },
 { h: 'Where you’ve seen this', p: [`Pip’s cheese surprise shrinks, jumping after cat scares. And it moves earlier: on walk 2, TD error ${fv(S2.err)} came a step early, entering a promising cell.`, 'In the 1990s Wolfram Schultz found monkey **dopamine** neurons firing at unexpected juice. Once a light reliably came first, they fired at the light, and dipped when promised juice didn’t come. TD powered the 1992 backgammon program TD-Gammon.'],
   say: [[["These dots are Pip's surprise at the cheese, across all two hundred walks. They sink, then jump after each cat.", [36, 284, 334, 132]]], ["Surprise also moves earlier.", ["On walk two, Pip got a small lift one step before the cheese.", [40, 16, 320, 256]]], ["Wolfram Schultz saw monkey dopamine neurons do the same. They fired for surprise juice, then for the light that announced it."]],
   shot: [4, 11],
   draw(t) { const n = Math.min(NWALK, Math.floor(t * 30)); maze({ V: FIN, t }); nums(FIN, .5); pip(cx(3), cy(0), 1, t); strip(n);
     txt('walk 1', 44, 396, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NWALK}`, 360, 396, { s: 12, c: C.soft, al: 'right' });
     txt('dots = surprise at the cheese · rose = cat', W / 2, 411, { s: 14, c: C.soft, a: n >= NWALK ? 1 : 0 }); } },
 { h: 'What TD cannot do yet', p: ['TD filled Pip’s notebook with how good each cell is **under its habit**, using only experience. But it never changed the habit.', 'To pick a better move from these numbers, Pip would need where each move leads and what it pays: the rulebook again. The fix, as in the Monte Carlo reel, is a number per move. Q-learning, next, learns those with TD’s one-step update.'],
   say: [[["So TD filled in the notebook from experience alone.", [40, 16, 320, 256]], ["But notice, the habit itself never changed.", [30, 292, 340, 87]]], [["Say Pip stands here and wants a better move.", [40, 16, 320, 256]], ["Where does each arrow lead? Watch the question marks.", [40, 16, 320, 256]], ["Without the rulebook, Pip can't tell.", [30, 292, 340, 87]]], [["The fix is a number for each move. That's Q-learning, coming next.", [30, 292, 340, 87]]]],
   shot: [3, 9],
   draw(t) { maze({ V: FIN, t, hi: [[2, 0, C.teal, .22]] }); nums(FIN, .6); pip(cx(2), cy(0), 1, t);
     [0, 1, 2, 3].forEach(m => { const u = clamp(t - .8 - m * .4); if (u > 0) { alpha(u, () => edgeArrow(2, 0, m, 1500 + m, C.ink, 17)); } });
     [[3, 0], [2, 1], [1, 0]].forEach(([c, r], i) => txt('?', cx(c), cy(r) + 22, { s: 20, c: C.roseInk, a: clamp(t - 2.6 - i * .3) }));
     card(['notebook: how good is this cell, under my habit', { s: 'better move?  need: where does each move lead?', c: C.roseInk }, { s: 'next: Q-learning, a number for each move', c: C.goldInk }], t - 3.6, { gap: 1.3 }); } },
 { h: 'Recap', p: ['**TD learning** estimates the values of the habit Pip follows, from experience alone, and updates after every step instead of waiting for the walk to end.', 'Its target is the Bellman rule with one real step in place of the average over moves. A fixed α makes the guesses hover around the true values; a smaller α wobbles less but learns slower.'],
   say: [["So that's TD learning.", ["Pip learns what its habit is worth from experience alone, updating after every single step.", [40, 16, 320, 256]]], [["The target is one real step: the reward, plus gamma times the guess where Pip landed.", [30, 292, 340, 110]], ["Alpha sets how far to move.", [30, 292, 340, 110]], "Big alpha learns fast and wobbles. Small alpha is steady, but slow."]],
   shot: [3, 9],
   draw(t) { maze({ V: FIN, t }); nums(FIN); arrows(.25); pip(cx(0), cy(3), 1, t);
     card(['target  =  reward  +  γ × guess where Pip landed', 'TD error  =  target − old guess', 'new guess  =  old  +  α × TD error', { s: 'walk over: target = just the reward', c: C.soft }], t - .3, { gap: 1 }); } },
];

RLStory.register('td', { title: 'Temporal-Difference Learning', track: 'Model-free', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { MC, TDOR, TDINV, NEXT13, ERR, OFF, MEAN2, WALKS, E, ES, EXACT, V30, VS, SUR, EXP, CATS, CAT1, LC, PC, REST2, VISITS, RG2, RG05, FIN, S2, LAST1, LAST2, runTD } });
})();
