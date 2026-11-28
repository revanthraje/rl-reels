// The Bellman equation as a story: Pip checks its walk-measured notebook against a one-step rule, then computes the
// notebook with no walking at all by sweeping that rule over every cell. Same maze, policy and seeds as values.js.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, key, maze, trail, walkPos, pip } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GAM = 0.9, EPS = 0.2, NW = 1000, SK = key(0, 3);
const ARR = new Array(20).fill(-1);
[[0, 3, 0], [0, 2, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [2, 1, 0], [1, 2, 3], [1, 3, 1], [2, 3, 1], [3, 3, 1], [4, 3, 0], [4, 2, 0], [4, 1, 0], [3, 2, 1]]
  .forEach(([c, r, m]) => { ARR[key(c, r)] = m; });
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);
const NAMES = ['up', 'right', 'down', 'left'];
const prob = (k, m) => EPS / 4 + (m === ARR[k] ? 1 - EPS : 0);          // 0.85 for the arrow, 0.05 for each other move
// ───── the walk notebook from the Value Functions reel (identical walks and seeds)
function walk(c, r, eps, R, keepTr) {
  const tr = []; let n = 0, f1 = null;
  for (let k = 0; k < 5000; k++) { let m, rnd = false; if (R() < eps) { rnd = true; m = Math.floor(R() * 4); } else m = ARR[key(c, r)];
    const [nc, nr] = G.step(c, r, m); if (keepTr) tr.push({ c, r, nc, nr, m, rnd }); if (!k) f1 = key(nc, nr); n++; c = nc; r = nr; const rw = G.term(c, r); if (rw) return { tr, n, f1, rew: rw, ret: rw * GAM ** (n - 1) }; }
  return { tr, n, f1, rew: 0, ret: 0 };
}
const NB = new Array(20).fill(0); let WALKSTEPS = 0, W1 = null, UPN = 0, UPS = 0;
CELLS.forEach(([c, r]) => { const R = rng(87000 + key(c, r)); let s = 0; for (let i = 0; i < NW; i++) { const w = walk(c, r, EPS, R, i === 0); s += w.ret; WALKSTEPS += w.n; if (i === 0 && c === 0 && r === 3) W1 = w; if (c === 0 && r === 3 && w.f1 === key(0, 2)) { UPN++; UPS += w.ret / GAM; } } NB[key(c, r)] = s / NW; });
// (from the start no first step can end the walk, so the rest's return is the whole return divided by 0.9)
// ───── sweeps of the Bellman rule (every cell uses the previous sweep's numbers)
function branches(V, c, r, opt) { const k = key(c, r); return [0, 1, 2, 3].map(m => { const [nc, nr] = G.step(c, r, m), rw = G.term(nc, nr), nv = rw ? 0 : V[key(nc, nr)]; return { m, p: opt ? null : prob(k, m), nc, nr, rw, nv, q: rw + GAM * nv }; }); }
function sweep(V, opt) { const N = V.slice(); CELLS.forEach(([c, r]) => { const b = branches(V, c, r, opt); N[key(c, r)] = opt ? Math.max(...b.map(x => x.q)) : b.reduce((s, x) => s + x.p * x.q, 0); }); return N; }
const HIST = [new Array(20).fill(0)]; for (let i = 1; i <= 60; i++) HIST.push(sweep(HIST[i - 1]));
const OPT = [new Array(20).fill(0)]; for (let i = 1; i <= 10; i++) OPT.push(sweep(OPT[i - 1], true));
const EX = HIST[60], maxDiff = (a, b) => Math.max(...CELLS.map(([c, r]) => Math.abs(a[key(c, r)] - b[key(c, r)])));
const SETTLE2 = HIST.findIndex((V, i) => i > 0 && CELLS.every(([c, r]) => V[key(c, r)].toFixed(2) === EX[key(c, r)].toFixed(2)));
const SETTLE3 = HIST.findIndex((V, i) => i > 0 && maxDiff(V, HIST[i - 1]) < .001);
const WGAP = maxDiff(NB, EX);
const OPTN = OPT.findIndex((V, i) => i > 0 && maxDiff(V, OPT[i - 1]) < 1e-12) - 1;   // last sweep that changed anything
// ───── formatting: hand-checks use the 2-decimal numbers shown on screen
const f2 = v => { const s = (Math.abs(v) < .005 ? 0 : v).toFixed(2); return s.replace('-', '−'); };
const r2 = v => +(+v).toFixed(2);
const ZERO = () => new Array(20).fill(0);
function nums(V, a = 1, Gx = G, s = 14, only) {
  for (const [c, r] of CELLS) { if (only && !only(c, r)) continue; const aa = typeof a === 'function' ? a(c, r) : a; if (aa <= 0) continue; const v = V[key(c, r)];
    txt(f2(v), Gx.ox + c * Gx.cs + Gx.cs - 5, Gx.oy + r * Gx.cs + (s >= 14 ? 12 : 10), { s, al: 'right', a: aa, c: Math.abs(v) < .005 ? C.soft : v < 0 ? C.roseInk : C.goldInk }); }
}
function arrows(a = 1, opt) { CELLS.forEach(([c, r]) => { const m = opt ? opt(c, r) : ARR[key(c, r)]; alpha(a, () => G.arrowIn(c, r, m, 400 + key(c, r), { c: C.goldInk, w: 1.5 })); }); }
function ghost2(c, r, str, a) { box(cx(c) - 16, cy(r) + 13, 32, 17, C.card, .9 * a); txt(str, cx(c), cy(r) + 21, { s: 13, c: '#3f7f7c', a }); }
function edgeArrow(c, r, m, seed, col, r0 = 12) { const [dx, dy] = G.MOVES[m]; arrow(cx(c) + dx * r0, cy(r) + dy * r0, cx(c) + dx * 29, cy(r) + dy * 29, seed, { c: col, w: 1.6, head: 6 }); }
function animWalk(w, t, dur, upto) { const n = upto ?? w.tr.length, m = walkPos(w.tr, 0, n, Math.max(0, t), dur); trail(w.tr.slice(0, Math.min(n, m.k + 1)).map(s => [s.c, s.r]), .55); pip(m.x, m.y, m.face, t); return m; }
// the four branches from one cell as card lines (numbers as displayed)
function branchLines(V, c, r, o = {}) {
  const b = branches(V, c, r), lines = [];
  [1, 0, 2, 3].forEach(m => { const x = b[m], nv = r2(x.nv), q = r2(x.rw + GAM * nv), stay = x.nc === c && x.nr === r;
    const what = x.rw > 0 ? 'cheese' : x.rw < 0 ? 'cat' : stay ? 'bump, stays' : 'next cell';
    lines.push({ s: `${NAMES[m]} ${x.p.toFixed(2)}:  ${x.rw > 0 ? '+1' : x.rw < 0 ? '−1' : '0'} + 0.9 × ${f2(nv)} = ${f2(q)}   (${what})`, c: x.rw < 0 ? C.roseInk : C.ink }); });
  const tot = b.reduce((s, x) => s + x.p * r2(x.rw + GAM * r2(x.nv)), 0);
  return { lines, tot, b };
}
function weightedLine(b) { const parts = [1, 0, 2, 3].map(m => `${b[m].p.toFixed(2)} × ${f2(r2(b[m].rw + GAM * r2(b[m].nv)))}`); return parts; }

// the two notebooks side by side
const GA = L.gridWorld({ cs: 38, ox: 8, oy: 44 }), GB = L.gridWorld({ cs: 38, ox: 202, oy: 44 });
// chart: value of the start cell after each sweep
function sweepChart(n, t) {
  const x0 = 66, x1 = 366, yb = 388, yt = 304, X = i => x0 + i / 20 * (x1 - x0), Y = v => yb - v / .5 * (yb - yt);
  rline(x0, yb, x1, yb, 960, { c: C.line, w: 1 }); rline(x0, yb, x0, yt - 4, 961, { c: C.line, w: 1 });
  txt('0', x0 - 10, yb, { s: 12, c: C.soft }); txt('0.5', x0 - 14, yt, { s: 12, c: C.soft }); txt('value of start', x0 + 4, yt - 12, { s: 13, c: C.soft, al: 'left' });
  txt('0', x0, yb + 11, { s: 12, c: C.soft }); txt('20', x1, yb + 11, { s: 12, c: C.soft }); txt('sweep number', (x0 + x1) / 2, yb + 11, { s: 12, c: C.soft });
  if (t > 1.5) { ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.grey; ctx.beginPath(); ctx.moveTo(x0, Y(EX[SK])); ctx.lineTo(x1, Y(EX[SK])); ctx.stroke(); ctx.restore(); txt(`settles at ${f2(EX[SK])}`, x1, Y(EX[SK]) - 10, { s: 13, c: C.soft, al: 'right' }); }
  for (let i = 0; i <= Math.min(n, 20); i++) { if (i) rline(X(i - 1), Y(HIST[i - 1][SK]), X(i), Y(HIST[i][SK]), 970 + i, { c: '#c99a3a', w: 1.3, once: true, j: .3 }); dot(X(i), Y(HIST[i][SK]), 3.2, C.gold); }
}
const SLV = [0, 1, 2, 3, 5, 7, 10, 13, 20];
function tapCell(x, y, st) { const c = Math.floor((x - OX) / CS), r = Math.floor((y - OY) / CS); if (c < 0 || c >= 5 || r < 0 || r >= 4 || G.isRock(c, r) || G.term(c, r)) return; st.cell = [c, r]; }

const S30 = branchLines(NB, 3, 0), T03 = GAM * (.85 * r2(NB[key(0, 2)]) + .05 * r2(NB[key(1, 3)]) + .10 * r2(NB[SK]));
const REST = W1.n - 1, RESTRET = GAM ** (REST - 1);
const STEPS = [
 { h: 'A shortcut?', p: [`Here is Pip’s notebook from the last reel: for each cell, the average return of 1000 walks that start there and follow policy A with one move in five random, on the dry floor. Measuring it took ${WALKSTEPS.toLocaleString('en-US')} steps of walking.`, 'Neighbouring numbers look related. If the value of a cell could be worked out from its neighbours, Pip could skip most of that walking.'],
   shot: [3, 9],
   say: [[["Here's Pip's notebook from the last reel.", [40, 16, 320, 256]], ["A thousand walks from every cell.", [40, 16, 320, 256]]], [["That took about seventy six thousand steps of walking.", [30, 292, 340, 87]]], [["But look at the neighbours. Their numbers seem related.", [40, 16, 320, 256]], "Could a cell's value be worked out from them?"]],
   draw(t) { maze({ V: NB, t }); nums(NB, (c, r) => clamp((t - .4 - key(c, r) * .06) * 3)); arrows(.25); pip(cx(0), cy(3), 1, t);
     if (t > 3.5) card(['notebook: average return of 1000 walks per cell', `policy A, 1 move in 5 random  ·  dry floor  ·  γ = 0.9`, { s: `cost: ${WALKSTEPS.toLocaleString('en-US')} steps of walking`, c: C.soft }], t - 3.5, { gap: 1 }); } },
 { h: 'Split a walk in two', p: [`One walk from the start: cheese on step ${W1.n}, return ${f2(W1.ret)}. Cut after the first step, the rest is its own walk from the cell above, with return 0.9 multiplied ${REST - 1} times: ${f2(RESTRET)}.`, `Seen from the start, everything after step 1 counts 0.9 times less. Return = first reward + 0.9 × return of the rest = 0 + 0.9 × ${f2(RESTRET)} = ${f2(W1.ret)}.`],
   shot: [4, 12],
   say: [[["Watch one walk from the start.", [40, 16, 320, 256]], ["Cheese on step seven. Return, zero point five three.", [40, 16, 320, 256]]], [["Now cut it after the first step.", [40, 16, 320, 256]], ["The rest is its own walk, from the cell above, worth zero point five nine.", [30, 292, 340, 87]]], [["The whole walk is just the first reward, plus zero point nine times the rest.", [30, 292, 340, 87]]]],
   draw(t) { maze({ t, hi: t > 1.2 ? [[0, 3, C.teal, .22], [0, 2, C.gold, .22]] : null }); arrows(.2); const m = animWalk(W1, t - .5, .5);
     if (t > 1.2) { txt('step 1', cx(0) + 2, cy(3) + 22, { s: 13, c: '#3f7f7c', a: clamp(t - 1.2) }); }
     if (t > 4.5) { trail(W1.tr.slice(1).map(s => [s.c, s.r]), .9); txt('the rest', cx(1) - 4, cy(0) + 22, { s: 13, c: C.goldInk, a: clamp(t - 4.5) }); }
     if (t > 5) card([`whole walk: cheese on step ${W1.n}  →  ${f2(W1.ret)}`, `the rest: ${REST} steps from the cell above  →  ${f2(RESTRET)}`, { s: `${f2(W1.ret)}  =  0  +  0.9 × ${f2(RESTRET)}`, c: C.goldInk }], t - 5, { gap: 1.4 }); } },
 { h: 'Average both sides', p: ['The split holds for every walk, so on average too. The rest is a fresh walk from the landing cell: Pip’s rule ignores the past, the Markov property.', `Check: ${UPN} walks went up first, about 850 expected. Their rests averaged ${f2(UPS / UPN)}, near that cell’s value, ${f2(NB[key(0, 2)])}. So value here = average of [ reward + 0.9 × value where Pip lands ]: the **Bellman equation**.`],
   shot: [4, 12],
   say: [[["That split works for every walk, so it works on average too.", [40, 16, 320, 256]], ["The rest is a fresh walk from wherever Pip lands.", [40, 16, 320, 256]]], [["Walks that went up first averaged zero point four five afterwards, close to that cell's value.", [30, 292, 340, 110]]], [["So, value here is the average of the reward plus zero point nine times the value where Pip lands. That's the Bellman equation.", [30, 292, 340, 110]]]],
   draw(t) { maze({ V: NB, t, hi: [[0, 3, C.teal, .22], [0, 2, C.gold, .3 * clamp(t - 2)]] }); nums(NB, (c, r) => (c === 0 && r === 2) ? 1 : .35); arrows(.2); pip(cx(0), cy(3), 1, t);
     if (t > 2) alpha(clamp(t - 2), () => edgeArrow(0, 3, 0, 1600, C.ink, 17));
     card(['every walk:  return = first reward + 0.9 × rest', { s: `${UPN} walks went up first: rests averaged ${f2(UPS / UPN)}`, c: C.ink }, { s: `value of the cell above: ${f2(NB[key(0, 2)])}`, c: C.goldInk }, { s: 'V(here) = average of [ reward + 0.9 × V(landing) ]', c: C.goldInk }], t - .5, { gap: 1.8 }); } },
 { h: 'What exactly is averaged?', p: ['Two kinds of chance. Pip’s own: which move, π(a | s): 0.85 for the arrow, 0.05 each other. The floor’s: where it lands, P(s′ | s, a).', 'Today’s dry floor sends each move to one known cell. On the mdp reel’s wet floor, a move goes ahead 8 times in 10, each side once in 10, and the rule averages over that too.'],
   shot: [4, 12],
   say: ["So what exactly gets averaged? Two kinds of chance.", [["Pip's own. Which move it picks, mostly the arrow.", [40, 16, 320, 256]]], [["And the floor's. Where each move lands.", [30, 292, 340, 110]], ["Today's floor is dry, so each move reaches one known cell.", [30, 292, 340, 110]]]],
   draw(t) { maze({ V: NB, t, hi: [[0, 3, C.teal, .22]] }); nums(NB, .35); arrows(.15); pip(cx(0), cy(3), 1, t);
     alpha(clamp(t - .8), () => { edgeArrow(0, 3, 0, 1600, C.ink, 17); edgeArrow(0, 3, 1, 1601, C.ink, 17); txt('0.85', cx(0) - 16, cy(3) - 22, { s: 13, c: C.ink }); txt('0.05', cx(0) + 30, cy(3) + 16, { s: 13, c: C.ink }); });
     card(['V(s) = Σ over moves a:  π(a | s) ×', '    Σ over landings s′:  P(s′ | s, a) × [ r + γ V(s′) ]', { s: 'dry floor: P = 1 for the one cell each move reaches', c: C.goldInk }, { s: 'wet floor: 0.8 ahead, 0.1 to each side', c: C.soft }], t - 2, { gap: 1.4, fs: 16 }); } },
 { h: 'Where can Pip land?', p: ['Take the cell beside the cheese. The arrow says right: 85 times in 100, 80 by habit and 5 by chance. Each other move: 5 times in 100.', 'Right: the cheese, +1, and the walk ends, so the next value is 0. Down: the cat, −1, also 0 after. Up bumps the wall: Pip stays, reward 0. Left: one cell back, reward 0.'],
   shot: [3, 11],
   say: [[["Take the cell beside the cheese.", [40, 16, 320, 256]], ["Watch the four moves appear, with their chances.", [40, 16, 320, 256]], ["Right, eighty five in a hundred. Each other move, five.", [40, 16, 320, 256]]], [["Right lands on the cheese, and down on the cat. Both end the walk.", [30, 292, 340, 87]], ["Up bumps the wall. Left goes one cell back.", [30, 292, 340, 87]]]],
   draw(t) { maze({ V: NB, t, hi: [[3, 0, C.teal, .2]] }); nums(NB, .4); pip(cx(3), cy(0), 1, t);
     const pos = [[cx(3), 7], [cx(4), cy(0) + 22], [cx(3), cy(1) + 25], [cx(2), cy(0) + 20]], lab = ['up 0.05: bump', 'right 0.85', 'down 0.05', 'left 0.05'];
     [1, 2, 0, 3].forEach((m, i) => { const u = clamp(t - .6 - i * 1.1); if (u <= 0) return; alpha(u, () => edgeArrow(3, 0, m, 1500 + m, m === 2 ? C.roseInk : C.ink, 17)); const [x, y] = pos[m], wd = lab[m].length * 6.4 + 10; box(x - wd / 2, y - 8, wd, 17, C.card, .92 * u); txt(lab[m], x, y, { s: 14, c: m === 2 ? C.roseInk : C.ink, a: u }); });
     if (t > 5.5) card([{ s: 'right: +1, walk ends (value after: 0)', c: C.goldInk }, { s: 'down: −1, walk ends (value after: 0)', c: C.roseInk }, 'up: bump, stays here  ·  left: next cell back'], t - 5.5, { gap: 1.1, fs: 16 }); } },
 { h: 'Check it with real numbers', p: [`Each move scores its reward plus 0.9 × the notebook value where it lands: on a dry floor, exactly its Q. Up stays here, worth ${f2(NB[key(3, 0)])}; left lands on ${f2(NB[key(2, 0)])}.`, `The last reel’s forced walks measured these Qs as 0.80 and 0.69, with a little noise. Weighted by how often each move happens: ${f2(S30.tot)}. The notebook says ${f2(NB[key(3, 0)])}: to two decimals, the rule holds.`],
   shot: [4, 12],
   say: [[["Now plug in real numbers.", [30, 288, 340, 130]], ["Each move scores its reward, plus zero point nine times the value where it lands.", [30, 288, 340, 130]]], [["Weight them by how often each move happens.", [30, 288, 340, 130]], ["Zero point eight seven.", [30, 288, 340, 130]], ["And the notebook says zero point eight seven. The rule holds.", [40, 16, 320, 256]]]],
   draw(t) { maze({ V: NB, t, hi: [[3, 0, C.teal, .2], [2, 0, C.gold, .25 * clamp(t - 1)]] }); nums(NB); pip(cx(3), cy(0), 1, t); [0, 1, 2, 3].forEach(m => edgeArrow(3, 0, m, 1500 + m, m === 2 ? C.roseInk : C.ink, 17));
     const L5 = S30.lines.concat([{ s: `0.85 × 1 + 0.05 × (${f2(r2(GAM * r2(NB[key(3, 0)])))} − 1 + ${f2(r2(GAM * r2(NB[key(2, 0)])))}) = ${f2(S30.tot)}  ✓`, c: C.goldInk }]);
     card(L5, t - .5, { gap: 1.3, fs: 15, lh: 20, y: 288 }); } },
 { h: 'Another cell', p: [`The start: up, 85 times in 100, lands on ${f2(NB[key(0, 2)])}; right on ${f2(NB[key(1, 3)])}. Down and left bump, so Pip stays on ${f2(NB[SK])}. All rewards are 0, so the rule is just 0.9 × the average landing value.`, `The rule gives ${f2(T03)}, the walks gave ${f2(NB[SK])}. Close, not equal: even 1000 walks leave a little noise. True values obey it exactly; soon we compute them.`],
   shot: [4, 12],
   say: [[["Try the start cell.", [40, 16, 320, 256]], ["Up lands on zero point four six, right on zero point four five.", [40, 16, 320, 256]], ["The two bumps keep Pip where it is.", [40, 16, 320, 256]]], [["The rule gives zero point four one. The walks gave zero point four.", [30, 292, 340, 87]], "Close, but not equal. Even a thousand walks leave a little noise."]],
   draw(t) { maze({ V: NB, t, hi: [[0, 3, C.teal, .2], [0, 2, C.gold, .22], [1, 3, C.gold, .15]] }); nums(NB); pip(cx(0), cy(3), 1, t);
     card([`0.9 × ( 0.85 × ${f2(NB[key(0, 2)])}  +  0.05 × ${f2(NB[key(1, 3)])}  +  0.10 × ${f2(NB[SK])} )`, `   = ${f2(T03)}   (up, right, two bumps)`, { s: `walks said ${f2(NB[SK])}: a small gap, from noise`, c: C.soft }], t - .6, { gap: 1.5, fs: 16 }); } },
 { h: 'A recipe with no walking', p: ['Turn the rule around to compute values. Start with a blank notebook, every number 0. Apply the rule to every cell, using the current numbers for the landing cells, and write the results into a fresh notebook.', 'One pass over all cells is a **sweep**. Repeat until the numbers stop changing; we will see why they do. Pip never takes a step.'],
   shot: [3, 10],
   say: [[["Now turn the rule around.", [40, 16, 320, 256]], ["Start with a blank notebook, every number zero.", [40, 16, 320, 256]]], [["Apply the rule to every cell, using the current numbers.", [30, 292, 340, 110]], ["That pass is a sweep. Repeat until nothing changes.", [30, 292, 340, 110]]], "And Pip never takes a single step."],
   draw(t) { maze({ V: ZERO(), t }); nums(ZERO(), clamp(t - .5)); arrows(.2);
     card(['1. start: every value 0', '2. sweep: every cell gets', '     average of [ reward + 0.9 × number where Pip lands ]', '3. repeat until nothing changes'], t - 1.5, { gap: 1.2 }); } },
 { h: 'Sweep 1', p: [`From all zeros, only cells next to the cheese or the cat change. Beside the cheese: 0.85 × 1 for the cheese, 0.05 × −1 for the cat, the rest are 0: ${f2(HIST[1][key(3, 0)])}. The cell below the cheese gets the same, ${f2(HIST[1][key(4, 1)])}.`, `The two other cells beside the cat get ${f2(HIST[1][key(2, 1)])}: a 5 in 100 chance of stepping onto it. Everything else stays 0.`],
   shot: [4, 11],
   say: [[["Here's sweep one. Watch which cells change.", [40, 16, 320, 256]], ["Only the cells touching the cheese or the cat.", [40, 16, 320, 256]]], [["Beside the cheese, it jumps to zero point eight.", [30, 292, 340, 87]], ["Beside the cat, a small minus, a five percent risk.", [30, 292, 340, 87]]], "Everything else still sees only zeros."],
   draw(t) { const u = clamp((t - 1.5) / 1); const V = HIST[0].map((v, i) => lerp(v, HIST[1][i], u)); const ch = (c, r) => Math.abs(HIST[1][key(c, r)]) > .001;
     maze({ V, t, hi: CELLS.filter(([c, r]) => ch(c, r)).map(([c, r]) => [c, r, HIST[1][key(c, r)] > 0 ? C.gold : C.rose, .25 * u]) }); nums(V); arrows(.2);
     if (t > 4) card([`beside the cheese: 0.85 × 1 + 0.05 × (−1) = ${f2(HIST[1][key(3, 0)])}`, `left of and below the cat: 0.05 × (−1) = ${f2(HIST[1][key(2, 1)])}`, { s: 'every other neighbour still says 0', c: C.soft }], t - 4, { gap: 1.2, fs: 16 }); caption('after sweep 1', clamp(t - 2.5)); } },
 { h: 'The news spreads', p: ['Each sweep, good news moves one cell further from the cheese. After 3 sweeps each number counts exactly the discounted rewards expected in Pip’s first 3 steps; after 7, in its first 7.', 'So the start stays near 0 until sweep 7: the cheese is 7 steps away. Then it jumps, then creeps up as longer walks add their smaller rewards.'],
   shot: [5, 12],
   say: [[["Now watch the sweeps run.", [40, 16, 320, 256]], ["Each one carries the good news one cell further from the cheese.", [40, 16, 320, 256]]], [["The start stays near zero for a while.", [40, 16, 320, 256]], ["Then, at sweep seven, it jumps. The cheese is seven steps away.", [100, 296, 200, 28]]]],
   draw(t) { const D = 1.3, k = clamp(Math.floor(t / D) + 1, 1, 9), u = ease((t - (k - 1) * D) / .7), V = HIST[k - 1].map((v, i) => lerp(v, HIST[k][i], u));
     maze({ V, t }); nums(V); arrows(.15); if (k >= 7) box(OX + 3, OY + 3 * CS + 3, CS - 6, CS - 6, C.teal, .25);
     caption(`after sweep ${k}`); txt(`start: ${f2(HIST[k][SK])}`, W / 2, 310, { s: 18, c: k >= 7 ? C.goldInk : C.soft }); } },
 { h: 'Until nothing changes', p: ['The start cell’s number after each sweep: across, sweep number; up, value. The dashed line is where it settles.', `By sweep ${SETTLE2} every number has its final two decimals; by sweep ${SETTLE3} none moves by more than 0.001. Why settle? The rule multiplies every landing value by 0.9, so each sweep leaves at most 0.9 times the old biggest error, here even less.`],
   shot: [3, 12],
   say: [[["This chart tracks the start cell, sweep by sweep.", [40, 285, 345, 115]], ["Watch it climb toward the dashed line.", [40, 285, 345, 115]]], [["By sweep sixteen, nothing moves by more than a thousandth.", [40, 285, 345, 115]]], "Why does it settle? Each sweep shrinks every error by at least zero point nine."],
   draw(t) { const n = clamp(Math.floor((t - 2.5) / .45), 0, 20); maze({ V: HIST[n], t }); nums(HIST[n]); arrows(.15); sweepChart(n, t); txt(`after sweep ${n}`, 366, 292, { s: 13, c: C.soft, al: 'right' }); } },
 { h: 'Walks versus sweeps', p: [`Left: 1000 walks per cell. Right: ${SETTLE3} sweeps. They agree within ${WGAP.toFixed(3)}, and the sweeps have no noise. From here on, the notebook shows swept values.`, `The walks took ${WALKSTEPS.toLocaleString('en-US')} steps. The sweeps did ${SETTLE3} passes over 16 cells, 4 moves each: ${(SETTLE3 * 64).toLocaleString('en-US')} small sums, each about as cheap as one step. A wet floor, with three landings per move, would triple the sums.`],
   shot: [4, 11],
   say: [[["Walks on the left, sweeps on the right.", [4, 20, 388, 180]], ["They agree within about a hundredth, and the sweeps have no noise.", [4, 20, 388, 180]]], [["The walks took seventy six thousand steps.", [30, 226, 340, 87]], ["The sweeps, about a thousand small sums.", [30, 226, 340, 87]]]],
   draw(t) { const a = clamp(t - .3), b = clamp(t - 1.6);
     alpha(a, () => { GA.maze({ V: NB, t }); nums(NB, 1, GA, 12); txt('from walks', GA.ox + 95, 28, { s: 15, c: C.goldInk }); });
     alpha(b, () => { GB.maze({ V: EX, t }); nums(EX, 1, GB, 12); txt('from sweeps', GB.ox + 95, 28, { s: 15, c: '#3f7f7c' }); });
     if (t > 3.5) card([`biggest difference: ${WGAP.toFixed(3)}  (noise in the walks)`, `walks: ${WALKSTEPS.toLocaleString('en-US')} steps`, `sweeps: ${SETTLE3} × 16 cells × 4 moves = ${(SETTLE3 * 64).toLocaleString('en-US')} sums`], t - 3.5, { y: 226, gap: 1.2 }); } },
 { h: 'The catch', p: ['Sweeps need the maze’s rules: for every move in every cell, where Pip can land, how likely, and what it pays. That is a **model**: here the world’s rulebook, not a neural network. The walks needed none of it.', 'Often no model is given. Later reels keep the Bellman rule but swap known probabilities for real experience, a step at a time: TD learning.'],
   shot: [3, 10],
   say: [[["The catch? Sweeps need the maze's rules.", [40, 16, 320, 256]], ["For every move, where Pip can land, how likely, and what it pays.", [30, 292, 340, 87]]], [["That's a model. The walks needed none of it.", [30, 292, 340, 87]]], "Often there's no model. Later reels keep the rule but learn from experience instead."],
   draw(t) { maze({ V: EX, t }); nums(EX, .5); arrows(.15); pip(cx(3), cy(0), 1, t); [0, 1, 2, 3].forEach(m => alpha(clamp(t - .5), () => edgeArrow(3, 0, m, 1500 + m, C.ink, 17)));
     card(['a model = for every cell and move:', '   where Pip can land, and how likely', '   what the step pays'], t - 1, { gap: 1.1 }); } },
 { h: 'The best Pip could do', p: ['The best any policy could do: pick the best move in every cell, and keep picking the best afterwards. Value here = the biggest, over moves, of reward + 0.9 × value where that move lands.', `That is the **Bellman optimality equation**. Its solution, **V star**, is the best policy’s value: no policy beats it in any cell. Swept, it settles after ${OPTN} sweeps.`],
   shot: [5, 9],
   say: ["Now ask for the best Pip could possibly do.", [["Instead of averaging over moves, take the biggest.", [30, 292, 340, 87]], ["That's the Bellman optimality equation.", [30, 292, 340, 87]]], [["Watch the sweeps. They settle after seven.", [40, 16, 320, 256]], ["The result, V star, is the best policy's value in every cell.", [40, 16, 320, 256]]]],
   draw(t) { const D = 1, k = clamp(Math.floor((t - .5) / D) + 1, 1, OPTN), u = ease((t - .5 - (k - 1) * D) / .6), V = OPT[k - 1].map((v, i) => lerp(v, OPT[k][i], u));
     maze({ V, t }); nums(V); caption(`optimality sweep ${k}`);
     card(['V*(here) = max over moves of', '                  [ reward + 0.9 × V*(where it lands) ]', { s: 'V* = value of the best policy, in every cell', c: C.soft }], t - .5, { gap: 1.2, fs: 16 }); } },
 { h: 'Reading the best map', p: ['Each cell’s best move has the biggest Q: reward + 0.9 × V star where it lands. These arrows lead straight to the cheese; on ties, as between the rocks, either is best.', `The best policy needs no random moves: the start is worth ${f2(OPT[OPTN][SK])}, against ${f2(EX[SK])} for Pip’s habit. On a wet floor, scores would first average over landings, then the max picks.`],
   shot: [4, 10],
   say: [[["In each cell, the best move has the biggest score.", [40, 16, 320, 256]], ["Watch the arrows. They lead straight to the cheese.", [40, 16, 320, 256]], ["Between the rocks, there's a tie, so either move is best.", [40, 16, 320, 256]]], [["The start is now worth zero point five three, against zero point four for Pip's habit.", [30, 292, 340, 87]]]],
   draw(t) { const V = OPT[OPTN]; maze({ V, t, hi: t > 3 ? [[1, 2, C.teal, .25]] : null }); nums(V); const best = (c, r) => { const b = branches(V, c, r, true); let bm = 0; b.forEach(x => { if (x.q > b[bm].q + 1e-9) bm = x.m; }); return bm; }; arrows(clamp(t - .5) * .9, best);
     if (t > 3) { ghost2(1, 2, 'tie', clamp(t - 3)); }
     card(['best move = biggest  reward + 0.9 × V*(landing)', `start: ${f2(V[SK])} best policy,  ${f2(EX[SK])} Pip’s habit`, { s: 'wet floor: average the landings, then take the max', c: C.soft }], t - 1.5, { gap: 1.3, fs: 16 }); } },
 { h: 'Your turn', ctl: { label: 'sweeps', values: SLV, def: 0, onChange: st => {} }, tap: tapCell,
   p: ['Drag the slider to run more sweeps of Pip’s policy, starting from all zeros, and watch the news spread from the cheese and the cat.', 'Tap any cell to see the rule at work there: its four moves, how likely each is, and the numbers from the sweep before.'],
   shot: [3],
   say: [[["Your turn. Drag the slider to run more sweeps.", [40, 16, 320, 256]], ["Watch the news spread out from the cheese and the cat.", [40, 16, 320, 256]]], [["Then tap any cell, and the card shows the rule at work there.", [30, 288, 340, 120]]]],
   draw(t, st) { const k = SLV[st.ci ?? 0], V1 = HIST[k], prev = st.prevCi != null ? HIST[SLV[st.prevCi]] : V1, u = ease((t - (st.changed ?? -9)) / .9), V = V1.map((v, i) => lerp(prev[i], v, u));
     maze({ V, t, hi: st.cell ? [[st.cell[0], st.cell[1], C.teal, .25]] : null }); nums(V); arrows(.15);
     if (st.cell && k > 0) { const [c, r] = st.cell, bl = branchLines(HIST[k - 1], c, r); card(bl.lines.concat([{ s: `weight by chance and add  →  ${f2(V1[key(c, r)])} after sweep ${k}`, c: C.goldInk }]), 9, { fs: 14, lh: 20, y: 288 }); }
     else { txt(k === 0 ? 'all zeros: no sweeps yet' : `after ${k} sweep${k > 1 ? 's' : ''}`, W / 2, 304, { s: 17, c: C.soft }); caption('tap a cell to see its rule'); } } },
 { h: 'Where it shows up', p: ['In the 1950s Richard Bellman called this **dynamic programming**: link each state to those one step later.', 'Planners with a model sweep it. Without one, TD learning nudges values toward reward + γ × the next state’s value. Q-learning and the 2015 Atari player DQN aim at reward + γ × the biggest Q in the next cell: the best next move’s value.'],
   shot: [4, 10],
   say: ["Richard Bellman called this dynamic programming, back in the nineteen fifties.", [["Planners with a model sweep it.", [30, 292, 340, 87]], ["Without one, TD learning nudges values toward the reward plus gamma times the next value.", [30, 292, 340, 87]]], [["And Q-learning, and the Atari player DQN, aim at the best next move's value.", [30, 292, 340, 87]]]],
   draw(t) { maze({ V: EX, t }); nums(EX, .6); arrows(.2);
     card(['dynamic programming: Bellman, 1950s', 'planning with a model: sweep the rule', { s: 'Q-learning, DQN: reward + γ × biggest next Q', c: C.soft }], t - .5, { gap: 1.2 }); } },
 { h: 'Recap', p: ['A walk’s return is its first reward plus γ times the return of the rest. Averaging gives the **Bellman equation**: value here = expected next reward + γ × expected value of where you land.', 'Sweep it over all cells until nothing changes to get values without walking, given the model. Put a max over moves inside for the best possible values.'],
   shot: [3, 9],
   say: [[["So, a walk's return is its first reward plus gamma times the rest.", [30, 292, 340, 110]], ["Average that, and you get the Bellman equation.", [30, 292, 340, 110]]], ["Sweep it until nothing changes, and you get values without walking. Put a max inside, and you get the best possible values."]],
   draw(t) { maze({ V: EX, t }); nums(EX); arrows(.25); pip(cx(0), cy(3), 1, t);
     card(['V(s) = average of [ reward + γ × V(next) ]', 'sweep from zeros until nothing changes', 'needs a model: landings, chances, rewards', 'V*(s) = max over moves of [ reward + γ × V*(next) ]'], t - .3, { gap: 1, fs: 16 }); } },
];

RLStory.register('bellman', { title: 'The Bellman Equation', track: 'Foundations', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { G, NB, EX, HIST, OPT, WALKSTEPS, W1, SETTLE2, SETTLE3, WGAP, OPTN, UPN, UPS, S30, T03, branches, sweep } });
})();
