// Successor representation as a story: Pip's notebook of values is split into a map of "where will I go" (expected
// discounted future arrivals in each cell, under Pip's usual habit, learned by TD) and "what does each cell pay".
// When the cheese and the cat trade places, values come back instantly by re-weighting the map, while plain TD has to
// relearn walk by walk. Same maze, habit, walks and α as the TD reel (rng(87000 + start key)); see notes/successor.md.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const G2 = L.gridWorld({ cols: 5, rows: 4, cheese: [3, 1], cat: [4, 0], rocks: [[2, 2], [1, 1]] });   // after the swap (same walls, same ending cells)
const { cx, cy, key, trail, walkPos, pip } = G;
const GAM = 0.9, EPS = 0.2, A0 = 0.2, NWALK = 200;
const SK = key(0, 3), KCH = key(4, 0), KCAT = key(3, 1), K30 = key(3, 0);
const isT = k => k === KCH || k === KCAT;
const no = k => k + 1;                                  // cell numbers 1…20, row by row, as in the loop reel
// Pip's habit: the arrows of the values / bellman / td reels (0 up, 1 right, 2 down, 3 left), 1 move in 5 random
const ARR = new Array(20).fill(-1);
[[0, 3, 0], [0, 2, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [2, 1, 0], [1, 2, 3], [1, 3, 1], [2, 3, 1], [3, 3, 1], [4, 3, 0], [4, 2, 0], [4, 1, 0], [3, 2, 1]]
  .forEach(([c, r, m]) => { ARR[key(c, r)] = m; });
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);
const ALL = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.isRock(c, r)) ALL.push([c, r]);   // map columns: every reachable cell, endings included
// the walks: the TD reel's 200 walks (same generator), then 200 more from the same generator for after the swap
const WALKS = (() => { const R = rng(87000 + SK), out = []; for (let i = 0; i < 2 * NWALK; i++) { let c = 0, r = 3; const tr = [];
  for (let k = 0; k < 400; k++) { const m = R() < EPS ? Math.floor(R() * 4) : ARR[key(c, r)], [nc, nr] = G.step(c, r, m); tr.push({ c, r, nc, nr, m }); c = nc; r = nr; if (isT(key(c, r))) break; }
  out.push(tr); } return out; })();
const zeros = () => Array.from({ length: 20 }, () => new Array(20).fill(0));
// exact map of the habit (needs the rulebook; only for checking): M[s][s'] = Σ over later steps t ≥ 1 of 0.9^(t−1) × chance of arriving in s' at step t
const MEX = (() => { let M = zeros(); for (let it = 0; it < 400; it++) { const N = zeros();
  CELLS.forEach(([c, r]) => { const k = key(c, r); for (let m = 0; m < 4; m++) { const p = EPS / 4 + (m === ARR[k] ? 1 - EPS : 0), [nc, nr] = G.step(c, r, m), k2 = key(nc, nr); N[k][k2] += p; if (!isT(k2)) for (let j = 0; j < 20; j++) N[k][j] += p * GAM * M[k2][j]; } }); M = N; } return M; })();
// TD on the map: after a step from s to s', row s moves α of the way toward (1 in column s') + 0.9 × row s' (row s' = 0 if the walk ended)
const SNAP = [];   // the start row after each walk
const ML = (() => { const M = zeros(); WALKS.slice(0, NWALK).forEach(tr => { tr.forEach(s => { const k = key(s.c, s.r), k2 = key(s.nc, s.nr);
  for (let j = 0; j < 20; j++) M[k][j] += A0 * ((j === k2 ? 1 : 0) + (isT(k2) ? 0 : GAM * M[k2][j]) - M[k][j]); }); SNAP.push(M[SK].slice()); }); return M; })();
const W1 = WALKS[0];
const ROW1 = (() => { const r = new Array(20).fill(0); r[key(W1[0].nc, W1[0].nr)] = A0; return r; })();   // the start row after walk 1, step 1
const val = (M, pay) => M.map(row => row.reduce((a, v, j) => a + v * (pay[j] || 0), 0));
const PAY = new Array(20).fill(0); PAY[KCH] = 1; PAY[KCAT] = -1;
const PAY2 = new Array(20).fill(0); PAY2[KCH] = -1; PAY2[KCAT] = 1;            // cheese and cat trade places
const VEX = val(MEX, PAY), VEX2 = val(MEX, PAY2), VL = val(ML, PAY), VL2 = val(ML, PAY2);
// plain TD on values (the TD reel's rule), same walks: identical to VL after 200 walks; then 200 more walks with the swapped pays
const TDV = (() => { const V = new Array(20).fill(0); WALKS.slice(0, NWALK).forEach(tr => tr.forEach(s => { const k = key(s.c, s.r), k2 = key(s.nc, s.nr); V[k] += A0 * (PAY[k2] + (isT(k2) ? 0 : GAM * V[k2]) - V[k]); })); return V; })();
const err = (V, E) => CELLS.reduce((a, [c, r]) => a + Math.abs(V[key(c, r)] - E[key(c, r)]), 0) / CELLS.length;
const SAME = Math.max(...CELLS.map(([c, r]) => Math.abs(TDV[key(c, r)] - VL[key(c, r)])));
const RELEARN = (() => { const V = TDV.slice(), e = [err(V, VEX2)], st = [V[SK]]; WALKS.slice(NWALK).forEach(tr => { tr.forEach(s => { const k = key(s.c, s.r), k2 = key(s.nc, s.nr); V[k] += A0 * (PAY2[k2] + (isT(k2) ? 0 : GAM * V[k2]) - V[k]); }); e.push(err(V, VEX2)); st.push(V[SK]); }); return { e, st, V }; })();
const ESR = err(VL2, VEX2), NEG = RELEARN.st.findIndex(v => v < 0);
// a crumb: +0.5 on every arrival in cell (0, 1), the walk goes on
const KCR = key(0, 1), PAY3 = PAY.slice(); PAY3[KCR] = .5; const VL3 = val(ML, PAY3), VEX3 = val(MEX, PAY3);

// ───── numbers and drawing helpers
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−');
const fv = v => Math.abs(v) < .0005 ? '0' : Math.abs(v) < .1 ? v.toFixed(3).replace('-', '−') : f2(v);
const TEAL = '#3f8582';
function nums(V, a = 1, Gx = G, only) { for (const [c, r] of CELLS) { if (only && !only(c, r)) continue; const aa = typeof a === 'function' ? a(c, r) : a; if (aa <= 0) continue; const v = V[key(c, r)];
  txt(fv(v), Gx.ox + c * Gx.cs + Gx.cs - 5, Gx.oy + r * Gx.cs + 12, { s: 14, al: 'right', a: aa, c: v < -.0005 ? C.roseInk : v > .0005 ? C.goldInk : C.soft }); } }
// a map row (future arrivals from one cell) drawn in teal; numbers in every reachable cell, ending cells included
function mapRow(row, a = 1, o = {}) { const Gx = o.G || G, mx = o.max || 1.2;
  ALL.forEach(([c, r]) => { const k = key(c, r), v = row[k], aa = typeof a === 'function' ? a(c, r) : a; if (aa <= 0) return;
    if (v > .004) box(Gx.ox + c * Gx.cs + 2, Gx.oy + r * Gx.cs + 2, Gx.cs - 4, Gx.cs - 4, C.teal, aa * lerp(.08, .55, clamp(v / mx)));
    const term = isT(k); txt(v < .005 ? '0' : v.toFixed(2), Gx.ox + c * Gx.cs + (term ? 6 : Gx.cs - 5), Gx.oy + r * Gx.cs + (term ? Gx.cs - 10 : 12), { s: 14, al: term ? 'left' : 'right', a: aa, c: v > .004 ? TEAL : C.soft }); }); }
function arrows(a = 1) { CELLS.forEach(([c, r]) => alpha(a, () => G.arrowIn(c, r, ARR[key(c, r)], 400 + key(c, r), { c: C.goldInk, w: 1.4 }))); }
function ring(c, r, col = C.ink, a = 1) { alpha(a, () => rrect(G.ox + c * G.cs + 3, G.oy + r * G.cs + 3, G.cs - 6, G.cs - 6, 600 + key(c, r), { c: col, w: 2 })); }
function crumb(c, r, a = 1) { alpha(a, () => { [[-6, 2], [3, -3], [5, 5], [-1, 7]].forEach(([dx, dy], i) => L.rcircle(cx(c) + dx, cy(r) + dy + 4, 2.6, 700 + i, { fill: '#e8c46a', j: .3 })); }); }
const dim = (a = .62) => box(0, 0, W, 285, C.bg, a);
// small cell numbers 1…20 in the top-left corner of each cell (as in the loop reel)
function cellNos(a = 1) { if (a <= 0) return; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.isRock(c, r)) txt(String(no(key(c, r))), G.ox + c * G.cs + 5, G.oy + r * G.cs + 10, { s: 12, c: C.soft, al: 'left', a }); }
// line chart of an error over walks
function errChart(n, o = {}) { const x0 = 60, x1 = 350, y0 = 386, y1 = 306, hi = 1, X = i => x0 + i / NWALK * (x1 - x0), Y = v => y0 - v / hi * (y0 - y1);
  rline(x0 - 4, y0, x1 + 4, y0, 960, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1 - 2, 961, { c: C.line, w: 1, once: true });
  [0, .5, 1].forEach(v => txt(String(v), x0 - 9, Y(v), { s: 12, c: C.soft, al: 'right' })); txt('swap', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(`${NWALK} walks later`, x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
  txt('average distance from the true new values', x0 - 8, y1 - 12, { s: 13, c: C.soft, al: 'left' });
  ctx.save(); ctx.strokeStyle = TEAL; ctx.lineWidth = 2.2; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(X(0), Y(ESR)); ctx.lineTo(X(NWALK), Y(ESR)); ctx.stroke(); ctx.restore();
  ctx.save(); ctx.strokeStyle = '#c99a3a'; ctx.lineWidth = 2; ctx.beginPath(); for (let i = 0; i <= Math.min(n, NWALK); i++) { const x = X(i), y = Y(RELEARN.e[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); }
const SCH = W1.map(s => key(s.nc, s.nr));   // walk 1's arrivals

const STEPS = [
 { h: 'Back in the maze', p: [`Same maze, habit and rules as in the TD reel: dry floor, the arrows with one move in five random, +1 on the step onto the cheese, −1 onto the cat, either ends the walk, γ (gamma) = 0.9.`, `After its ${NWALK} walks, TD gave Pip this notebook: ${f2(TDV[K30])} left of the cheese, ${f2(TDV[SK])} at the start. Each says how good a cell is, under this habit, for this cheese. They are noisy: the true values are ${f2(VEX[K30])} and ${f2(VEX[SK])}, and that noise will follow us through this reel.`],
   shot: [3, 9],
   draw(t) { G.maze({ V: TDV, t, labels: clamp(t - .5) }); nums(TDV, (c, r) => clamp((t - .8 - key(c, r) * .05) * 3)); arrows(.3 * clamp(t - 2)); pip(cx(0), cy(3), 1, t);
     card(['habit: the arrows, 1 move in 5 random', `notebook: TD after ${NWALK} walks`, { s: 'each number: reward to come, under this habit', c: C.soft }], t - 4, { gap: 1.1 }); } },
 { h: 'What if the cheese moves?', p: ['Tomorrow someone swaps the cheese and the cat. Walls, habit, the two ending cells: all the same. Only what the ending cells pay has changed.', 'Pip’s notebook is now wrong almost everywhere, because each number mixes two things together: where Pip tends to go, and what it finds there. Plain TD would have to unlearn and relearn, walk by walk.'],
   shot: [4, 11],
   draw(t) { const u = ease((t - 1.5) / 1.5), Gx = t < 2.25 ? G : G2; Gx.maze({ V: TDV, t, pop: { cheese: Math.abs(1 - 2 * u), cat: Math.abs(1 - 2 * u) } }); nums(TDV, .9); pip(cx(0), cy(3), 1, t);
     if (t > 3.5) txt('?', cx(0) - 4, cy(3) - 22, { s: 22, c: C.roseInk, a: clamp(t - 3.5) });
     card([{ s: 'notebook = where Pip goes  ×  what it finds', c: C.ink }, { s: 'the finds changed: every number is stale', c: C.roseInk }], t - 5, { gap: 1.3 }); } },
 { h: 'Two questions', p: ['So keep the two apart. Question one: from here, **where will I go**? That depends only on the maze and Pip’s habit. Question two: **what does each cell pay**? That is the part that changed.', 'Like knowing your town’s streets: whether today you want the bakery or the pool, the streets you would walk are the same. Only the errand changes.'],
   shot: [3, 10],
   draw(t) { G.maze({ t }); arrows(.5); pip(cx(0), cy(3), 1, t); dim(.55);
     txt('where will I go?', W / 2, 110, { s: 22, c: TEAL, a: clamp(t - .5) }); txt('maze + habit', W / 2, 140, { s: 16, c: C.soft, a: clamp(t - 1.2) });
     txt('what does each cell pay?', W / 2, 190, { s: 22, c: C.goldInk, a: clamp(t - 2.5) }); txt('the rewards', W / 2, 220, { s: 16, c: C.soft, a: clamp(t - 3.2) });
     card(['value  =  where I go  combined with  what it pays'], t - 5, { gap: 1 }); } },
 { h: 'Counting future visits', p: [`Answer question one by counting. Walk 1 from the start: Pip steps into cell ${no(SCH[0])}, then ${no(SCH[1])}, and so on, reaching the cheese on step ${W1.length}. Score each cell it steps into: 1 for the next step, 0.9 for the step after, 0.81 after that, one more 0.9 per step.`, 'That is the same discounting as for rewards. The cheese cell gets 0.9 multiplied ' + (W1.length - 1) + ` times: ${(GAM ** (W1.length - 1)).toFixed(2)}. Ending cells count too; after them the walk is over.`],
   shot: [5, 12],
   draw(t) { const n = W1.length, m = walkPos(W1, 0, n, Math.max(0, t - .5), 1.1), k = m.done ? n : m.k; G.maze({ t }); cellNos(); trail(W1.slice(0, k).map(s => [s.nc, s.nr]));
     W1.slice(0, k).forEach((s, i) => { const kk = key(s.nc, s.nr), term = isT(kk); txt((GAM ** i).toFixed(2).replace(/\.?0+$/, '') || '1', cx(s.nc) + (term ? -18 : 14), cy(s.nr) + (term ? 22 : -18), { s: 15, c: TEAL }); });
     pip(m.x, m.y, m.face, t); caption(`walk 1: step ${clamp(k, 1, n)} of ${n}`); } },
 { h: 'Pip’s map from the start', p: [`One walk is luck. Average such scores over all possible walks from the start, and you get Pip’s **map** of where it will go: ${MEX[SK][key(0, 2)].toFixed(2)} for the cell above the start (more than 1: Pip can step in more than once), ${MEX[SK][KCH].toFixed(2)} for the cheese cell, ${MEX[SK][KCAT].toFixed(2)} for the cat’s.`, `This map is called the **successor representation** (Peter Dayan, 1993). The start itself gets ${MEX[SK][SK].toFixed(2)}, from walks that wander back to it; counting begins with the next step, to match “reward on arrival”. Computed here with the rulebook, as a check.`],
   shot: [4, 11],
   draw(t) { G.maze({ t }); mapRow(MEX[SK], (c, r) => clamp((t - .6 - key(c, r) * .08) * 3)); ring(0, 3, C.ink, clamp(t - .3)); pip(cx(0), cy(3), 1, t);
     card([{ s: 'map row of the start cell:', c: C.ink }, { s: 'discounted future visits to each cell', c: TEAL }, { s: 'teal: more visits · from the rulebook, as a check', c: C.soft }], t - 4, { gap: 1.2 }); } },
 { h: 'Value from the map', p: [`Now question two. What each cell pays when Pip steps onto it: cheese +1, cat −1, every other cell 0. This works because the pay depends only on the cell Pip lands in. Multiply each cell’s visits by its pay, and add up.`, `From the start: ${MEX[SK][KCH].toFixed(2)} × 1 + ${MEX[SK][KCAT].toFixed(2)} × (−1) = ${f2(VEX[SK])}. That is exactly the start’s value under this habit, the number the sweeps found in the earlier reels.`],
   shot: [4, 11],
   draw(t) { G.maze({ t }); mapRow(MEX[SK], (c, r) => isT(key(c, r)) ? 1 : .35); ring(0, 3); pip(cx(0), cy(3), 1, t);
     if (t > 1.5) { txt('pays +1', cx(4), cy(0) + 21, { s: 13, c: C.goldInk, a: clamp(t - 1.5) }); txt('pays −1', cx(3), cy(1) + 21, { s: 13, c: C.roseInk, a: clamp(t - 2) }); }
     card([`${MEX[SK][KCH].toFixed(2)} × (+1)  +  ${MEX[SK][KCAT].toFixed(2)} × (−1)`, { s: `= ${f2(VEX[SK])}: the start’s value`, c: C.goldInk }, { s: 'value = sum over cells of visits × pay', c: C.soft }], t - 3, { gap: 1.3 }); } },
 { h: 'Learning the map by TD', p: ['Without the rulebook, Pip learns the map the TD way, one row per open cell: 16 × 18 numbers instead of TD’s 16. After each step, the whole row of the cell just left moves toward 1 for the landing cell plus 0.9 × the landing cell’s row; nothing is added if the walk ended.', `It is the TD rule with a 1 for “I arrived here” in place of the reward. Walk 1, step 1: Pip lands in cell ${no(SCH[0])}, whose row is still all 0, so the start’s entry for cell ${no(SCH[0])} moves a fifth of the way to 1: 0.2.`],
   shot: [4, 12],
   draw(t) { const m = walkPos(W1, 0, 1, clamp(t - 1, 0, 1), 1); G.maze({ t }); cellNos(.8); mapRow(t > 4 ? ROW1 : new Array(20).fill(0), (c, r) => (c === W1[0].nc && r === W1[0].nr) || (c === 0 && r === 3) ? 1 : .25); ring(0, 3, TEAL); pip(m.x, m.y, m.face, t);
     card(['row of the cell left  →  α of the way toward', { s: '   1 in the landing cell  +  0.9 × landing cell’s row', c: TEAL }, { s: 'α = 0.2, as in the TD reel', c: C.soft }], t - 2, { gap: 1.3 }); } },
 { h: `${NWALK} walks later`, p: [`After the same ${NWALK} walks, Pip’s learned start row: ${ML[SK][key(0, 2)].toFixed(2)} above the start, ${ML[SK][KCH].toFixed(2)} at the cheese, ${ML[SK][KCAT].toFixed(2)} at the cat; close to the rulebook’s map on the usual route, rougher elsewhere.`,
     `Multiply by the pays, and every cell matches the TD reel’s notebook exactly: ${f2(VL[SK])} at the start. Why? Same walks, same α, and the map update is linear, so multiplying it by the pays gives the TD update term by term. With fixed rewards the map buys nothing; it pays off when rewards change.`],
   shot: [5, 12],
   draw(t) { const n = clamp(Math.floor((t - .5) * 40), 1, NWALK); G.maze({ t }); mapRow(SNAP[n - 1], 1); ring(0, 3, TEAL); pip(cx(0), cy(3), 1, t);
     caption(n < NWALK ? `learned start row after walk ${n}` : `learned start row after walk ${NWALK}`);
     if (t > 6.5) card([`map × pays, start: ${f2(VL[SK])}   (TD notebook: ${f2(TDV[SK])})`, `left of the cheese: ${f2(VL[K30])}   (TD: ${f2(TDV[K30])})`, { s: 'identical in every cell', c: C.soft }], t - 6.5, { gap: 1.2 }); } },
 { h: 'The swap', p: [`Now the cheese and the cat trade places. Pip keeps its map: the habit and the walls did not change. It only rewrites the pays: +1 for the old cat cell, −1 for the old cheese cell, and multiplies again.`, `Instantly, with no new walk: the start reads ${f2(VL2[SK])}, the cell left of the old cheese ${f2(VL2[K30])}. Here the pays simply traded places, so every value flips its sign; the next example is less tidy.`],
   shot: [4, 11],
   draw(t) { const u = clamp((t - 2) / 1.2), Gx = u < .5 ? G : G2, V = u < .5 ? VL : VL2; Gx.maze({ V, t, pop: { cheese: Math.abs(1 - 2 * u), cat: Math.abs(1 - 2 * u) } }); nums(V, t < 2 || t > 3.2 ? 1 : 0); pip(cx(0), cy(3), 1, t);
     card(['same map', { s: 'new pays: cheese +1 and cat −1, swapped', c: C.goldInk }, { s: 'values = map × new pays: instant, no new walks', c: TEAL }], t - 5, { gap: 1.2 }); } },
 { h: 'Relearning, the slow way', p: [`Plain TD has only its old notebook. Let it walk ${NWALK} more times after the swap. Gold: how far its numbers are, on average, from the true new values under the old habit. Dashed teal: the instant re-weighted values, ${f2(ESR)} off.`,
     `TD starts ${f2(RELEARN.e[0])} off; its start value turns negative only after walk ${NEG}; after ${NWALK} walks it is ${f2(RELEARN.e[NWALK])} off. Both share the noise of a fixed α; the map got there at once.`],
   shot: [5, 13],
   draw(t) { const n = clamp(Math.floor((t - 1) * 30), 0, NWALK); G2.maze({ V: n ? RELEARN.V : TDV, t, pop: {} }); dim(.45); errChart(n);
     txt('gold: plain TD, relearning', 60, 411, { s: 13, c: '#b07d12', al: 'left', a: clamp(t - 1) }); txt('dashed teal: map × new pays', 350, 411, { s: 13, c: TEAL, al: 'right', a: clamp(t - 1) });
     txt(`walk ${n}`, 350, 294, { s: 13, c: C.soft, al: 'right' }); } },
 { get h() { return 'Any new reward'; }, p: [`Rewards can change anywhere, not only at the ending cells. Put the cheese and cat back, and add a **crumb** in cell ${no(KCR)}, on Pip’s usual route: +0.5 every time Pip steps onto it, and the walk goes on.`, `Same map, new pays: the start’s value jumps from ${f2(VL[SK])} to ${f2(VL3[SK])} at once, because the map says Pip steps onto that cell often and early. The rulebook’s answer is ${f2(VEX3[SK])}.`],
   shot: [4, 11],
   draw(t) { const V = t < 3 ? VL : VL3; G.maze({ V, t }); cellNos(.8); crumb(0, 1, clamp(t - 1)); nums(V, t < 2.6 || t > 3.4 ? 1 : 0); pip(cx(0), cy(3), 1, t);
     if (t > 1.2) txt('+0.5', cx(0) - 18, cy(1) - 18, { s: 14, c: C.goldInk, a: clamp(t - 1.2) });
     card([`crumb: +0.5 each visit to cell ${no(KCR)}`, `start: ${VL[SK].toFixed(3)} + 0.5 × ${ML[SK][KCR].toFixed(3)} visits = ${VL3[SK].toFixed(3)}`, { s: `rulebook: ${f2(VEX[SK])} + 0.5 × ${MEX[SK][KCR].toFixed(2)} = ${f2(VEX3[SK])}`, c: C.soft }], t - 4, { gap: 1.2 }); } },
 { h: 'Your turn', p: ['**Tap a cell** to put a crumb there, worth +0.5 each time Pip steps onto it. Tap it again to take it away. Pip’s values come straight from its learned map times the new pays, with no new walks.', 'Crumbs on the usual route lift many cells; crumbs in corners Pip rarely visits barely matter, because the map gives them few visits.'],
   shot: [3, 8],
   tap(x, y, st) { const c = Math.floor((x - G.ox) / G.cs), r = Math.floor((y - G.oy) / G.cs); if (c < 0 || r < 0 || c > 4 || r > 3 || G.isRock(c, r) || isT(key(c, r))) return;
     st.cr = st.cr || []; const k = key(c, r), i = st.cr.indexOf(k); if (i >= 0) st.cr.splice(i, 1); else st.cr.push(k); st.V = null; },
   draw(t, st) { const cr = st.cr || []; if (!st.V) { const p = PAY.slice(); cr.forEach(k => { p[k] += .5; }); st.V = val(ML, p); } G.maze({ V: st.V, t }); cr.forEach(k => crumb(k % 5, Math.floor(k / 5))); nums(st.V); pip(cx(0), cy(3), 1, t);
     txt(cr.length ? `start: ${f2(st.V[SK])}   (no crumbs: ${f2(VL[SK])})` : 'tap a cell to drop a crumb', W / 2, 320, { s: 18, c: cr.length ? C.goldInk : C.soft, a: cr.length ? 1 : .5 + .5 * Math.sin(t * 3) ** 2 }); } },
 { h: 'The catch', p: ['The map belongs to the habit. After the swap, Pip’s arrows still march toward the old cheese corner, now the cat. A better habit would take other routes, and a new habit has a new map, which must be relearned, at least in part.', 'To pick better moves without the rulebook, keep a map per move, like Q-values per move, and improve the habit as Q-learning does. The successor representation makes **new rewards** cheap, not **new habits**.'],
   shot: [3, 10],
   draw(t) { G2.maze({ V: VL2, t }); nums(VL2, .6); arrows(clamp(t - .5)); pip(cx(0), cy(3), 1, t);
     card([{ s: 'new rewards: re-weight the map, instant', c: TEAL }, { s: 'new habit: new map, must be learned', c: C.roseInk }], t - 3, { gap: 1.3 }); } },
 { h: 'Successor features', p: ['Big worlds have too many cells to map one by one. **Successor features** (André Barreto and colleagues, 2017) map future amounts of a few features, like food or water; a task gives each feature a weight, and value is map × weights.', 'Keep several old habits’ maps. For a new task, each habit’s value comes instantly; at every spot, follow the habit that looks best there. This **generalised policy improvement** is at least as good as each old habit.'],
   shot: [3, 10],
   draw(t) { G.maze({ t }); mapRow(ML[SK], .5); dim(.55);
     card(['1993  successor representation (Dayan)', '2017  successor features (Barreto and others)', '   value = feature map × feature weights', { s: '   new task: follow the best old habit per spot', c: C.soft }], t - .5, { y: 80, gap: 1.2 }); } },
  { h: 'A map in the brain?', p: ['The hippocampus holds **place cells**: neurons that fire when an animal is in a particular place. Kimberly Stachenfeld, Matthew Botvinick and Samuel Gershman (2017) proposed that they hold a predictive map like Pip’s: firing for where the animal is likely to go next, not just where it is.', 'This fits some recordings, such as place fields that stretch backward along a habitual path. How far the idea goes is still debated.'],
   shot: [3, 10],
   draw(t) { G.maze({ t }); mapRow(MEX[SK], clamp(t - .5) * .8); pip(cx(0), cy(3), 1, t); dim(.35);
     card(['place cells: fire at a particular place', { s: 'proposal: they encode where you will go', c: TEAL }, { s: 'Stachenfeld, Botvinick, Gershman, 2017', c: C.soft }], t - 2, { gap: 1.2 }); } },
{ h: 'Recap', p: ['The **successor representation** splits value into a map of where Pip will go, discounted future visits to every cell under its habit, and what each cell pays. Value is visits times pay, added up.', 'The map is learned by TD, with “I arrived here” in place of the reward. When rewards change, new values come instantly; when the habit changes, the map must be relearned.'],
   shot: [3, 9],
   draw(t) { G.maze({ V: VL, t }); nums(VL, .5); pip(cx(0), cy(3), 1, t); dim(.4);
     card(['map: discounted future visits per cell', 'value = Σ visits × pay', 'map learned by TD: 1 for “arrived here”', { s: 'new rewards: instant · new habit: relearn', c: C.soft }], t - .3, { gap: 1 }); } },
];
RLStory.register('successor', { title: 'Successor Features', track: 'Model-free', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { MEX, ML, VEX, VEX2, VL, VL2, TDV, SAME, RELEARN, ESR, NEG, VL3, VEX3, SNAP, WALKS } });
})();
