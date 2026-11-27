// Policy iteration as a story: holding the rulebook of Pip's wet maze, we start from a bad plan (every arrow up) and
// alternate evaluate (the plan's notebook, swept as in the Bellman reel) and improve (each arrow → its best-scoring move).
// Every number comes from the computation below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, hachure, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({});                                   // shared maze: cheese (4,0), cat (3,1), rocks (1,1) (2,2), start (0,3)
const { cx, cy, key, maze, trail, pip, MOVES } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GAM = 0.9, P = 0.2, TOL = .001, SK = key(0, 3), NAMES = ['up', 'right', 'down', 'left'];
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);

// ───── the rulebook of the wet floor (as in the mdp and valueiter reels): straight 0.8, 0.1 to each side, never backwards
function outcomes(c, r, m, p = P) {
  const out = [];
  for (const [mm, q] of [[m, 1 - p], [(m + 3) % 4, p / 2], [(m + 1) % 4, p / 2]]) {
    if (q <= 0) continue; const [nc, nr] = G.step(c, r, mm), o = out.find(o => o.nc === nc && o.nr === nr);
    if (o) o.prob += q; else out.push({ nc, nr, prob: q, mm });
  }
  return out;
}
// score of move m from (c, r) with notebook V = average over landings of [ reward + γ × V(landing) ]
function score(V, c, r, m) { return outcomes(c, r, m).reduce((s, o) => { const rw = G.term(o.nc, o.nr); return s + o.prob * (rw + (rw ? 0 : GAM * V[key(o.nc, o.nr)])); }, 0); }
const maxDiff = (a, b) => Math.max(...CELLS.map(([c, r]) => Math.abs(a[key(c, r)] - b[key(c, r)])));
// evaluate: sweep the plan's rule (only the plan's move) from V0 until the biggest change < 0.001; keep every sweep
function evaluate(plan, V0) { const hist = [V0.slice()]; for (let n = 0; n < 2000; n++) { const V = hist[hist.length - 1], N = V.slice(); CELLS.forEach(([c, r]) => { N[key(c, r)] = score(V, c, r, plan[key(c, r)]); }); hist.push(N); if (maxDiff(N, V) < TOL) break; } return hist; }
// improve: each arrow → the best-scoring move; keep the old arrow unless another move scores strictly higher
function improve(plan, V) { const np = plan.slice(), changed = []; CELLS.forEach(([c, r]) => { const k = key(c, r); let b = score(V, c, r, plan[k]), bm = plan[k];
  for (let m = 0; m < 4; m++) { const q = score(V, c, r, m); if (q > b + 1e-9) { b = q; bm = m; } } if (bm !== plan[k]) changed.push(k); np[k] = bm; }); return { np, changed }; }
const ZERO0 = () => new Array(20).fill(0);
const memo = {};
function run(m0) {
  if (memo[m0]) return memo[m0];
  let plan = new Array(20).fill(-1); CELLS.forEach(([c, r]) => { plan[key(c, r)] = m0; });
  let V = new Array(20).fill(0); const rounds = [];
  for (let i = 0; i < 30; i++) { const hist = evaluate(plan, V); V = hist[hist.length - 1]; const { np, changed } = improve(plan, V); rounds.push({ plan, hist, V, n: hist.length - 1, np, changed }); if (!changed.length) break; plan = np; }
  return (memo[m0] = { rounds, sweeps: rounds.reduce((s, x) => s + x.n, 0) });
}
const UP = run(0), RD = UP.rounds, FIN = RD[RD.length - 1];
// value iteration on the same floor, for the comparison (same stopping rule)
const VIH = [new Array(20).fill(0)]; for (let i = 1; i <= 400; i++) { const V = VIH[i - 1], N = V.slice(); CELLS.forEach(([c, r]) => { N[key(c, r)] = Math.max(...[0, 1, 2, 3].map(m => score(V, c, r, m))); }); VIH.push(N); }
const VIS = VIH.findIndex((V, i) => i > 0 && maxDiff(V, VIH[i - 1]) < TOL), VIX = VIH[400];
const VIPLAN = CELLS.map(([c, r]) => { const q = [0, 1, 2, 3].map(m => score(VIX, c, r, m)); return q.indexOf(Math.max(...q)); });
const SAME = CELLS.every(([c, r], i) => FIN.plan[key(c, r)] === VIPLAN[i]);
// "never worse": every cell's value, round by round, went up or stayed
const MONO = RD.every((x, i) => !i || CELLS.every(([c, r]) => x.V[key(c, r)] >= RD[i - 1].V[key(c, r)] - 1e-12));
const STARTS = [0, 1, 2, 3].map(m => run(m)), ALLSAME = STARTS.every(R => CELLS.every(([c, r]) => R.rounds[R.rounds.length - 1].plan[key(c, r)] === FIN.plan[key(c, r)]));
const MONOALL = STARTS.every(R => R.rounds.every((x, i) => !i || CELLS.every(([c, r]) => x.V[key(c, r)] >= R.rounds[i - 1].V[key(c, r)] - 1e-12)));
const PI_COST = UP.sweeps * 16 + RD.length * 64, VI_COST = VIS * 64;

// ───── numbers the story quotes
const f2 = v => { const s = (Math.abs(v) < .005 ? 0 : v).toFixed(2); return s.replace('-', '−'); };
const r2 = v => +(+v).toFixed(2);
const K30 = key(3, 0), K41 = key(4, 1), K32 = key(3, 2), K21 = key(2, 1), K13 = key(1, 3), K23 = key(2, 3);
const V1 = RD[0].V;
const S30 = [0, 1, 2, 3].map(m => score(V1, 3, 0, m));                         // the four scores beside the cheese after round 1's evaluation
const HAND30R = .8 * 1 - .1 + .1 * GAM * r2(V1[K30]);                           // right, with the 2-decimal number shown
const R2ZERO = evaluate(RD[1].plan, ZERO0()).length - 1;                        // round 2 evaluated from zeros instead
const QF30 = [0, 1, 2, 3].map(m => score(FIN.V, 3, 0, m));                     // final Q's beside the cheese
const V2_30 = RD[1].V[K30];
const UPS = [0, 1, 2].map(i => CELLS.filter(([c, r]) => r2(RD[i + 1].V[key(c, r)]) > r2(RD[i].V[key(c, r)])).length);
const nm = k => `${k % 5},${Math.floor(k / 5)}`;

// ───── drawing helpers
const ZERO = () => new Array(20).fill(0);
function nums(V, a = 1, Gx = G, s = 14) {
  for (const [c, r] of CELLS) { const aa = typeof a === 'function' ? a(c, r) : a; if (aa <= 0) continue; const v = V[key(c, r)];
    txt(f2(v), Gx.ox + c * Gx.cs + Gx.cs - 5, Gx.oy + r * Gx.cs + (s >= 14 ? 12 : 10), { s, al: 'right', a: aa, c: Math.abs(v) < .005 ? C.soft : v < 0 ? C.roseInk : C.goldInk }); }
}
function planArrows(plan, a = 1, Gx = G, o = {}) { CELLS.forEach(([c, r]) => { const aa = typeof a === 'function' ? a(c, r) : a; if (aa <= 0) return; const m = plan[key(c, r)], [dx, dy] = MOVES[m], x = Gx.cx(c), y = Gx.cy(r), s = Gx.cs / 64;
  alpha(aa, () => arrow(x - dx * 9 * s, y - dy * 9 * s + 4 * s, x + dx * 15 * s, y + dy * 15 * s + 4 * s, 400 + key(c, r) * 4 + m, { c: o.c ? o.c(c, r) : C.goldInk, w: o.w || 1.6, head: 7 * s })); }); }
function wet(a = 1, Gx = G) { for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.isRock(c, r)) continue; hachure(Gx.ox + c * Gx.cs, Gx.oy + r * Gx.cs, Gx.cs, Gx.cs, C.blue, 11 * Gx.cs / 64, 300 + c * 5 + r, .25 * a); } }
function hiCells(keys, col, a) { keys.forEach(k => box(OX + (k % 5) * CS + 3, OY + Math.floor(k / 5) * CS + 3, CS - 6, CS - 6, col, a)); }
function edgeArrow(c, r, m, seed, col, r0 = 12) { const [dx, dy] = MOVES[m]; arrow(cx(c) + dx * r0, cy(r) + dy * r0, cx(c) + dx * 29, cy(r) + dy * 29, seed, { c: col, w: 1.6, head: 6 }); }
function stayLoop(c, r, mm, col) { const [dx, dy] = MOVES[mm], x = cx(c) + dx * 21, y = cy(r) + dy * 21 + 4, a0 = Math.atan2(dy, dx); ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 7, a0 - Math.PI / 2 - .3, a0 + Math.PI / 2 + .3); ctx.stroke(); ctx.restore(); }
function fan(c, r, m, t, o = {}) {
  outcomes(c, r, m).forEach((q, i) => { const a = clamp((t - i * (o.gap ?? .6)) * 2); if (a <= 0) return; const col = q.mm === m ? C.goldInk : '#3f7f7c', lab = String(+q.prob.toFixed(2));
    alpha(a, () => { if (q.nc === c && q.nr === r) { stayLoop(c, r, q.mm, col); const [dx, dy] = MOVES[q.mm]; txt(lab, cx(c) + dx * 21 + (dy ? 19 : 0), cy(r) + dy * 21 + 4 + (dx ? -15 : 0), { s: 15, c: col, al: dy ? 'left' : 'center' }); }
      else { const dx = q.nc - c, dy = q.nr - r; arrow(cx(c) + dx * (dx ? 24 : 16), cy(r) + dy * 16 + 4, cx(q.nc) - dx * 18, cy(q.nr) - dy * 18 + 4, 610 + i * 7, { c: col, w: q.mm === m ? 2.2 : 1.5 });
        txt(lab, (cx(c) + cx(q.nc)) / 2 + (dy ? 7 : 0), (cy(r) + cy(q.nr)) / 2 + 4 + (dx ? -13 : 0), { s: 15, c: col, al: dy ? 'left' : 'center' }); } }); });
}
// numbers animating through an evaluation's sweeps, d seconds per sweep (long evaluations are sped up to fit in dur)
function evalFrame(hist, t, dur) { const n = hist.length - 1, d = Math.min(.45, dur / n), k = clamp(Math.floor(t / d) + 1, 1, n);
  return { V: hist[k], k, n, done: t >= n * d }; }
// one evaluate + improve round as an animation: numbers sweep for ev seconds, then changed arrows flash teal and turn
function roundScene(R, i, t, o = {}) {
  const x = R.rounds[i], ev = o.ev ?? 4, F = o.settled ? { V: x.V, k: x.n, n: x.n, done: true } : evalFrame(x.hist, t, ev), imp = t - ev - .4;
  maze({ V: F.V, t }); nums(F.V);
  if (imp > 0) hiCells(x.changed, C.teal, .3 * clamp(imp * 2));
  planArrows(imp > .8 ? x.np : x.plan, (c, r) => (x.changed.includes(key(c, r)) && imp > 0 && imp < .8 ? 1 - clamp((imp - .4) * 3) : 1), G, { c: (c, r) => (imp > .8 && x.changed.includes(key(c, r)) ? '#2f6f6c' : C.goldInk) });
  return { F, imp };
}
const GA = L.gridWorld({ cs: 38, ox: 8, oy: 44 }), GB = L.gridWorld({ cs: 38, ox: 202, oy: 44 });
const SLV = [0, 1, 2, 3], SLN = ['all up', 'all right', 'all down', 'all left'];
function tapCell(x, y, st) { const c = Math.floor((x - OX) / CS), r = Math.floor((y - OY) / CS); if (c < 0 || c >= 5 || r < 0 || r >= 4 || G.isRock(c, r) || G.term(c, r)) { st.cell = null; return; } st.cell = [c, r]; }

const STEPS = [
 { h: 'Another way to plan', p: ['Wet floor: a tried move goes that way 8 times in 10, sliding to each side 1 in 10. Bumps leave Pip in place. Stepping **onto** the cheese pays +1, onto the cat −1, both end the walk, other steps pay 0, and γ (gamma) is 0.9.', 'We hold the rulebook, the **model**, so we plan at a desk, improving a whole plan in rounds.'],
   shot: [5, 11],
   draw(t) { wet(clamp(t)); maze({ draw: t / 1.4, pop: { cheese: ease((t - 1.6) * 3), cat: ease((t - 2) * 3) }, labels: clamp(t - 2.6), t }); if (t > 2.4) pip(cx(0), cy(3), 1, t);
     if (t > 4.5) card(['wet floor: 0.8 straight, 0.1 slides to each side', 'cheese +1, cat −1 (walk ends) · else 0 · γ = 0.9', { s: 'we hold the rulebook; Pip just follows the plan', c: C.soft }], t - 4.5, { gap: 1.1, fs: 16 }); } },
 { h: 'A bad first plan', p: ['A **plan**, or **policy**, puts one arrow in every cell, and Pip always tries that move. Any randomness now comes from the floor alone.', 'Start with a plan that is clearly bad: every arrow points up. From the start, Pip walks up to the top corner and then pushes against the wall, moving on only when it happens to slide.'],
   shot: [3, 10],
   draw(t) { wet(); maze({ t }); planArrows(RD[0].plan, (c, r) => clamp((t - .5 - key(c, r) * .08) * 3));
     const path = [[0, 3], [0, 2], [0, 1], [0, 0]], t2 = t - 3.5, k = clamp(Math.floor(t2 / .7), 0, 3), u = ease((t2 % .7) / .7 * 1.3);
     if (t2 < 0) pip(cx(0), cy(3), 1, t); else if (t2 < 2.1) { const a = path[k], b = path[Math.min(3, k + 1)]; pip(cx(a[0]), lerp(cy(a[1]), cy(b[1]), u), 0, t); } else { const bump = Math.sin(Math.PI * clamp(((t2 - 2.1) % 1) / .6)) * 10; pip(cx(0), cy(0) - bump, 0, t); txt('bump!', cx(0) + 30, cy(0) - 20, { s: 14, c: C.soft, a: clamp(t2 - 2.4) }); }
     if (t > 6) card(['plan: every arrow up', { s: 'Pip tries its arrow; only the floor adds chance', c: C.soft }], t - 6, { gap: 1.1 }); } },
 { h: 'Round 1: evaluate', p: ['First, measure this plan: compute its notebook as in the Bellman reel, now averaging over landings. Each cell: the average of reward + 0.9 × value there, using only the plan’s move.', `Sweep from zeros until nothing moves more than 0.001: ${RD[0].n} sweeps, more than value iteration needed, because its walks are long and news travels slowly. The start is worth only ${f2(V1[SK])}.`],
   shot: [4, 12],
   draw(t) { wet(); const F = evalFrame(RD[0].hist, t - .8, 7); maze({ V: F.V, t }); nums(F.V); planArrows(RD[0].plan, .45);
     caption(F.done ? `settled after ${F.n} sweeps` : `evaluation sweep ${F.k}`);
     if (t > 8.5) card(['value here = average over landings of', '      [ reward + 0.9 × value there ]   (plan’s move only)', { s: `start: ${f2(V1[SK])}`, c: C.goldInk }], t - 8.5, { gap: 1, fs: 16 }); } },
 { h: 'Reading the bad notebook', p: [`Below the cheese the plan is already right: up, to the cheese, worth ${f2(V1[K41])}. Beside the cheese it pushes into the wall; only a slide reaches the cheese: ${f2(V1[K30])}.`, `Below the cat, up walks straight into it: ${f2(V1[K32])}, and bottom-row cells lower left drift in too. Left of the cat: ${f2(V1[K21])}, as a slide onto the cat loses about what moving up gains.`],
   shot: [3, 10],
   draw(t) { wet(); const a1 = clamp(t - .5), a2 = clamp(t - 4.5); maze({ V: V1, t, hi: [[4, 1, C.gold, .3 * a1], [3, 0, C.gold, .25 * a1], [3, 2, C.rose, .3 * a2]] }); nums(V1); planArrows(RD[0].plan, .5);
     card([{ s: `below the cheese: up = cheese → ${f2(V1[K41])}`, c: C.goldInk }, { s: `beside the cheese: up = wall → ${f2(V1[K30])}`, c: C.goldInk }, { s: `below the cat: up = cat → ${f2(V1[K32])}`, c: C.roseInk }], t - 1, { gap: 1.8, fs: 16 }); } },
 { h: 'Round 1: improve', p: ['Fix the plan: score each move in every cell by this notebook, averaging reward + 0.9 × value over landings. That is its **Q**: the move once, then the plan. Point each arrow at the best Q, keeping old arrows on ties, so no flip-flops.', `Beside the cheese, up: ${f2(S30[0])}; right: 0.8 × 1 − 0.1 + 0.1 × 0.9 × ${f2(V1[K30])} = ${f2(HAND30R)}. Switch.`],
   shot: [4, 12],
   draw(t) { wet(); maze({ V: V1, t, hi: [[3, 0, C.teal, .2]] }); nums(V1, (c, r) => (c === 3 && r === 0 ? 1 : .35)); planArrows(RD[0].plan, (c, r) => (c === 3 && r === 0 ? 1 - clamp(t - 9) : .35));
     if (t < 5) fan(3, 0, 1, t - .5, { gap: .7 }); if (t > 9) alpha(clamp(t - 9), () => G.arrowIn(3, 0, 1, 1700, { c: '#2f6f6c', w: 2.2 }));
     const ls = [1, 0, 3, 2].map(m => ({ s: `Q(${NAMES[m]}) = ${f2(S30[m])}${m === 1 ? '   ← best' : m === 0 ? '   (current arrow)' : ''}`, c: m === 1 ? C.goldInk : S30[m] < 0 ? C.roseInk : C.ink }));
     card(ls, t - 4.5, { gap: 1, fs: 16, lh: 22 }); } },
 { h: 'Ten arrows change', p: [`Done in every cell, ${RD[0].changed.length} of 16 arrows change. The top row now runs right; below the cat, the arrow turns away.`, 'Some look odd. Left of the cat it now pushes into the rock: up gains little, risks sliding onto the cat; the rock is safe. Two bottom-row cells point left. They win only by the bad plan’s notebook; next round rejudges them.'],
   shot: [3, 9],
   draw(t) { wet(); const u = clamp(t - 1.5); maze({ V: V1, t }); nums(V1, .5); hiCells(RD[0].changed, C.teal, .3 * u);
     planArrows(t > 2.5 ? RD[0].np : RD[0].plan, 1, G, { c: (c, r) => (t > 2.5 && RD[0].changed.includes(key(c, r)) ? '#2f6f6c' : C.goldInk) });
     if (t > 4) card([`round 1: ${RD[0].changed.length} arrows changed (teal)`, { s: 'judged with the old plan’s notebook', c: C.soft }], t - 4, { gap: 1.1 }); } },
 { h: 'Round 2', p: [`Evaluate the new plan, starting its sweeps from the last notebook: ${RD[1].n} sweeps (from zeros it would take ${R2ZERO}). The start jumps from ${f2(V1[SK])} to ${f2(RD[1].V[SK])}.`, `Improve again with the new numbers: only ${RD[1].changed.length} arrows change. Left of the cat it points up again, below the cat it now goes down, around the cat, and one bottom-row cell turns right.`],
   shot: [5, 12],
   draw(t) { wet(); const { F, imp } = roundScene(UP, 1, t - .5, { ev: 4.5 }); caption(imp > 0 ? `improve: ${RD[1].changed.length} arrows change` : F.done ? `settled after ${F.n} sweeps` : `round 2, evaluation sweep ${F.k}`);
     txt(`start: ${f2(F.V[SK])}`, W / 2, 312, { s: 18, c: C.goldInk }); } },
 { h: 'Rounds 3 and 4', p: [`Round 3: ${RD[2].n} sweeps, start ${f2(RD[2].V[SK])}, and just ${RD[2].changed.length} arrow changes, in the bottom row. Round 4: ${RD[3].n} sweeps, start ${f2(RD[3].V[SK])}, and no arrow changes at all.`, `When improving changes nothing, we stop. This is exactly the plan that value iteration, the previous reel, found on the same floor${SAME ? '' : ' (CHECK)'}, including the detour below the cat.`],
   shot: [5, 13],
   draw(t) { wet(); const i = t < 7 ? 2 : 3, tt = t < 7 ? t - .5 : t - 7.3, { F, imp } = roundScene(UP, i, tt, { ev: 3 });
     caption(imp > 0 ? (RD[i].changed.length ? `round ${i + 1}: ${RD[i].changed.length} arrow changes` : `round ${i + 1}: nothing changes → stop`) : `round ${i + 1}, evaluation sweep ${F.k}`);
     txt(`start: ${f2(F.V[SK])}`, W / 2, 312, { s: 18, c: C.goldInk }); } },
 { h: 'Round by round', p: ['Here are the four rounds in a table. Each row is one round’s plan: how many sweeps its evaluation took, what that plan makes the start worth, and how many arrows its improvement changed.', `Round 1’s fix shows up in row 2: the start jumps from ${f2(RD[0].V[SK])} to ${f2(RD[1].V[SK])}. After that, rounds only fine-tune: ${RD.map(x => x.changed.length).join(', ')} arrows changed.`],
   shot: [3, 10],
   draw(t) { wet(); maze({ V: FIN.V, t }); nums(FIN.V, .5); planArrows(FIN.plan, .7);
     const x0 = 40, y0 = 298, cols = [x0 + 20, x0 + 95, x0 + 185, x0 + 276]; box(x0 - 10, y0 - 12, 340, 106, C.card, .95); rrect(x0 - 10, y0 - 12, 340, 106, 1900, { c: C.line, w: 1.1 });
     ['round', 'sweeps', 'start (this plan)', 'arrows changed'].forEach((s, j) => txt(s, cols[j], y0 + 2, { s: 14, c: C.soft }));
     RD.forEach((x, i) => { const a = clamp(t - .8 - i * 1.1) ; if (a <= 0) return; const y = y0 + 24 + i * 20; [String(i + 1), String(x.n), f2(x.V[SK]), String(x.changed.length)].forEach((s, j) => txt(s, cols[j], y, { s: 16, a, c: j === 2 ? C.goldInk : C.ink })); }); } },
 { h: 'Why it never gets worse', p: [`Beside the cheese, the old plan, up, is worth ${f2(V1[K30])}. Right once, then the old plan: its Q, ${f2(S30[1])}, which is why right was picked.`, `Wherever Pip lands, the new arrow passed that test too, so using it twice is no worse, then three times, and on. The new plan, always: ${f2(V2_30)}. Across the maze, ${UPS.join(', ')} numbers rose per round; none fell.`],
   shot: [4, 12],
   draw(t) { wet(); const ph = t < 7 ? 0 : 1, V = ph ? RD[1].V : V1; maze({ V, t, hi: [[3, 0, C.teal, .25]] }); nums(V, (c, r) => (c === 3 && r === 0 ? 1 : .35)); planArrows(ph ? RD[1].plan : RD[0].plan, (c, r) => (c === 3 && r === 0 ? 1 : .3));
     card([{ s: `old plan (up) here:  ${f2(V1[K30])}`, c: C.ink }, { s: `right once, then old plan:  Q = ${f2(S30[1])}  ≥ ${f2(V1[K30])}`, c: C.goldInk }, { s: 'new arrows at the next cells too:  still ≥', c: C.goldInk }, { s: `new plan, always:  ${f2(V2_30)}  (round 2 notebook)`, c: '#2f6f6c' }], t - .5, { gap: 1.8, fs: 16, lh: 21, y: 289 });
     caption(ph ? 'round 2: the new plan evaluated' : 'round 1 notebook'); } },
 { h: 'Why it stops at the best', p: ['Switches need a strictly higher Q, so each changing round strictly gains; no plan returns, and with 4,294,967,296 plans the rounds must end.', `At the end each arrow has its cell’s best Q. Beside the cheese: right ${f2(QF30[1])}, up ${f2(QF30[0])}, left ${f2(QF30[3])}, down ${f2(QF30[2])}. Any other move, used once then ours, is no better, nor twice, and so on: no plan beats ours.`],
   shot: [4, 10],
   draw(t) { wet(); maze({ V: FIN.V, t, hi: [[3, 0, C.teal, .22]] }); nums(FIN.V); planArrows(FIN.plan, .8);
     card(['each change: a strictly higher Q → strictly better', '4,294,967,296 plans → no repeats → it ends', { s: 'at the end, beside the cheese:', c: C.goldInk }, { s: `   Q right ${f2(QF30[1])} ≥ up ${f2(QF30[0])}, left ${f2(QF30[3])}, down ${f2(QF30[2])}`, c: C.goldInk }, { s: 'other moves: no better once, twice, … → best plan', c: C.goldInk }], t - .5, { gap: 1.2, fs: 15, lh: 20, y: 289 }); } },
 { h: 'Rounds versus sweeps', p: [`Policy iteration needed ${RD.length} rounds; value iteration needed ${VIS} sweeps. But each round hides a whole evaluation: ${UP.sweeps} sweeps in total. Those sweeps are cheaper, one move per cell instead of four.`, `Counting move scores: ${PI_COST.toLocaleString('en-US')} here against ${VI_COST.toLocaleString('en-US')} for value iteration, about the same. Value iteration is the extreme case: its max picks the best arrow and does one evaluation sweep at once.`],
   shot: [4, 10],
   draw(t) { const a = clamp(t - .3), b = clamp(t - 1.6);
     alpha(a, () => { wet(1, GA); GA.maze({ V: FIN.V, t }); planArrows(FIN.plan, 1, GA, { w: 1.3 }); txt('policy iteration', GA.ox + 95, 28, { s: 15, c: '#3f7f7c' }); });
     alpha(b, () => { wet(1, GB); GB.maze({ V: VIX, t }); planArrows(VIPLAN.reduce((p, m, i) => { p[key(...CELLS[i])] = m; return p; }, new Array(20).fill(-1)), 1, GB, { w: 1.3 }); txt('value iteration', GB.ox + 95, 28, { s: 15, c: C.goldInk }); });
     if (t > 3.5) card([`policy: ${UP.sweeps} sweeps × 16 + ${RD.length} improves × 64 = ${PI_COST.toLocaleString('en-US')}`, `value: ${VIS} sweeps × 64 = ${VI_COST.toLocaleString('en-US')}  (move scores)`, { s: 'same arrows at the end', c: C.soft }], t - 3.5, { y: 226, gap: 1.2, fs: 16 }); } },
 { h: 'Your turn', ctl: { label: 'first plan', values: SLV, def: 0, fmt: v => SLN[v], onChange: st => { st.cell = null; } }, tap: tapCell,
   p: [`Pick a different first plan and watch the rounds. Starting all right takes ${STARTS[1].rounds.length} rounds; all down or all left take ${STARTS[2].rounds.length === STARTS[3].rounds.length ? STARTS[2].rounds.length + ' each' : STARTS[2].rounds.length + ' and ' + STARTS[3].rounds.length}, because they reach the cheese only by sliding: their early notebooks are mostly zeros, so news spreads slowly.`, `Every start ends at the same plan${ALLSAME ? '' : ' (CHECK)'}. Tap a cell to see its four Q values in the round on screen.`],
   shot: [4, 12],
   draw(t, st) { const R = STARTS[st.ci ?? 0], t0 = st.changed ?? 0, tt = Math.max(0, t - (t0 > 0 ? t0 : 0)), per = 3.4, i = Math.min(R.rounds.length - 1, Math.floor(tt / per));
     wet(); const { F, imp } = roundScene(R, i, tt - i * per, { ev: 1.8, settled: 1 }); const x = R.rounds[i];
     txt(`round ${i + 1} of ${R.rounds.length}: settled values  ·  start ${f2(F.V[SK])}`, W / 2, 300, { s: 17, c: C.goldInk });
     if (st.cell) { const [c, r] = st.cell, k = key(c, r); box(OX + c * CS + 3, OY + r * CS + 3, CS - 6, CS - 6, C.gold, .25);
       const q = [0, 1, 2, 3].map(m => score(x.V, c, r, m)); card([{ s: 'Q:  ' + [0, 1, 2, 3].map(m => `${NAMES[m]} ${f2(q[m])}`).join('   '), c: C.ink }, { s: `plan’s move: ${NAMES[x.plan[k]]}  ·  best: ${NAMES[x.np[k]]}`, c: C.goldInk }], 9, { y: 318, fs: 15, lh: 22 }); }
     else txt(`${R.rounds.length} rounds, ${R.sweeps} evaluation sweeps in all`, W / 2, 326, { s: 16, c: C.soft });
     caption(imp > 0 && i === R.rounds.length - 1 ? 'nothing changes → done' : 'tap a cell to see its Q values'); } },
 { h: 'The big template', p: ['Evaluate a plan, then improve it using that: this loop is everywhere in RL. Loosened, it is **generalised policy iteration**: evaluate with a few sweeps, improve only some cells, or shift the plan a little, then repeat.', 'Later reels, like actor-critic, learn without a rulebook: a **critic** estimates values from experience, and an **actor** nudges the plan toward moves the critic rates higher.'],
   shot: [4, 10],
   draw(t) { const x = W / 2, y = 150, a = clamp(t - .5);
     alpha(a, () => { rrect(x - 150, y - 30, 110, 60, 2000, { fill: C.card, c: C.line }); txt('evaluate', x - 95, y - 6, { s: 18 }); txt('plan → notebook', x - 95, y + 15, { s: 13, c: C.soft });
       rrect(x + 40, y - 30, 110, 60, 2001, { fill: C.card, c: C.line }); txt('improve', x + 95, y - 6, { s: 18 }); txt('notebook → plan', x + 95, y + 15, { s: 13, c: C.soft });
       arrow(x - 40, y - 18, x + 38, y - 18, 2002, { c: C.goldInk, w: 1.6 }); arrow(x + 40, y + 18, x - 38, y + 18, 2003, { c: '#3f7f7c', w: 1.6 });
       txt('notebook', x, y - 32, { s: 13, c: C.goldInk }); txt('new plan', x, y + 32, { s: 13, c: '#3f7f7c' }); });
          if (t > 3) card(['policy iteration: each half done fully', 'generalised: each half only partly', { s: 'actor-critic: critic evaluates, actor improves', c: C.soft }], t - 3, { gap: 1.2, fs: 16 }); } },
 { h: 'Where it shows up', p: ['Ronald Howard described policy iteration in 1960, in his book on dynamic programming and Markov processes. It remains a standard way to solve planning problems whose rules are known, like stock and maintenance decisions.', 'Its biggest legacy is the loop itself: judge the current behaviour, then nudge it toward what the judgement says is better.'],
   shot: [4, 10],
   draw(t) { wet(); maze({ V: FIN.V, t }); nums(FIN.V, .5); planArrows(FIN.plan, .6);
     card(['Howard, 1960: policy iteration', 'known rules: inventory, maintenance, routes', { s: 'the loop lives on: judge, then improve', c: C.soft }], t - .5, { gap: 1.2, fs: 16 }); } },
 { h: 'Recap', p: ['**Policy iteration** starts from any plan and repeats two halves: evaluate the plan’s notebook, then point each arrow at its best-scoring move.', 'Each round is at least as good as the last, in every cell. When no arrow changes, the plan is the best one. Few rounds, but each holds a full evaluation.'],
   shot: [3, 9],
   draw(t) { wet(); maze({ V: FIN.V, t }); nums(FIN.V); planArrows(FIN.plan, .8); pip(cx(0), cy(3), 1, t);
     card(['evaluate: sweep the plan’s rule until settled', 'improve: arrow → best average score', 'repeat until no arrow changes', 'never worse; ends at the best plan'], t - .3, { gap: 1, fs: 16 }); } },
];

RLStory.register('policyiter', { title: 'Policy Iteration', track: 'Planning', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { G, run, UP, RD, FIN, VIS, VIX, VIPLAN, SAME, MONO, MONOALL, ALLSAME, STARTS, PI_COST, VI_COST, S30, HAND30R, score, nm } });
})();
