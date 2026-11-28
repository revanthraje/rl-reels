// Value iteration as a story: holding the maze's rulebook, we sweep the best-move rule from all zeros, first on the dry
// floor, then on the mdp reel's wet floor (0.8 straight, 0.1 to each side). Every number comes from the sweeps below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, hachure, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({});                                   // shared maze: cheese (4,0), cat (3,1), rocks (1,1) (2,2), start (0,3)
const { cx, cy, key, maze, trail, pip, MOVES } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GAM = 0.9, WET = 0.2, SK = key(0, 3), NAMES = ['up', 'right', 'down', 'left'];
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);

// ───── the rulebook: tried move m with slip chance p → landings (same as the mdp reel: 1 − p straight, p/2 to each side)
function outcomes(c, r, m, p) {
  const out = [];
  for (const [mm, q] of [[m, 1 - p], [(m + 3) % 4, p / 2], [(m + 1) % 4, p / 2]]) {
    if (q <= 0) continue; const [nc, nr] = G.step(c, r, mm), o = out.find(o => o.nc === nc && o.nr === nr);
    if (o) o.prob += q; else out.push({ nc, nr, prob: q, mm });
  }
  return out;
}
// score of a move = average over landings of [ reward + γ × value there ] (ending cells are worth 0)
function score(V, c, r, m, p) { return outcomes(c, r, m, p).reduce((s, o) => { const rw = G.term(o.nc, o.nr); return s + o.prob * (rw + (rw ? 0 : GAM * V[key(o.nc, o.nr)])); }, 0); }
function bestOf(V, c, r, p) { const q = [0, 1, 2, 3].map(m => score(V, c, r, m, p)), b = Math.max(...q); return { q, b, ties: q.map((x, m) => (x > b - 1e-9 ? m : -1)).filter(m => m >= 0) }; }
function sweep(V, p) { const N = V.slice(); CELLS.forEach(([c, r]) => { N[key(c, r)] = bestOf(V, c, r, p).b; }); return N; }
const maxDiff = (a, b) => Math.max(...CELLS.map(([c, r]) => Math.abs(a[key(c, r)] - b[key(c, r)])));
const memo = {};
function VI(p) {
  if (memo[p]) return memo[p];
  const H = [new Array(20).fill(0)]; for (let i = 1; i <= 400; i++) H.push(sweep(H[i - 1], p));
  const EX = H[400], CH = H.map((V, i) => (i ? maxDiff(V, H[i - 1]) : 0));
  const S = CH.findIndex((d, i) => i > 0 && d < .001);                               // stopping rule: biggest change below 0.001
  const disp = V => CELLS.map(([c, r]) => bestOf(V, c, r, p).ties.join('')).join(',');
  // the arrows of sweep i are the winning moves used to compute H[i], i.e. scored with H[i − 1]
  const DF = disp(H[S - 1]); let AF = 0; for (let i = 1; i <= S; i++) if (disp(H[i - 1]) !== DF) AF = i + 1;   // first sweep from which the arrows equal the final ones
  const PI = new Array(20).fill(-1); CELLS.forEach(([c, r]) => { PI[key(c, r)] = bestOf(EX, c, r, p).ties[0]; });
  return (memo[p] = { H, EX, ST: H[S], CH, S, AF, PI, err: maxDiff(H[S], EX) });
}
const DRY = VI(0), WT = VI(WET);
// exact chance of ending on the cat when following arrows P on floor p (sweeps of the landing rule, no discount)
function catChance(P, p) { let V = new Array(20).fill(0); for (let it = 0; it < 4000; it++) { const N = V.slice(); CELLS.forEach(([c, r]) => { N[key(c, r)] = outcomes(c, r, P[key(c, r)], p).reduce((s, o) => { const t = G.term(o.nc, o.nr); return s + o.prob * (t ? (t < 0 ? 1 : 0) : V[key(o.nc, o.nr)]); }, 0); }); V = N; } return V; }
const KB = key(3, 2), OLDP = WT.PI.slice(); OLDP[KB] = 1;                            // wet arrows, but the old 'right' kept below the cat
const CAT_OLD = catChance(OLDP, WET)[KB], CAT_NEW = catChance(WT.PI, WET)[KB];
// one walk on floor p following arrows P, seeded
function slipWalk(c, r, P, p, R) { const tr = []; for (let k = 0; k < 60; k++) { const m = P[key(c, r)], u = R(); let mm = m; if (u >= 1 - p) mm = u < 1 - p / 2 ? (m + 3) % 4 : (m + 1) % 4;
  const [nc, nr] = G.step(c, r, mm); tr.push({ c, r, nc, nr, m, mm }); c = nc; r = nr; if (G.term(c, r)) break; } return tr; }
// 1000 walks from below the cat with each set of arrows (seed 11)
function walks(P) { const R = rng(11), out = { cat: 0, cheese: 0, steps: 0 }; for (let i = 0; i < 1000; i++) { const tr = slipWalk(3, 2, P, WET, R), l = tr[tr.length - 1]; out.steps += tr.length; if (G.term(l.nc, l.nr) < 0) out.cat++; else out.cheese++; } return out; }
const WK_OLD = walks(OLDP), WK_NEW = walks(WT.PI);
// demo walk with the wet arrows from below the cat: first seeded walk that really goes down first, slides at least once later, and reaches the cheese
const DEMO = (() => { for (let s = 1; s < 400; s++) { const tr = slipWalk(3, 2, WT.PI, WET, rng(500 + s)), l = tr[tr.length - 1]; if (G.term(l.nc, l.nr) > 0 && tr[0].mm === 2 && tr.slice(1).some(q => q.mm !== q.m) && tr.length <= 9) return { tr, seed: 500 + s }; } })();

// ───── numbers the story quotes
const f2 = v => { const s = (Math.abs(v) < .005 ? 0 : v).toFixed(2); return s.replace('-', '−'); };
const r2 = v => +(+v).toFixed(2);
const K30 = key(3, 0), K42 = key(4, 2), K33 = key(3, 3);
const DRYPATH = (() => { let [c, r] = [0, 3]; const P = [[c, r]]; while (!G.term(c, r) && P.length < 20) { [c, r] = G.step(c, r, DRY.PI[key(c, r)]); P.push([c, r]); } return P; })();
const SW1 = [0, 1, 2, 3].map(m => outcomes(3, 0, m, WET));                       // wet sweep-1 branches beside the cheese
// hand check below the cat with the 2-decimal numbers shown
const HV = k => r2(WT.ST[k]);
const QR = .8 * GAM * HV(K42) - .1 + .1 * GAM * HV(K33), QD = .8 * GAM * HV(K33) + .1 * GAM * HV(KB) + .1 * GAM * HV(K42);
const SLIPS = [0, .1, .2, .3, .4, .5];
// sweeps in which the start's arrow (scored with the notebook before the sweep) is a single 'right'
const ST_R = []; for (let k = 1; k <= WT.S; k++) { const t = bestOf(WT.H[k - 1], 0, 3, WET).ties; if (t.length === 1 && t[0] === 1) ST_R.push(k); }

// ───── drawing helpers
const ZERO = () => new Array(20).fill(0);
function nums(V, a = 1, Gx = G, s = 14, only) {
  for (const [c, r] of CELLS) { if (only && !only(c, r)) continue; const aa = typeof a === 'function' ? a(c, r) : a; if (aa <= 0) continue; const v = V[key(c, r)];
    txt(f2(v), Gx.ox + c * Gx.cs + Gx.cs - 5, Gx.oy + r * Gx.cs + (s >= 14 ? 12 : 10), { s, al: 'right', a: aa, c: Math.abs(v) < .005 ? C.soft : v < 0 ? C.roseInk : C.goldInk }); }
}
// best arrows for notebook V on floor p: one arrow if one move wins, two if two tie, '?' if three or four tie
function bestArrows(V, p, a = 1, Gx = G, o = {}) {
  CELLS.forEach(([c, r]) => { const aa = typeof a === 'function' ? a(c, r) : a; if (aa <= 0) return; const t = bestOf(V, c, r, p).ties, x = Gx.cx(c), y = Gx.cy(r), s = Gx.cs / 64;
    if (t.length > 2) { txt('?', x, y + 4 * s, { s: 20 * s + 2, c: '#c2b6a6', a: aa }); return; }
    t.forEach(m => { const [dx, dy] = MOVES[m]; alpha(aa, () => arrow(x - dx * 9 * s, y - dy * 9 * s + 4 * s, x + dx * 15 * s, y + dy * 15 * s + 4 * s, 400 + key(c, r) * 4 + m, { c: o.c || C.goldInk, w: o.w || 1.5, head: 7 * s })); }); });
}
function edgeArrow(c, r, m, seed, col, r0 = 12) { const [dx, dy] = MOVES[m]; arrow(cx(c) + dx * r0, cy(r) + dy * r0, cx(c) + dx * 29, cy(r) + dy * 29, seed, { c: col, w: 1.6, head: 6 }); }
function wet(a = 1, Gx = G) { for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.isRock(c, r)) continue; hachure(Gx.ox + c * Gx.cs, Gx.oy + r * Gx.cs, Gx.cs, Gx.cs, C.blue, 11 * Gx.cs / 64, 300 + c * 5 + r, .25 * a); } }
// a little U-turn inside a cell on the side of a blocked direction (a bump)
function stayLoop(c, r, mm, seed, col) {
  const [dx, dy] = MOVES[mm], x = cx(c) + dx * 21, y = cy(r) + dy * 21 + 4, a0 = Math.atan2(dy, dx);
  ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 7, a0 - Math.PI / 2 - .3, a0 + Math.PI / 2 + .3); ctx.stroke(); ctx.restore();
}
// arrows from (c, r) to every landing of move m with their chances
function fan(c, r, m, p, t, o = {}) {
  outcomes(c, r, m, p).forEach((q, i) => {
    const a = clamp((t - i * (o.gap ?? .6)) * 2); if (a <= 0) return; const col = q.mm === m ? C.goldInk : '#3f7f7c', lab = String(+q.prob.toFixed(2));
    alpha(a, () => {
      if (q.nc === c && q.nr === r) { stayLoop(c, r, q.mm, 620 + i * 3, col); const [dx, dy] = MOVES[q.mm]; txt(lab, cx(c) + dx * 21 + (dy ? 19 : 0), cy(r) + dy * 21 + 4 + (dx ? -15 : 0), { s: 15, c: col, al: dy ? 'left' : 'center' }); }
      else { const dx = q.nc - c, dy = q.nr - r; arrow(cx(c) + dx * (dx ? 24 : 16), cy(r) + dy * 16 + 4, cx(q.nc) - dx * 18, cy(q.nr) - dy * 18 + 4, 610 + i * 7, { c: col, w: q.mm === m ? 2.2 : 1.5 });
        txt(lab, (cx(c) + cx(q.nc)) / 2 + (dy ? (o.labLeft ? -7 : 7) : 0), (cy(r) + cy(q.nr)) / 2 + 4 + (dx ? -13 : 0), { s: 15, c: col, al: dy ? (o.labLeft ? 'right' : 'left') : 'center' }); }
    });
  });
}
// Pip during one recorded step: travels to the landing cell, or bumps toward the blocked side and comes back
function stepPos(s, u) {
  if (s.nc === s.c && s.nr === s.r) { const [dx, dy] = MOVES[s.mm], k = Math.sin(Math.PI * clamp(u)) * CS * .28; return { x: cx(s.c) + dx * k, y: cy(s.r) + dy * k }; }
  const e = ease(u); return { x: lerp(cx(s.c), cx(s.nc), e), y: lerp(cy(s.r), cy(s.nr), e) };
}
function playWalk(tr, t, dur) {
  const k = Math.floor(Math.max(0, t) / dur), done = k >= tr.length, s = tr[Math.min(k, tr.length - 1)], u = done ? 1 : (Math.max(0, t) % dur) / dur;
  trail(tr.slice(0, Math.min(tr.length, k + 1)).map(q => [q.c, q.r]), .55);
  const pos = done ? { x: cx(s.nc), y: cy(s.nr) } : stepPos(s, u); pip(pos.x, pos.y, Math.sign(s.nc - s.c) || 0, t);
  if (!done && s.mm !== s.m) txt('slide!', pos.x, pos.y - 26, { s: 15, c: '#3f7f7c' });
  return { done, k: Math.min(k, tr.length), last: tr[tr.length - 1] };
}
// chart: biggest change per sweep, on a scale where each line is ten times smaller than the one above
function changeChart(H, n, t) {
  const x0 = 70, x1 = 366, yt = 300, yb = 384, X = i => x0 + (i - .5) / 22 * (x1 - x0), Y = v => yt + (0 - Math.log10(Math.max(v, 1e-4))) / 4 * (yb - yt);
  [[1, '1'], [.1, '0.1'], [.01, '0.01'], [.001, '0.001']].forEach(([v, s], i) => { rline(x0, Y(v), x1, Y(v), 960 + i, { c: v === .001 ? C.rose : '#e6dccd', w: 1, once: true }); txt(s, x0 - 6, Y(v), { s: 12, c: v === .001 ? C.roseInk : C.soft, al: 'right' }); });
  rline(x0, yb, x1, yb, 965, { c: C.line, w: 1 }); txt('sweep number', (x0 + x1) / 2, yb + 11, { s: 12, c: C.soft }); txt('1', X(1), yb + 11, { s: 12, c: C.soft }); txt('22', X(22), yb + 11, { s: 12, c: C.soft });
  txt('biggest change in any cell', x0 + 2, yt - 10, { s: 13, c: C.soft, al: 'left' });
  for (let i = 1; i <= Math.min(n, 22); i++) { const v = H.CH[i]; if (v < 1e-4) continue; dot(X(i), Y(v), 3.2, i <= H.S ? C.gold : C.grey); if (i > 1 && H.CH[i - 1] >= 1e-4) rline(X(i - 1), Y(H.CH[i - 1]), X(i), Y(v), 970 + i, { c: '#c99a3a', w: 1.2, once: true, j: .3 }); }
}
const GA = L.gridWorld({ cs: 38, ox: 8, oy: 44 }), GB = L.gridWorld({ cs: 38, ox: 202, oy: 44 });
function tapCell(x, y, st, t) { const c = Math.floor((x - OX) / CS), r = Math.floor((y - OY) / CS); if (c < 0 || c >= 5 || r < 0 || r >= 4 || G.isRock(c, r) || G.term(c, r)) return;
  st.taps = (st.taps || 0) + 1; const p = SLIPS[st.ci ?? 2]; st.tr = slipWalk(c, r, VI(p).PI, p, rng(9000 + st.taps * 17 + key(c, r))); st.wt = t; st.wp = p; }

// sweep-1 lines beside the cheese on the wet floor (all values still 0)
const SW1L = [1, 0, 3, 2].map(m => { const o = SW1[m], parts = o.map(q => { const rw = G.term(q.nc, q.nr); return `${q.prob.toFixed(1)} × ${rw > 0 ? '1' : rw < 0 ? '(−1)' : '0'}`; }), v = score(ZERO(), 3, 0, m, WET);
  return { s: `${NAMES[m]}:  ${parts.join(' + ')}  =  ${String(+v.toFixed(2)).replace('-', '−')}${m === 1 ? '  ✓' : ''}`, c: m === 1 ? C.goldInk : v < 0 ? C.roseInk : C.ink }; });

const STEPS = [
 { h: 'Planning at a desk', p: ['Stepping **onto** the cheese pays +1, onto the cat −1, and either ends the walk. Other steps pay 0, bumps into walls or rocks leave Pip in place, and γ (gamma) is 0.9.', 'This time we hold the rulebook, its **model**: for every cell and move, where Pip lands and what the step pays. With it we plan at a desk while Pip waits.'],
   shot: [5, 11],
   say: [[["Back in Pip's maze. Watch it draw itself.", [40, 16, 320, 256]], ["Cheese plus one, cat minus one, gamma zero point nine.", [40, 16, 320, 256]]], [["This time, we hold the rulebook. Where every move lands, and what it pays.", [30, 292, 340, 87]]], "So we can plan at a desk, while Pip waits."],
   draw(t) { maze({ draw: t / 1.4, pop: { cheese: ease((t - 1.6) * 3), cat: ease((t - 2) * 3) }, labels: clamp(t - 2.6), t }); if (t > 2.4) pip(cx(0), cy(3), 1, t);
     if (t > 4.5) card(['the rulebook (model): every cell, every move →', '   where Pip lands, and what the step pays', { s: 'cheese +1, cat −1 (walk ends) · else 0 · γ = 0.9', c: C.soft }], t - 4.5, { gap: 1.1 }); } },
 { h: 'The best-move rule', p: ['The Bellman reel swept the rule for Pip’s distracted habit until it settled, then swapped the average over moves for the **best** move. That gives the best possible values, **V*** (V star): value here = the best over moves of reward + 0.9 × value where it lands.', 'Sweeping it over every cell, from all zeros, is **value iteration**. Watch the best arrows appear.'],
   shot: [3, 10],
   say: [[["Here's the rule from the Bellman reel, with the best move instead of an average.", [30, 292, 340, 110]], ["Value here is the best score over all moves.", [30, 292, 340, 110]]], [["Start with every number at zero.", [40, 16, 320, 256]], ["Then sweep that rule over every cell, again and again.", [30, 292, 340, 110]]], "That's value iteration."],
   draw(t) { maze({ V: ZERO(), t }); nums(ZERO(), clamp(t - .5)); pip(cx(0), cy(3), 1, t);
     card(['V*(here) = best over moves of', '        [ reward + 0.9 × V*(where it lands) ]', '1. start: every value 0', '2. sweep every cell with the rule  ·  3. repeat'], t - 1.2, { gap: 1.2 }); } },
 { h: 'One cell, four scores', p: ['Take the cell beside the cheese in sweep 1, with every number still 0. Right: the cheese, +1, and the walk ends, so nothing is added. Up bumps the wall: 0 + 0.9 × 0. Down is the cat: −1. Left: 0 + 0.9 × 0.', 'The best score, 1, becomes the cell’s new number, and its arrow points right, at the winning move.'],
   shot: [5, 11],
   say: [[["Take the cell beside the cheese, in sweep one.", [40, 16, 320, 256]], ["Watch the four moves get scored.", [40, 16, 320, 256]], ["Right reaches the cheese, plus one. Down is the cat, minus one. The others score zero.", [40, 16, 320, 256]]], [["The best score, one, becomes the cell's number.", [30, 292, 340, 64]], ["And its arrow points right, at the winning move.", [40, 16, 320, 256]]]],
   draw(t) { const u = clamp(t - 7.5); const V = ZERO(); V[K30] = u; maze({ V, t, hi: [[3, 0, C.teal, .2]] }); nums(V, (c, r) => (c === 3 && r === 0 ? 1 : .4));
     const pos = [[cx(3), 8], [cx(4), cy(0) + 24], [cx(3), cy(1) + 26], [cx(2), cy(0) + 22]], lab = ['up: 0 + 0.9 × 0 = 0', 'right: +1', 'down: −1', 'left: 0 + 0.9 × 0 = 0'];
     [1, 0, 2, 3].forEach((m, i) => { const a = clamp(t - .6 - i * 1.2); if (a <= 0) return; alpha(a * (m !== 1 ? 1 - .85 * u : 1 - u), () => edgeArrow(3, 0, m, 1500 + m, m === 2 ? C.roseInk : C.ink, 17)); const [x, y] = pos[m], wd = lab[m].length * 6.2 + 10;
       box(x - wd / 2, y - 8, wd, 17, m === 1 && t > 5.8 ? C.yellow : C.card, .92 * a); txt(lab[m], x, y, { s: 14, c: m === 2 ? C.roseInk : C.ink, a }); });
     if (u > 0) alpha(u, () => G.arrowIn(3, 0, 1, 1700, { c: C.goldInk, w: 2 }));
     if (t > 5.8) card(['best of  1,  0,  −1,  0   =   1', { s: 'new number 1, arrow → the winning move', c: C.goldInk }], t - 5.8, { gap: 1.4 }); } },
 { h: 'Sweep 1', p: ['Do that for all 16 cells, each using pre-sweep numbers. Only the two cells beside the cheese reach 1. Next to the cat the best score is 0: just don’t step on it.', 'Where three or four moves tie, there is no arrow yet, only a question mark. After one sweep most cells are like that: every move still scores 0.'],
   shot: [4, 10],
   say: [[["Now do that for every cell.", [40, 16, 320, 256]], ["Only the two cells beside the cheese reach one.", [40, 16, 320, 256]]], [["Beside the cat, the best is zero. Just don't step on it.", [30, 292, 340, 87]], ["Question marks mean several moves still tie.", [40, 16, 320, 256]]]],
   draw(t) { const u = clamp(t - 1), V = u > 0 ? DRY.H[1] : DRY.H[0]; maze({ V, t, hi: CELLS.filter(([c, r]) => DRY.H[1][key(c, r)] !== 0).map(([c, r]) => [c, r, C.teal, .25 * u * (1 - clamp(t - 3))]) }); nums(V); if (t > 2.5) bestArrows(DRY.H[0], 0, clamp(t - 2.5));
     caption('after sweep 1', clamp(t - 1)); if (t > 5) card(['beside the cheese: best = 1', 'beside the cat: best = 0 (any move but onto the cat)', { s: '?  = three or four moves tie  ·  two tie → two arrows', c: C.soft }], t - 5, { gap: 1.1, fs: 16 }); } },
 { h: 'The news spreads', p: [`Each sweep, the good news moves one cell further: ${f2(DRY.H[2][key(2, 0)])} two steps from the cheese, ${f2(DRY.H[3][key(1, 0)])} three steps away, and so on. A cell gets its arrow as soon as one move beats the others.`, `At sweep 7 the news reaches the start: ${f2(DRY.H[7][SK])}. Where two routes are equally short, two moves tie for best and both arrows stay, as at the start.`],
   shot: [5, 11],
   say: [[["Watch the sweeps run.", [40, 16, 320, 256]], ["Each one carries the good news one cell further from the cheese.", [40, 16, 320, 256]], ["A cell gets its arrow as soon as one move pulls ahead.", [40, 16, 320, 256]]], [["At sweep seven, the news reaches the start, zero point five three.", [30, 292, 340, 64]], "Where two routes are equally short, both arrows stay."]],
   draw(t) { const D = 1.3, k = clamp(Math.floor(t / D) + 2, 2, 7), u = ease((t - (k - 2) * D) / .6), V = DRY.H[k];
     maze({ V, t, hi: CELLS.filter(([c, r]) => DRY.H[k][key(c, r)] !== DRY.H[k - 1][key(c, r)]).map(([c, r]) => [c, r, C.teal, .25 * (1 - u)]) }); nums(V); bestArrows(DRY.H[k - 1], 0, 1); caption(`after sweep ${k}`); if (k === 7) box(OX + 3, OY + 3 * CS + 3, CS - 6, CS - 6, C.teal, .25 * clamp(t - 6 * D));
     if (t > 7.5) card([`start: ${f2(DRY.H[7][SK])} after sweep 7`, { s: 'two arrows = two moves score exactly the same', c: C.soft }], t - 7.5, { gap: 1.1 }); } },
 { h: 'When to stop', p: [`Sweep 8 changes nothing: these are V*. Dry, a number is final once it appears: three steps from the cheese, 0.9 × 0.9 = ${f2(DRY.EX[key(1, 0)])}, and no longer route beats that.`, `The arrows, the best-scoring move in each cell, form the best **policy**. Its walk to the cheese takes ${DRYPATH.length - 1} steps. Work: 8 × 16 cells × 4 moves = ${8 * 64} scores.`],
   shot: [3, 11],
   say: [[["Sweep eight changes nothing. We're done.", [30, 292, 340, 87]], ["On a dry floor, a number is final as soon as it appears.", [40, 16, 320, 256]]], [["The arrows form the best policy.", [40, 16, 320, 256]], ["Watch Pip follow them. Seven steps to the cheese.", [40, 16, 320, 256]]]],
   draw(t) { maze({ V: DRY.EX, t }); nums(DRY.EX); bestArrows(DRY.EX, 0, .8);
     const t2 = t - 3, kk = clamp(Math.floor(t2 / .6), 0, DRYPATH.length - 1), uu = ease((t2 % .6) / .6 * 1.3), a = DRYPATH[kk], b = DRYPATH[Math.min(DRYPATH.length - 1, kk + 1)];
     if (t2 > 0) { trail(DRYPATH.slice(0, kk + 1), .7); pip(lerp(cx(a[0]), cx(b[0]), uu), lerp(cy(a[1]), cy(b[1]), uu), b[0] - a[0], t); } else pip(cx(0), cy(3), 1, t);
     card(['sweep 8: biggest change 0  →  stop', `best policy: follow the arrows  ·  ${DRYPATH.length - 1} steps`, { s: '(at the start up and right tie; Pip takes up)', c: C.soft }], t - .5, { gap: 1.4, fs: 16 }); } },
 { h: 'A wet floor', p: ['Someone mops the floor, as in the MDP reel. A tried move goes that way 8 times in 10, sliding to each side 1 in 10, never backwards. Sliding into a wall or rock means staying put.', 'Rewards and γ are unchanged, so any change in numbers comes from slipping alone. The rulebook now lists every landing of each move, with its chance.'],
   shot: [3, 9],
   say: [[["Now someone mops the floor again.", [40, 16, 320, 256]], ["Watch Pip try right at the start.", [40, 16, 320, 256]], ["Eight times in ten it goes right. Otherwise it slides sideways.", [40, 16, 320, 256]]], [["Rewards and gamma don't change. Any new numbers come from slipping alone.", [30, 292, 340, 87]]]],
   draw(t) { wet(clamp(t)); maze({ t, hi: [[0, 3, C.teal, .2]] }); pip(cx(0), cy(3), 1, t); fan(0, 3, 1, WET, t - 1.5, { gap: .8 });
     if (t > 4.5) card(['try right at the start:', { s: '0.8 goes right · 0.1 slides up · 0.1 slides into the wall', c: C.ink }, { s: 'slip chance 0.2 in total, half to each side', c: C.soft }], t - 4.5, { gap: 1.1, fs: 16 }); } },
 { h: 'A move’s score is an average', p: ['A move’s score, its **Q**, averages its landings: chance × (reward + 0.9 × value there). The cell takes the best Q: Pip picks the move, not the landing.', 'Beside the cheese, sweep 1: right hits the cheese 8 times in 10, the cat 1 in 10: 0.8 − 0.1 = 0.7. Up, into the wall: 0.1, from a lucky slide onto the cheese.'],
   shot: [4, 12],
   say: [[["On a wet floor, a move's score is an average over where it might land.", [40, 16, 320, 256]], ["That's its Q.", [40, 16, 320, 256]]], [["Beside the cheese, right hits the cheese eight times in ten, and the cat one in ten.", [30, 290, 340, 120]], ["So its Q is zero point seven.", [30, 290, 340, 120]]], "The cell takes the best Q. Pip picks the move, not the landing."],
   draw(t) { wet(); const V = ZERO(); maze({ V, t, hi: [[3, 0, C.teal, .2]] }); pip(cx(3), cy(0), 1, t); fan(3, 0, 1, WET, t - .6, { gap: .8 });
     card(SW1L.concat([{ s: 'new value = best Q = 0.7  (Pip picks the move)', c: C.goldInk }]), t - 3, { gap: 1.1, fs: 15, lh: 20, y: 290 }); } },
 { h: 'Sweeping the wet floor', p: [`Numbers grow from the cheese again but don’t stop after 7 sweeps. The start is 0 until sweep 7, then ${f2(WT.H[7][SK])}, ${f2(WT.H[8][SK])}, ${f2(WT.H[9][SK])}, ${f2(WT.H[10][SK])}, still creeping at sweep 14.`, 'After 7 sweeps each number is the best expected from walks cut off at 7 steps; each sweep allows one more. Slides make walks long, so extra steps keep adding less and less, because of γ.'],
   shot: [4, 12],
   say: [[["Watch the wet floor sweeps.", [40, 16, 320, 256]], ["Numbers grow out from the cheese again, but they don't stop after seven.", [40, 16, 320, 256]]], [["The start keeps creeping up, sweep after sweep.", [100, 298, 200, 28]]], "Slides make walks longer, so each extra step adds a little more, and less each time."],
   draw(t) { const D = .8, k = clamp(Math.floor(t / D) + 1, 1, WT.S), V = WT.H[k];
     wet(); maze({ V, t, hi: [[0, 3, C.teal, .22]] }); nums(V); bestArrows(WT.H[k - 1], WET, .7); caption(`wet floor, after sweep ${k}`);
     txt(`start: ${f2(WT.H[k][SK])}`, W / 2, 312, { s: 18, c: k >= 7 ? C.goldInk : C.soft }); } },
 { h: 'When to stop, now', p: ['Each sweep’s **biggest change** in any cell: across, sweep number; up, the change, each line ten times smaller. Grey dots: unneeded sweeps.', `Stop below 0.001: sweep ${WT.S}. Each later change is at most 0.9 times the one before, so together they add at most 0.001 × (0.9 + 0.81 + …) = 9 × 0.001. Against 400 sweeps, the real gap is ${WT.err < .001 ? 'under 0.001' : WT.err.toFixed(4)}.`],
   shot: [3, 13],
   say: [[["This chart shows each sweep's biggest change.", [40, 285, 345, 115]], ["Each line down is ten times smaller.", [40, 285, 345, 115]]], [["Stop once it's below a thousandth. That's sweep sixteen.", [40, 285, 345, 115]]], "The remaining changes shrink by zero point nine each time, so together they can't add up to much."],
   draw(t) { const n = clamp(Math.floor((t - 2) / .4) + 1, 0, 22), k = clamp(n, 0, WT.S); wet(); maze({ V: WT.H[k], t }); nums(WT.H[k]); if (k > 0) bestArrows(WT.H[k - 1], WET, .5);
     changeChart(WT, n, t); if (n >= WT.S) txt(`stop: sweep ${WT.S}`, 300, 291, { s: 14, c: C.roseInk, a: clamp((t - 2 - WT.S * .4) * 2) }); } },
 { h: 'The arrows settle first', p: [`The arrows stop changing at sweep ${WT.AF}; the numbers creep on until sweep ${WT.S}. The start points right in sweeps ${ST_R[0]} to ${ST_R[ST_R.length - 1]}, then flips to up for good.`, 'The best move often shows before its exact value does. Mid-sweep you cannot tell the arrows are final, so the stopping rule watches the numbers. The next reel, policy iteration, works on arrows directly.'],
   shot: [4, 12],
   say: [[["Watch the arrows, sweep by sweep.", [40, 16, 320, 256]], ["Teal cells mark arrows that just changed.", [40, 16, 320, 256]]], [["The arrows are final by sweep ten.", [40, 290, 320, 75]], ["The numbers keep creeping until sweep sixteen.", [40, 290, 320, 75]]], "The best move often shows before its exact value. The next reel uses that."],
   draw(t) { const D = .75, k = clamp(Math.floor(t / D) + 1, 1, WT.S); wet(); maze({ V: WT.H[k], t }); nums(WT.H[k], .6);
     const prevT = CELLS.map(([c, r]) => (k > 1 ? bestOf(WT.H[k - 2], c, r, WET).ties.join('') : '0123')), chg = CELLS.filter(([c, r], i) => bestOf(WT.H[k - 1], c, r, WET).ties.join('') !== prevT[i]);
     chg.forEach(([c, r]) => box(OX + c * CS + 3, OY + r * CS + 3, CS - 6, CS - 6, C.teal, .3)); bestArrows(WT.H[k - 1], WET, 1);
     txt(`sweep ${k}`, W / 2, 304, { s: 18 }); txt(k < WT.AF ? `${chg.length} arrow${chg.length === 1 ? '' : 's'} appeared or changed (teal)` : `arrows final since sweep ${WT.AF}`, W / 2, 330, { s: 16, c: k < WT.AF ? '#3f7f7c' : C.goldInk });
     txt(`biggest number change: ${WT.CH[k] < .001 ? WT.CH[k].toFixed(4) : WT.CH[k].toFixed(3)}`, W / 2, 356, { s: 16, c: C.soft }); } },
 { h: 'Dry arrows, wet arrows', p: ['Left, the best arrows on the dry floor; right, the wet floor. The clearest change is right below the cat: dry, the best move was right; wet, it is down, away from the cat.', 'Where the dry floor had two equally short routes, slipping makes one better by a hair, so ties disappear. At the start, up beats right only in the third decimal.'],
   shot: [4, 10],
   say: [[["Dry floor arrows on the left, wet floor on the right.", [4, 20, 388, 180]], ["Look at the circled cell, right below the cat.", [4, 20, 388, 180]]], [["Dry, the best move was right. Wet, it's down, away from the cat.", [30, 226, 340, 64]], "And the ties disappear. Slipping makes one route better by a hair."]],
   draw(t) { const a = clamp(t - .3), b = clamp(t - 1.6);
     alpha(a, () => { GA.maze({ t }); bestArrows(DRY.EX, 0, 1, GA, { w: 1.3 }); txt('dry floor', GA.ox + 95, 28, { s: 15, c: C.goldInk }); });
     alpha(b, () => { wet(1, GB); GB.maze({ t }); bestArrows(WT.ST, WET, 1, GB, { w: 1.3 }); txt('wet floor', GB.ox + 95, 28, { s: 15, c: '#3f7f7c' }); });
     if (t > 3.5) { const u = clamp(t - 3.5); [GA, GB].forEach(g => rrect(g.ox + 3 * g.cs + 2, g.oy + 2 * g.cs + 2, g.cs - 4, g.cs - 4, 1800, { c: C.rose, w: 2 * u })); }
     if (t > 4.5) card(['below the cat:  dry → right,   wet → down', { s: 'dry ties (two arrows) become single arrows', c: C.soft }], t - 4.5, { y: 226, gap: 1.2 }); } },
 { h: 'Why go down?', p: ['Below the cat, going right risks a slide up, onto the cat, 1 time in 10. Going down risks nothing: its slides either bump into the rock and stay, or land one cell to the right.', `The detour costs two extra steps, yet the averages favour it: ${f2(QD)} for down against ${f2(QR)} for right.`],
   shot: [4, 12],
   say: [[["Why go down? Watch the slides from right below the cat.", [40, 16, 320, 256]], ["Going right risks a slide up, onto the cat.", [40, 16, 320, 256]]], [["Going down risks nothing. Its slides bump the rock or go right.", [40, 16, 320, 256]]], [["The detour is longer, but the averages favour it.", [30, 292, 340, 64]]]],
   draw(t) { wet(); maze({ V: WT.ST, t, hi: [[3, 2, C.teal, .2]] }); nums(WT.ST, (c, r) => ((c === 4 && r === 2) || (c === 3 && r === 3) || (c === 3 && r === 2) ? 1 : .3)); pip(cx(3), cy(2), 1, t);
     if (t < 5) fan(3, 2, 1, WET, t - .5, { gap: .7 }); else fan(3, 2, 2, WET, t - 5, { gap: .7, labLeft: 1 });
     const v = k => f2(HV(k));
     card([{ s: `right: 0.8 × 0.9 × ${v(K42)} + 0.1 × (−1) + 0.1 × 0.9 × ${v(K33)} = ${f2(QR)}`, c: C.ink }, { s: `down: 0.8 × 0.9 × ${v(K33)} + 0.1 × 0.9 × ${v(KB)} + 0.1 × 0.9 × ${v(K42)} = ${f2(QD)}  ✓`, c: C.goldInk }, { s: 'right risks the cat; down only risks a bump', c: C.soft }], t - 2, { gap: 3, fs: 14, lh: 21 }); } },
 { h: 'Does it matter?', p: [`Send Pip from below the cat 1000 times. With the dry floor’s arrow, right, ${WK_OLD.cat} end on the cat; only this cell differs. With the new arrow, down, only ${WK_NEW.cat}, though walks get longer.`, 'Why not zero? Both cells beside the cheese touch the cat, so the last step always risks sliding onto it. The policy can’t remove that risk, only stop adding more.'],
   shot: [5, 13],
   say: [[["Does it matter? Watch one walk with the new arrows.", [40, 16, 320, 256]]], [["From below the cat, the old arrow sent two hundred and seven walks in a thousand onto the cat.", [30, 292, 340, 87]], ["The new one, only a hundred and fourteen.", [30, 292, 340, 87]]], "The last step always carries some risk. The policy just stops adding more."],
   draw(t) { wet(); maze({ V: WT.ST, t }); bestArrows(WT.ST, WET, .45); const m = playWalk(DEMO.tr, t - .5, .7);
     if (m.done) bubble('+1', cx(4), cy(0) - 18, t - .5 - DEMO.tr.length * .7);
     caption(m.done ? `one walk with the new arrows: cheese after ${DEMO.tr.length} steps` : `one walk with the new arrows, step ${m.k + 1}`);
     if (t > 2) card([`old arrow (right):  ${WK_OLD.cat} of 1000 walks end on the cat`, `new arrow (down):  ${WK_NEW.cat} of 1000`, { s: 'every other cell: the wet-floor arrows', c: C.soft }], t - 2, { gap: 1.3, fs: 16 }); } },
 { h: 'Your turn', ctl: { label: 'slip', values: SLIPS, def: 2, fmt: v => String(v), onChange: st => { st.tr = null; } }, tap: tapCell,
   p: ['Drag the **slip chance**, the total sideways chance, so far 0.2. Value iteration reruns from zeros. More slipping means longer walks, so numbers creep for more sweeps, and arrows bend away from the cat. At 0.3 the start takes the bottom route; at 0.4 the cells beside the cheese push into the wall, awaiting a lucky slide.', 'Tap a cell to start Pip there.'],
   shot: [3],
   say: [[["Your turn. Drag the slip chance.", [40, 16, 320, 256]], ["Value iteration reruns from zeros.", [40, 16, 320, 256]]], [["More slipping, more sweeps, and arrows that bend away from the cat.", [40, 16, 320, 256]], ["Tap a cell to send Pip from there.", [60, 392, 280, 24]]]],
   draw(t, st) { const p = SLIPS[st.ci ?? 2], R1 = VI(p), V = R1.ST;
     if (p > 0) wet(Math.min(1, p / .2)); maze({ V, t }); nums(V); bestArrows(R1.ST, p, st.tr && st.wp === p ? .45 : 1);
     if (st.tr && st.wp === p) { const m = playWalk(st.tr, t - st.wt, .55); const e = G.term(m.last.nc, m.last.nr); caption(m.done ? `${e > 0 ? 'cheese' : e < 0 ? 'cat' : 'still walking'} after ${st.tr.length} steps` : `step ${m.k + 1}`); }
     else caption('tap a cell to send Pip from there');
     txt(p === 0 ? `dry floor: settled after ${R1.S} sweeps` : `slip ${p}: settled after ${R1.S} sweeps`, W / 2, 304, { s: 17, c: C.soft }); } },
 { h: 'The catch', p: ['Value iteration needs the whole rulebook, and every sweep visits every cell, move and landing. Here that is 16 cells. Go has so many legal positions that writing their number takes 171 digits: no table could hold them.', 'And often nobody hands you a rulebook at all. Later reels learn values from experience instead, and replace the table with a network.'],
   shot: [4, 10],
   say: [[["The catch? Every sweep visits every cell, every move, every landing.", [30, 292, 340, 87]], ["Here that's sixteen cells.", [40, 16, 320, 256]]], [["Go has so many positions that just writing the count takes a hundred and seventy one digits.", [30, 292, 340, 87]]], "And often there's no rulebook at all. Later reels learn from experience instead."],
   draw(t) { wet(); maze({ V: WT.ST, t }); nums(WT.ST, .5); bestArrows(WT.ST, WET, .4);
     card(['each sweep: every cell × every move × every landing', 'Pip’s maze: 16 cells.   Go: a 171-digit number', { s: 'no rulebook? learn from experience (later reels)', c: C.soft }], t - .5, { gap: 1.3, fs: 16 }); } },
 { h: 'Where it shows up', p: ['Richard Bellman introduced this step-by-step computation in the 1950s as **dynamic programming**. Value iteration is still a standard solver when the rules are known and small enough to list: inventory planning, maintenance schedules, robot routes on a known map.', 'Q-learning aims at a similar target, reward + γ × the best Q of the next cell, but estimates it from experience.'],
   shot: [4, 10],
   say: ["Richard Bellman called this dynamic programming, in the nineteen fifties.", [["It's still a standard solver when the rules are known and small.", [30, 292, 340, 87]], ["Inventory, maintenance, robots on a known map.", [30, 292, 340, 87]]], [["Q-learning aims at the same target, but learns it from experience.", [30, 292, 340, 87]]]],
   draw(t) { wet(); maze({ V: WT.ST, t }); nums(WT.ST, .5); bestArrows(WT.ST, WET, .6);
     card(['dynamic programming: Bellman, 1950s', 'known, small rules: inventory, maintenance, robot maps', { s: 'Q-learning: the same target, learned from experience', c: C.soft }], t - .5, { gap: 1.2, fs: 16 }); } },
 { h: 'Recap', p: ['Given the rulebook, **value iteration** sweeps one rule over every cell, from zeros: value here = the best Q, the top move’s average of reward + γ × landing value. It closes in on V*.', 'Stop when the biggest change is tiny; the best arrows are the best policy. On a slippery floor it bends away from danger, because the averages count the slides.'],
   shot: [3, 9],
   say: [[["So, value iteration sweeps the best move rule over every cell, from zeros.", [30, 292, 340, 110]], ["Stop when the changes are tiny.", [30, 292, 340, 110]]], ["The best arrows are the best policy. And on a slippery floor, they steer away from danger."]],
   draw(t) { wet(); maze({ V: WT.ST, t }); nums(WT.ST); bestArrows(WT.ST, WET, .8); pip(cx(0), cy(3), 1, t);
     card(['V*(s) = max over moves of average of', '            [ reward + γ × V*(landing) ]', 'sweep from zeros; stop when change < 0.001', 'best policy = the best move in each cell'], t - .3, { gap: 1, fs: 16 }); } },
];

RLStory.register('valueiter', { title: 'Value Iteration', track: 'Planning', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { G, VI, DRY, WT, CAT_OLD, CAT_NEW, WK_OLD, WK_NEW, DEMO, QR, QD, SW1L, DRYPATH, outcomes, score, bestOf } });
})();
