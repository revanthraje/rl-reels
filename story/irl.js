// Inverse RL as a story: instead of copying Mo's moves, Pip works out what Mo wants (a reward built from three features),
// meets the ambiguity (many rewards fit, even all-zero), picks the reward with the widest margin, and plans with it in a
// changed maze where copying fails. Dry floor, Mo deterministic; all numbers computed exactly below (no dice).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const GAM = .9, MO = '#9b7fc0', MOINK = '#5f4689', FN = ['cheese', 'cat', 'step'];
const G = L.gridWorld({}), { cx, cy, MOVES } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const G2 = L.gridWorld({ cheese: [4, 3], cat: [2, 3], rocks: [[1, 1], [3, 2]] });          // the changed maze
const no = (c, r) => r * 5 + c + 1;
function mk(s) { const pi = []; s.split('/').forEach((row, r) => row.split('').forEach((ch, c) => { pi[r * 5 + c] = { U: 0, R: 1, D: 2, L: 3 }[ch] ?? -1; })); return pi; }
const PA = mk('RRRR./U.U.U/UL.RU/URRRU');                   // Mo's rule card = policy A
const open = M => { const o = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!M.isRock(c, r) && !M.term(c, r)) o.push([c, r]); return o; };
// features of one step = of the cell it lands on: [reaches the cheese, reaches the cat, a step taken]
const phi = (M, c, r) => [M.term(c, r) > 0 ? 1 : 0, M.term(c, r) < 0 ? 1 : 0, 1];
const dotp = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
// discounted feature totals of a fixed rule card from a cell (follows it at most 300 steps); also returns the path
function totals(M, pi, from) { let [c, r] = from; const f = [0, 0, 0], P = [[c, r]]; let g = 1;
  for (let k = 0; k < 300; k++) { const [nc, nr] = M.step(c, r, pi[r * 5 + c]), p = phi(M, nc, nr); for (let i = 0; i < 3; i++) f[i] += g * p[i]; g *= GAM; c = nc; r = nr; P.push([c, r]); if (M.term(c, r)) break; }
  return { f, P, end: M.term(c, r) }; }
// the Wanderer's totals from the start, exactly (repeated sweeps of the rulebook)
const WAND = (() => { let F = {}; const O = open(G); O.forEach(([c, r]) => F[r * 5 + c] = [0, 0, 0]);
  for (let it = 0; it < 3000; it++) { const N = {}; O.forEach(([c, r]) => { const v = [0, 0, 0]; for (let m = 0; m < 4; m++) { const [nc, nr] = G.step(c, r, m), p = phi(G, nc, nr), t = G.term(nc, nr); for (let i = 0; i < 3; i++) v[i] += .25 * (p[i] + (t ? 0 : GAM * F[nr * 5 + nc][i])); } N[r * 5 + c] = v; }); F = N; }
  return F[15]; })();
// planning: Pip knows where moves lead, so for any reward guess w it computes the best value of every cell (value iteration)
function plan(M, w) { const O = open(M); let V = new Float64Array(20);
  const q = (c, r, m) => { const [nc, nr] = M.step(c, r, m); return dotp(w, phi(M, nc, nr)) + (M.term(nc, nr) ? 0 : GAM * V[nr * 5 + nc]); };
  for (let it = 0; it < 400; it++) { const N = new Float64Array(20); O.forEach(([c, r]) => { let b = -1e9; for (let m = 0; m < 4; m++) b = Math.max(b, q(c, r, m)); N[r * 5 + c] = b; }); V = N; }
  const pi = new Array(20).fill(0); O.forEach(([c, r]) => { let b = -1e9; for (let m = 0; m < 4; m++) { const v = q(c, r, m); if (v > b + 1e-9) { b = v; pi[r * 5 + c] = m; } } });
  return { pi, V }; }
// max margin: the unit-size w that maximises the smallest gap w·(Mo − other) over the other behaviours found so far
// = the shortest point of the hull of the gap vectors, scaled to size 1 (computed by many small exact steps)
function maxMargin(D) { let a = D.map(() => 1 / D.length);
  for (let it = 0; it < 20000; it++) { const p = [0, 1, 2].map(i => D.reduce((s, d, j) => s + a[j] * d[i], 0)); let jm = 0, vm = 1e9; D.forEach((d, j) => { const v = dotp(p, d); if (v < vm) { vm = v; jm = j; } });
    const q = D[jm], dd = [0, 1, 2].map(i => q[i] - p[i]), den = dotp(dd, dd); if (den < 1e-18) break; const s = clamp(-dotp(p, dd) / den); if (s < 1e-12) break; a = a.map((x, j) => x * (1 - s) + (j === jm ? s : 0)); }
  const p = [0, 1, 2].map(i => D.reduce((s, d, j) => s + a[j] * d[i], 0)), n = Math.hypot(...p); return { w: p.map(x => x / n), t: n }; }
const MOW = totals(G, PA, [0, 3]), muE = MOW.f;
const D1 = muE.map((x, i) => x - WAND[i]), R1 = maxMargin([D1]), P1 = plan(G, R1.w), T1 = totals(G, P1.pi, [0, 3]);
const D2 = muE.map((x, i) => x - T1.f[i]), R2 = maxMargin([D1, D2]), P2 = plan(G, R2.w), T2 = totals(G, P2.pi, [0, 3]);
const MATCH = Math.hypot(...T2.f.map((x, i) => x - muE[i])) < 1e-9;
const WN = R2.w.map(x => x / R2.w[0]);                       // divided by the cheese weight
// ambiguity: is Mo's walk a best walk from the start under these rewards?
const AMB = [['all zero', [0, 0, 0]], ['cheese only', [1, 0, 0]], ['usual: +1, −1', [1, -1, 0]], ['steps are fun', [0, 0, .1]]].map(([n, w]) => { const best = plan(G, w).V[15], mo = dotp(w, muE); return { n, w, best, mo, ok: mo >= best - 1e-9 }; });
// the changed maze: copying Mo's card (cells Mo has no arrow for copy the nearest arrow) vs planning with the inferred reward
const CARD2 = (() => { const pi = PA.slice(); open(G2).forEach(([c, r]) => { const k = r * 5 + c; if (pi[k] >= 0) return; let b = 1e9; open(G).forEach(([a, bb]) => { const d = Math.abs(a - c) + Math.abs(bb - r); if (d < b) { b = d; pi[k] = PA[bb * 5 + a]; } }); }); return pi; })();
const COPY2 = totals(G2, CARD2, [0, 3]), IRL2 = totals(G2, plan(G2, R2.w).pi, [0, 3]), TRUE2 = totals(G2, plan(G2, [1, -1, 0]).pi, [0, 3]);
const SAMEP = IRL2.P.map(p => p.join()).join() === TRUE2.P.map(p => p.join()).join();
const COPYSTUCK = COPY2.end === 0, COPYTO = COPY2.P.findIndex(([c, r], i) => i > 0 && COPY2.P[i - 1][0] === c && COPY2.P[i - 1][1] === r);   // first bump
const R1W = totals(G, P1.pi, [0, 3]);
const f2 = v => (v < -.005 ? '−' : v > .005 ? '+' : '') + Math.abs(v).toFixed(2), u2 = v => (v < -.005 ? '−' : '') + Math.abs(v).toFixed(2);

// ───── drawing
function teacher(x, y, face = 1, s = 1, t = 0) {            // Mo (people_world.md)
  L.mouse(x, y, face, s * 1.08, t); const c = ctx; c.save(); c.translate(x, y + Math.sin(t * 9) * 1.2); c.scale(face * s * 1.08, s * 1.08);
  c.strokeStyle = MO; c.lineWidth = 3.2; c.lineCap = 'round'; c.beginPath(); c.moveTo(5, -7); c.quadraticCurveTo(8, 1, 6, 8); c.stroke();
  c.beginPath(); c.moveTo(6, 6); c.lineTo(2, 12); c.stroke();
  c.strokeStyle = C.ink; c.lineWidth = 1.1; c.beginPath(); c.arc(13.5, -2, 3.3, 0, 7); c.stroke(); c.beginPath(); c.moveTo(10.2, -2.5); c.lineTo(7, -4); c.stroke();
  c.restore(); }
function dim(a = .6) { box(0, 0, W, 285, C.bg, a); }
function nums(M, a = 1) { if (a <= 0) return; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (M.isRock(c, r)) continue; txt(String(no(c, r)), OX + c * CS + 8, OY + r * CS + 11, { s: 12, al: 'left', c: C.soft, a }); } }
// a walker along a list of cells P (dur seconds per move); bumps nudge toward the wall
function walker(M, P, t, dur, who = 'pip', col = C.teal) { const n = P.length - 1, k = clamp(Math.floor(t / dur), 0, n), u = clamp((t - k * dur) / (dur * .7));
  for (let i = 1; i <= Math.min(k, n); i++) dot(M.cx(P[i][0]) + ((i * 7) % 11) - 5, M.cy(P[i][1]) + ((i * 5) % 11) - 5, 2.6, col, .6);
  let x, y, f = 1; if (k >= n || t < 0) { const p = P[t < 0 ? 0 : n]; x = M.cx(p[0]); y = M.cy(p[1]); } else { const a = P[k], b = P[k + 1];
    if (a[0] === b[0] && a[1] === b[1]) { x = M.cx(a[0]) + Math.sin(Math.PI * u) * 14; y = M.cy(a[1]); } else { const e = ease(u); x = lerp(M.cx(a[0]), M.cx(b[0]), e); y = lerp(M.cy(a[1]), M.cy(b[1]), e); f = Math.sign(b[0] - a[0]) || 1; } }
  if (who === 'mo') teacher(x, y + 2, f, .95, t); else L.mouse(x, y + 3, f, .95, t); return { k, done: k >= n }; }
// three weight bars: gold up for positive, rose down for negative
function wbars(w, x, y, a = 1, o = {}) { if (a <= 0) return; const h = o.h ?? 34, bw = o.bw ?? 34, gap = o.gap ?? 62; alpha(a, () => {
  rline(x - 10, y, x + gap * 2 + bw + 10, y, 990, { c: C.line, w: 1 });
  w.forEach((v, i) => { const X = x + i * gap, bh = clamp(Math.abs(v)) * h; box(X, v >= 0 ? y - bh : y, bw, bh, v >= 0 ? C.gold : C.rose, .85);
    txt(FN[i], X + bw / 2, y + (v >= 0 ? 12 : -12) + (v < 0 && bh > 10 ? 0 : 0), { s: 13, c: C.soft }); txt(f2(v), X + bw / 2, v >= 0 ? y - bh - 10 : y + bh + 11, { s: 14, c: v >= 0 ? C.goldInk : C.roseInk }); }); }); }
// a panel with a guess's three weights, over the dimmed maze
function panel(w, title, a = 1) { if (a <= 0) return; dim(.88 * a); alpha(a, () => { box(60, 40, 280, 205, C.card); rrect(60, 40, 280, 205, 995, { c: C.line, w: 1.1 }); });
  txt(title, W / 2, 58, { s: 15, c: C.soft, a }); wbars(w, 110, 152, a, { h: 60, bw: 40, gap: 80 }); }
// a table of feature totals
function table(rows, t, o = {}) { const x = o.x ?? 30, y = o.y ?? 292, w = 340, lh = o.lh ?? 22, h = 30 + rows.length * lh; if (t <= 0) return;
  alpha(clamp(t * 3), () => { box(x, y, w, h, C.card); rrect(x, y, w, h, 230 + rows.length, { c: C.line, w: 1.1 }); });
  const cols = [x + 150, x + 215, x + 285]; ['cheese', 'cat', 'steps'].forEach((s, i) => txt(s, cols[i], y + 15, { s: 14, c: C.soft, a: clamp(t * 3) }));
  rows.forEach((rw, j) => { const u = clamp((t - .6 - j * (o.gap ?? 1)) * 2.5); if (u <= 0) return; txt(rw[0], x + 12, y + 37 + j * lh, { s: 16, al: 'left', c: rw[4] || C.ink, a: u });
    for (let i = 0; i < 3; i++) txt(rw[i + 1].toFixed(2), cols[i], y + 37 + j * lh, { s: 16, c: rw[4] || C.ink, a: u }); }); }
const routeArrows = (M, P, col, a) => { if (a <= 0) return; for (let i = 0; i + 1 < P.length; i++) { const [c, r] = P[i], [nc, nr] = P[i + 1]; if (nc === c && nr === r) continue; const m = MOVES.findIndex(([dx, dy]) => dx === nc - c && dy === nr - r); alpha(a, () => M.arrowIn(c, r, m, 520 + i, { c: col, w: 1.7 })); } };

const STEPS = [
 { h: 'What does Mo want?', p: ['In the last reel, Pip copied Mo’s moves cell by cell. Copying has a second weakness: it breaks when the maze changes. Mo’s arrows point where the cheese used to be.', 'So here Pip asks a deeper question: not “what does Mo do?” but “what does Mo **want**?” Normal RL goes from a reward to behaviour. **Inverse RL** goes backwards, from behaviour to the reward that explains it.'],
   draw(t) { G.maze({ t }); teacher(cx(0), cy(3) + 2, 1, 1, t); L.mouse(cx(1) + 6, cy(3) + 5, -1, .85, t); box(0, 60, W, 140, C.bg, .9);
     const u1 = clamp(t - 2), u2 = clamp(t - 4.5); txt('RL:', 50, 105, { s: 17, al: 'left', a: u1 }); txt('reward  →  behaviour', 250, 105, { s: 19, a: u1 });
     txt('inverse RL:', 50, 155, { s: 17, al: 'left', c: MOINK, a: u2 }); txt('behaviour  →  reward', 250, 155, { s: 19, c: MOINK, a: u2 });
     card([{ s: 'copy the moves: imitation (last reel)', c: C.soft }, { s: 'find the goal behind the moves: inverse RL', c: MOINK }], t - 6.5, { gap: 1.2 }); } },
 { h: 'Why the goal travels', p: ['Think of a friend who always takes the longer road home. You might guess they care more about avoiding traffic than about distance. That guess is a reward, and it is useful: in a new city, you can predict their route without ever having seen it.', 'Moves belong to one place. A goal can be **reused** anywhere.'],
   draw(t) { const u = clamp(t - .5); alpha(u, () => { rrect(70, 40, 260, 200, 400, { c: C.line }); L.rline(90, 210, 310, 70, 401, { c: C.rose, w: 2.5 }); L.rline(90, 210, 90, 70, 402, { c: C.teal, w: 2.5 }); L.rline(90, 70, 310, 70, 403, { c: C.teal, w: 2.5 }); });
     txt('home', 320, 58, { s: 15, a: u }); txt('work', 86, 226, { s: 15, a: u }); txt('short, jammed', 230, 160, { s: 14, c: C.roseInk, a: clamp(t - 1.5) }); txt('long, clear', 150, 56, { s: 14, c: '#2f6f6c', a: clamp(t - 2.5) });
     for (let i = 0; i < 6; i++) dot(lerp(150, 290, i / 5), lerp(166, 88, i / 5), 3, C.rose, clamp(t - 1.5) * .8);
     card(['behaviour: always the long road', { s: 'guessed reward: traffic costs more than distance', c: MOINK }, { s: 'new city: predict the route from the reward', c: C.goldInk }], t - 3.5, { gap: 1.3, fs: 16 }); } },
 { h: 'Mo’s walk, dry floor', p: [`The floor is dry in this reel, so every move goes where it points. Mo walks from the start to the cheese in ${MOW.P.length - 1} steps, up the left side and along the top.`, 'Pip knows the maze: where each move leads. It does not know the rewards. Inverse RL usually assumes this: the learner can plan, but has to find out what is worth planning for.'],
   draw(t) { G.maze({ t }); routeArrows(G, MOW.P, MOINK, clamp(t - 1) * .9); walker(G, MOW.P, t - 1, 1, 'mo', MO);
     card([{ s: `Mo: ${MOW.P.length - 1} steps to the cheese`, c: MOINK }, 'Pip knows: where every move leads', { s: 'Pip doesn’t know: the rewards', c: C.roseInk }], t - 3); } },
 { h: 'Rewards from features', p: ['Pip describes every step with three **features**: does it reach the cheese, does it reach the cat, and the plain fact that a step was taken.', 'A guessed reward is one **weight** per feature. Reward of a step = cheese weight if it lands on the cheese, plus cat weight if it lands on the cat, plus the step weight, always. The maze’s usual rewards are the weights +1, −1 and 0.'],
   draw(t) { G.maze({ t, labels: 0 }); dim(.92); const u = [clamp(t - .5), clamp(t - 1.5), clamp(t - 2.5)];
     L.cheese(90, 80, 1); txt('reaches the cheese', 130, 80, { s: 17, al: 'left', a: u[0] }); L.cat(90, 140, t, 1); txt('reaches the cat', 130, 140, { s: 17, al: 'left', a: u[1] }); alpha(u[2], () => L.mouse(90, 200, 1, .8, t)); txt('takes a step (every step)', 130, 200, { s: 17, al: 'left', a: u[2] });
     txt('the usual rewards as weights:', 40, 298, { s: 15, al: 'left', c: C.soft, a: clamp(t - 4) }); wbars([1, -1, 0], 120, 356, clamp(t - 4.5), { h: 26 }); } },
 { h: 'Adding up a walk', p: [`Add each feature over the walk, with the usual discount: one more 0.9 per step of delay. Mo reaches the cheese on step ${MOW.P.length - 1}: cheese total 0.9⁶ = ${MOW.f[0].toFixed(2)}. Cat total 0. Step total 1 + 0.9 + … + 0.9⁶ = ${MOW.f[2].toFixed(2)}.`, `A walk’s return under any guess is then weight times total, added up. With the usual weights it is ${MOW.f[0].toFixed(2)}, the familiar return of a 7-step walk to the cheese.`],
   draw(t) { G.maze({ t }); routeArrows(G, MOW.P, MOINK, .6); teacher(cx(0), cy(3) + 2, 1, .9, t);
     for (let i = 1; i < MOW.P.length; i++) { const u = clamp((t - .5 - i * .5) * 2); txt((GAM ** (i - 1)).toFixed(2), cx(MOW.P[i][0]) - (i === MOW.P.length - 1 ? 12 : 0), cy(MOW.P[i][1]) + 22, { s: 13, c: C.soft, a: u }); }
     table([['Mo’s walk', ...MOW.f, MOINK]], t - 4, { gap: 1 }); if (t > 6.5) txt(`return = 1 × ${MOW.f[0].toFixed(2)} − 1 × 0 + 0 × ${MOW.f[2].toFixed(2)} = ${MOW.f[0].toFixed(2)}`, W / 2, 372, { s: 16, c: C.goldInk, a: clamp(t - 6.5) }); } },
 { h: 'Many rewards fit', p: [`Which rewards make Mo’s walk a best walk? The usual ones do. So does “only cheese counts”. And so does **all zero**: then every walk scores 0, and Mo’s is as good as any. All zero explains every behaviour at once.`, 'A reward where steps are fun does not fit: then the best walk never ends. Andrew Ng and Stuart Russell, who framed inverse RL in 2000, pointed out this **ambiguity**.'],
   draw(t) { G.maze({ t }); routeArrows(G, MOW.P, MOINK, .5); dim(.5);
     AMB.forEach((a, i) => { const u = clamp((t - .8 - i * 1.4) * 2); if (u <= 0) return; const y = 300 + i * 25; txt(a.n, 44, y + 8, { s: 16, al: 'left', a: u }); txt(`[${a.w.map(x => String(x).replace('-', '−')).join(', ')}]`, 190, y + 8, { s: 14, c: C.soft, a: u }); txt(a.ok ? 'fits' : 'doesn’t fit', 356, y + 8, { s: 16, al: 'right', c: a.ok ? '#2f6f6c' : C.roseInk, a: u }); });
     txt('weights for [cheese, cat, step]', W / 2, 285, { s: 14, c: C.soft, a: clamp(t - .5) }); } },
 { h: 'Pick the clearest win', p: ['So Pip needs a rule for choosing. A natural one: keep a list of rival behaviours, built up one at a time, and pick the reward under which Mo’s walk beats them all by the **widest margin**. All zero gives a margin of 0, so it is out. A walk with exactly Mo’s feature totals scores the same as Mo under every guess, so it is never a rival.', 'Doubling every weight changes no choice but doubles every gap, so the guesses are kept the same size: the three weights squared add up to 1.'],
   draw(t) { const y0 = 200; caption('a sketch, not data', clamp(t - 1)); txt('score under a guessed reward', W / 2, 40, { s: 16, c: C.soft, a: clamp(t - .5) });
     rline(60, y0, 340, y0, 980, { c: C.line }); const u = clamp(t - 1);
     [['Mo', 150, MOINK], ['others', 90, C.soft], ['others', 70, C.soft]].forEach(([n, h, c], i) => { box(90 + i * 90, y0 - h * u, 50, h * u, i ? C.grey : MO, .7); txt(n, 115 + i * 90, y0 + 14, { s: 14, c }); });
     if (t > 2.5) { alpha(clamp(t - 2.5), () => { arrow(160, y0 - 92, 160, y0 - 148, 981, { c: C.goldInk, w: 1.5 }); arrow(160, y0 - 148, 160, y0 - 92, 982, { c: C.goldInk, w: 1.5 }); }); txt('margin', 168, y0 - 120, { s: 15, al: 'left', c: C.goldInk, a: clamp(t - 2.5) }); }
     card(['wanted: Mo’s walk wins by the most', { s: 'all zero: every score 0, margin 0', c: C.roseInk }, { s: 'size fixed: weights² add up to 1', c: C.soft }], t - 4, { gap: 1.2 }); } },
 { h: 'Round 1: Mo against the Wanderer', p: [`Start with one other behaviour: the Wanderer, every move random. Its totals, worked out exactly from the rules: cheese ${WAND[0].toFixed(2)}, cat ${WAND[1].toFixed(2)}, steps ${WAND[2].toFixed(2)}. Mo takes far fewer steps.`, `So the reward with the widest margin says mostly “steps are bad”: cheese ${f2(R1.w[0])}, cat ${f2(R1.w[1])}, step ${f2(R1.w[2])}. The cheese weight is tiny partly because a step total is about 5 while a cheese total is at most 1: how features are scaled sways the choice.`],
   draw(t) { G.maze({ t }); routeArrows(G, MOW.P, MOINK, .5); dim(.35); table([['Mo', ...muE, MOINK], ['the Wanderer', ...WAND]], t - .5, { gap: 1.2 });
     panel(R1.w, 'guess 1: widest margin over the Wanderer', clamp(t - 4.5)); } },
 { h: 'Planning with guess 1', p: [`Pip plans the best walk under guess 1. With steps that costly, the quickest ending wins: Pip runs to the cat in ${T1.P.length - 1} steps instead of reaching the cheese in ${MOW.P.length - 1}.`, 'That is not what Mo does, so guess 1 is wrong. But it is useful: the cat run joins the list of behaviours Mo must beat.'],
   draw(t) { G.maze({ t }); routeArrows(G, MOW.P, MOINK, .35); const w = walker(G, T1.P, t - .8, .9); if (w.done) bubble('−1', cx(3), cy(1) - 16, t - .8 - (T1.P.length - 1) * .9);
     table([['Mo', ...muE, MOINK], ['the Wanderer', ...WAND], ['cat run (plan 1)', ...T1.f, C.roseInk]], t - 6, { gap: .8 }); } },
 { h: 'Round 2: beat both', p: [`Now find the reward under which Mo beats both the Wanderer and the cat run by the widest margin: cheese ${f2(R2.w[0])}, cat ${f2(R2.w[1])}, step ${f2(R2.w[2])}. Mo wins by ${R2.t.toFixed(2)} against each.`, `Pip plans with guess 2, and the plan ${MATCH ? 'has exactly Mo’s feature totals' : 'is close to Mo’s route'}. ${MATCH ? 'Mo’s route and the route along the bottom and up the right side tie: both are 7 steps to the cheese. The planner tries up first, so it takes Mo’s. Nothing is left to beat, so the search stops.' : ''}`],
   shot: [3, 13],
   draw(t) { G.maze({ t }); if (t > 5) { routeArrows(G, T2.P, C.goldInk, clamp(t - 5.5)); walker(G, T2.P, t - 6, .8); } panel(R2.w, 'guess 2: beats both by the widest margin', clamp(t - .3) * clamp(5.5 - t));
     card([{ s: `guess 2: cheese ${f2(R2.w[0])}, cat ${f2(R2.w[1])}, step ${f2(R2.w[2])}`, c: C.ink }, { s: `Mo wins by ${R2.t.toFixed(2)} against each`, c: C.soft }, { s: t > 6 + (T2.P.length - 1) * .8 ? `plan: ${T2.P.length - 1} steps to the cheese, Mo’s route` : ' ', c: C.goldInk }], t - 1.5, { gap: 1.5 }); } },
 { h: 'What Pip found', p: [`Divide guess 2 by its cheese weight: cheese +1, cat ${u2(WN[1])}, step ${u2(WN[2])}. Close to the maze’s real rewards, +1 and −1, plus a small dislike of steps that Mo’s walk cannot rule out.`, 'Why stop when the plan matches? If Pip’s plan collects the same feature totals as Mo, it earns the same return as Mo under **every** reward built from these features, Mo’s true one included. Pieter Abbeel and Andrew Ng built their 2004 method on this.'],
   draw(t) { G.maze({ t }); routeArrows(G, T2.P, C.goldInk, .6); panel(WN, 'guess 2, divided by the cheese weight', clamp(t - .5));
     table([['Mo', ...muE, MOINK], ['Pip’s plan', ...T2.f, C.goldInk]], t - 4, { gap: 1 }); } },
 { h: 'A new maze', p: ['Now the maze changes. The cheese moves to the bottom right, cell 20. The cat now sleeps in cell 18, and the rocks sit in cells 7 and 14. Mo has never walked here.', `Copying Mo’s rule card fails: Pip goes up and along the top to cell 5, where the cheese used to be. Mo’s full rule card from the old maze has no arrow there, because cell 5 held the cheese, so the nearest arrow says right: into the wall, ${COPYSTUCK ? 'again and again until the walk is stopped' : 'for a while'}.`],
   shot: [4, 12],
   draw(t) { G2.maze({ t }); nums(G2, .6); const P = COPY2.P.slice(0, Math.min(COPY2.P.length, 14)); routeArrows(G2, P.slice(0, 9), C.soft, .5); const w = walker(G2, P, t - 1, .75);
     if (t > 1 + 9 * .75) txt('bump, bump …', G2.cx(4) - 14, G2.cy(0) + 26, { s: 15, c: C.roseInk, a: clamp(t - 1 - 9 * .75) });
     card(['new maze: cheese in 20, cat in 18', { s: 'copying Mo’s moves: stuck at cell 5', c: C.roseInk }, { s: COPYSTUCK ? 'walk stopped, never reaches the cheese' : ' ', c: C.soft }], t - 3, { gap: 1.5 }); } },
 { h: 'Plan with what Mo wants', p: [`Pip plans in the new maze with guess 2. The plan goes up and around the cat: ${IRL2.P.length - 1} steps to the cheese, through cells ${IRL2.P.slice(1, -1).map(([c, r]) => no(c, r)).join(', ')}.`, `A route through cell 17 is just as short; the planner tries up first. ${SAMEP ? 'It is exactly the best route under the maze’s real rewards.' : 'It is close to the best route under the maze’s real rewards.'} The moves did not transfer; the goal did.`],
   shot: [5, 13],
   draw(t) { G2.maze({ t }); nums(G2, .5); routeArrows(G2, IRL2.P, C.goldInk, clamp(t - .5)); const w = walker(G2, IRL2.P, t - 2, .8); if (w.done) bubble('+1', G2.cx(4), G2.cy(3) - 18, t - 2 - (IRL2.P.length - 1) * .8);
     card([{ s: 'guess 2: cheese +1, cat ' + u2(WN[1]) + ', step ' + u2(WN[2]), c: C.goldInk }, `plan: ${IRL2.P.length - 1} steps, reaches the cheese`, { s: SAMEP ? 'same as the best route for the real rewards' : ' ', c: C.soft }], t - 3, { gap: 1.3 }); } },
 { h: 'Your turn', ctl: { label: 'reward', values: [0, 1, 2, 3], def: 3, fmt: v => ['all zero', 'cheese only', 'guess 1', 'guess 2'][v], onChange: st => { st.pt = st.changed; } }, tap: tapCell,
   p: ['Pick a reward guess, then tap any cell to move the cheese there. Pip plans with the guess in Pip’s usual maze and walks the plan.', 'Watch guess 1 run for the cat whenever the cheese is farther away than the cat, and all zero have no reason to go anywhere. Guess 2 finds the cheese wherever you put it in this small maze. In a much bigger maze, a far-off cheese could lose to a nearby cat.'],
   draw(t, st) { const M = st.M || G, w = WS[st.ci ?? 3]; if (!st.pl || st.plKey !== (st.ci ?? 3) + '|' + (st.key || '')) { st.pl = totals(M, plan(M, w).pi, [0, 3]); st.plKey = (st.ci ?? 3) + '|' + (st.key || ''); st.pt = st.pt ?? 0; }
     M.maze({ t }); const zero = !w.some(x => x);
     if (zero) { L.mouse(M.cx(0), M.cy(3) + 3, 1, .95, t); txt('?', M.cx(0) + 4, M.cy(3) - 24, { s: 18, c: C.soft }); caption('every walk scores 0: no reason to go anywhere'); }
     else { const P = st.pl.P.slice(0, 16), wk = walker(M, P, t - (st.pt || 0) - .3, .45); caption(wk.done ? (st.pl.end > 0 ? `cheese in ${P.length - 1} steps` : st.pl.end < 0 ? `cat in ${P.length - 1} steps` : 'never ends: going round in circles') : 'tap a cell to move the cheese'); }
     card([`reward guess: ${['all zero', 'cheese only', 'guess 1', 'guess 2'][st.ci ?? 3]}`, { s: `weights: cheese ${f2(w[0])}, cat ${f2(w[1])}, step ${f2(w[2])}`, c: C.soft }, { s: zero ? ' ' : `plan: ${st.pl.end > 0 ? 'cheese' : st.pl.end < 0 ? 'cat' : 'no ending'}, ${st.pl.end ? st.pl.P.length - 1 + ' steps' : ''}`, c: st.pl.end > 0 ? C.goldInk : C.roseInk }], 10, { y: 292, lh: 22, fs: 16 }); } },
 { h: 'Noisy experts: maximum entropy', p: ['Real experts are not perfect: they take different routes on different days. In 2008 Brian Ziebart and colleagues proposed **maximum entropy** inverse RL: assume the most random behaviour that still matches the expert’s feature totals. That works out as: each walk’s chance grows like e to the power of its return under the guessed reward.', 'The weights are tuned until these chancy walks match the expert’s totals on average. Ziebart’s team learned how taxi drivers weigh road types, speeds and turns from GPS traces.'],
   draw(t) { G.maze({ t }); dim(.2); routeArrows(G, MOW.P, MOINK, clamp(t - .5) * .8); 
     const Pb = [[0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [4, 2], [4, 1], [4, 0]]; routeArrows(G, Pb, MO, clamp(t - 1.5) * .55);
     card([{ s: 'two routes, equally good: 1 in 2 each', c: MOINK }, 'max entropy: match the totals,', '   otherwise stay as uncommitted as possible'], t - 2.5, { gap: 1.2 }); } },
 { h: 'Where it is used', p: ['Pieter Abbeel and colleagues used apprenticeship learning, built on inverse RL, to fly aerobatic manoeuvres with a model helicopter from an expert pilot’s demonstrations, around 2007 to 2010. Inverse RL is also used to model drivers and pedestrians.', 'Its costs: a planner runs inside every round, the features are chosen by hand, and some ambiguity always remains. **GAIL** (2016) never writes the reward down: a second network learns to tell the expert’s moves from the learner’s, and the learner is rewarded for fooling it.'],
   draw(t) { G2.maze({ t }); routeArrows(G2, IRL2.P, C.goldInk, .5); dim(.45);
     card(['used: helicopter aerobatics, driver models', { s: 'cost: a full plan in every round', c: C.soft }, { s: 'features picked by hand; ambiguity remains', c: C.soft }, { s: 'GAIL 2016: imitation with a hidden reward', c: MOINK }], t - .6, { gap: 1.2 }); } },
 { h: 'Recap', p: ['**Inverse RL** watches behaviour and works out the reward behind it. A reward is a reusable description of the goal: plan with it in a new maze, where copied moves fail.', 'Many rewards explain the same behaviour, even all zero, so a rule must choose: the **widest margin**, matching feature totals, or maximum entropy.'],
   draw(t) { G2.maze({ t }); routeArrows(G2, IRL2.P, C.goldInk, .7); L.mouse(G2.cx(0), G2.cy(3) + 3, 1, .95, t);
     card(['behaviour → reward: inverse RL', 'the reward transfers, copied moves don’t', 'many rewards fit: choose by margin', { s: 'match the features, match the expert', c: MOINK }], t - .3, { gap: 1 }); } },
];
const WS = [[0, 0, 0], [1, 0, 0], R1.w, R2.w];
function tapCell(x, y, st, t) { const c = Math.floor((x - OX) / CS), r = Math.floor((y - OY) / CS); if (c < 0 || c >= 5 || r < 0 || r >= 4) return;
  const M0 = st.M || G; if (M0.isRock(c, r) || (M0.cat && c === M0.cat[0] && r === M0.cat[1]) || (c === 0 && r === 3)) return;
  st.M = L.gridWorld({ cheese: [c, r], cat: G.cat, rocks: G.rocks }); st.key = c + ',' + r; st.pl = null; st.pt = t; }
RLStory.register('irl', { title: 'Inverse RL', track: 'Learning from people', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { MOW, WAND, R1, P1, T1, R2, T2, MATCH, WN, AMB, COPY2, IRL2, TRUE2, SAMEP, COPYSTUCK, plan, totals, G, G2 } });
})();
