// Policies as a story: Pip swaps a memorised list for a rule per cell, compares two rule cards by running them many times,
// and learns what a random (stochastic) policy is. All numbers come from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
// the shared Pip maze, with the wet floor from the MDP reel: a move goes as intended 8 times in 10, slides sideways 1 in 10 each way
const G = L.gridWorld({});
const { cx, cy, maze, pip, MOVES } = G, CS = G.cs;
const UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, SLIP = .2, GAMMA = .9, CAP = 100, NW = 5000, NAMES = ['up', 'right', 'down', 'left'];
const fmtN = n => n.toLocaleString('en-US'), F2 = v => (v < 0 ? '−' : '') + Math.abs(v).toFixed(2), F3 = v => (v < 0 ? '−' : '') + Math.abs(v).toFixed(3);

function slipMove(c, r, m, p, R) { const u = R(); let mm = m; if (u >= 1 - p) mm = u < 1 - p / 2 ? (m + 3) % 4 : (m + 1) % 4; const [nc, nr] = G.step(c, r, mm); return { c, r, m, mm, nc, nr }; }
// a policy = one move per cell (array of 20, index r * 5 + c); rows written top to bottom, '.' for rocks and ending cells
function mk(s) { const pi = []; s.split('/').forEach((row, r) => row.split('').forEach((ch, c) => { pi[r * 5 + c] = { U: 0, R: 1, D: 2, L: 3 }[ch] ?? 0; })); return pi; }
const PA = mk('RRRR./U.U.U/UL.RU/URRRU');          // policy A: shortest routes, start goes up
const PB = PA.slice(); PB[1 * 5 + 2] = DOWN;         // policy B: A with one arrow changed, at (2,1) it points down into the rock
const OPEN = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.isRock(c, r) && !G.term(c, r)) OPEN.push([c, r]);
// one walk from the start with its own dice (seed). eps = chance of ignoring the arrow and picking one of the 4 moves at random.
// return = γ^(steps − 1) × (+1 cheese, −1 cat); a walk still going after 100 steps is stopped and scores 0
function walk(pi, seed, eps = 0, from = [0, 3]) {
  const R = rng(seed); let [c, r] = from; const tr = [];
  for (let k = 1; k <= CAP; k++) {
    let m = pi[r * 5 + c], rnd = false; if (eps && R() < eps) { m = Math.floor(R() * 4); rnd = true; }
    const s = slipMove(c, r, m, SLIP, R); s.rnd = rnd; tr.push(s); c = s.nc; r = s.nr;
    const e = G.term(c, r); if (e) return { tr, ret: e * GAMMA ** (k - 1), end: e };
  }
  return { tr, ret: 0, end: 0 };
}
// many walks (seeds 1000 + i, the same dice for every policy): average return, outcome counts, running average
const memo = {};
function judge(pi, eps = 0, n = NW) {
  const key = pi.join('') + '|' + eps + '|' + n; if (memo[key]) return memo[key];
  let s = 0, s2 = 0; const cnt = [0, 0, 0], run = [], rets = [];
  for (let i = 0; i < n; i++) { const w = walk(pi, 1000 + i, eps); s += w.ret; s2 += w.ret * w.ret; rets.push(w.ret); cnt[w.end > 0 ? 0 : w.end < 0 ? 1 : 2]++; run.push(s / (i + 1)); }
  const avg = s / n; return (memo[key] = { avg, cnt, run, rets, se: Math.sqrt((s2 / n - avg * avg) / n) });
}
// the memorised list from the MDP reel (seed 5, slip 0.2, 1000 runs): how many reach the cheese, and run 2, which slips on move 2
const LIST = [UP, UP, UP, RIGHT, RIGHT, RIGHT, RIGHT];
const LISTRUNS = (() => { const R = rng(5), runs = []; let ch = 0;
  for (let i = 0; i < 1000; i++) { let c = 0, r = 3; const tr = []; for (const m of LIST) { const s = slipMove(c, r, m, SLIP, R); tr.push(s); c = s.nc; r = s.nr; if (G.term(c, r)) break; } if (G.term(c, r) > 0) ch++; if (i < 5) runs.push(tr); }
  return { ch, run2: runs[1] }; })();
const W1 = { a: walk(PA, 1000), b: walk(PB, 1000) }, W2 = { a: walk(PA, 1001), b: walk(PB, 1001) };
const RECOVER = walk(PA, 1015);                      // a walk of A with a sideways slip on move 2 that the arrows repair
const JA = judge(PA), JB = judge(PB);
// noise: standard error of each average, and of the A − B gap (paired, since both use the same dice)
const SED = (() => { const d = JA.rets.map((v, i) => v - JB.rets[i]), m = d.reduce((a, b) => a + b, 0) / NW; return Math.sqrt((d.reduce((a, b) => a + (b - m) ** 2, 0) / NW) / NW); })();
// exact values from the rulebook: evaluate a policy, and find the best policy by value iteration (start cell)
function outs(c, r, m) { return [[m, 1 - SLIP], [(m + 3) % 4, SLIP / 2], [(m + 1) % 4, SLIP / 2]].map(([mm, q]) => { const [a, b] = G.step(c, r, mm); return [a, b, q]; }); }
function qv(V, c, r, m) { let v = 0; for (const [a, b, q] of outs(c, r, m)) { const rw = G.term(a, b); v += q * (rw + (rw ? 0 : GAMMA * V[b * 5 + a])); } return v; }
const EXACT = (() => { const VA = new Float64Array(20), VS = new Float64Array(20), opt = PA.slice();
  for (let it = 0; it < 3000; it++) OPEN.forEach(([c, r]) => { VA[r * 5 + c] = qv(VA, c, r, PA[r * 5 + c]); let bv = -9; for (let m = 0; m < 4; m++) { const v = qv(VS, c, r, m); if (v > bv + 1e-12) { bv = v; opt[r * 5 + c] = m; } } VS[r * 5 + c] = bv; });
  return { A: VA[15], best: VS[15], opt, diff: OPEN.filter(([c, r]) => opt[r * 5 + c] !== PA[r * 5 + c]).map(([c, r]) => r * 5 + c + 1) }; })();
const JOPT = judge(EXACT.opt);
const EPS = [0, .1, .2, .5, 1];
const NPOL = 4 ** 16;

// ───── drawing helpers ─────
function dim(a = .7, y0 = 0, y1 = 285) { box(0, y0, W, y1 - y0, C.bg, a); }
function ghost(M, c, r, s, a = 1, o = {}) { txt(s, M.cx(c) + (o.dx ?? 0), M.cy(r) + (o.dy ?? 20), { s: o.s || 14, c: o.c || '#9a8f84', a }); }
// policy arrow in a cell of grid M (scaled to the cell size)
function arrowAt(M, c, r, m, seed, o = {}) { const [dx, dy] = MOVES[m], k = M.cs / 64, y = M.cy(r) + (o.dy ?? 0); arrow(M.cx(c) - dx * 12 * k, y - dy * 12 * k, M.cx(c) + dx * 15 * k, y + dy * 15 * k, seed, Object.assign({ c: C.goldInk, w: 1.6, head: 7 * Math.max(.7, k) }, o)); }
function policyArrows(M, pi, t, o = {}) { OPEN.forEach(([c, r], i) => { if (o.skip && o.skip(c, r)) return; const a = clamp((t - i * (o.gap ?? 0)) * 3) * (o.a ?? 1); if (a <= 0) return; const hi = o.hi && o.hi(c, r);
  alpha(a, () => arrowAt(M, c, r, pi[r * 5 + c], 400 + i * 7 + (o.seed || 0), { c: hi ? C.roseInk : (o.col || C.goldInk), w: hi ? 2.2 : 1.6, dy: o.dy })); }); }
// where a walker is during transition list tr at time t (dur seconds per step); bumps go a little way toward the wall and back
function walkAt(M, tr, t, dur) {
  const k = Math.floor(t / dur);
  if (k >= tr.length) { const l = tr[tr.length - 1]; return { x: M.cx(l.nc), y: M.cy(l.nr), k: tr.length, done: true, s: l }; }
  if (k < 0) return { x: M.cx(tr[0].c), y: M.cy(tr[0].r), k: 0, s: null };
  const s = tr[k], u = clamp((t - k * dur) / (dur * .7));
  if (s.nc === s.c && s.nr === s.r) { const [dx, dy] = MOVES[s.mm], b = Math.sin(Math.PI * u) * M.cs * .28; return { x: M.cx(s.c) + dx * b, y: M.cy(s.r) + dy * b, k, s, u }; }
  const e = ease(u); return { x: lerp(M.cx(s.c), M.cx(s.nc), e), y: lerp(M.cy(s.r), M.cy(s.nr), e), k, s, u };
}
function trailOf(M, tr, k, a = .6) { for (let i = 0; i < Math.min(k, tr.length); i++) { const s = tr[i]; dot(M.cx(s.nc) + ((i * 7) % 11) - 5, M.cy(s.nr) + ((i * 5) % 11) - 5, 2.4, C.teal, a); } }
function wet(M = G, a = 1) { for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (M.isRock(c, r)) continue; L.hachure(M.ox + c * M.cs, M.oy + r * M.cs, M.cs, M.cs, C.blue, 11 * M.cs / 64 + 4, 300 + c * 5 + r, .22 * a); } }
const MA = L.gridWorld({ cs: 34, ox: 22, oy: 64 }), MB = L.gridWorld({ cs: 34, ox: 208, oy: 64 });
function mouseAt(M, x, y, face, t) { L.mouse(x, y + 2, face || 1, .95 * M.cs / 64, t); }
// two mini mazes, A left and B right, with headers
function twoMazes(t, o = {}) {
  [[MA, PA, 'policy A', C.goldInk], [MB, PB, 'policy B', C.roseInk]].forEach(([M, pi, name, col], j) => {
    wet(M); M.maze({ t }); policyArrows(M, pi, 9, { a: o.arrowA ?? .8, hi: j === 1 ? (c, r) => c === 2 && r === 1 : null, seed: j * 50 });
    txt(name, M.ox + 85, 46, { s: 17, c: col });
  });
}
// a random policy for the "how many" step: arrows chosen by a seeded die
function randomPolicy(seed) { const R = rng(seed), pi = PA.slice(); OPEN.forEach(([c, r]) => { pi[r * 5 + c] = Math.floor(R() * 4); }); return pi; }

const STEPS = [
 { h: 'The list that broke', p: ['In the last reel, Pip memorised the shortest route as a list of seven moves. On the wet floor, only 229 of 1000 runs reached the cheese.', 'Here is why. On move 2, Pip slid right into a new cell. The list still said “up” and then “right”, which here point into rocks. In this run, Pip bumped five times, and the list ran out.'],
   shot: [3, 10.5],
   draw(t) { wet(); maze({ t }); let c = 0, r = 3; LIST.forEach((m, i) => { alpha(.4, () => G.arrowIn(c, r, m, 900 + i * 7, { c: C.goldInk, w: 1.4 })); [c, r] = G.step(c, r, m); });
     const tr = LISTRUNS.run2, P = walkAt(G, tr, t - 1, .9); trailOf(G, tr, P.k); pip(P.x, P.y, P.s && P.s.nc !== P.s.c ? Math.sign(P.s.nc - P.s.c) : 0, t);
     if (P.s && P.s.mm !== P.s.m && P.u > .3) bubble('slip!', cx(P.s.c) + 20, cy(P.s.r) - 24, (t - 1 - P.k * .9) - .2, '#3f7f7c');
     if (P.s && !P.done && P.s.nc === P.s.c && P.s.nr === P.s.r) bubble('bump', cx(P.s.c), cy(P.s.r) - 26, (t - 1 - P.k * .9) - .2, C.soft);
     caption(P.done ? 'list used up: Pip never reached the cheese' : P.s ? `move ${P.k + 1} of 7: the list says ${NAMES[P.s.m]}` : 'faint arrows: the memorised list');
     if (t > 8.5) card([{ s: 'a list only works if nothing goes wrong', c: C.ink }, { s: 'Pip needs a move for wherever it lands', c: C.goldInk }], t - 8.5, { gap: 1 }); } },

 { h: 'A rule for every cell', p: ['So Pip writes a rule card instead: for **every cell**, one move. This is a **policy**, written **π (pi)**. π of s is the action the policy picks in state s.', 'Why is the cell enough? The maze is Markov: the cell tells Pip everything about what can happen next. So what Pip must find, per cell, is a move. This card follows shortest routes; at the start, where up and right tie, it picks up. Call it policy A.'],
   shot: [3, 9],
   draw(t) { wet(); maze({ t }); policyArrows(G, PA, t - .5, { gap: .18 }); pip(cx(0), cy(3), 1, t);
     if (t > 4.5) card(['π(s) = the move to make in state s', { s: '16 cells, 16 arrows: one move each', c: C.goldInk }, { s: 'rocks need none; cheese and cat end the walk', c: C.soft }], t - 4.5, { gap: 1 }); } },

 { h: 'Slip, then recover', p: ['Watch policy A on the wet floor. On move 2, Pip slides right instead of going up. No problem: the new cell has its own arrow, pointing back to the route.', 'Pip reaches the cheese in 9 steps instead of 7. A policy keeps working after surprises, because it never assumes where Pip will be. It only asks: where am I now?'],
   shot: [2.3, 4.5, 10],
   draw(t) { wet(); maze({ t }); policyArrows(G, PA, 9, { a: .55 }); const tr = RECOVER.tr, d = .85, P = walkAt(G, tr, t - .8, d); trailOf(G, tr, P.k);
     if (P.s && !P.done) box(G.ox + P.s.c * CS + 3, G.oy + P.s.r * CS + 3, CS - 6, CS - 6, C.teal, .18);
     pip(P.x, P.y, P.s && P.s.nc !== P.s.c ? Math.sign(P.s.nc - P.s.c) : 0, t);
     if (P.s && P.s.mm !== P.s.m && P.u > .3 && !P.done) bubble('slip!', cx(P.s.c) + 20, cy(P.s.r) - 24, (t - .8 - P.k * d) - .2, '#3f7f7c');
     if (P.done) { bubble('+1', cx(4), cy(0) - 20, t - .8 - tr.length * d); caption(`cheese after ${tr.length} steps`); }
     else caption(P.s ? `step ${P.k + 1}: the arrow here says ${NAMES[P.s.m]}${P.s.mm !== P.s.m ? ', but Pip slides ' + NAMES[P.s.mm] : ''}` : 'shaded: the cell Pip is reading its arrow from'); } },

 { h: 'How many rule cards?', p: ['How many different policies are there? Each of the 16 cells can hold any of 4 arrows, so 4 × 4 × … sixteen times: about **4.3 billion**.', 'Most of them are silly: arrows into walls, or arrows that send Pip in circles. Scoring one card fairly takes thousands of walks, so trying them all is hopeless. Finding a good one without trying everything is what the rest of reinforcement learning is about.'],
   shot: [3, 9],
   draw(t) { wet(); maze({ t }); const k = Math.floor(t / .8), pi = t < 7 ? randomPolicy(77 + k) : PA; policyArrows(G, pi, 9, { col: t < 7 ? C.soft : C.goldInk, seed: (t < 7 ? k : 0) * 3 });
     pip(cx(0), cy(3), 1, t); caption(t < 7 ? `random rule card number ${k + 1}` : 'back to policy A', 1);
     card([`4 choices in each of 16 cells`, { s: `4 × 4 × … sixteen times = ${fmtN(NPOL)}`, c: C.goldInk }], t - .8, { gap: 1.2 }); } },

 { h: 'Two candidates', p: ['Here is a second card, **policy B**. It is a copy of A with a single arrow changed: in cell 8, just left of the cat, B points down, into a rock.', 'Pip only lands in cell 8 after slipping down from the top row, so B looks just as good most of the time. Is it really worse? Pip does not know the maze’s rules, so it has to find out by walking.'],
   shot: [3, 8],
   draw(t) { twoMazes(t); const a = clamp(t - 1.5) * (.6 + .4 * Math.sin(t * 4)); alpha(a, () => rrect(MB.ox + 2 * 34 + 1, MB.oy + 34 + 1, 32, 32, 990, { c: C.rose, w: 2 }));
     MA.pip(MA.cx(0), MA.cy(3), 1, t); MB.pip(MB.cx(0), MB.cy(3), 1, t);
     if (t > 3) card(['A and B differ only in cell 8, left of the cat', { s: 'A: up, back to the top row', c: C.goldInk }, { s: 'B: down, into the rock', c: C.roseInk }], t - 3, { gap: 1 }); } },

 { h: 'Scoring a walk', p: ['To compare policies, score each walk by its **return**: the rewards added up, each discounted by **γ (gamma)** = 0.9 per step of delay.', 'Here the only reward comes at the end, so the return is 0.9 to the power of steps minus one, plus for the cheese, minus for the cat. Seven steps: 0.9 to the sixth, 0.53. A walk still going after 100 steps is stopped and scores 0, as no reward has come; with A and B it never happens.'],
   shot: [3, 10],
   draw(t) { wet(); maze({ t }); policyArrows(G, PA, 9, { a: .4 }); let c = 0, r = 3; const P = [[0, 3]];
     for (let i = 0; i < 7; i++) { [c, r] = G.step(c, r, PA[r * 5 + c]); P.push([c, r]); }
     P.slice(1).forEach(([c2, r2], i) => { const a = clamp((t - .5 - i * .5) * 3); txt(String(i + 1), cx(c2) - 18, cy(r2) - 18, { s: 15, c: '#3f7f7c', a }); });
     pip(cx(0), cy(3), 1, t);
     if (t > 4.2) card(['return = ± 0.9 × 0.9 × … (steps − 1 times)', { s: '7 steps: 0.9 × 0.9 × 0.9 × 0.9 × 0.9 × 0.9 = 0.53', c: C.goldInk }, { s: 'still walking after 100 steps: 0', c: C.soft }], t - 4.2, { gap: 1.1 }); } },

 { h: 'Walk 1: a tie', p: ['Pip walks once with each card. To be fair, both walks use **the same dice**: the same slips at the same moments. Then any difference comes from the arrows, not from luck.', 'Walk 1: twice Pip slides into the outer wall and stays put, but both walks follow the same arrows to the cheese in 9 steps. Return 0.43 each. This walk tells Pip nothing about the difference.'],
   shot: [4, 11],
   draw(t) { twoMazes(t, { arrowA: .45 }); const d = .75;
     [[MA, W1.a], [MB, W1.b]].forEach(([M, w]) => { const P = walkAt(M, w.tr, t - .8, d); trailOf(M, w.tr, P.k, .7); M.pip(P.x, P.y, P.s && P.s.nc !== P.s.c ? Math.sign(P.s.nc - P.s.c) : 0, t);
       if (P.done) txt(`return ${F2(w.ret)}`, M.ox + 85, 222, { s: 17, c: w.ret > 0 ? C.goldInk : C.roseInk, a: clamp(t - .8 - w.tr.length * d) }); });
     if (t > 8.5) card([`walk 1:  A ${F2(W1.a.ret)},  B ${F2(W1.b.ret)}`, { s: 'same dice, same route: no difference shows', c: C.soft }], t - 8.5, { gap: 1 }); } },

 { h: 'Walk 2: the trap', p: ['Walk 2. On the top row, Pip slips down into the changed cell. With A, the arrow points up: Pip climbs back and reaches the cheese in 10 steps, return 0.39.', 'With B, the arrow points into the rock. Pip bumps, then a slip throws it right into the cat: −0.48. Not just bad luck: from cell 8, B can only bump, slip into the other rock, or slip into the cat. Once there, B always ends on the cat.'],
   shot: [5, 12],
   draw(t) { twoMazes(t, { arrowA: .45 }); const d = .75;
     [[MA, W2.a], [MB, W2.b]].forEach(([M, w], j) => { const P = walkAt(M, w.tr, t - .8, d); trailOf(M, w.tr, P.k, .7); M.pip(P.x, P.y, P.s && P.s.nc !== P.s.c ? Math.sign(P.s.nc - P.s.c) : 0, t);
       if (P.s && !P.done && P.s.nc === 2 && P.s.nr === 1 && P.u > .6) txt('slip!', M.cx(2), M.cy(1) - 22, { s: 14, c: '#3f7f7c' });
       if (P.done) txt(`return ${F2(w.ret)}`, M.ox + 85, 222, { s: 17, c: w.ret > 0 ? C.goldInk : C.roseInk, a: clamp(t - .8 - w.tr.length * d) }); });
     if (t > 9.5) card([{ s: `walk 2:  A ${F2(W2.a.ret)},  B ${F2(W2.b.ret)}`, c: C.ink }, { s: 'B’s bad arrow finally mattered', c: C.roseInk }], t - 9.5, { gap: 1 }); } },

 { h: 'Five thousand walks', p: [`Up the side: the **average return** of all walks so far. Along the bottom: the number of walks. Early on the lines jump; then they flatten. After 5000: A **${F3(JA.avg)}**, B ${F3(JB.avg)}, each give or take about ${(Math.ceil(Math.max(JA.se, JB.se) * 1000) / 1000).toFixed(3)} of leftover luck, so the gap is real.`, `B ends on the cat ${fmtN(JB.cnt[1])} times, A ${fmtN(JA.cnt[1])}. Judge a policy by its average return. Measured from every cell, that average is its **value**: next reel.`],
   shot: [3, 11],
   draw(t) { wet(); maze({ t }); policyArrows(G, PA, 9, { a: .5, skip: (c, r) => c === 2 && r === 1 });
     arrow(cx(2) - 12, cy(1) + 12, cx(2) - 12, cy(1) - 14, 991, { c: C.goldInk, w: 2 }); arrow(cx(2) + 12, cy(1) - 14, cx(2) + 12, cy(1) + 12, 992, { c: C.roseInk, w: 2 });
     txt('A', cx(2) - 22, cy(1) - 18, { s: 13, c: C.goldInk }); txt('B', cx(2) + 22, cy(1) + 16, { s: 13, c: C.roseInk });
     dim(.35);
     const x0 = 64, x1 = 360, y0 = 392, y1 = 306, lo = -.05, hi = .55, X = i => x0 + i / (NW - 1) * (x1 - x0), Y = v => y0 - (clamp(v, lo, hi) - lo) / (hi - lo) * (y0 - y1);
     const n = Math.round(lerp(1, NW, ease(clamp((t - 1.6) / 6))));
     alpha(clamp(t * 2), () => { rline(x0, y0, x1, y0, 960, { c: C.line, w: 1 }); rline(x0, y0, x0, y1, 961, { c: C.line, w: 1 });
       [0, .1, .2, .3, .4, .5].forEach(v => { txt(v.toFixed(1), x0 - 8, Y(v), { s: 12, al: 'right', c: C.soft }); ctx.save(); ctx.globalAlpha = .35; ctx.strokeStyle = C.line; ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.moveTo(x0, Y(v)); ctx.lineTo(x1, Y(v)); ctx.stroke(); ctx.restore(); }); txt('average return so far', x0, y1 - 10, { s: 14, al: 'left', c: C.soft });
       txt('walk 1', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt('walk 5000', x1, y0 + 10, { s: 12, c: C.soft, al: 'right' }); });
     [[JA, '#c99a3a', 'A'], [JB, '#c0504d', 'B']].forEach(([J, col], j) => { ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.beginPath();
       for (let i = 0; i < n; i += Math.max(1, Math.floor(i / 300))) { const x = X(i), y = Y(J.run[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore();
       if (n > 1) txt(`${j ? 'B' : 'A'} ${F3(J.run[n - 1])}`, X(n - 1) + 4, Y(J.run[n - 1]) + (j ? 10 : -10), { s: 14, al: n > NW * .8 ? 'right' : 'left', c: j ? C.roseInk : C.goldInk }); });
     caption(`after ${fmtN(n)} walks each`); } },

 { h: 'Rolling dice on purpose', p: ['So far each cell held one fixed move: a **deterministic** policy. A policy can also hold chances: a **stochastic** policy. We write π(a | s), read “pi of a given s”: the probability of picking action a in state s.', 'The random Pip from the first reel was one too: a quarter for each move. Careful, there are now two kinds of chance. The wet floor is the world’s randomness. These dice are Pip’s own choice.'],
   shot: [3, 9],
   draw(t) { wet(); maze({ t }); const c0 = 1, r0 = 2, pr = [.05, .05, .05, .85];
     pip(cx(c0), cy(r0), -1, t);
     const LP = [[0, -40], [56, -14], [0, 46], [-46, -18]];
     NAMES.forEach((s, m) => { const a = clamp((t - .8 - m * .6) * 2); if (a <= 0) return; const [dx, dy] = MOVES[m], len = 12 + 36 * Math.sqrt(pr[m]), o0 = dx ? 24 : 16, main = m === 3;
       alpha(a, () => arrow(cx(c0) + dx * o0, cy(r0) + dy * o0 + 4, cx(c0) + dx * (o0 + len), cy(r0) + dy * (o0 + len) + 4, 700 + m * 9, { c: main ? C.goldInk : C.soft, w: main ? 3 : 1.4 }));
       const x = cx(c0) + LP[m][0], y = cy(r0) + LP[m][1]; box(x - 30, y - 9, 60, 18, C.bg, .85 * a); txt(`${s} ${pr[m]}`, x, y, { s: 14, c: main ? C.goldInk : '#6d645b', a }); });
     if (t > 4) card(['π(a | s): chance of each move in state s', { s: 'here: left 0.85, each other move 0.05', c: C.goldInk }, { s: 'floor slips = the world · these dice = Pip’s choice', c: C.soft }], t - 4, { gap: 1.1 }); } },

 { h: 'Why roll dice?', p: ['Two good reasons. A learner must **explore**: slips carry Pip into new cells, but if it always picks its arrow, it never finds out what the other moves would do. And against an opponent, being predictable gets punished: in rock, paper, scissors, play each one third of the time.', 'A fixed maze has no opponent. In each cell, some move is at least as good as the others, so putting all the chance on it cannot hurt: the best policy needs no dice.'],
   shot: [3, 9],
   draw(t) { const a1 = clamp(t - .5), a2 = clamp(t - 2.5);
     alpha(a1, () => { MA.maze({ t }); const R = rng(9); for (let i = 0; i < 6; i++) { const [c, r] = OPEN[Math.floor(R() * OPEN.length)]; txt('?', MA.cx(c), MA.cy(r) + 2, { s: 18, c: '#3f7f7c' }); } MA.pip(MA.cx(0), MA.cy(3), 1, t); txt('explore: try other moves', MA.ox + 85, 222, { s: 15, c: '#3f7f7c' }); });
     alpha(a2, () => { const xs = [250, 300, 350]; ['rock', 'paper', 'scissors'].forEach((s, i) => { rcircle(xs[i], 130, 20, 970 + i, { fill: C.card }); txt(['✊', '✋', '✌'][i], xs[i], 131, { s: 20 }); txt(s, xs[i], 165, { s: 14, c: C.soft }); txt('1/3', xs[i], 92, { s: 15, c: C.goldInk }); });
       txt('games: stay unpredictable', 300, 222, { s: 15, c: C.goldInk }); });
     if (t > 5) card(['explore while learning', 'stay unpredictable against an opponent', { s: 'fixed maze: the best policy needs no dice', c: C.soft }], t - 5, { gap: 1.1 }); } },

 { h: 'Dial in randomness', ctl: { label: 'ε', values: EPS, def: 2, fmt: v => String(v), onChange: () => {} },
   p: ['Drag **ε (epsilon)**: on each move, with chance ε, Pip ignores its arrow and picks one of the four moves at random. Otherwise it follows policy A.', `At 0, plain policy A: ${F3(JA.avg)}. At 0.2, the 0.85 and 0.05 card from two steps back: ${F3(judge(PA, .2).avg)}. At 1, Pip ignores its card: ${F3(judge(PA, 1).avg)}. That is close to 0, not clearly negative, because discounting shrinks the late endings of long random walks.`],
   shot: [3, 8],
   draw(t, st) { const e = EPS[st.ci ?? 2], J = judge(PA, e), J0 = st.prevCi != null ? judge(PA, EPS[st.prevCi]) : J, u = ease((t - (st.changed ?? -9)) / .8);
     wet(); maze({ t }); policyArrows(G, PA, 9, { a: lerp(.75, .15, e) });
     const w = walk(PA, 1000 + Math.floor(t / 12), e), tr = w.tr.slice(0, 40), P = walkAt(G, tr, (t % 12) - .5, .45); trailOf(G, tr, P.k, .6);
     pip(P.x, P.y, P.s && P.s.nc !== P.s.c ? Math.sign(P.s.nc - P.s.c) : 0, t);
     if (P.s && P.s.rnd && !P.done) txt('dice!', P.x, P.y - 26, { s: 14, c: C.orange });
     const avg = lerp(J0.avg, J.avg, u), cnt = J.cnt.map((v, i) => lerp(J0.cnt[i], v, u));
     txt(`average return over 5000 walks:  ${F3(avg)}`, 40, 300, { s: 17, al: 'left' });
     const xs = [100, 200, 300], base = 374, hm = 46, lab = ['cheese', 'cat', 'stopped at 100'], col = [C.gold, C.rose, C.grey]; rline(50, base, 350, base, 906, { c: C.line, w: 1 });
     xs.forEach((x, i) => { const h = cnt[i] / NW * hm; box(x - 26, base - h, 52, h, col[i]); txt(lab[i], x, base + 11, { s: 14, c: C.soft }); txt(fmtN(Math.round(cnt[i])), x, base - h - 9, { s: 14 }); });
     caption('bars: how the 5000 walks ended  ·  orange “dice!” = a random move'); } },

 { h: 'Your turn: write a card', tap: tapTurn, p: ['Tap any cell to turn its arrow a quarter turn clockwise. After each change, Pip walks your card 5000 times from the start, with the same dice as before.', `Policy A scores ${F3(JA.avg)}. Can you beat it? Barely. Worked out exactly from the rulebook, the best card is worth ${EXACT.best.toFixed(4)} from the start, and A ${EXACT.A.toFixed(4)}. The best card differs from A in one cell; on these walks it scores ${F3(JOPT.avg)}. Or break A, and see which cells matter.`],
   shot: [3, 8],
   draw(t, st) { const pi = st.pi || PA, J = judge(pi); wet(); maze({ t });
     policyArrows(G, pi, 9, { hi: (c, r) => pi[r * 5 + c] !== PA[r * 5 + c] }); pip(cx(0), cy(3), 1, t);
     if (st.tc != null) { const [c, r] = st.tc, a = 1 - clamp((t - st.tt) / .8); if (a > 0) box(G.ox + c * CS + 3, G.oy + r * CS + 3, CS - 6, CS - 6, C.gold, .35 * a); }
     const d = J.avg - JA.avg;
     card([{ s: `your card: average return ${F3(J.avg)}`, c: C.ink }, { s: Math.abs(d) < .0005 ? 'same as policy A' : `${d > 0 ? 'better' : 'worse'} than A by ${F3(Math.abs(d))}`, c: d > .0005 ? '#3f7f7c' : d < -.0005 ? C.roseInk : C.soft },
       { s: `cheese ${fmtN(J.cnt[0])} · cat ${fmtN(J.cnt[1])} · stopped ${fmtN(J.cnt[2])}`, c: C.soft }], 9, { gap: 0 });
     caption(st.pi ? 'red arrows: changed from policy A' : 'tap a cell to turn its arrow'); } },

 { h: 'Where policies live', p: ['A thermostat runs a tiny deterministic policy: too cold, heat on; warm enough, heat off. A robot’s controller maps sensor readings to motor commands.', 'AlphaGo, which beat Lee Sedol at Go in 2016, used a **policy network**: given a board, it output a probability for every legal move, a stochastic policy learned first from millions of human moves, then improved by playing against itself.'],
   shot: [3, 9],
   draw(t) { const a1 = clamp(t - .5), a2 = clamp(t - 2), a3 = clamp(t - 3.5);
     alpha(a1, () => { rrect(52, 92, 56, 76, 940, { fill: C.card }); txt('19°', 80, 118, { s: 20 }); txt('heat on', 80, 146, { s: 14, c: C.orange }); txt('thermostat', 80, 192, { s: 16 }); });
     alpha(a2, () => { L.robot(200, 135, 2, t); txt('robot controller', 200, 192, { s: 16 }); });
     alpha(a3, () => { for (let i = 0; i <= 4; i++) { rline(290 + i * 18, 92, 290 + i * 18, 164, 950 + i, { c: C.line, w: 1, once: true }); rline(290, 92 + i * 18, 362, 92 + i * 18, 956 + i, { c: C.line, w: 1, once: true }); }
       dot(308, 110, 6, C.ink); dot(326, 128, 6, '#f5efe6'); rcircle(326, 128, 6, 962, { j: .3 }); [[344, 110, .6], [326, 146, .25], [290, 146, .1]].forEach(([x, y, q]) => { dot(x, y, 7 * Math.sqrt(q) + 2, C.gold, .8); });
       txt('Go: chance per move', 326, 192, { s: 16 }); });
     if (t > 5) card(['deterministic: same situation, same action', 'stochastic: a probability for each action', { s: 'either way: situation in, action out', c: C.soft }], t - 5, { gap: 1 }); } },

 { h: 'Recap', p: ['A **policy** π is a rule from state to action: an arrow in every cell, so it keeps working after any slip. It can be deterministic, or stochastic with π(a | s).', 'To compare policies, run each many times and average the returns. One walk is luck. That average, from every cell, is next: the **value** of a policy.'],
   shot: [4, 9],
   draw(t) { wet(); maze({ t }); policyArrows(G, PA, 9, { a: .8 }); pip(cx(0), cy(3), 1, t);
     card(['policy π: state in, action out', 'deterministic π(s) or stochastic π(a | s)', 'judge a policy by its average return', { s: `policy A ${F3(JA.avg)} vs B ${F3(JB.avg)} over 5000 walks`, c: C.soft }], t - .3, { gap: 1 }); } },
];

function tapTurn(x, y, st, t) {
  const c = Math.floor((x - G.ox) / CS), r = Math.floor((y - G.oy) / CS);
  if (c < 0 || c > 4 || r < 0 || r > 3 || G.isRock(c, r) || G.term(c, r)) return;
  const pi = (st.pi || PA).slice(); pi[r * 5 + c] = (pi[r * 5 + c] + 1) % 4; st.pi = pi; st.tc = [c, r]; st.tt = t;
}

RLStory.register('policy', { title: 'Policies', track: 'Foundations', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { listCheese: LISTRUNS.ch, run2: LISTRUNS.run2.map(s => s.nc + ',' + s.nr), w1: [W1.a.ret, W1.a.tr.length, W1.b.ret, W1.b.tr.length], w2: [W2.a.ret, W2.a.tr.length, W2.b.ret, W2.b.tr.length],
    w1path: W1.a.tr.map(s => s.nc + '' + s.nr + (s.mm !== s.m ? '*' : '')).join(' '), w2a: W2.a.tr.map(s => s.nc + '' + s.nr + (s.mm !== s.m ? '*' : '')).join(' '), w2b: W2.b.tr.map(s => s.nc + '' + s.nr + (s.mm !== s.m ? '*' : '')).join(' '),
    recover: [RECOVER.ret, RECOVER.tr.length, RECOVER.tr.map(s => s.nc + '' + s.nr + (s.mm !== s.m ? '*' : '')).join(' ')], A: [JA.avg, JA.cnt], B: [JB.avg, JB.cnt],
    eps: () => EPS.map(e => [e, judge(PA, e).avg, judge(PA, e).cnt]), npol: NPOL, se: [JA.se, JB.se, SED], exact: EXACT, opt: JOPT.avg, optCnt: JOPT.cnt, runMin: () => [Math.min(...JA.run), Math.min(...JB.run), Math.max(...JA.run), Math.max(...JB.run)] } });
})();
