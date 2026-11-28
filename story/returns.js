// Rewards and returns as a story: Pip adds up its rewards, discovers that the total ignores how long a walk took,
// and tries a step cost. Uses the same 100 random episodes as the loop reel (seed 769). All numbers come from below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, maze, trail, pip, ox: OX, oy: OY, cs: CS } = G;
const no = (c, r) => r * G.cols + c + 1;
const NAMES = ['up', 'right', 'down', 'left'];

// ───── simulation: the loop reel's random walks (uniform over 4 actions), seed 769 ─────
const SEED = 769;
function walks(seed, n) {
  const R = L.rng(seed), out = [];
  for (let e = 0; e < n; e++) { let [c, r] = G.start; const tr = [];
    for (let k = 0; k < 5000; k++) { const m = Math.floor(R() * 4), [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr); tr.push({ c, r, m, nc, nr, rew, bump: nc === c && nr === r }); c = nc; r = nr; if (rew) break; }
    out.push(tr); }
  return out;
}
const EPS = walks(SEED, 100), LENS = EPS.map(e => e.length), ENDS = EPS.map(e => e[e.length - 1].rew);
// reward of one step with a step cost: landing on cheese or cat pays +1 or −1, any other step pays −cost
const rewC = (q, cost) => q.rew ? q.rew : -cost;
const ret = (tr, cost, from = 0) => { let g = 0; for (let k = tr.length - 1; k >= from; k--) g = rewC(tr[k], cost) + g; return g; };
const r1 = v => { const x = Math.round(v * 100) / 100; return Object.is(x, -0) ? 0 : x; };
const fmt = (v, d = 1) => { const x = r1(v); const s = (Math.abs(x) < 1e-9 ? 0 : Math.abs(x)).toFixed(d).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1'); return x < -1e-9 ? '−' + s : s; };
const fmtS = (v, d) => (r1(v) > 1e-9 ? '+' : '') + fmt(v, d);
// fixed routes (cells as [c, r]); the shortest way to the cheese and to the cat
const SHORT = [[0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [4, 2], [4, 1], [4, 0]];
const QCAT = [[0, 3], [1, 3], [2, 3], [3, 3], [3, 2], [3, 1]];
const toTr = P => P.slice(0, -1).map(([c, r], i) => { const [nc, nr] = P[i + 1]; return { c, r, nc, nr, m: G.MOVES.findIndex(([a, b]) => a === nc - c && b === nr - r), rew: G.term(nc, nr), bump: false }; });
const TS = toTr(SHORT), TQ = toTr(QCAT), E1 = EPS[0];
// which episodes to show as "same return, different length"
const SAME = [9, 0, 4].map(i => ({ i, n: LENS[i] }));      // episodes 10, 1, 5: all reached the cheese
const COST = .1, RET01 = EPS.map(e => ret(e, COST)), MEAN01 = RET01.reduce((a, b) => a + b, 0) / 100;
const BESTR = Math.max(...RET01), WORSTR = Math.min(...RET01), OFF = LENS.filter(l => l > 150).length;
const COSTS = [0, .01, .05, .1, .2, .5, 1, 1.5];

// ───── helpers ─────
const F = s => `${s}px 'Patrick Hand', cursive`;
function width(s, size) { ctx.save(); ctx.font = F(size); const w = ctx.measureText(s).width; ctx.restore(); return w; }
function rich(parts, x, y, o = {}) {
  const s = o.s || 17, ss = Math.round(s * .66);
  const ws = parts.map(p => typeof p === 'string' ? width(p, s) : width(p[0], s) + width(p[1], ss) + 1), tot = ws.reduce((a, b) => a + b, 0);
  let X = o.al === 'center' ? x - tot / 2 : o.al === 'right' ? x - tot : x;
  parts.forEach((p, i) => { if (typeof p === 'string') txt(p, X, y, { s, c: o.c, a: o.a, al: 'left' }); else { const wb = width(p[0], s); txt(p[0], X, y, { s, c: o.c, a: o.a, al: 'left' }); txt(p[1], X + wb + 1, y + s * .3, { s: ss, c: o.c, a: o.a, al: 'left' }); } X += ws[i]; });
  return tot;
}
function rcard(rows, t, o = {}) {
  const x = o.x ?? 30, y = o.y ?? 292, w = o.w ?? 340, lh = o.lh ?? 23, h = 18 + rows.length * lh, a = clamp(t * 3);
  if (a <= 0) return; box(x, y, w, h, C.card, a); rrect(x, y, w, h, 200 + rows.length, { c: C.line, w: 1.1 });
  rows.forEach((row, i) => { const u = clamp((t - (o.gap ?? 1.1) * i) * 2.5); if (u <= 0) return; const parts = typeof row === 'string' ? [row] : Array.isArray(row) ? row : row.parts; rich(parts, x + 14, y + 20 + i * lh, { s: o.fs || 17, a: u, c: row.c }); });
}
function nums(a = 1) { for (let r = 0; r < G.rows; r++) for (let c = 0; c < G.cols; c++) { if (G.isRock(c, r)) continue; txt(String(no(c, r)), OX + c * CS + 8, OY + r * CS + 11, { s: 12, al: 'left', c: C.soft, a }); } }
function dim(a = .7) { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, 286); ctx.restore(); }
function walkAt(tr, from, to, t, dur) {
  const k = Math.floor(t / dur);
  if (k >= to - from) { const l = tr[to - 1]; return { x: cx(l.nc), y: cy(l.nr), k: to - from, done: true, face: Math.sign(l.nc - l.c) || 0 }; }
  const s = tr[from + k], u = ease((t % dur) / dur * 1.4), [dx, dy] = G.MOVES[s.m];
  if (s.bump) { const b = Math.sin(clamp((t % dur) / dur * 1.6) * Math.PI) * 11; return { x: cx(s.c) + dx * b, y: cy(s.r) + dy * b, k, s, face: dx || 0 }; }
  return { x: lerp(cx(s.c), cx(s.nc), u), y: lerp(cy(s.r), cy(s.nr), u), k, s, face: dx || 0 };
}
// a route drawn as a hand-drawn polyline through cell centres
function route(P, col, off = 0, p = 1, seed = 500) { const n = (P.length - 1) * clamp(p); for (let i = 0; i < Math.ceil(n); i++) { const u = clamp(n - i), [a, b] = [P[i], P[i + 1]]; rline(cx(a[0]) + off, cy(a[1]) + off, lerp(cx(a[0]), cx(b[0]), u) + off, lerp(cy(a[1]), cy(b[1]), u) + off, seed + i, { c: col, w: 3, once: true, j: .6 }); } }
// a row of reward chips: vals = numbers, shown n of them
function chips(vals, x0, y, n, o = {}) {
  const gap = o.gap ?? 24;
  for (let i = 0; i < Math.min(n, vals.length); i++) { const v = vals[i], x = x0 + i * gap; rcircle(x, y, 10, 600 + i, { fill: v > 0 ? C.yellow : v < 0 ? C.pink : C.card, c: v > 0 ? C.goldInk : v < 0 ? C.roseInk : C.line, j: .5 });
    txt(v === 0 ? '0' : fmtS(v), x, y + 1, { s: o.fs || (String(v).length > 3 ? 11 : 13), c: v > 0 ? C.goldInk : v < 0 ? C.roseInk : C.soft }); }
}
// tiny boat for the CoastRunners story
function boat(x, y, a) { ctx.save(); ctx.translate(x, y); ctx.rotate(a); ctx.beginPath(); ctx.moveTo(-12, -5); ctx.lineTo(10, -5); ctx.lineTo(15, 0); ctx.lineTo(10, 5); ctx.lineTo(-12, 5); ctx.closePath(); ctx.fillStyle = '#e9e1d6'; ctx.fill(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3; ctx.stroke(); ctx.fillStyle = C.rose; ctx.fillRect(-4, -3, 7, 6); ctx.restore(); }

// return-vs-length chart (step cost 0.1): x = steps in the episode (0..150), y = return (+1 .. −16)
const RC = { x0: 60, x1: 370, X: n => 60 + Math.min(n, 150) / 150 * 310, Y: g => 50 + (1 - g) / 17 * 220 };
function retChart(t) {
  const { x0, x1, X, Y } = RC;
  rline(x0, Y(0), x1, Y(0), 930, { c: C.line, w: 1 }); rline(x0, Y(1.5), x0, Y(-16), 931, { c: C.line, w: 1 });
  [0, -5, -10, -15].forEach(g => txt(fmt(g), x0 - 6, Y(g), { s: 12, c: C.soft, al: 'right' })); txt('return', x0 - 4, Y(1.5) - 12, { s: 14, c: C.soft, al: 'left' });
  txt('steps in the episode →', x1, Y(0) - 12, { s: 14, c: C.soft, al: 'right', a: clamp(t - .5) });
  [50, 100, 150].forEach(n => { txt(String(n), X(n), Y(0) + 11, { s: 12, c: C.soft, a: clamp(t - .5) }); rline(X(n), Y(0) - 3, X(n), Y(0) + 3, 934 + n, { c: C.line, w: 1, once: true }); });
  const a1 = clamp((t - 1.5) / 1.5), a2 = clamp((t - 3.5) / 1.5);
  if (a1 > 0) rline(X(1), Y(1), lerp(X(1), X(150), a1), lerp(Y(1), Y(1 - .1 * 149), a1), 932, { c: C.teal, w: 1.6 });
  if (a2 > 0) rline(X(1), Y(-1), lerp(X(1), X(140), a2), lerp(Y(-1), Y(-1 - .1 * 139), a2), 933, { c: C.rose, w: 1.6 });
  if (t > 5.5) EPS.forEach((e, i) => { const a = clamp((t - 5.5 - i * .02) * 3); if (a > 0 && LENS[i] <= 150) dot(X(LENS[i]), Y(RET01[i]), 3.2, ENDS[i] > 0 ? C.teal : C.rose, a); });
}

// ───── the steps ─────
const STEPS = [
 { h: 'One number per step', p: ['Back in Pip’s maze, with the same rules: cheese +1, cat −1, every other step 0. In the last reel Pip walked at random, and its first episode took 14 steps.', 'Here are its rewards, one per step: thirteen zeros, then +1. How good was that episode as a whole?'],
   shot: [4, 10],
   say: [[["Back in Pip's maze. Watch Pip replay its first random walk.", [40, 16, 320, 256]]], [["Each step drops one reward into the row below.", [28, 298, 350, 50]], ["Zero, zero, zero, and only at the very end, plus one.", [28, 298, 350, 50]]], "So how good was that walk, as a whole?"],
   draw(t) { const m = walkAt(E1, 0, E1.length, t, .6); trail(E1.slice(0, m.k + 1).map(q => [q.c, q.r]), .9); maze({ t }); pip(m.x, m.y, m.face, t);
     chips(E1.map(q => q.rew), 42, 312, m.done ? E1.length : m.k, { gap: 24.5 });
     txt('reward of each step, in order', 30, 338, { s: 14, al: 'left', c: C.soft, a: clamp(t - 1) });
     if (m.done) caption('episode 1: 14 steps, then the cheese', clamp(t - 9.5)); } },

 { h: 'Add them up: the return', p: ['Add up all the rewards from a moment until the episode ends. That sum is called the **return**. For episode 1, counted from the start: thirteen zeros plus one, so the return is 1.', 'The return is the score for a whole stretch of the walk, not for one step.'],
   shot: [3, 9],
   say: [[["Watch the rewards fly into the jar.", [20, 290, 370, 110]], ["Thirteen zeros add nothing. Then the cheese adds one.", [20, 290, 370, 110]]], ["That total is the return. It scores a whole stretch of the walk, not one single step."]],
   draw(t) { maze({ t }); dim(.55); trail(E1.map(q => [q.c, q.r]), .5); pip(cx(4), cy(0), 1, t);
     const vals = E1.map(q => q.rew), fly = i => clamp((t - 1 - i * .3) / .6), jar = { x: 340, y: 332 };
     vals.forEach((v, i) => { const u = ease(fly(i)), x = lerp(42 + i * 24.5, jar.x, u), y = lerp(312, jar.y, u) - Math.sin(u * Math.PI) * 30; if (u >= 1) return; rcircle(x, y, 10, 600 + i, { fill: v > 0 ? C.yellow : C.card, c: v > 0 ? C.goldInk : C.line, j: .5 }); txt(String(v ? '+1' : 0), x, y + 1, { s: 13, c: v > 0 ? C.goldInk : C.soft }); });
     rrect(jar.x - 26, jar.y - 22, 52, 50, 640, { fill: C.card, c: C.ink, w: 1.4 });
     const got = vals.filter((_, i) => fly(i) >= 1).reduce((a, b) => a + b, 0); txt(String(got), jar.x, jar.y + 3, { s: 26, c: got ? C.goldInk : C.soft });
     txt('return', jar.x, jar.y + 38, { s: 14, c: C.soft });
     if (t > 6.5) txt('0 + 0 + … + 0 + 1  =  1', 40, 340, { s: 18, al: 'left', a: clamp(t - 6.5) });
     if (t > 7.5) txt('thirteen zeros, then the cheese', 40, 368, { s: 14, al: 'left', c: C.soft, a: clamp(t - 7.5) }); } },

 { h: 'Writing it down', p: ['The return from time t, **G t**, adds every reward after t: R t+1 plus R t+2 and so on, up to R T. **T**, the end time, is the episode’s number of moves.', 'It starts at t+1 because action A t’s reward arrives one tick later. After the end nothing is left: G T = 0. Episode 1: T = 14, G0 = 1.'],
   shot: [5, 11],
   say: [[["Here's the return written down. People use the letter G.", [20, 50, 360, 110]], ["It adds every reward after time t, all the way to the end.", [20, 50, 360, 110]]], [["Notice it starts one tick later, because each reward arrives just after its move.", [20, 50, 360, 110]], ["For episode one, the return from the start is one.", [20, 180, 360, 30]]]],
   draw(t) {
     rich([['G', 't'], '  =  ', ['R', 't+1'], ' + ', ['R', 't+2'], ' + … + ', ['R', 'T']], W / 2, 90, { s: 30, al: 'center', a: clamp(t - .4) });
     const lab = (s, x, y, a, c) => txt(s, x, y, { s: 15, c: c || C.soft, a });
     lab('return from time t', 94, 136, clamp(t - 1.6)); lab('first reward after time t', 200, 50, clamp(t - 2.6), C.roseInk); lab('last one: at the end, time T', 300, 136, clamp(t - 3.6));
     alpha(clamp(t - 1.6), () => arrow(94, 126, 94, 108, 650, { c: C.soft })); alpha(clamp(t - 2.6), () => arrow(186, 60, 172, 74, 651, { c: C.roseInk })); alpha(clamp(t - 3.6), () => arrow(320, 126, 316, 108, 652, { c: C.soft }));
     if (t > 5) { rich(['episode 1:  T = 14,   ', ['G', '0'], ' = 0 + 0 + … + 0 + 1 = 1'], W / 2, 196, { s: 17, al: 'center', a: clamp(t - 5) }); }
     if (t > 6.5) txt('G is just the usual letter for a return', W / 2, 236, { s: 14, c: C.soft, a: clamp(t - 6.5) });
     rcard([[['R', 't+1'], ' = reward for the action at time t'], ['T = the time the episode ends (its number of moves)'], { parts: [['G', 'T'], ' = 0: nothing is left after the end'], c: C.soft }], t - 7.5, { gap: 1.1 }); } },

 { h: 'Why the total counts', p: ['Why not judge a move by its own reward? From cell 16, going up, going right and bumping the wall all pay 0. The next reward cannot tell them apart.', 'What differs comes later, and the return covers the whole rest of the episode. One walk’s return credits all its moves, good or bad alike; averaging many walks sorts them out, as later reels show.'],
   shot: [4, 10],
   say: [[["Why not judge a move by its own reward?", [40, 16, 320, 256]], ["Look at the four moves from the start.", [40, 16, 320, 256]], ["Every one pays zero next.", [40, 16, 320, 256]]], [["The next reward can't tell them apart.", [30, 302, 340, 87]], ["What differs comes later, and that's what the return counts.", [30, 302, 340, 87]]]],
   draw(t) { maze({ t }); pip(cx(0), cy(3), 1, t);
     [[0, 'up'], [1, 'right'], [2, 'down'], [3, 'left']].forEach(([m, n], i) => { const a = clamp((t - .6 - i * .8) * 2); if (a <= 0) return; const [dx, dy] = G.MOVES[m];
       alpha(a, () => arrow(cx(0) + dx * 20, cy(3) + dy * 20, cx(0) + dx * 40, cy(3) + dy * 40, 660 + m, { c: C.goldInk, w: 1.8 }));
       const L4 = { up: [cx(0), cy(3) - 52, 'center', '0'], right: [cx(0) + 46, cy(3) + 4, 'left', '0'], down: [cx(0) + 10, cy(3) + 44, 'left', '0'], left: [OX - 4, cy(3) - 10, 'right', '0'] }[n];
       txt(L4[3], L4[0], L4[1], { s: 15, c: C.goldInk, a, al: L4[2] }); });
     card(['every move from cell 16 pays 0 next', '(down and left bump: still 0)', { s: 'the return looks at everything that follows', c: C.goldInk }], t - 4, { gap: 1.3, y: 302 }); } },

 { h: 'Same return, very different walks', p: ['Now compare three random episodes that all ended at the cheese: episode 10 took 10 steps, episode 1 took 14, and episode 5 took 147. Each one’s return is exactly 1.', 'With these rewards, the return ignores how long Pip wandered. A detour of a hundred steps costs nothing, so Pip has no reason to hurry.'],
   shot: [4, 11],
   say: [[["Three random walks, all ending at the cheese.", [40, 16, 320, 256]], ["Watch how differently they wander.", [40, 16, 320, 256]]], [["Ten steps, fourteen steps, and a hundred and forty seven.", [30, 296, 340, 90]], ["Yet every return is exactly one.", [30, 296, 340, 90]]], "With these rewards, a long detour costs nothing. Pip has no reason to hurry."],
   draw(t) { maze({ t }); const which = clamp(Math.floor(t / 2.6), 0, 2), e = EPS[SAME[which].i], u = clamp((t - which * 2.6) / 2.2);
     trail(e.slice(0, Math.max(1, Math.floor(clamp(u) * e.length))).map(q => [q.c, q.r]), .8); const q = e[Math.min(e.length - 1, Math.floor(clamp(u) * e.length))]; pip(cx(u >= 1 ? q.nc : q.c), cy(u >= 1 ? q.nr : q.r), 1, t);
     SAME.forEach((s, j) => { const a = clamp((t - 2.2 - j * 2.6) * 2), y = 312 + j * 28; if (t < j * 2.6) return;
       txt(`episode ${s.i + 1}`, 40, y, { s: 16, al: 'left', a: clamp((t - j * 2.6) * 2) }); txt(`${s.n} steps`, 140, y, { s: 16, al: 'left', c: C.soft, a: clamp((t - j * 2.6) * 2) });
       txt('return 1', 250, y, { s: 17, al: 'left', c: C.goldInk, a }); });
     if (t > 8.5) caption('the total only sees the +1 at the end', clamp(t - 8.5)); } },

 { h: 'A step cost', p: ['A common fix is a **step cost**: the designer changes the rules so every step that does not land on the cheese or cat pays −0.1, bumps included. Cheese +1 and cat −1 stay, and both still end the walk.', 'Every number from here on uses this new rule. Why 0.1? Just a small choice; we will see that the size matters.'],
   shot: [4, 10],
   say: [[["A common fix is a step cost.", [40, 16, 320, 256]], ["Watch the cells. Every ordinary step now costs a tenth.", [40, 16, 320, 256]], ["Bumps too.", [40, 16, 320, 256]]], [["The cheese and the cat stay the same, and both still end the walk.", [30, 292, 340, 87]]], "Why a tenth? Just a small choice. Its size will matter."],
   draw(t) { maze({ t }); pip(cx(0), cy(3), 1, t);
     for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.isRock(c, r)) continue; const v = G.term(c, r), a = clamp((t - .8 - (v ? 1.6 : (r * 5 + c) * .04)) * 2);
       txt(v > 0 ? '+1' : v < 0 ? '−1' : '−0.1', cx(c), cy(r) + (v ? -22 : 24), { s: v ? 20 : 14, c: v > 0 ? C.goldInk : v < 0 ? C.roseInk : C.roseInk, a: v ? a : a * .8 }); }
     card(['number in a cell = reward for stepping onto it', 'every ordinary step, bumps included: −0.1', { s: 'cheese +1 and cat −1 as before, both end the walk', c: C.soft }], t - 3.2, { gap: 1.2 }); } },

 { h: 'Shorter walks win now', p: ['Recount with the step cost. A cheese walk of n steps has n − 1 ordinary steps, so its return is 1 − 0.1 × (n − 1). A cat walk scores −1 − 0.1 × (n − 1).', `The shortest walk, 7 steps, scores 0.4. Episode 10 scores 0.1, episode 1 −0.3, and episode 5 ${fmt(ret(EPS[4], COST))}. The total now prefers short walks.`],
   shot: [3, 11],
   say: [[["Recount with the step cost.", [40, 16, 320, 256]], ["The gold path is the shortest way to the cheese, seven steps.", [40, 16, 320, 256]]], [["It now scores zero point four.", [30, 292, 340, 100]], ["The long hundred and forty seven step walk sinks to minus thirteen point six.", [30, 292, 340, 100]]], "Now the total prefers short walks."],
   draw(t) { maze({ t }); route(SHORT, C.gold, 0, clamp((t - .3) / 2), 520); const m = walkAt(TS, 0, TS.length, clamp(t - .3, 0, 9) * 3.2, 1); pip(m.x, m.y, m.face || 1, t);
     const rows = [['shortest walk', 7, ret(TS, COST)], ...SAME.map(s => [`episode ${s.i + 1}`, s.n, ret(EPS[s.i], COST)])];
     rows.forEach(([n, l, g], j) => { const a = clamp((t - 2.6 - j * 1.4) * 2), y = 308 + j * 24; if (a <= 0) return;
       txt(n, 40, y, { s: 16, al: 'left', a, c: j ? C.ink : C.goldInk }); txt(`${l} steps`, 150, y, { s: 15, al: 'left', c: C.soft, a }); txt(`1 − 0.1 × ${l - 1} = ${fmt(g)}`, 228, y, { s: 15, al: 'left', a, c: g >= 0 ? C.goldInk : C.roseInk }); }); } },

 { h: 'A picture of all 100', p: [`All 100 random episodes. Across: steps. Height: return, with the step cost. Most returns are negative, so zero sits near the top; ${OFF} long episodes fall off the right edge.`, 'Teal ended at the cheese, rose at the cat. Each dot sits on one of two lines, the cat line 2 below. Every extra step costs 0.1, so on each line shorter is better.'],
   shot: [4, 12],
   say: [[["Here are all hundred random walks.", [30, 20, 350, 270]], ["Further right means longer. Higher means a better return.", [30, 20, 350, 270]]], [["Teal dots ended at the cheese, rose dots at the cat.", [30, 20, 350, 270]], ["They sit on two sloping lines. On each line, shorter is better.", [30, 20, 350, 270]]]],
   draw(t) { retChart(t);
     if (t > 1.5) txt('ended at the cheese', RC.X(70) + 6, RC.Y(1 - .1 * 69) - 14, { s: 14, c: '#2f7f7b', a: clamp(t - 2.5), al: 'left' });
     if (t > 3.5) txt('ended at the cat', RC.X(40) - 4, RC.Y(-1 - .1 * 39) + 16, { s: 14, c: C.roseInk, a: clamp(t - 4.5), al: 'right' });
     card(['one dot = one episode', 'each extra step: return 0.1 lower', { s: `${OFF} episodes over 150 steps are off the chart`, c: C.soft }], t - 6.5, { gap: 1.1 }); } },

 { h: 'A return from every step', p: ['Returns can start at any time. On the shortest walk, from cell 10, the last before the cheese, only the +1 is left: G6 = 1. One step earlier: −0.1 + 1 = 0.9.', 'Each return is the next reward plus the return after it. Backward: 1, 0.9, 0.8, down to 0.4 at the start. Later reels average such returns into a **value**.'],
   shot: [5, 12],
   say: [[["Returns can start from any step. Look at the shortest walk.", [40, 16, 320, 256]], ["Next to the cheese, only the plus one is left. Its return is one.", [40, 16, 320, 256]]], [["One step back, pay the step cost first. Zero point nine.", [40, 16, 320, 256]]], [["Each return is the next reward, plus the return after it.", [30, 292, 340, 87]], "All the way back to zero point four at the start."]],
   draw(t) { maze({ t }); route(SHORT, C.gold, 0, 1, 520); pip(cx(0), cy(3), 1, t);
     const n = TS.length; for (let k = n - 1; k >= 0; k--) { const a = clamp((t - .8 - (n - 1 - k) * .9) * 2); if (a <= 0) continue; const q = TS[k], g = ret(TS, COST, k);
       box(cx(q.c) - 18, cy(q.r) - 30, 36, 18, C.card, .9 * a); rich([['G', String(k)], ' ' + fmt(g)], cx(q.c), cy(q.r) - 21, { s: 15, al: 'center', a, c: C.goldInk }); }
     rcard([[['G', '6'], ' = ', ['R', '7'], ' = 1     (the cheese step)'], [['G', '5'], ' = ', ['R', '6'], ' + ', ['G', '6'], ' = −0.1 + 1 = 0.9'], [['G', 't'], ' = ', ['R', 't+1'], ' + ', ['G', 't+1'], '     next reward + the rest']], t - 7, { gap: 1.3 }); } },

 { h: 'The return is a gamble', p: [`From the same start, random Pip’s returns vary wildly: the best of 100 scored ${fmt(BESTR)}, the worst ${fmt(WORSTR)}.`, `The return depends on every random move, so it is a random number; here it averages ${fmt(MEAN01)}. Averaged over many episodes from a cell, under one way of choosing moves, it is what a later reel calls a value. Choose moves differently and that changes.`],
   shot: [4, 11],
   say: [[["From the very same start, random Pip's returns are all over the place.", [30, 286, 340, 110]], ["The best scored zero point three. The worst, about minus thirty.", [30, 286, 340, 110]]], [["So the return is a random number. Here it averages minus six point seven.", [30, 286, 340, 110]]], "Average it over many walks from a cell, and you get what a later reel calls a value."],
   draw(t) { maze({ t }); dim(.5); pip(cx(0), cy(3), 1, t);
     const x0 = 40, x1 = 360, lo = -30, hi = 1, X = g => x0 + (Math.max(g, lo) - lo) / (hi - lo) * (x1 - x0), y0 = 380;
     rline(x0, y0, x1, y0, 940, { c: C.line, w: 1 }); [-30, -20, -10, 0].forEach(g => { txt(g ? fmt(g) : '0', X(g), y0 + 11, { s: 12, c: C.soft }); rline(X(g), y0 - 3, X(g), y0 + 3, 941 + g, { c: C.line, w: 1, once: true }); });
     txt('return of each episode (step cost 0.1)', x0, 296, { s: 14, al: 'left', c: C.soft }); 
     const stack = {}; RET01.forEach((g, i) => { const b = Math.round(X(g) / 6), k = stack[b] = (stack[b] || 0) + 1, a = clamp((t - .5 - i * .04) * 3); if (a > 0) dot(b * 6, y0 - 4 - (k - 1) * 6, 2.6, ENDS[i] > 0 ? C.teal : C.rose, a); });
     if (t > 5) { const xm = X(MEAN01); alpha(clamp(t - 5), () => rline(xm, 306, xm, y0, 945, { c: C.ink, w: 1.4 })); txt(`average ${fmt(MEAN01)}`, xm + 4, 312, { s: 14, al: 'left', a: clamp(t - 5) }); } } },

 { h: 'Careful: the cost changes the goal', p: ['Size matters. With a cost of 1.5 per step, the shortest cheese walk scores 1 − 1.5 × 6 = −8, but running into the cat, 5 steps, scores −1 − 1.5 × 4 = −7.', 'Now rushing to the cat has the highest return, so an agent maximising it would do that. It chases the return you wrote down, not the one you meant.'],
   shot: [4, 11],
   say: [[["But the size of the cost matters.", [40, 16, 320, 256]], ["Make it one and a half per step.", [40, 16, 320, 256]], ["Compare the gold walk to the cheese with the rose walk into the cat.", [40, 16, 320, 256]]], [["The cheese walk scores minus eight. The cat walk, minus seven.", [30, 292, 340, 87]], ["The cat wins!", [30, 292, 340, 87]]], "The agent chases the return you wrote down, not the one you meant."],
   draw(t) { maze({ t }); route(SHORT, C.gold, -4, clamp((t - .5) / 1.5), 520); route(QCAT, C.rose, 4, clamp((t - 2.5) / 1.5), 540); pip(cx(0), cy(3), 1, t);
     rcard([{ parts: ['cheese, 7 steps:   1 − 1.5 × 6  =  −8'], c: C.goldInk }, { parts: ['cat, 5 steps:   −1 − 1.5 × 4  =  −7    higher!'], c: C.roseInk }, { parts: ['a cost above 1 per step makes the cat the best plan'], c: C.soft }], t - 3.6, { gap: 1.5 }); } },

 { h: 'Reward what you want', p: ['This happens with real agents. In 2016, OpenAI trained an agent on the boat racing game CoastRunners, where points come from hitting targets along the course.', 'The agent found a lagoon where three targets kept reappearing. It kept circling there, crashing and catching fire, never finishing the race, and still scored more than finishing would. The reward said points, not winning.'],
   shot: [4, 10],
   say: ["This happens with real agents.", [["In a boat racing game, points came from hitting targets.", [30, 20, 360, 240]], ["Watch the boat circle the lagoon, where targets keep coming back.", [30, 20, 360, 240]]], [["It crashes, catches fire, never finishes, and still scores more than winning would.", [30, 292, 340, 87]], "The reward said points, not winning."]],
   draw(t) {
     // lagoon + race course
     ctx.save(); ctx.fillStyle = '#d7e8ec'; ctx.beginPath(); ctx.ellipse(150, 150, 110, 80, 0, 0, 7); ctx.fill(); ctx.restore();
     ctx.save(); ctx.strokeStyle = C.line; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.ellipse(150, 150, 110, 80, 0, 0, 7); ctx.stroke(); ctx.restore();
     rline(260, 150, 380, 150, 700, { c: C.line, w: 1.4 }); rline(260, 110, 380, 110, 701, { c: C.line, w: 1.4 }); txt('the race goes on →', 320, 96, { s: 14, c: C.soft }); rline(372, 110, 372, 150, 702, { c: C.ink, w: 2 }); txt('finish', 372, 166, { s: 13, c: C.soft });
     const ang = t * 1.6, bx = 150 + Math.cos(ang) * 60, by = 150 + Math.sin(ang) * 42, TA = [-2.3, -.6, 1.4]; let hits = 0;
     TA.forEach((ai, i) => { const d = ang - ai, laps = d > 0 ? Math.floor(d / (2 * Math.PI)) + 1 : 0, since = d > 0 ? d % (2 * Math.PI) : 9; hits += laps;
       if (since > .9) L.coin(150 + Math.cos(ai) * 60, 150 + Math.sin(ai) * 42, 9, '★'); else if (since < .6) bubble('+10', 150 + Math.cos(ai) * 60, 150 + Math.sin(ai) * 42 - 10, since / .6 * .9, C.goldInk); });
     boat(bx, by, ang + Math.PI / 2);
     if (t > 2) { const fl = (t * 3) % 1; txt('🔥', bx - 6, by - 14 - fl * 4, { s: 14, a: .6 + .4 * Math.sin(t * 9) }); }
     const pts = hits * 10; txt(`points: ${pts}`, 40, 40, { s: 18, al: 'left', c: C.goldInk });
     card(['reward: points for hitting targets', 'what the designers meant: win the race', { s: 'the agent found a loop that scores more', c: C.roseInk }], t - 3.5, { gap: 1.3 }); } },

 { h: 'Tasks that never end', p: ['Pip’s walks always end. Many tasks don’t: a thermostat or a trading program just keeps going. These are **continuing tasks**; their sum has no last term.', 'Earning +1 every step, such an agent’s return would be 1, 2, 3, growing forever. Two very different agents could both have an infinite return, and we could not compare them. The next reel fixes this.'],
   shot: [4, 11],
   say: [[["Pip's walks always end. Many tasks don't.", [20, 20, 370, 100]], ["Up top, the episode stops, and the sum is one.", [20, 20, 370, 100]]], [["Below, a task that just keeps going, plus one every step.", [20, 120, 370, 110]], ["Watch the sum grow, and grow, forever.", [20, 120, 370, 110]]], ["Two agents with infinite returns can't be compared. The next reel fixes that.", [30, 292, 340, 87]]],
   draw(t) {
     txt('episodic: the sum stops', 30, 40, { s: 16, al: 'left', c: C.soft });
     chips([0, 0, 0, 0, 0, 0, 1], 44, 72, Math.floor(t * 2) + 1, { gap: 28 }); if (t > 3.5) txt('= 1', 250, 72, { s: 18, al: 'left', c: C.goldInk, a: clamp(t - 3.5) }); txt('end', 236, 98, { s: 13, c: C.soft, a: clamp(t - 3.5) });
     txt('continuing: +1 every step, forever', 30, 140, { s: 16, al: 'left', c: C.soft, a: clamp(t - 3) });
     const n = clamp(Math.floor((t - 3.5) * 2), 0, 12); chips(new Array(12).fill(1), 44, 172, n, { gap: 28 }); if (n >= 12) txt('…', 384, 172, { s: 20, al: 'right' });
     if (n > 0) txt(`sum so far: ${n}`, 30, 210, { s: 18, al: 'left', c: C.goldInk }); if (t > 10) txt('→ ∞', 160, 210, { s: 20, al: 'left', c: C.roseInk, a: clamp(t - 10) });
     card(['episodic task: has a last step T', 'continuing task: no last step, the sum can blow up', { s: 'fix in the next reel: the discount factor γ', c: C.soft }], t - 6.5, { gap: 1.2 }); } },

 { h: 'Your turn', ctl: { label: 'step cost', values: COSTS, def: 3, fmt: v => String(v) },
   p: ['Drag the step cost and watch three walks: the shortest way to the cheese, the 14-step episode 1, and the quickest way into the cat. Bars show each walk’s return.', 'At 0 the two cheese walks tie. Small costs reward hurrying. Just above 0.2 the quick cat walk overtakes episode 1, and above 1 it beats even the shortest cheese walk.'],
   shot: [3],
   say: [[["Your turn. Drag the step cost.", [20, 286, 370, 110]], ["Three walks, three bars.", [20, 286, 370, 110]], ["The shortest cheese walk, episode one, and the quick dash into the cat.", [40, 16, 320, 256]]], [["At zero, the cheese walks tie.", [20, 286, 370, 110]], ["Push the cost up and watch the cat walk climb, until above one it's the best plan.", [20, 286, 370, 110]]]],
   draw(t, st) { const cost = COSTS[st.ci ?? 3], prev = st.prevCi != null ? COSTS[st.prevCi] : cost, u = ease((t - (st.changed ?? -9)) / .7), cc = lerp(prev, cost, u);
     const W3 = [{ n: 'shortest cheese, 7 steps', tr: TS, col: C.gold, ink: C.goldInk, P: SHORT, off: -5 }, { n: 'episode 1, 14 steps', tr: E1, col: C.teal, ink: '#2f7f7b', off: 0 }, { n: 'quick cat, 5 steps', tr: TQ, col: C.rose, ink: C.roseInk, P: QCAT, off: 5 }];
     maze({ t }); trail(E1.map(q => [q.c, q.r]), .9); route(SHORT, C.gold, -5, 1, 520); route(QCAT, C.rose, 5, 1, 540); pip(cx(0), cy(3), 1, t);
     const gs = W3.map(w => ret(w.tr, cost)), gsAnim = W3.map(w => ret(w.tr, cc)), best = Math.max(...gs);
     const bx = 296, mxN = Math.max(.01, ...gsAnim.map(g => -g)), mxP = Math.max(.01, ...gsAnim), sc = Math.min(110 / mxN, 60 / mxP); rline(bx, 296, bx, 392, 950, { c: C.ink, w: 1.2 }); txt('0', bx, 290, { s: 12, c: C.soft });
     W3.forEach((w, j) => { const y = 302 + j * 33, g = gsAnim[j], len = g * sc, top = Math.abs(gs[j] - best) < 1e-9;
       txt(w.n + (top ? '   ★ best' : ''), 26, y, { s: 14, al: 'left', c: w.ink }); box(g >= 0 ? bx : bx + len, y + 8, Math.max(1.5, Math.abs(len)), 13, w.col, .85);
       txt(fmt(gs[j]), g >= 0 ? bx + Math.abs(len) + 6 : bx + len - 6, y + 15, { s: 15, al: g >= 0 ? 'left' : 'right', c: w.ink }); });
     caption(Math.abs(gs[0] - gs[2]) < 1e-9 ? 'cheese and cat tie: at a cost of 1, only speed counts' : gs[2] > gs[0] ? 'the cat is the best plan now!' : gs[2] > gs[1] ? 'the quick cat walk now beats the 14-step cheese walk' : cost === 0 ? 'no cost: both cheese walks score 1' : 'shorter cheese walks score higher'); } },

 { h: 'The big bet', p: ['RL rests on one idea, the **reward hypothesis**: any goal can be written as getting the largest **expected return**, the total reward averaged over the agent’s own random choices and the environment’s. Richard Sutton, a founder of the field, framed it.', 'It is a working assumption, not a theorem, and still debated. In practice it has carried RL far, given carefully designed rewards.'],
   shot: [5],
   say: ["All of RL rests on one bet, the reward hypothesis.", [["Any goal can be written as getting the largest expected return.", [30, 292, 340, 87]], "Expected just means averaged over all the luck along the way."], "It's an assumption, not a theorem. But with careful rewards, it has carried RL a long way."],
   draw(t) { maze({ t }); route(SHORT, C.gold, 0, 1, 520); pip(cx(0), cy(3), 1, t);
     card(['the reward hypothesis:', 'every goal = the largest expected return', { s: 'expected = averaged over all the luck along the way', c: C.soft }], t - .5, { gap: 1.3 }); } },

 { h: 'Recap', p: ['A **reward** scores one step. The **return** adds up all the rewards from a step until the episode ends, and that total is what an agent tries to make large.', 'How you set the rewards decides what “good” means: a step cost makes short walks win, and too big a cost can make the cat win.'],
   shot: [6],
   say: [[["So, a reward scores one step.", [40, 16, 320, 256]], ["The return adds up every reward from a step until the end, and that's what an agent tries to make big.", [30, 292, 340, 110]]], ["And how you set the rewards decides what good means. Pick them carefully."]],
   draw(t) { maze({ t }); route(SHORT, C.gold, 0, 1, 520); pip(cx(0), cy(3), 1, t);
     rcard([[['G', 't'], ' = ', ['R', 't+1'], ' + ', ['R', 't+2'], ' + … + ', ['R', 'T']], [['G', 't'], ' = ', ['R', 't+1'], ' + ', ['G', 't+1'], '   (count backward)'], 'judge moves by the return, not the next reward', { parts: ['the return you write is the goal the agent chases'], c: C.goldInk }], t - .3, { gap: 1 }); } },
];

RLStory.register('returns', { title: 'Rewards & Returns', track: 'Foundations', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { SEED, SAME, short01: ret(TS, COST), ep10: ret(EPS[9], COST), ep1: ret(E1, COST), ep5: ret(EPS[4], COST), qcat01: ret(TQ, COST), back: TS.map((_, k) => r1(ret(TS, COST, k))),
    MEAN01, BESTR, WORSTR, costs: COSTS.map(c => [c, r1(ret(TS, c)), r1(ret(E1, c)), r1(ret(TQ, c))]), lens: [TS.length, TQ.length] } });
})();
