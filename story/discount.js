// The discount factor as a story: Pip learns why later rewards should count less. Same maze and the same 100 random
// episodes as the loop and returns reels (seed 769); plus a seeded "the walk may be cut short" simulation and a crumb choice.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, maze, trail, pip, ox: OX, oy: OY, cs: CS } = G;
const no = (c, r) => r * G.cols + c + 1;

// ───── simulations ─────
const SEED = 769, CUT_SEED = 7;
function walks(seed, n) {
  const R = L.rng(seed), out = [];
  for (let e = 0; e < n; e++) { let [c, r] = G.start; const tr = [];
    for (let k = 0; k < 5000; k++) { const m = Math.floor(R() * 4), [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr); tr.push({ c, r, m, nc, nr, rew }); c = nc; r = nr; if (rew) break; }
    out.push(tr); }
  return out;
}
const EPS = walks(SEED, 100), LENS = EPS.map(e => e.length), ENDS = EPS.map(e => e[e.length - 1].rew);
const dret = (rews, g, from = 0) => { let s = 0; for (let k = rews.length - 1; k >= from; k--) s = rews[k] + g * s; return s; };   // R_{t+1} + γ R_{t+2} + …
const SHORT = [[0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [4, 2], [4, 1], [4, 0]];
const SR = SHORT.slice(1).map(([c, r]) => G.term(c, r));     // rewards along the shortest walk: 0 ×6, then +1
const epR = e => e.map(q => q.rew);
const SAME = [9, 0, 4];                                        // episodes 10, 1, 5 (all cheese; 10, 14, 147 steps)
// "the walk may be cut short": after every step, with probability 0.1, the episode stops with nothing more. Pip walks the shortest route.
function cutSim(n, p, seed) { const R = L.rng(seed); let got = 0; const ends = new Array(8).fill(0); for (let i = 0; i < n; i++) { let k = 1; for (; k < 7; k++) if (R() < p) break; ends[k]++; if (k === 7) got++; } return { got, n, ends }; }
const CUT = cutSim(10000, .1, CUT_SEED);
// a never-ending +1 stream: partial sums of 1 + γ + γ² + …
const psum = (g, n) => { let s = 0; for (let k = 0; k < n; k++) s += g ** k; return s; };
// the crumb choice: a crumb worth +0.3 in cell 6 also ends the walk. Value iteration finds the best plan for each γ.
const CRUMB = [0, 1], CRUMBV = .3;
const termC = (c, r) => (c === CRUMB[0] && r === CRUMB[1]) ? CRUMBV : G.term(c, r);
function solveCrumb(g) {
  const V = new Float64Array(20);
  for (let it = 0; it < 200; it++) for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.isRock(c, r) || termC(c, r)) continue; let b = -1e9; for (let m = 0; m < 4; m++) { const [nc, nr] = G.step(c, r, m), rw = termC(nc, nr); b = Math.max(b, rw + (rw ? 0 : g * V[G.key(nc, nr)])); } V[G.key(c, r)] = b; }
  // greedy path from the start
  let [c, r] = G.start; const P = [[c, r]];
  for (let k = 0; k < 20 && !termC(c, r); k++) { let b = -1e9, bm = 0; for (let m = 0; m < 4; m++) { const [nc, nr] = G.step(c, r, m); if (nc === c && nr === r) continue; const rw = termC(nc, nr), q = rw + (rw ? 0 : g * V[G.key(nc, nr)]); if (q > b + 1e-12) { b = q; bm = m; } } [c, r] = G.step(c, r, bm); P.push([c, r]); }
  return { V, P, v0: V[G.key(...G.start)], goal: termC(c, r) === CRUMBV ? 'crumb' : termC(c, r) > 0 ? 'cheese' : 'cat' };
}
const GAMMAS = [.1, .3, .5, .7, .75, .8, .85, .9, .95, .99];
const SOL = {}; GAMMAS.concat([.9]).forEach(g => { SOL[g] = solveCrumb(g); });
const crumbPlan = g => CRUMBV * g, cheesePlan = g => g ** 6;      // crumb: reward on step 2 (×γ); cheese: reward on step 7 (×γ⁶)
const SWITCH = CRUMBV ** (1 / 5);                                   // γ⁵ = 0.3
// spread of discounted returns over the 100 random episodes
const spread = g => EPS.map(e => dret(epR(e), g));

// ───── helpers ─────
const F = s => `${s}px 'Patrick Hand', cursive`;
function width(s, size) { ctx.save(); ctx.font = F(size); const w = ctx.measureText(s).width; ctx.restore(); return w; }
// rich text: parts = strings, [base, sub] or { sup: [base, sup] }
function rich(parts, x, y, o = {}) {
  const s = o.s || 17, ss = Math.round(s * .66);
  const wOf = p => typeof p === 'string' ? width(p, s) : p.sup ? width(p.sup[0], s) + width(p.sup[1], ss) + 1 : width(p[0], s) + width(p[1], ss) + 1;
  const ws = parts.map(wOf), tot = ws.reduce((a, b) => a + b, 0);
  let X = o.al === 'center' ? x - tot / 2 : o.al === 'right' ? x - tot : x;
  parts.forEach((p, i) => {
    if (typeof p === 'string') txt(p, X, y, { s, c: o.c, a: o.a, al: 'left' });
    else { const [b, e] = p.sup || p, wb = width(b, s); txt(b, X, y, { s, c: o.c, a: o.a, al: 'left' }); txt(e, X + wb + 1, y + (p.sup ? -s * .32 : s * .3), { s: ss, c: o.c, a: o.a, al: 'left' }); }
    X += ws[i];
  });
  return tot;
}
function rcard(rows, t, o = {}) {
  const x = o.x ?? 30, y = o.y ?? 292, w = o.w ?? 340, lh = o.lh ?? 23, h = 18 + rows.length * lh, a = clamp(t * 3);
  if (a <= 0) return; box(x, y, w, h, C.card, a); rrect(x, y, w, h, 200 + rows.length, { c: C.line, w: 1.1 });
  rows.forEach((row, i) => { const u = clamp((t - (o.gap ?? 1.1) * i) * 2.5); if (u <= 0) return; const parts = typeof row === 'string' ? [row] : Array.isArray(row) ? row : row.parts; rich(parts, x + 14, y + 20 + i * lh, { s: o.fs || 17, a: u, c: row.c }); });
}
const g2 = (b, e) => ({ sup: [b, e] });
const f2 = v => { const x = Math.round(v * 100) / 100; return (Object.is(x, -0) ? 0 : x).toFixed(2).replace(/0$/, '').replace(/\.0?$/, '').replace('-', '−'); };
function dim(a = .7) { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, 286); ctx.restore(); }
function route(P, col, off = 0, p = 1, seed = 500) { const n = (P.length - 1) * clamp(p); for (let i = 0; i < Math.ceil(n); i++) { const u = clamp(n - i), [a, b] = [P[i], P[i + 1]]; rline(cx(a[0]) + off, cy(a[1]) + off, lerp(cx(a[0]), cx(b[0]), u) + off, lerp(cy(a[1]), cy(b[1]), u) + off, seed + i, { c: col, w: 3, once: true, j: .6 }); } }
function walkPath(P, t, dur) { const k = clamp(Math.floor(t / dur), 0, P.length - 1), u = ease((t % dur) / dur * 1.3), a = P[k], b = P[Math.min(P.length - 1, k + 1)]; return k >= P.length - 1 ? { x: cx(a[0]), y: cy(a[1]), k, done: true, face: 1 } : { x: lerp(cx(a[0]), cx(b[0]), u), y: lerp(cy(a[1]), cy(b[1]), u), k, face: b[0] - a[0] }; }
function crumb(x, y, s = 1) { [[-5, 2, 4], [3, -1, 3.2], [1, 5, 2.4], [-1, -5, 1.8]].forEach(([a, b, r], i) => rcircle(x + a * s, y + b * s, r * s, 880 + i, { fill: '#c08a54', c: C.ink, w: 1, j: .4 })); }
// weight bars: bar k = γ^k, how much a reward k steps after the first one counts
function weightBars(g, n, o = {}) {
  const x0 = o.x0 ?? 50, x1 = o.x1 ?? 370, y0 = o.y0 ?? 250, h = o.h ?? 150, w = (x1 - x0) / n, show = o.show ?? n;
  rline(x0, y0, x1, y0, 960, { c: C.line, w: 1 }); rline(x0, y0, x0, y0 - h - 6, 961, { c: C.line, w: 1 });
  txt('1', x0 - 6, y0 - h, { s: 12, c: C.soft, al: 'right' }); txt('0', x0 - 6, y0, { s: 12, c: C.soft, al: 'right' });
  for (let k = 0; k < Math.min(show, n); k++) { const v = g ** k, bh = v * h, a = o.grow ? clamp(o.grow(k)) : 1; box(x0 + k * w + w * .15, y0 - bh * a, w * .7, bh * a, o.col || C.gold, .9);
    if (o.labels && a >= 1) txt(f2(v), x0 + k * w + w / 2, y0 - bh - 9, { s: 12, c: C.goldInk }); if (o.ticks && (n <= 12 || k % 5 === 0)) txt(String(k), x0 + k * w + w / 2, y0 + 11, { s: 12, c: C.soft }); }
}

// ───── the steps ─────
const STEPS = [
 { h: 'Two loose ends', p: ['The last reel left two problems. With plain rewards, a 10-step and a 147-step walk to the cheese both return 1; a step cost fixed that, but too big a cost made the cat the best plan.', 'And a never-ending task’s sum can grow forever. **Discounting** helps with both. Its knob must be chosen too, but it can never make the cat look good.'],
   shot: [4, 10],
   draw(t) { maze({ t }); const e = EPS[4], n = Math.min(e.length, Math.floor(t * 30)); trail(e.slice(0, n).map(q => [q.c, q.r]), .8); const q = e[Math.max(0, n - 1)]; pip(cx(n >= e.length ? q.nc : q.c), cy(n >= e.length ? q.nr : q.r), 1, t);
     card(['10 steps or 147 steps: return 1 either way', 'a step cost: too big, and the cat wins', { s: 'never-ending tasks: the sum can blow up', c: C.soft }], t - 1, { gap: 1.6 });
     caption(n < e.length ? `episode 5, step ${n}` : 'episode 5: 147 steps, return 1'); } },

 { h: 'Sooner is better', p: ['Would you take 100 rupees today or 100 rupees in a year? Most people take it today. You can use it now, and who knows what happens in a year.', 'So a reward far in the future should count for a little less than the same reward now. RL builds this in with one number.'],
   shot: [4, 10],
   draw(t) { dim(0);
     const coinsAt = (x, y, n, a) => { for (let i = 0; i < n; i++) alpha(a, () => L.coin(x + (i % 5) * 16 - 32, y - Math.floor(i / 5) * 9, 9, '₹')); };
     txt('today', 110, 70, { s: 20, a: clamp(t - .4) }); txt('in a year', 290, 70, { s: 20, a: clamp(t - 1.2) });
     coinsAt(110, 160, 10, clamp(t - .6)); coinsAt(290, 160, 10, clamp(t - 1.4)); txt('100', 110, 190, { s: 22, c: C.goldInk, a: clamp(t - .8) }); txt('100', 290, 190, { s: 22, c: C.goldInk, a: clamp(t - 1.6) });
     if (t > 3) { const u = ease((t - 3) / .8); rrect(60, 46, 100, 160, 980, { c: C.teal, w: 2.4, p: u }); txt('most people pick this', 110, 228, { s: 15, c: '#2f7f7b', a: u }); }
     if (t > 5) txt('?', 290, 120, { s: 40, c: C.soft, a: clamp(t - 5) * .7 });
     card(['you can use it sooner', 'the future is uncertain', { s: 'so later rewards count a bit less', c: C.goldInk }], t - 5.5, { gap: 1.2 }); } },

 { h: 'The discount factor γ', p: ['That number is the **discount factor**, written **γ (gamma)**, from 0 to 1. The first reward after now counts fully. A reward one step after it is multiplied by γ; two steps after, by γ times γ, written γ²; k steps after, by γ to the power k.', 'The bars show γ = 0.9: 1, 0.9, 0.81, 0.73, and on. γ = 1 means no discount; γ = 0 means only the first reward counts.'],
   shot: [4, 11],
   draw(t) { weightBars(.9, 12, { y0: 230, h: 160, labels: true, ticks: true, grow: k => (t - 1 - k * .45) * 2.5 });
     txt('how much a reward counts', 50, 50, { s: 15, al: 'left', c: C.soft }); txt('steps after the first reward →', 370, 260, { s: 14, al: 'right', c: C.soft, a: clamp(t - 1) });
     rcard([[g2('γ', '0'), ' = 1,   ', g2('γ', '1'), ' = 0.9,   ', g2('γ', '2'), ' = 0.9 × 0.9 = 0.81'], [g2('γ', '3'), ' = 0.73,  …,   ', g2('γ', '10'), ' = 0.35'], { parts: ['γ near 1: shrinks slowly · γ near 0: fast'], c: C.soft }], t - 7, { gap: 1.2 }); } },

 { h: 'The discounted return', p: ['Now redefine the return. The first reward counts fully, the next is multiplied by γ, the one after by γ², and so on.', 'G t = R t+1 + γ R t+2 + γ² R t+3 + … With γ = 1 this is the plain sum from the last reel. With γ below 1, every reward still counts, just less the later it comes.'],
   shot: [5, 11],
   draw(t) { maze({ t }); dim(.94);
     rich([['G', 't'], ' = ', ['R', 't+1'], ' + γ ', ['R', 't+2'], ' + ', g2('γ', '2'), ' ', ['R', 't+3'], ' + …'], W / 2, 90, { s: 28, al: 'center', a: clamp(t - .4) });
     const lab = (s, x, y, a) => txt(s, x, y, { s: 14, c: C.soft, a });
     lab('counts fully', 150, 134, clamp(t - 1.6)); lab('× 0.9', 232, 134, clamp(t - 2.4)); lab('× 0.81', 318, 134, clamp(t - 3.2));
     if (t > 4.5) rich(['γ = 1:  ', ['G', 't'], ' = ', ['R', 't+1'], ' + ', ['R', 't+2'], ' + …   the plain sum'], W / 2, 190, { s: 17, al: 'center', a: clamp(t - 4.5), c: C.soft });
     card(['the first reward: weight 1', 'each step later: one more factor of γ', { s: 'a reward k steps after the first: × γ to the k', c: C.goldInk }], t - 6, { gap: 1.2 }); } },

 { h: 'Back in the maze', p: ['Drop the step cost: cheese +1, cat −1, other steps 0 again, and take γ = 0.9. On the shortest walk, the +1 is the seventh reward, six steps after the first.', 'So it counts 0.9 to the sixth: 0.9 × 0.9 × 0.9 × 0.9 × 0.9 × 0.9 = 0.53. The discounted return of the shortest walk is 0.53.'],
   shot: [4, 11],
   draw(t) { maze({ t }); route(SHORT, C.gold, 0, clamp((t - .3) / 2), 520); const m = walkPath(SHORT, clamp(t - 2.4, 0, 99), .55); pip(m.x, m.y, m.face, t);
     SHORT.slice(1).forEach(([c, r], k) => { const a = clamp((t - 2.6 - k * .55) * 2); if (a <= 0) return; rich(['× ', g2('0.9', String(k))], cx(c), cy(r) + 22, { s: 13, al: 'center', a, c: C.soft }); });
     if (t > 2.6 && t < 7.6) caption('label in a cell = weight of the reward for stepping onto it');
     if (t > 6.6) bubble('+1 × 0.53', cx(4) - 10, cy(0) - 16, t - 6.6);
     rcard([['reward 7 counts  ', g2('0.9', '6'), '  =  0.53'], { parts: ['discounted return of the shortest walk:  0.53'], c: C.goldInk }], t - 7.6, { gap: 1.2 }); } },

 { h: 'Long walks lose, cats stay bad', p: [`Now the same three random episodes. Episode 10, 10 steps: 0.9 to the ninth, 0.39. Episode 1, 14 steps: 0.25. Episode 5, 147 steps: about 0.0000002, practically nothing.`, 'Shorter walks win, with no step cost at all. And a discounted cheese is always above 0 while a discounted cat is always below 0, so the cat can never become the best plan. Wandering forever earns 0: better than the cat, worse than any cheese.'],
   shot: [4, 11],
   draw(t) { maze({ t }); dim(.35); const rows = [['shortest walk', 7, cheeseD(7)], ...SAME.map(i => [`episode ${i + 1}`, LENS[i], dret(epR(EPS[i]), .9)])];
     rows.forEach(([n, l, g], j) => { const a = clamp((t - .6 - j * 1.4) * 2), y = 306 + j * 24; if (a <= 0) return;
       txt(n, 36, y, { s: 16, al: 'left', a, c: j ? C.ink : C.goldInk }); txt(`${l} steps`, 146, y, { s: 15, al: 'left', c: C.soft, a }); rich([g2('0.9', String(l - 1)), ' = ' + (g < .001 ? '0.0000002' : f2(g))], 224, y, { s: 15, a, c: C.goldInk }); });
     if (t > 6.5) { box(cx(4) - 182, cy(0) - 12, 150, 24, C.bg, .8 * clamp(t - 6.5)); txt('cheese, discounted: > 0', cx(4) - 34, cy(0), { s: 16, c: C.goldInk, a: clamp(t - 6.5), al: 'right' }); box(cx(3) - 150, cy(1) - 8, 124, 24, C.bg, .8 * clamp(t - 7.2)); txt('cat, discounted: < 0', cx(3) - 28, cy(1) + 4, { s: 16, c: C.roseInk, a: clamp(t - 7.2), al: 'right' }); } } },

 { h: 'Counting backward again', p: ['As before, count from every step of the shortest walk. From the last cell before the cheese, the return is 1. One step earlier, the +1 comes one step later: 0.9. Then 0.81, 0.73, and so on, down to 0.53 at the start.', 'The rule: G t = R t+1 + γ × G t+1. A later reel builds on this rule; it is the seed of what is called the Bellman equation.'],
   shot: [5, 12],
   draw(t) { maze({ t }); route(SHORT, C.gold, 0, 1, 520); pip(cx(0), cy(3), 1, t);
     const n = SR.length; for (let k = n - 1; k >= 0; k--) { const a = clamp((t - .8 - (n - 1 - k) * .9) * 2); if (a <= 0) continue; const [c, r] = SHORT[k], g = dret(SR, .9, k);
       box(cx(c) - 20, cy(r) - 30, 40, 18, C.card, .9 * a); rich([['G', String(k)], ' ' + f2(g)], cx(c), cy(r) - 21, { s: 15, al: 'center', a, c: C.goldInk }); }
     rcard([[['G', '6'], ' = ', ['R', '7'], ' = 1'], [['G', '5'], ' = ', ['R', '6'], ' + 0.9 × ', ['G', '6'], ' = 0 + 0.9 × 1 = 0.9'], [['G', 't'], ' = ', ['R', 't+1'], ' + γ × ', ['G', 't+1']]], t - 7, { gap: 1.3 }); } },

 { h: 'Reason 1: the future may not come', p: ['Why multiply by γ? Here is one honest reason. Suppose that after every step there is a 1 in 10 chance the walk is simply cut short: the lights go out, or Pip is called home, and nothing more is earned, good or bad.', `We sent Pip along the shortest walk 10,000 times like that. It reached the cheese ${CUT.got.toLocaleString('en-US')} times: ${(CUT.got / CUT.n * 100).toFixed(1)} percent, close to 0.9 to the sixth, 0.53.`],
   shot: [5, 12],
   draw(t) { maze({ t }); route(SHORT, C.gold, 0, 1, 520);
     const runs = Math.min(10000, Math.floor(clamp((t - 1) / 6) ** 2 * 10000)); const R = L.rng(CUT_SEED); let got = 0; const ends = new Array(8).fill(0);
     for (let i = 0; i < runs; i++) { let k = 1; for (; k < 7; k++) if (R() < .1) break; ends[k]++; if (k === 7) got++; }
     const demo = walkPath(SHORT, (t % 4.2), .55); pip(demo.x, demo.y, demo.face, t);
     // stopped-early counts on the path cells
     for (let k = 1; k < 7; k++) { const [c, r] = SHORT[k]; if (ends[k]) txt(`✕ ${ends[k]}`, cx(c), cy(r) + 22, { s: 13, c: C.roseInk }); }
     card([`runs: ${runs.toLocaleString('en-US')}`, `reached the cheese: ${got.toLocaleString('en-US')}  (${runs ? (got / runs * 100).toFixed(1) : '0'}%)`, { s: '✕ = walks cut short in that cell · 0.9⁶ = 53.1%', c: C.soft }], t - .4, { gap: .6 }); } },

 { h: 'γ as a survival chance', p: ['So γ = 0.9 behaves like a 9 in 10 chance, at every step, that the world keeps going. A reward six steps away survives six such chances: 0.9 to the sixth.', 'On average over the cut-offs, the plain return in such a fragile world equals the discounted return, exactly. That is reason one: **the far future is less certain**. It is a way to read γ; in practice γ is a setting the designer picks.'],
   shot: [9],
   draw(t) { maze({ t }); dim(.92);
     for (let k = 0; k < 7; k++) { const a = clamp((t - .6 - k * .6) * 2), x = 50 + k * 50; if (a <= 0) continue; rcircle(x, 140, 16, 990 + k, { fill: k < 6 ? '#cfe3e2' : C.yellow, c: C.ink, j: .6 }); txt(k < 6 ? '0.9' : '+1', x, 141, { s: 14, a }); if (k < 6) txt('×', x + 25, 141, { s: 16, c: C.soft, a }); }
     txt('keep going? 9 in 10, six times, then the cheese', W / 2, 186, { s: 15, c: C.soft, a: clamp(t - 4.5) });
     card(['γ = chance the world goes on, each step', 'reward k steps after the first × γ to the k', { s: 'average over luck = discounted return', c: C.goldInk }], t - 5.5, { gap: 1.2 }); } },

 { h: 'Reason 2: sums stay finite', p: ['Back to the never-ending task that pays +1 every step. The plain sum grows forever. Discounted with γ = 0.9: 1 + 0.9 + 0.81 + … reaches 6.5 after 10 steps, 8.8 after 20, 9.9 after 50.', 'It never passes 10: by the geometric series from algebra, the sum of γ to the k is 1 ÷ (1 − γ). Every agent now has a finite return to compare.'],
   shot: [5, 12],
   draw(t) { const n = clamp(Math.floor((t - .5) * 6), 0, 60), x0 = 50, x1 = 370, y0 = 250, Y = v => y0 - v / 10 * 180, X = k => x0 + k / 60 * (x1 - x0);
     rline(x0, y0, x1, y0, 970, { c: C.line, w: 1 }); rline(x0, y0, x0, Y(10.6), 971, { c: C.line, w: 1 }); rline(x0, Y(10), x1, Y(10), 972, { c: C.rose, w: 1.2 });
     txt('10', x0 - 6, Y(10), { s: 12, c: C.roseInk, al: 'right' }); txt('limit', x1, Y(10) - 9, { s: 12, c: C.roseInk, al: 'right' }); txt('0', x0 - 6, y0, { s: 12, c: C.soft, al: 'right' }); txt('total so far', x0 + 4, Y(11.3), { s: 14, c: C.soft, al: 'left' }); txt('steps →', x1, y0 + 12, { s: 13, c: C.soft, al: 'right' });
     [10, 20, 50].forEach(k => { if (n >= k) { dot(X(k), Y(psum(.9, k)), 4, C.goldInk); txt(psum(.9, k).toFixed(1), X(k) + 4, Y(psum(.9, k)) + 14, { s: 13, c: C.goldInk, al: 'left' }); txt(String(k), X(k), y0 + 12, { s: 12, c: C.soft }); } });
     for (let k = 1; k <= n; k++) rline(X(k - 1), Y(psum(.9, k - 1)), X(k), Y(psum(.9, k)), 973 + k, { c: C.gold, w: 2, once: true, j: .3 });
     for (let k = 1; k <= Math.min(n, 10); k++) dot(X(k), Y(k), 2.4, C.soft); if (n > 3) txt('plain sum: off the chart', X(9) + 6, Y(10.6), { s: 13, c: C.soft, al: 'left' });
     rcard([['1 + 0.9 + 0.81 + …  =  1 ÷ (1 − 0.9)  =  10'], { parts: ['the limit 1 ÷ (1 − γ) for any γ below 1'], c: C.soft }], t - 8, { gap: 1.2 }); } },

 { h: 'How far ahead does it look?', p: ['That same number, 1 ÷ (1 − γ), is a handy rule of thumb for how far ahead an agent effectively looks. With γ = 0.9 it is 10 steps; with γ = 0.99 it is 100.', 'Why that number? In the fragile-world reading, a world that ends with chance 1 in 10 per step lasts 10 steps on average. It is not a hard cut-off: at that distance a reward still counts about 0.35.'],
   shot: [5, 12],
   draw(t) { const x0 = 50, x1 = 370, y0 = 240, h = 170, N = 150, X = k => x0 + k / N * (x1 - x0), Y = v => y0 - v * h;
     rline(x0, y0, x1, y0, 960, { c: C.line, w: 1 }); rline(x0, y0, x0, y0 - h - 6, 961, { c: C.line, w: 1 }); txt('1', x0 - 6, Y(1), { s: 12, c: C.soft, al: 'right' }); txt('0', x0 - 6, y0, { s: 12, c: C.soft, al: 'right' });
     txt('how much a reward k steps later counts', x0 + 4, Y(1) - 16, { s: 14, c: C.soft, al: 'left' }); [0, 50, 100, 150].forEach(k => txt(String(k), X(k), y0 + 12, { s: 12, c: C.soft })); txt('k →', x1, y0 + 24, { s: 13, c: C.soft, al: 'right' });
     [[.9, C.gold, C.goldInk, 1], [.99, C.teal, '#2f7f7b', 4]].forEach(([g, col, ink, t0]) => { const p = clamp((t - t0) / 2); if (p <= 0) return; ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.beginPath(); for (let k = 0; k <= N * p; k++) k ? ctx.lineTo(X(k), Y(g ** k)) : ctx.moveTo(X(k), Y(1)); ctx.stroke(); ctx.restore();
       const hz = 1 / (1 - g), a = clamp(t - t0 - 2); if (a > 0) { alpha(a, () => rline(X(hz), Y(g ** hz) - 4, X(hz), y0, 965 + t0, { c: ink, w: 1.2 })); dot(X(hz), Y(g ** hz), 4, ink, a); txt(`γ = ${g}: about ${Math.round(hz)} steps`, X(hz) + 6, Y(g ** hz) - 12, { s: 14, c: ink, al: 'left', a }); } });
     rcard([['look-ahead  ≈  1 ÷ (1 − γ)'], ['γ = 0.9 → 10 steps,   γ = 0.99 → 100 steps'], { parts: ['at that distance a reward still counts about 0.35'], c: C.soft }], t - 7.5, { gap: 1.2 }); } },

 { h: 'A crumb or the cheese?', p: ['Now γ changes which plan is best. Put a crumb worth +0.3 in cell 6, two steps up. Eating it also ends the walk. The cheese, +1, is seven steps away along the bottom.', `Crumb plan: 0.3 × γ. Cheese plan: γ to the sixth. With γ = 0.5 the crumb wins, 0.15 against 0.016. With γ = 0.9 the cheese wins, 0.53 against 0.27. They swap at γ ≈ ${SWITCH.toFixed(2)}.`],
   shot: [4, 11],
   draw(t) { maze({ t }); crumb(cx(0), cy(1), 1.4); txt('+0.3', cx(0) + 14, cy(1) - 18, { s: 15, c: C.goldInk }); route(SOL[.5].P, C.orange, -4, clamp((t - 1) / 1.2), 540); route(SHORT, C.gold, 4, clamp((t - 2.4) / 2), 520); pip(cx(0), cy(3), 1, t);
     txt('2 steps', cx(0) + 26, cy(2) + 4, { s: 13, c: C.goldInk, al: 'left', a: clamp(t - 2) }); txt('7 steps', cx(2), cy(3) + 22, { s: 13, c: C.goldInk, a: clamp(t - 4) });
     rcard([['crumb:  0.3 × γ           cheese:  ', g2('γ', '6')], ['γ = 0.5:   0.15   vs   0.016   → crumb'], ['γ = 0.9:   0.27   vs   0.53     → cheese']], t - 5, { gap: 1.4 }); } },

 { h: 'Your turn', ctl: { label: 'γ', values: GAMMAS, def: 2, fmt: v => v },
   p: ['Drag **γ**. Pip does not know the maze, but we do, so for each setting we compute the best plan, counting rewards with the discount, and let Pip walk it. Later reels show how Pip can learn this. Bars: each plan’s worth from the start.', `Below about ${SWITCH.toFixed(2)}, the crumb wins: the cheese is too far away to matter. Above it, Pip is patient enough to walk to the cheese.`],
   shot: [3, 8],
   draw(t, st) { const g = GAMMAS[st.ci ?? 2], S = SOL[g], since = t - (st.changed ?? 0);
     maze({ t }); crumb(cx(0), cy(1), 1.4); txt('+0.3', cx(0) + 14, cy(1) - 18, { s: 15, c: C.goldInk });
     route(S.P, S.goal === "crumb" ? C.orange : C.gold, 0, clamp(since / 1.2), 560); const m = walkPath(S.P, Math.max(0, since - 1.4), .5); pip(m.x, m.y, m.face, t);
     const vs = [['crumb plan', crumbPlan(g)], ['cheese plan', cheesePlan(g)]], bx = 150, bw = 200;
     vs.forEach(([n, v], j) => { const y = 312 + j * 34, best = (S.goal === 'crumb') === (j === 0); txt(n, 30, y, { s: 16, al: 'left', c: best ? C.goldInk : C.soft }); box(bx, y - 9, Math.max(1.5, v * bw), 18, best ? C.gold : C.grey, .85); txt(v >= .1 ? f2(v) : v >= .001 ? v.toFixed(3) : v.toPrecision(1), bx + v * bw + 6, y, { s: 15, al: 'left', c: best ? C.goldInk : C.soft }); if (best) txt('★', 120, y, { s: 15, c: C.goldInk }); });
     caption(`γ = ${g}: the best plan is the ${S.goal}${g >= .95 ? ', and even far rewards count a lot' : g <= .3 ? ', almost nothing beyond the next step counts' : ''}`); } },

 { h: 'The price of patience', p: ['Why not always use γ close to 1? The more the far future counts, the more its luck counts too. Here are random Pip’s 100 discounted returns for three values of γ.', 'With γ = 0.5 they all squash to almost 0: steady, but they barely tell the cheese from the cat. With γ = 0.99 they spread almost from −1 to +1, so judging the start from a few walks would depend heavily on which walks you happened to see.'],
   shot: [4, 11],
   draw(t) { const rowsG = [.5, .9, .99], x0 = 60, x1 = 360, X = v => x0 + (v + 1) / 2 * (x1 - x0);
     txt('discounted return of each random episode', 30, 44, { s: 15, al: 'left', c: C.soft });
     rowsG.forEach((g, j) => { const y = 112 + j * 66, a = clamp((t - .6 - j * 1.6) * 2); txt(`γ = ${g}`, 380, y - 38, { s: 15, al: 'right', a });
       rline(x0, y, x1, y, 975 + j, { c: C.line, w: 1 }); [-1, 0, 1].forEach(v => txt(v > 0 ? '+1' : v < 0 ? '−1' : '0', X(v), y + 13, { s: 12, c: C.soft, a }));
       const vals = spread(g), stack = {}; vals.forEach((v, i) => { const b = Math.round(X(v) / 5), k = stack[b] = (stack[b] || 0) + 1, aa = clamp((t - .8 - j * 1.6 - i * .01) * 3); if (aa > 0) dot(b * 5, y - 4 - Math.min(k - 1, 8) * 4, 2.2, ENDS[i] > 0 ? C.teal : C.rose, aa); });
       if (t > 1.6 + j * 1.6) Object.keys(stack).forEach(b => { if (stack[b] > 9) txt(`×${stack[b]}`, b * 5 + 6, y - 40, { s: 12, c: C.ink, al: 'left' }); }); });
     card(['small γ: little spread, but short-sighted', 'γ near 1: far-sighted, but varies a lot', { s: 'tall stacks are capped; ×n = how many', c: C.soft }], t - 6.5, { gap: 1.2 }); } },

 { h: 'Where you meet γ', p: ['Discounting is old: economists discount future money the same way, through interest rates. In RL, γ is part of the problem setup, picked by the designer. DeepMind’s Atari player DQN used γ = 0.99.', 'Games that always end, like Go, can use γ = 1, with no discount at all. And people? Studies suggest we care much more about today versus tomorrow than about day 100 versus day 101, unlike a constant γ.'],
   shot: [8],
   draw(t) {
     const rows = [['Atari, DQN 2015', 'γ = 0.99', 'about 100 steps'], ['typical RL tasks', 'γ = 0.9 to 0.999', '10 to 1,000 steps'], ['Go, chess', 'often γ = 1', 'every game ends']];
     txt('setting', 34, 60, { s: 15, al: 'left', c: C.soft }); txt('discount', 170, 60, { s: 15, al: 'left', c: C.soft }); txt('looks ahead', 282, 60, { s: 15, al: 'left', c: C.soft });
     rows.forEach((r, j) => { const a = clamp((t - .8 - j * 1.2) * 2); r.forEach((s, i) => txt(s, [34, 170, 282][i], 100 + j * 44, { s: 15, al: 'left', a })); });
     card(['γ is chosen by the designer, as part of the task', 'change γ and the best plan can change', { s: 'as the crumb showed', c: C.soft }], t - 4.5, { gap: 1.2 }); } },

 { h: 'Recap', p: ['The **discount factor γ** makes a reward k steps after the first count γ to the k. Later rewards count less because the future is uncertain and sooner is better, and the sum stays finite even in never-ending tasks.', 'Small γ is impatient, γ near 1 is patient but its returns vary more. The agent looks roughly 1 ÷ (1 − γ) steps ahead.'],
   shot: [6],
   draw(t) { maze({ t }); route(SHORT, C.gold, 0, 1, 520); pip(cx(0), cy(3), 1, t);
     rcard([[['G', 't'], ' = ', ['R', 't+1'], ' + γ ', ['R', 't+2'], ' + ', g2('γ', '2'), ' ', ['R', 't+3'], ' + …'], [['G', 't'], ' = ', ['R', 't+1'], ' + γ × ', ['G', 't+1']], 'small γ: impatient · γ near 1: patient', { parts: ['looks about 1 ÷ (1 − γ) steps ahead'], c: C.goldInk }], t - .3, { gap: 1 }); } },
];
function cheeseD(n) { return .9 ** (n - 1); }

RLStory.register('discount', { title: 'The Discount Factor γ', track: 'Foundations', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { short: dret(SR, .9), back: SR.map((_, k) => +dret(SR, .9, k).toFixed(3)), eps: SAME.map(i => [i + 1, LENS[i], dret(epR(EPS[i]), .9)]), CUT, cutFrac: CUT.got / CUT.n, g6: .9 ** 6,
    psum: [10, 20, 50].map(k => +psum(.9, k).toFixed(3)), w10: .9 ** 10, w100: .99 ** 100, SWITCH, crumb: GAMMAS.map(g => [g, SOL[g].goal, +SOL[g].v0.toFixed(4), +crumbPlan(g).toFixed(4), +cheesePlan(g).toFixed(4), SOL[g].P.length - 1]),
    spread: [.5, .9, .99].map(g => { const v = spread(g), m = v.reduce((a, b) => a + b, 0) / 100; return [g, +Math.min(...v).toFixed(3), +Math.max(...v).toFixed(3), +Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / 100).toFixed(4)]; }) } });
})();
