// Reward shaping as a story: in a bigger maze the cheese is hard to find; hints help, a careless hint gets farmed,
// and potential-based hints (Ng, Harada & Russell 1999) are safe. Learner = the qlearning reel's Q-learning.
// All numbers come from the seeded runs below (showcase seed 1; averages over seeds 1-20).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
// ───── world: 8 x 6, cheese top right, start bottom left, five rocks, no cat
const COLS = 8, ROWS = 6, NS = COLS * ROWS;
const G = L.gridWorld({ cols: COLS, rows: ROWS, cheese: [7, 0], cat: null, rocks: [[2, 1], [5, 2], [3, 3], [1, 4], [6, 4]], start: [0, 5], cs: 44, ox: 24, oy: 10 });
const { cx, cy, key, maze, trail, pip } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GAM = 0.9, ALPHA = 0.2, EPS = 0.2, CAP = 1000, NW = 30, NRUN = 20, SEED = 1;
const dist = (c, r) => Math.abs(7 - c) + r;              // columns + rows to the cheese (rocks never make the real way longer here)
const PHI = (c, r) => 1 - dist(c, r) / 12;               // the safe hint's potential: a rough guess of what is still to come
// hint paid on a move from (c, r) to (nc, nr); done = landed on the cheese (walk over, potential 0)
const HINT = {
  none: () => 0,
  hot: (c, r, nc, nr) => .1 * (dist(c, r) - dist(nc, nr)),
  closer: (c, r, nc, nr) => (dist(nc, nr) < dist(c, r) ? .3 : 0),
  pot: (c, r, nc, nr, done) => GAM * (done ? 0 : PHI(nc, nr)) - PHI(c, r),
  neg: (c, r, nc, nr, done) => GAM * (done ? 0 : -.1 * dist(nc, nr)) + .1 * dist(c, r),
};
// Q-learning exactly as in the qlearning reel (eps-greedy over 4 moves, random ties, update on landing), learning from reward + hint
function runQ(mode, seed, nw, keep) {
  const R = rng(seed), Q = new Float64Array(NS * 4), E = [], F = HINT[mode];
  for (let e = 0; e < nw; e++) {
    let [c, r] = G.start; const tr = []; let n = 0, end = 0;
    for (let k = 0; k < CAP; k++) {
      let m; if (R() < EPS) m = Math.floor(R() * 4); else { let b = -1e9, ms = []; for (let a = 0; a < 4; a++) { const q = Q[key(c, r) * 4 + a]; if (q > b + 1e-9) { b = q; ms = [a]; } else if (q > b - 1e-9) ms.push(a); } m = ms[Math.floor(R() * ms.length)]; }
      const [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr), done = rew !== 0, h = F(c, r, nc, nr, done);
      const nx = done ? 0 : Math.max(Q[key(nc, nr) * 4], Q[key(nc, nr) * 4 + 1], Q[key(nc, nr) * 4 + 2], Q[key(nc, nr) * 4 + 3]), i = key(c, r) * 4 + m;
      const old = Q[i]; Q[i] += ALPHA * (rew + h + GAM * nx - Q[i]);
      if (keep) tr.push({ c, r, m, nc, nr, rew, h, old, nw: Q[i], bump: nc === c && nr === r });
      c = nc; r = nr; n++; if (done) { end = 1; break; }
    }
    E.push({ n, end, tr });
  }
  return { E, Q };
}
const argQ = (Q, c, r) => { let b = -1e9, bm = 0; for (let m = 0; m < 4; m++) if (Q[key(c, r) * 4 + m] > b + 1e-12) { b = Q[key(c, r) * 4 + m]; bm = m; } return bm; };
// Pip follows its biggest numbers from the start (no random moves), at most 40 steps
function follow(Q) { let [c, r] = G.start; const P = [[c, r]]; for (let k = 0; k < 40 && !G.term(c, r); k++) { [c, r] = G.step(c, r, argQ(Q, c, r)); P.push([c, r]); } return { P, ok: G.term(...P[P.length - 1]) !== 0 }; }
// averages over NRUN runs (seeds 1..20), memoised per hint
const AV = {};
function avg(mode) {
  if (AV[mode]) return AV[mode];
  const cur = new Float64Array(NW); let ok = 0, beside = 0;
  for (let s = 1; s <= NRUN; s++) { const R = runQ(mode, s, NW); R.E.forEach((e, i) => { cur[i] += e.n / NRUN; }); const F = follow(R.Q); if (F.ok) ok++; else if (F.P.slice(-2).some(([c, r]) => dist(c, r) === 1)) beside++; }
  const first10 = cur.slice(0, 10).reduce((a, b) => a + b, 0) / 10, last10 = cur.slice(NW - 10).reduce((a, b) => a + b, 0) / 10;
  return (AV[mode] = { cur, first10, last10, ok, beside });
}
const SHOW = {}; const show = mode => SHOW[mode] ||= (() => { const R = runQ(mode, SEED, NW, true); return Object.assign(R, { F: follow(R.Q) }); })();
// exact best values with each hint (value iteration, knows the rules: used only to check claims)
function VI(mode) {
  const F = HINT[mode], V = new Float64Array(NS);
  const q = (c, r, m) => { const [nc, nr] = G.step(c, r, m), rw = G.term(nc, nr); return rw + F(c, r, nc, nr, !!rw) + (rw ? 0 : GAM * V[key(nc, nr)]); };
  for (let it = 0; it < 600; it++) for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) { if (G.isRock(c, r) || G.term(c, r)) continue; V[key(c, r)] = Math.max(q(c, r, 0), q(c, r, 1), q(c, r, 2), q(c, r, 3)); }
  let notCloser = 0; for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) { if (G.isRock(c, r) || G.term(c, r)) continue; let b = -1e9, bm = 0; for (let m = 0; m < 4; m++) { const v = q(c, r, m); if (v > b + 1e-9) { b = v; bm = m; } } const [nc, nr] = G.step(c, r, bm); if (dist(nc, nr) !== dist(c, r) - 1) notCloser++; }
  return { V, notCloser };
}
const VIM = {}; const vi = m => VIM[m] ||= VI(m);
// discounted hint total along a list of cells (potential hint)
function hintTotal(P) { let s = 0; for (let i = 0; i < P.length - 1; i++) { const [c, r] = P[i], [nc, nr] = P[i + 1]; s += GAM ** i * HINT.pot(c, r, nc, nr, G.term(nc, nr) !== 0); } return s; }
const ROUTE = [[0, 5], [0, 4], [0, 3], [0, 2], [0, 1], [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0]];
const DANCE = (() => { const P = ROUTE.slice(0, 12); for (let k = 0; k < 1000; k++) P.push(k % 2 ? [6, 0] : [5, 0]); return P; })();
const f2 = v => L.num(v, 2), f3 = v => L.num(v, 3), f1 = v => (Math.round(v * 10) / 10).toFixed(1), f0 = v => String(Math.round(v));
const sgn = (v, d = 2) => (v > 5e-4 ? '+' : v < -5e-4 ? '−' : '') + L.num(Math.abs(v), d);

// ───── drawing helpers
function dim(a = .62) { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, 285); ctx.restore(); }
function distNums(a, only) { if (a <= 0) return; for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) { if (G.isRock(c, r) || G.term(c, r)) continue; if (only && !only(c, r)) continue; txt(String(dist(c, r)), OX + c * CS + CS - 6, OY + r * CS + 10, { s: 12, al: 'right', c: C.soft, a }); } }
function potNums(a) { if (a <= 0) return; for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) { if (G.isRock(c, r) || G.term(c, r)) continue; const v = PHI(c, r); box(OX + c * CS + 3, OY + r * CS + 3, CS - 6, CS - 6, C.gold, .08 + .5 * v * a); txt(f2(v), cx(c), cy(r) + 12, { s: 12, c: C.goldInk, a }); } }
// a walker along a list of cells, dur seconds per step
function along(P, t, dur) { t = Math.max(0, t); const k = Math.min(P.length - 1, Math.floor(t / dur)), u = ease((t % dur) / dur * 1.3), a = P[k], b = P[Math.min(P.length - 1, k + 1)];
  return { x: lerp(cx(a[0]), cx(b[0]), k >= P.length - 1 ? 0 : u), y: lerp(cy(a[1]), cy(b[1]), k >= P.length - 1 ? 0 : u), k, face: b[0] - a[0], done: k >= P.length - 1 }; }
const COL = { none: C.grey, hot: C.teal, closer: C.rose, pot: '#c99a3a', neg: C.blue };
const NAME = { none: 'no hints', hot: 'hot and cold', closer: '+0.3 per step closer', pot: 'potential hint', neg: 'potential −0.1 × distance' };
// chart: steps per walk (average of 20 runs), walks 1..30; y capped at 300
function chart(modes, t, o = {}) {
  const x0 = 62, x1 = 368, y0 = 388, y1 = 306, X = i => x0 + i * (x1 - x0) / (NW - 1), YM = o.ymax ?? 300, Y = v => y0 - Math.min(v, YM) / YM * (y1 - y0) * -1;
  const Yv = v => y0 - Math.min(v, YM) / YM * (y0 - y1);
  rline(x0 - 4, y0, x1 + 4, y0, 950, { c: C.line, w: 1 }); rline(x0 - 4, y0, x0 - 4, y1 - 4, 951, { c: C.line, w: 1 });
  (YM > 300 ? [0, 200, 400, YM] : [0, 100, 200, 300]).filter((v, i, a) => a.indexOf(v) === i).forEach(v => txt(v === 300 && YM === 300 ? '300+' : String(v), x0 - 9, Yv(v), { s: 12, c: C.soft, al: 'right' }));
  txt('steps per walk, average of 20 runs', x0, y1 - 12, { s: 13, c: C.soft, al: 'left' });
  txt('walk 1', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NW}`, x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
  modes.forEach((m, j) => { const A = avg(m).cur, n = Math.floor(clamp((t - j * (o.stagger ?? 0)) / (o.dur ?? 3)) * NW); for (let i = 0; i < n; i++) { if (i) rline(X(i - 1), Yv(A[i - 1]), X(i), Yv(A[i]), 960 + i + j * 40, { c: COL[m], w: 1.8, once: true, j: .3 }); dot(X(i), Yv(A[i]), 2.2, COL[m]); } });
  return { X, Y: Yv };
}
function legend(modes, x, y, a = 1) { box(x - 8, y - 12, 192, modes.length * 16 + 8, C.card, .92 * a); modes.forEach((m, i) => { box(x, y + i * 16 - 3, 14, 6, COL[m], a); txt(NAME[m], x + 20, y + i * 16, { s: 13, al: 'left', a, c: C.ink }); }); }

// ───── story moments
const S0 = show('none'), W1 = S0.E[0].tr, W1N = W1.length;
const NONE = () => avg('none'), HOT = () => avg('hot'), CLO = () => avg('closer'), POT = () => avg('pot');
const SC = show('closer'), DP = SC.F.P;                                    // the greedy walk of the seed-1 '+0.3 closer' run (dances)
const DANCE_AT = DP.findIndex((p, i) => i > 1 && p[0] === DP[i - 2][0] && p[1] === DP[i - 2][1]) - 2;
const SP = show('pot'), PW1 = SP.E[0].tr.map(s => [s.c, s.r]).concat([[7, 0]]);
const EAT = 1 + .3, DANCEV = () => vi('closer').V[key(6, 0)];

const STEPS = [
 { h: 'A bigger maze', p: ['Pip has a **bigger maze** for this reel, so the cheese is harder to find: 8 cells across, 6 up, five rocks, the cheese top right, the start bottom left. No cat, to keep things simple.', 'The rules are as before: +1 on the step onto the cheese, which ends the walk; every other step pays 0; γ (gamma) = 0.9. A walk that has not found the cheese after 1000 steps is called off.'],
   shot: [3, 9],
   draw(t) { maze({ t, draw: clamp(t / 1.5) }); pip(cx(0), cy(5), 1, t); txt('+1', cx(7) - 12, cy(0) - 12, { s: 14, c: '#b07d12', a: clamp(t - 2) }); txt('start', cx(0), cy(5) - 14, { s: 12, c: C.soft, a: clamp(t - 2) });
     card(['8 × 6 cells, no cat', '+1 on the step onto the cheese, walk over', { s: 'every other step 0;  γ = 0.9', c: C.soft }], t - 3, { gap: 1.1 }); } },
 { h: 'The learner', p: ['Pip learns with **Q-learning**, as in the Q-learning reel: four numbers per cell, one per move, all starting at 0. It takes the move with the biggest number, ties broken at random, and one move in five is random.', 'After each move, the move’s number is nudged a fifth of the way, α (alpha) = 0.2, toward the reward plus 0.9 times the best number where Pip landed.'],
   draw(t) { maze({ t }); pip(cx(0), cy(5), 1, t);
     card(['Q-learning: a number per cell and move', 'act: biggest number;  1 move in 5 random', 'target = reward + 0.9 × best number there', { s: 'new = old + 0.2 × (target − old)', c: C.soft }], t - .5, { gap: 1.1 }); } },
 { h: 'Searching in the dark', p: [`Every number starts at 0, so every choice is a coin toss. Pip’s first walk took **${W1N} steps**. Each update said “0 stays 0”.`, 'Only the last step, onto the cheese, changed anything: one number out of 168 rose to 0.2. All that wandering taught Pip nothing. A reward that almost never shows up is called a **sparse reward**.'],
   shot: [6, 13],
   draw(t) { const dur = .045, tEnd = W1N * dur, k = Math.min(W1N, Math.floor(t / dur)), pts = W1.slice(0, k).map(s => [s.c, s.r]);
     maze({ t }); trail(pts, .55); const s = W1[Math.min(W1N - 1, k)], p = k >= W1N ? [7, 0] : [s.c, s.r]; pip(cx(p[0]), cy(p[1]), 1, t);
     caption(k < W1N ? `walk 1, step ${k} (sped up)` : `walk 1: ${W1N} steps, then the cheese`);
     if (t > tEnd + .5) { bubble('+1', cx(7), cy(0) - 14, t - tEnd - .5); card([`steps: ${W1N}, rewards: ${W1N - 1} zeros, then +1`, 'numbers changed: 1 of 168', { s: 'the cheese is far, and the news comes late', c: C.soft }], t - tEnd - 1.5, { gap: 1.2 }); } } },
 { h: 'How slow is slow?', get p() { return ['To see the speed, run 20 Pips, each with its own luck, for 30 walks each. The chart’s line goes across from walk 1 to walk 30; its height is how many steps that walk took, averaged over the 20 Pips. Lower is better: the shortest way takes 12 steps.', `Without hints, the first 10 walks average **${f0(NONE().first10)} steps**. By walk 30 it is ${f0(NONE().cur[NW - 1])}.`]; },
   shot: [4, 10],
   draw(t) { maze({ t }); dim(.5); pip(cx(0), cy(5), 1, t); chart(['none'], t - 1.5, { dur: 4 }); legend(['none'], 200, 236, clamp(t - 5.5)); } },
 { h: 'Give hints: hot and cold', p: ['We, the designers, know how far each cell is from the cheese: columns plus rows. Here no rock makes the real way longer. Pip does not know this.', 'So we add a hint to the reward, like the game **hot and cold**: +0.1 for a step that brings Pip one closer, −0.1 for a step away, 0 for a bump. Adding extra rewards like this is called **reward shaping**.'],
   shot: [4, 11],
   draw(t) { maze({ t }); distNums(clamp(t - .8)); const P = [[2, 4], [2, 3], [3, 4]]; pip(cx(2), cy(4), 1, t);
     if (t > 3) { alpha(clamp(t - 3), () => { arrow(cx(2), cy(4) - 8, cx(2), cy(3) + 12, 700, { c: C.teal, w: 2 }); arrow(cx(2), cy(4) + 8, cx(2), cy(5) - 10, 701, { c: C.rose, w: 2 }); });
       txt('+0.1', cx(2) - 8, cy(3) + 10, { s: 15, c: '#3f8582', a: clamp(t - 3.4), al: 'right' }); txt('−0.1', cx(2) - 8, cy(5) + 8, { s: 15, c: C.roseInk, a: clamp(t - 3.8), al: 'right' }); }
     caption('small grey number = steps to the cheese', clamp(t - 1.2));
     card(['hint: +0.1 for a step closer', '−0.1 for a step away, 0 for a bump', { s: 'Pip learns from reward + hint', c: C.soft }], t - 5, { gap: 1.1, y: 296 }); } },
 { h: 'Much faster', get p() { return [`The teal line is the same 20 Pips with hot-and-cold hints. Their first 10 walks average **${f0(HOT().first10)} steps** instead of ${f0(NONE().first10)}.`, 'Why? Every step now carries a little news, so the numbers start pointing toward the cheese long before Pip ever reaches it.']; },
   shot: [4, 9],
   draw(t) { maze({ t }); dim(.5); pip(cx(0), cy(5), 1, t); chart(['none', 'hot'], t + 6, { dur: 4, stagger: 6.5 }); legend(['none', 'hot'], 200, 236, 1); } },
 { h: 'A tempting shortcut', p: ['A designer thinks: why punish steps away at all? Just reward progress, and make it stronger: **+0.3 for every step closer**, nothing otherwise.', 'It sounds like the same hint. Watch what Pip learns.'],
   draw(t) { maze({ t }); distNums(clamp(t - .5)); pip(cx(2), cy(4), 1, t);
     if (t > 1.5) { alpha(clamp(t - 1.5), () => { arrow(cx(2), cy(4) - 8, cx(2), cy(3) + 12, 700, { c: C.teal, w: 2 }); arrow(cx(2), cy(4) + 8, cx(2), cy(5) - 10, 701, { c: C.line, w: 2 }); });
       txt('+0.3', cx(2) - 8, cy(3) + 10, { s: 15, c: '#3f8582', a: clamp(t - 1.8), al: 'right' }); txt('0', cx(2) - 8, cy(5) + 8, { s: 15, c: C.soft, a: clamp(t - 2.2), al: 'right' }); }
     card(['hint: +0.3 for a step closer', { s: 'a step away or a bump: 0', c: C.soft }], t - 3, { gap: 1.1 }); } },
 { h: 'Pip dances', get p() { return [`After 30 walks, Pip follows its biggest numbers from the start, with no random moves. It runs toward the cheese, then steps back and forth **beside it**. Without random moves it would never eat.`, `${20 - CLO().ok} of the 20 Pips end up dancing: ${CLO().beside} right beside the cheese, the rest farther away, a sign their learning is unfinished. Even early, this hint slows learning: ${f0(CLO().first10)} steps per walk over the first 10 walks, against ${f0(NONE().first10)} with no hints.`]; },
   shot: [6, 12],
   draw(t) { const sp = show('closer'); maze({ t }); const m = along(DP, t - .5, .32); trail(DP.slice(0, m.k + 1), .6); pip(m.x, m.y, m.face, t);
     if (m.k > DANCE_AT + 1) txt('back and forth…', cx(5) + 22, cy(1) - 4, { s: 15, c: C.roseInk, a: clamp((t - .5 - (DANCE_AT + 2) * .32)) });
     card([`${20 - CLO().ok} of 20 Pips dance, ${CLO().beside} beside the cheese`, `last 10 walks: ${f0(CLO().last10)} steps on average`, { s: 'hot and cold, last 10 walks: ' + f0(HOT().last10) + ' steps', c: C.soft }], t - 7, { gap: 1.1 }); } },
 { h: 'Why dancing pays', p: [`Beside the cheese, Pip has two plans. Eat now: +1, plus 0.3 for the step closer, then nothing: **${f2(EAT)}**. Or dance: a free step away, +0.3 back, again and again. Counted with the 0.9 discount, the dance is worth **${f2(DANCEV())}**.`, 'Beside the cheese, the hint did not just speed things up. It changed **what is best**. Pip is not broken; it found the best way to collect our hints. This is called **farming** a reward.'],
   shot: [5, 12],
   draw(t) { maze({ t }); const u = (t * .7) % 2, x = u < 1 ? lerp(cx(6), cx(5), ease(u)) : lerp(cx(5), cx(6), ease(u - 1)); pip(x, cy(0), u < 1 ? -1 : 1, t);
     if (t > 1) { alpha(clamp(t - 1), () => arrow(cx(6) + 8, cy(0) + 14, cx(7) - 12, cy(0) + 14, 720, { c: C.goldInk, w: 1.6 })); txt('eat', cx(7), cy(0) + 30, { s: 13, c: C.goldInk, a: clamp(t - 1) }); }
     card([`eat now: 1 + 0.3 = ${f2(EAT)}`, `dance: 0 + 0.9 × 0.3 + 0 + 0.9³ × 0.3 + … = ${f2(DANCEV())}`, { s: '(step k counts 0.9 to the power k)', c: C.soft }, { s: `dancing wins: the hint changed what is best`, c: C.roseInk }], t - 2.5, { gap: 1.4 }); } },
 { h: 'Why hot and cold was fine', p: ['With hot and cold, a step away **takes back** what the step closer gave: +0.1, then −0.1. Out and back, the hints sum to nothing; with the discount, almost nothing, +0.01, far too little to beat the cheese here.', 'Call minus a tenth of the distance a cell’s **potential**. Each hint is the new cell’s potential minus the old one’s, so along any walk the middle terms cancel: only start and end count.'],
   shot: [4, 12],
   draw(t) { maze({ t }); [[3, 1], [4, 1]].forEach(([c, r]) => txt(String(dist(c, r)), cx(c), cy(r), { s: 20, c: C.soft, a: clamp(t - .5) })); txt('steps to the cheese', cx(3) + CS / 2, cy(1) - 30, { s: 13, c: C.soft, a: clamp(t - .5) });
     if (t > 1) { alpha(clamp(t - 1), () => { arrow(cx(3) + 6, cy(1) + 12, cx(4) - 6, cy(1) + 12, 730, { c: C.teal, w: 2 }); arrow(cx(4) - 6, cy(1) + 20, cx(3) + 6, cy(1) + 20, 731, { c: C.rose, w: 2 }); });
       txt('closer +0.1', cx(4) + 26, cy(1) + 12, { s: 14, c: '#3f8582', a: clamp(t - 1.4), al: 'left' }); txt('back −0.1', cx(4) + 26, cy(1) + 28, { s: 14, c: C.roseInk, a: clamp(t - 2), al: 'left' }); }
     card(['potential = −0.1 × distance', 'hint = potential(new) − potential(old)', { s: 'sum over a walk = potential(end) − potential(start)', c: C.soft }], t - 4, { gap: 1.3 }); } },
 { h: 'The safe recipe', p: ['In 1999, Andrew Ng, Daishi Harada and Stuart Russell made this exact. Give each cell any potential, 0 once the walk is over, and pay **hint = 0.9 × potential of the new cell − potential of the old one**.', 'Counted with the discount, any whole walk’s hints from a cell then add up to minus its potential, loops included. So every move’s number in that cell shifts by the same amount, and the ranking of moves is unchanged: **the best plan stays best**. This is **potential-based shaping**.'],
   shot: [4, 13],
   draw(t) { maze({ t }); dim(.8);
     txt('hint  =  γ × Φ(new cell)  −  Φ(old cell)', W / 2, 60, { s: 21, a: clamp(t * 2) });
     [['Φ (phi): the potential, any number per cell', 1.2], ['Φ = 0 once the walk is over', 2], ['γ = 0.9, the same discount as the reward', 2.8], ['any whole walk from a cell:', 4], ['hints add up to −Φ(that cell)', 4.6]].forEach(([s, d], i) => txt(s, 50, 104 + i * 30 + (i > 2 ? 10 : 0), { s: 17, al: 'left', a: clamp((t - d) * 2), c: i === 4 ? C.goldInk : C.ink }));
     card(['same total for every plan', '→ which move is best cannot change', { s: 'hot and cold lacks the 0.9: fine here, no guarantee', c: C.soft }], t - 6, { gap: 1.2 }); } },
 { h: 'Choosing a potential', p: ['Any potential is safe, but a good one also helps. A natural choice is a rough guess of how much reward is still to come. Ours: **1 minus the distance divided by 12**. It is 0 at the start, 12 steps away, and 0.92 beside the cheese.', 'On the cheese the walk is over, so its potential is 0. The step onto the cheese pays +1 of reward and a hint of −0.92: the guess is cashed in.'],
   shot: [4, 11],
   draw(t) { maze({ t }); potNums(clamp(t - .6)); pip(cx(0), cy(5), 1, t);
     card(['potential Φ = 1 − distance ÷ 12', 'start: 0;   beside the cheese: 0.92', { s: 'onto the cheese: +1 reward, hint 0 − 0.92', c: C.soft }], t - 3, { gap: 1.2 }); } },
 { h: 'What the hints feel like', p: [`From a cell 8 steps away: a step closer pays ${sgn(HINT.pot(2, 3, 2, 2, false), 3)}, a bump ${sgn(HINT.pot(2, 3, 2, 3, false), 3)}, a step away ${sgn(HINT.pot(2, 3, 2, 4, false), 3)}.`, 'So getting closer pays, wasting a step costs a little, going back costs more. These small costs are not a step cost that changes the plan: the cheese step pays them back, so every plan’s total is the same. What matters is the form of the recipe, with the 0.9 built in.'],
   shot: [4, 10],
   draw(t) { maze({ t }); pip(cx(2), cy(3), 1, t);
     const H = [[0, HINT.pot(2, 3, 2, 2, false), C.teal], [2, HINT.pot(2, 3, 2, 4, false), C.rose]];
     H.forEach(([m, v, col], i) => { const a = clamp(t - 1 - i * .8), dy = m === 0 ? -1 : 1; if (a <= 0) return; alpha(a, () => arrow(cx(2), cy(3) + dy * 10, cx(2), cy(3) + dy * 32, 740 + i, { c: col, w: 2 })); box(cx(2) + 12, cy(3) + dy * 30 - 9, 52, 18, C.card, .9 * a); txt(sgn(v, 3), cx(2) + 14, cy(3) + dy * 30, { s: 15, al: 'left', a, c: col === C.teal ? '#3f8582' : C.roseInk }); });
     if (t > 2.6) { const a = clamp(t - 2.6); box(cx(2) - 72, cy(3) - 9, 52, 18, C.card, .9 * a); txt('bump ' + sgn(HINT.pot(2, 3, 2, 3, false), 3), cx(2) - 20, cy(3), { s: 14, al: 'right', a, c: C.soft }); }
     card([`closer: 0.9 × ${f3(PHI(2, 2))} − ${f3(PHI(2, 3))} = ${sgn(HINT.pot(2, 3, 2, 2, false), 3)}`, `bump:   0.9 × ${f3(PHI(2, 3))} − ${f3(PHI(2, 3))} = ${sgn(HINT.pot(2, 3, 2, 3, false), 3)}`, `away:   0.9 × ${f3(PHI(2, 4))} − ${f3(PHI(2, 3))} = ${sgn(HINT.pot(2, 3, 2, 4, false), 3)}`], t - 4, { gap: 1.2, fs: 16 }); } },
 { h: 'Same total, every time', p: [`Check it from the start, where the potential is 0. Three walks: the shortest route, Pip’s first wandering walk with these hints, ${PW1.length - 1} steps, and a walk that dances beside the cheese for 1000 steps.`, `Their hints, counted with the discount, add up to ${f2(hintTotal(ROUTE))}, ${f2(hintTotal(PW1))} and ${f2(hintTotal(DANCE))}. Dancing earns nothing extra. Only the reward differs: the route’s cheese is worth ${f2(GAM ** 11)} from the start, the dance 0.`],
   shot: [5, 12],
   draw(t) { maze({ t });
     const R1 = ROUTE, a1 = clamp((t - .5) / 2), a2 = clamp((t - 3) / 2.5), a3 = clamp((t - 6) / 2);
     trail(PW1.slice(0, Math.floor(a2 * PW1.length)), .55);
     for (let i = 0; i < Math.floor(a1 * (R1.length - 1)); i++) rline(cx(R1[i][0]) - 4, cy(R1[i][1]) - 4, cx(R1[i + 1][0]) - 4, cy(R1[i + 1][1]) - 4, 760 + i, { c: '#c99a3a', w: 2.6, once: true });
     if (a3 > 0) { const u = (t * 1.4) % 2, x = u < 1 ? lerp(cx(6), cx(5), ease(u)) : lerp(cx(5), cx(6), ease(u - 1)); pip(x, cy(0), u < 1 ? -1 : 1, t); } else pip(cx(0), cy(5), 1, t);
     card([{ s: `shortest route, 12 steps:   ${f2(hintTotal(ROUTE))}`, c: C.goldInk }, { s: `first walk, ${PW1.length - 1} steps:   ${f2(hintTotal(PW1))}`, c: '#3f8582' }, { s: `dance for 1000 steps:   ${f2(hintTotal(DANCE))}`, c: C.roseInk }, { s: `reward on top: route ${f2(GAM ** 11)}, dance 0`, c: C.soft }], t - 2, { gap: 2.4, lh: 20, y: 289 });
     caption('hint totals from the start, counted with the 0.9 discount', clamp(t - 9)); } },
 { h: 'Fast and safe', get p() { return [`Gold: 20 Pips with the potential hint. Their first 10 walks average **${f0(POT().first10)} steps**, almost as fast as hot and cold, and all ${POT().ok} end on a 12-step route.`, `The best numbers Pip heads for are the usual ones minus the cell’s potential. At the start, where the potential is 0, that is ${f2(vi('pot').V[key(0, 5)])}, the same as without hints. The teal hot-and-cold line hides under the gold one.`]; },
   shot: [4, 11],
   draw(t) { maze({ t }); dim(.5); pip(cx(0), cy(5), 1, t); chart(['none', 'hot', 'pot'], t + 12, { dur: 4, stagger: 6.5 }); legend(['none', 'hot', 'pot'], 200, 228, 1); } },
 { h: 'Safe is not the same as fast', get p() { return [`The guarantee is about what is best, not speed. Put hot and cold’s potential, minus a tenth of the distance, into the exact recipe: now even a bump earns a small positive hint, so moves Pip has tried look better than untried ones.`, `The first 10 walks average **${f0(avg('neg').first10)} steps**: slower than no hints at all. This potential differs from ours mostly by a constant. A constant cannot change the best plan, but because Pip’s numbers all start at 0, it changes how fast Pip learns.`]; },
   shot: [4, 11],
   draw(t) { maze({ t }); dim(.5); pip(cx(0), cy(5), 1, t); chart(['none', 'pot', 'neg'], t + 6, { dur: 4, stagger: 6.5, ymax: Math.ceil(Math.max(...avg('neg').cur) / 100) * 100 }); legend(['none', 'pot', 'neg'], 200, 228, 1); } },
 { h: 'Your turn', ctl: { label: 'hint', values: ['none', 'hot', 'closer', 'pot'], def: 2, fmt: v => ({ none: 'none', hot: 'hot/cold', closer: '+0.3', pot: 'potential' })[v] },
   p: ['Pick one of four hints: none, hot and cold, +0.3 per step closer, or the potential hint. The picture shows one Pip after 30 walks, following its biggest numbers with no random moves. The card gives averages over all 20 Pips.', 'Which hints speed Pip up, and which one teaches it the wrong thing?'],
   shot: [3, 10],
   draw(t, st) { const mode = ['none', 'hot', 'closer', 'pot'][st.ci ?? 2], sh = show(mode), A = avg(mode), t0 = t - (st.changed ?? 0);
     maze({ t }); const P = sh.F.P, m = along(P, Math.max(0, t0 - .6), .3); trail(P.slice(0, m.k + 1), .6); pip(m.x, m.y, m.face, t);
     caption(m.done ? (sh.F.ok ? `reached the cheese in ${P.length - 1} steps` : 'still dancing after 40 steps') : `following its numbers, step ${m.k}`);
     card([NAME[mode], `first 10 walks: ${f0(A.first10)} steps on average`, `last 10 walks: ${f0(A.last10)} steps`, { s: `Pips that reach the cheese: ${A.ok} of 20`, c: A.ok < 20 ? C.roseInk : C.soft }], 10, { lh: 21, fs: 16, y: 292 }); } },
 { h: 'In the real world', p: ['The word comes from animal training: B. F. Skinner taught pigeons by rewarding small steps toward a trick. In RL, Jette Randløv and Preben Alstrøm in 1998 rewarded a simulated bicycle for progress toward a goal, with no penalty for riding away. It learned to ride **in circles**, collecting reward on every loop, just like Pip’s dance.', 'That story helped motivate the 1999 recipe. Hand-made hints are still common in robot learning.'],
   shot: [3, 10],
   draw(t) { maze({ t }); dim(.4); const u = t * 1.2; ctx.save(); ctx.setLineDash([4, 5]); ctx.strokeStyle = C.line; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.ellipse(cx(3) + 22, cy(4) + 22, 40, 20, 0, 0, 7); ctx.stroke(); ctx.restore(); pip(cx(3) + 22 + Math.cos(u) * 40, cy(4) + 22 + Math.sin(u) * 20, Math.sin(u) > 0 ? -1 : 1, t);
     card(['1940s  Skinner: shaping pigeons’ behaviour', '1998  Randløv and Alstrøm: the circling bicycle', '1999  Ng, Harada, Russell: potential-based shaping'], t - .5, { gap: 1.1, fs: 16 }); } },
 { h: 'Recap', p: ['**Sparse rewards** make learning slow: most steps teach nothing. **Shaping** adds hints so every step carries news.', 'Careless hints change what is best and get **farmed**. Hints of the form γ × potential(new) − potential(old) can never change the best plan; a good potential also makes learning fast.'],
   draw(t) { maze({ t }); const sp = show('pot'); const m = along(sp.F.P, t % 6, .3); pip(m.x, m.y, m.face, t);
     card(['sparse reward: slow, the news comes late', 'hints make every step informative', 'careless hints get farmed', 'safe: hint = γ Φ(new) − Φ(old)'], t - .3, { gap: 1 }); } },
];
RLStory.register('shaping', { title: 'Reward Shaping', track: 'Rewards', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { runQ, W1N, avg, show, vi, hintTotal, ROUTE, DANCE, PW1, DP, DANCE_AT, HINT, PHI, follow } });
})();
