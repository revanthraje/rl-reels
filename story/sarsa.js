// SARSA as a story: in a room with a row of cats, SARSA learns the safe route (its own random moves count), Q-learning the risky shortest one.
// All numbers come from the seeded runs below (story seed 28; comparisons over seeds 1-100).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, txt, arrow, card, bubble, caption, dot, box, num } = L;
let ctx;
// ───── the cat row room: 6 x 4, no rocks, start bottom-left, cheese bottom-right, cats on the bottom row in between
const COLS = 6, ROWS = 4, CHEESE = [5, 3], CATS = [[1, 3], [2, 3], [3, 3], [4, 3]], START = [0, 3], GAM = .9, ALPHA = .2, NEP = 500, MV = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const G = L.gridWorld({ cols: COLS, rows: ROWS, cheese: CHEESE, cat: null, rocks: [], cs: 60, ox: 20, oy: 16, start: START });
const { cx, cy, trail, pip } = G, OX = G.ox, OY = G.oy, CS = G.cs, N = ['up', 'right', 'down', 'left'];
const key = (c, r) => r * COLS + c, qk = (c, r, m) => key(c, r) * 4 + m;
const term = (c, r) => (c === CHEESE[0] && r === CHEESE[1]) ? 1 : CATS.some(([x, y]) => x === c && y === r) ? -1 : 0;
const step = (c, r, m) => { const nc = c + MV[m][0], nr = r + MV[m][1]; return (nc < 0 || nr < 0 || nc >= COLS || nr >= ROWS) ? [c, r] : [nc, nr]; };
// algo 'q': pick move, step, update with the max, repeat.  algo 's' (SARSA): pick the next move first, update with ITS number, then make it.
function learn(algo, eps, seed, rec) {
  const R = rng(seed), Q = new Float64Array(COLS * ROWS * 4), E = [];
  const best = (c, r) => Math.max(Q[qk(c, r, 0)], Q[qk(c, r, 1)], Q[qk(c, r, 2)], Q[qk(c, r, 3)]);
  const act = (c, r) => { if (R() < eps) return [Math.floor(R() * 4), true]; let b = -1e9, ms = []; for (let m = 0; m < 4; m++) { const q = Q[qk(c, r, m)]; if (q > b + 1e-9) { b = q; ms = [m]; } else if (q > b - 1e-9) ms.push(m); } return [ms[Math.floor(R() * ms.length)], false]; };
  for (let e = 0; e < NEP; e++) {
    let [c, r] = START, A = algo === 's' ? act(c, r) : null, k = 0, rew = 0; const tr = rec ? [] : null;
    for (k = 0; k < 3000; k++) {
      if (algo === 'q') A = act(c, r);
      const [m, rnd] = A, [nc, nr] = step(c, r, m); rew = term(nc, nr); const done = rew !== 0, bestNext = done ? 0 : best(nc, nr); let next = 0, A2 = null;
      if (!done) { if (algo === 's') { A2 = act(nc, nr); next = Q[qk(nc, nr, A2[0])]; } else next = bestNext; }
      const old = Q[qk(c, r, m)], target = rew + GAM * next, nw = old + ALPHA * (target - old); Q[qk(c, r, m)] = nw;
      if (rec) tr.push({ c, r, m, rnd, nc, nr, rew, old, next, target, nw, bestNext, m2: A2 && A2[0], rnd2: A2 && A2[1] });
      c = nc; r = nr; A = A2; if (done) break;
    }
    E.push({ tr, end: rew, ret: rew * GAM ** k, len: k + 1, Q: rec ? Array.from(Q) : null });
  }
  return { E, Q: Array.from(Q) };
}
const argQ = (Q, c, r) => { let b = -1e9, bm = 0; for (let m = 0; m < 4; m++) if (Q[qk(c, r, m)] > b + 1e-12) { b = Q[qk(c, r, m)]; bm = m; } return bm; };
const maxQ = (Q, c, r) => Math.max(Q[qk(c, r, 0)], Q[qk(c, r, 1)], Q[qk(c, r, 2)], Q[qk(c, r, 3)]);
function route(Q) { let [c, r] = START; const P = [[c, r]], seen = new Set([key(c, r)]); for (let k = 0; k < 30 && !term(c, r); k++) { [c, r] = step(c, r, argQ(Q, c, r)); if (seen.has(key(c, r))) return P; seen.add(key(c, r)); P.push([c, r]); } return P; }
const rkey = P => P.map(p => p.join(',')).join(' ');
// exact expected return from the start for "follow Q's arrows, but a random move with probability eps" (repeated sweeps)
function evalPol(Q, eps) { let V = new Float64Array(COLS * ROWS); for (let it = 0; it < 600; it++) { const Nw = new Float64Array(V.length); for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) { if (term(c, r)) continue; const g = argQ(Q, c, r); let v = 0;
  for (let m = 0; m < 4; m++) { const p = (m === g ? 1 - eps : 0) + eps / 4, [nc, nr] = step(c, r, m), rw = term(nc, nr); v += p * (rw + (rw ? 0 : GAM * V[key(nc, nr)])); } Nw[key(c, r)] = v; } V = Nw; } return V[key(...START)]; }
// ───── the story run (seed 28, eps 0.2)
const SEED = 28, EPS = .2, RQ = learn('q', EPS, SEED, true), RS = learn('s', EPS, SEED, true), FQ = RQ.Q, FS = RS.Q, PQ = route(FQ), PS = route(FS);
const EDGE = rkey(PQ), SAFE = rkey(PS);
// the concrete SARSA moment: lands on the cat row's top neighbour, next move picked = a random 'down'
const MOM = (() => { for (let e = 0; e < 60; e++) { const tr = RS.E[e].tr; for (let i = 0; i < tr.length; i++) { const t = tr[i]; if (!t.rew && t.nr === 2 && t.nc >= 1 && t.nc <= 4 && t.rnd2 && t.m2 === 2 && t.next < -.1 && t.bestNext > .1) return { e, i, t, n: tr[i + 1] }; } } return null; })();
const MT = MOM.t, MN = MOM.n, QTGT = MT.rew + GAM * MT.bestNext, QNEW = MT.old + ALPHA * (QTGT - MT.old);
function QatS(e, i) { const Q = e > 0 ? RS.E[e - 1].Q.slice() : new Array(96).fill(0); for (let k = 0; k < i; k++) { const t = RS.E[e].tr[k]; Q[qk(t.c, t.r, t.m)] = t.nw; } return Q; }
const VQ2 = evalPol(FQ, EPS), VS2 = evalPol(FS, EPS), VQ0 = evalPol(FQ, 0), VS0 = evalPol(FS, 0);
// ───── comparisons over seeds 1..100 (memoised per eps)
const NSEED = 100, NB = 20, BW = NEP / NB, aggs = {};
function agg(eps) { if (aggs[eps]) return aggs[eps]; const o = { awayS: 0, bq: new Array(NB).fill(0), bs: new Array(NB).fill(0), rq: 0, rs: 0, cq: 0, cs: 0, nq: {}, ns: {} };
  for (let s = 1; s <= NSEED; s++) { const q = learn('q', eps, s), sa = learn('s', eps, s); if (!route(sa.Q).some(([c, r]) => r === 2 && c >= 1 && c <= 4)) o.awayS++; o.nq[rkey(route(q.Q))] = (o.nq[rkey(route(q.Q))] || 0) + 1; o.ns[rkey(route(sa.Q))] = (o.ns[rkey(route(sa.Q))] || 0) + 1;
    q.E.forEach((e, i) => { o.rq += e.ret; o.cq += e.end < 0; o.bq[Math.floor(i / BW)] += e.ret / BW / NSEED; }); sa.E.forEach((e, i) => { o.rs += e.ret; o.cs += e.end < 0; o.bs[Math.floor(i / BW)] += e.ret / BW / NSEED; }); }
  const T = NSEED * NEP; o.rq /= T; o.rs /= T; o.cq /= T; o.cs /= T;
  const top = m => Object.entries(m).sort((a, b) => b[1] - a[1])[0]; o.topQ = top(o.nq); o.topS = top(o.ns); o.edgeQ = o.nq[EDGE] || 0; o.edgeS = o.ns[EDGE] || 0; o.safeS = o.ns[SAFE] || 0;
  return (aggs[eps] = o); }
const EPSV = [0, .05, .1, .2, .3];
const sgn = x => x.replace('-', '−'), f2 = v => sgn(num(v)), pct = v => Math.round(v * 100);

// ───── drawing
function room(o = {}) { G.maze({ t: o.t || 0, draw: o.draw, pop: o.pop }); const a = o.cats ?? 1; if (a > 0) CATS.forEach(([c, r], i) => { ctx.save(); ctx.globalAlpha = a; L.cat(cx(c), cy(r) + 4, (o.t || 0) + i * .3, CS / 64); ctx.restore(); }); }
function tri(xm, ym, x0, y0, x1, y1, m, col, a) { const P = [[[x0, y0], [x1, y0]], [[x1, y0], [x1, y1]], [[x1, y1], [x0, y1]], [[x0, y1], [x0, y0]]][m];
  ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(xm, ym); ctx.lineTo(...P[0]); ctx.lineTo(...P[1]); ctx.closePath(); ctx.fill(); ctx.restore(); }
function wedges(Q, c, r, x0, y0, x1, y1, hl, lines = true) { const xm = (x0 + x1) / 2, ym = (y0 + y1) / 2;
  for (let m = 0; m < 4; m++) { const v = Q[qk(c, r, m)]; if (Math.abs(v) >= .004) tri(xm, ym, x0 + 2, y0 + 2, x1 - 2, y1 - 2, m, v > 0 ? C.gold : C.rose, lerp(.14, .62, clamp(Math.abs(v) / .9))); }
  if (hl) tri(xm, ym, x0 + 2, y0 + 2, x1 - 2, y1 - 2, hl[0], hl[1] || C.teal, hl[2] ?? .35);
  if (lines) { ctx.save(); ctx.globalAlpha = .55; rline(x0 + 3, y0 + 3, x1 - 3, y1 - 3, 300 + c * 9 + r, { c: C.line, w: .9, once: true, j: .6 }); rline(x1 - 3, y0 + 3, x0 + 3, y1 - 3, 330 + c * 9 + r, { c: C.line, w: .9, once: true, j: .6 }); ctx.restore(); } }
function qroom(Q, o = {}) { for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) { if (term(c, r)) continue; const x0 = OX + c * CS, y0 = OY + r * CS; wedges(Q, c, r, x0, y0, x0 + CS, y0 + CS, o.hl && o.hl[key(c, r)]); } room(o); }
function qnums(Q, c, r, x0, y0, s, fs, d3) { const xm = x0 + s / 2, ym = y0 + s / 2, P = [[xm, y0 + s * .17], [x0 + s * .8, ym + 1], [xm, y0 + s * .85], [x0 + s * .2, ym + 1]];
  for (let m = 0; m < 4; m++) { const v = Q[qk(c, r, m)]; txt(sgn(d3 ? num(v, 3) : num(v)), P[m][0], P[m][1], { s: fs, c: v < -.004 ? C.roseInk : v > .004 ? C.goldInk : C.soft }); } }
function zoom(Q, c, r, x, y, s, o = {}) { const a = o.a ?? 1; if (a <= 0) return; ctx.save(); ctx.globalAlpha = a; box(x, y, s, s, C.card); wedges(Q, c, r, x, y, x + s, y + s, o.hl); L.rrect(x, y, s, s, 640 + c * 5 + r, { w: 1.4 });
  qnums(Q, c, r, x, y, s, o.d3 ? 14 : 15, o.d3); if (o.label) txt(o.label, x + s / 2, y + s + 11, { s: 13, c: o.lc || C.soft }); ctx.restore(); }
function routeLine(P, col, a = 1, w = 3, off = 0) { if (a <= 0) return; ctx.save(); ctx.globalAlpha = a; for (let i = 1; i < P.length; i++) rline(cx(P[i - 1][0]) + off, cy(P[i - 1][1]) + off, cx(P[i][0]) + off, cy(P[i][1]) + off, 700 + i, { c: col, w, once: true, j: .8 }); ctx.restore(); }
function arrowsQ(Q, a = 1, col = C.goldInk) { if (a <= 0) return; ctx.save(); ctx.globalAlpha = a; for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) { if (term(c, r)) continue; G.arrowIn(c, r, argQ(Q, c, r), 400 + key(c, r), { c: col, w: 1.5 }); } ctx.restore(); }
// small room for side-by-side views
function mini(x0, y0, cs, Q, P, col, label, a = 1) { if (a <= 0) return; ctx.save(); ctx.globalAlpha = a;
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) if (!term(c, r)) wedges(Q, c, r, x0 + c * cs, y0 + r * cs, x0 + (c + 1) * cs, y0 + (r + 1) * cs, null, false);
  for (let k = 0; k <= ROWS; k++) rline(x0, y0 + k * cs, x0 + COLS * cs, y0 + k * cs, 10 + k, { c: k % ROWS ? C.line : C.ink, w: k % ROWS ? .9 : 1.5 });
  for (let k = 0; k <= COLS; k++) rline(x0 + k * cs, y0, x0 + k * cs, y0 + ROWS * cs, 30 + k, { c: k % COLS ? C.line : C.ink, w: k % COLS ? .9 : 1.5 });
  CATS.forEach(([c, r]) => L.cat(x0 + c * cs + cs / 2, y0 + r * cs + cs / 2 + 2, 0, cs / 80)); L.cheese(x0 + 5.5 * cs, y0 + 3.5 * cs, cs / 70);
  for (let i = 1; i < P.length; i++) rline(x0 + (P[i - 1][0] + .5) * cs, y0 + (P[i - 1][1] + .5) * cs, x0 + (P[i][0] + .5) * cs, y0 + (P[i][1] + .5) * cs, 720 + i, { c: col, w: 3, once: true, j: .6 });
  txt(label, x0 + COLS * cs / 2, y0 - 12, { s: 16, c: col }); ctx.restore(); }
function ghost(c, r, s, a, dy) { if (a <= 0) return; box(cx(c) - 36, cy(r) + dy - 9, 72, 18, C.card, .85 * a); txt(s, cx(c), cy(r) + dy, { s: 13, c: '#6d645b', a }); }
function dim(a = .62) { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, 285); ctx.restore(); }
// chart: average return per block of 25 walks, over 100 runs. Q-learning orange, SARSA teal.
const QC = C.orange, SC = '#3f8f8b';
function blockChart(A, n, o = {}) {
  const x0 = 58, x1 = 362, y0 = 392, y1 = 306, lo = -.2, hi = .3, X = i => x0 + (i + .5) * (x1 - x0) / NB, Y = v => lerp(y0, y1, (v - lo) / (hi - lo));
  rline(x0 - 4, y0, x1 + 4, y0, 960, { c: C.line, w: 1 }); rline(x0 - 4, Y(0), x1 + 4, Y(0), 961, { c: '#e4d8c7', w: 1, once: true });
  [[.3, '0.3'], [0, '0'], [-.2, '−0.2']].forEach(([v, s]) => txt(s, x0 - 8, Y(v), { s: 12, c: C.soft, al: 'right' }));
  txt('walk 1', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NEP}`, x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
  txt('average return per walk, 100 runs each', x0 - 30, y1 - 14, { s: 14, al: 'left', c: C.soft });
  [[A.bq, QC, o.q ?? 1], [A.bs, SC, o.s ?? 1]].forEach(([B, col, a]) => { if (a <= 0) return; ctx.save(); ctx.globalAlpha = a; for (let i = 0; i < Math.min(n, NB); i++) { if (i) rline(X(i - 1), Y(B[i - 1]), X(i), Y(B[i]), 970 + i, { c: col, w: 1.6, once: true }); dot(X(i), Y(B[i]), 3, col); } ctx.restore(); });
  if (o.legend) { txt('Q-learning', x1, Y(A.bq[NB - 1]) + 14, { s: 14, al: 'right', c: QC, a: o.q ?? 1 }); txt('SARSA', x1, Y(A.bs[NB - 1]) - 13, { s: 14, al: 'right', c: SC, a: o.s ?? 1 }); }
}
const A2 = () => agg(EPS);

const STEPS = [
 { h: 'A new room', p: ['Pip has a new room: six cells wide, four tall, no rocks. It starts bottom left, the cheese sits bottom right, and between them, along the bottom row, **four cats** sleep.', 'The old rules hold: dry floor; stepping onto the cheese pays +1, onto a cat −1, either ending the walk; γ (gamma) is 0.9; bumps leave Pip in place.'],
   shot: [4, 10],
   draw(t) { room({ t, draw: t / 1.6, pop: { cheese: ease((t - 1.8) * 3) }, cats: clamp((t - 2.4) * 2) }); if (t > 2.8) pip(cx(0), cy(3), 1, t);
     txt('+1', cx(5), cy(3) - 24, { s: 15, c: C.goldInk, a: clamp(t - 3.4) }); CATS.forEach(([c, r], i) => txt('−1', cx(c) - 18, cy(r) - 20, { s: 14, c: C.roseInk, a: clamp(t - 3.6 - i * .15) })); txt('start', cx(0), cy(3) + 24, { s: 13, c: C.soft, a: clamp(t - 3.4) });
     if (t > 4.5) card(['new: 6 × 4 room, a row of 4 cats, no rocks', 'same: dry floor, +1 cheese, −1 cat, both end', { s: 'same: γ = 0.9, α = 0.2, 1 move in 5 random', c: C.soft }], t - 4.5, { gap: 1 }); } },
 { h: 'Two ways across', p: ['The short way runs along the cats: up one, five steps right, down onto the cheese. Seven steps: 0.9 multiplied six times, 0.53. One row higher takes nine steps: 0.43.', 'Without random moves, the short way wins. But Pip moves at random one time in five: on the row above the cats, one move in twenty points down at a cat.'],
   draw(t) { room({ t }); pip(cx(0), cy(3), 1, t); routeLine(PQ, QC, clamp((t - .6) * 1.2)); routeLine(PS, SC, clamp((t - 2.6) * 1.2));
     box(cx(2.5) - 66, cy(2) + 13, 132, 19, C.bg, .9 * clamp(t - 1.6)); txt(`short: ${PQ.length - 1} steps, 0.53`, cx(2.5), cy(2) + 22, { s: 15, c: QC, a: clamp(t - 1.6) });
     box(cx(2.5) - 80, cy(1) - 30, 160, 19, C.bg, .9 * clamp(t - 3.6)); txt(`one row up: ${PS.length - 1} steps, 0.43`, cx(2.5), cy(1) - 20, { s: 15, c: SC, a: clamp(t - 3.6) });
     if (t > 5) card(['random move 1 time in 5, any of 4 directions', 'so “down” by accident: 1 move in 20', { s: 'on the short way, that means a cat', c: C.roseInk }], t - 5, { gap: 1 }); } },
 { h: 'What Q-learning learns', p: [`First, Q-learning, exactly as in the last reel: ${NEP} walks, one move in five random. Its arrows take the short way, right along the cats.`, 'That fits what Q-learning measures. Every target assumes the best next move, so its numbers describe a Pip that never moves at random. In the top rows, which Pip rarely visits, the arrows are rough guesses.'],
   draw(t) { qroom(FQ, { t }); arrowsQ(FQ, clamp(t - 1)); routeLine(PQ, QC, clamp(t - 3), 3, 0); pip(cx(0), cy(3), 1, t);
     if (t > 4) card(['Q-learning, 500 walks, ε = 0.2', 'target = r + γ × biggest number where you land', { s: 'arrows: along the cats', c: QC }], t - 4, { gap: 1 }); } },
 { h: 'SARSA’s one change', p: ['**SARSA** changes one thing. When Pip lands, it first picks its next move, with the same habit: the biggest number, or one time in five a random move. Then it updates using the number of **that** move, the one it will really make, not the biggest one. Then it makes it.', 'Its name lists what one update uses: State, Action, Reward, next State, next Action.'],
   draw(t) { qroom(FS, { t }); pip(cx(0), cy(3), 1, t); dim(.72);
     const items = [['S', 'cell Pip left'], ['A', 'move it made'], ['R', 'reward it got'], ['S′', 'cell where it landed'], ['A′', 'next move, already picked']];
     items.forEach(([a, b], i) => { const u = clamp((t - .6 - i * .8) * 2); txt(a, 90, 60 + i * 38, { s: 28, a: u, c: i === 4 ? SC : C.ink }); txt(b, 130, 62 + i * 38, { s: 18, al: 'left', a: u, c: i === 4 ? SC : C.ink }); });
     if (t > 5) card(['Q-learning: target = r + γ × biggest Q in S′', 'SARSA:       target = r + γ × Q of A′ in S′', { s: 'A′ is picked first, then really made', c: C.soft }], t - 5, { gap: 1.1 }); } },
 { h: 'In symbols', p: ['With a′, said “a prime”, the next move Pip has already picked in s′:', 'Q(s, a) becomes Q(s, a) + α × (r + γ × Q(s′, a′) − Q(s, a)). The only difference from Q-learning is inside the target: Q(s′, a′) instead of max Q(s′, ·). If the walk is over, there is no next move and nothing is added.'],
   draw(t) { qroom(FS, { t }); pip(cx(0), cy(3), 1, t); dim(.75);
     txt('Q(s, a)  ←  Q(s, a) + α ( r + γ Q(s′, a′) − Q(s, a) )', W / 2, 80, { s: 19, a: clamp(t * 2) });
     txt('SARSA', W / 2, 50, { s: 16, c: SC, a: clamp(t * 2) });
     txt('Q(s, a)  ←  Q(s, a) + α ( r + γ max Q(s′, ·) − Q(s, a) )', W / 2, 160, { s: 19, a: clamp(t - 1.5), c: C.soft }); txt('Q-learning', W / 2, 130, { s: 16, c: QC, a: clamp(t - 1.5) });
     if (t > 3) { ctx.save(); ctx.globalAlpha = clamp(t - 3); rline(226, 94, 296, 94, 980, { c: SC, w: 2.2 }); rline(210, 174, 300, 174, 981, { c: QC, w: 2.2 }); ctx.restore(); }
     card(['a′: the move Pip will make next in s′', 'chosen by the same ε-greedy habit', { s: 'so sometimes a random move', c: C.soft }], t - 4, { gap: 1 }); } },
 { h: 'One SARSA update', p: [`Walk ${MOM.e + 1} of SARSA’s own run. Pip steps right onto the cell above a cat, and picks its next move: a random one, **down**, whose number is ${f2(MT.next)}. So the target is 0 + 0.9 × ${f2(MT.next)} = ${f2(MT.target)}, and the old ${f2(MT.old)} drops to ${f2(MT.nw)}.`, `Q-learning would have used the best number there, ${f2(MT.bestNext)}: target ${f2(QTGT)}, new ${f2(QNEW)}.`],
   shot: [4, 12],
   draw(t) { const Qb = QatS(MOM.e, MOM.i), Q = Qb.slice(); if (t > 7) Q[qk(MT.c, MT.r, MT.m)] = lerp(MT.old, MT.nw, ease((t - 7) / 1));
     const mv = G.walkPos(RS.E[MOM.e].tr, MOM.i, MOM.i + 1, clamp(t - .4, 0, 1), 1);
     qroom(Q, { t, hl: { [key(MT.c, MT.r)]: [MT.m, C.teal, .4 * clamp(t - 1.4)] } }); pip(mv.x, mv.y, mv.face || 1, t);
     if (t > 2) { ctx.save(); ctx.globalAlpha = clamp(t - 2); arrow(cx(MT.nc), cy(MT.nr) + 14, cx(MT.nc), cy(MT.nr) + 42, 990, { c: C.rose, w: 2.2 }); ctx.restore(); ghost(MT.nc, MT.nr, 'next: down (random)', clamp(t - 2.4), -22); }
     zoom(Q, MT.nc, MT.nr, 26, 294, 92, { label: 'landed here', a: clamp(t - 2.8), hl: [2, C.rose, .3] });
     card([`SARSA: 0 + 0.9 × ${f2(MT.next)} = ${f2(MT.target)}`, `new ${f2(MT.old)} → ${f2(MT.nw)}`, { s: `Q-learning: 0.9 × ${f2(MT.bestNext)} = ${f2(QTGT)} → ${f2(QNEW)}`, c: QC }], t - 3.6, { x: 132, w: 244, gap: 1.4, fs: 16 }); } },
 { h: 'Then it really goes down', p: [`SARSA commits to the move it picked: Pip steps down onto the cat. Reward −1, walk over, so that number becomes ${f2(MN.nw)}.`, 'Picking a move in advance is what makes SARSA honest about Pip. Its targets contain the random moves Pip really makes, so cells where a random move is dangerous get lower numbers.'],
   draw(t) { const Qb = QatS(MOM.e, MOM.i + 1), Q = Qb.slice(); if (t > 2.6) Q[qk(MN.c, MN.r, MN.m)] = lerp(MN.old, MN.nw, ease((t - 2.6) / .8));
     const mv = G.walkPos(RS.E[MOM.e].tr, MOM.i + 1, MOM.i + 2, clamp(t - .5, 0, 1), 1);
     qroom(Q, { t, hl: { [key(MN.c, MN.r)]: [2, C.rose, .3 * clamp(t - 1.4)] } }); pip(mv.x, mv.y, 1, t); if (t > 1.2) bubble('−1', cx(MN.nc), cy(MN.nr) - 18, t - 1.4);
     zoom(Q, MN.c, MN.r, 26, 294, 92, { label: 'above the cat', a: clamp(t - 1.8), hl: [2, C.rose, .3] });
     card([`down: ${f2(MN.old)} + 0.2 × (${f2(-1 - MN.old)}) = ${f2(MN.nw)}`, 'walk over: no next move', { s: 'this random move was in the last target', c: C.soft }], t - 3, { x: 132, w: 244, gap: 1.3, fs: 16 }); } },
 { h: 'The edge gets a bad name', p: [`After ${NEP} walks, compare one cell on the row above the cats, second from the left. Q-learning says “right” is worth ${f2(FQ[qk(1, 2, 1)])}: the perfect value of the short way. SARSA says only ${f2(FS[qk(1, 2, 1)])}, because walking on along the cats, a random move will sometimes take Pip into a cat.`, `For SARSA, “up”, away from the cats, is the best move here: ${f2(FS[qk(1, 2, 0)])}.`],
   draw(t) { qroom(FS, { t }); pip(cx(0), cy(3), 1, t);
     zoom(FQ, 1, 2, 40, 294, 92, { label: 'Q-learning', lc: QC, a: clamp(t - 1), }); zoom(FS, 1, 2, 268, 294, 92, { label: 'SARSA', lc: SC, a: clamp(t - 2.5), });
     txt('same cell,', W / 2, 318, { s: 16, c: C.soft, a: clamp(t - 3.5) }); txt('after 500 walks', W / 2, 338, { s: 16, c: C.soft, a: clamp(t - 3.5) }); txt('big grid: SARSA', W / 2, 364, { s: 14, c: SC, a: clamp(t - 4) }); } },
 { h: 'Two routes', p: ['From the start, follow each learner’s arrows. Q-learning hugs the cats. SARSA climbs two rows, keeping an empty row between itself and the cats.', `In this run, with Pip still random one move in five, Q-learning’s arrows are worth ${f2(VQ2)} from the start, SARSA’s ${f2(VS2)}. These exact values come from sweeps over the room’s rules, as in the Bellman reel; Pip never sees them.`],
   draw(t) { mini(20, 66, 28, FQ, PQ.slice(0, 1 + Math.floor(clamp(t - .5, 0, 9) * 2)), QC, 'Q-learning'); mini(212, 66, 28, FS, PS.slice(0, 1 + Math.floor(clamp(t - 1, 0, 9) * 2)), SC, 'SARSA');
     txt('start', 20 + 14, 66 + 4 * 28 + 12, { s: 12, c: C.soft }); txt('start', 212 + 14, 66 + 4 * 28 + 12, { s: 12, c: C.soft });
     card([`random 1 move in 5:   Q-learning ${f2(VQ2)}   SARSA ${f2(VS2)}`, { s: 'expected return from the start, exact', c: C.soft }], t - 5.5, { y: 240, gap: 1 }); } },
 { h: 'Is this one run a fluke?', p: [`To check, run both learners ${NSEED} times with different random seeds, ${NEP} walks each. The chart shows what Pip actually earned while learning: each dot is the average return of 25 walks, averaged over the ${NSEED} runs. Across is the walk number; up is the return.`, 'Orange is Q-learning, teal is SARSA.'],
   draw(t) { room({ t }); pip(cx(0), cy(3), 1, t); dim(.5); const A = A2(); blockChart(A, Math.floor(clamp(t - 1.5, 0, 9) * 3), { q: 1, s: clamp(t - 5), legend: t > 9 }); } },
 { h: 'SARSA earns more while learning', p: [`Over all walks, SARSA earned ${f2(A2().rs)} per walk, Q-learning ${f2(A2().rq)}. SARSA’s route is longer, but only ${pct(A2().cs)} in 100 of its walks met a cat, against ${pct(A2().cq)} for Q-learning.`, `Early on both wander, earning about the same. Then SARSA’s numbers settle away from the cats, while Q-learning’s route keeps costing cats. Q-learning chose the cat-side route in ${A2().edgeQ} of ${NSEED} runs, SARSA in ${A2().edgeS || 'none'}.`],
   draw(t) { room({ t }); pip(cx(0), cy(3), 1, t); dim(.5); const A = A2(); blockChart(A, NB, { legend: true });
     card([`average return:  Q-learning ${f2(A.rq)}   SARSA ${f2(A.rs)}`, `walks ending on a cat:  ${pct(A.cq)} %   vs   ${pct(A.cs)} %`, `route along the cats:  ${A.edgeQ} runs   vs   ${A.edgeS}`], t - .6, { y: 120, gap: 1.2 }); } },
 { h: 'So who is right?', p: [`Both, about different Pips. Without random moves, Q-learning’s route would be worth ${f2(VQ0)}, SARSA’s only ${f2(VS0)}. While Pip still explores, SARSA’s is worth more.`, 'Q-learning learns the best way to act, whatever Pip does meanwhile. SARSA learns the best way **given its own habit**, random moves included. Rule of thumb: if learning mistakes cost something real, use SARSA; if only the final route matters, Q-learning.'],
   draw(t) { mini(20, 66, 28, FQ, PQ, QC, 'Q-learning', 1); mini(212, 66, 28, FS, PS, SC, 'SARSA', 1);
     card([`never random:          Q-learning ${f2(VQ0)}    SARSA ${f2(VS0)}`, `random 1 in 5:         Q-learning ${f2(VQ2)}    SARSA ${f2(VS2)}`, { s: 'expected return from the start, this run', c: C.soft }], t - .6, { y: 240, gap: 1.2 }); } },
 { h: 'Your turn', ctl: { label: 'ε', values: EPSV, def: 3, fmt: v => v },
   p: [`Drag **ε**. Each setting reruns both ${NSEED} times and draws each one’s most common route. The more random Pip is, the more SARSA avoids the cats.`, `At 0.2 the other SARSA runs take the top row or a mix; in ${A2().awayS} of ${NSEED} its arrows never touch a cell above a cat. At 0 the rules coincide; with little randomness SARSA earns a bit less.`],
   draw(t, st) { const e = EPSV[st.ci ?? 3], A = agg(e), rq = A.topQ[0].split(' ').map(s => s.split(',').map(Number)), rs = A.topS[0].split(' ').map(s => s.split(',').map(Number));
     mini(20, 66, 28, new Array(96).fill(0), rq, QC, 'Q-learning'); mini(212, 66, 28, new Array(96).fill(0), rs, SC, 'SARSA');
     txt(`${A.topQ[1]} of ${NSEED} runs`, 20 + 84, 66 + 112 + 13, { s: 13, c: C.soft }); txt(`${A.topS[1]} of ${NSEED} runs`, 212 + 84, 66 + 112 + 13, { s: 13, c: C.soft });
     card([`ε = ${e}: random move ${e === 0 ? 'never' : e === .05 ? '1 time in 20' : e === .1 ? '1 time in 10' : e === .2 ? '1 time in 5' : '3 times in 10'}`, `average return while learning:`, `     Q-learning ${f2(A.rq)}     SARSA ${f2(A.rs)}`, { s: `walks ending on a cat:  ${pct(A.cq)} %  vs  ${pct(A.cs)} %`, c: C.soft }], 10, { y: 250, lh: 24, fs: 17 }); } },
 { h: 'Where it came from', p: ['Gavin Rummery and Mahesan Niranjan described the method in 1994 as “modified connectionist Q-learning”; Richard Sutton suggested the name SARSA. A room like this one is the **cliff walking** example from Sutton and Barto’s textbook.', 'The lesson carries over: when mistakes made while learning are costly, as for a real robot near stairs, learning about the behaviour you actually have can be the safer choice.'],
   draw(t) { mini(20, 66, 28, FQ, PQ, QC, 'Q-learning'); mini(212, 66, 28, FS, PS, SC, 'SARSA');
     card(['1994  Rummery and Niranjan', '1996  the name SARSA (Sutton)', { s: 'cliff walking: the classic test of this contrast', c: C.soft }], t - .5, { y: 240, gap: 1.1 }); } },
 { h: 'Recap', p: ['SARSA picks its next move first, then updates toward the reward plus γ times **that move’s** number. So its numbers include Pip’s own random moves.', 'Near danger it learns a safer route and earns more while exploring. Q-learning, with its max, learns the best route for a Pip that never moves at random.'],
   draw(t) { qroom(FS, { t }); arrowsQ(FS, .7, '#2f6f6c'); pip(cx(0), cy(3), 1, t);
     card(['pick a′ first (ε-greedy), then update, then do it', 'target = r + γ × Q(s′, a′)', 'values include your own exploring', { s: 'safer near danger while learning', c: C.soft }], t - .3, { gap: 1 }); } },
];
RLStory.register('sarsa', { title: 'SARSA', track: 'Model-free', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { RQ, RS, FQ, FS, PQ, PS, MOM, QTGT, QNEW, VQ2, VS2, VQ0, VS0, agg, EPSV, qk } });
})();
