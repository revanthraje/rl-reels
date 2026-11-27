// REINFORCE as a story: Pip learns its door chances, then its maze chances, from its own tries. After each whole
// episode every move's preference is nudged by α × (return that followed it) × (gradient of its log-probability).
// Door room and maze as in briefs/notes/pg_world.md. All numbers come from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
// ===== pg kit (shared helpers for the Policy Gradients track; copy, do not import) =====
const softmax = th => { const m = Math.max(...th), e = th.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
const DCOL = ['#8fb3d9', '#ef8f5a', '#6fb7b4'], DINK = ['#4f7fb0', '#c4581f', '#3f7f7c'], DNAME = ['A', 'B', 'C'], DX = [90, 200, 310], DP = [.2, .5, .8];
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−'), fz = v => Math.abs(v) < .005 ? '0' : f2(v), sg = v => (v > .005 ? '+' : '') + fz(v);
function pie(x, y, r, pr, o = {}) {
  let a0 = -Math.PI / 2; pr.forEach((p, i) => { const a1 = a0 + p * Math.PI * 2; ctx.save(); ctx.globalAlpha = o.a ?? 1; ctx.fillStyle = (o.cols || DCOL)[i];
    ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, r, a0, a1); ctx.closePath(); ctx.fill(); ctx.restore();
    if (o.labels && p > .06) { const am = (a0 + a1) / 2; txt((o.names || DNAME)[i], x + Math.cos(am) * r * .62, y + Math.sin(am) * r * .62, { s: 13, a: o.a ?? 1 }); } a0 = a1; });
  rcircle(x, y, r, 880, { j: .6 });
}
function door(i, o = {}) {
  const x = DX[i], y0 = 50, w = 62, h = 96;
  box(x - w / 2, y0, w, h, DCOL[i], o.glow ? .6 : .3); rrect(x - w / 2, y0, w, h, 820 + i * 5, { w: o.glow ? 2.4 : 1.6 }); dot(x + w / 2 - 10, y0 + h / 2 + 4, 2.6, C.ink);
  txt(DNAME[i], x, y0 - 14, { s: 20, c: DINK[i] }); box(x - 18, y0 + 14, 36, 20, C.card); rrect(x - 18, y0 + 14, 36, 20, 840 + i, { c: C.line, w: 1 });
  txt(o.secret ? String(DP[i]) : '?', x, y0 + 24, { s: 15, c: o.secret ? C.goldInk : C.soft });
}
// door room: doors, preference + chance rows, pie, Pip. o = { th, secret, glow, rows, pie, pip, mx, my }
function room(t, o = {}) {
  rline(20, 146, 380, 146, 801, { w: 1.6 }); for (let i = 0; i < 3; i++) door(i, { secret: o.secret, glow: o.glow === i });
  const th = o.th || [0, 0, 0], p = softmax(th);
  if (o.rows !== false) for (let i = 0; i < 3; i++) { txt(`pref ${f2(th[i])}`, DX[i], 162, { s: 14, c: C.soft }); txt(`chance ${f2(p[i])}`, DX[i], 180, { s: 15, c: DINK[i] }); }
  if (o.pie !== false) pie(o.px ?? 300, o.py ?? 238, 34, p, { labels: true });
  if (o.pip !== false) L.mouse(o.mx ?? 110, o.my ?? 244, o.face ?? 1, 1, t);
}
// arrow fans in a maze cell: one arrow per move, length ∝ chance (chance 1 reaches the cell edge)
function fans(M, TH, o = {}) {
  for (let r = 0; r < M.rows; r++) for (let c = 0; c < M.cols; c++) {
    if (M.term(c, r) || M.isRock(c, r) || (o.skip && o.skip(c, r))) continue; const k = M.key(c, r), p = softmax(Array.from(TH.slice(k * 4, k * 4 + 4))), best = p.indexOf(Math.max(...p)), x = M.cx(c), y = M.cy(r) + 4;
    p.forEach((q, m) => { const [dx, dy] = M.MOVES[m], len = 3 + q * (M.cs / 2 - 7), big = m === best && q > .4;
      alpha(o.a ?? 1, () => arrow(x + dx * 4, y + dy * 4, x + dx * (4 + len), y + dy * (4 + len), 600 + k * 4 + m, { c: big ? C.goldInk : C.soft, w: big ? 2 : 1.2, head: 4 + q * 4 })); });
  }
}
// maze REINFORCE: walk from the start with the current chances (dice rng(seed)), cap 100 steps (stopped walk scores 0),
// then for every step t: θ(s_t, ·) += α × A_t × (onehot(a_t) − chance(·|s_t)), A_t = G_t − baseline(s_t) (baseline 0 here).
// opts: { keep: store transitions and preference snapshots, base: 'v' learns V(s) by averaging returns with step ab }
const GAM = .9, CAP = 100;
function soft4(th, s) { const o = s * 4, m = Math.max(th[o], th[o + 1], th[o + 2], th[o + 3]), a = Math.exp(th[o] - m), b = Math.exp(th[o + 1] - m), c = Math.exp(th[o + 2] - m), d = Math.exp(th[o + 3] - m), z = a + b + c + d; return [a / z, b / z, c / z, d / z]; }
function runMaze(G, seed, alphaStep, nep, opts = {}) {
  const R = rng(seed), th = new Float64Array(80), V = new Float64Array(20), rets = new Float64Array(nep), eps = [];
  for (let e = 0; e < nep; e++) {
    let c = G.start[0], r = G.start[1]; const tr = [];
    for (let k = 0; k < CAP; k++) { const s = G.key(c, r), p = soft4(th, s); let u = R(), m = 0; while (m < 3 && u >= p[m]) { u -= p[m]; m++; }
      const [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr); tr.push({ c, r, nc, nr, s, m, rew, p }); c = nc; r = nr; if (rew) break; }
    let g = 0; for (let i = tr.length - 1; i >= 0; i--) { g = tr[i].rew + GAM * g; tr[i].G = g; }
    for (const q of tr) { q.b = opts.base === 'v' ? V[q.s] : 0; const A = q.G - q.b; for (let a = 0; a < 4; a++) th[q.s * 4 + a] += alphaStep * A * ((a === q.m) - q.p[a]); }
    if (opts.base === 'v') for (const q of tr) V[q.s] += (opts.ab ?? .1) * (q.G - V[q.s]);
    rets[e] = tr[0].G; if (opts.keep) eps.push({ tr, ret: tr[0].G, end: tr[tr.length - 1].rew, th: Float64Array.from(th), V: Float64Array.from(V) });
  }
  return { rets, eps };
}
// ===== end pg kit =====
const G = L.gridWorld({}), { cx, cy, key } = G;
const A0 = .2, NEP = 300, NRUN = 100, ALPHAS = [.05, .1, .2, .5, 1], ADOOR = .5;
// ── door room: one try. The story uses the first seed (from 5000 up) whose first pick is door C and pays a crumb.
const TRY = (() => { for (let k = 5000; ; k++) { const R = rng(k), u = R(), a = u < 1 / 3 ? 0 : u < 2 / 3 ? 1 : 2, r = R() < DP[a] ? 1 : 0; if (a === 2 && r) return { seed: k, a, r }; } })();
const P0 = [1 / 3, 1 / 3, 1 / 3], TH1 = P0.map((p, i) => ADOOR * TRY.r * ((i === TRY.a) - p)), P1 = softmax(TH1);
// ── many tries at the uniform policy (rng(31)): running average of reward × gradient of log-probability, per preference
const NAVG = 2000, AVG = (() => { const R = rng(31), sum = [0, 0, 0], run = []; for (let i = 0; i < NAVG; i++) { const u = R(), a = u < 1 / 3 ? 0 : u < 2 / 3 ? 1 : 2, r = R() < DP[a] ? 1 : 0;
  for (let k = 0; k < 3; k++) sum[k] += r * ((k === a) - 1 / 3); run.push(sum.map(s => s / (i + 1))); } return run; })();
const EXACT = [0, 1, 2].map(i => (DP[i] - .5) / 3);
// ── maze: story run (seed 5000) and 100 runs per step size
const S = runMaze(G, 5000, A0, NEP, { keep: true }), E = S.eps;
const memo = {};
function many(a) { if (memo[a]) return memo[a]; const runs = []; for (let k = 0; k < NRUN; k++) runs.push(runMaze(G, 5000 + k, a, NEP).rets);
  const mean = new Float64Array(NEP); runs.forEach(rr => rr.forEach((v, i) => mean[i] += v / NRUN));
  const blk = (rr, i) => { let s = 0; for (let j = i; j < i + 25; j++) s += rr[j]; return s / 25; };
  const stuck = runs.filter(rr => { for (let j = NEP - 25; j < NEP; j++) if (rr[j] !== 0) return false; return true; }).length, end = mean.slice(NEP - 25).reduce((x, y) => x + y, 0) / 25;
  return (memo[a] = { runs, mean, sm: smooth(mean, 10), stuck, end, at: i => blk(mean, i) }); }
function smooth(v, w) { const o = new Float64Array(v.length); for (let i = 0; i < v.length; i++) { let s = 0, n = 0; for (let j = Math.max(0, i - w + 1); j <= i; j++) { s += v[j]; n++; } o[i] = s / n; } return o; }
const M2 = many(A0);
// numbers the story quotes
const W1 = E[0], W1N = W1.tr.length, W1G = W1.tr.map(q => q.G);
const CH1 = E.findIndex(e => e.end > 0), WC = E[CH1];
const SM = smooth(S.rets, 25), FIRST = [0, 1, 2, 3, 4].map(k => runMaze(G, 5000 + k, A0, NEP).rets);
const ROUTE = (() => { let [c, r] = G.start; const P = [[c, r]]; const th = E[NEP - 1].th; for (let k = 0; k < 30 && !G.term(c, r); k++) { const s = key(c, r), p = softmax(Array.from(th.slice(s * 4, s * 4 + 4))); [c, r] = G.step(c, r, p.indexOf(Math.max(...p))); P.push([c, r]); } return P; })();
const CATS = E.filter(e => e.end < 0).length, STOPS = E.filter(e => e.end === 0).length;
const STARTP = i => softmax(Array.from(E[i].th.slice(key(0, 3) * 4, key(0, 3) * 4 + 4)));
const at100 = FIRST.map(rr => rr.slice(75, 100).reduce((a, b) => a + b, 0) / 25);
const M1 = many(1), STUCKRUN = M1.runs.findIndex(rr => rr.slice(NEP - 25).every(v => v === 0)), STUCK = runMaze(G, 5000 + STUCKRUN, 1, NEP, { keep: true });
const STK_ENDCAT = (() => { const e = STUCK.eps; const firstStop = e.findIndex((x, i) => e.slice(i).every(y => y.end === 0)); return { firstStop, cats: e.slice(0, firstStop).filter(x => x.end < 0).length, ches: e.slice(0, firstStop).filter(x => x.end > 0).length }; })();
const BEST = GAM ** 6, RMIN = Math.min(...S.rets), RMAX = Math.max(...S.rets);

// ───── drawing helpers
const W1S = W1.tr;
function walkAt(tr, t, dur) {
  const k = Math.floor(t / dur);
  if (k >= tr.length) { const l = tr[tr.length - 1]; return { x: cx(l.nc), y: cy(l.nr), k: tr.length, done: true, s: l }; }
  if (k < 0) return { x: cx(tr[0].c), y: cy(tr[0].r), k: 0 };
  const s = tr[k], u = clamp((t - k * dur) / (dur * .7));
  if (s.nc === s.c && s.nr === s.r) { const [dx, dy] = G.MOVES[s.m], b = Math.sin(Math.PI * u) * 18; return { x: cx(s.c) + dx * b, y: cy(s.r) + dy * b, k, s, face: dx || 0 }; }
  const e = ease(u); return { x: lerp(cx(s.c), cx(s.nc), e), y: lerp(cy(s.r), cy(s.nr), e), k, s, face: s.nc - s.c };
}
const thBefore = i => i > 0 ? E[i - 1].th : new Float64Array(80);
// learning-curve chart. series: [{ v, col, w, dots }], explained in the text before use
function chart(o) {
  const x0 = o.x0 ?? 56, x1 = o.x1 ?? 350, y0 = o.y0 ?? 388, y1 = o.y1 ?? 304, lo = o.lo ?? -1, hi = o.hi ?? .6, n = o.n ?? NEP;
  const X = i => x0 + i / (NEP - 1) * (x1 - x0), Y = v => y0 - (clamp(v, lo, hi) - lo) / (hi - lo) * (y0 - y1);
  rline(x0 - 4, Y(0), x1 + 4, Y(0), 960, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1 - 2, 961, { c: C.line, w: 1, once: true });
  [[0, '0'], [hi, f2(hi)], [lo, f2(lo)]].forEach(([v, s]) => txt(s, x0 - 18, Y(v), { s: 12, c: C.soft }));
  if (o.best) { ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.grey; ctx.lineWidth = 1.1; ctx.beginPath(); ctx.moveTo(x0, Y(BEST)); ctx.lineTo(x1, Y(BEST)); ctx.stroke(); ctx.restore(); txt(`best ${f2(BEST)}`, x1 + 4, Y(BEST), { s: 12, c: C.soft, al: 'left' }); }
  txt('walk 1', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NEP}`, x1, y0 + 11, { s: 12, c: C.soft, al: 'right' });
  if (o.title) txt(o.title, x0 - 10, y1 - 12, { s: 13, c: C.soft, al: 'left' });
  if (o.ticks) for (let i = 0; i < n; i++) { const e = E[i].end; if (e > 0) continue; rline(X(i), y0 - 1, X(i), y0 - 8, 1200 + i, { c: e < 0 ? C.rose : C.grey, w: 1.3, once: true, j: .2 }); }
  (o.series || []).forEach(s => { const m = Math.min(n, s.v.length);
    if (s.dots) { for (let i = 0; i < m; i++) dot(X(i), Y(s.v[i]), 1.8, s.v[i] < -.001 ? C.rose : s.v[i] > .001 ? C.gold : C.grey, .8); return; }
    ctx.save(); ctx.globalAlpha = s.a ?? 1; ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 1.6; ctx.lineJoin = 'round'; ctx.beginPath();
    for (let i = 0; i < m; i++) { const x = X(i), y = Y(s.v[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); });
  return { X, Y };
}
function gLabel(q, a) { box(cx(q.c) - 22, cy(q.r) - 29, 44, 16, C.card, .85 * a); txt(f2(q.G), cx(q.c), cy(q.r) - 21, { s: 13, c: q.G < 0 ? C.roseInk : C.goldInk, a }); }
// which steps of walk 1 to label: the first visit of each cell, keeping the latest return there
const LAB1 = (() => { const seen = {}; W1S.forEach((q, i) => { seen[q.s] = i; }); return Object.values(seen).map(i => W1S[i]); })();

const STEPS = [
 { h: 'Learning from tries', p: ['In the last reel, Pip climbed toward the best door using the exact gradient. But that needed the hidden crumb chances, which Pip does not have.', '**REINFORCE** fixes this. Pip simply tries: it picks a door with its current chances, sees what happens, and nudges its preferences. No hidden chances, no rulebook, only its own experience.'],
   shot: [3, 9],
   draw(t) { room(t, { secret: t < 2.5 });
     if (t > 3) card([{ s: 'exact gradient: needs hidden chances  ✗', c: C.roseInk }, { s: 'REINFORCE: learn from Pip’s own tries', c: DINK[2] }], t - 3, { gap: 1.3 }); } },
 { h: 'One try', p: ['All preferences start at 0, so each door has chance 0.33. Pip rolls its dice and picks door C. A crumb appears: reward 1.', `The idea is simple: an action followed by a good return becomes **more likely**. So C’s preference goes up, and the other two go down a little.`],
   shot: [4, 11],
   draw(t) { const go = ease((t - 1) / 1.2), back = ease((t - 4) / 1.2), x = lerp(110, DX[2] - 60, go * (1 - back)), y = lerp(244, 214, go * (1 - back));
     const u = ease((t - 6) / 1.5), th = TH1.map(v => v * u); room(t, { th, pip: false, glow: t > 1 ? 2 : null });
     L.mouse(x, y, t > 4 && t < 5.2 ? -1 : 1, 1, t); bubble('crumb! +1', 180, 262, t - 2.3);
     if (t > 6) card([`C up: ${sg(TH1[2])}    A and B down: ${sg(TH1[0])} each`, { s: `new chances: A ${f2(P1[0])}  B ${f2(P1[1])}  C ${f2(P1[2])}`, c: DINK[2] }], t - 6, { gap: 1.4 }); } },
 { h: 'The nudge, exactly', p: [`Each preference moves by **step size** α (alpha) × return × a factor. For the chosen door the factor is 1 − its chance: 1 − 0.33 = 0.67. For every other door it is minus its chance, −0.33.`, `With α = ${ADOOR} and return 1: C gets ${sg(TH1[2])}, A and B ${sg(TH1[0])}. Had the reward been 0, nothing would change at all.`],
   shot: [4, 11],
   draw(t) { room(t, { th: TH1, glow: 2 });
     card(['chosen door:  + α × return × (1 − its chance)', 'other doors:   − α × return × their chance', { s: `C:  ${ADOOR} × 1 × 0.67 = ${sg(TH1[2])}`, c: DINK[2] }, { s: `A, B:  ${ADOOR} × 1 × 0.33 = ${f2(-TH1[0])} down`, c: C.roseInk }], t - .8, { gap: 1.4, fs: 16 }); } },
 { h: 'How surprising was the choice?', p: ['That factor is the gradient of the **log-probability**, the log of the chosen door’s chance. The nudge is scaled by the return and by how surprising the choice was. A door picked with chance 0.9 gets a factor of only 0.1; one picked with chance 0.1 gets 0.9.', 'Why? Likely doors get picked, and nudged, often; rare doors seldom, so each nudge counts more. The card shows the division by the chance exactly cancels how often a door is picked.'],
   shot: [4, 11],
   draw(t) { const xs = [110, 290];
     [[.9, 'likely pick'], [.1, 'surprising pick']].forEach(([p, s], j) => { const a = clamp(t - .5 - j * 2); if (a <= 0) return;
       alpha(a, () => { pie(xs[j], 110, 46, [p, 1 - p], { cols: [DCOL[2], '#e4d8c7'] }); txt(s, xs[j], 40, { s: 16 }); txt(`chance ${p}`, xs[j], 176, { s: 15, c: DINK[2] });
         const len = (1 - p) * 70; arrow(xs[j], 250, xs[j], 250 - len - 4, 920 + j, { c: DINK[2], w: 3 }); txt(`factor ${f2(1 - p)}`, xs[j] + 52, 236, { s: 15, c: C.goldInk }); }); });
     if (t > 5) card(['factor = change in chance ÷ chance', '   (for the chosen door: 1 − its chance)', 'on average, door picked with its chance:', '   chance × reward × change ÷ chance', { s: '   = reward × change in chance: exact gradient', c: C.goldInk }], t - 5, { gap: 1.2, fs: 15, lh: 20, y: 286 }); } },
 { h: 'On average, uphill', p: [`One try is a rough guess. Here Pip tries ${NAVG.toLocaleString('en-US')} times without changing its chances, and we average reward × factor for each door. Each line is a running average; the dashed lines are the exact gradient from the last reel: −0.10, 0 and +0.10.`, `After ${NAVG.toLocaleString('en-US')} tries the averages are ${f2(AVG[NAVG - 1][0])}, ${fz(AVG[NAVG - 1][1])} and ${f2(AVG[NAVG - 1][2])}. REINFORCE’s nudges are noisy, but on average they point **uphill**.`],
   shot: [4, 12],
   draw(t) { room(t, { secret: 1, rows: false, pie: false, pip: false }); const n = clamp(Math.floor((t - 1) * 200), 0, NAVG);
     const x0 = 60, x1 = 340, y0 = 372, y1 = 200, Y = v => y0 - (v + .4) / .8 * (y0 - y1), X = i => x0 + Math.log(i + 1) / Math.log(NAVG) * (x1 - x0);
     rline(x0, Y(0), x1, Y(0), 930, { c: C.line, w: 1, once: true }); rline(x0, y0, x0, y1, 931, { c: C.line, w: 1, once: true });
     txt('+0.4', x0 - 18, Y(.4), { s: 12, c: C.soft }); txt('0', x0 - 12, Y(0), { s: 12, c: C.soft }); txt('−0.4', x0 - 18, Y(-.4), { s: 12, c: C.soft });
     txt('try 1', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt('10', X(9), y0 + 11, { s: 12, c: C.soft }); txt('100', X(99), y0 + 11, { s: 12, c: C.soft }); txt(NAVG.toLocaleString('en-US'), x1, y0 + 11, { s: 12, c: C.soft, al: 'right' });
     txt('average nudge per door (tries on a log scale)', x0 - 6, y1 - 12, { s: 13, c: C.soft, al: 'left' });
     [0, 1, 2].forEach(k => { ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = DCOL[k]; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(x0, Y(EXACT[k])); ctx.lineTo(x1, Y(EXACT[k])); ctx.stroke(); ctx.restore();
       ctx.save(); ctx.strokeStyle = DINK[k]; ctx.lineWidth = 1.8; ctx.beginPath(); for (let i = 0; i < n; i++) { const x = X(i), y = Y(clamp(AVG[i][k], -.4, .4)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore();
       if (n > 0) txt(DNAME[k], x1 + 10, Y(EXACT[k]), { s: 14, c: DINK[k] }); });
     caption(n ? `after ${n.toLocaleString('en-US')} tries` : ''); } },
 { h: 'A whole walk', p: ['Now the maze: dry floor, +1 onto the cheese, −1 onto the cat, both end the walk, γ (gamma) 0.9. All preferences start at 0.', 'REINFORCE waits for the walk to finish, like Monte Carlo, and credits every move with the **return that followed it**, since a move cannot change earlier rewards. Textbook REINFORCE also shrinks late moves’ nudges by 0.9 per step since the start; like most implementations, we leave that out.'],
   shot: [3, 9],
   draw(t) { G.maze({ t, labels: clamp(t - .5) }); fans(G, new Float64Array(80), { skip: (c, r) => c === 0 && r === 3 }); G.pip(cx(0), cy(3), 1, t);
     if (t > 4) card(['return of a move = final reward × 0.9 per later move', { s: 'cheese 3 moves later: 0.9 × 0.9 × 1 = 0.81', c: C.goldInk }, { s: 'a walk stopped after 100 moves scores 0', c: C.soft }], t - 4, { gap: 1.3, fs: 16 }); } },
 { h: 'Walk 1', p: [`Walk 1 takes ${W1N} moves. Pip wanders, bumps into walls ${W1S.filter(q => q.nc === q.c && q.nr === q.r).length} times, and finally steps up from the cell below the cat onto the cat: reward −1.`, `The walk is over, so every move now gets its return: −1 for the last one, ${f2(W1G[W1N - 2])} for the one before, and so on back to ${f2(W1G[0])} for Pip’s first move. The labels show the return of the last move made in each cell.`],
   shot: [4, 12],
   draw(t) { const dur = .45, P = walkAt(W1S, t - .5, dur); G.maze({ t }); fans(G, new Float64Array(80), { a: .5 });
     for (let i = 0; i < Math.min(P.k, W1N); i++) dot(cx(W1S[i].nc) + ((i * 7) % 11) - 5, cy(W1S[i].nr) + ((i * 5) % 11) - 5, 2.4, C.teal, .6);
     G.pip(P.x, P.y, P.face, t); const tEnd = .5 + W1N * dur; if (t > tEnd) bubble('−1', cx(3), cy(1) - 18, t - tEnd);
     if (t > tEnd + 1) LAB1.forEach((q, i) => gLabel(q, clamp((t - tEnd - 1 - i * .25) * 3)));
     if (t > tEnd + 3) card([`last move −1, the one before ${f2(W1G[W1N - 2])}, then ${f2(W1G[W1N - 3])} …`, { s: `first move: 0.9 to the ${W1N - 1}th × (−1) = ${f2(W1G[0])}`, c: C.roseInk }], t - tEnd - 3, { gap: 1.3, fs: 16 });
     if (!P.done) caption(`walk 1, move ${Math.min(W1N, P.k + 1)}`); } },
 { h: 'Every move pushed down', p: ['Now the nudges. Each move’s preference changes by α × its return × its factor, with α = 0.2. All returns were negative, so **every move** Pip made became less likely.', 'Fair for the last move, up into the cat: in that cell its chance drops from 0.25 to ' + f2(softmax(Array.from(E[0].th.slice(key(3, 2) * 4, key(3, 2) * 4 + 4)))[0]) + '. But Pip’s first move, up from the start, was a fine move. Its chance drops too, to ' + f2(softmax(Array.from(E[0].th.slice(key(0, 3) * 4, key(0, 3) * 4 + 4)))[0]) + ', punished only because this walk happened to end badly.'],
   shot: [3, 10],
   draw(t) { const u = ease((t - 1) / 2), th = E[0].th.map(v => v * u); G.maze({ t, hi: t > 3 ? [[3, 2, C.rose, .25], [0, 3, C.teal, .2]] : null }); fans(G, th);
     const pc = (c, r) => softmax(Array.from(E[0].th.slice(key(c, r) * 4, key(c, r) * 4 + 4)))[0];
     if (t > 3) card([{ s: `below the cat, up: 0.25 → ${f2(pc(3, 2))}   fair`, c: C.roseInk }, { s: `start, up: 0.25 → ${f2(pc(0, 3))}   unlucky`, c: '#3f7f7c' }, { s: 'small steps (α = 0.2): the arrows barely change', c: C.soft }, { s: 'a cell visited twice gets two nudges', c: C.soft }], t - 3, { gap: 1.3 }); } },
 { h: 'The first cheese', p: [`Walk ${CH1 + 1} is the first to reach the cheese, after ${WC.tr.length} moves. The last move gets return 1, but Pip’s first move only ${f2(WC.ret)}: the cheese was too many steps away to matter much.`, 'So learning starts near the cheese, where returns are big, and spreads back toward the start as walks get shorter.'],
   shot: [3, 10],
   draw(t) { const th = E[CH1].th, th0 = thBefore(CH1), u = ease((t - 2) / 2); const T = th0.map((v, i) => lerp(v, th[i], u));
     G.maze({ t }); fans(G, T); for (let i = 0; i < WC.tr.length; i++) dot(cx(WC.tr[i].nc) + ((i * 7) % 11) - 5, cy(WC.tr[i].nr) + ((i * 5) % 11) - 5, 2.2, C.teal, .35);
     if (t > 3) gLabel(WC.tr[WC.tr.length - 1], clamp(t - 3)); gLabel(WC.tr[0], clamp(t - 5)); bubble('+1', cx(4), cy(0) - 18, t - .5); G.pip(cx(4), cy(0) + 6, 1, t);
     caption(`walk ${CH1 + 1}: cheese after ${WC.tr.length} moves`); } },
 { h: `${NEP} walks`, p: ['A new chart. Along the bottom: walk number. Up the side: that walk’s return from the start. Dots: gold for cheese, rose for cat. Long walks score near 0, so bottom ticks also mark cat walks, rose, and stopped walks, grey. The line averages the last 25 walks.', `Early walks are long and score near 0. Then the returns climb: the average of the last 25 walks reaches ${f2(SM[NEP - 1])}, near the best possible ${f2(BEST)}.`],
   shot: [4, 13],
   draw(t) { const n = clamp(Math.floor((t - 1.5) * 30), 0, NEP); G.maze({ t }); fans(G, n ? E[n - 1].th : new Float64Array(80), { a: .9 });
     box(0, 284, W, 120, C.bg); chart({ n, series: [{ v: S.rets, dots: true }, { v: SM, col: C.goldInk, w: 2 }], best: true, title: 'return of each walk', ticks: true }); caption(n ? `after walk ${n}` : ''); } },
 { h: 'What Pip learned', p: [`After ${NEP} walks, following the biggest arrow from the start gives a route of ${ROUTE.length - 1} moves to the cheese, the shortest possible. At the start, “up” now has chance ${f2(STARTP(NEP - 1)[0])}.`, `Along the way, ${CATS} walks ended on the cat and ${STOPS} were stopped at 100 moves. Stopped walks scored 0 and changed nothing. Off the route, cells were rarely visited, so their arrows stay rough.`],
   shot: [3, 10],
   draw(t) { G.maze({ t }); fans(G, E[NEP - 1].th); const n = clamp(Math.floor((t - 1.5) / .4), 0, ROUTE.length); for (let i = 1; i < n; i++) { const [a, b] = ROUTE[i - 1], [c, d] = ROUTE[i]; rline(cx(a), cy(b) - 14, cx(c), cy(d) - 14, 940 + i, { c: C.teal, w: 3 }); } G.pip(cx(0), cy(3), 1, t);
     if (t > 5) card([`biggest arrows: ${ROUTE.length - 1}-move route to the cheese`, { s: `${CATS} cat walks (pushed down), ${STOPS} stopped (no change)`, c: C.soft }], t - 5, { gap: 1.3 }); } },
 { h: 'Same recipe, other dice', p: [`Run REINFORCE again with different dice. The thin lines are five runs, each the average of its last 25 walks. Around walk 100 they range from ${f2(Math.min(...at100))} to ${f2(Math.max(...at100))}: some runs learn early, some late.`, `The thick line averages ${NRUN} runs. It climbs smoothly, to ${f2(M2.end)} over the last 25 walks. A single run is a very noisy version of it.`],
   shot: [4, 12],
   draw(t) { const n = clamp(Math.floor((t - .8) * 60), 0, NEP); const cols = ['#c99a3a', '#4f7fb0', '#3f9a96', '#c4581f', '#a06cb0'];
     chart({ y0: 300, y1: 60, n, series: FIRST.map((rr, k) => ({ v: smooth(rr, 25), col: cols[k], w: 1.2, a: .8 })).concat(t > 6 ? [{ v: M2.sm, col: C.ink, w: 3 }] : []), best: true, lo: -.2, title: 'average return of the last 25 walks' });
     if (t > 6) card([`thin: runs 1 to 5, around walk 100 from ${f2(Math.min(...at100))} to ${f2(Math.max(...at100))}`, { s: `thick: average of ${NRUN} runs`, c: C.ink }], t - 6, { gap: 1.2, y: 322 }); } },
 { h: 'Why so noisy?', p: ['Each walk’s luck colours **every move** in it. One random detour into the cat punishes all the good moves before it, as walk 1 showed; one lucky walk rewards the bumps along the way.', `The bars count Pip’s ${NEP} walks by their return from the start, from ${f2(RMIN)} to ${f2(RMAX)}. Every nudge is scaled by returns like these, so the nudges swing as widely. This is **high variance**: right on average, but each single estimate jumps around. The next reel calms it.`],
   shot: [3, 9],
   draw(t) { const x0 = 50, x1 = 360, base = 250, bw = .05, lo = -.6, nb = 24, cnt = new Array(nb).fill(0); S.rets.forEach(v => cnt[Math.min(nb - 1, Math.floor((v - lo) / bw))]++);
     const mx = Math.max(...cnt), X = v => x0 + (v - lo) / (nb * bw) * (x1 - x0), u = ease((t - .5) / 2);
     txt(`returns of Pip’s ${NEP} walks (how many walks per bar)`, W / 2, 40, { s: 15, c: C.soft });
     cnt.forEach((k, b) => { if (!k) return; const v = lo + b * bw, h = Math.sqrt(k / mx) * 170 * u; box(X(v) + 1, base - h, X(v + bw) - X(v) - 2, h, v < -.001 ? C.rose : C.gold); });
     rline(x0, base, x1, base, 1300, { c: C.line }); [-.5, 0, .5].forEach(v => txt(String(v).replace('-', '−'), X(v), base + 12, { s: 12, c: C.soft }));
     txt('bar height grows with the square root of the count', W / 2, base + 30, { s: 12, c: C.soft, a: u });
     card([{ s: `walk returns from ${f2(RMIN)} to ${f2(RMAX)}`, c: C.ink }, { s: 'one walk’s return credits all its moves', c: C.soft }, { s: 'right on average, noisy each time: high variance', c: C.soft }], t - 2.5, { gap: 1.3, y: 300 }); } },
 { h: 'Your turn', ctl: { label: 'α', values: ALPHAS, def: 2, onChange: st => {} },
   p: [`Drag the step size **α**. Each setting reruns ${NRUN} runs of ${NEP} walks with the same dice. The thick line is their average return; the thin lines are runs 1 to 5.`, 'Small α learns slowly but steadily. Big α learns faster at first, but watch the count of runs that end up stuck.'],
   shot: [3, 9],
   draw(t, st) { const a = ALPHAS[st.ci ?? 2], M = many(a), n = clamp(Math.floor((t - (st.changed ?? 0) - .3) * 80), 0, NEP);
     chart({ y0: 270, y1: 60, lo: -.2, n, series: M.runs.slice(0, 5).map(rr => ({ v: smooth(rr, 25), col: C.grey, w: 1, a: .9 })).concat([{ v: M.sm, col: C.goldInk, w: 3 }]), best: true, title: `α = ${a}: average return (thick), runs 1–5 (thin)` });
     card([`average over the last 25 walks: ${f2(M.end)}`, { s: `stuck runs (last 25 walks all stopped): ${M.stuck} of ${NRUN}`, c: M.stuck > 5 ? C.roseInk : C.soft }], 9, { gap: 0, y: 304 }); } },
 { h: 'Too big a step', p: [`With α = 1, ${M1.stuck} of ${NRUN} runs get stuck. In run ${STUCKRUN + 1}, the early walks hit the cat ${STK_ENDCAT.cats} times and never the cheese. The big pushes left most big arrows pointing left or down, away from the cheese.`, `From walk ${STK_ENDCAT.firstStop + 1} on, all ${NEP - STK_ENDCAT.firstStop} walks hit the 100-move stop: return 0, no nudge. No chance is exactly 0, so an ending could still come, but here it practically never does. Big steps on noisy estimates can wreck a policy.`],
   shot: [3, 10],
   draw(t) { const e = STUCK.eps[NEP - 1]; G.maze({ t }); fans(G, e.th); const tr = e.tr.slice(0, 40), P = walkAt(tr, (t % 18) - .5, .35); G.pip(P.x, P.y, P.face, t);
     if (t > 4) card([`early walks: cat ${STK_ENDCAT.cats} times, cheese ${STK_ENDCAT.ches} times`, 'big pushes away from the right side', { s: 'walks hit the 100-move stop → return 0 → no nudge', c: C.roseInk }], t - 4, { gap: 1.3 });
     caption(`run ${STUCKRUN + 1}, α = 1, walk ${NEP}: stopped after 100 moves, return 0`); } },
 { h: 'Where it is used', p: ['Ronald **Williams** published REINFORCE in 1992. AlphaGo, described in 2016, improved its policy network this way: games against earlier versions of itself, with each move nudged by whether the game was won or lost.', 'REINFORCE-style updates are still used today, for example to fine-tune language models on a reward score, usually together with the noise-calming tricks of the next reels.'],
   shot: [3, 9],
   draw(t) { alpha(clamp(t - .3), () => { for (let i = 0; i <= 4; i++) { rline(60 + i * 22, 70, 60 + i * 22, 158, 950 + i, { c: C.line, w: 1, once: true }); rline(60, 70 + i * 22, 148, 70 + i * 22, 956 + i, { c: C.line, w: 1, once: true }); }
       dot(82, 92, 7, C.ink); dot(104, 114, 7, '#f5efe6'); rcircle(104, 114, 7, 962, { j: .3 }); txt('Go: win = +1, loss = −1', 104, 186, { s: 15 }); });
     alpha(clamp(t - 2), () => { rrect(240, 80, 110, 60, 970, { fill: C.card }); txt('reply A  ★ 0.8', 295, 100, { s: 14 }); txt('reply B  ★ 0.3', 295, 122, { s: 14, c: C.soft }); txt('language models', 295, 186, { s: 15 }); });
     if (t > 4) card(['whole episode, then nudge each action', { s: 'by the return that followed it', c: C.soft }], t - 4, { gap: 1.2 }); } },
 { h: 'Recap', p: ['**REINFORCE** learns a policy from complete episodes. After each one, every action’s log-probability is nudged by step size × the return that followed it.', 'Rare choices get bigger nudges, so on average the nudges follow the true gradient. But one episode’s luck colours every action, so single updates are noisy, and big steps can wreck the policy.'],
   shot: [3, 9],
   draw(t) { G.maze({ t }); fans(G, E[NEP - 1].th); G.pip(cx(0), cy(3), 1, t);
     card(['play a whole episode with the current chances', 'return after each move: reward discounted by γ', 'pref += α × return × (onehot − chance)', { s: 'right on average, noisy: high variance', c: C.soft }], t - .3, { gap: 1, fs: 16 }); } },
];

RLStory.register('reinforce', { title: 'REINFORCE', track: 'Policy Gradients', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { TRY, TH1, P1, avgEnd: AVG[NAVG - 1], W1N, W1G, W1path: W1S.map(q => q.c + '' + q.r + 'URDL'[q.m]).join(' '), CH1, WCn: WC.tr.length, WCret: WC.ret, SMend: SM[NEP - 1], route: ROUTE.length - 1,
    startUp: STARTP(NEP - 1)[0], CATS, STOPS, at100, M2end: M2.end, M2stuck: M2.stuck, M1stuck: M1.stuck, STUCKRUN, STK_ENDCAT, alphas: () => ALPHAS.map(a => [a, many(a).end, many(a).stuck]) } });
})();
