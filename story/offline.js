// Offline RL as a story: Pip learns only from an old diary of walks (no new steps). With a table nothing breaks; with a
// small network every move gets a number, untried moves get made-up ones, and Q-learning's max inflates them
// (extrapolation error). CQL pushes untried moves down and fixes it. Diary dice rng(11), network start knobs rng(2),
// test dice rng(1000 + i), i < 1000 (as in the policy and imitation reels). Training runs lazily on first use.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({}), { cx, cy, key, MOVES } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const SLIP = .2, GAM = .9, NWALK = 50, NAMES = ['up', 'right', 'down', 'left'], MO = '#9b7fc0', MOINK = '#5f4689';
const H = 8, ROUNDS = 30, GD = 60, LR = .2, NETSEED = 2, A_CQL = .5, AV = [0, .05, .1, .5, 5];
function mk(s) { const pi = []; s.split('/').forEach((row, r) => row.split('').forEach((ch, c) => { pi[r * 5 + c] = { U: 0, R: 1, D: 2, L: 3 }[ch] ?? 0; })); return pi; }
const PA = mk('RRRR./U.U.U/UL.RU/URRRU');                   // Mo's rule card = policy A
const OPEN = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.isRock(c, r) && !G.term(c, r)) OPEN.push([c, r]);
const no = (c, r) => r * 5 + c + 1, sgn = v => (v < -.005 ? '−' : '') + Math.abs(v).toFixed(2);
function slipMove(c, r, m, R) { const u = R(); let mm = m; if (u >= 1 - SLIP) mm = u < 1 - SLIP / 2 ? (m + 3) % 4 : (m + 1) % 4; const [nc, nr] = G.step(c, r, mm); return { c, r, m, mm, nc, nr, rew: G.term(nc, nr) }; }
function walk(pi, R, from = [0, 3], cap = 100) { let [c, r] = from; const tr = [];
  for (let k = 1; k <= cap; k++) { const s = slipMove(c, r, pi[r * 5 + c], R); tr.push(s); c = s.nc; r = s.nr; if (s.rew) return { tr, end: s.rew, ret: s.rew * GAM ** (k - 1) }; }
  return { tr, end: 0, ret: 0 }; }
// ───── the diary: 50 old walks of Pip with Mo's rule card on the wet floor, every step with its reward
const DIARY = (() => { const R = rng(11), D = []; for (let i = 0; i < NWALK; i++) D.push(walk(PA, R)); return D; })();
const DATA = DIARY.flatMap(w => w.tr), NSTEP = DATA.length, DCH = DIARY.filter(w => w.end > 0).length, DCAT = NWALK - DCH;
const TRIED = new Set(DATA.map(t => (t.r * 5 + t.c) * 4 + t.m)), DCELLS = new Set(DATA.map(t => t.r * 5 + t.c)).size;
const AGG = (() => { const m = new Map(); DATA.forEach(t => { const k = [t.c, t.r, t.m, t.nc, t.nr].join(); if (!m.has(k)) m.set(k, Object.assign({ n: 0 }, t)); m.get(k).n++; }); return [...m.values()]; })();
// test: 1000 walks from the start, same dice for every rule
function judge(pi) { let s = 0, ch = 0, ct = 0; for (let i = 0; i < 1000; i++) { const w = walk(pi, rng(1000 + i)); s += w.ret; if (w.end > 0) ch++; else if (w.end < 0) ct++; } return { avg: s / 1000, ch, ct, to: 1000 - ch - ct }; }
const argmax = q => { let b = -1e9, bm = 0; for (let a = 0; a < 4; a++) if (q[a] > b) { b = q[a]; bm = a; } return bm; };
// exact values from the rulebook (for the narrator only: the best possible value at the start, and Mo's card there)
function qv(V, c, r, m) { let v = 0; [[m, 1 - SLIP], [(m + 3) % 4, SLIP / 2], [(m + 1) % 4, SLIP / 2]].forEach(([mm, p]) => { const [a, b] = G.step(c, r, mm), rw = G.term(a, b); v += p * (rw + (rw ? 0 : GAM * V[b * 5 + a])); }); return v; }
const EXACT = (() => { const VA = new Float64Array(20), VS = new Float64Array(20); for (let it = 0; it < 2000; it++) OPEN.forEach(([c, r]) => { VA[r * 5 + c] = qv(VA, c, r, PA[r * 5 + c]); VS[r * 5 + c] = Math.max(...[0, 1, 2, 3].map(m => qv(VS, c, r, m))); }); return { A: VA[15], best: VS[15] }; })();
// ───── a table, replayed like the on/off reel (step 0.2, 200 readings): untried moves stay 0
const TAB = (() => { const Q = new Float64Array(80); for (let p = 0; p < 200; p++) DATA.forEach(t => { const k = (t.r * 5 + t.c) * 4 + t.m, nx = t.rew ? 0 : Math.max(Q[(t.nr * 5 + t.nc) * 4], Q[(t.nr * 5 + t.nc) * 4 + 1], Q[(t.nr * 5 + t.nc) * 4 + 2], Q[(t.nr * 5 + t.nc) * 4 + 3]); Q[k] += .2 * (t.rew + GAM * nx - Q[k]); });
  const pi = new Array(20).fill(0); OPEN.forEach(([c, r]) => { pi[r * 5 + c] = argmax([0, 1, 2, 3].map(a => Q[(r * 5 + c) * 4 + a])); }); return { Q, pi, J: judge(pi) }; })();
// ───── the network: column and row (scaled to −1 … 1) → 8 tanh units → 4 numbers, one per move
const feat = (c, r) => [c / 2 - 1, r / 1.5 - 1], NKNOB = 2 * H + H + H * 4 + 4;
function newNet(seed) { const R = rng(seed), w = { W1: [], b1: new Float64Array(H), W2: [], b2: new Float64Array(4) }; for (let j = 0; j < H; j++) w.W1.push([R() * 2 - 1, R() * 2 - 1]); for (let k = 0; k < 4; k++) w.W2.push(Array.from({ length: H }, () => (R() * 2 - 1) / Math.sqrt(H))); return w; }
function fwd(w, c, r) { const x = feat(c, r), h = w.W1.map((v, j) => Math.tanh(v[0] * x[0] + v[1] * x[1] + w.b1[j])); return { x, h, q: w.W2.map((v, k) => v.reduce((s, a, j) => s + a * h[j], w.b2[k])) }; }
// fitted Q-learning on the diary: each round, targets = reward + 0.9 × the biggest number at the landing cell (just the
// reward if the walk ended); then 60 small steps of the knobs toward them. CQL adds α × (soft max of the 4 numbers − the
// diary's move's number) at every diary step: all moves pushed down, hardest the biggest; the diary's move pushed up.
const memo = {};
function train(al) { if (memo[al]) return memo[al]; const w = newNet(NETSEED), hist = [Math.max(...fwd(w, 0, 3).q)];
  for (let it = 0; it < ROUNDS; it++) { const tg = AGG.map(t => t.rew ? t.rew : GAM * Math.max(...fwd(w, t.nc, t.nr).q));
    for (let g = 0; g < GD; g++) { const gW1 = w.W1.map(() => [0, 0]), gb1 = new Float64Array(H), gW2 = w.W2.map(() => new Float64Array(H)), gb2 = new Float64Array(4);
      AGG.forEach((t, i) => { const f = fwd(w, t.c, t.r), dq = [0, 0, 0, 0]; dq[t.m] += t.n * (f.q[t.m] - tg[i]);
        if (al) { const mx = Math.max(...f.q), ex = f.q.map(q => Math.exp(q - mx)), Z = ex.reduce((a, b) => a + b, 0); for (let a = 0; a < 4; a++) dq[a] += t.n * al * (ex[a] / Z - (a === t.m ? 1 : 0)); }
        for (let k = 0; k < 4; k++) { gb2[k] += dq[k]; for (let j = 0; j < H; j++) gW2[k][j] += dq[k] * f.h[j]; }
        for (let j = 0; j < H; j++) { let d = 0; for (let k = 0; k < 4; k++) d += dq[k] * w.W2[k][j]; d *= 1 - f.h[j] * f.h[j]; gb1[j] += d; gW1[j][0] += d * f.x[0]; gW1[j][1] += d * f.x[1]; } });
      for (let j = 0; j < H; j++) { w.b1[j] -= LR * gb1[j] / NSTEP; w.W1[j][0] -= LR * gW1[j][0] / NSTEP; w.W1[j][1] -= LR * gW1[j][1] / NSTEP; }
      for (let k = 0; k < 4; k++) { w.b2[k] -= LR * gb2[k] / NSTEP; for (let j = 0; j < H; j++) w.W2[k][j] -= LR * gW2[k][j] / NSTEP; } }
    hist.push(Math.max(...fwd(w, 0, 3).q)); }
  const Q = new Float64Array(80), pi = new Array(20).fill(0); OPEN.forEach(([c, r]) => { const q = fwd(w, c, r).q; for (let a = 0; a < 4; a++) Q[(r * 5 + c) * 4 + a] = q[a]; pi[r * 5 + c] = argmax(q); });
  let umax = -1e9, ucell = null; OPEN.forEach(([c, r]) => { for (let a = 0; a < 4; a++) if (!TRIED.has((r * 5 + c) * 4 + a) && Q[(r * 5 + c) * 4 + a] > umax) { umax = Q[(r * 5 + c) * 4 + a]; ucell = [c, r, a]; } });
  const offData = OPEN.filter(([c, r]) => !TRIED.has((r * 5 + c) * 4 + pi[r * 5 + c])).length;
  return (memo[al] = { w, Q, pi, hist, J: judge(pi), start: Array.from(Q.slice(60, 64)), umax, ucell, offData, startWalk: walk(pi, rng(1000)) }); }
const PL = () => train(0), CQ = () => train(A_CQL);
const STARTM = PA[15];                                        // the diary's move at the start: up

// ───── drawing (four triangles per cell, as in the Q-learning and on/off reels)
const qk = (c, r, m) => (r * 5 + c) * 4 + m;
function tri(xm, ym, x0, y0, x1, y1, m, col, a) { const P = [[[x0, y0], [x1, y0]], [[x1, y0], [x1, y1]], [[x1, y1], [x0, y1]], [[x0, y1], [x0, y0]]][m];
  ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(xm, ym); ctx.lineTo(...P[0]); ctx.lineTo(...P[1]); ctx.closePath(); ctx.fill(); ctx.restore(); }
function wedges(Q, c, r, x0, y0, x1, y1, o = {}) { const xm = (x0 + x1) / 2, ym = (y0 + y1) / 2;
  for (let m = 0; m < 4; m++) { const v = Q[qk(c, r, m)]; if (o.triedOnly && !TRIED.has(qk(c, r, m))) continue; if (Math.abs(v) >= .004) tri(xm, ym, x0 + 2, y0 + 2, x1 - 2, y1 - 2, m, v > 0 ? C.gold : C.rose, lerp(.14, .62, clamp(Math.abs(v) / .9)) * (o.wa ?? 1)); }
  alpha(.55, () => { rline(x0 + 3, y0 + 3, x1 - 3, y1 - 3, 300 + c * 9 + r, { c: C.line, w: .9, once: true, j: .6 }); rline(x1 - 3, y0 + 3, x0 + 3, y1 - 3, 330 + c * 9 + r, { c: C.line, w: .9, once: true, j: .6 }); }); }
function wet(a = 1) { for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.isRock(c, r)) continue; L.hachure(OX + c * CS, OY + r * CS, CS, CS, C.blue, 15, 300 + c * 5 + r, .22 * a); } }
function qmaze(Q, t, o = {}) { if (Q) OPEN.forEach(([c, r]) => { const x0 = OX + c * CS, y0 = OY + r * CS; wedges(Q, c, r, x0, y0, x0 + CS, y0 + CS, o); }); if (o.wet ?? 1) wet(.6); G.maze({ t }); }
function arrowsPi(pi, a = 1, col = C.goldInk, hi) { if (a <= 0) return; OPEN.forEach(([c, r]) => { const k = r * 5 + c, off = hi && !TRIED.has(k * 4 + pi[k]); alpha(a, () => G.arrowIn(c, r, pi[k], 400 + k, { c: off ? '#8f2f2f' : col, w: off ? 2.8 : 1.5 })); }); }
function triedMarks(a = 1) { if (a <= 0) return; OPEN.forEach(([c, r]) => { const k = r * 5 + c; alpha(a, () => G.arrowIn(c, r, PA[k], 450 + k, { c: MOINK, w: 1.7 })); }); }
function zoom(Q, c, r, x, y, s, o = {}) { const a = o.a ?? 1; if (a <= 0) return; alpha(a, () => { box(x, y, s, s, C.card); wedges(Q, c, r, x, y, x + s, y + s); rrect(x, y, s, s, 640, { w: 1.4 });
  const xm = x + s / 2, ym = y + s / 2, P = [[xm, y + s * .17], [x + s * .8, ym + 1], [xm, y + s * .85], [x + s * .2, ym + 1]];
  for (let m = 0; m < 4; m++) { const v = Q[qk(c, r, m)], tr = TRIED.has(qk(c, r, m)); txt(sgn(v), P[m][0], P[m][1], { s: 14, c: tr ? MOINK : v < 0 ? C.roseInk : C.goldInk }); }
  if (o.label) txt(o.label, xm, y + s + 11, { s: 13, c: o.lc || C.soft }); }); }
function dim(a = .6) { box(0, 0, W, 285, C.bg, a); }
// walker along transitions (bumps nudge toward the wall)
function walkAt(tr, t, dur) { const k = Math.floor(t / dur); if (k >= tr.length) { const l = tr[tr.length - 1]; return { x: cx(l.nc), y: cy(l.nr), k: tr.length, done: true }; }
  if (k < 0) return { x: cx(tr[0].c), y: cy(tr[0].r), k: 0 }; const s = tr[k], u = clamp((t - k * dur) / (dur * .7));
  if (s.nc === s.c && s.nr === s.r) { const [dx, dy] = MOVES[s.mm], b = Math.sin(Math.PI * u) * CS * .28; return { x: cx(s.c) + dx * b, y: cy(s.r) + dy * b, k, s, u, bump: true }; }
  const e = ease(u); return { x: lerp(cx(s.c), cx(s.nc), e), y: lerp(cy(s.r), cy(s.nr), e), k, s, u }; }
let face = 1; function play(tr, t, dur) { const p = walkAt(tr, t, dur); for (let i = 0; i < Math.min(p.k, tr.length); i++) dot(cx(tr[i].nc) + ((i * 7) % 11) - 5, cy(tr[i].nr) + ((i * 5) % 11) - 5, 2.4, C.teal, .55);
  if (p.s) { const f = Math.sign(MOVES[p.s.mm][0]); if (f) face = f; } L.mouse(p.x, p.y + 3, face, .95, t); if (p.bump && p.u < .9) txt('bump', p.x, p.y - 28, { s: 14, c: C.soft }); return p; }
// the network picture: 2 inputs → 8 hidden → 4 outputs
function netPic(x, y, w, h, a = 1, t = 99) { if (a <= 0) return; const I = [[x, y + h * .35], [x, y + h * .65]], Hd = Array.from({ length: H }, (_, j) => [x + w / 2, y + h * (j + .5) / H]), O = [0, 1, 2, 3].map(k => [x + w, y + h * (k + .8) / 5]);
  alpha(a * .5, () => { I.forEach(p => Hd.forEach((q, j) => rline(p[0], p[1], q[0], q[1], 700 + j, { c: C.line, w: .8, once: true, j: .3 }))); Hd.forEach((p, j) => O.forEach((q, k) => rline(p[0], p[1], q[0], q[1], 720 + j * 4 + k, { c: C.line, w: .8, once: true, j: .3 }))); });
  alpha(a, () => { I.forEach(p => dot(p[0], p[1], 6, C.teal)); Hd.forEach(p => dot(p[0], p[1], 5, C.grey)); O.forEach(p => dot(p[0], p[1], 6, C.gold)); });
  txt('column', x - 10, I[0][1], { s: 13, al: 'right', c: C.soft, a }); txt('row', x - 10, I[1][1], { s: 13, al: 'right', c: C.soft, a });
  NAMES.forEach((n, k) => txt(n, x + w + 12, O[k][1], { s: 13, al: 'left', c: C.goldInk, a })); }
// chart: the start cell's biggest number after each round
function chart(series, t, o = {}) { const x0 = 70, x1 = 360, y0 = 250, y1 = 40, lo = -.5, hi = 3.2, Y = v => lerp(y0, y1, (clamp(v, lo, hi) - lo) / (hi - lo)), X = k => lerp(x0, x1, k / ROUNDS);
  rline(x0, y0, x1 + 6, y0, 950, { c: C.line }); rline(x0, y0, x0, y1 - 6, 951, { c: C.line }); [0, 1, 2, 3].forEach(v => txt(String(v), x0 - 8, Y(v), { s: 13, al: 'right', c: C.soft }));
  [0, 10, 20, 30].forEach(k => txt(String(k), X(k), y0 + 13, { s: 13, c: C.soft })); txt('round', x1, y0 + 28, { s: 13, al: 'right', c: C.soft }); txt('start cell: biggest number', x0 - 10, y1 - 18, { s: 13, al: 'left', c: C.soft });
  if (t > (o.lineT ?? 1)) { ctx.save(); ctx.setLineDash([5, 5]); ctx.globalAlpha = clamp(t - (o.lineT ?? 1)); ctx.strokeStyle = C.ink; ctx.beginPath(); ctx.moveTo(x0, Y(1)); ctx.lineTo(x1, Y(1)); ctx.stroke(); ctx.strokeStyle = '#2f6f6c'; ctx.beginPath(); ctx.moveTo(x0, Y(EXACT.best)); ctx.lineTo(x1, Y(EXACT.best)); ctx.stroke(); ctx.restore();
    txt('1: no walk can earn more', x1, Y(1) - 10, { s: 13, al: 'right', a: clamp(t - (o.lineT ?? 1)) }); txt(`best possible here: ${EXACT.best.toFixed(2)}`, x1, Y(EXACT.best) + 12, { s: 13, al: 'right', c: '#2f6f6c', a: clamp(t - (o.lineT ?? 1)) }); }
  series.forEach(([h, col, t0, lab]) => { const n = clamp(Math.floor((t - t0) / .2), 0, ROUNDS); if (t < t0) return; for (let k = 1; k <= n; k++) rline(X(k - 1), Y(h[k - 1]), X(k), Y(h[k]), 960 + k, { c: col, w: 1.8, once: true, j: .3 }); dot(X(n), Y(h[n]), 4, col); if (n >= ROUNDS && lab) txt(lab, X(n) - 4, Y(h[n]) - 14, { s: 14, al: 'right', c: col }); }); }

const STEPS = [
 { h: 'Learning without walking', p: ['In the on/off reel, Pip learned from the Wanderer’s diary without taking a single step itself. Now make that the whole rule: a fixed record of old experience, and **no new walks, ever**.', 'This is **offline RL**. It matters wherever trying things is costly or dangerous: you cannot test random treatments on patients, but hospitals keep years of records.'],
   draw(t) { qmaze(null, t, { wet: 0 }); L.mouse(cx(0), cy(3) + 3, 1, .95, t); dim(.4);
     alpha(clamp(t - 1), () => { rrect(130, 90, 140, 100, 980, { fill: C.card, c: C.ink, w: 1.4 }); for (let i = 0; i < 5; i++) rline(145, 112 + i * 15, 255, 112 + i * 15, 981 + i, { c: C.line, w: 1, once: true }); });
     txt('the diary', 200, 76, { s: 16, a: clamp(t - 1) });
     card(['online: act, see what happens, learn, repeat', { s: 'offline: learn from a fixed record only', c: C.goldInk }, { s: 'no new walks, no new trials', c: C.roseInk }], t - 3, { gap: 1.2 }); } },
 { h: 'Pip’s old diary', p: [`Pip once walked ${NWALK} times with Mo’s rule card on the wet floor, where a move goes where it points 8 times in 10 and slides to each side 1 in 10. Each step was logged: cell, move, reward, landing cell. ${NSTEP} steps; ${DCH} walks reached the cheese, ${DCAT} the cat.`, `Slides reached all ${DCELLS} cells, but each holds only **one move**: Mo’s arrow.`],
   draw(t) { const w = Math.min(NWALK - 1, Math.floor(t / .2)), tr = DIARY[w].tr; qmaze(null, t); triedMarks(clamp(t - 6) * .9);
     if (t < 10.5) { for (let i = 0; i < tr.length; i++) dot(cx(tr[i].nc) + ((i * 7) % 11) - 5, cy(tr[i].nr) + ((i * 5) % 11) - 5, 2.4, C.teal, .7); const l = tr[tr.length - 1]; L.mouse(cx(l.nc), cy(l.nr) + 3, 1, .9, t); }
     card([`diary: walk ${w + 1} of ${NWALK}`, `steps so far: ${DIARY.slice(0, w + 1).reduce((s, x) => s + x.tr.length, 0)}`, { s: t > 6 ? 'plum arrows: the only move tried in each cell' : ' ', c: MOINK }], 10); } },
 { h: 'With a table, no harm done', p: [`Replay with a table, as in the on/off reel: four numbers per cell, drawn as triangles, updated by Q-learning over many readings. Untried moves stay 0.`, `Harmless here: every tried move scores above 0, so Pip follows Mo’s arrows: cheese ${TAB.J.ch} times in 1000 tests. The start’s number, ${TAB.Q[60].toFixed(2)}, beats the true ${EXACT.A.toFixed(2)} only because this diary was lucky: ${DCH} of ${NWALK} reached the cheese.`],
   draw(t) { qmaze(TAB.Q, t); arrowsPi(TAB.pi, clamp(t - 2) * .8); L.mouse(cx(0), cy(3) + 3, 1, .95, t);
     zoom(TAB.Q, 0, 3, 30, 294, 92, { label: 'start cell (table)', a: clamp(t - 1) });
     card([{ s: 'untried moves: 0, no clue', c: C.soft }, { s: 'tried move: plum number', c: MOINK }, { s: `test: cheese ${TAB.J.ch} of 1000`, c: C.goldInk }], t - 3, { x: 140, w: 230, fs: 16 }); } },
 { h: 'Big problems need a network', p: ['In big problems a table is impossible, as the function approximation reel showed. A network instead computes the numbers from a description of the state.', `Pip’s tiny network takes the column and row, scaled to between −1 and 1, passes them through ${H} hidden units, and gives **four numbers**, one per move: ${NKNOB} knobs in all. Every move gets a number, tried or not.`],
   draw(t) { qmaze(null, t, { wet: .3 }); dim(.85); netPic(130, 50, 150, 200, clamp(t - .8));
     card(['in: column and row of the cell', 'out: one number for each of the 4 moves', { s: 'untried moves get numbers too', c: C.roseInk }], t - 4, { gap: 1.2 }); } },
 { h: 'Replaying with the network', p: [`As with the table, each round every diary step gets a target: the reward plus 0.9 times the **biggest** of the network’s four numbers at the landing cell, or just the reward if the walk ended there.`, `Then, with targets held fixed, all the knobs take ${GD} small steps toward them. ${ROUNDS} rounds in all. Nothing else changes; Pip never takes a new step.`],
   draw(t) { qmaze(null, t, { wet: .4 }); dim(.8); txt('target = r + 0.9 × max of the 4 numbers', W / 2, 90, { s: 18, a: clamp(t - .5) }); txt('at the landing cell (r alone if the walk ended)', W / 2, 116, { s: 14, c: C.soft, a: clamp(t - .5) });
     txt(`then ${GD} small knob steps toward the targets`, W / 2, 170, { s: 17, a: clamp(t - 2.5) }); txt(`repeat: ${ROUNDS} rounds`, W / 2, 200, { s: 17, c: C.goldInk, a: clamp(t - 4) });
     card(['the diary: fixed', 'the targets: from the network itself', { s: 'the max: looks at all 4 moves, tried or not', c: C.roseInk }], t - 5.5, { gap: 1.2 }); } },
 { h: 'Numbers that cannot be true', get p() { const P = PL(); return [`After ${ROUNDS} rounds, the start cell’s numbers are: up ${sgn(P.start[0])}, right ${sgn(P.start[1])}, down ${sgn(P.start[2])}, left ${sgn(P.start[3])}. Only “up” appears in the diary.`, `But no walk can earn more than 1: the cheese pays +1 once, and the discount only shrinks it. Under the best possible rule, the start is worth ${EXACT.best.toFixed(2)}. The untried “left”, ${sgn(P.start[3])}, is pure invention.`]; },
   draw(t) { const P = PL(); qmaze(P.Q, t); L.mouse(cx(0), cy(3) + 3, 1, .95, t); zoom(P.Q, 0, 3, 30, 294, 92, { label: 'start cell (network)', a: clamp(t - .8) });
     card([{ s: 'plum: the move the diary tried', c: MOINK }, { s: `left, never tried: ${sgn(P.start[3])}`, c: C.goldInk }, { s: 'no return can be above 1', c: C.roseInk }], t - 3, { x: 140, w: 230, fs: 16 }); } },
 { h: 'How the error grows', get p() { const P = PL(); return ['Across: rounds 0 to 30; round 0 is random knobs. Up: the start’s biggest number. Dashed lines: 1, the most any walk can earn, and the true best value.', `It climbs past 1, to ${sgn(P.hist[ROUNDS])}. Why? Each target’s max takes the landing cell’s biggest number, including untried moves’ made-up ones. It finds the most optimistic invention, and the next round builds on it: **extrapolation error**.`]; },
   shot: [5, 13],
   draw(t) { const P = PL(); chart([[P.hist, C.roseInk, 4, 'network, plain']], t, { lineT: 2 }); } },
 { h: 'Nobody ever checks', get p() { const P = PL(); return ['Online, the mistake would fix itself: Pip would try “left”, bump the wall, earn nothing, and lower the number each try. Offline, nobody tries it.', `Following the biggest numbers, Pip pushes into the wall at the start again and again. It reaches the cheese ${P.J.ch} times in 1000 tests. In ${P.offData} of ${OPEN.length} cells it picks a move the diary never tried.`]; },
   shot: [5, 12],
   draw(t) { const P = PL(); qmaze(P.Q, t, { wa: .45 }); arrowsPi(P.pi, .9, C.goldInk, true); const tr = P.startWalk.tr.slice(0, 12); play(tr, t - 1, .7);
     card([{ s: 'rose arrows: choices the diary never tried', c: C.roseInk }, `test: cheese ${P.J.ch} of 1000`, { s: P.J.to ? `stopped after 100 steps: ${P.J.to}` : ' ', c: C.soft }], t - 2, { gap: 1.2 }); } },
 { h: 'Stay close to the data', p: ['The fix sounds simple: **don’t trust moves the diary never tried**. If the data says nothing about a move, assume it is not great.', 'Offline methods do this in different ways. Some only ever pick moves that look like the diary’s. Others lower the numbers of untried moves, so the max stops finding inventions. The next step shows one of those.'],
   draw(t) { const P = PL(); qmaze(P.Q, t, { triedOnly: t > 2 }); triedMarks(clamp(t - 2) * .9); dim(.15);
     card([{ s: 'tried moves: learn from the data', c: MOINK }, { s: 'untried moves: assume they are not great', c: C.roseInk }], t - 3, { gap: 1.3 }); } },
 { h: 'Conservative Q-learning', p: ['**CQL**, from Aviral Kumar, Aurick Zhou, George Tucker and Sergey Levine in 2020, adds a penalty at each diary step: a soft maximum of the cell’s four numbers, minus the diary move’s. Lowering it pushes all four down, the biggest hardest, and the diary’s move back up.', `Targets hold tried moves up; untried ones only sink. Dial α (alpha), here ${A_CQL}, sets how hard.`],
   draw(t) { qmaze(null, t, { wet: .3 }); dim(.8); const x = 200, y = 140, s = 120; const u = clamp(t - 1.5);
     alpha(1, () => { box(x - s / 2, y - s / 2, s, s, C.card); rrect(x - s / 2, y - s / 2, s, s, 660, { w: 1.4 }); rline(x - s / 2, y - s / 2, x + s / 2, y + s / 2, 661, { c: C.line, once: true }); rline(x + s / 2, y - s / 2, x - s / 2, y + s / 2, 662, { c: C.line, once: true }); });
     txt('up (in the diary)', x, y - s / 2 - 14, { s: 14, c: MOINK }); alpha(u, () => { arrow(x, y - 20, x, y - 48, 663, { c: MOINK, w: 2 }); [[1, 0], [0, 1], [-1, 0]].forEach(([dx, dy], i) => arrow(x + dx * 34, y + dy * 34 - 12, x + dx * 34, y + dy * 34 + 12, 664 + i, { c: C.roseInk, w: 2 })); });
     txt('pushed up', x - 12, y - 34, { s: 13, c: MOINK, a: u, al: 'right' }); txt('pushed down', x + 64, y + 40, { s: 13, c: C.roseInk, a: u, al: 'left' });
     card(['plus, at every diary step:', { s: '   all 4 numbers down (biggest hardest)', c: C.roseInk }, { s: '   the diary’s move up', c: MOINK }, { s: `strength α = ${A_CQL}`, c: C.soft }], t - 4, { gap: 1.1 }); } },
 { h: 'After CQL', get p() { const P = PL(), Q = CQ(); return [`Same diary, knobs and rounds, with CQL. Start: up ${sgn(Q.start[0])}, right ${sgn(Q.start[1])}, down ${sgn(Q.start[2])}, left ${sgn(Q.start[3])}. The diary’s move is on top, above its real worth, ${EXACT.A.toFixed(2)}, and far below 1.`, `The new line stays down. Pip reaches the cheese ${Q.J.ch} times in 1000, against ${P.J.ch} without CQL. One invention survives: up in cell 14, into the cat, so it trails the diary’s rule, ${TAB.J.ch}.`]; },
   shot: [4, 13],
   draw(t) { const P = PL(), Q = CQ(); chart([[P.hist, C.roseInk, -99, 'plain'], [Q.hist, C.goldInk, 3, 'with CQL']], t, { lineT: -1 });
     zoom(Q.Q, 0, 3, 40, 294, 92, { label: 'start, with CQL', lc: C.goldInk, a: clamp(t - 9) }); card([`with CQL: cheese ${Q.J.ch} of 1000`, { s: `plain: cheese ${P.J.ch} of 1000`, c: C.roseInk }], t - 10, { x: 150, w: 220, fs: 16 }); } },
 { h: 'How hard to push', get p() { const Ts = AV.map(a => train(a)); return [`At α 0.05, inventions survive: cheese ${Ts[1].J.ch}. At 0.5 numbers are sane, but cell 14’s invention costs walks: ${Ts[3].J.ch}. At 5 Pip copies the diary, ${Ts[4].J.ch}, but the start’s number, ${sgn(Ts[4].start[STARTM])}, tops 1 again.`, 'Strong pushes win walks, middle ones honest numbers. With one move per cell, CQL cannot beat the diary’s rule. Offline RL’s promise, combining the best of varied walks, is not shown.']; },
   draw(t) { const Ts = AV.map(a => train(a)); qmaze(null, t, { wet: .3 }); dim(.93); txt('strength', 60, 30, { s: 13, al: 'left', c: C.soft }); txt('start’s biggest number', 170, 30, { s: 13, c: C.soft }); txt('cheese, of 1000', 300, 30, { s: 13, c: C.soft });
     AV.forEach((a, i) => { const u = clamp((t - .5 - i * .9) * 2); if (u <= 0) return; const y = 60 + i * 42, T = Ts[i]; txt(`α = ${a}`, 60, y, { s: 17, al: 'left', a: u });
       txt(`start ${sgn(T.start[argmax(T.start)])}`, 170, y, { s: 16, c: T.start[argmax(T.start)] > 1 ? C.roseInk : C.ink, a: u }); txt(`cheese ${T.J.ch}`, 300, y, { s: 16, c: T.J.ch > 800 ? C.goldInk : C.roseInk, a: u }); });
     card(['too gentle: inventions survive', { s: 'in between: sane numbers, one invention', c: C.goldInk }, 'very hard: copies the diary, numbers inflated'], t - 6, { gap: 1.2 }); } },
 { h: 'Your turn', ctl: { label: 'α', values: AV, def: 0, fmt: v => String(v), onChange: st => { st.w = null; } }, tap: tapCell,
   p: ['Slide α, the strength of CQL’s push, from 0, plain offline Q-learning, to 5. Each setting trains its own network on the same diary.', 'Rose arrows are choices the diary never tried. Tap a cell to drop Pip there and watch one walk with that network’s choices, slides and all.'],
   draw(t, st) { const T = train(AV[st.ci ?? 0]); qmaze(T.Q, t, { wa: .45 }); arrowsPi(T.pi, .9, C.goldInk, true);
     if (st.w) { const p = play(st.w.tr.slice(0, 30), t - st.wt, .5); caption(p.done && st.w.tr.length <= 30 ? (st.w.end > 0 ? `cheese after ${st.w.tr.length} steps` : st.w.end < 0 ? `cat after ${st.w.tr.length} steps` : 'stopped') : p.done ? 'still going after 30 steps…' : `step ${p.k + 1}`); }
     else { L.mouse(cx(0), cy(3) + 3, 1, .95, t); caption('tap a cell to drop Pip there'); }
     card([`α = ${AV[st.ci ?? 0]}: start cell’s biggest number ${sgn(T.start[argmax(T.start)])}`, { s: `untried choices: ${T.offData} of ${OPEN.length} cells`, c: T.offData ? C.roseInk : C.soft }, { s: `test: cheese ${T.J.ch}, cat ${T.J.ct} of 1000`, c: C.goldInk }], 10, { y: 292, lh: 22, fs: 16 }); } },
 { h: 'Other ways to stay close', p: ['**IQL**, implicit Q-learning, by Ilya Kostrikov, Ashvin Nair and Sergey Levine in 2021, ignores untried moves: its targets use only logged moves. Where a cell has several, it learns which did best and leans toward those.', 'Earlier, BCQ, batch-constrained Q-learning, in 2019, kept the learner to moves like the data’s. All share one idea: offline, do not trust what the data does not show.'],
   draw(t) { const Q = CQ(); qmaze(Q.Q, t, { triedOnly: true }); triedMarks(.8); dim(.25);
     card([{ s: 'CQL: push untried moves down', c: C.goldInk }, { s: 'IQL: targets from logged moves only', c: MOINK }, { s: 'BCQ: only moves like the data’s', c: C.soft }], t - .6, { gap: 1.2 }); } },
 { h: 'Where it matters', p: ['Offline RL is studied wherever logs are plentiful and experiments are costly: treatment records in healthcare, logs of what people clicked in recommendation systems, and recorded runs of robots.', 'Often it gives a careful **warm start**, followed by a little cautious practice in the real world. The toy here shows the main danger, not how large systems behave.'],
   draw(t) { const Q = CQ(); qmaze(Q.Q, t); arrowsPi(Q.pi, .7, C.goldInk, true); dim(.45);
     card(['healthcare: treatment records', 'recommendations: click logs', 'robots: recorded runs', { s: 'then: a little careful practice', c: C.goldInk }], t - .6, { gap: 1.1 }); } },
 { h: 'Recap', p: ['**Offline RL** learns from a fixed record, with no new trials. With a network, untried moves get made-up numbers, and Q-learning’s max hunts down the most optimistic: **extrapolation error**.', 'Nothing corrects it, because nobody tries those moves. The cure is to stay close to the data: CQL pushes untried moves down, IQL never asks about them.'],
   draw(t) { const Q = CQ(); qmaze(Q.Q, t); arrowsPi(Q.pi, .7, C.goldInk, true); L.mouse(cx(0), cy(3) + 3, 1, .95, t);
     card(['offline: a fixed record, no new trials', 'untried moves get invented numbers', 'the max finds them; nobody checks', { s: 'stay close to the data: CQL, IQL', c: C.goldInk }], t - .3, { gap: 1 }); } },
];
let taps = 0;
function tapCell(x, y, st, t) { const c = Math.floor((x - OX) / CS), r = Math.floor((y - OY) / CS); if (c < 0 || c >= 5 || r < 0 || r >= 4 || G.isRock(c, r) || G.term(c, r)) return;
  st.w = walk(train(AV[st.ci ?? 0]).pi, rng(7000 + taps++), [c, r]); st.wt = t; }
RLStory.register('offline', { title: 'Offline RL', track: 'Learning from people', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { DIARY, NSTEP, DCH, DCAT, DCELLS, TRIED, TAB, EXACT, train, PL, CQ, AGG } });
})();
