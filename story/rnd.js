// RND & Go-Explore as a story: Pip in a long meadow where no spot ever repeats, so visit counts are useless.
// A copier network chasing a fixed random network gives a novelty bonus that spreads to nearby spots (RND);
// an archive of reached places, returned to and explored from, gives Go-Explore. All numbers from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, gauss, rline, rrect, rcircle, txt, arrow, card, caption, dot, box, alpha } = L;
let ctx;
// ───── the long meadow: x in [0, 4], y in [0, 1]; start (0.1, 0.5); cheese circle at (3.85, 0.5), radius 0.12
const XW = 4, STEP = 0.15, WOB = 0.02, CH = [3.85, 0.5], CR = 0.12, TL = 40, NW = 30, START = [0.1, 0.5], EPS = 0.1;
const MV = [[0, 1], [1, 0], [0, -1], [-1, 0]], cl = (v, a, b) => Math.max(a, Math.min(b, v));
const inCheese = (x, y) => Math.hypot(x - CH[0], y - CH[1]) < CR;
function move(x, y, m, R) { return [cl(x + MV[m][0] * STEP + gauss(R) * WOB, 0, XW), cl(y + MV[m][1] * STEP + gauss(R) * WOB, 0, 1)]; }

// ───── tiny networks 2 → H (tanh) → 4. Input: the spot, scaled to −1 … 1
const K = 4, HT = 16, HP = 32;
function net(nH, R, s1, b1, s2) { const n = { nH, W1: new Float64Array(nH * 2), B1: new Float64Array(nH), W2: new Float64Array(K * nH), B2: new Float64Array(K), h: new Float64Array(nH), o: new Float64Array(K) };
  for (let i = 0; i < n.W1.length; i++) n.W1[i] = (R() * 2 - 1) * s1; for (let i = 0; i < nH; i++) n.B1[i] = (R() * 2 - 1) * b1; for (let i = 0; i < n.W2.length; i++) n.W2[i] = (R() * 2 - 1) * s2; return n; }
const copyNet = n => ({ nH: n.nH, W1: n.W1.slice(), B1: n.B1.slice(), W2: n.W2.slice(), B2: n.B2.slice(), h: new Float64Array(n.nH), o: new Float64Array(K) });
function fwd(n, x, y) { const a = x / 2 - 1, b = 2 * y - 1, H = n.nH; for (let j = 0; j < H; j++) n.h[j] = Math.tanh(n.W1[j * 2] * a + n.W1[j * 2 + 1] * b + n.B1[j]);
  for (let k = 0; k < K; k++) { let s = n.B2[k]; for (let j = 0; j < H; j++) s += n.W2[k * H + j] * n.h[j]; n.o[k] = s; } return n.o; }
const TARGET = net(HT, rng(1), 2, 2, .25);                  // the fixed random network: never trained (seed 1: before any lesson, the copier is not more wrong on the cheese side)
const TOUT = new Float64Array(K);
function errAt(P, x, y) { const t = fwd(TARGET, x, y); for (let k = 0; k < K; k++) TOUT[k] = t[k]; const p = fwd(P, x, y); let e = 0; for (let k = 0; k < K; k++) e += (p[k] - TOUT[k]) ** 2; return e / K; }
// one copying lesson at (x, y): a gradient step on the squared copying error
function lesson(P, x, y, lr = .05) { errAt(P, x, y); const H = P.nH, a = x / 2 - 1, b = 2 * y - 1, g = [0, 0, 0, 0]; for (let k = 0; k < K; k++) g[k] = (P.o[k] - TOUT[k]) * 2 / K;
  for (let j = 0; j < H; j++) { let gh = 0; for (let k = 0; k < K; k++) gh += g[k] * P.W2[k * H + j]; gh *= 1 - P.h[j] ** 2;
    for (let k = 0; k < K; k++) P.W2[k * H + j] -= lr * g[k] * P.h[j]; P.W1[j * 2] -= lr * gh * a; P.W1[j * 2 + 1] -= lr * gh * b; P.B1[j] -= lr * gh; }
  for (let k = 0; k < K; k++) P.B2[k] -= lr * g[k]; }
const newCopier = seed => net(HP, rng(seed), .5, .5, .05);

// ───── walkers. 'rand': every move at random. 'rnd': peek at the 4 landing spots (0.15 away, no wobble) and take the one with the
// biggest copying error, except 1 move in 10 at random; after each move, one lesson at the new spot. Walks of 40 moves, then back to start.
function walkers(mode, seed, nw = NW, stopAtCheese = false) {
  const R = rng(seed), P = mode === 'rnd' ? newCopier(seed + 5) : null, walks = [], snaps = [];
  let first = -1;
  for (let w = 0; w < nw; w++) { let x = START[0], y = START[1]; const pts = [[x, y]];
    for (let t = 0; t < TL; t++) { let m;
      if (mode === 'rnd' && R() >= EPS) { let b = -1; for (let k = 0; k < 4; k++) { const e = errAt(P, cl(x + MV[k][0] * STEP, 0, XW), cl(y + MV[k][1] * STEP, 0, 1)); if (e > b) { b = e; m = k; } } }
      else m = Math.floor(R() * 4);
      [x, y] = move(x, y, m, R); pts.push([x, y]); if (P) lesson(P, x, y);
      if (inCheese(x, y)) { if (first < 0) first = w + 1; break; } }
    walks.push(pts); if (P) snaps.push(copyNet(P)); if (stopAtCheese && first > 0) break; }
  return { walks, snaps, first, far: walks.map(p => Math.max(...p.map(q => q[0]))) };
}
// ───── Go-Explore: an archive of cells (squares 0.25 on a side). Each round: pick a cell (weight 1 / √(times seen + 1)), restore Pip
// to the saved spot of that cell, then 40 random moves. A new cell is saved with its spot; a known cell keeps the spot reached in the fewest moves.
const cellOf = (x, y) => Math.min(15, Math.floor(x / .25)) * 4 + Math.min(3, Math.floor(y / .25));
function goExplore(seed, nr = 60) {
  const R = rng(seed), A = new Map(), rounds = []; A.set(cellOf(...START), { x: START[0], y: START[1], len: 0, seen: 1, born: 0 }); let first = -1;
  for (let r = 0; r < nr && first < 0; r++) {
    const keys = [...A.keys()], wts = keys.map(k => 1 / Math.sqrt(A.get(k).seen + 1)); let u = R() * wts.reduce((s, v) => s + v, 0), i = 0; while (u > wts[i]) { u -= wts[i]; i++; }
    const pick = keys[i], c = A.get(pick); let x = c.x, y = c.y, len = c.len; const pts = [[x, y]], added = [];
    for (let t = 0; t < TL; t++) { [x, y] = move(x, y, Math.floor(R() * 4), R); len++; pts.push([x, y]); const k = cellOf(x, y), e = A.get(k);
      if (!e) { A.set(k, { x, y, len, seen: 1, born: r + 1 }); added.push(k); } else { e.seen++; if (len < e.len) { e.len = len; e.x = x; e.y = y; } }
      if (inCheese(x, y)) { first = r + 1; break; } }
    rounds.push({ pick, from: [c.x, c.y], pts, added, size: A.size, far: Math.max(...[...A.values()].map(v => v.x)) });
  }
  return { rounds, first, A };
}
// ───── story runs (seed 33, near the median for Go-Explore; RND found the cheese a little sooner than its median) and averages over 20 seeds (100–119)
const SS = 33, SEEDS = 20;
const RAND = walkers('rand', SS), RNDW = walkers('rnd', SS), GE = goExplore(SS);
const ALLR = RAND.walks.flat(), MOVED = RAND.walks.flatMap(p => p.slice(1)), REPEATS = (() => { const s = new Set(MOVED.map(p => p[0] + ',' + p[1])); return MOVED.length - s.size; })();
const RANDFAR = Math.max(...RAND.far);
// the copier taught on the random walker's spots (3 passes over all 30 walks)
const TAUGHT = (() => { const P = newCopier(SS + 5), P0 = copyNet(P); for (let pass = 0; pass < 3; pass++) for (const p of ALLR) lesson(P, p[0], p[1]); return { P, P0 }; })();
const STRIPE = errAt(TAUGHT.P0, 2, .5), EDGE = errAt(TAUGHT.P0, 2, .94), errL = errAt(TAUGHT.P, .5, .5), errR = errAt(TAUGHT.P, 3, .5), errL0 = errAt(TAUGHT.P0, .5, .5), errR0 = errAt(TAUGHT.P0, 3, .5);
const AVG = (() => { const fr = [], fn = [], fg = [], farR = new Float64Array(NW), farN = new Float64Array(NW);
  for (let i = 0; i < SEEDS; i++) { const s = 100 + i, a = walkers('rand', s), b = walkers('rnd', s), g = goExplore(s);
    fr.push(a.first < 0 ? Infinity : a.first); fn.push(b.first < 0 ? Infinity : b.first); fg.push(g.first < 0 ? Infinity : g.first);
    let ma = 0, mb = 0; for (let w = 0; w < NW; w++) { ma = Math.max(ma, a.far[w]); mb = Math.max(mb, b.far[w]); farR[w] += ma / SEEDS; farN[w] += mb / SEEDS; } }
  const srt = a => a.slice().sort((p, q) => p - q), med = a => { const b = srt(a); return (b[9] + b[10]) / 2; };
  return { randHit: fr.filter(isFinite).length, rndHit: fn.filter(isFinite).length, geHit: fg.filter(isFinite).length, rndMed: med(fn), geMed: med(fg), farR, farN };
})();
const f1 = v => (+v.toFixed(1)).toString(), f2 = v => (+v.toFixed(2)).toString(), f3 = v => (+v.toFixed(3)).toString(), r0 = v => String(Math.round(v));
const sim = { STRIPE, EDGE, MOVED, RAND, RNDW, GE, REPEATS, RANDFAR, errL, errR, errL0, errR0, AVG, TARGET, errAt, walkers, goExplore };

// ───── drawing: the long meadow strip
const MX = 20, MWD = 360, U = MWD / XW;
function strip(o = {}) {
  const y0 = o.y0 ?? 70, h = U, sx = x => MX + x * U, sy = y => y0 + (1 - y) * h, p = o.draw ?? 1;
  box(MX, y0, MWD * p, h, '#eef3dc', .6);
  if (o.heat) { const n = 32, m = 8, dw = MWD / n, dh = h / m; for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { const v = o.heat((i + .5) * XW / n, (j + .5) / m); box(MX + i * dw, y0 + h - (j + 1) * dh, dw + .3, dh + .3, o.hcol || C.orange, (o.heatA ?? 1) * lerp(.04, .85, clamp(v / (o.vmax || 1)))); } }
  if (o.cells) o.cells.forEach(([k, a, col]) => { const cx = Math.floor(k / 4), cy = k % 4; box(MX + cx * .25 * U, y0 + h - (cy + 1) * .25 * U, .25 * U, .25 * U, col || C.teal, a); });
  if (o.grid) alpha(o.grid, () => { for (let k = 1; k < 16; k++) rline(sx(k * .25), y0, sx(k * .25), y0 + h, 900 + k, { c: C.line, w: .7, once: true, j: .2 }); for (let k = 1; k < 4; k++) rline(MX, sy(k * .25), MX + MWD, sy(k * .25), 930 + k, { c: C.line, w: .7, once: true, j: .2 }); });
  rrect(MX, y0, MWD, h, 940, { w: 1.8, p: p >= 1 ? 1 : clamp(p * 1.4) });
  if ((o.cheese ?? 1) > 0) alpha(o.cheese ?? 1, () => { ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = '#c99a3a'; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.arc(sx(CH[0]), sy(CH[1]), CR * U, 0, 7); ctx.stroke(); ctx.restore(); L.cheese(sx(CH[0]), sy(CH[1]) + 2, .6); });
  if (o.ticks) alpha(o.ticks, () => [0, 1, 2, 3, 4].forEach(v => txt(String(v), sx(v), y0 + h + 10, { s: 12, c: C.soft })));
  return { sx, sy, y0, h };
}
function dots(S, pts, n, col, a = 1, r = 1.8) { const m = Math.min(n ?? pts.length, pts.length); for (let i = 0; i < m; i++) dot(S.sx(pts[i][0]), S.sy(pts[i][1]), r, col, a); }
function pathLine(S, pts, n, col, a = .6) { const m = Math.min(n, pts.length); if (m < 2) return; ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = 1.3; ctx.beginPath();
  for (let i = 0; i < m; i++) { const x = S.sx(pts[i][0]), y = S.sy(pts[i][1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); }
function pipAt(S, x, y, t, face = 1, s = .55) { L.mouse(S.sx(x), S.sy(y) + 2, face, s, t); }
// error map cache per copier
const HM = new Map();
function errMap(P) { if (HM.has(P)) return HM.get(P); const n = 32, m = 8, out = new Float64Array(n * m); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out[i * m + j] = errAt(P, (i + .5) * XW / n, (j + .5) / m);
  const f = (x, y) => out[Math.min(n - 1, Math.floor(x / XW * n)) * m + Math.min(m - 1, Math.floor(y * m))]; HM.set(P, f); return f; }
const VMAX = (() => { const f = errMap(TAUGHT.P0); let s = 0; for (let i = 0; i < 32; i++) for (let j = 0; j < 8; j++) s += f((i + .5) / 8, (j + .5) / 8); return s / 256; })();   // colour scale: the untaught copier's average error
// replay walks: walk index and move index at time t (dur seconds per move)
function walkAt(walks, t, dur) { let k = Math.floor(Math.max(0, t) / dur); for (let w = 0; w < walks.length; w++) { const n = walks[w].length - 1; if (k <= n) return { w, k }; k -= n + 1; } const w = walks.length - 1; return { w, k: walks[w].length - 1, done: true }; }

// the square meadow for the recall step (a stroll of 50 random moves with the meadow's rules, cheese and cat ignored)
const STROLL = (() => { const R = rng(8), out = [[.1, .1]]; let x = .1, y = .1; for (let i = 0; i < 50; i++) { const m = Math.floor(R() * 4); x = cl(x + MV[m][0] * STEP + gauss(R) * WOB, 0, 1); y = cl(y + MV[m][1] * STEP + gauss(R) * WOB, 0, 1); out.push([x, y]); } return out; })();
function square(t) { const X0 = 110, Y0 = 26, S = 200, px = x => X0 + x * S, py = y => Y0 + (1 - y) * S;
  box(X0, Y0, S, S, '#eef3dc', .6); rrect(X0, Y0, S, S, 950, { w: 1.8 });
  L.cheese(px(.85), py(.85) + 2, .6); L.cat(px(.6), py(.6) + 3, t, .6);
  const n = Math.max(1, Math.floor(clamp((t - 1) / 5) * STROLL.length)); for (let i = 0; i < n; i++) dot(px(STROLL[i][0]), py(STROLL[i][1]), 2.2, C.teal, .7);
  L.mouse(px(STROLL[n - 1][0]), py(STROLL[n - 1][1]) + 2, 1, .55, t);
  txt('x', X0 + S / 2, Y0 + S + 14, { s: 13, c: C.soft }); txt('y', X0 - 12, Y0 + S / 2, { s: 13, c: C.soft }); }
// chart: farthest x reached so far (average of 20 runs) against walk number
const CHT = { x0: 60, x1: 360, y0: 250, y1: 50 }, chX = w => CHT.x0 + (w - 1) / (NW - 1) * (CHT.x1 - CHT.x0), chY = v => CHT.y0 - v / XW * (CHT.y0 - CHT.y1);
function axes(a = 1) { alpha(a, () => { rline(CHT.x0, CHT.y0, CHT.x1 + 6, CHT.y0, 601, { w: 1.2 }); rline(CHT.x0, CHT.y0, CHT.x0, CHT.y1 - 8, 602, { w: 1.2 });
  [0, 1, 2, 3, 4].forEach(v => { txt(String(v), CHT.x0 - 8, chY(v), { s: 12, c: C.soft, al: 'right' }); if (v) rline(CHT.x0, chY(v), CHT.x1, chY(v), 610 + v, { c: '#e9dfd0', w: 1, once: true, j: .3 }); });
  [1, 10, 20, 30].forEach(v => txt(String(v), chX(v), CHT.y0 + 12, { s: 12, c: C.soft }));
  txt('walk number', CHT.x1 - 30, CHT.y0 + 27, { s: 13, c: C.soft }); txt('farthest x reached so far', CHT.x0 - 4, CHT.y1 - 22, { s: 13, c: C.soft, al: 'left' });
  ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = '#c99a3a'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(CHT.x0, chY(CH[0] - CR)); ctx.lineTo(CHT.x1, chY(CH[0] - CR)); ctx.stroke(); ctx.restore();
  txt('cheese edge', CHT.x0 + 6, chY(CH[0] - CR) - 9, { s: 12, c: '#b07d12', al: 'left' }); }); }
function curve(A, col, p, label, lx, ly) { const n = Math.floor(clamp(p) * NW); if (n < 1) return; ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.lineJoin = 'round'; ctx.beginPath();
  for (let w = 1; w <= n; w++) { const x = chX(w), y = chY(A[w - 1]); w > 1 ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); if (p >= 1 && label) txt(label, lx, ly, { s: 14, c: col, al: 'left' }); }
// little network icon
function netIcon(x, y, label, col, sub) { rrect(x - 52, y - 26, 104, 52, 960 + Math.round(x), { fill: C.card, w: 1.3 }); for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) dot(x - 30 + i * 30, y - 15 + j * 10, 2.6, col, .8);
  for (let j = 0; j < 4; j++) rline(x - 30, y - 15 + j * 10, x, y - 15 + ((j + 1) % 4) * 10, 970 + j + Math.round(x), { c: col, w: .8, once: true, j: .3 });
  txt(label, x, y + 40, { s: 15, c: col }); if (sub) txt(sub, x, y + 58, { s: 12, c: C.soft }); }
const LESS = (() => { const P = copyNet(TAUGHT.P0), out = []; for (let i = 0; i < 6; i++) { out.push(errAt(P, 2, .5)); lesson(P, 2, .5); } return out; })();
const MIN_MOVES = Math.ceil((CH[0] - CR - START[0]) / STEP);
const ORANGE = '#b8602c', TEALI = '#3f8582', GOLDI = '#8a5d08', ROSEI = '#a54444';
const GER = Math.max(3, GE.rounds.findIndex((r, i) => i >= 3 && r.from[0] > .6)), GEF = GE.first, GEC = GE.rounds[GE.rounds.length - 1].size;

// your turn: tap to give the copier lessons around a spot (starts from the untaught copier)
function yourTap(x, y, st, t) { if (y < 100 || y > 100 + U || x < MX || x > MX + MWD) { if (y > 300) { st.P = null; st.n = 0; st.ver = (st.ver || 0) + 1; } return; }
  const mx = (x - MX) / U, my = 1 - (y - 100) / U; if (!st.P) { st.P = copyNet(TAUGHT.P0); st.n = 0; }
  const R = rng(1000 + (st.n || 0)); for (let i = 0; i < 30; i++) lesson(st.P, cl(mx + gauss(R) * .08, 0, XW), cl(my + gauss(R) * .08, 0, 1));
  st.n = (st.n || 0) + 1; st.last = [mx, my, t]; st.ver = (st.ver || 0) + 1; st.map = null; }

const STEPS = [
 { h: 'Back in the meadow', p: ['Remember the meadow from the function approximation reel? Pip’s position there is two numbers, x and y. Each move goes 0.15 in its direction, plus a little random wobble.', 'Because of the wobble, Pip practically never stands on exactly the same spot twice. Keep that in mind: it is about to break the counting trick from the curiosity reel.'],
   draw(t) { square(t); if (t > 7) card(['position: two numbers, x and y', { s: 'each move: 0.15, plus a small random wobble', c: C.soft }], t - 7, { gap: 1.1 }); }, shot: [4, 10] },
 { h: 'A long meadow', p: [`Pip’s new meadow is 4 long and 1 wide. Pip starts at the left. The cheese circle sits at the far right end: **+1 on the move that lands inside it**, and the walk ends. There is no cat.`, `Each walk is cut off after 40 moves, and Pip is put back at the start. Even walking straight right, the cheese takes at least ${MIN_MOVES} moves.`],
   draw(t) { const S = strip({ y0: 100, draw: clamp(t / 1.5), ticks: clamp(t - 1.5), cheese: clamp(t - 2) }); pipAt(S, START[0], START[1], t);
     txt('x', 200, 214, { s: 13, c: C.soft, a: clamp(t - 1.5) }); txt('start', S.sx(START[0]) + 4, S.y0 - 12, { s: 13, c: C.soft, a: clamp(t - 2) }); txt('+1 on arrival', S.sx(CH[0]) - 20, S.y0 - 12, { s: 13, c: '#b07d12', a: clamp(t - 2.5) });
     if (t > 5.5) card(['walk: at most 40 moves, then back to start', { s: `cheese: at least ${MIN_MOVES} moves away`, c: C.soft }], t - 5.5, { gap: 1.1 }); }, shot: [4, 10] },
 { h: 'Random wandering', p: [`First, a Pip that picks every move at random, for 30 walks. Each walk is drawn as its own trail of dots.`, `The farthest it ever gets is x = ${f2(RANDFAR)}, not even halfway. It never reaches the cheese. Random steps cancel out, as they did in the maze.`],
   draw(t) { const S = strip({ y0: 100, ticks: 1 }); const nw = Math.min(NW, Math.floor(Math.max(0, t - .5) * 3) + 1);
     for (let w = 0; w < nw; w++) dots(S, RAND.walks[w], null, C.teal, w === nw - 1 ? .9 : .35);
     const lw = RAND.walks[nw - 1], q = lw[lw.length - 1]; pipAt(S, q[0], q[1], t);
     caption(`walk ${nw} of ${NW}`); if (t > 10.5) card([`farthest point: x = ${f2(RANDFAR)}`, { s: 'cheese reached: never', c: ROSEI }], t - 10.5, { gap: 1.1 }); }, shot: [5, 13] },
 { h: 'Nothing to count', p: [`Those walks landed on ${MOVED.length} spots. Only ${REPEATS} were exact repeats, all in the two left corners, where the fence squeezes Pip. So a count bonus would pay the first-visit 0.1 almost everywhere, even a hair away from a spot Pip has crossed many times.`, 'Counting only works when places repeat. Here, "near" never counts as "the same".'],
   draw(t) { const S = strip({ y0: 100, ticks: 1 }); RAND.walks.forEach(w => dots(S, w, null, C.teal, .35));
     const a = clamp(t - 3); if (a > 0) alpha(a, () => { const zx = 200, zy = 48; rcircle(zx, zy + 10, 30, 980, { fill: C.card }); dot(zx - 8, zy + 8, 4, C.teal); dot(zx + 8, zy + 13, 4, C.teal);
       txt('two spots a hair apart:', zx + 38, zy - 2, { s: 13, c: TEALI, al: 'left' }); txt('each counted once', zx + 38, zy + 16, { s: 13, c: TEALI, al: 'left' }); rline(S.sx(.9), S.sy(.5), zx - 30, zy + 20, 981, { c: C.line, w: 1 }); });
     if (t > 6) card([`${MOVED.length} spots, ${REPEATS} exact repeats`, { s: 'almost every count is 1: no signal', c: ROSEI }], t - 6, { gap: 1.1 }); }, shot: [4, 10] },
 { h: 'What we want instead', p: ['We want a sense of familiarity that **spreads to nearby spots**: if Pip has been around here a lot, here should feel boring, even if this exact spot is new.', 'Networks do exactly that kind of spreading. In the function approximation reel, a nudge at one spot also moved the guesses at nearby spots.'],
   draw(t) { const S = strip({ y0: 100, ticks: 1, heat: x => clamp((x - 1) / 1.8), heatA: clamp(t - 2), vmax: 1, hcol: C.orange }); RAND.walks.forEach(w => dots(S, w, null, C.teal, .35));
     if (t > 2) { txt('familiar', S.sx(.7), S.y0 - 12, { s: 14, c: TEALI, a: clamp(t - 2) }); txt('new', S.sx(3.3), S.y0 - 12, { s: 14, c: ORANGE, a: clamp(t - 2.6) }); }
     if (t > 5) card([{ s: 'orange: feels new (a wish, not yet a method)', c: ORANGE }, { s: 'familiarity should fade smoothly with distance', c: C.soft }], t - 5, { gap: 1.1 }); }, shot: [4, 10] },
 { h: 'A random network', p: ['Burda and colleagues proposed a neat trick in 2018. Take a small network, fill its knobs with random numbers, and **never train it**. Feed it a spot, and out come 4 numbers that mean nothing at all.', 'Call it the **fixed random network**. (Not the target network of the DQN reels: that one is a copy of the Q-network.) The map shows its first number across the meadow: gold positive, rose negative. Meaningless, but smooth and fixed.'],
   draw(t) { const S = strip({ y0: 120, ticks: 1, cheese: .4 }); const a = clamp(t - 3);
     if (a > 0) { const n = 32, m = 8, dw = MWD / n, dh = U / m; for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { const v = fwd(TARGET, (i + .5) * XW / n, (j + .5) / m)[0]; box(MX + i * dw, S.y0 + U - (j + 1) * dh, dw + .3, dh + .3, v >= 0 ? C.gold : C.rose, a * lerp(.05, .8, clamp(Math.abs(v) / .5))); } rrect(MX, S.y0, MWD, U, 940, { w: 1.8 }); }
     netIcon(200, 46, '', C.ink); txt('random knobs, never trained', 200, 92, { s: 13, c: C.soft, a: clamp(t - 1) });
     if (t > 7) card(['fixed random network: spot in, 4 numbers out', { s: 'map: its first number (gold +, rose −)', c: C.soft }], t - 7, { gap: 1.1 }); }, shot: [4, 11] },
 { h: 'The copier', p: ['A second network, the **copier**, tries to give the same 4 numbers. But it only gets lessons at spots Pip actually visits: after each move, one small gradient step that shrinks its copying error at that spot.', 'The **copying error** is the average squared difference between the copier’s 4 numbers and the fixed random network’s 4 numbers.'],
   draw(t) { const a1 = clamp(t - .5), a2 = clamp(t - 2), a3 = clamp(t - 4);
     alpha(a1, () => { txt('a spot (x, y)', 70, 140, { s: 15 }); arrow(118, 130, 150, 90, 990, { c: C.soft }); arrow(118, 150, 150, 190, 991, { c: C.soft }); netIcon(205, 80, 'fixed random network', C.ink); });
     alpha(a2, () => { netIcon(205, 200, 'copier (learns)', TEALI); });
     alpha(a3, () => { txt('4 numbers', 310, 80, { s: 14 }); txt('4 numbers', 310, 200, { s: 14, c: TEALI }); arrow(310, 96, 330, 128, 992, { c: ORANGE }); arrow(310, 184, 330, 152, 993, { c: ORANGE }); txt('error', 352, 140, { s: 16, c: ORANGE }); });
     if (t > 6.5) card(['copying error = average of (copy − random net)²', { s: 'lessons only where Pip has been', c: C.soft }], t - 6.5, { gap: 1.1 }); }, shot: [5, 10] },
 { h: 'Error marks the new', p: [`We give the copier lessons at the random walker’s ${MOVED.length} spots, three times over. Before: error about ${f2(errL0)} on the left, ${f2(errR0)} at x = 3. After: ${f3(errL)} on the left, where Pip walked, and ${f3(errR)} at x = 3, where it never went.`, 'Both maps use the same colour scale: orange marks big error. Where Pip has been, the copy is good; far away, poorer. So **copying error works as a novelty bonus**. That is **Random Network Distillation**, or RND.'],
   draw(t) { const u = t < 4 ? TAUGHT.P0 : TAUGHT.P, S = strip({ y0: 100, ticks: 1, heat: errMap(u), vmax: VMAX, cheese: .5 }); RAND.walks.forEach(w => dots(S, w, null, C.teal, .3));
     caption(t < 4 ? 'before the lessons: orange = big copying error' : 'after the lessons: orange = big copying error');
     if (t > 7) card([`before: ${f2(errL0)} left, ${f2(errR0)} at x = 3`, `after: ${f3(errL)} left, ${f3(errR)} at x = 3`], t - 7, { gap: 1.1, y: 260 }); }, shot: [3, 11] },
 { h: 'Familiar by accident', p: [`Look again at the copier before any lessons: a pale band runs along the middle, where Pip has never been. At x = 2 its error there is ${f3(STRIPE)}, against ${f3(EDGE)} near the top fence. By chance, the untrained copier already roughly matches the random network there.`, `Spreading cuts both ways: lessons on the left also lowered the error at x = 3, from ${f2(errR0)} to ${f3(errR)}. Such accidents can mislead Pip a little, until lessons sort them out.`],
   draw(t) { const S = strip({ y0: 100, ticks: 1, heat: errMap(TAUGHT.P0), vmax: VMAX, cheese: .5 }); const a = clamp(t - 2);
     alpha(a, () => { rcircle(S.sx(2), S.sy(.5), 9, 1500, { c: TEALI, w: 1.6 }); txt(f3(STRIPE), S.sx(2) + 14, S.sy(.5), { s: 13, c: TEALI, al: 'left' }); rcircle(S.sx(2), S.sy(.94), 7, 1501, { c: ORANGE, w: 1.6 }); txt(f3(EDGE), S.sx(2) + 14, S.y0 - 10, { s: 13, c: ORANGE, al: 'left' }); });
     caption('the untaught copier: no lessons anywhere yet');
     if (t > 6) card(['pale band: low error by chance, not by visits', { s: 'networks spread what they learn, and their accidents', c: C.soft }], t - 6, { gap: 1.1, y: 250 }); }, shot: [4, 10] },
 { h: 'Why a random network?', p: ['Any fixed pattern would do; random knobs are just the easiest. It **never changes** and looks only at what Pip sees **now**, so lessons can learn it: the bars show a fresh copier’s error at x = 2 falling.', 'Next-view prediction must guess the future, and a random future, like the noisy TV, can never be guessed. RND never predicts the future. But static that is new each time is still a new input: less fooled, not immune.'],
   draw(t) { netIcon(130, 80, 'random net', C.ink, 'same view, same numbers'); netIcon(290, 80, 'copier', TEALI, 'gets better with lessons');
     if (t > 3) { const k = Math.min(6, Math.floor((t - 3) * 1.5) + 1), mx = LESS[0]; for (let i = 0; i < k; i++) { const h = LESS[i] / mx * 80, x = 112 + i * 34; box(x, 270 - h, 24, h, C.orange, .8); txt(f3(LESS[i]), x + 12, 262 - h, { s: 12, c: ORANGE }); txt(String(i), x + 12, 280, { s: 12, c: C.soft }); }
       txt('fresh copier: error at x = 2 after 0, 1, 2 … lessons there', 200, 298, { s: 13, c: C.soft }); } }, shot: [4, 9] },
 { h: 'Pip peeks', p: ['Pip knows roughly where each move takes it, but not where the cheese is. Before each move it checks the copying error at the four landing spots and takes the biggest, except 1 move in 10 at random. Then the copier gets one lesson at the new spot.', 'This toy Pip only explores and does not learn from the cheese. Real RND adds the bonus to the game reward and trains a policy on the sum.'],
   draw(t) { const S = strip({ y0: 100, ticks: 1, heat: errMap(TAUGHT.P), vmax: VMAX, cheese: .5 }); const x = 1.8, y = .5; pipAt(S, x, y, t);
     if (t > 2) MV.forEach(([dx, dy], k) => { const e = errAt(TAUGHT.P, x + dx * STEP, y + dy * STEP), a = clamp((t - 2 - k * .5) * 2); alpha(a, () => { arrow(S.sx(x) + dx * 12, S.sy(y) - dy * 12, S.sx(x + dx * STEP * 1.6), S.sy(y + dy * STEP * 1.9), 1000 + k, { c: ORANGE, w: 1.4 }); });
       txt(f3(e), S.sx(x + dx * .5) + (dy ? 26 : 0), S.sy(y + dy * .3) + (dy ? 0 : -12), { s: 13, c: ORANGE, a }); });
     if (t > 6) card(['peek: error at each landing spot', { s: 'take the biggest (1 move in 10: random)', c: C.soft }], t - 6, { gap: 1.1 }); }, shot: [5, 10] },
 { h: 'The RND explorer', p: [`Same start, fresh copier. The map shows the copier’s error at the start of each walk. As places become familiar, the orange retreats, and Pip’s peeks lead it into fresher grass.`, `In walk 1 it gets to x = ${f2(RNDW.far[0])}. It first lands in the cheese in walk ${RNDW.first}. Note that the bonus knows nothing about cheese: Pip finds it only by going where it has not been.`],
   draw(t) { const S0 = { y0: 100 }, p = walkAt(RNDW.walks.slice(0, RNDW.first), t - .5, .022), P = p.w ? RNDW.snaps[p.w - 1] : newCopierCached();
     const S = strip({ y0: 100, ticks: 1, heat: errMap(P), vmax: VMAX }); const pts = RNDW.walks[p.w]; dots(S, pts, p.k + 1, C.teal, .8); const q = pts[p.k]; pipAt(S, q[0], q[1], t);
     caption(p.done ? `walk ${p.w + 1}: cheese!` : `walk ${p.w + 1} · move ${p.k}`);
     if (t > 13) card([`walk 1: x = ${f2(RNDW.far[0])} at best`, { s: `cheese first reached in walk ${RNDW.first}`, c: TEALI }], t - 13, { gap: 1.1, y: 250 }); }, shot: [4, 14] },
 { h: 'A chart of reach', p: [`To compare fairly, we run ${SEEDS} different seeds of 30 walks each. Across: the walk number. Up: the farthest x reached so far, averaged over the ${SEEDS} runs. The dashed line is the near edge of the cheese circle.`, 'A line that climbs means Pip keeps pushing into new ground.'],
   draw(t) { axes(clamp(t - .5)); if (t > 6) card(['across: walk 1 to 30', `up: farthest x so far, average of ${SEEDS} runs`], t - 6, { gap: 1.1 }); }, shot: [4, 9] },
 { h: 'Novelty pushes outward', p: [`Averages over ${SEEDS} runs, so they differ from the one run we watched. Gold, the random walker: it creeps to ${f2(AVG.farR[NW - 1])} after 30 walks. Teal, the RND explorer: ${f2(AVG.farN[0])} in walk 1 and ${f2(AVG.farN[NW - 1])} by walk 30, at the far end.`, `The bonus ignores the cheese, so hitting the circle, radius 0.12, is partly luck: RND landed in it in ${AVG.rndHit} of ${SEEDS} runs, the random walker in ${AVG.randHit}. A learning agent would then use that +1.`],
   draw(t) { axes(); curve(AVG.farR, GOLDI, (t - .5) / 3, 'random walker', chX(16), chY(AVG.farR[15]) + 18); curve(AVG.farN, TEALI, (t - 4) / 3, 'RND explorer', chX(16), chY(AVG.farN[15]) + 20);
     if (t > 8) card([`cheese within 30 walks: RND ${AVG.rndHit} of ${SEEDS} runs`, { s: `random walker: ${AVG.randHit} of ${SEEDS}`, c: C.soft }], t - 8, { gap: 1.1 }); }, shot: [4, 11] },
 { h: 'Montezuma’s Revenge', p: ['The famous hard case is an Atari game from 1984, Montezuma’s Revenge. An explorer climbs through a pyramid of rooms, 24 on the first level, with keys, doors, ladders and deadly falls. Points are rare, and each needs a long chain of exact moves.', 'In 2015 the DQN agent, which mastered many Atari games, scored 0 here. In 2018, RND became the first method to beat the average human score without human demonstrations.'],
   draw(t) { const rows = [1, 3, 5, 7]; rows.forEach((n, r) => { for (let i = 0; i < n; i++) { const a = clamp((t - .5 - r * .6) * 2); if (a <= 0) continue; const x = 200 + (i - (n - 1) / 2) * 44, y = 40 + r * 46; rrect(x - 20, y - 18, 40, 36, 1100 + r * 10 + i, { fill: C.card, w: 1.2, fa: a }); } });
     alpha(clamp(t - 3), () => { L.coin(156, 132, 6); txt('key', 156, 146, { s: 12, c: C.soft }); rrect(240, 118, 8, 20, 1150, { fill: '#c9a46b', w: 1 }); txt('door', 244, 148, { s: 12, c: C.soft }); L.mouse(200, 88, 1, .45, t); });
     txt('a sketch, not the real map', 200, 236, { s: 12, c: C.soft, a: clamp(t - 3) });
     if (t > 6) card(['rare points, long exact chains', { s: 'DQN (2015): 0 · RND (2018): above average human', c: C.soft }], t - 6, { gap: 1.1 }); }, shot: [4, 10] },
 { h: 'Forgetting the frontier', p: [`Even with a bonus, every walk starts back at the start. The RND explorer reached x = ${f2(RNDW.far[0])} in walk 1, but walk 2 began from scratch and got only to ${f2(RNDW.far[1])}. Our meadow has no fatal moves, so RND copes here. In a game where a few wrong moves end the try, a promising place may never be reached again.`, 'Ecoffet and colleagues turned this around with **Go-Explore**, first shown in 2019 and published in 2021: **first return, then explore**.'],
   draw(t) { const S = strip({ y0: 100, ticks: 1 }); const w = RNDW.walks; dots(S, w[0], null, C.teal, t > 4 ? .45 : .7); if (t > 4) dots(S, w[1], Math.floor(clamp((t - 4) / 2) * w[1].length), C.orange, .7); const far = w[0].reduce((b, p) => p[0] > b[0] ? p : b, w[0][0]);
     alpha(clamp(t - 2), () => { rcircle(S.sx(far[0]), S.sy(far[1]), 10, 1200, { c: GOLDI, w: 1.6 }); txt('reached once', S.sx(far[0]), S.y0 - 12, { s: 13, c: GOLDI }); });
     alpha(clamp(t - 4), () => { arrow(S.sx(far[0]) - 14, S.y0 + U + 22, S.sx(START[0]) + 14, S.y0 + U + 22, 1201, { c: ROSEI }); txt('next walk: back to the start', 200, 236, { s: 13, c: ROSEI }); });
     if (t > 6.5) card(['Go-Explore: remember good places,', { s: 'go back to one, then explore from there', c: C.soft }], t - 6.5, { gap: 1.1, y: 250 }); }, shot: [4, 10] },
 { h: 'How Go-Explore works', p: ['Pip lumps spots into coarse **cells**, squares 0.25 on a side, and keeps an **archive**: one saved spot per cell, the one reached in the fewest moves. Yes, that is counting again: coarse cells make places repeat, at the price of choosing the cell size by hand. RND needs no such choice.', 'Each round: pick a cell, favouring ones Pip has landed in rarely. Restore its saved spot, like loading a saved game. Then take 40 random moves.'],
   draw(t) { const r = GE.rounds[GER], A = [...GE.A.entries()].filter(([k, v]) => v.born <= GER);
     const S = strip({ y0: 100, ticks: 1, grid: clamp(t - .5), cells: t > 2 ? A.map(([k]) => [k, .3]) : [] }); if (t > 2) A.forEach(([k, v]) => dot(S.sx(v.x), S.sy(v.y), 2.6, TEALI, clamp(t - 2)));
     if (t > 4.5) { const c = GE.A.get(r.pick); rcircle(S.sx(r.from[0]), S.sy(r.from[1]), 9, 1210, { c: GOLDI, w: 1.8 }); txt('picked', S.sx(r.from[0]), S.y0 - 12, { s: 13, c: GOLDI }); }
     if (t > 6) { const n = Math.floor(clamp((t - 6) / 3) * r.pts.length); dots(S, r.pts, n, C.orange, .8); }
     if (t > 9.5) card(['pick weight = 1 ÷ √(landings in the cell + 1)', 'restore its spot → 40 random moves', { s: 'new cells join the archive', c: C.soft }], t - 9.5, { gap: 1.1, y: 236 }); }, shot: [5, 13] },
 { h: 'Go-Explore in the meadow', p: [`Round by round, the archive grows outward. Shaded squares are cells in the archive. In this run, Pip lands in the cheese in round ${GEF}, with ${GEC} cells saved.`, `Over ${SEEDS} runs it found the cheese within 60 rounds in ${AVG.geHit} of them, a median of ${r0(AVG.geMed)} rounds. Its moves are random: remembering and returning do the work. A round has the moves of one walk but never pays for the trip back, and a real robot usually cannot restore a saved spot.`],
   draw(t) { const n = Math.min(GE.rounds.length, Math.floor(Math.max(0, t - .5) * 3) + 1), r = GE.rounds[n - 1];
     const cells = [...GE.A.entries()].filter(([k, v]) => v.born <= n).map(([k, v]) => [k, v.born === n ? .7 : .3, v.born === n ? C.orange : C.teal]);
     const S = strip({ y0: 100, ticks: 1, grid: .5, cells }); dots(S, r.pts, null, C.orange, .7); const q = r.pts[r.pts.length - 1]; pipAt(S, q[0], q[1], t);
     caption(n >= GE.rounds.length ? `round ${n}: cheese! · ${r.size} cells saved` : `round ${n} · ${r.size} cells saved`);
     if (t > 12) card([`cheese in round ${GEF} (this run)`, { s: `median over ${SEEDS} runs: ${r0(AVG.geMed)} rounds`, c: C.soft }], t - 12, { gap: 1.1, y: 250 }); }, shot: [5, 13] },
 { h: 'Your turn', tap: yourTap, p: ['**Tap the meadow** to give the copier 30 lessons around that spot. Watch the orange, its copying error, fade there and a little around it.', 'Tap a few spots in a row to paint a familiar trail. Tap below the meadow to start over with an untaught copier.'],
   draw(t, st) { const P = st.P || TAUGHT.P0; if (!st.map || st.mapVer !== st.ver) { HM.delete(P); st.map = errMap(P); st.mapVer = st.ver; }
     const S = strip({ y0: 100, ticks: 1, heat: st.map, vmax: VMAX, cheese: .4 });
     if (st.last) { const [x, y, t0] = st.last; rcircle(S.sx(x), S.sy(y), 9, 1300, { c: TEALI, w: 1.6 }); txt(`error here now ${errAt(P, x, y).toFixed(3)}`, 200, 244, { s: 16, c: ORANGE }); }
     txt(`lessons given: ${30 * (st.n || 0)}`, 200, 272, { s: 15, c: C.soft }); caption(st.n ? 'tap again, or tap down here to reset' : 'tap the meadow'); }, shot: [3] },
 { h: 'Where it worked', p: ['RND, from OpenAI in 2018, was the first to beat average human play on Montezuma’s Revenge without demonstrations. Go-Explore, from Uber AI Labs, scored far higher there and also did well on Pitfall, another very sparse game.', 'Go-Explore explored by restoring saved game states, then trained a network policy by imitating its own best routes, so it could follow them even when the game behaves a little randomly.'],
   draw(t) { const items = [['RND, 2018', 'copying error as a novelty bonus'], ['Go-Explore, 2019 and 2021', 'remember, return, then explore'], ['both', 'cracked Montezuma’s Revenge']];
     items.forEach(([a, b], i) => { const y = 70 + i * 70, al = clamp((t - .6 - i * 2) * 2); if (al <= 0) return; rrect(28, y - 26, 344, 54, 1400 + i, { fill: C.card, c: C.line, fa: al });
       txt(a, 44, y - 8, { s: 18, al: 'left', a: al }); txt(b, 44, y + 13, { s: 15, al: 'left', c: C.soft, a: al }); });
     if (t > 7) card(['our meadow is a toy:', { s: 'real agents use deep networks on game screens', c: C.soft }], t - 7, { gap: 1.1 }); }, shot: [4, 10] },
 { h: 'Recap', p: ['When places never repeat, **counting** fails. **RND** measures novelty as the error of a copier chasing a fixed random network: small where you have been, large elsewhere, and it spreads to nearby spots.', '**Go-Explore** keeps an archive of reached places, returns to one, and explores from there. Reward novelty, and remember the frontier.'],
   draw(t) { const S = strip({ y0: 60, grid: .5, cells: [...GE.A.keys()].map(k => [k, .3]) }); txt('Go-Explore’s archive at the end of its run', 200, 168, { s: 13, c: C.soft });
     card(['places never repeat: counts fail', 'RND: bonus = copying error of a random net', 'Go-Explore: remember, return, explore', { s: 'reward novelty, and remember the frontier', c: C.soft }], t - .3, { gap: 1, y: 200 }); }, shot: [6] },
];
let NC0 = null; function newCopierCached() { return NC0 || (NC0 = newCopier(SS + 5)); }

RLStory.register('rnd', { title: 'RND & Go-Explore', track: 'Rewards', onCtx: c => { ctx = c; }, steps: STEPS, sim });
})();
