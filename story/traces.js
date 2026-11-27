// Eligibility traces & TD(λ) as a story: Pip learns the values of its arrows habit from walks, and a fading mark on
// recently left cells lets one surprise update the whole recent path. TD(0) (λ = 0) and Monte Carlo-like (λ = 1) are the
// two ends of the dial. All λ values learn from the very same walks (the habit is fixed). Numbers come from the runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, key, maze, trail, pip } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GAM = 0.9, EPS = 0.2, ALPHA = 0.2, SK = key(0, 3), K30 = key(3, 0);
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);
const ZERO = () => new Array(20).fill(0);
const f2 = v => { const s = (Math.abs(v) < .005 ? 0 : v).toFixed(2); return s.replace('-', '−'); };
const f3 = v => (Math.abs(v) < .0005 ? '0' : String(+v.toFixed(3))).replace('-', '−');

// ───── the arrows habit (values reel): follow the arrow, 1 move in 5 a random direction; bump = stay, still a step
const ARR = new Array(20).fill(-1);
[[0, 3, 0], [0, 2, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [2, 1, 0], [1, 2, 3], [1, 3, 1], [2, 3, 1], [3, 3, 1], [4, 3, 0], [4, 2, 0], [4, 1, 0], [3, 2, 1]]
  .forEach(([c, r, m]) => { ARR[key(c, r)] = m; });
function walk(c, r, eps, R) {
  const tr = [];
  for (let k = 0; k < 5000; k++) { let m, rnd = false; if (R() < eps) { rnd = true; m = Math.floor(R() * 4); } else m = ARR[key(c, r)];
    const [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr); tr.push({ c, r, nc, nr, m, rnd, rew, bump: nc === c && nr === r }); c = nc; r = nr; if (rew) break; }
  return tr;
}
function exactV() { const V = ZERO(); for (let it = 0; it < 3000; it++) CELLS.forEach(([c, r]) => { const k = key(c, r); let v = 0; for (let m = 0; m < 4; m++) { const p = EPS / 4 + (m === ARR[k] ? 1 - EPS : 0), [nc, nr] = G.step(c, r, m), rw = G.term(nc, nr); v += p * (rw + (rw ? 0 : GAM * V[key(nc, nr)])); } V[k] = v; }); return V; }
const EX = exactV();
// TD(λ), replacing marks: on leaving a cell its mark is set to 1; TD error δ = r + γ V(next) − V(left); every cell += α δ mark; marks × γλ
function tdl(walks, lam, keep, al = ALPHA) {
  const V = new Float64Array(20), hist = [], log = [];
  walks.forEach((tr, wi) => { const E = new Float64Array(20);
    tr.forEach(s => { const k = key(s.c, s.r), d = s.rew + GAM * (s.rew ? 0 : V[key(s.nc, s.nr)]) - V[k]; E[k] = 1;
      if (keep && wi < keep) log.push({ wi, d, E: Array.from(E), Vb: Array.from(V) });
      for (let i = 0; i < 20; i++) if (E[i]) { V[i] += al * d * E[i]; E[i] *= GAM * lam; } });
    hist.push(Array.from(V)); });
  return { hist, log };
}
// the story's walks: the same start walks as the values and Monte Carlo reels (rng(87000 + start key))
const NW = 200, WK = []; { const R = rng(87000 + SK); for (let i = 0; i < NW; i++) WK.push(walk(0, 3, EPS, R)); }
const LAMS = [0, .3, .5, .8, .9, 1];
const RUNS = {}; const RUN = l => RUNS[l] ||= tdl(WK, l, 2);
// fair race: 200 sets of 200 walks (set 0 = the story's walks), every λ learns from the same set; error = mean |V − true| over Pip's usual route
const ROUTE = [[0, 3], [0, 2], [0, 1], [0, 0], [1, 0], [2, 0], [3, 0]], RK = ROUTE.map(([c, r]) => key(c, r));
const err = V => RK.reduce((s, k) => s + Math.abs(V[k] - EX[k]), 0) / RK.length;
const NSETS = 200;
const ERR = (() => { const out = {}; LAMS.forEach(l => { out[l] = new Array(NW).fill(0); });
  for (let sd = 0; sd < NSETS; sd++) { const ws = sd === 0 ? WK : (() => { const R = rng(500000 + sd), a = []; for (let i = 0; i < NW; i++) a.push(walk(0, 3, EPS, R)); return a; })();
    LAMS.forEach(l => { const h = tdl(ws, l).hist; for (let i = 0; i < NW; i++) out[l][i] += err(h[i]) / NSETS; }); }
  return out; })();
// each λ with its own best α (grid 0.05 … 0.5), same 200 sets of walks. Precomputed (the scan takes ~20 s); alphaScan() recomputes it for the headless check.
const ALS = [.05, .1, .2, .3, .4, .5];
const ABEST = { 0: [.5, .102, .1, .045], .3: [.4, .090, .05, .040], .5: [.3, .086, .05, .037], .8: [.2, .077, .05, .038], .9: [.2, .073, .05, .041], 1: [.1, .078, .05, .046] };   // [α20, error20, α200, error200]
function alphaScan() { const sets = [WK]; for (let sd = 1; sd < NSETS; sd++) { const R = rng(500000 + sd), a = []; for (let i = 0; i < NW; i++) a.push(walk(0, 3, EPS, R)); sets.push(a); }
  const out = {}; LAMS.forEach(l => { let b20 = [0, 9], b200 = [0, 9]; ALS.forEach(al => { let e20 = 0, e200 = 0; sets.forEach(ws => { const h = tdl(ws, l, 0, al).hist; e20 += err(h[19]) / NSETS; e200 += err(h[199]) / NSETS; }); if (e20 < b20[1]) b20 = [al, e20]; if (e200 < b200[1]) b200 = [al, e200]; }); out[l] = [b20[0], +b20[1].toFixed(3), b200[0], +b200[1].toFixed(3)]; }); return out; }
const ABEST200 = LAMS.reduce((b, l) => ABEST[l][3] < ABEST[b][3] ? l : b, 0), ABEST20 = LAMS.reduce((b, l) => ABEST[l][1] < ABEST[b][1] ? l : b, 0);
const BEST20 = LAMS.reduce((b, l) => ERR[l][19] < ERR[b][19] ? l : b, 0), BEST200 = LAMS.reduce((b, l) => ERR[l][199] < ERR[b][199] ? l : b, 0);

// ───── story numbers
const w1 = WK[0], N1 = w1.length;
const after1 = l => RUN(l).hist[0];
const MARKS08 = (() => { const lg = RUN(.8).log.filter(q => q.wi === 0); return lg[lg.length - 1].E; })();   // marks at the cheese step, λ = 0.8
const TD10START = RUN(0).hist[9][SK], L08_10START = RUN(.8).hist[9][SK], L1_10START = RUN(1).hist[9][SK];
const MARK_LIFE = (() => { let k = 0, v = 1; while (v > .01) { v *= GAM * .8; k++; } return k; })();
const FIRSTSTART0 = RUN(0).hist.findIndex(V => Math.abs(V[SK]) > 1e-9) + 1;   // first walk after which TD(0)'s start guess is not 0

// ───── drawing helpers
const no = (c, r) => r * 5 + c + 1;
function nums(V, a = 1, only, fmt = f2) { for (const [c, r] of CELLS) { const k = key(c, r); if (only && !only(c, r)) continue; const v = V[k], aa = typeof a === 'function' ? a(c, r) : a; if (aa <= 0) continue;
  txt(fmt(v), OX + c * CS + CS - 5, OY + r * CS + 12, { s: 14, al: 'right', a: aa, c: v < -.004 ? C.roseInk : v > .004 ? C.goldInk : C.soft }); } }
function arrows(a = 1) { CELLS.forEach(([c, r]) => alpha(a, () => G.arrowIn(c, r, ARR[key(c, r)], 400 + key(c, r), { c: C.goldInk, w: 1.4 }))); }
function dim(a = .62) { box(0, 0, W, 285, C.bg, a); }
function cellMark(c, r, s, a = 1, o = {}) { box(cx(c) - 24, cy(r) + (o.dy ?? 4) - 10, 48, 20, C.card, .88 * a); txt(s, cx(c), cy(r) + (o.dy ?? 4), { s: o.s || 16, c: o.c || C.goldInk, a }); }
function walkAt(tr, t, dur) {
  const k = Math.floor(Math.max(0, t) / dur);
  if (k >= tr.length) { const l = tr[tr.length - 1]; return { x: cx(l.nc), y: cy(l.nr), k: tr.length, done: true, face: Math.sign(l.nc - l.c) || 0 }; }
  const s = tr[k], u = ease((Math.max(0, t) % dur) / dur * 1.4), [dx, dy] = G.MOVES[s.m];
  if (s.bump) { const b = Math.sin(clamp((Math.max(0, t) % dur) / dur * 1.6) * Math.PI) * 11; return { x: cx(s.c) + dx * b, y: cy(s.r) + dy * b, k, s, face: dx || 0 }; }
  return { x: lerp(cx(s.c), cx(s.nc), u), y: lerp(cy(s.r), cy(s.nr), u), k, s, face: dx || 0 };
}
// marks (traces) after k steps of a walk with this λ, replacing: cell left gets 1, all × γλ after each step
function marksAt(tr, k, lam) { const E = ZERO(); for (let i = 0; i < k; i++) { for (let j = 0; j < 20; j++) E[j] *= GAM * lam; E[key(tr[i].c, tr[i].r)] = 1; } return E; }
// a mark drawn as a teal bar at the bottom of the cell (full cell width = 1) with its number
function drawMarks(E, a = 1, o = {}) { CELLS.forEach(([c, r]) => { const v = E[key(c, r)]; if (v < .005) return; const x = OX + c * CS + 6, y = OY + r * CS + CS - 14, w = (CS - 12) * v;
  box(x, y, CS - 12, 9, '#e4d8c7', .7 * a); box(x, y, w, 9, C.teal, .9 * a); if (o.n !== false) txt(f2(v), cx(c), y - 9, { s: 13, c: '#2f6f6c', a }); }); }
// mini maze
function mini(Gm, V, label, t, col) { Gm.maze({ V, t }); CELLS.forEach(([c, r]) => { const v = V[key(c, r)]; if (Math.abs(v) < .0005) return; txt(f2(v), Gm.ox + c * Gm.cs + Gm.cs / 2, Gm.oy + r * Gm.cs + Gm.cs / 2 + 2, { s: 13, c: C.goldInk }); });
  txt(label, Gm.ox + Gm.cs * 2.5, Gm.oy - 14, { s: 15, c: col || C.ink }); }
const GA = L.gridWorld({ cs: 37, ox: 8, oy: 70 }), GB = L.gridWorld({ cs: 37, ox: 207, oy: 70 });
// error chart: across = walks 1..XM, up = error 0..YM
const COLS = { 0: '#3f8f8b', .3: '#9cc3a6', .5: '#c9b45a', .8: C.gold, .9: C.orange, 1: C.rose };
const CHX0 = { x0: 62, x1: 356, yt: 300, yb: 380 }, YM = .6;
function errChart(lams, n, XM, o = {}) {
  const CHX = o.g || CHX0; const X = i => CHX.x0 + i / (XM - 1) * (CHX.x1 - CHX.x0), Y = v => CHX.yb - clamp(v / YM) * (CHX.yb - CHX.yt), a = o.ax ?? 1;
  alpha(a, () => { rline(CHX.x0, CHX.yb, CHX.x1, CHX.yb, 960, { c: C.line, w: 1 }); rline(CHX.x0, CHX.yt, CHX.x0, CHX.yb, 961, { c: C.line, w: 1 });
    txt('0', CHX.x0 - 8, CHX.yb, { s: 12, c: C.soft, al: 'right' }); txt(String(YM), CHX.x0 - 8, CHX.yt, { s: 12, c: C.soft, al: 'right' });
    txt('error', CHX.x0 - 8, (CHX.yt + CHX.yb) / 2, { s: 12, c: C.soft, al: 'right' });
    txt('walk 1', CHX.x0, CHX.yb + 11, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${XM}`, CHX.x1, CHX.yb + 11, { s: 12, c: C.soft, al: 'right' }); });
  lams.forEach(l => { const e = ERR[l]; ctx.save(); ctx.strokeStyle = COLS[l]; ctx.lineWidth = o.hi === l ? 3 : 2; ctx.globalAlpha = o.hi != null && o.hi !== l ? .35 : 1; ctx.beginPath();
    for (let i = 0; i < Math.min(n, XM); i++) i ? ctx.lineTo(X(i), Y(e[i])) : ctx.moveTo(X(i), Y(e[i])); ctx.stroke(); ctx.restore();
    if (n >= XM && o.labels !== false) txt(`λ ${l}`, X(XM - 1) + 4, Y(e[XM - 1]) + (o.dy?.[l] ?? 0), { s: 13, c: COLS[l] === C.gold ? C.goldInk : COLS[l] === C.rose ? C.roseInk : COLS[l], al: 'left' }); });
}
// bars: error after 20 and after 200 walks for each λ
function errBars(t, sel, o = {}) {
  const x0 = 46, y0 = o.y0 ?? 384, h = o.h ?? 64, gw = 152, bw = 20, E = (l, i) => o.tuned ? ABEST[l][i === 19 ? 1 : 3] : ERR[l][i];
  const mx = Math.max(...LAMS.map(l => Math.max(ERR[l][19], ERR[l][199])));   // one shared scale for both panels
  [[19, 'after 20 walks', x0], [199, 'after 200 walks', x0 + 172]].forEach(([i, lab, gx], g) => {
    const best = LAMS.reduce((b, l) => E(l, i) < E(b, i) ? l : b, 0);
    txt(lab, gx + gw / 2, y0 - h - 26, { s: 14, c: C.soft }); rline(gx, y0, gx + gw, y0, 970 + g, { c: C.line, w: 1 });
    LAMS.forEach((l, j) => { const u = clamp((t - j * .15 - g * .8) * 2), v = E(l, i), bh = v / mx * h * u, x = gx + 4 + j * (bw + 4.8);
      box(x, y0 - bh, bw, bh, COLS[l], sel == null || sel === l ? .95 : .3); txt(String(l), x + bw / 2, y0 + 10, { s: 12, c: C.soft });
      if (u >= 1 && o.vals !== false) txt(v.toFixed(2).slice(1), x + bw / 2, y0 - bh - 8, { s: 12, c: l === best ? C.ink : C.soft });
      if (l === best && u >= 1) rrect(x - 2, y0 - bh - 1, bw + 4, bh + 1, 980 + g, { c: C.ink, w: 1.4 });
      if (o.tuned && u >= 1) txt('α ' + String(ABEST[l][i === 19 ? 0 : 2]).slice(1), x + bw / 2, y0 + 23, { s: 11, c: C.soft }); });
  });
  txt('λ', 28, y0 + 10, { s: 13, c: C.soft });
}
// n-step lookahead arcs from a cell along walk 1
function lookArc(i, n, a, col, off = 0) { const pts = [[w1[i].c, w1[i].r]]; for (let k = i; k < Math.min(N1, i + n); k++) pts.push([w1[k].nc, w1[k].nr]);
  ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.setLineDash([6, 4]); ctx.beginPath(); pts.forEach(([c, r], k) => { const x = cx(c) + off, y = cy(r) + off; k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.restore();
  const [c2, r2] = pts[pts.length - 1]; dot(cx(c2) + off, cy(r2) + off, 4.5, col, a); }

const LSL = [0, .3, .5, .8, .9, 1];
const STEPS = [
 { h: 'Two ways to pass the news', p: ['Pip follows its habit, one move in five random, learning each cell’s value. Cheese +1 and cat −1 come on stepping onto them, ending the walk. γ (gamma) is 0.9.', '**TD learning** updates every step from the next cell’s guess; **Monte Carlo** waits for the real return. The best of both? The trick also works for Q-values, in SARSA and Q-learning.'],
   shot: [4, 11],
   draw(t) { maze({ t, labels: clamp(t - .5) }); arrows(.35); const m = walkAt(w1, t - 1, .5); trail(w1.slice(0, Math.min(N1, m.k + 1)).map(s => [s.c, s.r]), .5); pip(m.x, m.y, m.face, t);
     if (m.done) bubble('+1', cx(4), cy(0) - 18, t - 1 - N1 * .5);
     if (t > 5) card(['TD: after every step, reward + 0.9 × next guess', 'Monte Carlo: after the walk, the real return', { s: 'both nudge by α = 0.2, a fifth of the way', c: C.soft }], t - 5, { gap: 1.2 }); } },
 { h: 'After walk 1', p: [`Walk 1: the familiar ${N1}-step trip to the cheese. With TD only the cell beside the cheese learns: 0 + 0.2 × 1 = 0.2. Other steps had nothing to pass on.`, `Fixed-fraction Monte Carlo moves every cell the same fifth of the way to its return: 0.2 beside the cheese, down to ${f2(after1(1)[SK])} at the start. One walk reached the whole path.`],
   shot: [4, 10],
   draw(t) { alpha(clamp(t - .3), () => mini(GA, after1(0), 'TD', t, '#3f8f8b')); alpha(clamp(t - 2.5), () => mini(GB, after1(1), 'Monte Carlo', t, C.roseInk));
     if (t > 5) card(['TD: 1 cell changed', `Monte Carlo: all ${new Set(w1.map(s => key(s.c, s.r))).size} cells on the path changed`, { s: 'but Monte Carlo had to wait for the end', c: C.soft }], t - 5, { gap: 1.2, y: 300 }); } },
 { h: 'Who deserves the credit?', p: [`Which earlier moves earned the cheese? This is the **credit assignment** problem. TD passes credit back about one cell per walk: here its guess for the start first moves on walk ${FIRSTSTART0}, and after 10 walks is only ${TD10START.toFixed(4)}.`, 'Monte Carlo credits every cell at once, but only after the walk, and one unlucky ending shakes the whole path. Is there a middle ground?'],
   shot: [4, 10],
   draw(t) { const n = clamp(Math.floor(t / .7) + 1, 1, 10), V = RUN(0).hist[n - 1]; maze({ V, t, hi: [[0, 3, C.teal, .25]] }); nums(V); arrows(.2); pip(cx(0), cy(3), 1, t);
     txt(`TD, after walk ${n}`, W / 2, 300, { s: 16, c: C.soft });
     if (t > 7) card([{ s: `start after 10 walks: TD ${TD10START.toFixed(4)}`, c: '#3f8f8b' }, { s: 'the news moves about one cell back per walk', c: C.soft }], t - 7, { y: 316, gap: 1.2 }); } },
 { h: 'Looking n steps ahead', p: ['TD’s target looks one step ahead; Monte Carlo looks to the end.', 'In between is the **n-step target**: the next n real rewards, discounted by 0.9 per step as in a return, then 0.9 multiplied n times × the guess where Pip is n steps later. If the walk ends sooner, stop there. One step is TD, the whole walk Monte Carlo.'],
   shot: [4, 11],
   draw(t) { maze({ t }); arrows(.15); pip(cx(0), cy(3), 1, t);
     [[1, '#3f8f8b', 1, -12], [3, C.orange, 3, 0], [N1, C.rose, 5, 12]].forEach(([n, col, t0, off]) => { const a = clamp(t - t0); if (a > 0) lookArc(0, n, a, col, off); });
     card([{ s: '1 step:  reward + 0.9 × guess one cell on   (TD)', c: '#3f8f8b' }, { s: '3 steps:  R + 0.9 R + 0.81 R + 0.9³ × guess', c: '#c4581f' }, { s: 'to the end:  the discounted return   (Monte Carlo)', c: C.roseInk }], t - 1, { gap: 2, fs: 16 }); } },
 { h: 'A fading memory', p: ['Smoother than waiting n steps: give each cell a **mark**, its **eligibility trace**, and set a dial **λ (lambda)** from 0 to 1.', 'Each step, every mark first shrinks, times 0.9 × λ; then the cell Pip just left gets mark 1. With λ = 0.8, after step 1 the start has 1; after step 2, 0.72, the cell above it 1.'],
   shot: [3.5, 9],
   draw(t) { const D = .9, m = walkAt(w1, t - .5, D); maze({ t }); arrows(.15); drawMarks(marksAt(w1, m.k, .8)); pip(m.x, m.y, m.face, t);
     caption(m.done ? 'teal bar = mark: 1 for the cell just left, fading behind' : `step ${Math.min(N1, m.k + 1)}`);
     if (t > N1 * D + 1.2) card(['each step: 1) every mark × 0.9 × λ', '                2) the cell just left: mark = 1', { s: 'λ = 0.8: × 0.72 per step', c: C.soft }], t - N1 * D - 1.2, { gap: 1 }); } },
 { h: 'One surprise, many cells', p: ['Each step Pip computes the TD error as usual: reward + 0.9 × next guess − old guess. Then **every** cell moves by α × TD error × its mark.', `On walk 1 every TD error is 0 until the cheese: 1. That single surprise updates the whole recent path: 0.2 beside the cheese, then ${f3(after1(.8)[key(2, 0)])}, ${f3(after1(.8)[key(1, 0)])}, and on, down to ${f3(after1(.8)[SK])} at the start.`],
   shot: [3, 11],
   draw(t) { const V = after1(.8), u = ease((t - 4) / 1.5); maze({ V: V.map(v => v * u), t }); arrows(.15); drawMarks(MARKS08, 1 - clamp(t - 6.5)); pip(cx(4), cy(0), 1, t); bubble('+1', cx(4), cy(0) - 18, t - .3);
     if (t > 6) w1.forEach(s => { const k = key(s.c, s.r); cellMark(s.c, s.r, f3(V[k]), clamp(t - 6), { dy: 6 }); });
     card(['TD error at the cheese: 1 + 0 − 0 = 1', 'each cell: + 0.2 × 1 × its mark', { s: `mark 1 → 0.2,  mark 0.72 → ${f3(.2 * .72)},  …`, c: C.goldInk }], t - 1.5, { gap: 1.3 }); } },
 { h: 'The dial λ', p: ['At **λ = 0** marks vanish after one step: plain TD. At **λ = 1** marks shrink by 0.9 per step, like discounting, so on walk 1 each cell gets the +1 as its return would: exactly Monte Carlo. Later, guesses shift mid-walk, so only close.', 'Between, it acts like an average of all n-step targets, longer ones weighted less. This is **TD(λ)**.'],
   shot: [2.5, 5.5, 8.5, 11.5],
   draw(t) { const i = clamp(Math.floor(t / 3), 0, 3), l = [0, .5, .8, 1][i], V = after1(l); maze({ V, t }); arrows(.15); nums(V, (c, r) => Math.abs(V[key(c, r)]) > .0005 ? 1 : 0, null, f3); pip(cx(4), cy(0), 1, t);
     txt(`λ = ${l}${l === 0 ? '  (TD)' : l === 1 ? '  (Monte Carlo-like)' : ''}`, W / 2, 300, { s: 19, c: COLS[l] === C.gold ? C.goldInk : COLS[l] === C.rose ? C.roseInk : COLS[l] });
     card([`after walk 1: ${CELLS.filter(([c, r]) => Math.abs(V[key(c, r)]) > .0005).length} cell${l === 0 ? '' : 's'} changed`, { s: `start: ${f3(V[SK])}`, c: C.soft }], 3, { y: 318, gap: 0 }); } },
 { h: 'A fair race', p: ['Pip’s habit never changes, so every λ can learn from the **very same walks**. Each is scored by its **error**: its average gap from the true values on the 7 cells of Pip’s usual route.', `True values: the values reel’s notebook, 0.87 beside the cheese, ${f2(EX[SK])} at the start. To avoid a lucky draw, we average the race over ${NSETS} sets of walks.`],
   shot: [4, 11],
   draw(t) { maze({ V: EX, t }); nums(EX, 1, (c, r) => ROUTE.some(q => q[0] === c && q[1] === r)); arrows(.15); pip(cx(0), cy(3), 1, t);
     alpha(clamp(t - 1), () => ROUTE.forEach(([c, r]) => box(OX + c * CS + 3, OY + r * CS + 3, CS - 6, CS - 6, C.teal, .18)));
     card(['same walks for every λ', 'error = average |guess − true value|, 7 route cells', { s: `true: start ${f2(EX[SK])}, beside the cheese ${f2(EX[K30])}`, c: C.goldInk }], t - 3, { gap: 1.2 }); } },
 { h: 'The first 50 walks', p: ['Across: walks 1 to 50. Up: the error, averaged over all the sets of walks. Lower is better.', `After 10 walks, TD’s error is still ${f2(ERR[0][9])}; with λ = 0.8 it is ${f2(ERR[.8][9])}, with λ = 1 only ${f2(ERR[1][9])}. Spreading each surprise over the recent path makes early learning much faster. But watch the rose line creep back up.`],
   shot: [3, 11],
   draw(t) { const g = { x0: 66, x1: 360, yt: 50, yb: 270 }, ax = clamp(t - .3), n = Math.floor(clamp((t - 2.5) / 5) * 50);
     txt('error on the route, walk by walk (lower is better)', g.x0 - 30, g.yt - 22, { s: 14, c: C.soft, al: 'left', a: ax });
     errChart([0, .8, 1], n, 50, { ax, g, labels: false });
     [[0, 'λ = 0  (TD)'], [.8, 'λ = 0.8'], [1, 'λ = 1']].forEach(([l, s], i) => alpha(clamp(t - 1 - i * .4), () => { rline(250, 70 + i * 20, 272, 70 + i * 20, 1900 + i, { c: COLS[l], w: 2.5, once: true }); txt(s, 278, 70 + i * 20, { s: 14, al: 'left', c: C.ink }); }));
     if (t > 7.6) { const X = 66 + 9 / 49 * (360 - 66); alpha(clamp(t - 7.6), () => rline(X, 50, X, 270, 1910, { c: C.soft, w: 1, once: true })); txt('walk 10', X + 4, 262, { s: 12, c: C.soft, al: 'left', a: clamp(t - 7.6) }); }
     if (t > 8) card([{ s: `after 10 walks:  TD ${f2(ERR[0][9])}`, c: '#3f8f8b' }, { s: `λ 0.8: ${f2(ERR[.8][9])}    λ 1: ${f2(ERR[1][9])}`, c: C.ink }], t - 8, { y: 312, gap: .8 }); } },
 { h: 'Fast early, noisy later', p: [`Bars: error after 20 and 200 walks, one scale. At 20 walks λ = ${BEST20}, in the middle, wins. After 200, plain TD wins: ${f2(ERR[0][199])} against ${f2(ERR[1][199])} for λ = 1.`, 'Why? Large λ leans on the real rest of the walk, so one cat ending jolts many cells, as in Monte Carlo. Small λ leans on neighbours’ guesses, which already average many walks.'],
   shot: [4, 11],
   draw(t) { txt('error by λ, all with α = 0.2  (outlined = best)', W / 2, 26, { s: 13, c: C.soft, a: clamp(t) }); errBars(t - .5, null, { y0: 250, h: 150 });
     if (t > 4) card([`20 walks: best λ = ${BEST20}  (error ${f2(ERR[BEST20][19])})`, `200 walks: best λ = ${BEST200}  (error ${f2(ERR[BEST200][199])})`, { s: 'big λ spreads news fast but shakes more', c: C.soft }], t - 4, { y: 300, gap: 1.2 }); } },
{ h: 'Is one α fair?', p: ['Every λ used α = 0.2, but large λ spreads surprises widely and may want smaller steps. Now each gets its best α, 0.05 to 0.5.', `After 20 walks λ = ${ABEST20} wins, ${ABEST[ABEST20][1].toFixed(2)}. After 200, smaller steps help all, and λ = ${ABEST200} edges ahead: ${ABEST[ABEST200][3].toFixed(3)}, against ${ABEST[0][3].toFixed(3)} for TD and ${ABEST[1][3].toFixed(3)} for λ = 1. TD’s late win was partly the shared α.`],
   shot: [4, 11],
   draw(t) { txt('error by λ, each with its best α  (outlined = best)', W / 2, 26, { s: 13, c: C.soft, a: clamp(t) }); errBars(t - .5, null, { y0: 250, h: 150, tuned: 1 });
     if (t > 4) card([`20 walks: best λ = ${ABEST20}  (α ${ABEST[ABEST20][0]}, error ${ABEST[ABEST20][1].toFixed(2)})`, `200 walks: best λ = ${ABEST200}  (α ${ABEST[ABEST200][2]}, error ${ABEST[ABEST200][3].toFixed(3)})`, { s: 'the best λ also depends on α and on how long', c: C.soft }], t - 4, { y: 300, gap: 1.2, fs: 16 }); } },
 { h: 'Your turn', ctl: { label: 'λ', values: LSL, def: 3, onChange: st => {} },
   p: ['Drag **λ**. The maze shows this run’s notebook after 10 walks, and the bars show the averaged error after 20 and 200 walks for your setting.', 'At 0, watch the news stay near the cheese; at 1, it reaches the start at once. Which λ would you pick for 20 walks? For 200?'],
   shot: [3],
   draw(t, st) { const l = LSL[st.ci ?? 3], V1 = RUN(l).hist[9], prev = st.prevCi != null ? RUN(LSL[st.prevCi]).hist[9] : V1, u = ease((t - (st.changed ?? -9)) / .8), V = V1.map((v, i) => lerp(prev[i], v, u));
     maze({ V, t }); nums(V); arrows(.15); pip(cx(0), cy(3), 1, t); txt(`λ = ${l}: this run’s notebook after 10 walks`, W / 2, 284, { s: 13, c: C.soft }); errBars(9, l, { y0: 382, h: 46 }); } },
 { h: 'The cost', p: [`Traces cost: each step updates every marked cell, not one. With λ = 0.8, a mark stays above 0.01 for ${MARK_LIFE} steps, so on a long walk Pip updates up to ${MARK_LIFE} cells per step instead of 1. Smaller marks can be dropped. Here, a revisited cell’s mark resets to 1, not adding up.`, 'And λ is one more dial to tune, with α.'],
   shot: [4, 10],
   draw(t) { maze({ t }); arrows(.15); const tr = WK[1], D = .5, m = walkAt(tr, t - .5, D), k = Math.min(tr.length, m.k + 1); drawMarks(marksAt(tr, Math.min(tr.length, m.k), .8), 1, { n: false }); pip(m.x, m.y, m.face, t);
     const E = marksAt(tr, Math.min(tr.length, m.k), .8), nm = E.filter(v => v > .01).length;
     card([`walk 2, step ${k}`, { s: `cells with a mark to update: ${nm}`, c: '#2f6f6c' }, { s: 'TD alone: 1 cell per step', c: C.soft }], .5, { gap: 0 }); } },
 { h: 'Where you’ve seen this', p: ['Richard Sutton introduced TD(λ) in 1988. Gerald Tesauro’s **TD-Gammon** (1992) used it to learn backgammon by playing itself, coming close to the best human players.', 'The same λ lives on in **GAE** (generalised advantage estimation, 2015). It judges how much better a move turned out than the cell’s value, blending short and long lookaheads with λ; the popular PPO method commonly uses it.'],
   shot: [4, 10],
   draw(t) { const a = clamp(t - .3); alpha(a, () => { rrect(70, 40, 260, 150, 1500, { fill: '#e9dcc6' }); rline(200, 40, 200, 190, 1501, { c: C.ink, w: 2 });
       for (let i = 0; i < 6; i++) { const x = 80 + i * 20, y = 45; ctx.save(); ctx.fillStyle = i % 2 ? C.rose : C.teal; ctx.globalAlpha = .7; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 10, y + 60); ctx.lineTo(x + 20, y); ctx.fill(); ctx.beginPath(); ctx.moveTo(x + 130, 185); ctx.lineTo(x + 140, 125); ctx.lineTo(x + 150, 185); ctx.fill(); ctx.restore(); }
       [[100, 170], [120, 170], [250, 60], [290, 60], [310, 80]].forEach(([x, y], i) => rcircle(x, y, 9, 1510 + i, { fill: i < 2 ? C.card : '#4a4540', j: .4 })); });
     txt('backgammon', 200, 210, { s: 16, c: C.soft, a: clamp(t - 1) });
     if (t > 3) card(['1988: TD(λ), Richard Sutton', '1992: TD-Gammon, Gerald Tesauro', { s: '2015: GAE, the same λ inside PPO', c: C.soft }], t - 3, { gap: 1.2 }); } },
 { h: 'Recap', p: ['An **eligibility trace** is a fading mark on recently left cells. Each TD error updates every marked cell, in proportion to its mark, so one surprise reaches the whole recent path.', '**λ** sets how slowly marks fade: 0 is TD, 1 is Monte Carlo-like. Large λ learns fast early but is noisier; in between is often a good compromise.'],
   shot: [4, 10],
   draw(t) { maze({ V: RUN(.8).hist[NW - 1], t }); nums(RUN(.8).hist[NW - 1]); arrows(.2); pip(cx(0), cy(3), 1, t); txt('λ = 0.8, α = 0.2: this run’s notebook after 200 walks', W / 2, 282, { s: 13, c: C.soft });
     card(['each step: marks × 0.9 λ, then cell just left = 1', 'TD error = reward + 0.9 × next − old', 'every cell += α × TD error × its mark', { s: 'λ = 0: TD  ·  λ = 1: Monte Carlo-like', c: C.soft }], t - .3, { gap: 1 }); } },
];

RLStory.register('traces', { title: 'Eligibility Traces & TD(λ)', track: 'Model-free', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { alphaScan, ABEST, WK, EX, ERR, LAMS, RUN, BEST20, BEST200, MARKS08, TD10START, L08_10START, L1_10START, FIRSTSTART0, after1, f2, f3, marksAt } });
})();
