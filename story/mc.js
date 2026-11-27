// Monte Carlo learning as a story: Pip learns values from the walks it actually takes (always from the start),
// writing a return into every cell it passed once a walk is over. Part 1 judges the values reel's arrows; part 2
// drops the arrows and improves its own moves with averages per move (Q). All numbers come from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, key, maze, trail, pip } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GAM = 0.9, EPS = 0.2, NW = 1000, SK = key(0, 3);
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);
const ZERO = () => new Array(20).fill(0);
const f2 = v => { const s = (Math.abs(v) < .005 ? 0 : v).toFixed(2); return s.replace('-', '−'); };
const fr = v => f2(v).replace(/\.?0+$/, '') || '0';            // 1.00 → 1, 0.90 → 0.9 (for single returns)

// ───── Part 1: the values reel's arrows policy and its exact start walks (same code, same seed: rng(87000 + start key))
const ARR = new Array(20).fill(-1);
[[0, 3, 0], [0, 2, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [2, 1, 0], [1, 2, 3], [1, 3, 1], [2, 3, 1], [3, 3, 1], [4, 3, 0], [4, 2, 0], [4, 1, 0], [3, 2, 1]]
  .forEach(([c, r, m]) => { ARR[key(c, r)] = m; });
function walk(c, r, eps, R) {
  const tr = [];
  for (let k = 0; k < 5000; k++) {
    let m, rnd = false; if (R() < eps) { rnd = true; m = Math.floor(R() * 4); } else m = ARR[key(c, r)];
    const [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr); tr.push({ c, r, nc, nr, m, rnd, rew, bump: nc === c && nr === r }); c = nc; r = nr; if (rew) break;
  }
  let g = 0; for (let i = tr.length - 1; i >= 0; i--) { g = tr[i].rew + GAM * g; tr[i].g = g; }
  return tr;
}
const WK = []; { const R = rng(87000 + SK); for (let i = 0; i < NW; i++) WK.push(walk(0, 3, EPS, R)); }
// first-visit flags
WK.forEach(tr => { const seen = new Set(); tr.forEach(s => { const k = key(s.c, s.r); s.first = !seen.has(k); seen.add(k); }); });
// notebook after each walk (first-visit averages) + every return recorded per cell
const VS = [ZERO()], NS = [ZERO()], RL = Array.from({ length: 20 }, () => []);
{ const S = ZERO(), N = ZERO(), V = ZERO();
  WK.forEach((tr, i) => { tr.forEach(s => { if (!s.first) return; const k = key(s.c, s.r); N[k]++; V[k] += (s.g - V[k]) / N[k]; RL[k].push({ i, g: s.g, rew: tr[tr.length - 1].rew }); }); VS.push(V.slice()); NS.push(N.slice()); }); }
const everyV = (() => { const S = ZERO(), N = ZERO(); WK.forEach(tr => tr.forEach(s => { const k = key(s.c, s.r); S[k] += s.g; N[k]++; })); return S.map((s, i) => N[i] ? s / N[i] : 0); })();
// the values reel's notebook: 1000 walks started from every cell (rng(87000 + key)), for comparison
const NBV = (() => { const V = ZERO(); CELLS.forEach(([c, r]) => { const R = rng(87000 + key(c, r)); let s = 0; for (let i = 0; i < NW; i++) s += walk(c, r, EPS, R)[0].g; V[key(c, r)] = s / NW; }); return V; })();
function exactV(eps) { const V = ZERO(); for (let it = 0; it < 3000; it++) CELLS.forEach(([c, r]) => { const k = key(c, r); let v = 0; for (let m = 0; m < 4; m++) { const p = eps / 4 + (m === ARR[k] ? 1 - eps : 0), [nc, nr] = G.step(c, r, m), rw = G.term(nc, nr); v += p * (rw + (rw ? 0 : GAM * V[key(nc, nr)])); } V[k] = v; }); return V; }
const V1000 = VS[NW], N1000 = NS[NW];
const ROUTE = [[0, 3], [0, 2], [0, 1], [0, 0], [1, 0], [2, 0], [3, 0]];
const ROUTE_DIFF = Math.max(...ROUTE.map(([c, r]) => Math.abs(V1000[key(c, r)] - NBV[key(c, r)])));
const FE_DIFF = Math.max(...CELLS.map(([c, r]) => Math.abs(V1000[key(c, r)] - everyV[key(c, r)])));
const RARE = [3, 2], RK = key(3, 2);                                   // the cell below the cat
const START_R = RL[SK].map(q => q.g), RMIN = Math.min(...START_R), RMAX = Math.max(...START_R);
const HIST = (() => { const h = {}; START_R.forEach(g => { const s = g.toFixed(3); h[s] = (h[s] || 0) + 1; }); return Object.entries(h).map(([g, n]) => ({ g: +g, n })); })();
const HMAX = Math.max(...HIST.map(q => q.n)), H053 = HIST.find(q => Math.abs(q.g - .531) < .001).n, NCAT = START_R.filter(g => g < 0).length;

// ───── Part 2: no arrows. Q(cell, move) = first-visit average of returns; biggest Q (ties at random), 1 in 5 random. Seed 35 like the TD reel.
const SEED = 35, NC = 200;
function control(seed, n) {
  const R = rng(seed), Q = new Float64Array(80), N = new Float64Array(80), eps = [];
  for (let e = 0; e < n; e++) {
    let [c, r] = G.start; const tr = [];
    for (let k = 0; k < 5000; k++) {
      let m, rnd = false; const b = key(c, r) * 4;
      if (R() < EPS) { rnd = true; m = Math.floor(R() * 4); }
      else { let best = -1e9, ms = []; for (let a = 0; a < 4; a++) { const q = Q[b + a]; if (q > best + 1e-9) { best = q; ms = [a]; } else if (q > best - 1e-9) ms.push(a); } m = ms[Math.floor(R() * ms.length)]; }
      const [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr); tr.push({ c, r, nc, nr, m, rnd, rew, bump: nc === c && nr === r }); c = nc; r = nr; if (rew) break;
    }
    let g = 0; for (let i = tr.length - 1; i >= 0; i--) { g = tr[i].rew + GAM * g; tr[i].g = g; }
    const seen = new Set(); tr.forEach(s => { const j = key(s.c, s.r) * 4 + s.m; if (seen.has(j)) return; seen.add(j); N[j]++; Q[j] += (s.g - Q[j]) / N[j]; });
    eps.push({ tr, n: tr.length, end: tr[tr.length - 1].rew, Q: Array.from(Q), N: Array.from(N) });
  }
  return eps;
}
const CE = control(SEED, NC), CL = CE.map(e => e.n), CEND = CE.map(e => e.end);
function bestMove(Q, c, r) { const b = key(c, r) * 4; let best = -1e9, bm = -1, tie = false; for (let a = 0; a < 4; a++) { const q = Q[b + a]; if (q > best + 1e-9) { best = q; bm = a; tie = false; } else if (q > best - 1e-9) tie = true; } return tie ? -1 : bm; }
function qPath(Q) { let [c, r] = G.start; const P = [[c, r]]; for (let k = 0; k < 30 && !G.term(c, r); k++) { const m = bestMove(Q, c, r); if (m < 0) return { P, ok: false }; [c, r] = G.step(c, r, m); P.push([c, r]); } return { P, ok: G.term(c, r) > 0 }; }
const QFIN = CE[NC - 1].Q, QP = qPath(QFIN);
const W4 = CE[3], W4BUMPS = W4.tr.filter(s => s.bump).length;
const W4REP = (() => { const n = {}; W4.tr.forEach(s => { if (s.bump) { const j = key(s.c, s.r) * 4 + s.m; n[j] = (n[j] || 0) + 1; } }); let bj = 0, bn = 0; for (const j in n) if (n[j] > bn) { bn = n[j]; bj = +j; } return { c: Math.floor(bj / 4) % 5, r: Math.floor(Math.floor(bj / 4) / 5), m: bj % 4, n: bn }; })();
const Q3 = CE[2].Q, Q00UP = Q3[0].toFixed(3).replace('-', '−');
const SETTLE = (() => { for (let i = 0; i < NC; i++) if (CL.slice(i).filter(l => l > 15).length <= 3) return i + 1; return NC; })();
const LATE = CL.slice(SETTLE - 1), LATE710 = LATE.filter(l => l >= 7 && l <= 10).length / LATE.length;
// other seeds: does the greedy route reach the cheese after 200 walks? (checked once, quoted in text)
const SEEDS = (() => { const res = []; for (let s = 1; s <= 40; s++) { const e = control(s, NC); const p = qPath(e[NC - 1].Q); res.push(p.ok ? p.P.length - 1 : 0); } return res; })();
const SEEDS_OK = SEEDS.filter(x => x > 0 && x <= 9).length, N7 = SEEDS.filter(x => x === 7).length, N9S = SEEDS.map((x, i) => x === 9 ? i + 1 : 0).filter(x => x);
const N9FIX = N9S.filter(sd => { const e = control(sd, 1000), p = qPath(e[999].Q); return p.ok && p.P.length - 1 === 7; }).length;   // the 9-step runs, given 1000 walks
// how Pip left the corner trap: first walk after which the corner's best move is right (positive) for good
const ESCW = (() => { for (let e = 0; e < NC; e++) if (CE.slice(e).every(x => bestMove(x.Q, 0, 0) === 1 && x.Q[1] > 0)) return e + 1; return -1; })(), ESCQ = CE[ESCW - 1].Q[1];
// off-route cells (visited, not on the route, not the 3-return cell): how many returns, how far off the values reel
const OFFC = CELLS.filter(([c, r]) => !ROUTE.some(q => q[0] === c && q[1] === r) && key(c, r) !== key(3, 2));
const OFFN = OFFC.map(([c, r]) => N1000[key(c, r)]), OFF_ERR = OFFC.reduce((a, [c, r]) => a + Math.abs(V1000[key(c, r)] - NBV[key(c, r)]), 0) / OFFC.length;

// ───── drawing helpers
const no = (c, r) => r * 5 + c + 1;
function nums(V, N, a = 1, o = {}) {
  for (const [c, r] of CELLS) { const k = key(c, r); if (N && !N[k]) continue; const aa = typeof a === 'function' ? a(c, r) : a; if (aa <= 0) continue; const v = V[k];
    txt(f2(v), OX + c * CS + CS - 5, OY + r * CS + 12, { s: 14, al: 'right', a: aa, c: v < -.004 ? C.roseInk : v > .004 ? C.goldInk : C.soft });
    if (o.n && N) { const s = 'n ' + N[k]; box(OX + c * CS + 3, OY + r * CS + CS - 18, s.length * 6 + 6, 15, C.bg, .85 * aa); txt(s, OX + c * CS + 6, OY + r * CS + CS - 10, { s: 12, al: 'left', a: aa, c: C.ink }); } }
}
function cellMark(c, r, s, a = 1, o = {}) { if (o.bg) box(cx(c) - 22, cy(r) + (o.dy ?? 4) - 10, 44, 20, C.card, .85 * a); txt(s, cx(c) + (o.dx ?? 0), cy(r) + (o.dy ?? 4), { s: o.s || 17, c: o.c || C.goldInk, a }); }
function hourglass(x, y) { box(x - 11, y - 13, 22, 26, C.card, .9); ctx.save(); ctx.beginPath(); ctx.moveTo(x - 7, y - 10); ctx.lineTo(x + 7, y - 10); ctx.lineTo(x - 7, y + 10); ctx.lineTo(x + 7, y + 10); ctx.closePath(); ctx.strokeStyle = C.soft; ctx.lineWidth = 1.4; ctx.stroke();
  ctx.fillStyle = C.yellow; ctx.beginPath(); ctx.moveTo(x - 4, y + 8); ctx.lineTo(x + 4, y + 8); ctx.lineTo(x, y + 3); ctx.closePath(); ctx.fill(); ctx.restore(); }
function dim(a = .62) { box(0, 0, W, 285, C.bg, a); }
function arrows(a = 1) { CELLS.forEach(([c, r]) => alpha(a, () => G.arrowIn(c, r, ARR[key(c, r)], 400 + key(c, r), { c: C.goldInk, w: 1.4 }))); }
function edgeArrow(c, r, m, seed, col, r0 = 12, r1 = 29) { const [dx, dy] = G.MOVES[m]; arrow(cx(c) + dx * r0, cy(r) + dy * r0 + 4, cx(c) + dx * r1, cy(r) + dy * r1 + 4, seed, { c: col, w: 1.6, head: 6 }); }
// walker along a recorded walk; bumps wiggle against the wall
function walkAt(tr, t, dur) {
  const k = Math.floor(Math.max(0, t) / dur);
  if (k >= tr.length) { const l = tr[tr.length - 1]; return { x: cx(l.nc), y: cy(l.nr), k: tr.length, done: true, face: Math.sign(l.nc - l.c) || 0 }; }
  const s = tr[k], u = ease((Math.max(0, t) % dur) / dur * 1.4), [dx, dy] = G.MOVES[s.m];
  if (s.bump) { const b = Math.sin(clamp((Math.max(0, t) % dur) / dur * 1.6) * Math.PI) * 11; return { x: cx(s.c) + dx * b, y: cy(s.r) + dy * b, k, s, face: dx || 0 }; }
  return { x: lerp(cx(s.c), cx(s.nc), u), y: lerp(cy(s.r), cy(s.nr), u), k, s, face: dx || 0 };
}
function animWalk(tr, t, dur, o = {}) {
  const m = walkAt(tr, t, dur); trail(tr.slice(0, Math.min(tr.length, m.k + 1)).map(s => [s.c, s.r]), o.ta ?? .55);
  pip(m.x, m.y, m.face, t); if (!m.done && m.s && m.s.rnd && o.q !== false) txt('?', m.x + 14, m.y - 24, { s: 20, c: C.roseInk });
  return m;
}
// returns written back along a finished walk, last step first; second visits greyed and struck through
function writeBack(tr, t, per, o = {}) {
  const shown = new Set(); const n = tr.length;
  for (let j = 0; j < n; j++) { const i = n - 1 - j, s = tr[i], a = clamp((t - j * per) * 2.5); if (a <= 0) continue;
    const firstIdx = tr.findIndex(q => q.c === s.c && q.r === s.r), isFirst = firstIdx === i;
    if (isFirst) cellMark(s.c, s.r, fr(s.g), a, { s: 17, c: s.g < 0 ? C.roseInk : C.goldInk, dy: o.dy ?? 4, bg: 1 });
    else { const dy = (o.dy ?? 4) + 18; box(cx(s.c) - 18, cy(s.r) + dy - 8, 36, 16, C.card, .85 * a); txt(fr(s.g), cx(s.c), cy(s.r) + dy, { s: 14, c: C.soft, a: a * .9 }); alpha(a * .8, () => rline(cx(s.c) - 14, cy(s.r) + dy, cx(s.c) + 14, cy(s.r) + dy, 700 + i, { c: C.soft, w: 1.2, once: true })); }
  }
}
// number line from −1 to +1 for returns (used by the spread chart and the your-turn card)
const NL = { x0: 56, x1: 352 }, NX = v => NL.x0 + (v + 1) / 2 * (NL.x1 - NL.x0);
function numberLine(y, a = 1, o = {}) {
  alpha(a, () => { rline(NL.x0 - 6, y, NL.x1 + 6, y, 950, { c: C.line, w: 1.1 });
    [-1, -.5, 0, .5, 1].forEach((v, i) => { rline(NX(v), y - 3, NX(v), y + 3, 951 + i, { c: C.line, w: 1, once: true }); txt(v === 0 ? '0' : (v < 0 ? '−' : '+') + Math.abs(v), NX(v), y + 13, { s: 12, c: C.soft }); });
    if (o.label) txt(o.label, NL.x1 + 6, y + 13, { s: 12, c: C.soft, al: 'right' }); });
}
// histogram of the 1000 start returns: one bar per distinct return
function histogram(t) {
  const y0 = 238, hmax = 150, a = clamp(t - .3);
  numberLine(y0, a); txt('return of one walk, from the start', NL.x0 - 6, y0 + 30, { s: 13, c: C.soft, al: 'left', a });
  txt('how many walks', NL.x0 - 6, y0 - hmax - 14, { s: 13, c: C.soft, al: 'left', a });
  const u = ease((t - 1.5) / 2.5);
  HIST.forEach((q, i) => { const h = q.n / HMAX * hmax * u; if (h <= 0) return; box(NX(q.g) - 3.5, y0 - h, 7, h, q.g > 0 ? C.gold : C.rose, .9); });
  if (t > 4.4) { const v = V1000[SK], aa = clamp(t - 4.4); alpha(aa, () => { rline(NX(v), y0 + 4, NX(v), y0 - hmax + 8, 990, { c: C.ink, w: 1.6, once: true }); }); txt(`average ${f2(v)}`, NX(v) - 6, y0 - hmax + 2, { s: 15, al: 'right', a: aa }); }
  if (t > 3.8) txt(`${NCAT} cat walks`, NX(-.5), y0 - 34, { s: 13, c: C.roseInk, a: clamp(t - 3.8) });
  if (t > 3.5) txt(`${H053} walks: 0.53`, NX(.531) + 4, y0 - H053 / HMAX * hmax - 10, { s: 13, c: C.goldInk, a: clamp(t - 3.5) });
}
// strip of walk lengths (part 2): one bar per walk, capped at CAP steps
const CAP = 100;
function strip(n, o = {}) {
  const x0 = 40, x1 = 360, y0 = 374, y1 = 302, w = (x1 - x0) / NC;
  rline(x0, y0, x1, y0, 900, { c: C.line, w: 1 }); rline(x0, y0 - (y0 - y1) * 10 / CAP, x1, y0 - (y0 - y1) * 10 / CAP, 901, { c: '#ece2d4', w: 1, once: true });
  txt('steps per walk', x0, y1 - 17, { s: 14, al: 'left', c: C.soft }); txt('10', x0 - 4, y0 - (y0 - y1) * 10 / CAP, { s: 12, c: C.soft, al: 'right' }); txt(String(CAP), x0 - 4, y1, { s: 12, c: C.soft, al: 'right' });
  for (let i = 0; i < Math.min(n, NC); i++) { const h = Math.min(CAP, CL[i]) / CAP * (y0 - y1); ctx.fillStyle = CEND[i] > 0 ? C.teal : C.rose; ctx.fillRect(x0 + i * w + .2, y0 - h, Math.max(1, w - .5), h);
    if (CL[i] > CAP) txt('▲', x0 + i * w + w / 2, y1 - 4, { s: 9, c: CEND[i] > 0 ? '#3f7f7c' : C.roseInk }); }
  txt('walk 1', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NC}`, x1, y0 + 11, { s: 12, c: C.soft, al: 'right' });
}
function qArrows(Q, a = 1) { CELLS.forEach(([c, r]) => { const m = bestMove(Q, c, r); if (m < 0) { txt('?', cx(c), cy(r) + 5, { s: 17, c: '#c2b6a6', a }); return; } alpha(a, () => G.arrowIn(c, r, m, 400 + key(c, r), { c: C.goldInk, w: 1.5 })); }); }
function cards() { // playing cards + a chip, for the real-world step
  [[96, 70, -.18, 'A', '♠', C.ink], [140, 64, .02, '8', '♥', C.roseInk], [184, 72, .2, 'K', '♣', C.ink]].forEach(([x, y, rot, v, s, col], i) => {
    ctx.save(); ctx.translate(x, y); ctx.rotate(rot); rrect(-24, -34, 48, 68, 1600 + i * 4, { fill: C.card }); txt(v, -12, -20, { s: 17, c: col }); txt(s, 4, 6, { s: 26, c: col }); ctx.restore(); });
  const x = 250, y = 22, g = 22; for (let i = 0; i < 6; i++) { rline(x, y + i * g, x + 5 * g, y + i * g, 1660 + i, { c: C.line, w: 1 }); rline(x + i * g, y, x + i * g, y + 5 * g, 1670 + i, { c: C.line, w: 1 }); }
  [[1, 1, 0], [2, 1, 1], [2, 2, 0], [3, 2, 1], [1, 3, 1], [3, 3, 0], [4, 2, 0]].forEach(([a, b, wh], i) => rcircle(x + a * g, y + b * g, g * .42, 1680 + i, { fill: wh ? C.card : '#4a4540', j: .4 }));
  txt('blackjack', 140, 122, { s: 15, c: C.soft }); txt('Go', 305, 148, { s: 15, c: C.soft });
}

// your-turn taps: select a cell
function tapCell(x, y, st) { const c = Math.floor((x - OX) / CS), r = Math.floor((y - OY) / CS); if (c < 0 || c >= 5 || r < 0 || r >= 4 || G.isRock(c, r) || G.term(c, r)) return; st.sel = [c, r]; st.selT = null; }
const NWALKS = [1, 3, 10, 100, 1000];

const w1 = WK[0], w2 = WK[1], w3 = WK[2];
const STEPS = [
 { h: 'Back in the maze', p: ['Pip is back in its maze, and the floor is **dry** again: every move goes where Pip intends. Stepping onto the cheese pays +1, onto the cat −1, and either one ends the walk; other steps pay 0. A bump leaves Pip in place but counts as a step. γ (gamma) is 0.9.', 'Pip follows its **habit** from the values reel: the arrow, except that one move in five goes in a random direction.'],
   shot: [5, 11],
   draw(t) { maze({ draw: t / 1.6, pop: { cheese: ease((t - 1.8) * 3), cat: ease((t - 2.3) * 3) }, labels: clamp(t - 3), t }); if (t > 2.8) pip(cx(0), cy(3), 1, t); if (t > 4) arrows(clamp(t - 4) * .8);
     if (t > 5.5) card(['cheese +1, cat −1, on the step onto them; walk ends', 'other steps 0  ·  bump: stay put, still a step', { s: 'dry floor  ·  habit: arrow 4 in 5, random 1 in 5', c: C.soft }], t - 5.5, { gap: 1, fs: 16 }); } },
 { h: 'No rulebook', p: ['Pip has no rulebook: nobody told it where moves lead or what they pay, so it can’t work values out on paper. It can only walk.', 'In the values reel, Pip was placed in every cell for 1000 walks each: that averaging was already Monte Carlo, the wasteful kind. Real Pip always starts in the corner. **Monte Carlo learning** learns from the complete walks it actually takes. The name comes from the famous casino.'],
   shot: [3, 10],
   draw(t) { maze({ t }); arrows(.3); pip(cx(0), cy(3), 1, t); const a = clamp(t - .6);
     alpha(a, () => { box(118, 96, 164, 66, C.card, 1); rrect(118, 96, 164, 66, 1100, { c: C.line }); txt('rulebook', 200, 116, { s: 17 }); txt('where moves lead, what they pay', 200, 140, { s: 13, c: C.soft }); rline(124, 102, 276, 156, 1101, { c: C.rose, w: 2.2 }); rline(276, 102, 124, 156, 1102, { c: C.rose, w: 2.2 }); });
     if (t > 4) card(['values reel: 1000 walks from every cell', 'now: only the walks Pip takes, from the start', { s: 'learn from complete walks: Monte Carlo', c: C.goldInk }], t - 4, { gap: 1.3 }); } },
 { h: 'One walk, seven returns', p: [`Walk 1, as in the values reel: cheese in ${w1.length} steps. Once the walk is over, Pip writes a **return** into every cell it passed, not just the start.`, `A cell’s return counts the rewards from the step that leaves it until the walk ends. Beside the cheese that is 1. Each cell further back waited one step longer, one more × 0.9, down to ${f2(w1[0].g)} at the start.`],
   shot: [4, 12],
   draw(t) { maze({ t }); arrows(.25); const m = animWalk(w1, t - .5, .55), tEnd = .5 + w1.length * .55;
     if (m.done) bubble('+1', cx(4), cy(0) - 18, t - tEnd); else caption(`step ${Math.min(w1.length, m.k + 1)}`);
     if (t > tEnd + .8) writeBack(w1, t - tEnd - .8, .7);
     if (t > tEnd + 1) card(['return from a cell = rewards after leaving it', 'beside the cheese: 1  ·  one step back: × 0.9', { s: `start: 0.9 multiplied ${w1.length - 1} times = ${f2(w1[0].g)}`, c: C.goldInk }], t - tEnd - 1, { gap: 2 }); } },
 { h: 'A notebook of averages', p: ['For every cell the notebook keeps two things: how many returns it has, **n**, and their average.', 'A new return nudges the average: new = old + (return − old) ÷ n. The hundredth return moves it only a hundredth of the way. Some versions nudge by a fixed fraction instead, so old walks fade: useful when the habit keeps changing.'],
   shot: [3, 10],
   draw(t) { const V = VS[1], N = NS[1]; maze({ V, t }); arrows(.2); trail(w1.map(s => [s.c, s.r]), .3); nums(V, N, (c, r) => clamp(t - .4 - ROUTE.findIndex(q => q[0] === c && q[1] === r) * .25), { n: 1 }); pip(cx(4), cy(0), 1, t);
     card(['corner: average  ·  bottom: n, returns so far', 'new = old + (return − old) ÷ n', { s: 'walk 1: n = 1, so average = that return', c: C.soft }], t - 2.2, { gap: 1.5 }); } },
 { h: 'Walk 2: twice in one cell', p: ['On walk 2 Pip steps back down once and bumps the top wall once, so it passes three cells twice. Which return counts? Here, only the one from the **first visit** in each walk. Counting every visit also works: after 1000 walks the two notebooks differ by at most ' + f2(FE_DIFF) + '.', `The start averages (0.53 + 0.39) ÷ 2 = ${f2(VS[2][SK])}. Beside the cheese: (1 + 0.9) ÷ 2 = ${f2(VS[2][key(3, 0)])}.`],
   shot: [5, 13],
   draw(t) { const D = .42, tEnd = .4 + w2.length * D, done = t > tEnd + .6 + w2.length * .45 + .6;
     maze({ V: done ? VS[2] : VS[1], t }); arrows(.2); const m = animWalk(w2, t - .4, D); if (m.done) bubble('+1', cx(4), cy(0) - 18, t - tEnd); else caption(`step ${Math.min(w2.length, m.k + 1)}`);
     nums(done ? VS[2] : VS[1], done ? NS[2] : NS[1], 1, { n: done || t < tEnd + .6 });
     if (t > tEnd + .6 && !done) writeBack(w2, t - tEnd - .6, .45);
     if (done) card([`start: (0.53 + 0.39) ÷ 2 = ${f2(VS[2][SK])}`, `beside the cheese: (1 + 0.9) ÷ 2 = ${f2(VS[2][key(3, 0)])}`, { s: 'a cell passed twice: only its first-visit return counts', c: C.soft }], t - tEnd - .6 - w2.length * .45 - .6, { gap: 1.1 });
     else if (t > tEnd + .6) caption('struck-through grey: a second visit, not counted'); } },
 { h: 'Walk 3: same path, opposite lesson', p: ['Walk 3 starts along the same path. Halfway, Pip can’t write anything: whether these cells lead to cheese or cat is still unknown.', `Then, right beside the cheese, a random move points down, onto the cat. Every return on the path is negative. The start’s average drops to ${f2(VS[3][SK])}, beside the cheese to ${f2(VS[3][key(3, 0)])}. A single walk can swing the notebook a lot.`],
   shot: [2.6, 13],
   draw(t) { const D = .5, tEnd = .4 + w3.length * D, t2 = tEnd + .8 + w3.length * .4, done = t > t2 + .5;
     maze({ V: done ? VS[3] : VS[2], t }); arrows(.2); const mw = walkAt(w3, t - .4, D);
     if (!mw.done) { w3.slice(0, mw.k + 1).forEach(s => hourglass(cx(s.c), cy(s.r) + 4)); caption('mid-walk: hourglass = return not known yet'); }
     const m = animWalk(w3, t - .4, D); nums(done ? VS[3] : VS[2], done ? NS[3] : NS[2], 1, { n: done || !m.done });
     if (!m.done) {}
     else { bubble('−1', cx(3), cy(1) - 18, t - tEnd, C.roseInk); if (t > tEnd + .8 && !done) writeBack(w3, t - tEnd - .8, .4); }
     if (done) card([`start: (0.53 + 0.39 − 0.48) ÷ 3 = ${f2(VS[3][SK])}`, `beside the cheese: (1 + 0.9 − 1) ÷ 3 = ${f2(VS[3][key(3, 0)])}`, { s: 'one cat walk pulls every cell on the path down', c: C.roseInk }], t - t2 - .5, { gap: 1.1 }); } },
 { h: 'Many walks', p: [`Pip keeps walking from the start. After 10 walks the start reads ${f2(VS[10][SK])}, after 100 walks ${f2(VS[100][SK])}, after 1000 walks ${f2(V1000[SK])}: exactly the values reel’s number, because these are the same 1000 walks.`, `But now every cell on the route got its own returns from those walks too, at least ${Math.min(...ROUTE.map(([c, r]) => N1000[key(c, r)]))} each, with no extra walking.`],
   shot: [3, 7, 13],
   draw(t) { let n; if (t < 3) n = 3 + Math.floor(clamp(t / 2.5) * 7); else if (t < 7) n = 10 + Math.floor(clamp((t - 3) / 3) * 90); else n = 100 + Math.floor(clamp((t - 7) / 4) * 900);
     maze({ V: VS[n], t }); arrows(.2); const w = WK[n - 1]; trail(w.map(s => [s.c, s.r]), .35); nums(VS[n], NS[n], 1, { n: 1 }); pip(cx(w[w.length - 1].nc), cy(w[w.length - 1].nr), 1, t);
     txt(`after walk ${n} of ${NW}`, W / 2, 298, { s: 16, c: C.soft });
     card([`start after 10 walks: ${f2(VS[10][SK])}`, `after 100: ${f2(VS[100][SK])}   ·   after 1000: ${f2(V1000[SK])}`, { s: 'values reel, same 1000 walks: 0.40', c: C.goldInk }], n >= 1000 ? t - 11.2 : -1, { y: 312, gap: 1 }); } },
 { h: 'Where Pip rarely goes', p: [`Compare with the values reel, which started 1000 walks in every cell. Along Pip’s usual route the two notebooks agree within ${f2(ROUTE_DIFF)}.`, `Off the route, cells got only ${Math.min(...OFFN)} to ${Math.max(...OFFN)} returns, mostly from the same few walks, and are off by about ${OFF_ERR.toFixed(1)}: the cell below the cheese reads ${f2(V1000[key(4, 1)])} against ${f2(NBV[key(4, 1)])}. Below the cat: only ${N1000[RK]} returns, ${f2(V1000[RK])} against ${f2(NBV[RK])}. Monte Carlo only learns where Pip actually goes.`],
   shot: [4, 11],
   draw(t) { const a1 = clamp(t - 3.5); maze({ V: V1000, t, hi: [[3, 0, C.gold, .25 * clamp(t - .5) * (1 - a1)], [3, 2, C.rose, .3 * a1]] }); nums(V1000, N1000, 1, { n: 1 });
     card([{ s: `beside the cheese: n ${N1000[key(3, 0)]}, average ${f2(V1000[key(3, 0)])}  (values reel ${f2(NBV[key(3, 0)])})`, c: C.goldInk }, { s: `below the cat: n ${N1000[RK]}, average ${f2(V1000[RK])}  (values reel ${f2(NBV[RK])})`, c: C.roseInk }, { s: `other off-route cells: n ${Math.min(...OFFN)} to ${Math.max(...OFFN)}, off by about ${OFF_ERR.toFixed(1)}`, c: C.soft }], t - 1, { gap: 2.6, fs: 15 }); } },
 { h: 'How much one return varies', p: ['Here are all 1000 returns from the start. Across is the return of one walk, from −1 to +1. Each bar counts how many walks gave that return: gold walks reached the cheese, rose ones the cat.', `They spread from ${f2(RMIN)} to ${f2(RMAX)}, because every random move until the end of the walk changes the return. Only the average of many walks settles, near ${f2(V1000[SK])}.`],
   shot: [3, 11],
   draw(t) { histogram(t);
     if (t > 6) card([`${H053} walks: the shortest route, 0.53`, `${NCAT} walks: the cat, a negative return`, { s: `one walk alone: anywhere from ${f2(RMIN)} to ${f2(RMAX)}`, c: C.soft }], t - 6, { gap: 1.1 }); } },
 { h: 'Your turn', ctl: { label: 'walks', values: NWALKS, def: 2, onChange: st => {} }, tap: tapCell,
   p: ['Drag the slider to choose how many of Pip’s walks the notebook has seen. Tap any cell to see every return it has collected, as dots on a line from −1 to +1, with its average.', 'Try the start, then a cell in the bottom right. With few walks, see how far one return can pull an average.'],
   shot: [3],
   draw(t, st) { const n = NWALKS[st.ci ?? 2], V = VS[n], N = NS[n], sel = st.sel || [0, 3], k = key(sel[0], sel[1]);
     maze({ V, t, hi: [[sel[0], sel[1], C.teal, .25]] }); arrows(.15); nums(V, N, 1, { n: 1 }); pip(cx(0), cy(3), 1, t);
     const rs = RL[k].filter(q => q.i < n); box(20, 286, 360, 106, C.card, .95); rrect(20, 286, 360, 106, 1700, { c: C.line });
     txt(`${k === SK ? 'start' : 'this cell'}: ${rs.length} return${rs.length === 1 ? '' : 's'} from ${n} walk${n === 1 ? '' : 's'}`, 36, 303, { s: 16, al: 'left' });
     numberLine(366, 1);
     rs.forEach((q, i) => dot(NX(q.g), 348 - ((i * 37) % 23), rs.length > 50 ? 2 : 3.6, q.rew > 0 ? C.gold : C.rose, rs.length > 50 ? .5 : .95));
     if (rs.length) { const v = V[k]; rline(NX(v), 326, NX(v), 364, 1710, { c: C.ink, w: 1.8, once: true }); txt(`average ${f2(v)}`, Math.min(330, Math.max(70, NX(v))), 320, { s: 14, c: v < 0 ? C.roseInk : C.goldInk }); }
     else txt('no returns yet: Pip never got here', W / 2, 345, { s: 15, c: C.soft });
     caption('tap a cell to see its returns'); } },
 { h: 'From judging to improving', p: ['So far the arrows came from someone else; the notebook only judged them. To improve, a cell’s value isn’t enough: without a rulebook, Pip can’t tell where each move leads.', 'So Pip keeps an average per move: **Q(cell, move)**, said “Q of cell and move”: the average return after taking that move there, first visit per walk. Picking moves by Q changes the habit: policy iteration’s evaluate-then-improve loop, done a little after every walk.'],
   shot: [4, 11],
   draw(t) { maze({ t, hi: [[2, 0, C.teal, .2]] }); pip(cx(2), cy(0), 1, t); const qa = clamp(t - 1.5); [0, 1, 2, 3].forEach(m => alpha(qa, () => edgeArrow(2, 0, m, 1500 + m, C.ink, 17)));
     const lab = [[cx(2), 8, 'Q up'], [cx(3) + 4, cy(0) + 22, 'Q right'], [cx(2), cy(1) + 24, 'Q down'], [cx(1) - 4, cy(0) + 22, 'Q left']];
     lab.forEach(([x, y, s], i) => { const u = clamp(t - 2 - i * .4); box(x - 26, y - 9, 52, 18, C.card, .9 * u); txt(s, x, y, { s: 14, a: u }); });
     card(['value: one number per cell', 'Q: one number per cell and move', { s: 'after a walk: each move used gets its return', c: C.soft }], t - 4.5, { gap: 1.2 }); } },
 { h: 'A habit built from averages', p: ['Pip drops the arrows and starts from scratch: every Q is 0. In each cell it takes the move with the biggest Q, ties broken at random. One move in five is random anyway, so moves that look bad still get tried.', `Walk 1 wanders for ${CL[0]} steps and ends on the cat. Every move it used now has a negative average, so next time an untried move, still at 0, looks better.`],
   shot: [5, 12.5],
   draw(t) { const D = Math.min(.14, 7 / CL[0]), tEnd = .5 + CL[0] * D, done = t > tEnd + .6, tr = CE[0].tr;
     maze({ t }); const m = animWalk(tr, t - .5, D, { ta: .35 }); if (!m.done) caption(`walk 1, step ${Math.min(CL[0], m.k + 1)}`); else bubble('−1', cx(3), cy(1) - 18, t - tEnd, C.roseInk);
     if (done) { const Q = CE[0].Q, u = clamp(t - tEnd - .6); CELLS.forEach(([c, r]) => [0, 1, 2, 3].forEach(a => { if (Q[key(c, r) * 4 + a] < 0) alpha(u * .85, () => edgeArrow(c, r, a, 1800 + key(c, r) * 4 + a, C.roseInk, 10, 26)); })); caption('rose arrows: moves with a negative Q after walk 1', u); }
     card(['take the move with the biggest Q (ties: random)', '1 move in 5: random, to keep trying things', { s: done ? `walk 1: ${CL[0]} steps, the cat` : ' ', c: C.roseInk }], t - .3, { gap: 1.2 }); } },
 { h: 'The catch: wait for the end', p: [`Walk 4 takes ${CL[3]} steps. In the top-left corner, “up” has the least bad average, ${Q00UP}, because the walks that tried it were so long that the cat’s −1 had shrunk to almost nothing. So Pip bumps that wall ${W4REP.n} times.`, 'Its averages can’t change until the walk is over, and a walk that never ended would teach nothing. **TD learning**, in a coming reel, updates after every step.'],
   shot: [5, 12],
   draw(t) { const tr = W4.tr, D = 9 / CL[3], m = walkAt(tr, t - .5, D), k = Math.min(tr.length, m.k + 1);
     maze({ t, hi: [[W4REP.c, W4REP.r, C.rose, .22]] }); trail(tr.slice(0, k).map(s => [s.c, s.r]), .25); pip(m.x, m.y, m.face, t);
     if (m.done) bubble('−1', cx(3), cy(1) - 18, t - .5 - CL[3] * D, C.roseInk);
     const cb = tr.slice(0, k).filter(s => s.bump && s.c === W4REP.c && s.r === W4REP.r && s.m === W4REP.m).length;
     card([`walk 4, step ${k}   ·   bumps into the top-left wall: ${cb}`, { s: `corner Q:  up (bump) ${Q00UP},  down ${Q3[2].toFixed(3).replace('-', '−')}`, c: C.ink }, { s: `               left (bump) ${f2(Q3[3])},  right ${f2(Q3[1])}`, c: C.ink }, { s: m.done ? 'walk over: now the averages update' : 'averages frozen until the walk ends', c: m.done ? C.goldInk : C.soft }], 2, { gap: 0, fs: 15, lh: 21 }); } },
 { h: 'How Pip got out', p: ['After walk 4, up and down in the corner both averaged just below 0, the best there, so long walks went on. The way out: a walk that reaches the cheese gives every move it used a positive return.', `On walk ${ESCW} Pip left the corner to the right, and that walk reached the cheese. “Right” got the corner’s only positive average, ${f2(ESCQ)}, so Pip now turns right there.`],
   shot: [3, 12],
   draw(t) { const n = clamp(4 + Math.floor(t / .8), 4, SETTLE), Q = CE[n - 1].Q; maze({ t, hi: [[0, 0, C.teal, .2]] });
     CELLS.forEach(([c, r]) => { const m = bestMove(Q, c, r); if (m >= 0 && Q[key(c, r) * 4 + m] > 0) G.arrowIn(c, r, m, 400 + key(c, r), { c: C.goldInk, w: 1.6 }); });
     const l = CE[n - 1].tr[CE[n - 1].n - 1]; pip(cx(l.nc), cy(l.nr), 1, t);
     const q = a => { const v = Q[a]; return (Math.abs(v) < .01 ? v.toFixed(3) : f2(v)).replace('-', '−'); };
     card([`after walk ${n}: ${CL[n - 1]} steps, ${CEND[n - 1] > 0 ? 'cheese' : 'cat'}`, { s: `corner Q: up ${q(0)}, right ${q(1)}, down ${q(2)}, left ${q(3)}`, c: C.ink }, { s: 'gold arrow: best move, with a positive average', c: C.goldInk }], 2, { gap: 0, fs: 15 }); } },
 { h: `${NC} walks later`, p: [`One bar per walk: height = steps (up to 100, faint line at 10); teal cheese, rose cat. After about ${SETTLE} walks most take 7 to 10 steps.`, `The biggest-Q arrows lead Pip to the cheese in ${QP.P.length - 1} steps, the shortest route. In 40 more runs with different luck, ${N7} found it; ${N9S.length} kept a 9-step detour: the shorter way, tried mostly by random moves, still looked worse. With 1000 walks, ${N9FIX} of the ${N9S.length} find 7.`],
   shot: [4, 16.5],
   draw(t) { let n; if (t < 6) n = 1 + Math.floor(t / .5); else n = 13 + Math.floor((t - 6) / .035); const done = n >= NC; n = Math.min(n, NC);
     const Q = CE[n - 1].Q; maze({ t }); qArrows(Q, .9);
     if (done && t > 12.8) { const t2 = t - 12.8, kk = clamp(Math.floor(t2 / .5), 0, QP.P.length - 1); trail(QP.P.slice(0, kk + 1), .7); pip(cx(QP.P[kk][0]), cy(QP.P[kk][1]), 1, t); }
     else { const l = CE[n - 1].tr[CE[n - 1].n - 1]; pip(cx(l.nc), cy(l.nr), 1, t); }
     strip(n); caption(done ? (t > 12.8 + QP.P.length * .5 ? 'rarely visited cells: rough arrows, never mattered  (? = a tie)' : `following the arrows: ${NC} walks learned`) : `walk ${n} of ${NC}`); } },
 { h: 'Where you’ve seen this', p: ['The name dates from the 1940s, when Stanislaw Ulam and John von Neumann used random sampling for physics calculations at Los Alamos.', 'A classic RL example is **blackjack**: the value of holding 18 is estimated by playing many hands to the end and averaging the results. AlphaGo, in 2016, also judged Go positions partly by playing fast games to the end and averaging who won.'],
   shot: [4, 10],
   draw(t) { alpha(clamp(t - .3), cards); txt('play to the end, then average', W / 2, 200, { s: 17, c: C.soft, a: clamp(t - 1.5) });
     if (t > 3) card(['1940s: Monte Carlo methods, random sampling', 'blackjack: average result of many hands', { s: 'AlphaGo 2016: fast games played to the end', c: C.soft }], t - 3, { gap: 1.2 }); } },
 { h: 'Recap', p: ['**Monte Carlo learning** needs no rulebook. After each complete walk, every cell passed gets its return, and its average moves toward it.', 'The returns are real outcomes, but they vary a lot, and cells Pip rarely visits stay rough. Averages per move, Q, let Pip improve its own habit. The price: nothing is learned until the walk ends.'],
   shot: [4, 10],
   draw(t) { maze({ V: V1000, t }); nums(V1000, N1000); arrows(.25); pip(cx(0), cy(3), 1, t);
     card(['after a walk: return for each cell passed', 'new = old + (return − old) ÷ n', 'Q per move → pick the biggest, 1 in 5 random', { s: 'noisy returns · must wait for the end', c: C.soft }], t - .3, { gap: 1 }); } },
];

RLStory.register('mc', { title: 'Monte Carlo Learning', track: 'Model-free', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { WK, VS, NS, RL, everyV, NBV, exactV, V1000, N1000, ROUTE_DIFF, FE_DIFF, RMIN, RMAX, HIST, H053, NCAT, CE, CL, CEND, QP, W4BUMPS, W4REP, SETTLE, LATE710, SEEDS, SEEDS_OK, f2, control, qPath, bestMove } });
})();
