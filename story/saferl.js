// Safe and constrained RL as a story: a room with a glass floor. Reward says "reach the cheese", a separate cost counts
// steps onto glass. A Lagrange multiplier (a price on glass) rises while Pip overspends its budget and settles at the tipping point.
// All numbers come from the seeded run below (seed 2: 300 free walks, then 700 walks with the price switched on).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const COLS = 6, ROWS = 4, NS = COLS * ROWS;
const G = L.gridWorld({ cols: COLS, rows: ROWS, cheese: [5, 3], cat: null, rocks: [], start: [0, 3], cs: 56, ox: 32, oy: 14 });
const { cx, cy, key, maze, trail, pip } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GL = [[2, 2], [3, 2], [2, 3], [3, 3]], isGlass = (c, r) => GL.some(([a, b]) => a === c && b === r);
const GAM = 0.9, ALPHA = 0.2, EPS = 0.2, CAP = 100, ETA = 0.01, WARM = 300, NW = 1000, SEED = 2, BUD = 1;
// Q-learning with two tables: Qr (reward to come, discounted 0.9) and Qc (glass steps to come, not discounted: walks end, and the budget counts raw steps). Pip acts on Qr − lam × Qc (eps-greedy, random ties);
// both targets use the move that is best by that score in the landing cell. lam stays 0 for the first WARM walks, then after
// every walk: lam = max(0, lam + ETA × (glass steps this walk − budget)).
function run(budget, seed = SEED) {
  const R = rng(seed), Qr = new Float64Array(NS * 4), Qc = new Float64Array(NS * 4), E = []; let lam = 0;
  const score = (c, r, a) => Qr[key(c, r) * 4 + a] - lam * Qc[key(c, r) * 4 + a];
  for (let e = 0; e < NW; e++) {
    let [c, r] = G.start, cost = 0, n = 0; const tr = [];
    for (let k = 0; k < CAP; k++) {
      let m; if (R() < EPS) m = Math.floor(R() * 4); else { let b = -1e9, ms = []; for (let a = 0; a < 4; a++) { const v = score(c, r, a); if (v > b + 1e-9) { b = v; ms = [a]; } else if (v > b - 1e-9) ms.push(a); } m = ms[Math.floor(R() * ms.length)]; }
      const [nc, nr] = G.step(c, r, m), rw = G.term(nc, nr), cs = isGlass(nc, nr) ? 1 : 0, done = rw !== 0;
      let b = -1e9, bm = 0; if (!done) for (let a = 0; a < 4; a++) { const v = score(nc, nr, a); if (v > b + 1e-12) { b = v; bm = a; } }
      const i = key(c, r) * 4 + m; Qr[i] += ALPHA * (rw + (done ? 0 : GAM * Qr[key(nc, nr) * 4 + bm]) - Qr[i]); Qc[i] += ALPHA * (cs + (done ? 0 : Qc[key(nc, nr) * 4 + bm]) - Qc[i]);   // glass: no discount, like the budget
      tr.push([c, r]); cost += cs; n++; c = nc; r = nr; if (done) { tr.push([c, r]); break; }
    }
    E.push({ cost, n, lam, tr }); if (e === WARM - 1) E.snapQ = { Qr: Qr.slice(), Qc: Qc.slice() };
    if (e >= WARM) lam = Math.max(0, lam + ETA * (cost - budget));
  }
  return { E, Qr, Qc, lam };
}
const mean = (a, f) => a.reduce((s, x) => s + f(x), 0) / a.length;
const RUNS = {}, runB = b => RUNS[b] ||= run(b);
const RN = runB(BUD), E = RN.E;
// story numbers
const SHORT_R = GAM ** 4, DETOUR_R = GAM ** 8, SHORT_C = 2, TIP = (SHORT_R - DETOUR_R) / SHORT_C;   // exact, from the rules
const FREE = E.slice(WARM - 100, WARM), FREECOST = mean(FREE, e => e.cost);
const LAMS = E.map(e => e.lam), PEAK = Math.max(...LAMS), PEAKW = LAMS.indexOf(PEAK) + 1;
// a walk is a detour walk if it ever reaches the top two rows; otherwise it went along the bottom
const isDet = e => e.tr.some(([c, r]) => r <= 1);
const LAST = E.slice(NW - 300), LASTCOST = mean(LAST, e => e.cost), LASTLAM = mean(LAST, e => e.lam), SHARE = LAST.filter(e => !isDet(e)).length;
const DETCOST = mean(LAST.filter(isDet), e => e.cost), BOTCOST = mean(LAST.filter(e => !isDet(e)), e => e.cost);
const GLASS_FREE = E.slice(0, WARM).reduce((s, e) => s + e.cost, 0), GLASS_CON = E.slice(WARM).reduce((s, e) => s + e.cost, 0);
// at the branching cell (1,3): right onto glass vs up around, with the final numbers
const BQ = (Q, m) => Q[key(1, 3) * 4 + m], BR = { r: BQ(RN.Qr, 1), c: BQ(RN.Qc, 1) }, BU = { r: BQ(RN.Qr, 0), c: BQ(RN.Qc, 0) }, BTIP = (BR.r - BU.r) / (BR.c - BU.c);
// greedy path (no random moves) at a given price, from given tables
function follow(Qr, Qc, lam) { let [c, r] = G.start; const P = [[c, r]]; for (let k = 0; k < 20 && !G.term(c, r); k++) { let b = -1e9, bm = 0; for (let a = 0; a < 4; a++) { const v = Qr[key(c, r) * 4 + a] - lam * Qc[key(c, r) * 4 + a]; if (v > b + 1e-12) { b = v; bm = a; } } [c, r] = G.step(c, r, bm); P.push([c, r]); } return P; }
const PFREE = follow(E.snapQ.Qr, E.snapQ.Qc, 0);
const SHORTP = [[0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3]], DETP = [[0, 3], [1, 3], [1, 2], [1, 1], [2, 1], [3, 1], [4, 1], [4, 2], [4, 3], [5, 3]];
const BV = [0.25, 0.5, 1, 1.5, 2, 3];
const SUM = {}; const summ = b => SUM[b] ||= (() => { const o = runB(b), last = o.E.slice(NW - 300); const bot = last.filter(e => !isDet(e)), det = last.filter(isDet); return { o, cost: mean(last, e => e.cost), lam: mean(last, e => e.lam), share: bot.length, botC: bot.length ? mean(bot, e => e.cost) : 0, detC: det.length ? mean(det, e => e.cost) : 0, peak: Math.max(...o.E.map(e => e.lam)), lamEnd: o.lam }; })();
const f2 = v => L.num(v, 2), f3 = v => L.num(v, 3), f0 = v => String(Math.round(v)), f1 = v => (Math.round(v * 10) / 10).toFixed(1);

// ───── drawing
function glass(a = 1, cracked) { GL.forEach(([c, r], i) => { const x = OX + c * CS, y = OY + r * CS; box(x + 2, y + 2, CS - 4, CS - 4, '#cfe6f2', .85 * a);
  alpha(a * .8, () => { rline(x + 10, y + 12, x + 22, y + 22, 700 + i, { c: '#9cc5da', w: 1.2, once: true }); rline(x + 30, y + 38, x + 44, y + 30, 710 + i, { c: '#9cc5da', w: 1.2, once: true }); });
  if (cracked && cracked[i] > 0) alpha(clamp(cracked[i]), () => { const mx = x + CS / 2, my = y + CS / 2; [[-14, -10], [12, -14], [16, 8], [-8, 15], [2, -18]].forEach(([dx, dy], j) => rline(mx, my, mx + dx, my + dy, 720 + i * 7 + j, { c: C.roseInk, w: 1.1, once: true })); }); }); }
function room(t, o = {}) { glass(o.ga ?? 1, o.cracked); maze({ t, draw: o.draw }); if (o.labels !== 0) { const a = o.labels ?? 1; txt('+1', cx(5) - 14, cy(3) - 16, { s: 14, c: '#b07d12', a }); txt('start', cx(0), cy(3) - 18, { s: 12, c: C.soft, a }); } }
function dim(a = .62) { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, 285); ctx.restore(); }
function route(P, col, off, p = 1, seed = 500) { const n = (P.length - 1) * clamp(p); for (let i = 0; i < Math.ceil(n); i++) { const u = clamp(n - i), a = P[i], b = P[i + 1]; rline(cx(a[0]) + off, cy(a[1]) + off, lerp(cx(a[0]), cx(b[0]), u) + off, lerp(cy(a[1]), cy(b[1]), u) + off, seed + i, { c: col, w: 3, once: true, j: .6 }); } }
function along(P, t, dur) { t = Math.max(0, t); const k = Math.min(P.length - 1, Math.floor(t / dur)), u = ease((t % dur) / dur * 1.3), a = P[k], b = P[Math.min(P.length - 1, k + 1)];
  return { x: lerp(cx(a[0]), cx(b[0]), k >= P.length - 1 ? 0 : u), y: lerp(cy(a[1]), cy(b[1]), k >= P.length - 1 ? 0 : u), k, face: b[0] - a[0], done: k >= P.length - 1 }; }
// two stacked strips over walks WARM+1..NW: price (top) and glass per walk, average of the last 25 walks (bottom), budget dashed
const MA = (o, i) => { let s = 0, n = 0; for (let j = Math.max(WARM, i - 24); j <= i; j++) { s += o.E[j].cost; n++; } return s / n; };
function strips(o, n, budget, opt = {}) {
  const x0 = 70, x1 = 368, X = i => x0 + (i - WARM) / (NW - 1 - WARM) * (x1 - x0), ymax = opt.ymax ?? .6;
  const T = { y0: 336, y1: 296 }, B = { y0: 394, y1: 350 }, Yl = v => T.y0 - clamp(v / ymax) * (T.y0 - T.y1), Yc = v => B.y0 - clamp(v / 3) * (B.y0 - B.y1);
  rline(x0, T.y0, x1, T.y0, 950, { c: C.line, w: 1, once: true }); rline(x0, B.y0, x1, B.y0, 951, { c: C.line, w: 1, once: true });
  txt('price λ', x0 - 6, T.y1 + 20, { s: 13, c: C.goldInk, al: 'right' }); txt('0', x0 - 6, T.y0, { s: 12, c: C.soft, al: 'right' }); txt(f2(ymax), x0 - 6, T.y1 + 2, { s: 12, c: C.soft, al: 'right' });
  txt('glass', x0 - 6, B.y1 + 2, { s: 13, c: C.roseInk, al: 'right' }); txt('per walk', x0 - 6, B.y1 + 15, { s: 12, c: C.soft, al: 'right' }); txt('0', x0 - 6, B.y0, { s: 12, c: C.soft, al: 'right' });
  if (budget != null) { ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.ink; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, Yc(budget)); ctx.lineTo(x1, Yc(budget)); ctx.stroke(); ctx.restore(); txt(`budget ${budget}`, x1, Yc(budget) - 7, { s: 12, al: 'right' }); }
  const end = Math.min(NW - 1, WARM + n);
  ctx.save(); ctx.lineWidth = 1.8; ctx.strokeStyle = C.gold; ctx.beginPath(); for (let i = WARM; i <= end; i++) { const x = X(i), y = Yl(o.E[i].lam); i === WARM ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } ctx.stroke();
  ctx.strokeStyle = C.rose; ctx.beginPath(); for (let i = WARM; i <= end; i++) { const x = X(i), y = Yc(MA(o, i)); i === WARM ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } ctx.stroke(); ctx.restore();
  txt(`walk ${WARM + 1}`, x0, B.y0 + 9, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NW}`, x1, B.y0 + 9, { s: 12, c: C.soft, al: 'right' });
  return { X, Yl, Yc, end };
}

const STEPS = [
 { h: 'A room with a glass floor', p: ['A new room for Pip: 6 cells across, 4 up, no rocks, no cat. Start bottom left, cheese bottom right: +1 on the step onto it, which ends the walk; every other step pays 0; γ (gamma) = 0.9.', 'In the middle of the floor are four **glass panels**. Stepping on them changes nothing in Pip’s reward. That is exactly the problem.'],
   shot: [3, 9],
   draw(t) { room(t, { ga: clamp(t - 1.5), draw: clamp(t / 1.5), labels: clamp(t - 1) }); pip(cx(0), cy(3), 1, t);
     card(['6 × 4 room; cheese +1 on arrival, walk over', 'other steps 0;  γ = 0.9', { s: 'glass panels: reward does not care', c: C.soft }], t - 3, { gap: 1.1 }); } },
 { h: 'A second number: cost', p: ['We care about the glass: every step onto a panel risks breaking it. So we count a second signal, the **cost**: 1 per step onto glass, 0 otherwise.', 'Cost stays **separate** from reward: reward says how well Pip did, cost how much harm it risked. Ideally harm stays low even while Pip learns; we will see how far this method gets.'],
   draw(t) { room(t, { cracked: [0, 0, clamp((t - 1.5) * 2), 0] }); const m = along([[1, 3], [2, 3]], t - .5, 1); pip(m.x, m.y, 1, t);
     if (t > 1.5) bubble('cost 1', cx(2), cy(3) - 20, t - 1.6, C.roseInk);
     card(['reward: +1 for the cheese', { s: 'cost: 1 for each step onto glass', c: C.roseInk }, { s: 'two numbers, kept apart', c: C.soft }], t - 3, { gap: 1.2 }); } },
 { h: 'Two routes', p: [`The **shortcut** runs along the bottom: 5 steps, 2 of them on glass. The **detour** goes up and around: 9 steps, no glass.`, `Counted from the start with the 0.9 discount, the shortcut’s cheese is worth 0.9 to the 4th, ${f2(SHORT_R)}. The detour’s is 0.9 to the 8th, ${f2(DETOUR_R)}. Reward alone prefers the shortcut.`],
   shot: [4, 11],
   draw(t) { room(t); route(SHORTP, C.rose, 6, clamp((t - .5) / 2), 600); route(DETP, C.teal, -4, clamp((t - 2.5) / 3), 620); pip(cx(0), cy(3), 1, t);
     card([{ s: `shortcut: 5 steps, 2 on glass, reward ${f2(SHORT_R)}`, c: C.roseInk }, { s: `detour: 9 steps, no glass, reward ${f2(DETOUR_R)}`, c: '#3f8582' }, { s: 'reward counted from the start, with 0.9', c: C.soft }], t - 5.5, { gap: 1.2, fs: 16 }); } },
 { h: 'Pip ignores the glass', p: [`First Pip trains for ${WARM} walks with Q-learning on reward alone, as in earlier reels, one move in five random. It learns the shortcut.`, `Over walks ${WARM - 99} to ${WARM} it stepped on glass **${f1(FREECOST)} times per walk** on average: the 2 of the shortcut, plus random moves that land on glass.`],
   shot: [5, 12],
   draw(t) { const cr = [0, 0, 0, 0]; const m = along(PFREE, (t % 5) - .5, .5); PFREE.slice(0, m.k + 1).forEach(([c, r]) => { const i = GL.findIndex(([a, b]) => a === c && b === r); if (i >= 0) cr[i] = 1; });
     room(t, { cracked: cr }); trail(PFREE.slice(0, m.k + 1), .6); pip(m.x, m.y, m.face, t);
     card([`after ${WARM} walks: the shortcut`, `glass steps per walk: ${f1(FREECOST)}`, { s: '(walks 201 to 300, random moves included)', c: C.soft }], t - 2, { gap: 1.2 }); } },
 { h: 'Why not a fixed penalty?', p: [`We could subtract a fixed penalty w per glass step, counted plainly, undiscounted. The shortcut is then worth ${f2(SHORT_R)} − ${f2(SHORT_C)} × w, the detour ${f2(DETOUR_R)}.`, `Below w = ${f2(TIP)} the shortcut always wins, two glass steps every walk, random moves aside; above it the detour always wins, none. Only w exactly at the tie could mix routes, a knife-edge found only with all these numbers.`],
   shot: [4, 12],
   draw(t) { room(t, { labels: 0 }); dim(.9); const x0 = 70, x1 = 360, y0 = 230, y1 = 60, wmax = .3, X = w => x0 + w / wmax * (x1 - x0), Y = v => y0 - (v / .7) * (y0 - y1);
     rline(x0, y0, x1, y0, 960, { c: C.line, w: 1 }); rline(x0, y0, x0, y1, 961, { c: C.line, w: 1 }); txt('penalty w →', x1, y0 - 12, { s: 13, c: C.soft, al: 'right' }); txt('value from the start', x0, y1 - 22, { s: 13, c: C.soft, al: 'left' });
     [0, .1, .2, .3].forEach(w => txt(String(w), X(w), y0 + 14, { s: 12, c: C.soft }));
     const a = clamp((t - .5) / 2); rline(X(0), Y(SHORT_R), X(lerp(0, .3, a)), Y(Math.max(0, SHORT_R - SHORT_C * lerp(0, .3, a))), 962, { c: C.rose, w: 2.2, once: true }); rline(X(0), Y(DETOUR_R), X(lerp(0, .3, a)), Y(DETOUR_R), 963, { c: C.teal, w: 2.2, once: true });
     txt('shortcut', X(.03), Y(SHORT_R) + 22, { s: 13, c: C.roseInk, al: 'left', a }); txt('detour', X(.24), Y(DETOUR_R) - 10, { s: 13, c: '#3f8582', a });
     if (t > 3) { const b = clamp(t - 3); dot(X(TIP), Y(DETOUR_R), 4, C.ink, b); rline(X(TIP), Y(DETOUR_R), X(TIP), y0, 964, { c: C.ink, w: 1, once: true, p: b }); txt(`w = ${f2(TIP)}`, X(TIP) + 6, y0 - 12, { s: 13, al: 'left', a: b }); }
     card([`shortcut: ${f2(SHORT_R)} − ${f2(SHORT_C)} × w  (2 glass)`, `detour: ${f2(DETOUR_R)}  (0 glass)`, { s: 'one side or the other, except exactly at the tie', c: C.soft }], t - 4.5, { gap: 1.2, fs: 16 }); } },
 { h: 'Say what you want', p: ['**Constrained RL** states the goal directly: maximise reward, with the usual 0.9 discount, **subject to** the average cost per walk staying under a chosen **budget**.', `Ours: at most 1 glass step per walk, on average. The shortcut costs at least 2, the detour almost nothing, so meeting the budget best means taking the shortcut only some of the time. Pip must find that balance.`],
   draw(t) { room(t); dim(.6);
     txt('maximise  reward (discounted, 0.9)', W / 2, 90, { s: 22, a: clamp(t - .4) }); txt('subject to  average cost per walk  ≤  budget', W / 2, 130, { s: 19, a: clamp(t - 1.4), c: C.roseInk });
     txt('budget = 1 glass step per walk', W / 2, 176, { s: 17, a: clamp(t - 2.4) });
     card(['reward: what we want more of', { s: 'cost: what we must keep under a limit', c: C.roseInk }, { s: 'the limit holds on average, over many walks', c: C.soft }], t - 3.5, { gap: 1.2 }); } },
 { h: 'A price on glass', p: ['The trick: a **price** λ (lambda) per glass step: Pip maximises reward minus λ times cost. After each walk, λ rises by 0.01 times the overspend, glass steps minus budget, never below 0.', 'Overspend and the price rises; underspend and it falls. In maths, λ is a **Lagrange multiplier**, turning a limit into a price. The 0.01 is a knob: larger reacts faster but swings more; smaller overspends longer.'],
   draw(t) { room(t); dim(.65);
     txt('Pip maximises:  reward − λ × cost', W / 2, 70, { s: 21, a: clamp(t - .4) });
     txt('after each walk:', 60, 120, { s: 17, al: 'left', a: clamp(t - 1.4) }); txt('λ ← λ + 0.01 × (glass steps − budget)', 76, 150, { s: 18, al: 'left', a: clamp(t - 1.8), c: C.goldInk });
     txt('(never below 0)', 76, 176, { s: 14, al: 'left', a: clamp(t - 2.4), c: C.soft });
     card(['3 glass steps, budget 1: λ up by 0.02', '0 glass steps, budget 1: λ down by 0.01', { s: 'a price that rises while you overspend', c: C.goldInk }], t - 3.4, { gap: 1.2 }); } },
 { h: 'Two notebooks', p: ['Pip keeps **two numbers per move**: reward to come, discounted by 0.9, and glass steps to come, counted plainly like the budget. After each move, both are nudged, as in Q-learning, toward this step’s result plus the next number, scaled by 0.9 for reward, not for glass.', '“Next” means the move Pip would pick by its score: reward minus λ times glass. So the glass number reflects how Pip really plans to act, and a new price changes its choices at once.'],
   draw(t) { room(t); box(OX + CS + 3, OY + 3 * CS + 3, CS - 6, CS - 6, C.yellow, .35); pip(cx(0), cy(3), 1, t); const Q = RN;
     [[1, '#3f8582', 'right'], [0, '#3f8582', 'up']].forEach(([m, col, nm], i) => { const a = clamp(t - 1 - i * .6); if (a <= 0) return; alpha(a, () => G.arrowIn(1, 3, m, 740 + i, { c: m === 1 ? C.rose : C.teal, w: 2 })); });
     card([`yellow cell, “right”: reward ${f2(BR.r)}, glass ${f2(BR.c)}`, `“up”: reward ${f2(BU.r)}, glass ${f2(BU.c)}`, { s: 'score = reward − λ × glass', c: C.goldInk }], t - 2.5, { gap: 1.2, fs: 16 });
     caption(`numbers Pip ends up with after ${NW} walks`, clamp(t - 2.5)); } },
 { h: 'Reading the chart', p: [`The chart starts at walk ${WARM + 1}, when the price is switched on. Across: walks, up to ${NW}. Gold, on top: the price λ. Rose, below: glass steps per walk, averaged over the last 25 walks. The dashed line is the budget, 1.`, 'Watch the two lines push against each other.'],
   draw(t) { room(t); dim(.5); pip(cx(0), cy(3), 1, t); strips(RN, Math.floor(clamp((t - 2) / 8) * (NW - WARM)), BUD); } },
 { h: 'The price at work', p: [`At first Pip overspends and the price climbs, peaking at ${f2(PEAK)} on walk ${PEAKW}. The detour then wins and glass drops below budget, so the price drifts down until the shortcut sometimes wins again.`, `Over the last 300 walks the price averages ${f2(LASTLAM)}. Pip takes the bottom on ${SHARE} of them, about ${f1(BOTCOST)} glass steps each, and around on the rest, ${f1(DETCOST)} each from random moves: **${f2(LASTCOST)} per walk**, right on budget.`],
   shot: [5, 12],
   draw(t) { room(t); dim(.4); const n = NW - WARM, k = Math.floor(clamp(t / 9) * n), w = WARM + k, e = E[Math.min(NW - 1, w)];
     const P = e.tr, m = along(P, (t * 3) % (P.length * .25), .25); trail(P, .5); pip(m.x, m.y, m.face, t); strips(RN, k, BUD);
     caption(`walk ${Math.min(NW, w + 1)}: price ${f2(e.lam)}`, 0); txt(`walk ${Math.min(NW, w + 1)}, price ${f2(e.lam)}`, W / 2, 262, { s: 14, c: C.soft }); } },
 { h: 'The tipping point', p: [`At the cell where the routes split, “right” promises ${f2(BR.r)} reward and ${f2(BR.c)} glass; “up” ${f2(BU.r)} and ${f2(BU.c)}. The scores tie at a price of ${f2(BTIP)}. From the start it was ${f2(TIP)}: reward shrinks with distance, glass does not, so the tie depends on where you stand.`, `The price settles near there, ${f2(LASTLAM)} on average. Just below, Pip takes the shortcut; just above, the detour: small swings mix the routes.`],
   shot: [4, 11],
   draw(t) { room(t); box(OX + CS + 3, OY + 3 * CS + 3, CS - 6, CS - 6, C.yellow, .35); pip(cx(0), cy(3), 1, t); const lam = (BTIP) * (1 + .4 * Math.sin(t * 1.3)), sR = BR.r - lam * BR.c, sU = BU.r - lam * BU.c, best = sR > sU ? 1 : 0;
     G.arrowIn(1, 3, 1, 750, { c: best === 1 ? C.rose : C.line, w: best === 1 ? 2.6 : 1.2 }); G.arrowIn(1, 3, 0, 751, { c: best === 0 ? C.teal : C.line, w: best === 0 ? 2.6 : 1.2 });
     card([`try a price: ${f3(lam)}`, { s: `right: ${f2(BR.r)} − λ × ${f2(BR.c)} = ${f3(sR)}`, c: C.roseInk }, { s: `up: ${f2(BU.r)} − λ × ${f2(BU.c)} = ${f3(sU)}`, c: '#3f8582' }, { s: `equal at λ = ${f2(BTIP)}`, c: C.goldInk }], t - 1, { gap: 1, fs: 16 }); } },
 { h: 'Glass broken while learning', p: [`The budget holds on average, once the price settles. Before that, Pip stepped on glass a lot: ${GLASS_FREE} times in the first ${WARM} walks, ${GLASS_CON} more in the next ${NW - WARM}. Even on the detour, random moves hit glass ${f1(DETCOST)} times per walk.`, 'If one break is a disaster, averages are not enough. **Safe exploration** adds a **shield**, a hand-written rule blocking known-dangerous moves while Pip learns, or a safe starting habit.'],
   shot: [4, 11],
   draw(t) { room(t, { cracked: [1, 1, 1, 1].map((v, i) => clamp(t - .5 - i * .4)) }); pip(cx(0), cy(3), 1, t);
     card([`glass steps, walks 1 to ${WARM}: ${GLASS_FREE}`, `glass steps, walks ${WARM + 1} to ${NW}: ${GLASS_CON}`, { s: 'the budget limits the average, not each walk', c: C.roseInk }], t - 2, { gap: 1.2 }); } },
 { h: 'Your turn', ctl: { label: 'budget', values: BV, def: 2, fmt: v => v },
   p: [`Pick a budget, in glass steps per walk. The run repeats: ${WARM} free walks, then ${NW - WARM} with the price.`, 'Whenever routes must be mixed, the price settles near the tipping point; what changes is how often Pip takes the shortcut. A budget below even the detour’s cost cannot be met, and the price keeps climbing. A loose budget leaves the price near 0.'],
   shot: [3, 10],
   draw(t, st) { const b = BV[st.ci ?? 2], S = summ(b); room(t); dim(.35); const e = S.o.E[NW - 1], P = e.tr, m = along(P, (t * 3) % (P.length * .25 + 1), .25); pip(m.x, m.y, m.face, t);
     strips(S.o, Math.floor(clamp((t - (st.changed ?? 0)) / 4) * (NW - WARM)), b, { ymax: S.peak > 1.2 ? 2 : S.peak > .6 ? 1.2 : .6 });
     box(36, 18, 328, 96, C.card, .92); txt(`budget ${b}: price ${f2(S.lam)} on average, ${f2(S.lamEnd)} at the end`, 46, 34, { s: 15, al: 'left' }); txt(`glass per walk, last 300: ${f2(S.cost)}`, 46, 56, { s: 15, al: 'left', c: C.roseInk }); txt(`along the bottom: ${S.share} walks, ${f1(S.botC)} glass each`, 46, 78, { s: 15, al: 'left', c: C.soft }); txt(`around: ${300 - S.share} walks, ${f1(S.detC)} glass each`, 46, 100, { s: 15, al: 'left', c: C.soft }); } },
 { h: 'In the real world', p: ['Eitan Altman’s 1999 book set out **constrained MDPs**. In 2017, Joshua Achiam and colleagues proposed Constrained Policy Optimisation for deep RL, and in 2019 OpenAI released **Safety Gym**, simulated robots that must reach goals while avoiding hazards, where price-based methods like PPO-Lagrangian are standard baselines.', 'The same idea suits robots with limits on force or energy. Real deployments add many more safeguards.'],
   draw(t) { room(t); dim(.4); pip(cx(0), cy(3), 1, t);
     card(['1999  Altman: constrained MDPs', '2017  Achiam and colleagues: CPO', '2019  OpenAI Safety Gym; PPO-Lagrangian baseline'], t - .5, { gap: 1.1, fs: 16 }); } },
 { h: 'Limits to know', p: ['The constraint is about the **average**: single walks can still go over, and learning itself can break the limit, as we saw. Someone must choose the budget.', 'And cost is written by people, just like reward. A badly specified cost can be gamed too, as the reward hacking reel showed.'],
   draw(t) { room(t); pip(cx(0), cy(3), 1, t); dim(.3);
     card(['holds on average, not on every walk', 'learning breaks it before the price settles', { s: 'cost can be mis-specified, like reward', c: C.roseInk }], t - .5, { gap: 1.2 }); } },
 { h: 'Recap', p: ['Keep harm in its own signal, the **cost**, separate from reward. Constrained RL maximises reward while the average cost stays under a budget.', 'A **Lagrange multiplier** is a price on cost that rises while you overspend and falls while you underspend; it settles at the tipping point between options.'],
   draw(t) { room(t); const m = along(DETP, t % 5, .45); pip(m.x, m.y, m.face, t);
     card(['cost: a separate signal for harm', 'maximise reward, average cost ≤ budget', 'price λ: up when over, down when under', 'averages allow mistakes while learning'], t - .3, { gap: 1 }); } },
];
RLStory.register('saferl', { title: 'Safe & Constrained RL', track: 'Rewards', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { run, runB, BOTCOST, SHORT_R, DETOUR_R, SHORT_C, TIP, FREECOST, PEAK, PEAKW, LASTCOST, LASTLAM, SHARE, DETCOST, GLASS_FREE, GLASS_CON, BR, BU, BTIP, PFREE, summ, BV } });
})();
