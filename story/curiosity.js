// Curiosity & intrinsic motivation as a story: Pip in a big maze with one far-away cheese. Random exploring wanders;
// an inner reward for novelty (a count bonus, then a surprise bonus) sends Pip to fresh ground; a noisy TV traps surprise.
// Every number comes from the seeded runs below (see briefs/notes/curiosity.md).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, caption, dot, box, alpha } = L;
let ctx;
// ───── the big maze: 12 × 9 cells, walls between cells (seeded depth-first maze + 10 extra openings for loops)
const COLS = 12, ROWS = 9, N = COLS * ROWS, MV = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const START = (ROWS - 1) * COLS, CHEESE = COLS - 1, KV = 64, TV = 6 * COLS + 2;   // start bottom left, cheese top right, TV at column 3, row 7
const GAM = 0.9, ALPHA = 0.5, EPS = 0.1, BETA = 0.1, Q0 = BETA / (1 - GAM);       // Q0 = 1: an endless run of brand-new cells
const OPEN = (() => { const R = rng(3), open = Array.from({ length: N }, () => [0, 0, 0, 0]), seen = new Uint8Array(N), st = [0]; seen[0] = 1;
  while (st.length) { const k = st[st.length - 1], c = k % COLS, r = (k / COLS) | 0, opts = [];
    for (let m = 0; m < 4; m++) { const nc = c + MV[m][0], nr = r + MV[m][1]; if (nc < 0 || nr < 0 || nc >= COLS || nr >= ROWS) continue; const nk = nr * COLS + nc; if (!seen[nk]) opts.push([m, nk]); }
    if (!opts.length) { st.pop(); continue; } const [m, nk] = opts[Math.floor(R() * opts.length)]; open[k][m] = 1; open[nk][(m + 2) % 4] = 1; seen[nk] = 1; st.push(nk); }
  let e = 0; while (e < 10) { const k = Math.floor(R() * N), m = Math.floor(R() * 4), c = k % COLS, r = (k / COLS) | 0, nc = c + MV[m][0], nr = r + MV[m][1];
    if (nc < 0 || nr < 0 || nc >= COLS || nr >= ROWS || open[k][m]) continue; const nk = nr * COLS + nc; open[k][m] = 1; open[nk][(m + 2) % 4] = 1; e++; }
  return open; })();
const MS = new Int8Array(4);   // tied best moves (scratch)
const next = (s, a) => OPEN[s][a] ? s + MV[a][1] * COLS + MV[a][0] : s;
const DIST = (() => { const d = new Int16Array(N).fill(-1), q = [START]; d[START] = 0; while (q.length) { const k = q.shift(); for (let a = 0; a < 4; a++) { const n = next(k, a); if (d[n] < 0) { d[n] = d[k] + 1; q.push(n); } } } return d; })();
const SHORT = DIST[CHEESE], TVD = DIST[TV];

// ───── one run: Q-learning on (cheese reward + inner bonus); the walk restarts at the start after the cheese.
// mode: 'eps' (no bonus, guesses start at 0), 'count' (0.1 / √visits of the arrival cell), 'pred' (0.1 × prediction error)
function run(mode, seed, T, tv = -1) {
  const R = rng(seed), q0 = mode === 'eps' ? 0 : Q0, Q = new Float64Array(N * 4).fill(q0), cnt = new Uint32Array(N);
  // per (cell, move): belief in each view [usual view, static 1..64], stored as raw × scale so 'halve everything' is one multiply
  const P = mode === 'pred' ? new Float64Array(N * 4 * (KV + 1)) : null, SC = mode === 'pred' ? new Float64Array(N * 4).fill(1) : null;
  const path = new Uint8Array(T + 1), bon = new Float32Array(T + 1); let s = START; cnt[s] = 1; path[0] = s; let first = -1, vis = 1, tvT = 0; const cov = new Int16Array(T + 1); cov[0] = 1;
  for (let t = 0; t < T; t++) {
    let a; if (R() < EPS) a = Math.floor(R() * 4); else { let b = -1e9, n = 0; for (let m = 0; m < 4; m++) { const q = Q[s * 4 + m]; if (q > b + 1e-12) { b = q; MS[0] = m; n = 1; } else if (q > b - 1e-12) MS[n++] = m; } a = MS[Math.floor(R() * n)]; }
    const s2 = next(s, a); if (!cnt[s2]) vis++; cnt[s2]++; const ext = s2 === CHEESE ? 1 : 0; let bonus = 0;
    if (mode === 'count') bonus = BETA / Math.sqrt(cnt[s2]);
    else if (mode === 'pred') { const sa = s * 4 + a, off = sa * (KV + 1), v = s2 === tv ? 1 + Math.floor(R() * KV) : 0, err = 1 - P[off + v] * SC[sa];
      SC[sa] *= .5; P[off + v] += .5 / SC[sa]; bonus = BETA * err;   // the guess moves halfway toward what was seen
      if (SC[sa] < 1e-100) { for (let j = 0; j <= KV; j++) P[off + j] *= SC[sa]; SC[sa] = 1; } }
    if (s2 === tv) tvT++;
    let mx = 0; if (!ext) { mx = -1e9; for (let m = 0; m < 4; m++) mx = Math.max(mx, Q[s2 * 4 + m]); }
    Q[s * 4 + a] += ALPHA * (ext + bonus + GAM * mx - Q[s * 4 + a]);
    if (ext && first < 0) first = t + 1; s = ext ? START : s2; if (ext) cnt[s]++;
    path[t + 1] = ext ? CHEESE : s; bon[t + 1] = bonus; cov[t + 1] = vis;
  }
  return { first, path, bon, cov, tvT, vis, cnt, Q };
}
// ───── averages over 101 runs (seeds 1000–1100), at load
const SEEDS = 101, TMAX = 5000, TEPS = 10000, CURVE = 2000;
function stats(mode, T, tv) {
  const F = [], tvT = [], curve = new Float64Array(CURVE / 50 + 1);
  for (let i = 0; i < SEEDS; i++) { const r = run(mode, 1000 + i, T, tv); F.push(r.first < 0 ? Infinity : r.first); tvT.push(r.tvT);
    for (let j = 0; j <= CURVE / 50; j++) curve[j] += r.cov[j * 50] / SEEDS; }
  const med = a => a.slice().sort((x, y) => x - y)[(SEEDS - 1) / 2];
  return { medFirst: med(F), medTv: med(tvT), never: F.filter(f => f === Infinity).length, T, curve };
}
// counting ignores what Pip sees, so a TV changes nothing for it (same runs; checked in the notes)
const AV = { eps: stats('eps', TEPS), count: stats('count', TMAX), opt: stats('opt', TMAX), pred: stats('pred', TMAX), predTV: stats('pred', TMAX, TV) };
// story runs: one seed (1030, picked because every explorer behaves close to its median) shown for every explorer
const SS = 1030, TS = 3000;
const RUN = { eps: run('eps', SS, TS), count: run('count', SS, TS), pred: run('pred', SS, TS), predTV: run('pred', SS, TS, TV) };
const r0 = v => String(Math.round(v)), f2 = v => (+v.toFixed(2)).toString(), f3 = v => (+v.toFixed(3)).toString();
const visitsAt = (R, t) => { const c = new Int32Array(N); for (let k = 0; k <= t; k++) c[R.path[k]]++; return c; };
const E1000 = visitsAt(RUN.eps, 1000), EMAX = Math.max(...E1000), EMAXK = E1000.indexOf(EMAX), ESTART = E1000[START];
const TVB = (() => { let s = 0, n = 0; const R = RUN.predTV; for (let t = 1; t <= TS; t++) if (R.path[t] === TV) { s += R.bon[t]; n++; } return s / n; })();
const TVFIRST = (() => { for (let t = 1; t <= TS; t++) if (RUN.predTV.path[t] === TV) return t; return -1; })();
const sim = { TVD, SHORT, AV, RUN, run, TV, E1000: { EMAX, EMAXK, ESTART }, TVB, TVFIRST };

// ───── drawing
const CS = 28, OX = 32, OY = 20, cx = k => OX + (k % COLS) * CS + CS / 2, cy = k => OY + ((k / COLS) | 0) * CS + CS / 2;
const TEAL = '#3f8582', GOLDI = '#8a5d08', ROSEI = '#a54444';
function walls(p = 1) {
  const x1 = OX + COLS * CS, y1 = OY + ROWS * CS; let sd = 400;
  for (let k = 0; k < N; k++) { const c = k % COLS, r = (k / COLS) | 0, x = OX + c * CS, y = OY + r * CS;
    if (c < COLS - 1 && !OPEN[k][1]) rline(x + CS, y, x + CS, y + CS, sd++, { w: 1.6, j: .8, once: true, p: clamp(p * 2 - r / ROWS) });
    if (r < ROWS - 1 && !OPEN[k][2]) rline(x, y + CS, x + CS, y + CS, sd++, { w: 1.6, j: .8, once: true, p: clamp(p * 2 - r / ROWS) }); }
  rrect(OX, OY, COLS * CS, ROWS * CS, 390, { w: 2 });
}
function tvSet(k, t, a = 1) { const x = cx(k) - 3, y = cy(k) - 3; alpha(a, () => { box(x - 10, y - 9, 20, 15, '#5b544d');
  const R = rng(Math.floor(t * 10) + 7); for (let i = 0; i < 18; i++) box(x - 8 + Math.floor(R() * 8) * 2, y - 7 + Math.floor(R() * 6) * 2, 2, 2, R() < .5 ? '#f4efe6' : '#a8a096');
  rline(x - 5, y + 6, x - 8, y + 11, 391, { once: true, w: 1 }); rline(x + 5, y + 6, x + 8, y + 11, 392, { once: true, w: 1 }); }); }
// maze scene. o = { cnt (visits per cell), cmax, pip (cell), face, t, tv (cell), tvA, labels, route, draw, mark: [[cell, text, colour]] }
function scene(o = {}) {
  const t = o.t || 0;
  if (o.cnt) for (let k = 0; k < N; k++) if (o.cnt[k]) box(OX + (k % COLS) * CS + 1, OY + ((k / COLS) | 0) * CS + 1, CS - 2, CS - 2, C.teal, .1 + .8 * clamp(Math.log(o.cnt[k]) / Math.log(o.cmax || 60)));
  if (o.route) { const pts = routePts(); const n = Math.floor(clamp(o.route) * (pts.length - 1)); ctx.save(); ctx.strokeStyle = C.gold; ctx.lineWidth = 2.4; ctx.setLineDash([4, 4]); ctx.beginPath();
    pts.slice(0, n + 1).forEach((k, i) => i ? ctx.lineTo(cx(k), cy(k)) : ctx.moveTo(cx(k), cy(k))); ctx.stroke(); ctx.restore(); }
  walls(o.draw ?? 1);
  if ((o.draw ?? 1) >= .8) L.cheese(cx(CHEESE), cy(CHEESE) + 1, .6);
  if (o.tv != null && o.tv >= 0) tvSet(o.tv, t, o.tvA ?? 1);
  if (o.labels) { txt('start', cx(START), OY + ROWS * CS + 10, { s: 13, c: C.soft, a: o.labels }); txt('cheese: +1 on arrival', cx(CHEESE) - 50, OY - 10, { s: 13, c: '#b07d12', a: o.labels }); }
  if (o.mark) o.mark.forEach(([k, s, col]) => txt(s, cx(k), cy(k) + 1, { s: 13, c: col || C.ink }));
  if (o.pip != null) { const onTv = o.pip === o.tv; L.mouse(cx(o.pip) + (onTv ? 8 : 2 * (o.face || 1)), cy(o.pip) + (onTv ? 7 : 2), o.face || 1, onTv ? .4 : .5, t); }
}
let RP = null; function routePts() { if (RP) return RP; const prev = new Int16Array(N).fill(-1), q = [START]; prev[START] = START;
  while (q.length) { const k = q.shift(); for (let a = 0; a < 4; a++) { const n = next(k, a); if (prev[n] < 0) { prev[n] = k; q.push(n); } } }
  const out = [CHEESE]; while (out[0] !== START) out.unshift(prev[out[0]]); return (RP = out); }
// replay run R from step 0 to step k: visits so far, Pip's cell and facing (memo per run)
const memo = new Map();
function replay(R, k) { k = clamp(Math.floor(k), 0, R.path.length - 1); let m = memo.get(R); if (!m || m.k > k) { m = { k: 0, c: new Int32Array(N), face: 1 }; m.c[R.path[0]] = 1; memo.set(R, m); }
  while (m.k < k) { m.k++; const a = R.path[m.k - 1], b = R.path[m.k]; m.c[b]++; const dx = (b % COLS) - (a % COLS); if (dx) m.face = Math.sign(dx); }
  return { c: m.c, pip: R.path[k], face: m.face, k, seen: R.cov[k] }; }
// play run R up to step K over dur seconds starting at t0
function playRun(R, t, t0, K, dur, o = {}) { const k = Math.floor(clamp((t - t0) / dur) * K), s = replay(R, k);
  scene(Object.assign({ cnt: s.c, cmax: o.cmax || 60, pip: s.pip, face: s.face, t }, o)); return s; }
// coverage chart: x = step 0..2000, y = cells seen 0..108
const CH = { x0: 60, x1: 360, y0: 250, y1: 40 };
const chX = s => CH.x0 + s / CURVE * (CH.x1 - CH.x0), chY = v => CH.y0 - v / N * (CH.y0 - CH.y1);
function axes(a = 1) { alpha(a, () => {
  rline(CH.x0, CH.y0, CH.x1 + 6, CH.y0, 601, { w: 1.2 }); rline(CH.x0, CH.y0, CH.x0, CH.y1 - 8, 602, { w: 1.2 });
  [0, 54, 108].forEach(v => { txt(String(v), CH.x0 - 8, chY(v), { s: 12, c: C.soft, al: 'right' }); if (v) rline(CH.x0, chY(v), CH.x1, chY(v), 610 + v, { c: '#e9dfd0', w: 1, once: true, j: .3 }); });
  [0, 1000, 2000].forEach(v => txt(String(v), chX(v), CH.y0 + 12, { s: 12, c: C.soft }));
  txt('steps', CH.x1 - 10, CH.y0 + 27, { s: 13, c: C.soft }); txt('cells seen (of 108)', CH.x0 - 4, CH.y1 - 22, { s: 13, c: C.soft, al: 'left' }); }); }
function curve(A, col, p, label, lx, ly) { const n = Math.floor(clamp(p) * (A.length - 1)); if (n < 1) return;
  ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.lineJoin = 'round'; ctx.beginPath(); ctx.moveTo(chX(0), chY(A[0]));
  for (let j = 1; j <= n; j++) ctx.lineTo(chX(j * 50), chY(A[j])); ctx.stroke(); ctx.restore();
  if (p >= 1 && label) txt(label, lx, ly, { s: 14, c: col, al: 'left' }); }
// horizontal bars of median steps to the first cheese
function medBars(rows, t, o = {}) { const x0 = 130, x1 = 350, max = o.max || 2000, y0 = o.y0 ?? 60, gap = o.gap ?? 52;
  rows.forEach(([name, v, col, note], i) => { const a = clamp((t - .5 - i * 1.2) * 2); if (a <= 0) return; const y = y0 + i * gap, w = (x1 - x0) * Math.min(v, max) / max * ease(clamp((t - .5 - i * 1.2) / 1));
    txt(name, x0 - 10, y, { s: 15, al: 'right', c: col, a }); box(x0, y - 11, w, 22, col, .75 * a); txt(r0(v), x0 + w + 6, y, { s: 15, al: 'left', c: col, a });
    if (note) txt(note, x0, y + 21, { s: 12, al: 'left', c: C.soft, a }); });
  alpha(clamp(t - .5), () => rline(130, y0 - 20, 130, y0 + gap * rows.length - 24, 640, { c: C.line, w: 1 })); }
const dim = (a = .75) => box(0, 0, W, 285, C.bg, a);

// your turn: tap a cell to put the TV there; slider picks the explorer
const YT = new Map();
function yourRun(mode, tv) { const key = mode + tv; if (!YT.has(key)) YT.set(key, run(mode, SS, TS, tv)); return YT.get(key); }
function cellAt(x, y) { const c = Math.floor((x - OX) / CS), r = Math.floor((y - OY) / CS); if (c < 0 || r < 0 || c >= COLS || r >= ROWS) return -1; return r * COLS + c; }

const A = AV, RE = RUN.eps, RC = RUN.count, RPd = RUN.pred, RT = RUN.predTV;
const STEPS = [
 { h: 'A much bigger maze', p: [`Pip moves to a big maze: 12 columns by 9 rows, 108 cells, with walls between them. The cheese is top right, ${SHORT} steps away by the shortest route. No cat here.`, 'Same rules: **+1 on the step onto the cheese**, then Pip restarts at the bottom left. Other steps pay 0. Bumping a wall leaves Pip in place but still counts as a step.'],
   get say() { return [["Welcome to a much bigger maze."], [["A hundred and eight cells, with walls between them.", [26, 14, 348, 262]], `The cheese is top right, ${SHORT} steps away by the shortest route. The gold dashes show it.`], ["No cat here. Just one plus one, on the step onto the cheese."]]; },
   draw(t) { scene({ draw: clamp(t / 2.5), labels: clamp(t - 2.5), pip: t > 2 ? START : null, t, route: t > 6 ? clamp((t - 6) / 2) : 0 });
     if (t > 8.5) card([`gold dashes: the shortest route, ${SHORT} steps`, { s: 'only we can see it: Pip has no map', c: C.soft }], t - 8.5, { gap: 1.1 }); }, shot: [4, 11] },
 { h: 'A reward that almost never comes', p: ['Until Pip first steps onto the cheese, every step pays 0: a **sparse reward**, as in the shaping reel. Why not add hints? With walls everywhere, "closer as the crow flies" can point into dead ends, and the designer may not know the maze at all.', 'So Pip learns nothing until it stumbles onto the cheese once. Everything depends on how it explores before that.'],
   say: [["Until Pip first finds the cheese, every step pays nothing. A sparse reward."], ["Why not add hints? With walls everywhere, closer as the crow flies can point into dead ends. And the designer may not know the maze at all."], [["So everything depends on how Pip explores before that first cheese.", [30, 292, 340, 87]]]],
   draw(t) { const R = rng(5), ks = []; for (let k = 0; k < N; k++) if (k !== CHEESE && k !== START) ks.push(k); ks.sort(() => R() - .5);
     scene({ pip: START, t, labels: 1, mark: ks.slice(0, Math.floor(clamp((t - .8) / 4) * ks.length)).map(k => [k, '0', C.soft]) });
     if (t > 6) card(['every step: reward 0', { s: 'the cheese step: +1, the only signal in the maze', c: GOLDI }], t - 6, { gap: 1.1 }); }, shot: [5, 10] },
 { h: 'Pip’s learner', p: ['Pip keeps a guess per cell and move. Each step moves that guess halfway toward the reward plus 0.9 times the new cell’s best guess: **Q-learning**, γ (gamma) 0.9. One move in 10 is random, the rest are best-guess: **ε-greedy**, ε (epsilon) 0.1.', 'Step size and ε differ from earlier reels. Guesses start at 0 and stay 0 without reward; ties are broken at random.'],
   say: [["Pip learns with Q-learning, one guess per cell and move."], [["One move in ten is random. The rest follow the best guess.", [30, 292, 340, 87]]], ["Every guess starts at zero, and stays zero without reward. So at first, every choice is a tie, broken at random."]],
   draw(t) { const x = 200, y = 130, s = 110; rrect(x - s / 2, y - s / 2, s, s, 700, { fill: C.card, w: 1.6 }); L.mouse(x, y + 4, 1, .9, t);
     [[0, -1], [1, 0], [0, 1], [-1, 0]].forEach(([dx, dy], m) => { const a = clamp(t - 1 - m * .4); if (a <= 0) return;
       alpha(a, () => { arrow(x + dx * 30, y + dy * 30 + 3, x + dx * 62, y + dy * 62 + 3, 710 + m, { c: GOLDI, w: 1.6 }); }); txt('0', x + dx * 82, y + dy * 78 + 3, { s: 18, c: C.soft, a }); });
     if (t > 3.5) txt('guess for each move: 0', x, y + 98, { s: 15, c: C.soft, a: clamp(t - 3.5) });
     if (t > 6) card(['all four guesses equal: a tie', { s: 'so before the cheese, every move is random', c: ROSEI }], t - 6, { gap: 1.1 }); }, shot: [4, 10] },
 { h: 'Wandering in circles', p: [`Watch 1000 steps of this random walk, sped up. Shading shows how often Pip stood in a cell: the darker, the more visits.`, `After 1000 steps Pip has seen only ${RE.cov[1000]} of the 108 cells. It stood on the start cell ${ESTART} times and on one cell nearby ${EMAX} times. Random steps keep undoing each other, so Pip circles near home.`],
   say: [["Watch a thousand steps of this random walk, sped up."], [["The darker the cell, the more often Pip stood there.", [26, 14, 348, 262]], ["Look how it piles up near home.", [26, 14, 348, 262]]], ["Random steps keep undoing each other, so Pip circles near the start."]],
   draw(t) { const s = playRun(RE, t, .6, 1000, 8, { cmax: 64 }); caption(`step ${s.k} · cells seen ${s.seen} of 108`);
     if (t > 9) card([`start cell: ${ESTART} visits   darkest cell: ${EMAX}`, { s: 'the top rows, where the cheese is: untouched', c: C.soft }], t - 9, { gap: 1.1, y: 296 }); }, shot: [5, 12] },
 { h: 'Found, eventually', p: [`Pip keeps wandering. It finally steps onto the cheese at step ${RE.first}, more than a hundred times the ${SHORT} steps of the shortest route.`, `Was this run unlucky? A bit, but it was typical. Over ${SEEDS} runs with different luck, the middle run, the **median**, needs ${r0(A.eps.medFirst)} steps to find the cheese the first time.`],
   get say() { return [["Pip keeps wandering."], [`It finally finds the cheese at step ${RE.first}.`, ["More than a hundred times the shortest route.", [26, 14, 348, 262]]], ["Unlucky? A bit. But over many runs, a typical Pip still needs well over a thousand steps."]]; },
   draw(t) { const K = RE.first, s = playRun(RE, t, .4, K, 7, { cmax: 64 }); caption(s.k >= K ? `cheese at step ${K}!` : `step ${s.k} · cells seen ${s.seen}`);
     if (t > 8) card([`this run: cheese at step ${RE.first}`, `median of ${SEEDS} runs: ${r0(A.eps.medFirst)} steps`], t - 8, { gap: 1.1 }); }, shot: [4, 11] },
 { h: 'A reward from inside', p: ['Children explore because new things are interesting, not for pay. Give Pip the same drive: an **intrinsic reward**, a bonus Pip pays itself for going somewhere new.', 'The cheese is an **extrinsic reward**, from the world. Pip learns from the sum, with the same learner. Doesn’t a bonus change what is best, as in the shaping reel? Yes, on purpose. We come back to that.'],
   say: [["Children explore because new things are interesting, not because someone pays them."], ["So let's give Pip the same drive. An intrinsic reward, a bonus it pays itself for going somewhere new."], [["The cheese is an extrinsic reward, from the world. Pip learns from the sum.", [30, 292, 340, 87]]]],
   draw(t) { scene({ pip: START, t, labels: 1 }); dim(.55); L.mouse(150, 150, 1, 1.4, t);
     alpha(clamp(t - 1), () => { rrect(196, 70, 150, 58, 720, { fill: C.card, w: 1.3 }); txt('somewhere new!', 271, 90, { s: 17, c: TEAL }); txt('+ a little bonus', 271, 112, { s: 15, c: TEAL }); rline(196, 120, 172, 138, 721, { w: 1.2 }); });
     if (t > 4) card([{ s: 'extrinsic: from the world (cheese +1)', c: GOLDI }, { s: 'intrinsic: from Pip itself (novelty bonus)', c: TEAL }, { s: 'Pip learns from the sum of the two', c: C.soft }], t - 4, { gap: 1.1 }); }, shot: [3, 9] },
 { h: 'Counting visits', p: ['The simplest novelty bonus: Pip counts its visits to each cell. On arrival it pays itself **0.1 divided by the square root of the count**, this visit included.', 'The bars show the first ten visits: 0.1 on the first, 0.05 on the fourth. The hundredth pays 0.01. Familiar cells get boring, but slowly; unvisited cells keep the full 0.1.'],
   say: [["The simplest novelty bonus counts visits."], [["Watch the bars. The first visit to a cell pays the most.", [30, 14, 340, 240]], ["Each later visit pays a bit less. Familiar cells get boring, but slowly.", [30, 14, 340, 240]]], ["Cells Pip has never visited keep the full bonus."]],
   draw(t) { const vals = []; for (let n = 1; n <= 10; n++) vals.push(BETA / Math.sqrt(n));
     const n = Math.floor(clamp((t - 1) / 5) * 10 + .0001); txt('bonus on arrival', 40, 24, { s: 14, al: 'left', c: C.soft });
     for (let i = 0; i < n; i++) { const x = 52 + i * 31, h = vals[i] / .1 * 150; box(x, 230 - h, 22, h, C.teal, .8); txt(f3(vals[i]), x + 11, 220 - h, { s: 12, c: TEAL }); txt(String(i + 1), x + 11, 242, { s: 13, c: C.soft }); }
     rline(44, 230, 360, 230, 730, { c: C.line, w: 1 }); txt('visit number', 200, 262, { s: 13, c: C.soft });
     if (t > 7) card(['bonus = 0.1 ÷ √(visits so far)', { s: 'visit 1: 0.1 · visit 4: 0.05 · visit 100: 0.01', c: C.soft }], t - 7, { gap: 1.1 }); }, shot: [4, 11] },
 { h: 'Untried moves look promising', p: [`An untried move starts with the guess ${f2(Q0)}, not 0. Why? Endless new cells, each paying 0.1, are worth 0.1, plus 0.9 times 0.1, plus 0.81 times 0.1, and so on: exactly 1.`, 'So "untried" means "maybe fresh ground": optimism, as in the bandit reels. The surprise explorer later starts at 1 too. The cheese-only random walker started at 0.'],
   say: [["Here's a subtle point. An untried move doesn't start at zero. It starts at one."], [["Why? Endless new cells, each paying a bonus, add up to exactly one.", [30, 40, 340, 210]]], ["So untried means maybe fresh ground. That's optimism, like in the bandit reels."]],
   draw(t) { const terms = [1, .9, .81, .729, .656, .59]; txt('a run of brand-new cells:', 40, 50, { s: 15, al: 'left', c: C.soft });
     terms.forEach((g, i) => { const a = clamp((t - .8 - i * .6) * 2); if (a <= 0) return; const x = 30 + i * 48; rrect(x, 70, 42, 40, 740 + i, { fill: C.card, w: 1.2 }); txt('0.1', x + 21, 84, { s: 14, c: TEAL, a });
       txt('× ' + g, x + 21, 100, { s: 12, c: C.soft, a }); });
     if (t > 4.6) txt('… forever', 345, 90, { s: 14, c: C.soft, a: clamp(t - 4.6) });
     if (t > 5.5) txt('total = 0.1 ÷ (1 − 0.9) = 1', 200, 160, { s: 20, c: TEAL, a: clamp(t - 5.5) });
     if (t > 8) card([`untried move: guess starts at ${f2(Q0)}`, { s: 'tried moves: learned from what really happened', c: C.soft }], t - 8, { gap: 1.1 }); }, shot: [4, 11] },
 { h: 'The counting explorer', p: [`Same maze, same luck seed. Now each visit makes a cell a little more boring, so Pip’s guesses point toward the least visited parts of the maze, and the shading spreads like a flood.`, `This Pip steps onto the cheese at step ${RC.first}, after seeing ${RC.cov[RC.first]} cells. By that step the random walker had seen ${RE.cov[RC.first]}; it needed ${RE.first} steps for the cheese.`],
   get say() { return [["Same maze, same luck, now with the counting bonus."], [["Watch the shading spread, like a flood.", [26, 14, 348, 262]], "Pip's guesses keep pointing toward the least visited parts of the maze."], [`This Pip finds the cheese at step ${RC.first}. The random walker needed ${RE.first}.`]]; },
   draw(t) { const K = RC.first, s = playRun(RC, t, .5, K, 7, { cmax: 64 }); caption(s.k >= K ? `cheese at step ${K}!` : `step ${s.k} · cells seen ${s.seen} of 108`);
     if (t > 8.5) card([{ s: `counting explorer: cheese at step ${RC.first}`, c: TEAL }, { s: `random walker: step ${RE.first}`, c: GOLDI }], t - 8.5, { gap: 1.1, y: 296 }); }, shot: [4, 11] },
 { h: 'A chart of exploring', p: [`One run proves little, so here are averages over ${SEEDS} runs. The chart: left to right, steps taken. Upward, how many of the 108 cells Pip has seen at least once by then. A line that reaches the top has seen the whole maze.`, 'The grey line shows the fastest possible case, one new cell per step.'],
   say: [["One run proves little. So let's average many."], [["Across is steps taken. Up is how many cells Pip has seen.", [40, 30, 330, 240]], ["A line that reaches the top has seen the whole maze.", [40, 30, 330, 240]]], ["Grey shows the fastest possible case, one new cell every step."]],
   draw(t) { axes(clamp(t - .5)); if (t > 4) alpha(clamp(t - 4), () => { rline(chX(0), chY(1), chX(107), chY(108), 650, { c: '#d9cdbc', w: 2 }); rline(chX(107), chY(108), chX(2000), chY(108), 651, { c: '#d9cdbc', w: 2 }); txt('one new cell per step', chX(150), chY(100), { s: 13, c: C.soft, al: 'left' }); });
     if (t > 6.5) card(['across: steps, 0 to 2000', `up: cells seen, average of ${SEEDS} runs`], t - 6.5, { gap: 1.1 }); }, shot: [4, 10] },
 { h: 'Counting covers the maze', p: [`Gold, the random walker: after 1000 steps it has seen ${r0(A.eps.curve[20])} cells on average. Teal, the counting explorer: ${r0(A.count.curve[20])}, almost all of them.`, `The random walker is still at ${r0(A.eps.curve[40])} after 2000 steps. Being pulled toward rarely visited cells turns wandering into a sweep.`],
   say: [[["Gold is the random walker. It creeps along.", [40, 30, 330, 240]], ["Teal is the counting explorer. It sees almost every cell in a thousand steps.", [40, 30, 330, 240]]], ["Being pulled toward rare cells turns wandering into a sweep."]],
   draw(t) { axes(); curve(A.eps.curve, GOLDI, (t - .5) / 3, 'random walker', chX(1150), chY(A.eps.curve[23]) + 18);
     curve(A.count.curve, TEAL, (t - 4) / 3, 'counting explorer', chX(1150), chY(A.count.curve[23]) + 20);
     if (t > 7.5) card([{ s: `step 1000:  counting ${r0(A.count.curve[20])},  random ${r0(A.eps.curve[20])}`, c: C.ink }, { s: `average cells seen, ${SEEDS} runs`, c: C.soft }], t - 7.5, { gap: 1.1 }); }, shot: [4, 11] },
 { h: 'Time to the first cheese', p: [`Now the race to the first cheese. Each bar is the median over ${SEEDS} runs. Random walker: ${r0(A.eps.medFirst)} steps. Counting explorer: ${r0(A.count.medFirst)}, more than five times faster.`, `Is it just the hopeful start of 1? With that start but no bonus, the median is ${r0(A.opt.medFirst)}: it helps, but those hopes fade once each move is tried. The bonus keeps pulling Pip onward.`],
   say: [["Now the race to the first cheese."], [["Each bar is the middle run of many.", [100, 40, 270, 200]], ["The counting explorer is more than five times faster.", [100, 40, 270, 200]]], ["Is it just the hopeful start? Without the bonus, those hopes fade once each move is tried. The bonus keeps pulling Pip onward."]],
   draw(t) { medBars([['random', A.eps.medFirst, GOLDI], ['counting', A.count.medFirst, TEAL], ['hopeful start only', A.opt.medFirst, C.soft]], t, { y0: 80, gap: 62 });
     if (t > 5.5) card([`median steps to the first cheese, ${SEEDS} runs`, { s: `shortest route: ${SHORT} steps`, c: C.soft }], t - 5.5, { gap: 1.1 }); }, shot: [5, 11] },
 { h: 'Curiosity as surprise', p: ['Pathak and colleagues, in 2017, paid for surprise instead. For each cell and move, Pip keeps **chances** for what it sees next, here the cell it lands in. The **prediction error** is 1 minus the chance given to what it saw. Bonus: 0.1 times that.', 'Each move’s chances shift halfway toward the outcome. A new move has error 1, then 0.5, 0.25, 0.125.'],
   say: [["In twenty seventeen, Pathak and colleagues paid for surprise instead."], [["Pip keeps chances for where each move will land. The less it expected what it saw, the bigger the bonus.", [30, 292, 340, 87]]], ["A new move is a full surprise. Then a half, then a quarter, as Pip learns."]],
   draw(t) { const errs = [1, .5, .25, .125, .0625]; txt('error on the 1st, 2nd, 3rd … try of one move (chance 0, 0.5, 0.75 …)', 40, 50, { s: 14, al: 'left', c: C.soft });
     errs.forEach((e, i) => { const a = clamp((t - 1 - i * .9) * 2); if (a <= 0) return; const x = 70 + i * 60, h = e * 150 * ease(a); box(x, 230 - h, 34, h, C.orange, .8); txt(String(e), x + 17, 220 - h, { s: 13, c: '#b8602c', a }); txt(String(i + 1), x + 17, 242, { s: 13, c: C.soft, a }); });
     rline(56, 230, 370, 230, 760, { c: C.line, w: 1 });
     if (t > 6.5) card(['bonus = 0.1 × prediction error', { s: 'error = 1 − chance Pip gave to what it saw', c: C.soft }], t - 6.5, { gap: 1.1 }); }, shot: [4, 10] },
 { h: 'Surprise works too', p: [`Same seed, the surprise explorer finds the cheese at step ${RPd.first}. Its median over ${SEEDS} runs: ${r0(A.pred.medFirst)} steps, slower than counting’s ${r0(A.count.medFirst)}, far faster than the random walker’s ${r0(A.eps.medFirst)}.`, 'Moves here are not random, so surprise is a count per cell and move, halving each try. It likely lags because it fades faster than the square root and is kept per move, not per cell.'],
   get say() { return [[`Same seed, the surprise explorer finds the cheese at step ${RPd.first}.`, ["Over many runs, it's slower than counting, but far faster than random.", [30, 296, 340, 87]]], ["Here, surprise is basically a count per move, halving each try. It fades faster than counting does."]]; },
   draw(t) { const K = RPd.first, s = playRun(RPd, t, .5, K, 7, { cmax: 64 }); caption(s.k >= K ? `cheese at step ${K}!` : `step ${s.k} · cells seen ${s.seen} of 108`);
     if (t > 8.5) card([{ s: `surprise explorer: median ${r0(A.pred.medFirst)} steps`, c: '#b8602c' }, { s: `counting ${r0(A.count.medFirst)} · random ${r0(A.eps.medFirst)}`, c: C.soft }], t - 8.5, { gap: 1.1, y: 296 }); }, shot: [4, 11] },
 { h: 'The noisy TV', p: [`A TV now sits ${TVD} steps from the start, showing static: one of 64 random patterns each time. There Pip must also predict the picture, and no chance beats 1 in 64.`, `So the error stays near 1: the bonus there averages ${f3(TVB)} forever. Pip reaches the TV at step ${TVFIRST} and mostly stays: each wall bump keeps it there, with a fresh picture.`],
   say: [["Now a twist. A TV sits in the maze, showing random static."], [["On that cell, Pip must also predict the picture. And no one can predict static.", [26, 14, 348, 262]]], [["So the surprise never fades. Watch Pip reach the TV, and stay.", [26, 14, 348, 262]]]],
   draw(t) { const s = playRun(RT, t, .5, TS, 11, { cmax: 300, tv: TV }); caption(`step ${s.k} · steps on the TV cell ${tvSoFar(s.k)}`);
     if (t < 3) txt('TV', cx(TV), cy(TV) - 22, { s: 14, c: ROSEI, a: clamp(t) });
     if (t > 11.8) card([`bonus on the TV cell: about ${f3(TVB)}, every time`, { s: `first visit to the TV: step ${TVFIRST}`, c: C.soft }], t - 11.8, { gap: 1.1, y: 296 }); }, shot: [3, 14] },
 { h: 'Trapped by noise', p: [`In 3000 steps this Pip spent ${RT.tvT} on the TV cell, saw only ${RT.cov[TS]} cells, and never found the cheese. Over ${SEEDS} runs of 5000 steps, median time to cheese: ${r0(A.predTV.medFirst)}, with the ${A.predTV.never} failed runs counted as over 5000.`, 'Counting is not fooled: it counts cells, not pictures. Counting each picture as a new place would make static always new, and trap counting too.'],
   get say() { return [[`In three thousand steps, this Pip spent ${RT.tvT} on the TV cell.`, ["It saw only a small part of the maze, and never found the cheese.", [30, 292, 340, 87]]], ["Counting isn't fooled. It counts cells, not pictures. But count each picture as a new place, and static would trap counting too."]]; },
   draw(t) { medBars([['random', A.eps.medFirst, GOLDI], ['counting', A.count.medFirst, TEAL], ['surprise', A.pred.medFirst, '#b8602c'], ['surprise + TV', A.predTV.medFirst, ROSEI]], t, { y0: 60, gap: 54, max: 3000 });
     if (t > 6) card([`median steps to the first cheese, ${SEEDS} runs`, { s: `surprise + TV: ${A.predTV.never} of ${SEEDS} runs never found it in 5000`, c: ROSEI }], t - 6, { gap: 1.1 }); }, shot: [6, 12] },
 { h: 'Can a bonus be farmed?', p: ['After the hacking reel, a fair question: can Pip farm its own bonus? A count bonus fades: the hundredth visit pays only 0.01. Farming pays less and less, so in the end the cheese’s +1 wins.', `A surprise bonus fades only if the world is predictable. Noise never is, so it pays about ${f3(TVB)} forever. The noisy TV is reward hacking of an inner reward.`],
   say: [["After the hacking reel, a fair question. Can Pip farm its own bonus?"], [["A count bonus fades. Farming it pays less and less, until the cheese wins.", [30, 292, 340, 87]]], ["A surprise bonus fades only if the world is predictable. Noise never is. The noisy TV is reward hacking of an inner reward."]],
   draw(t) { const x0 = 60, x1 = 360, y0 = 230, y1 = 60, n = 100, X = v => x0 + (v - 1) / (n - 1) * (x1 - x0), Y = v => y0 - v / .1 * (y0 - y1);
     rline(x0, y0, x1 + 6, y0, 1601, { w: 1.2 }); rline(x0, y0, x0, y1 - 8, 1602, { w: 1.2 }); ['0', '0.05', '0.1'].forEach((l, i) => txt(l, x0 - 8, Y(i * .05), { s: 12, c: C.soft, al: 'right' }));
     [1, 50, 100].forEach(v => txt(String(v), X(v), y0 + 12, { s: 12, c: C.soft })); txt('visit number', x1 - 30, y0 + 27, { s: 13, c: C.soft }); txt('bonus on that visit', x0 - 4, y1 - 22, { s: 13, c: C.soft, al: 'left' });
     const p1 = clamp((t - 1) / 3), p2 = clamp((t - 4.5) / 3); ctx.save(); ctx.lineWidth = 2.4;
     if (p1 > 0) { ctx.strokeStyle = TEAL; ctx.beginPath(); for (let v = 1; v <= 1 + p1 * (n - 1); v++) { const yy = Y(BETA / Math.sqrt(v)); v > 1 ? ctx.lineTo(X(v), yy) : ctx.moveTo(X(v), yy); } ctx.stroke(); }
     if (p2 > 0) { ctx.strokeStyle = ROSEI; ctx.beginPath(); ctx.moveTo(X(1), Y(TVB)); ctx.lineTo(X(1 + p2 * (n - 1)), Y(TVB)); ctx.stroke(); } ctx.restore();
     if (p1 >= 1) txt('count bonus: fades', X(55), Y(BETA / Math.sqrt(55)) + 16, { s: 14, c: TEAL, al: 'left' });
     if (p2 >= 1) txt('surprise at the TV: never fades', X(30), Y(TVB) + 16, { s: 14, c: ROSEI, al: 'left' });
     if (t > 8.5) card(['fading bonus: farming stops paying', { s: 'noise: an inner reward that can be farmed forever', c: ROSEI }], t - 8.5, { gap: 1.1 }); }, shot: [5, 11] },
 { h: 'Your turn', ctl: { label: 'explorer', values: ['surprise', 'counting'], def: 0 }, tap(x, y, st, t) { const k = cellAt(x, y); if (k < 0 || k === START || k === CHEESE) return; st.tv = k; st.t0 = t; },
   p: ['**Tap a cell** to put the TV there. The slider picks the explorer: left is surprise, right is counting; its name shows beside the slider and above the maze. Pip replays 3000 steps, same seed, sped up.', 'Try the TV far away, or in a dead end. When does surprise get hooked, and does counting ever care?'],
   say: [["Your turn. Tap a cell to put the TV there. The slider picks the explorer."], [["Watch Pip replay three thousand steps.", [26, 14, 348, 262]]], ["Try the TV far away, or in a dead end. When does surprise get hooked? Does counting ever care?"]],
   draw(t, st) { const tv = st.tv ?? TV, mode = st.ci === 1 ? 'count' : 'pred', R = yourRun(mode, tv), t0 = Math.max(st.t0 ?? 0, st.changed ?? 0), tt = t - t0;
     const s = playRun(R, tt, .3, TS, 12, { cmax: 300, tv }); txt(mode === 'pred' ? 'showing: the surprise explorer' : 'showing: the counting explorer', 200, 10, { s: 14, c: mode === 'pred' ? ROSEI : TEAL });
     let onTv = 0; for (let k = 1; k <= s.k; k++) if (R.path[k] === tv) onTv++;
     txt(`step ${s.k}  ·  cells seen ${s.seen}  ·  on the TV ${onTv}`, 200, 300, { s: 16 });
     txt(R.first > 0 && R.first <= s.k ? `first cheese at step ${R.first}` : (s.k >= TS ? 'no cheese in 3000 steps' : 'no cheese yet'), 200, 326, { s: 16, c: R.first > 0 && R.first <= s.k ? GOLDI : C.soft });
     caption('tap a cell to move the TV'); }, shot: [4, 13] },
 { h: 'Where you meet this', p: ['Pathak and colleagues’ prediction-error curiosity let agents explore Doom and Super Mario Bros games with little or no game reward. Their Mario agent cleared part of level one on curiosity alone.', 'Bellemare and colleagues stretched count bonuses to game screens in 2016, with big gains on Montezuma’s Revenge, the Atari game where DQN scored 0. In 2018 Burda and colleagues demonstrated the noisy TV trap.'],
   say: [["Pathak's curiosity let agents explore Doom and Super Mario with little or no game reward."], ["Bellemare and colleagues stretched counting to game screens, with big gains on Montezuma's Revenge, where DQN scored zero."], [["And in twenty eighteen, Burda and colleagues demonstrated the noisy TV trap.", [30, 292, 340, 87]]]],
   draw(t) { const items = [['Pathak et al., 2017', 'surprise: Doom, Super Mario Bros'], ['Bellemare et al., 2016', 'counts on game screens: Montezuma’s Revenge'], ['Burda et al., 2018', 'curiosity alone in many games; the noisy TV']];
     items.forEach(([a, b], i) => { const y = 70 + i * 70, al = clamp((t - .6 - i * 2) * 2); if (al <= 0) return; rrect(28, y - 26, 344, 54, 770 + i, { fill: C.card, c: C.line, fa: al });
       txt(a, 44, y - 8, { s: 18, al: 'left', a: al }); txt(b, 44, y + 13, { s: 15, al: 'left', c: C.soft, a: al }); });
     if (t > 7) card(['a toy maze shows the idea;', { s: 'real games need bonuses that work on pictures', c: C.soft }], t - 7, { gap: 1.1 }); }, shot: [4, 10] },
 { h: 'Recap', p: ['When reward is **sparse**, random exploring wanders in circles. An **intrinsic reward** for novelty sends the agent toward fresh ground.', 'Count visits and pay more for rare places, or pay for **prediction error**. Beware the **noisy TV**: pure randomness stays surprising forever. Next: what to do when places never repeat, so there is nothing to count.'],
   say: [["So, when reward is sparse, random exploring wanders in circles. A bonus for novelty sends Pip toward fresh ground."], [["Count visits, or pay for surprise.", [30, 290, 340, 87]], "But beware the noisy TV. Pure randomness stays surprising forever."]],
   draw(t) { const s = replay(RC, RC.first); scene({ cnt: s.c, cmax: 64, pip: CHEESE, t }); dim(.55);
     card(['sparse reward: random exploring wanders', 'intrinsic reward: a bonus for novelty', 'count bonus 0.1 ÷ √visits, or surprise', { s: 'noisy TV: randomness never gets boring', c: ROSEI }], t - .3, { gap: 1, y: 290, lh: 22 }); }, shot: [6] },
];
function tvSoFar(k) { let n = 0; for (let j = 1; j <= k; j++) if (RT.path[j] === TV) n++; return n; }

RLStory.register('curiosity', { title: 'Curiosity & Intrinsic Motivation', track: 'Rewards', onCtx: c => { ctx = c; }, steps: STEPS, sim });
})();
