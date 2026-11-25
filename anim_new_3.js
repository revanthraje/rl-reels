// New-topic simulations, part 3: Decision Transformer, GRPO, Safe RL. Loaded after anim.js (uses its helpers, A, RLAnim).
'use strict';
{
const CAP = RLAnim.CAP, TOUCH = RLAnim.TOUCH;
const ctl = (S, k, d) => (S.ctl && S.ctl[k] != null ? S.ctl[k] : d);

// ───────── dt: a game written as a sentence of tokens ─────────
A.n3_tokens = (c, t) => {
  const T = 15, u = t % T, shown = Math.min(9, 1 + Math.floor(u / .8));
  const toks = [['R̂ 15', C.gold], ['s1', C.blue], ['→', C.pink], ['R̂ 15', C.gold], ['s2', C.blue], ['↑', C.pink], ['R̂ 10', C.gold], ['s3', C.blue], ['?', C.pink]];
  const x0 = 13, w = 38, gap = 4, y = 150, h = 38, reading = u > 7.6 && u < 10.5, done = u >= 10.5;
  rr(c, 90, 42, 220, 50, 14, C.panel, reading ? C.purple : C.line, reading ? 3 : 1.5);
  txt(c, 'Transformer', 200, 60, { s: 15, b: 1, c: reading ? C.purple : C.fg });
  txt(c, reading ? 'reading the whole sequence…' : 'predicts the next token', 200, 79, { s: 11, c: C.dim });
  for (let i = 0; i < shown; i++) {
    const x = x0 + i * (w + gap), [s, col] = toks[i], last = i === 8;
    rr(c, x, y, w, h, 8, alpha(col, last && done ? .55 : .22), col, last && done ? 3 : 1.5);
    txt(c, last && done ? '↑' : s, x + w / 2, y + h / 2, { s: s.length > 3 ? 11 : 14, b: 1 });
    if (reading) line(c, x + w / 2, y - 2, lerp(x + w / 2, 200, .7), 96, alpha(C.purple, .5), 1.2);
    if (i % 3 === 0 && i < 9) txt(c, 'step ' + (i / 3 + 1), x + (3 * w + 2 * gap) / 2, y + h + 14, { s: 10, c: C.dim });
  }
  if (done) { arrow(c, 200, 96, x0 + 8 * (w + gap) + w / 2, y - 4, C.pink, 2.5); popup(c, 'predicted: jump ↑', 300, 128, Math.min(1, (u - 10.5) / 3) * .6, C.pink); }
  txt(c, 'gold = return-to-go   blue = state   pink = action', 200, 228, { s: 11, c: C.dim });
  const lab = u < 7.6 ? 'Each step adds three tokens: score still wanted, what it sees, what it did' : reading ? 'The model reads everything so far…' : '…and fills in the next action, like the next word';
  txt(c, lab, 200, 18, { s: 12, b: 1, c: C.gold });
};
CAP.n3_tokens = 'Tokens appear one by one: gold boxes are the score still wanted, blue boxes what the agent sees, pink boxes its actions. The purple transformer reads them all and fills in the missing action.';

// ───────── dt: ask for a score ─────────
const PATHS = [
  { R: 2, col: C.red, cells: [[0, 4], [0, 3], [0, 2], [0, 1], [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], coins: ['0,2', '3,0'] },
  { R: 5, col: C.blue, cells: [[0, 4], [1, 4], [1, 3], [2, 3], [2, 2], [3, 2], [3, 1], [4, 1], [4, 0], [5, 0]], coins: ['1,4', '2,3', '2,2', '3,1', '4,0'] },
  { R: 8, col: C.green, cells: [[0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [5, 3], [4, 3], [4, 2], [5, 2], [5, 1], [5, 0]], coins: ['1,4', '2,4', '3,4', '4,4', '5,4', '4,3', '4,2', '5,1'] },
];
A.n3_askScore = (c, t, S) => {
  const tg = Math.round(ctl(S, 'target', 8));
  let k = 0; PATHS.forEach((p, i) => { if (Math.abs(p.R - tg) < Math.abs(PATHS[k].R - tg)) k = i; });
  if (S.tg !== tg) { S.tg = tg; S.t0 = t; }
  const P = PATHS[k], gx = 208, gy = 52, cs = 30, step = .7, n = P.cells.length;
  const el = (t - S.t0) % (n * step + 2.5), idx = Math.min(n - 1, Math.floor(el / step));
  txt(c, 'Logged games (the dataset)', 100, 30, { s: 12, b: 1, c: C.dim });
  PATHS.forEach((p, i) => {
    rr(c, 14, 44 + i * 40, 172, 32, 10, i === k ? alpha(p.col, .25) : C.panel, i === k ? p.col : C.line, i === k ? 2.5 : 1);
    txt(c, `run ${'ABC'[i]}: final score ${p.R}`, 100, 60 + i * 40, { s: 12, b: i === k });
  });
  txt(c, 'You ask for: ' + tg, 100, 180, { s: 17, b: 1, c: C.gold });
  const warn = tg > 9 ? 'Above every logged run: just guessing' : tg < 2 ? 'Below every run: plays badly on purpose' : 'Copies the run with about that score';
  txt(c, warn, 100, 204, { s: 10.5, c: tg > 9 ? C.red : C.dim });
  for (let y = 0; y < 5; y++) for (let x = 0; x < 6; x++) rr(c, gx + x * cs, gy + y * cs, cs - 2, cs - 2, 5, C.panel);
  PATHS.forEach((p, i) => { if (i !== k) poly(c, p.cells.map(([x, y]) => [gx + x * cs + 14, gy + y * cs + 14]), alpha(p.col, .25), 2, [3, 3]); });
  poly(c, P.cells.map(([x, y]) => [gx + x * cs + 14, gy + y * cs + 14]), alpha(P.col, .7), 2.5);
  let got = 0;
  P.cells.forEach(([x, y], i) => { if (P.coins.includes(x + ',' + y)) { if (i <= idx) got++; else txt(c, '🪙', gx + x * cs + 14, gy + y * cs + 14, { s: 14 }); } });
  txt(c, '🏁', gx + 5 * cs + 14, gy + 14, { s: 14 }); txt(c, 'start', gx + 14, gy + 4 * cs + 38, { s: 10, c: C.dim });
  const [ax, ay] = P.cells[idx]; dot(c, gx + ax * cs + 14, gy + ay * cs + 14, 9, C.pink, '#fff');
  txt(c, `return-to-go: ${tg - got}`, 298, 34, { s: 13, b: 1, c: C.gold });
  txt(c, `coins so far: ${got}`, 298, 222, { s: 12, c: C.fg });
};
CAP.n3_askScore = 'Left: three logged games and their scores. Right: the agent, in pink, follows the run that matches the score you ask for. Each coin it grabs lowers the return-to-go shown on top.';

// ───────── dt: copying whole runs vs stitching pieces ─────────
const RUN1 = [[0, 3], [1, 3], [1, 2], [2, 2], [2, 1], [1, 1], [0, 1], [0, 0]], RUN2 = [[4, 3], [3, 3], [3, 2], [2, 2], [2, 1], [3, 1], [3, 0], [4, 0]];
A.n3_stitch = (c, t) => {
  const T = 13, u = t % T, cs = 30;
  const paths = { dt: RUN1, q: [...RUN1.slice(0, 5), ...RUN2.slice(5)] };
  [['Copy whole runs (DT)', 14, paths.dt, C.purple, 'score 0: wandered off'], ['Stitch pieces (Q-learning)', 210, paths.q, C.teal, 'score +10: reached 🏁']].forEach(([lab, x0, path, col, end]) => {
    const y0 = 52; txt(c, lab, x0 + 88, 30, { s: 11, b: 1, c: col });
    for (let y = 0; y < 4; y++) for (let x = 0; x < 5; x++) rr(c, x0 + 12 + x * cs, y0 + y * cs, cs - 2, cs - 2, 5, C.panel);
    const P = ([x, y]) => [x0 + 12 + x * cs + 14, y0 + y * cs + 14];
    poly(c, RUN1.map(P), alpha(C.blue, .45), 2, [4, 3]); poly(c, RUN2.map(P), alpha(C.gold, .45), 2, [4, 3]);
    txt(c, '🏁', ...P([4, 0]), { s: 14 }); txt(c, 'S', ...P([0, 3]), { s: 12, b: 1, c: C.dim });
    const i = Math.min(path.length - 1, Math.floor(Math.max(0, u - 1) / .75)); dot(c, ...P(path[i]), 8, col, '#fff');
    if (i === path.length - 1) txt(c, end, x0 + 88, 190, { s: 12, b: 1, c: col === C.teal ? C.green : C.red });
  });
  txt(c, 'blue dashes = logged run 1 (from S)   gold dashes = logged run 2 (reaches 🏁)', 200, 222, { s: 10.5, c: C.dim });
  txt(c, 'Both runs cross the same middle square', 200, 242, { s: 11, c: C.fg });
};
CAP.n3_stitch = 'Two logged runs cross in the middle. The left agent copies run 1 all the way and misses the flag. The right one switches to run 2 at the crossing and reaches the flag: that is stitching.';

// ───────── grpo: a group of answers, graded and compared ─────────
const ANS_BAD = ['41', '54', '48', '61', '57', '49'];
A.n3_group = (c, t, S) => {
  if (!S.g) S.g = { seed: 3, t0: t };
  while (S.taps.length) { S.taps.shift(); S.g.seed++; S.g.t0 = t; }
  if (t - S.g.t0 > 15) { S.g.seed++; S.g.t0 = t; }
  const u = t - S.g.t0, r = mulberry(Math.imul(S.g.seed + 11, 2654435761)), n = 8; r(); r();
  const ok = Array.from({ length: n }, () => r() < .45), ans = ok.map(o => o ? '51' : ANS_BAD[Math.floor(r() * ANS_BAD.length)]);
  const mean = ok.filter(Boolean).length / n, shown = Math.min(n, Math.floor(u / .45) + 1), graded = u > 4.2, adv = u > 6.5;
  txt(c, 'Question: 17 × 3 = ?   (8 tries)', 200, 20, { s: 13, b: 1, c: C.gold });
  const x0 = 12, w = 44, gap = 4.6, y = 38, yb = 186;
  for (let i = 0; i < shown; i++) {
    const x = x0 + i * (w + gap);
    rr(c, x, y, w, 44, 8, graded ? alpha(ok[i] ? C.green : C.red, .22) : C.panel, graded ? (ok[i] ? C.green : C.red) : C.line, 1.5);
    txt(c, ans[i], x + w / 2, y + 16, { s: 14, b: 1 });
    if (graded) txt(c, ok[i] ? '✓ 1' : '✗ 0', x + w / 2, y + 34, { s: 11, b: 1, c: ok[i] ? C.green : C.red });
    if (adv) { const a = (ok[i] ? 1 : 0) - mean, g = Math.min(1, (u - 6.5) / 1.2); bar(c, x + 10, yb, w - 20, a * 70 * g, a >= 0 ? C.green : C.red); if (u > 7.7 && a !== 0) txt(c, (a > 0 ? '+' : '') + a.toFixed(2), x + w / 2, yb - a * 70 + (a > 0 ? -9 : 9), { s: 10, b: 1, c: a > 0 ? C.green : C.red }); }
  }
  if (graded) { line(c, 8, yb, 392, yb, C.dim, 1, [4, 3]); txt(c, `group average ${mean.toFixed(2)}`, 392, yb + 12, { s: 10, c: C.dim, a: 'right' }); }
  const allSame = mean === 0 || mean === 1;
  const lab = !graded ? 'Step 1 · sample a group of answers' : !adv ? 'Step 2 · a checker grades: right = 1, wrong = 0' : allSame ? 'All the same: every advantage is 0, nothing to learn' : 'Step 3 · advantage = score − average: green up, red down';
  txt(c, lab, 200, 246, { s: 12, b: 1, c: allSame && adv ? C.red : C.fg });
};
TOUCH.add('n3_group');
CAP.n3_group = 'Eight answers to one maths question appear, then a checker marks them green for right and red for wrong. Bars show each answer’s score minus the group average: green ones get reinforced, red ones discouraged.';

// ───────── grpo: when does a group give any signal? ─────────
A.n3_zeroSignal = (c, t, S) => {
  const p = ctl(S, 'p', .5), n = Math.round(ctl(S, 'n', 8)), useful = 1 - Math.pow(p, n) - Math.pow(1 - p, n);
  const ep = Math.floor(t / 2.5), rows = 9, d = Math.min(9, 150 / n);
  txt(c, 'sample groups', 95, 18, { s: 11, b: 1, c: C.dim });
  let good = 0;
  for (let k = 0; k < rows; k++) {
    const r = mulberry(ep * 131 + k * 17 + 5), o = Array.from({ length: n }, () => r() < p), s = o.filter(Boolean).length, mixed = s > 0 && s < n; good += mixed;
    const y = 36 + k * 21;
    o.forEach((v, i) => dot(c, 14 + i * (d + 1.5) + d / 2, y, d / 2, mixed ? (v ? C.green : C.red) : alpha(C.dim, .5)));
    if (!mixed) txt(c, 'no signal', 182, y, { s: 9, c: C.dim, a: 'right' });
  }
  const px0 = 222, px1 = 386, py = 190, ph = 140, X = q => px0 + (px1 - px0) * q, Y = v => py - v * ph;
  line(c, px0, py, px1, py, C.line, 1.5); line(c, px0, py, px0, py - ph - 4, C.line, 1.5);
  const pts = []; for (let i = 0; i <= 60; i++) { const q = i / 60; pts.push([X(q), Y(1 - Math.pow(q, n) - Math.pow(1 - q, n))]); }
  poly(c, pts, C.teal, 2.5); dot(c, X(p), Y(useful), 6, C.gold, '#fff');
  txt(c, 'chance one try is right →', px1, py + 13, { s: 10, c: C.dim, a: 'right' }); txt(c, 'useful groups', px0 + 4, py - ph - 8, { s: 10, c: C.dim, a: 'left' });
  txt(c, `${Math.round(useful * 100)}% of groups teach something`, 200, 236, { s: 13, b: 1, c: C.gold });
};
CAP.n3_zeroSignal = 'Each row of dots is one group of tries: green right, red wrong. Rows that are all one colour turn grey because they teach nothing. The teal curve shows how often a group is useful.';

// ───────── grpo: answers grow longer as accuracy rises (illustrative) ─────────
A.n3_lengthGrow = (c, t) => {
  const T = 15, u = t % T, f = Math.min(1, u / 10), x0 = 40, x1 = 370, yb = 196, h = 140, X = q => x0 + (x1 - x0) * q;
  const acc = q => .15 + .6 * (1 - Math.exp(-3 * q)) + .02 * Math.sin(q * 30), len = q => .1 + .75 * q * q + .03 * Math.sin(q * 23);
  line(c, x0, yb, x1, yb, C.line, 1.5); line(c, x0, yb, x0, yb - h - 4, C.line, 1.5);
  const N = Math.max(2, Math.floor(f * 80)); const A1 = [], L1 = [];
  for (let i = 0; i < N; i++) { const q = i / 79; A1.push([X(q), yb - acc(q) * h]); L1.push([X(q), yb - len(q) * h]); }
  poly(c, A1, C.teal, 2.5); poly(c, L1, C.gold, 2.5);
  txt(c, 'accuracy', x0 + 6, 32, { s: 11, b: 1, c: C.teal, a: 'left' }); txt(c, 'answer length', x0 + 80, 32, { s: 11, b: 1, c: C.gold, a: 'left' });
  txt(c, 'training steps →', x1, yb + 9, { s: 10, c: C.dim, a: 'right' });
  const early = 'Answer: 41', late = '…17 × 3 is 51. Let me check: 50 + 1. Answer: 51';
  const q = f, bub = q < .45 ? early : late, bx = q < .45 ? 150 : 225;
  rr(c, bx - 115, 216, 230, 24, 8, alpha(q < .45 ? C.red : C.green, .2), q < .45 ? C.red : C.green, 1.2);
  txt(c, bub, bx, 228, { s: 10.5 });
  txt(c, 'illustrative shape, not real data', x1, 48, { s: 9.5, c: C.dim, a: 'right' });
};
CAP.n3_lengthGrow = 'Two curves over training: teal is the share of correct answers, gold is how long answers are. The bubble shows a typical answer: short and wrong early, longer with a self-check later. Illustrative only.';

// ───────── saferl: one fixed penalty weight ─────────
A.n3_tradeoff = (c, t, S) => {
  const w = ctl(S, 'w', 2), sp = clamp(w > 0 ? Math.sqrt(1 / (1.8 * w)) : 1, .08, 1), bump = .6 * sp * sp * sp;
  if (!S.tr || S.tr.w !== w) S.tr = { w, x: 0, trips: 0, bumps: 0, hit: -9, r: mulberry(9), lt: t };
  const T = S.tr, dt = clamp(t - T.lt, 0, .05); T.lt = t;
  T.x += dt * sp * 170; if (T.x > 360) { T.x -= 360; T.trips++; if (T.r() < bump) { T.bumps++; T.hit = t; } }
  rr(c, 14, 70, 372, 70, 12, C.panel, C.line);
  for (let k = 0; k < 12; k++) line(c, 30 + k * 30, 105, 44 + k * 30, 105, alpha(C.dim, .5), 2);
  [120, 250, 330].forEach(x => txt(c, '🚧', x, 128, { s: 13 }));
  const hitNow = t - T.hit < .8; txt(c, hitNow ? '💥' : '🚗', 24 + T.x, 98, { s: 22 });
  txt(c, `speed ${Math.round(sp * 100)}%`, 70, 166, { s: 13, b: 1, c: C.teal });
  txt(c, `bumps per 100 trips: ${(bump * 100).toFixed(1)}`, 230, 166, { s: 13, b: 1, c: bump > .05 ? C.red : C.green });
  txt(c, `this run: ${T.trips} trips, ${T.bumps} bumps`, 200, 192, { s: 11, c: C.dim });
  const verdict = w < 1 ? 'Penalty too small: fast but reckless' : w > 6 ? 'Penalty too big: safe but crawling' : 'Somewhere in between: is it the right balance?';
  txt(c, `reward = speed − ${w.toFixed(1)} × bumps`, 200, 30, { s: 13, b: 1, c: C.gold });
  txt(c, verdict, 200, 228, { s: 12, b: 1 });
};
CAP.n3_tradeoff = 'A delivery car does laps past road works. Its speed is whatever earns the most reward once each bump costs the penalty weight. Watch the speed and the bump rate change as the weight changes.';

// ───────── saferl: a penalty that tunes itself (Lagrangian) ─────────
function lagSim(budget) {
  return memo('lag' + budget, () => {
    const out = []; let lam = 0, cost = 10;
    for (let k = 0; k < 60; k++) { cost = .8 * cost + .2 * 10 * Math.exp(-.9 * lam); lam = Math.max(0, lam + .08 * (cost - budget)); out.push({ cost, lam }); }
    return out;
  });
}
A.n3_lagrange = (c, t, S) => {
  const b = Math.round(ctl(S, 'budget', 2) * 2) / 2, D = lagSim(b);
  if (S.lb !== b) { S.lb = b; S.t0 = t; }
  const n = Math.min(60, 1 + Math.floor((t - S.t0) / 10 * 60)), x0 = 46, x1 = 380, X = k => x0 + (x1 - x0) * k / 59;
  const top = 34, th = 96, bot = 156, bh = 66, cMax = 11, lMax = Math.max(3, ...D.map(d => d.lam)) * 1.1;
  line(c, x0, top + th, x1, top + th, C.line, 1.2); line(c, x0, bot + bh, x1, bot + bh, C.line, 1.2);
  const Yc = v => top + th - v / cMax * th, Yl = v => bot + bh - v / lMax * bh;
  line(c, x0, Yc(b), x1, Yc(b), C.gold, 1.5, [5, 4]); txt(c, 'budget ' + b, x1, Yc(b) - 8, { s: 10, c: C.gold, a: 'right' });
  poly(c, D.slice(0, n).map((d, k) => [X(k), Yc(d.cost)]), C.red, 2.5);
  poly(c, D.slice(0, n).map((d, k) => [X(k), Yl(d.lam)]), C.purple, 2.5);
  const d = D[n - 1]; dot(c, X(n - 1), Yc(d.cost), 4, C.red); dot(c, X(n - 1), Yl(d.lam), 4, C.purple);
  txt(c, 'cost (bumps)', 8, top + 6, { s: 11, b: 1, c: C.red, a: 'left' }); txt(c, 'penalty weight λ', 8, bot - 4, { s: 11, b: 1, c: C.purple, a: 'left' });
  txt(c, 'training rounds →', x1, bot + bh + 12, { s: 10, c: C.dim, a: 'right' });
  txt(c, d.cost > b + .05 ? 'cost over budget → λ goes up' : 'cost under budget → λ eases down', 200, 248, { s: 12, b: 1, c: d.cost > b + .05 ? C.red : C.green });
};
CAP.n3_lagrange = 'Top: the red cost curve against the gold budget line. Bottom: the purple penalty weight. When cost is over budget the weight climbs, which pushes cost down; then the weight relaxes and both settle.';

// ───────── saferl: a shield blocks dangerous moves ─────────
const LAVA = new Set(['2,0', '2,1', '4,3', '4,4', '5,1', '6,3', '3,4']);
A.n3_shield = (c, t, S) => {
  if (!S.sh) S.sh = { on: true, x: 0, y: 2, last: t, r: mulberry(4), falls: 0, blocked: 0, goals: 0, fx: null, tapT: -99, modeT: t, boom: -9 };
  const H = S.sh;
  const reset = on => { H.on = on; H.x = 0; H.y = 2; H.falls = H.blocked = H.goals = 0; H.modeT = t; H.fx = null; };
  while (S.taps.length) { S.taps.shift(); reset(!H.on); H.tapT = t; }
  if (t - H.tapT > 20 && t - H.modeT > 13) reset(!H.on);
  while (t - H.last >= .55) {
    H.last += .55;
    const D = [[1, 0], [0, -1], [0, 1], [-1, 0]];
    let a = H.r() < .55 ? (H.r() < .7 ? 0 : (H.y < 2 ? 2 : H.y > 2 ? 1 : (H.r() < .5 ? 1 : 2))) : Math.floor(H.r() * 4);
    const nx = q => clamp(H.x + D[q][0], 0, 7), ny = q => clamp(H.y + D[q][1], 0, 4), bad = q => LAVA.has(nx(q) + ',' + ny(q));
    if (H.on && bad(a)) { H.blocked++; H.fx = { x: nx(a), y: ny(a), t: H.last }; const safe = [0, 1, 2, 3].filter(q => !bad(q)); a = safe[Math.floor(H.r() * safe.length)]; }
    H.x = nx(a); H.y = ny(a);
    if (LAVA.has(H.x + ',' + H.y)) { H.falls++; H.boom = H.last; H.bx = H.x; H.by = H.y; H.x = 0; H.y = 2; }
    if (H.x === 7 && H.y === 2) { H.goals++; H.x = 0; H.y = 2; }
  }
  const gx = 14, gy = 40, cs = 34;
  for (let y = 0; y < 5; y++) for (let x = 0; x < 8; x++) { const l = LAVA.has(x + ',' + y); rr(c, gx + x * cs, gy + y * cs, cs - 2, cs - 2, 5, l ? alpha(C.red, .55) : C.panel); if (l) txt(c, '🔥', gx + x * cs + 16, gy + y * cs + 16, { s: 13 }); }
  txt(c, '🏁', gx + 7 * cs + 16, gy + 2 * cs + 16, { s: 15 }); txt(c, 'S', gx + 16, gy + 2 * cs + 16, { s: 12, b: 1, c: C.dim });
  if (H.fx && t - H.fx.t < .7) { const fx = gx + H.fx.x * cs + 16, fy = gy + H.fx.y * cs + 16; txt(c, '✋', fx, fy, { s: 18 }); }
  if (t - H.boom < .8) txt(c, '💥', gx + H.bx * cs + 16, gy + H.by * cs + 16, { s: 20 });
  dot(c, gx + H.x * cs + 16, gy + H.y * cs + 16, 9, C.teal, '#fff');
  rr(c, 292, 40, 98, 64, 12, H.on ? alpha(C.green, .25) : alpha(C.red, .2), H.on ? C.green : C.red, 2);
  txt(c, H.on ? '🛡 shield ON' : 'shield OFF', 341, 62, { s: 13, b: 1, c: H.on ? C.green : C.red }); txt(c, 'tap to switch', 341, 84, { s: 10, c: C.dim });
  txt(c, `falls: ${H.falls}`, 341, 124, { s: 13, b: 1, c: H.falls ? C.red : C.fg });
  txt(c, `blocked: ${H.blocked}`, 341, 146, { s: 13, b: 1, c: C.gold });
  txt(c, `flags: ${H.goals}`, 341, 168, { s: 13, b: 1, c: C.green });
  txt(c, H.on ? '✋ = a move into fire was swapped for a safe one' : 'Without the shield, exploring means falling in', 200, 228, { s: 12, b: 1 });
  txt(c, 'teal dot = the exploring agent', 200, 18, { s: 11, c: C.dim });
};
TOUCH.add('n3_shield');
CAP.n3_shield = 'A teal agent explores a grid with fire squares. With the shield on, any move into fire is caught, shown as a hand, and swapped for a safe move. With it off, the agent keeps falling in.';
}
