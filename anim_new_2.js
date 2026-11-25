// New-topic simulations, part 2: MuZero, Successor Features, Contextual Bandits.
// Loaded after anim.js (uses its helpers, A and RLAnim). Interactive ones read S.ctl / S.taps.
'use strict';
(() => {
const CAP = RLAnim.CAP, TOUCH = RLAnim.TOUCH;
const inBox = (q, x, y, w, h) => q.x >= x && q.x <= x + w && q.y >= y && q.y <= y + h;
const f2 = v => (Math.round(v * 100) / 100).toFixed(2);
const f1 = v => (Math.round(v * 10) / 10).toFixed(1);
const step = (c, s, col) => { rr(c, 12, 222, 376, 30, 10, alpha(C.blue, .12), C.line, 1); txt(c, s, 200, 237, { s: 12.5, b: 1, c: col || C.fg }); };

// ═════════════ MuZero ═════════════
function chip(c, x, y, lab, on, col = C.purple) {
  dot(c, x, y, 17, on ? alpha(col, .9) : alpha(col, .25), on ? C.fg : null);
  txt(c, lab, x, y, { s: 12, b: 1, c: on ? C.bg : C.dim });
}
function policyBars(c, x, y, ps, al = 1) {
  ps.forEach((p, k) => { rr(c, x - 19 + k * 13, y - p * 26, 10, Math.max(2, p * 26), 2, alpha([C.teal, C.gold, C.pink][k], al)); });
}

// 1. the three learned functions, step by step
A.n2_mzPipeline = (c, t) => {
  const P = 2.6, ph = Math.floor(t / P) % 6;
  txt(c, 'MuZero’s learned model', 200, 16, { s: 12, c: C.gold });
  // observation
  rr(c, 12, 62, 58, 58, 8, C.panel, ph === 0 ? C.teal : C.line, ph === 0 ? 3 : 1.5);
  const r = mulberry(3); for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) c.fillStyle = alpha([C.blue, C.teal, C.pink, C.gold][Math.floor(r() * 4)], .7), c.fillRect(17 + i * 8, 67 + j * 8, 7, 7);
  txt(c, 'screen', 41, 132, { s: 11, c: C.dim });
  arrow(c, 72, 91, 104, 91, ph === 0 ? C.teal : C.line, 2);
  rr(c, 106, 72, 34, 38, 8, ph === 0 ? alpha(C.teal, .25) : C.panel, C.line, 1); txt(c, 'h', 123, 91, { s: 15, b: 1, c: C.teal });
  // hidden states
  const xs = [175, 265, 355], acts = ['➡', '⬅'], rews = ['+1', '0'];
  chip(c, xs[0], 91, 's0', ph >= 0);
  arrow(c, 141, 91, 156, 91, ph === 0 ? C.teal : C.line, 2);
  for (let k = 0; k < 2; k++) {
    const on = ph >= k + 1, cur = ph === k + 1;
    const x1 = xs[k] + 20, x2 = xs[k + 1] - 20;
    arrow(c, x1, 91, x2, 91, cur ? C.pink : on ? C.line : alpha(C.dim, .25), cur ? 3 : 2);
    rr(c, (x1 + x2) / 2 - 17, 56, 34, 22, 6, cur ? alpha(C.pink, .3) : C.panel, C.line, 1);
    txt(c, 'g ' + acts[k], (x1 + x2) / 2, 67, { s: 12, b: 1, c: on ? C.pink : C.dim });
    if (on) txt(c, 'reward ' + rews[k], (x1 + x2) / 2, 112, { s: 11, b: 1, c: C.gold });
    chip(c, xs[k + 1], 91, 's' + (k + 1), on);
  }
  // predictions
  if (ph >= 3) {
    const pol = [[.2, .7, .1], [.6, .3, .1], [.3, .3, .4]], val = ['0.8', '0.9', '0.4'];
    xs.forEach((x, k) => {
      rr(c, x - 30, 140, 60, 66, 8, ph === 3 ? alpha(C.green, .12) : C.panel, ph === 3 ? C.green : C.line, ph === 3 ? 2 : 1);
      txt(c, 'f', x - 22, 150, { s: 12, b: 1, c: C.green });
      policyBars(c, x, 190, pol[k]); txt(c, 'v ' + val[k], x + 16, 150, { s: 11, b: 1, c: C.green });
      line(c, x, 109, x, 139, alpha(C.green, .6), 1.5, [3, 3]);
    });
  }
  step(c, ['1 · h turns the real screen into a hidden state s0', '2 · g imagines: after ➡ comes state s1 and reward +1',
    '3 · g again: after ⬅ comes s2 and reward 0', '4 · f guesses good moves (bars) and value v for each state',
    '5 · all of this happens inside the network', '6 · no screen is ever redrawn, no rules are used'][ph], ph >= 4 ? C.gold : C.fg);
};
CAP.n2_mzPipeline = 'Left, the real screen goes through h into a purple hidden state. Pink g boxes imagine the next state and its reward for a move. Green f boxes guess good moves and a value for each state.';

// 2. redraw pixels vs predict three numbers
A.n2_pixelsVsValue = (c, t) => {
  const T = 14, tt = t % T, u = clamp(tt / 9, 0, 1);
  rr(c, 10, 30, 186, 186, 12, C.panel, C.line, 1.5); rr(c, 204, 30, 186, 186, 12, C.panel, C.purple, 2);
  txt(c, 'World model: redraw the screen', 103, 18, { s: 11.5, b: 1, c: C.dim });
  txt(c, 'MuZero: just 3 things', 297, 18, { s: 11.5, b: 1, c: C.purple });
  const r = mulberry(11), n = 12, filled = Math.floor(u * n * n);
  for (let k = 0; k < n * n; k++) {
    const i = k % n, j = Math.floor(k / n), base = (i + j) % 5 === 0 ? C.gold : j > 8 ? C.green : C.blue, noise = r();
    c.fillStyle = k < filled ? mix(base, C.panel, .25 + noise * .5) : alpha(C.dim, .08);
    c.fillRect(22 + i * 13.5, 40 + j * 13.5, 12, 12);
  }
  txt(c, 'numbers to predict: ' + Math.round(u * 27648).toLocaleString('en'), 103, 207, { s: 11, b: 1, c: C.fg });
  const items = [['reward', '+1', C.gold], ['value', '0.8', C.green], ['good moves', '', C.teal]];
  items.forEach(([lab, v, col], k) => {
    const on = tt > 1 + k * 1.4; if (!on) return;
    const y = 66 + k * 46;
    rr(c, 216, y - 18, 162, 36, 9, alpha(col, .15), col, 1.5);
    txt(c, lab, 228, y, { s: 13, b: 1, a: 'left' });
    if (v) txt(c, v, 362, y, { s: 16, b: 1, c: col, a: 'right' }); else policyBars(c, 352, y + 12, [.25, .65, .1]);
  });
  if (tt > 5.5) txt(c, 'numbers to predict: about 5 ✓', 297, 207, { s: 11, b: 1, c: C.green });
  txt(c, 'Only what matters for choosing a move has to be right.', 200, 240, { s: 12, b: 1, c: tt > 7 ? C.gold : C.dim });
};
CAP.n2_pixelsVsValue = 'Left, a model that redraws every pixel slowly fills thousands of squares. Right, MuZero only predicts a reward, a value and which moves look good: a handful of numbers.';

// 3. imagine moves yourself (tap ⬅ / ➡)
A.n2_mzImagine = (c, t, S) => {
  S.taps ||= [];
  const M = S.mi ||= { seq: [], last: t, done: 0, auto: t + 4 };
  const GOAL = 5, START = 2;
  const pos = () => M.seq.reduce((p, a) => clamp(p + a, 0, 5), START);
  const push = a => { if (M.seq.length < 4 && !M.done) { M.seq.push(a); M.last = t; if (M.seq.length === 4 || pos() === GOAL) M.done = t; } };
  while (S.taps.length) { const q = S.taps.shift(); M.auto = t + 7; if (M.done) continue; if (inBox(q, 40, 196, 140, 52)) push(-1); else if (inBox(q, 220, 196, 140, 52)) push(1); }
  if (t > M.auto && !M.done && t - M.last > 1.6) push(pos() < GOAL ? 1 : -1);
  if (M.done && t - M.done > 3.2) { M.seq = []; M.done = 0; M.last = t; }
  // real game strip (frozen)
  txt(c, 'real game (paused while MuZero thinks)', 200, 14, { s: 11, c: C.dim });
  for (let i = 0; i < 6; i++) rr(c, 92 + i * 36, 24, 32, 26, 5, i === GOAL ? alpha(C.gold, .25) : C.bg, C.line, 1);
  txt(c, '🪙', 92 + GOAL * 36 + 16, 37, { s: 14 }); txt(c, '🤖', 92 + START * 36 + 16, 37, { s: 15 });
  // imagined chain
  txt(c, 'imagined inside the network', 200, 66, { s: 11, b: 1, c: C.purple });
  let p = START, ret = 0, gone = false;
  const chain = [{ p, r: 0, v: Math.pow(.9, GOAL - p) }];
  M.seq.forEach((a, k) => {
    const np = clamp(p + a, 0, 5); const r = !gone && np === GOAL ? 1 : 0; if (np === GOAL) gone = true; p = np;
    chain.push({ p, r, v: gone ? 0 : Math.pow(.9, Math.abs(GOAL - p)), a });
  });
  chain.forEach((s, k) => {
    const x = 40 + k * 80, y = 112;
    if (k) { arrow(c, x - 60, y, x - 20, y, C.pink, 2); txt(c, s.a > 0 ? '➡' : '⬅', x - 40, y - 14, { s: 13 }); if (s.r) txt(c, '+1', x - 40, y + 15, { s: 12, b: 1, c: C.gold }); }
    chip(c, x, y, 's' + k, true);
    txt(c, 'v ' + f2(s.v), x, y + 32, { s: 11, b: 1, c: C.green });
  });
  for (let k = chain.length; k < 5; k++) { const x = 40 + k * 80; dot(c, x, 112, 17, alpha(C.dim, .08), alpha(C.dim, .3)); }
  chain.forEach((s, k) => { if (k) ret += Math.pow(.9, k - 1) * s.r; });
  const last = chain[chain.length - 1], total = ret + Math.pow(.9, chain.length - 1) * last.v;
  txt(c, M.seq.length ? `imagined score = rewards + value at the end = ${f2(total)}` : 'start: MuZero’s value guess here is ' + f2(chain[0].v), 200, 168, { s: 12, b: 1, c: M.done ? C.gold : C.fg });
  if (M.done) txt(c, last.p === GOAL ? 'reached the coin in imagination! 🪙' : 'plan scored, try another one…', 200, 186, { s: 11.5, c: C.gold });
  rr(c, 40, 198, 140, 46, 12, alpha(C.pink, .18), C.pink, 2); txt(c, '⬅ imagine left', 110, 221, { s: 13, b: 1 });
  rr(c, 220, 198, 140, 46, 12, alpha(C.pink, .18), C.pink, 2); txt(c, 'imagine right ➡', 290, 221, { s: 13, b: 1 });
};
TOUCH.add('n2_mzImagine');
CAP.n2_mzImagine = 'The real game at the top stays frozen. Each move you imagine adds a purple hidden state, with a gold reward if it reaches the coin and a green value guess. The score adds them up.';

// ═════════════ Successor features ═════════════
const SW = 6, SH = 4, SG = .8, SX0 = 22, SY0 = 34, SS = 38;
const sid = (x, y) => y * SW + x;
const MOVES = [[0, -1], [1, 0], [0, 1], [-1, 0]], MP = [.2, .35, .2, .25];
function srMove(x, y, a) { const nx = clamp(x + MOVES[a][0], 0, SW - 1), ny = clamp(y + MOVES[a][1], 0, SH - 1); return [nx, ny]; }
const SR = () => memo('n2_sr', () => { // M[s][s'] = expected discounted visits under the fixed wandering policy
  const N = SW * SH; let M = Array.from({ length: N }, (_, s) => Array.from({ length: N }, (_, j) => +(s === j)));
  for (let it = 0; it < 80; it++) {
    M = M.map((row, s) => { const x = s % SW, y = Math.floor(s / SW); const out = row.map((_, j) => +(s === j));
      MOVES.forEach((_, a) => { const [nx, ny] = srMove(x, y, a), ns = sid(nx, ny); for (let j = 0; j < N; j++) out[j] += SG * MP[a] * M[ns][j]; }); return out; });
  }
  return M;
});
function srGrid(c, val, o = {}) {
  for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
    const v = val[sid(x, y)], px = SX0 + x * SS, py = SY0 + y * SS;
    rr(c, px, py, SS - 3, SS - 3, 6, heat(v), C.line, 1);
    if (o.nums) txt(c, o.fmt ? o.fmt(o.raw[sid(x, y)]) : f1(v), px + (SS - 3) / 2, py + SS - 11, { s: 10, c: v > .55 ? C.bg : C.fg, b: 1 });
  }
}

// 4. where will I end up? discounted visits pile up
A.n2_srOccupancy = (c, t) => {
  const SX = [1, 2], NW = 14, L = 12, STEP = .38;
  const walks = memo('n2_walks', () => { const r = mulberry(21), W = []; for (let w = 0; w < NW; w++) { let [x, y] = SX; const p = [[x, y]]; for (let k = 1; k < L; k++) { let u = r(), a = 0; while (a < 3 && u > MP[a]) { u -= MP[a]; a++; }[x, y] = srMove(x, y, a); p.push([x, y]); } W.push(p); } return W; });
  const per = L * STEP + .9, T = NW * per + 4, tt = t % T, w = Math.min(NW - 1, Math.floor(tt / per)), k = Math.min(L - 1, Math.floor((tt - w * per) / STEP));
  const occ = new Array(SW * SH).fill(0), done = tt >= NW * per ? NW : w;
  for (let i = 0; i < done; i++) walks[i].forEach(([x, y], j) => occ[sid(x, y)] += Math.pow(SG, j));
  if (done < NW) for (let j = 0; j <= k; j++) { const [x, y] = walks[w][j]; occ[sid(x, y)] += Math.pow(SG, j); }
  const n = Math.max(1, done + (done < NW ? 1 : 0)), avg = occ.map(v => v / n), mx = Math.max(...avg);
  srGrid(c, avg.map(v => v / mx * .95), { nums: true, raw: avg, fmt: f1 });
  if (done < NW) {
    const path = walks[w].slice(0, k + 1).map(([x, y]) => [SX0 + x * SS + 17, SY0 + y * SS + 17]);
    poly(c, path, alpha(C.pink, .8), 2.5); const e = path[path.length - 1]; dot(c, e[0], e[1], 7, C.pink, C.fg);
    if (k) txt(c, '+' + f2(Math.pow(SG, k)), e[0], e[1] - 14, { s: 11, b: 1, c: C.gold });
  }
  rr(c, SX0 + SX[0] * SS + 2, SY0 + SX[1] * SS + 2, SS - 7, SS - 7, 5, null, C.fg, 2);
  const lx = 262;
  txt(c, 'walks averaged: ' + Math.min(n, NW), lx, 48, { s: 12, b: 1, a: 'left' });
  txt(c, 'step ' + (done < NW ? k : '-') + ' adds ' + SG + '^step', lx, 70, { s: 11, a: 'left', c: C.gold });
  txt(c, 'each square shows', lx, 104, { s: 11, a: 'left', c: C.dim });
  txt(c, 'future visits, with', lx, 120, { s: 11, a: 'left', c: C.dim });
  txt(c, 'later ones shrunk', lx, 136, { s: 11, a: 'left', c: C.dim });
  txt(c, 'by γ = 0.8 per step', lx, 152, { s: 11, a: 'left', c: C.dim });
  txt(c, 'bright = often & soon', lx, 178, { s: 11, b: 1, a: 'left', c: C.teal });
  txt(c, 'white box = start', lx, 196, { s: 11, a: 'left', c: C.fg });
  step(c, done < NW ? `walk ${w + 1}: wander from the start, count where you go` : 'average of all walks = the successor representation', done < NW ? C.fg : C.gold);
};
CAP.n2_srOccupancy = 'A pink walker wanders from the white start square. Each step adds a shrinking amount, shown in gold, to the square it lands on. Averaged over walks, bright squares are where you’ll be often and soon.';

// 5. values for free: drag the reward weights
A.n2_srSwap = (c, t, S) => {
  const ctl = S.ctl || {}, w1 = ctl.apple ?? 1, w2 = ctl.water ?? 0, M = SR();
  const APPLE = sid(5, 0), WATER = sid(0, 3), START = sid(1, 2), N = SW * SH;
  const V = M.map(row => w1 * row[APPLE] + w2 * row[WATER]), mx = Math.max(.3, ...V.map(Math.abs));
  const key = w1 + ',' + w2; if (S.wk !== key) { S.wk = key; S.wt = t; S.ag = START; S.an = t + .8; S.trail = [START]; }
  srGrid(c, V.map(v => v / mx * .95), { nums: true, raw: V, fmt: f1 });
  txt(c, '🍎', SX0 + 5 * SS + 17, SY0 + 13, { s: 16 }); txt(c, '💧', SX0 + 17, SY0 + 3 * SS + 13, { s: 16 });
  // a walker climbs the new values (nothing relearned)
  if (t > S.an && S.trail.length < 10 && S.ag !== APPLE && S.ag !== WATER) {
    const x = S.ag % SW, y = Math.floor(S.ag / SW); let best = S.ag, bv = V[S.ag];
    MOVES.forEach((_, a) => { const [nx, ny] = srMove(x, y, a), ns = sid(nx, ny); if (V[ns] > bv + 1e-9) { bv = V[ns]; best = ns; } });
    if (best === S.ag) S.trail.length = 10; else { S.ag = best; S.trail.push(best); }
    S.an = t + .7;
  }
  if (t > S.an + 2.5 && (S.trail.length >= 10 || S.ag === APPLE || S.ag === WATER)) { S.ag = START; S.trail = [START]; S.an = t + .8; }
  poly(c, S.trail.map(s => [SX0 + (s % SW) * SS + 17, SY0 + Math.floor(s / SW) * SS + 17]), alpha(C.pink, .8), 2.5);
  const ax = SX0 + (S.ag % SW) * SS + 17, ay = SY0 + Math.floor(S.ag / SW) * SS + 17; dot(c, ax, ay, 7, C.pink, C.fg);
  const lx = 262;
  txt(c, 'value =', lx, 50, { s: 12, b: 1, a: 'left' });
  txt(c, 'where I’ll end up', lx, 68, { s: 11.5, a: 'left', c: C.teal });
  txt(c, '× what I like there', lx, 86, { s: 11.5, a: 'left', c: C.gold });
  txt(c, '🍎 weight ' + f1(w1), lx, 118, { s: 12, b: 1, a: 'left', c: w1 >= 0 ? C.green : C.red });
  txt(c, '💧 weight ' + f1(w2), lx, 138, { s: 12, b: 1, a: 'left', c: w2 >= 0 ? C.green : C.red });
  txt(c, 'new values appear', lx, 168, { s: 11, a: 'left', c: C.dim });
  txt(c, 'instantly: no new', lx, 184, { s: 11, a: 'left', c: C.dim });
  txt(c, 'experience needed', lx, 200, { s: 11, a: 'left', c: C.dim });
  step(c, t - S.wt < 1.2 ? 'weights changed → values recomputed in one multiply ⚡' : 'the pink agent climbs toward higher values', t - S.wt < 1.2 ? C.gold : C.fg);
};
CAP.n2_srSwap = 'Square colours and numbers are values. They come from multiplying the stored future-visit counts by how much you like 🍎 and 💧. The pink agent walks uphill on the new values.';

// 6. goal moves: SR adapts at once, TD relearns slowly
A.n2_srVsRelearn = (c, t) => {
  const N = 8, T = 18, tt = t % T, MOVE = 2.5, g = .8;
  const vOf = goal => Array.from({ length: N }, (_, i) => Math.pow(g, Math.abs(goal - i)));
  const old = vOf(N - 1), neu = vOf(0);
  const tdHist = memo('n2_td', () => { const H = [old.slice()]; let V = old.slice();
    for (let e = 0; e < 30; e++) { V = V.map((v, i) => i === 0 ? v + .5 * (1 - v) : v + .35 * (g * Math.max(V[i - 1], V[Math.min(N - 1, i + 1)]) - v)); H.push(V.slice()); } return H; });
  const moved = tt >= MOVE, ep = moved ? Math.min(tdHist.length - 1, Math.floor((tt - MOVE) / .5)) : 0;
  const srV = !moved ? old : tt < MOVE + .6 ? old.map((v, i) => lerp(v, neu[i], (tt - MOVE) / .6)) : neu;
  const tdV = tdHist[ep];
  const panel = (y, lab, V, col, sub) => {
    txt(c, lab, 14, y - 52, { s: 12, b: 1, a: 'left', c: col }); txt(c, sub, 386, y - 52, { s: 11, a: 'right', c: C.dim });
    for (let i = 0; i < N; i++) { const x = 30 + i * 44; rr(c, x, y + 4, 38, 8, 3, C.panel); bar(c, x + 4, y + 2, 30, V[i] * 42, alpha(col, .85)); }
    const goal = moved ? 0 : N - 1; txt(c, '🍎', 30 + goal * 44 + 19, y + 22, { s: 14 });
  };
  panel(96, 'successor representation', srV, C.teal, moved ? 'adapted at once ⚡' : 'value bars');
  panel(196, 'plain TD, relearning', tdV, C.pink, moved ? `episodes since move: ${ep}` : 'value bars');
  if (moved && tt < MOVE + 2) txt(c, '🍎 moved to the left end!', 200, 16, { s: 13, b: 1, c: C.gold });
  else txt(c, moved ? 'TD has to wait for the change to spread back step by step' : 'apple at the right end: both agents agree', 200, 16, { s: 12, c: C.dim });
};
CAP.n2_srVsRelearn = 'Two rows of value bars over the same corridor. When the apple jumps to the left end, the teal successor agent re-multiplies and adapts at once; the pink TD agent needs many episodes.';

// ═════════════ Contextual bandits ═════════════
const USERS = [['🧑‍🎓', 'student'], ['👵', 'retiree'], ['🧑‍💼', 'worker']], ARTS = [['⚽', 'sport'], ['💻', 'tech'], ['🌻', 'garden']];
const TRUE = [[.2, .7, .05], [.1, .1, .6], [.5, .3, .1]];

// 7. plain vs contextual, one visitor at a time
A.n2_ctxNews = (c, t) => {
  const P = 1.7, n = Math.floor(t / P), u = (t % P) / P;
  const sim = memo('n2_news', () => { const r = mulberry(5), out = []; let a = 0, b = 0;
    for (let i = 0; i < 400; i++) { const who = Math.floor(r() * 3), x = r(), cb = TRUE[who][1] > x, cc = TRUE[who][TRUE[who].indexOf(Math.max(...TRUE[who]))] > x; a += cb; b += cc; out.push({ who, cb, cc, a, b }); } return out; });
  const k = n % 400, h = sim[k], best = TRUE[h.who].indexOf(Math.max(...TRUE[h.who]));
  txt(c, 'visitor ' + (k + 1), 200, 16, { s: 12, c: C.dim });
  txt(c, USERS[h.who][0], 200, 52, { s: 34 }); txt(c, USERS[h.who][1], 200, 80, { s: 12, b: 1 });
  const row = (y, lab, art, click, tot, col) => {
    rr(c, 14, y, 372, 50, 12, C.panel, col, 1.5);
    txt(c, lab, 26, y + 15, { s: 11.5, b: 1, a: 'left', c: col });
    txt(c, 'shows ' + ARTS[art][0] + ' ' + ARTS[art][1], 26, y + 35, { s: 13, b: 1, a: 'left' });
    if (u > .35) txt(c, click ? '👆 click!' : 'no click', 250, y + 25, { s: 13, b: 1, c: click ? C.green : C.dim });
    txt(c, 'clicks ' + tot, 376, y + 25, { s: 12, b: 1, a: 'right', c: col });
  };
  row(100, 'plain bandit: one best article for all', 1, h.cb, h.a, C.pink);
  row(160, 'contextual: looks at who arrives', best, h.cc, h.b, C.teal);
  txt(c, 'long-run click rate:  plain 37%  ·  contextual 60%', 200, 232, { s: 12, b: 1, c: C.gold });
};
CAP.n2_ctxNews = 'Visitors arrive one at a time. The pink plain bandit shows everyone tech, its best article on average. The teal contextual bandit picks per visitor, and its click count pulls ahead.';

// 8. one UCB row per context, with a bonus slider
A.n2_ctxTable = (c, t, S) => {
  const bonus = Math.round(((S.ctl || {}).bonus ?? .6) * 10) / 10;
  if (S.bk !== bonus) { S.bk = bonus; S.t0 = t; }
  const sim = memo('n2_ctx' + bonus, () => { const r = mulberry(17), N = TRUE.map(() => [0, 0, 0]), W = TRUE.map(() => [0, 0, 0]), H = []; let clicks = 0, best = 0;
    for (let i = 0; i < 400; i++) {
      const u = Math.floor(r() * 3), n = N[u].reduce((a, b) => a + b, 0);
      let a = 0, sc = -1; for (let j = 0; j < 3; j++) { const s = N[u][j] ? W[u][j] / N[u][j] + bonus * Math.sqrt(Math.log(n + 1) / N[u][j]) : 1e9 - j; if (s > sc) { sc = s; a = j; } }
      const click = r() < TRUE[u][a]; N[u][a]++; W[u][a] += click; clicks += click; best += TRUE[u][a] === Math.max(...TRUE[u]);
      H.push({ u, a, click, N: N.map(x => x.slice()), W: W.map(x => x.slice()), clicks, best });
    } return H; });
  const tt = t - S.t0, i = Math.min(399, tt < 10 ? Math.floor(tt) : 10 + Math.floor((tt - 10) * 4)), h = sim[i];
  txt(c, tt < 10 ? 'one visitor per second' : '⏩ fast-forward', 386, 14, { s: 11, a: 'right', c: C.dim });
  txt(c, 'bonus ' + f1(bonus), 14, 14, { s: 12, b: 1, a: 'left', c: C.gold });
  ARTS.forEach(([e, n], j) => txt(c, e + ' ' + n, 150 + j * 86, 36, { s: 12, b: 1 }));
  USERS.forEach(([e, n], u) => {
    const y = 52 + u * 50;
    txt(c, e, 30, y + 20, { s: 22 }); txt(c, n, 70, y + 20, { s: 11, c: C.dim, a: 'left' });
    for (let j = 0; j < 3; j++) {
      const x = 110 + j * 86, cur = h.u === u && h.a === j, N = h.N[u][j], est = N ? h.W[u][j] / N : 0;
      rr(c, x, y, 80, 42, 8, cur ? alpha(h.click ? C.green : C.pink, .3) : C.panel, cur ? (h.click ? C.green : C.pink) : C.line, cur ? 2.5 : 1);
      bar(c, x + 6, y + 36, est * 68, 6, alpha(C.teal, .9)); c.fillStyle = C.gold; c.fillRect(x + 6 + TRUE[u][j] * 68, y + 30, 2, 12);
      txt(c, N ? Math.round(est * 100) + '%' : '?', x + 20, y + 15, { s: 12, b: 1 }); txt(c, '×' + N, x + 72, y + 15, { s: 10, c: C.dim, a: 'right' });
    }
  });
  txt(c, `visitors ${i + 1}  ·  clicks ${h.clicks}  ·  best article shown ${Math.round(100 * h.best / (i + 1))}%`, 200, 216, { s: 12, b: 1 });
  txt(c, 'teal bar = guess · gold tick = true rate · ×N = times shown', 200, 240, { s: 10.5, c: C.dim });
};
CAP.n2_ctxTable = 'One row per kind of visitor, one column per article. Teal bars are guessed click rates, gold ticks the true ones. Each row runs its own UCB: highlighted cells are what was just shown.';

// 9. LinUCB: score = features × weights + bonus
A.n2_ctxLine = (c, t) => {
  const P = 1.5, NP = 7, n = Math.floor(t / P), u = n % (NP * 3), who = Math.floor(u / NP), ph = u % NP;
  const F = [[.3, .1, .9], [.2, .9, .1], [.8, .4, .4]], Wt = [[.8, 0, .1], [0, -.2, .8], [0, .7, 0]], B = [.06, .1, .22];
  const FN = ['likes sport', 'older', 'works in tech'];
  txt(c, USERS[who][0] + ' ' + USERS[who][1], 70, 22, { s: 14, b: 1 });
  FN.forEach((f, k) => { const y = 52 + k * 34; txt(c, f, 14, y, { s: 11, a: 'left', c: C.dim }); rr(c, 14, y + 8, 110, 8, 3, C.panel); bar(c, 14, y + 16, F[who][k] * 110, 8, alpha(C.blue, .9)); });
  txt(c, 'user features', 70, 160, { s: 11, b: 1, c: C.blue });
  ARTS.forEach(([e, nme], j) => {
    const y = 40 + j * 56, on = ph >= j + 1, cur = ph === j + 1;
    const score = F[who].reduce((s, f, k) => s + f * Wt[j][k], 0);
    txt(c, e, 160, y + 14, { s: 20 });
    if (cur) txt(c, F[who].map((f, k) => f1(f) + '×' + f1(Wt[j][k])).join(' + '), 186, y - 2, { s: 10, c: C.dim, a: 'left' });
    rr(c, 184, y + 8, 120, 12, 4, C.panel);
    if (on) bar(c, 184, y + 20, Math.max(0, score) * 120, 12, alpha(C.teal, .9));
    if (ph >= 4) { const x0 = 184 + Math.max(0, score) * 120; rr(c, x0, y + 8, B[j] * 120, 12, 3, alpha(C.gold, .6)); }
    if (on) txt(c, f2(score) + (ph >= 4 ? ' + ' + f2(B[j]) : ''), 386, y + 14, { s: 11, b: 1, a: 'right' });
    if (ph >= 5) { const tot = F[who].map((_, jj) => F[who].reduce((s, f, k) => s + f * Wt[jj][k], 0) + B[jj]); if (tot.indexOf(Math.max(...tot)) === j) rr(c, 150, y - 6, 240, 40, 10, null, C.green, 2.5); }
  });
  step(c, ['1 · read the visitor’s features', '2 · score ⚽: features × its learned weights', '3 · score 💻 the same way', '4 · score 🌻',
    '5 · add a gold bonus where the guess is shaky', '6 · show the article with the highest total', '6 · show the article with the highest total'][ph], ph >= 5 ? C.green : C.fg);
};
CAP.n2_ctxLine = 'Blue bars describe the visitor. For each article, teal bars are features times that article’s learned weights. Gold adds an uncertainty bonus, and the green box marks the article shown.';
})();
