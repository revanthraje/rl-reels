// Part 2 simulations: planning, model-free, deep RL (valueiter … targetnet). Loaded after anim.js.
'use strict';
(() => {
const CAP = RLAnim.CAP, TOUCH = RLAnim.TOUCH;
const f1 = v => (Math.round(v * 10) / 10).toFixed(1);
const ctl = (S, k, d) => (S.ctl && S.ctl[k] != null && !isNaN(S.ctl[k])) ? +S.ctl[k] : d;
function every(S, t, key, period) { // true once per period (ignoring time jumps)
  const k = 'nx_' + key; if (S[k] == null || t < S[k] - period * 2) S[k] = t + period;
  if (t >= S[k]) { S[k] = t + period; return true; } return false;
}
function label(c, s, x, y, col, size) { txt(c, s, x, y, { s: size || 12, b: 1, c: col || C.gold }); }

// ───────── value iteration on a grid: tap to move the goal ─────────
const VG = { W: 6, H: 4, x0: 62, y0: 30, s: 44, walls: new Set(['2,1', '2,2', '4,2']) };
function viSnaps(gx, gy, g) {
  return memo(`p2vi${gx},${gy},${g}`, () => {
    let V = new Float32Array(24); const out = [V.slice()];
    for (let k = 0; k < 14; k++) {
      const N = new Float32Array(24);
      for (let y = 0; y < VG.H; y++) for (let x = 0; x < VG.W; x++) {
        if (VG.walls.has(x + ',' + y) || (x === gx && y === gy)) continue;
        let best = -1e9;
        for (const [dx, dy] of DIRS) {
          let nx = x + dx, ny = y + dy; if (nx < 0 || ny < 0 || nx >= VG.W || ny >= VG.H || VG.walls.has(nx + ',' + ny)) { nx = x; ny = y; }
          const v = (nx === gx && ny === gy) ? 10 : g * V[ny * VG.W + nx]; if (v > best) best = v;
        }
        N[y * VG.W + x] = best;
      }
      V = N; out.push(V.slice());
    }
    return out;
  });
}
A.p2_viGoal = (c, t, S) => {
  const g = Math.round(ctl(S, 'g', .9) * 100) / 100;
  S.goal ||= [5, 0]; if (S.t0 == null) S.t0 = t;
  while (S.taps && S.taps.length) {
    const q = S.taps.shift(), x = Math.floor((q.x - VG.x0) / VG.s), y = Math.floor((q.y - VG.y0) / VG.s);
    if (x >= 0 && y >= 0 && x < VG.W && y < VG.H && !VG.walls.has(x + ',' + y)) { S.goal = [x, y]; S.t0 = t; }
  }
  const snaps = viSnaps(S.goal[0], S.goal[1], g), el = t - S.t0, k = Math.min(14, Math.floor(el / 1.3));
  if (el > 14 * 1.3 + 5) S.t0 = t; // loop
  const V = snaps[k], P = snaps[Math.max(0, k - 1)], u = clamp((el - k * 1.3) / .5, 0, 1);
  for (let y = 0; y < VG.H; y++) for (let x = 0; x < VG.W; x++) {
    const px = VG.x0 + x * VG.s, py = VG.y0 + y * VG.s, i = y * VG.W + x, wall = VG.walls.has(x + ',' + y), goal = x === S.goal[0] && y === S.goal[1];
    const changed = Math.abs(V[i] - P[i]) > .01 && k > 0;
    rr(c, px + 1.5, py + 1.5, VG.s - 3, VG.s - 3, 6, wall ? C.wall : goal ? alpha(C.green, .5) : (V[i] > 0 ? heat(V[i] / 10) : C.panel), changed && u < 1 ? C.fg : null, 2.5);
    if (goal) txt(c, '🏁', px + VG.s / 2, py + VG.s / 2, { s: 18 });
    else if (!wall) txt(c, V[i] > 0 ? f1(V[i]) : '0', px + VG.s / 2, py + VG.s / 2, { s: 13, b: 1, c: V[i] > 4 ? C.bg : C.fg });
  }
  if (k >= 13) { // converged: show the plan
    for (let y = 0; y < VG.H; y++) for (let x = 0; x < VG.W; x++) {
      if (VG.walls.has(x + ',' + y) || (x === S.goal[0] && y === S.goal[1])) continue;
      let best = -1, ba = 0; DIRS.forEach(([dx, dy], a) => { const nx = x + dx, ny = y + dy; if (nx < 0 || ny < 0 || nx >= VG.W || ny >= VG.H || VG.walls.has(nx + ',' + ny)) return; const v = (nx === S.goal[0] && ny === S.goal[1]) ? 10.01 : V[ny * VG.W + nx]; if (v > best) { best = v; ba = a; } });
      const cx = VG.x0 + x * VG.s + VG.s / 2, cy = VG.y0 + y * VG.s + VG.s / 2; cellArrow(c, cx + DIRS[ba][0] * 12, cy + DIRS[ba][1] * 12 + 2, ba, 10, alpha(C.bg, .8), 2);
    }
  }
  label(c, k >= 13 ? 'done: arrows = best move from each square' : `sweep ${k}: value spreads one step further`, 200, 15, C.gold, 13);
  txt(c, `👆 tap a square to move the goal   ·   γ = ${g}`, 200, 222, { s: 12, c: C.dim });
  txt(c, 'number = value (goal pays 10)', 200, 242, { s: 11, c: C.dim });
};
TOUCH.add('p2_viGoal');
CAP.p2_viGoal = 'A maze with a goal flag. Every 1.3 seconds one **sweep** updates all squares: the numbers spread out from the goal, fading with distance. At the end, arrows show the best move.';

// ───────── value iteration in a corridor, γ slider ─────────
A.p2_viLine = (c, t, S) => {
  const g = ctl(S, 'g', .9), n = 6, x0 = 26, w = 58, y0 = 170, T = 1.8, K = 6, L = K * T + 4, tt = t % L, k = Math.min(K, Math.floor(tt / T)), u = clamp((tt - k * T) / .8, 0, 1);
  const val = (i, kk) => { const d = n - 1 - i; return d >= 1 && d <= kk ? 10 * Math.pow(g, d - 1) : 0; };
  for (let i = 0; i < n; i++) {
    const x = x0 + i * w, goal = i === n - 1, v = val(i, k), vp = val(i, k - 1), fresh = v !== vp && k > 0;
    rr(c, x + 3, y0, w - 6, 40, 8, goal ? alpha(C.green, .5) : C.panel, fresh ? C.gold : C.line, fresh ? 3 : 1.5);
    txt(c, goal ? '🏁 +10' : f1(fresh ? lerp(vp, v, u) : v), x + w / 2, y0 + 20, { s: 14, b: 1 });
    const h = (fresh ? lerp(vp, v, u) : v) * 10; if (!goal) bar(c, x + 14, y0 - 6, w - 28, h, alpha(C.teal, .85));
    if (fresh && i < n - 2) { curve(c, [x + w * 1.5, y0 - 6 - val(i + 1, k) * 10 - 4], [x + w / 2, y0 - 6 - h - 4], -18, C.gold, 4, 2); txt(c, `× ${g.toFixed(2)}`, x + w, y0 - 14 - Math.max(h, val(i + 1, k) * 10) - 18, { s: 12, b: 1, c: C.gold }); }
  }
  label(c, k === 0 ? 'start: every value is 0' : k >= K - 1 ? 'values fade with distance from the goal' : `sweep ${k}: one more square learns about the goal`, 200, 16, C.gold, 13);
  txt(c, `γ = ${g.toFixed(2)}: each step back keeps ${Math.round(g * 100)}% of the value`, 200, 236, { s: 12, c: C.dim });
};
CAP.p2_viLine = 'A row of squares with the goal on the right. Each sweep, one more square learns its value: the goal’s 10 shrinks by γ for every step of distance. Teal bars show the values.';

// ───────── policy iteration in a corridor: tap to step ─────────
function piSeq() {
  return memo('p2pi', () => {
    const n = 6, g = .9, step = (i, a) => Math.max(0, Math.min(n - 1, i + (a ? 1 : -1)));
    let pi = [0, 1, 0, 0, 1, -1]; const out = [];
    const evalPi = p => p.map((_, i) => { if (i === n - 1) return 0; let s = i, ret = 0, d = 1; for (let k = 0; k < 20; k++) { const ns = step(s, p[s]); if (ns === n - 1) { ret = 10 * d; break; } if (ns === s) break; s = ns; d *= g; } return ret; });
    for (let it = 0; it < 6; it++) {
      const V = evalPi(pi); out.push({ ph: 'eval', pi: pi.slice(), V, it });
      const np = pi.map((a, i) => { if (i === n - 1) return -1; const q = [0, 1].map(b => { const ns = step(i, b); return ns === n - 1 ? 10 : (ns === i ? 0 : g * V[ns]); }); return q[1] > q[0] + 1e-9 ? 1 : q[0] > q[1] + 1e-9 ? 0 : a; });
      const changed = np.map((a, i) => a !== pi[i]);
      out.push({ ph: 'impr', pi: np, V, it, changed }); if (!changed.some(Boolean)) break; pi = np;
    }
    return out;
  });
}
A.p2_piLoop = (c, t, S) => {
  const seq = piSeq(); S.ph ??= 0;
  if (S.taps && S.taps.length) { S.taps.length = 0; S.ph = (S.ph + 1) % seq.length; S.nx_pi = t + 3; }
  else if (every(S, t, 'pi', S.ph === seq.length - 1 ? 5 : 3)) S.ph = (S.ph + 1) % seq.length;
  const st = seq[S.ph], n = 6, x0 = 26, w = 58, y0 = 110;
  label(c, st.ph === 'eval' ? `① Evaluate: how good is the current plan?` : st.changed.some(Boolean) ? '② Improve: switch to moves that look better' : '② Improve: nothing changes → this plan is optimal ✓', 200, 22, st.ph === 'eval' ? C.teal : C.gold, 13);
  txt(c, `round ${st.it + 1}`, 200, 44, { s: 12, c: C.dim });
  for (let i = 0; i < n; i++) {
    const x = x0 + i * w, goal = i === n - 1;
    rr(c, x + 3, y0, w - 6, 56, 8, goal ? alpha(C.green, .5) : st.V[i] > 0 ? heat(st.V[i] / 10 * .9) : C.panel, st.changed && st.changed[i] ? C.gold : C.line, st.changed && st.changed[i] ? 3 : 1.5);
    if (goal) txt(c, '🏁', x + w / 2, y0 + 28, { s: 20 });
    else {
      const a = st.pi[i]; arrow(c, x + w / 2 - (a ? 14 : -14), y0 + 20, x + w / 2 + (a ? 14 : -14), y0 + 20, st.changed && st.changed[i] ? C.gold : C.fg, 3, 9);
      txt(c, st.ph === 'eval' || S.ph > 0 ? f1(st.V[i]) : '?', x + w / 2, y0 + 42, { s: 13, b: 1, c: st.V[i] > 5 ? C.bg : C.fg });
    }
  }
  txt(c, 'arrow = the plan in that square   ·   number = its value', 200, 192, { s: 12, c: C.dim });
  txt(c, '👆 tap to take the next step', 200, 222, { s: 12, c: C.gold });
};
TOUCH.add('p2_piLoop');
CAP.p2_piLoop = 'Arrows are the current plan in each square. **Evaluate** fills in how much each square is worth under that plan, **improve** flips arrows (gold) that point the wrong way. Repeat until nothing flips.';

// ───────── Monte Carlo averaging: tap to play an episode ─────────
A.p2_mcAvg = (c, t, S) => {
  const sd = ctl(S, 'luck', 2.5); S.r ||= mulberry(11);
  if (!S.rets || S.sd !== sd) { S.rets = sd === 2.5 && !S.rets ? [10, 4, 7, 9] : []; S.sd = sd; S.nx_mc = t + 1.2; }
  const add = () => { if (S.rets.length >= 30) S.rets = []; const fixed = [10, 4, 7, 9]; S.rets.push(S.rets.length < 4 && sd === 2.5 ? fixed[S.rets.length] : clamp(7.5 + gauss(S.r) * sd, 0, 15)); S.flash = t; };
  if (S.taps && S.taps.length) { S.taps.length = 0; add(); S.nx_mc = t + 2; } else if (every(S, t, 'mc', 1.4)) add();
  const x0 = 40, x1 = 380, y0 = 200, sc = 10, X = i => x0 + (i + .5) * (x1 - x0) / 30, Y = v => y0 - v * sc;
  line(c, x0, y0, x1, y0, C.line, 1.5); line(c, x0, y0, x0, Y(15), C.line, 1.5);
  [0, 5, 10, 15].forEach(v => txt(c, v, x0 - 12, Y(v), { s: 10, c: C.dim }));
  line(c, x0, Y(7.5), x1, Y(7.5), C.gold, 2, [5, 5]); txt(c, 'true value', x1 - 4, Y(7.5) - 9, { s: 10, c: C.gold, a: 'right' });
  let sum = 0; const avg = [];
  S.rets.forEach((r, i) => { sum += r; avg.push([X(i), Y(sum / (i + 1))]); dot(c, X(i), Y(r), i === S.rets.length - 1 && t - S.flash < .6 ? 6 : 4, alpha(C.blue, .9)); });
  if (avg.length > 1) poly(c, avg, C.pink, 3); if (avg.length) dot(c, ...avg.at(-1), 5, C.pink);
  const n = S.rets.length, last = S.rets[n - 1];
  label(c, n ? `game ${n}: return ${f1(last)}  →  average ${f1(sum / n)}` : 'play some games…', 200, 18, C.fg, 13);
  dot(c, 70, 44, 4, C.blue); txt(c, 'one game’s return', 78, 44, { a: 'left', s: 11, c: C.dim });
  line(c, 190, 44, 210, 44, C.pink, 3); txt(c, 'running average = estimate', 216, 44, { a: 'left', s: 11, c: C.dim });
  txt(c, '👆 tap to play another game', 200, 228, { s: 12, c: C.gold });
};
TOUCH.add('p2_mcAvg');
CAP.p2_mcAvg = 'Each blue dot is the total reward of one finished game from the same state. The pink line is their running average, the value estimate. It wobbles at first, then settles near the gold true value.';

// ───────── TD on a commute ─────────
A.p2_tdCommute = (c, t) => {
  const pts = [['leave office', 30], ['raining at car', 35], ['highway jam', 40], ['home', 43]], T = 2.6, L = pts.length * T + 4, tt = t % L, k = Math.min(pts.length - 1, Math.floor(tt / T)), u = clamp((tt - k * T) / 1, 0, 1);
  const x0 = 92, dx = 92, X = i => x0 + i * dx;
  line(c, X(0), 70, X(3), 70, C.line, 6); pts.forEach((p, i) => { dot(c, X(i), 70, 7, i <= k ? C.gold : C.panel, C.line); txt(c, p[0], X(i), 50, { s: 11, c: i <= k ? C.fg : C.dim }); });
  const carX = k < 3 ? lerp(X(k), X(k + 1), clamp((tt - k * T - 1.2) / 1.4, 0, 1)) : X(3); txt(c, '🚗', carX, 70, { s: 20 });
  txt(c, 'predicted total time', 200, 98, { s: 11, c: C.dim });
  const row = (y, name, col, upd) => {
    txt(c, name, 12, y + 14, { a: 'left', s: 13, b: 1, c: col });
    pts.forEach((p, i) => {
      if (i === 3) return; let v = 30;
      if (upd === 'td') v = i < k ? pts[i + 1][1] : i === k ? 30 + (k ? 0 : 0) : 30;
      if (upd === 'td' && i === k - 1) v = lerp(i === 0 ? 30 : pts[i][1], pts[k][1], u);
      if (upd === 'td' && i < k - 1) v = pts[i + 1][1];
      if (upd === 'td' && i === k) v = i === 0 ? 30 : pts[i][1];
      if (upd === 'mc') v = k === 3 ? lerp(i === 0 ? 30 : pts[i][1], 43, u) : (i === 0 ? 30 : pts[i][1]);
      if (i > k) { txt(c, '…', X(i), y + 14, { s: 13, c: C.dim }); return; }
      rr(c, X(i) - 26, y, 52, 28, 8, C.panel, col, 1.5); txt(c, Math.round(v) + ' min', X(i), y + 14, { s: 12, b: 1 });
    });
  };
  row(110, 'TD', C.teal, 'td'); row(160, 'MC', C.pink, 'mc');
  if (k > 0 && k < 3) curve(c, [X(k), 124], [X(k - 1) + 26, 124], -10, C.teal, 2, 2);
  label(c, k === 0 ? 'start: guess 30 minutes' : k < 3 ? 'TD: fix the last guess right away; MC must wait' : 'home after 43 min: MC finally updates everything', 200, 210, k < 3 ? C.teal : C.pink, 12);
  txt(c, 'each box = guess made at that point', 200, 236, { s: 11, c: C.dim });
};
CAP.p2_tdCommute = 'A drive home with checkpoints. Each box is the guess for the total trip time made at that point. The **TD** row corrects the previous guess at every checkpoint; the **Monte Carlo** row waits until you arrive.';

// ───────── TD updates with a learning-rate slider ─────────
A.p2_tdAlpha = (c, t, S) => {
  const a = ctl(S, 'a', .2); S.r ||= mulberry(5);
  if (!S.hist || S.a !== a) { S.hist = [{ v: 0, y: null }]; S.a = a; }
  if (every(S, t, 'td', .55)) { const v = S.hist.at(-1).v, y = 6 + gauss(S.r) * 2; S.hist.push({ v: v + a * (y - v), y, e: y - v }); if (S.hist.length > 48) S.hist.shift(); }
  const x0 = 30, x1 = 300, y0 = 200, sc = 13, X = i => x0 + i * (x1 - x0) / 47, Y = v => y0 - v * sc;
  line(c, x0, y0, x1, y0, C.line, 1.5); line(c, x0, Y(6), x1, Y(6), C.gold, 2, [5, 5]); txt(c, 'true value 6', x1, Y(6) - 9, { a: 'right', s: 10, c: C.gold });
  S.hist.forEach((h, i) => { if (h.y != null) dot(c, X(i), Y(clamp(h.y, 0, 13)), 3, alpha(C.blue, .8)); });
  poly(c, S.hist.map((h, i) => [X(i), Y(h.v)]), C.pink, 3); dot(c, X(S.hist.length - 1), Y(S.hist.at(-1).v), 5, C.pink);
  const e = S.hist.at(-1).e || 0; rr(c, 318, 60, 70, 150, 10, C.panel, C.line);
  txt(c, 'TD error', 353, 74, { s: 11, b: 1 }); line(c, 326, 135, 380, 135, C.line, 1);
  bar(c, 340, 135, 26, clamp(e, -5, 5) * 12, e >= 0 ? C.green : C.red); txt(c, (e >= 0 ? '+' : '') + f1(e), 353, 198, { s: 12, b: 1, c: e >= 0 ? C.green : C.red });
  label(c, `guess += ${a.toFixed(2)} × error`, 165, 18, C.fg, 13);
  dot(c, 40, 42, 3, C.blue); txt(c, 'noisy target (reward + next guess)', 48, 42, { a: 'left', s: 11, c: C.dim });
  line(c, 40, 230, 58, 230, C.pink, 3); txt(c, 'the value guess', 64, 230, { a: 'left', s: 11, c: C.dim });
};
CAP.p2_tdAlpha = 'Blue dots are noisy targets, one per step. The pink line is the value guess: each step it moves a fraction α of the **TD error** (the green or red bar) toward the target.';

// ───────── Q-learning arithmetic ─────────
A.p2_qUpdate = (c, t) => {
  const T = 1.9, L = 6 * T + 3, k = Math.min(5, Math.floor((t % L) / T));
  rr(c, 30, 60, 100, 100, 12, C.panel, k >= 5 ? C.gold : C.line, 2); txt(c, 'here', 80, 76, { s: 11, c: C.dim });
  txt(c, 'Q(right)', 80, 104, { s: 12 }); txt(c, k >= 5 ? '2.35' : '2', 80, 128, { s: 22, b: 1, c: k >= 5 ? C.gold : C.fg });
  agent(c, 80, 150, C.pink, 8);
  rr(c, 260, 60, 110, 100, 12, C.panel, C.line, 2); txt(c, 'next square', 315, 76, { s: 11, c: C.dim });
  [['↑', 1], ['→', 5], ['↓', 0], ['←', 2]].forEach(([d, v], i) => { const best = v === 5 && k >= 2; rr(c, 270 + (i % 2) * 48, 88 + Math.floor(i / 2) * 34, 42, 28, 6, best ? alpha(C.gold, .35) : C.bg, best ? C.gold : null); txt(c, `${d} ${v}`, 291 + (i % 2) * 48, 102 + Math.floor(i / 2) * 34, { s: 13, b: 1 }); });
  arrow(c, 136, 110, 252, 110, k >= 1 ? C.pink : C.line, 3, 10); if (k >= 1) txt(c, 'move right: reward +1', 194, 96, { s: 12, b: 1, c: C.green });
  const lines = ['', 'got reward 1', 'best next move is worth 5', 'target = 1 + 0.9 × 5 = 5.5', 'error = 5.5 − 2 = 3.5', 'new Q = 2 + 0.1 × 3.5 = 2.35'];
  for (let i = 1; i <= k; i++) txt(c, lines[i], 200, 160 + i * 16, { s: i === k ? 14 : 12, b: i === k ? 1 : 0, c: i === k ? C.gold : C.dim });
  label(c, `step ${k} of 5`, 200, 22, C.fg, 13);
};
CAP.p2_qUpdate = 'One Q-learning update, slowly. The robot moves right and gets 1. The best value in the next square is 5. Target 5.5, error 3.5, and the old 2 moves a tenth of the way: 2.35.';

// ───────── Q-table filling up in a corridor ─────────
function qTableRun() {
  return memo('p2qt', () => {
    const n = 5, Q = [...Array(n)].map(() => [0, 0]), r = mulberry(8), out = []; let ep = 0;
    while (out.length < 90) {
      let s = 0; ep++;
      while (s < n - 1 && out.length < 90) {
        const a = r() < .3 ? (r() < .5 ? 0 : 1) : (Q[s][1] > Q[s][0] ? 1 : Q[s][0] > Q[s][1] ? 0 : (r() < .5 ? 0 : 1));
        const ns = clamp(s + (a ? 1 : -1), 0, n - 1), rew = ns === n - 1 ? 10 : 0, tgt = rew + (ns === n - 1 ? 0 : .9 * Math.max(...Q[ns]));
        Q[s][a] += .5 * (tgt - Q[s][a]); out.push({ s, a, ns, Q: Q.map(q => q.slice()), ep }); s = ns;
      }
    }
    return out;
  });
}
A.p2_qTable = (c, t) => {
  const run = qTableRun(), k = Math.floor(t / .8) % (run.length + 5), d = run[Math.min(k, run.length - 1)], n = 5, x0 = 70, w = 54;
  for (let i = 0; i < n; i++) { rr(c, x0 + i * w + 2, 34, w - 4, 34, 7, i === n - 1 ? alpha(C.green, .5) : C.panel); txt(c, i === n - 1 ? '🏁 +10' : 'S' + (i + 1), x0 + i * w + w / 2, 51, { s: 12, b: 1 }); }
  const fr = clamp((t % .8) / .5, 0, 1), ax = lerp(d.s, d.ns, ease(fr)); agent(c, x0 + ax * w + w / 2, 51, C.pink, 9);
  txt(c, 'state', 40, 96, { s: 11, c: C.dim }); txt(c, '← left', 40, 128, { s: 11, b: 1 }); txt(c, '→ right', 40, 162, { s: 11, b: 1 });
  for (let i = 0; i < n - 1; i++) {
    txt(c, 'S' + (i + 1), x0 + i * w + w / 2, 96, { s: 12, b: 1, c: C.dim });
    for (let a = 0; a < 2; a++) {
      const v = d.Q[i][a], hot = d.s === i && d.a === (a ? 1 : 0);
      rr(c, x0 + i * w + 3, 112 + (1 - a) * 0 + a * 34, w - 6, 30, 6, heat(v / 10), hot ? C.gold : null, 3);
      txt(c, f1(v), x0 + i * w + w / 2, 127 + a * 34, { s: 13, b: 1, c: v > 4 ? C.bg : C.fg });
    }
  }
  label(c, `episode ${d.ep}  ·  gold box = the number just updated`, 200, 14, C.gold, 12);
  txt(c, 'Q-table: one number per state and move', 200, 212, { s: 12, c: C.dim });
  txt(c, 'values for “right” grow back from the goal', 200, 232, { s: 12, c: C.dim });
};
CAP.p2_qTable = 'A robot explores a short corridor. Below is its **Q-table**: one box per square and move. After every step the gold box updates, and good values spread backwards from the goal.';

// ───────── cliff: Q-learning vs SARSA with an ε slider ─────────
const CW = 10;
function cliffLearn(eps, sarsa) {
  if (eps === 0 && sarsa) return cliffLearn(0, false); // with no exploration the two methods agree
  return memo(`p2cl${eps}${sarsa}`, () => {
    const Q = new Float32Array(40 * 4), r = mulberry(21 + (sarsa ? 1 : 0)), e = Math.max(eps, .02);
    const step = (x, y, a) => { let nx = clamp(x + DIRS[a][0], 0, CW - 1), ny = clamp(y + DIRS[a][1], 0, 3); if (ny === 3 && nx > 0 && nx < CW - 1) return [0, 3, -100, false]; return [nx, ny, -1, ny === 3 && nx === CW - 1]; };
    const pick = (x, y) => { const s = (y * CW + x) * 4; if (r() < e) return Math.floor(r() * 4); let b = 0; for (let a = 1; a < 4; a++) if (Q[s + a] > Q[s + b]) b = a; return b; };
    for (let ep = 0; ep < 1600; ep++) {
      const lr = ep < 1000 ? .4 : .1; let x = 0, y = 3, a = pick(x, y);
      for (let k = 0; k < 300; k++) {
        const [nx, ny, rew, done] = step(x, y, a), s = (y * CW + x) * 4, ns = (ny * CW + nx) * 4, na = pick(nx, ny);
        const nv = done ? 0 : sarsa ? Q[ns + na] : Math.max(Q[ns], Q[ns + 1], Q[ns + 2], Q[ns + 3]);
        Q[s + a] += lr * (rew + nv - Q[s + a]); if (done) break; x = nx; y = ny; a = na;
      }
    }
    const path = [[0, 3]]; let x = 0, y = 3;
    for (let k = 0; k < 40 && !(x === CW - 1 && y === 3); k++) { const s = (y * CW + x) * 4; let b = 0; for (let a = 1; a < 4; a++) if (Q[s + a] > Q[s + b]) b = a; const [nx, ny, rew] = step(x, y, b); if (rew === -100) break; x = nx; y = ny; path.push([x, y]); }
    return { path, Q };
  });
}
A.p2_cliffEps = (c, t, S) => {
  const eps = Math.round(ctl(S, 'e', .1) * 20) / 20, s = 32, x0 = 40, y0 = 66, P = (x, y) => [x0 + x * s + s / 2, y0 + y * s + s / 2];
  for (let y = 0; y < 4; y++) for (let x = 0; x < CW; x++) { const cl = y === 3 && x > 0 && x < CW - 1; rr(c, x0 + x * s + 1, y0 + y * s + 1, s - 2, s - 2, 4, cl ? alpha(C.red, .45) : C.panel); }
  txt(c, 'C L I F F  −100', x0 + 5 * s, y0 + 3.5 * s, { s: 12, b: 1, c: C.red }); txt(c, 'S', ...P(0, 3), { b: 1 }); txt(c, 'G', ...P(CW - 1, 3), { b: 1, c: C.green });
  const runs = [[false, C.pink, 'Q-learning'], [true, C.teal, 'SARSA']];
  S.w ||= {}; S.falls ||= { false: 0, true: 0 }; S.trips ||= { false: 0, true: 0 };
  if (S.eps !== eps) { S.eps = eps; S.w = {}; S.falls = { false: 0, true: 0 }; S.trips = { false: 0, true: 0 }; S.r = mulberry(3); }
  const tick = every(S, t, 'cl', .32);
  runs.forEach(([sarsa, col], j) => {
    const { path, Q } = cliffLearn(eps, sarsa); poly(c, path.map(q => P(...q)), alpha(col, .55), 3, [5, 4]);
    const w = S.w[sarsa] ||= { x: 0, y: 3, fall: 0 };
    if (tick) {
      if (w.fall) { w.fall--; if (!w.fall) { w.x = 0; w.y = 3; } }
      else {
        const st = (w.y * CW + w.x) * 4; let a = 0; for (let b = 1; b < 4; b++) if (Q[st + b] > Q[st + a]) a = b; if (S.r() < eps) a = Math.floor(S.r() * 4);
        w.x = clamp(w.x + DIRS[a][0], 0, CW - 1); w.y = clamp(w.y + DIRS[a][1], 0, 3);
        if (w.y === 3 && w.x > 0 && w.x < CW - 1) { w.fall = 4; S.falls[sarsa]++; S.trips[sarsa]++; }
        else if (w.y === 3 && w.x === CW - 1) { w.fall = 2; S.trips[sarsa]++; }
      }
    }
    const [px, py] = P(w.x, w.y); dot(c, px + (j ? 4 : -4), py, 9, w.fall && w.y === 3 && w.x > 0 && w.x < CW - 1 ? C.red : col);
    dot(c, 30 + j * 190, 18, 7, col); txt(c, `${runs[j][2]}: fell ${S.falls[sarsa]} / ${S.trips[sarsa]} trips`, 42 + j * 190, 18, { a: 'left', s: 12, b: 1 });
  });
  txt(c, `ε = ${eps.toFixed(2)}: chance of a random step   ·   dashed = learned route`, 200, 44, { s: 11, c: C.dim });
};
CAP.p2_cliffEps = 'Both robots explore with random steps. Pink, Q-learning, learned the route along the cliff edge, so random steps make it fall. Teal, SARSA, counted its own randomness and walks further from the edge.';

// ───────── on-policy vs off-policy ─────────
A.p2_onOff = (c, t) => {
  const r = mulberry(Math.floor(t / 7) + 1), T = 7, tt = t % T;
  const pan = (x0, title, col, off) => {
    rr(c, x0, 40, 180, 180, 14, C.panel, C.line); txt(c, title, x0 + 90, 56, { s: 13, b: 1, c: col });
    const n = 5, s = 28, gx = x0 + 20, gy = 72;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) rr(c, gx + x * s + 1, gy + y * s + 1, s - 2, s - 2, 4, x === 4 && y === 0 ? alpha(C.green, .5) : C.bg);
    const pts = [[0, 4]]; let x = 0, y = 4; const rr2 = mulberry(off ? 7 : 9);
    for (let k = 0; k < 14; k++) { const a = off ? Math.floor(rr2() * 4) : (rr2() < .2 ? Math.floor(rr2() * 4) : (rr2() < .5 ? 0 : 1)); x = clamp(x + DIRS[a][0], 0, 4); y = clamp(y + DIRS[a][1], 0, 4); pts.push([x, y]); }
    const f = clamp(tt / 5, 0, 1) * (pts.length - 1), cp = pts.map(q => [gx + q[0] * s + s / 2, gy + q[1] * s + s / 2]);
    poly(c, cp.slice(0, Math.floor(f) + 1).concat([lerpPath(cp, f)]), alpha(C.blue, .7), 2.5); dot(c, ...lerpPath(cp, f), 6, C.blue);
    if (tt > 2.5) { const u = clamp((tt - 2.5) / 1.5, 0, 1); c.globalAlpha = u; for (let y2 = 0; y2 < n; y2++) for (let x2 = 0; x2 < n; x2++) { if (x2 === 4 && y2 === 0) continue; const a = (x2 < 4 && (y2 === 0 || (x2 + y2) % 2)) ? 1 : 0; cellArrow(c, gx + x2 * s + s / 2, gy + y2 * s + s / 2, a, 10, off ? C.gold : C.teal, 2); } c.globalAlpha = 1; }
    txt(c, off ? 'acts randomly · learns the best plan' : 'acts with its plan · learns that plan', x0 + 90, 236, { s: 11, c: C.dim });
  };
  pan(14, 'ON-policy (SARSA)', C.teal, false); pan(206, 'OFF-policy (Q-learning)', C.gold, true);
  txt(c, 'blue trail = how it acts   ·   arrows = the plan it learns', 200, 18, { s: 12, b: 1 });
};
CAP.p2_onOff = 'Two learners. The blue trail is how each one actually behaves; the arrows are the plan it is learning. On the left they are the same plan. On the right, random wandering still teaches the best plan.';

// ───────── eligibility traces with a λ slider ─────────
A.p2_lambda = (c, t, S, p = {}) => {
  const lam = p.fp ? .8 : ctl(S, 'l', .8), n = 8, x0 = 22, w = 45, T = .9, L = (n - 1) * T + 4.5, tt = t % L, k = Math.min(n - 1, Math.floor(tt / T)), done = tt > (n - 1) * T + .4;
  for (let i = 0; i < n; i++) {
    const x = x0 + i * w, tr = i <= k ? Math.pow(lam, k - i) : 0, goal = i === n - 1;
    rr(c, x + 3, 150, w - 6, 36, 7, goal ? alpha(C.green, .5) : C.panel, C.line); txt(c, goal ? '+1' : (p.fp ? '👣' : 's' + (i + 1)), x + w / 2, 168, { s: 12, b: 1, c: goal ? C.fg : C.dim, al: p.fp && !goal ? (i <= k ? .25 + .75 * tr : .15) : 1 });
    if (!goal && i <= k) { bar(c, x + 10, 142, w - 20, tr * 72, alpha(C.purple, .85)); if (tr > .05) txt(c, f1(tr), x + w / 2, 134 - tr * 72, { s: 10, c: C.purple }); }
    if (done && !goal && i <= k) { const u = clamp((tt - (n - 1) * T - .4) / 1.2, 0, 1); bar(c, x + 10, 222, w - 20, u * tr * 26, C.green); }
  }
  agent(c, x0 + lerp(k, Math.min(n - 1, k + 1), ease(clamp((tt - k * T) / .5, 0, 1)) * (k < n - 1 ? 1 : 0)) * w + w / 2, 168, C.pink, 9);
  label(c, done ? 'reward! every state learns in proportion to its trace' : 'purple bar = trace: how recently the state was visited', 200, 18, done ? C.green : C.purple, 12);
  txt(c, `λ = ${lam.toFixed(2)}: each step, old traces shrink to ${Math.round(lam * 100)}%`, 200, 40, { s: 12, c: C.dim });
  if (done) txt(c, 'green = value increase', 200, 238, { s: 11, c: C.green });
};
CAP.p2_lambda = 'The robot walks right. Every square it passes gets a purple **trace** bar that shrinks by λ each step. When the reward arrives, each square gets credit in proportion to its trace (green).';

// ───────── function approximation: tap to add data ─────────
A.p2_fit = (c, t, S, p = {}) => {
  const rbf = p.mode === 'rbf', x0 = 40, x1 = 380, y0 = 205, y1 = 45, X = u => lerp(x0, x1, u), Y = v => lerp(y0, y1, v), iu = x => (x - x0) / (x1 - x0), iv = y => (y - y0) / (y1 - y0);
  line(c, x0, y0, x1, y0, C.line, 1.5); line(c, x0, y0, x0, y1, C.line, 1.5);
  if (!rbf) {
    S.pts ||= [[.15, .2], [.35, .38], [.55, .5], [.75, .72]];
    while (S.taps && S.taps.length) { const q = S.taps.shift(), u = iu(q.x), v = iv(q.y); if (u > 0 && u < 1 && v > 0 && v < 1) { S.pts.push([u, v]); if (S.pts.length > 12) S.pts.shift(); S.flash = t; } }
    const n = S.pts.length, mx = S.pts.reduce((a, q) => a + q[0], 0) / n, my = S.pts.reduce((a, q) => a + q[1], 0) / n;
    let sxy = 0, sxx = 0; S.pts.forEach(([x, y]) => { sxy += (x - mx) * (y - my); sxx += (x - mx) ** 2; });
    const m = sxx ? sxy / sxx : 0, b = my - m * mx; S.m = S.m == null ? m : lerp(S.m, m, .08); S.b = S.b == null ? b : lerp(S.b, b, .08);
    line(c, X(0), Y(S.b), X(1), Y(S.m + S.b), C.pink, 3);
    S.pts.forEach((q, i) => dot(c, X(q[0]), Y(q[1]), i === n - 1 && t - (S.flash ?? -9) < .6 ? 7 : 5, C.blue));
    const qu = .5 + .42 * Math.sin(t * .5), qv = S.m * qu + S.b; line(c, X(qu), y0, X(qu), Y(qv), alpha(C.gold, .6), 1.5, [3, 3]); txt(c, '🏠?', X(qu), Y(qv) - 12, { s: 14 }); dot(c, X(qu), Y(qv), 4, C.gold);
    txt(c, 'house size →', x1, y0 + 14, { a: 'right', s: 11, c: C.dim }); txt(c, 'price', x0 + 4, y1 - 8, { a: 'left', s: 11, c: C.dim });
    label(c, 'blue = houses seen   ·   pink = learned rule', 200, 16, C.fg, 12);
    txt(c, '👆 tap to add a house · 🏠? = a house never seen', 200, 240, { s: 12, c: C.gold });
    return;
  }
  // rbf mode: one update moves the neighbours too
  const K = 7, cen = [...Array(K)].map((_, i) => i / (K - 1)), phi = u => cen.map(m => Math.exp(-((u - m) ** 2) / (2 * .13 ** 2)));
  S.w ||= [.3, .45, .5, .4, .45, .55, .5].map(v => v); S.old ||= null;
  const demo = [[.3, .85], [.7, .15], [.5, .9], [.85, .8]];
  let tgt = null;
  if (S.taps && S.taps.length) { const q = S.taps.shift(), u = iu(q.x), v = iv(q.y); if (u > 0 && u < 1) tgt = [u, clamp(v, 0, 1)]; S.nx_rb = t + 6; }
  else if (every(S, t, 'rb', 4.5)) { S.di = ((S.di ?? -1) + 1) % demo.length; tgt = demo[S.di]; }
  if (tgt) { S.tg = tgt; S.old = S.w.slice(); S.steps = 0; }
  if (S.tg && S.steps < 40) { const f = phi(S.tg[0]), pred = f.reduce((a, v, i) => a + v * S.w[i], 0), err = S.tg[1] - pred, nn = f.reduce((a, v) => a + v * v, 0); S.w = S.w.map((w, i) => w + .06 * err * f[i] / nn); S.steps++; }
  const curvePts = w => [...Array(61)].map((_, i) => { const u = i / 60; return [X(u), Y(phi(u).reduce((a, v, j) => a + v * w[j], 0))]; });
  if (S.old) poly(c, curvePts(S.old), alpha(C.dim, .7), 2, [5, 4]);
  poly(c, curvePts(S.w), C.pink, 3);
  if (S.tg) { dot(c, X(S.tg[0]), Y(S.tg[1]), 7, C.gold); txt(c, 'new target', X(S.tg[0]), Y(S.tg[1]) - 14, { s: 11, b: 1, c: C.gold }); line(c, X(S.tg[0]), y0, X(S.tg[0]), y1, alpha(C.gold, .3), 1, [3, 3]); }
  label(c, 'grey dashed = before   ·   pink = after one target', 200, 16, C.fg, 12);
  txt(c, 'nearby states moved too, even ones we never trained on', 200, 222, { s: 11, c: C.dim });
  txt(c, '👆 tap anywhere to set a new target', 200, 241, { s: 12, c: C.gold });
};
TOUCH.add('p2_fit');
CAP.p2_fit = 'Blue dots are houses we have seen, the pink line is the learned rule. The gold 🏠 is a house never seen: the rule still predicts its price. Add points and the rule refits.';

// ───────── DQN: one frame vs four ─────────
A.p2_frames = (c, t) => {
  const pos = s => { const u = (s * .35) % 2, x = 40 + 100 * Math.abs(((s * .23) % 2) - 1); return [x, 80 + 90 * Math.abs(u - 1)]; };
  const box = (x0, title) => { rr(c, x0, 40, 170, 170, 12, '#05060d', C.line); txt(c, title, x0 + 85, 26, { s: 12, b: 1 }); for (let i = 0; i < 6; i++) rr(c, x0 + 8 + i * 26.5, 50, 24, 9, 2, [C.red, C.gold, C.green, C.blue, C.purple, C.teal][i]); rr(c, x0 + 60, 196, 50, 6, 3, C.fg); };
  box(18, '1 frame'); box(212, 'last 4 frames stacked');
  const [bx, by] = pos(t); dot(c, 18 + bx, by, 6, C.fg); txt(c, '↑ or ↓ ?', 103, 226, { s: 13, b: 1, c: C.red });
  for (let k = 3; k >= 0; k--) { const [x, y] = pos(t - k * .25); dot(c, 212 + x, y, 6, alpha('#eef0ff', 1 - k * .22)); }
  const [px, py] = pos(t - .25); arrow(c, 212 + px, py, 212 + bx + (bx - px) * 2.5, by + (by - py) * 2.5, C.gold, 2.5, 8);
  txt(c, 'direction visible ✓', 297, 226, { s: 13, b: 1, c: C.green });
};
CAP.p2_frames = 'A ball in a brick game. On the left, one still frame: you can’t tell where the ball is going. On the right, the last four frames overlap as a fading trail, so the direction is obvious.';

// ───────── replay buffer: shuffled batches / prioritised sampling ─────────
A.p2_buffer = (c, t, S, p = {}) => {
  if (p.mode === 'prio') {
    const pr = ctl(S, 'pr', .6), errs = [.2, 1.6, .4, .3, 2.4, .5, .25, 1.1, .35, .6], n = errs.length, x0 = 30, w = 34, y0 = 170;
    if (S.pr !== pr) { S.pr = pr; S.cnt = new Array(n).fill(0); S.r = mulberry(4); S.last = -1; }
    const wts = errs.map(e => Math.pow(e, pr)), Z = wts.reduce((a, b) => a + b, 0);
    if (every(S, t, 'pb', .3)) { let u = S.r() * Z, i = 0; while (u > wts[i] && i < n - 1) { u -= wts[i]; i++; } S.cnt[i]++; S.last = i; S.lt = t; }
    const mx = Math.max(4, ...S.cnt);
    errs.forEach((e, i) => {
      const x = x0 + i * w, hot = S.last === i && t - S.lt < .25;
      rr(c, x + 3, y0, w - 6, 26, 6, hot ? C.gold : C.panel, C.line); txt(c, '📼', x + w / 2, y0 + 13, { s: 13 });
      bar(c, x + 8, y0 - 4, w - 16, e * 22, alpha(C.red, .7)); txt(c, f1(e), x + w / 2, y0 - 12 - e * 22, { s: 10, c: C.red });
      bar(c, x + 8, y0 + 32, w - 16, -S.cnt[i] / mx * 40, alpha(C.teal, .85)); txt(c, S.cnt[i], x + w / 2, y0 + 82, { s: 10, c: C.teal });
    });
    txt(c, 'red = surprise (TD error)   ·   teal = times replayed', 200, 18, { s: 12, b: 1 });
    txt(c, pr < .05 ? 'priority 0: every memory equally likely' : `priority ${pr.toFixed(2)}: surprising memories picked more`, 200, 40, { s: 12, c: C.gold });
    return;
  }
  const cols = [C.red, C.gold, C.green, C.blue, C.purple], n = 30, k = Math.floor(t / .6), x0 = 40, s = 32;
  txt(c, 'replay buffer (colour = moment in the game)', 200, 16, { s: 12, b: 1 });
  for (let i = 0; i < n; i++) { const id = k - n + 1 + i; if (id < 0) continue; const cx = x0 + (i % 10) * s, cy = 30 + Math.floor(i / 10) * 30; rr(c, cx + 2, cy + 2, s - 4, 26, 5, cols[Math.floor(id / 6) % 5], i === n - 1 ? C.fg : null, 2); }
  const r = mulberry(Math.floor(t / 2.4) + 3), last = [...Array(4)].map((_, j) => k - 3 + j), rnd = [...Array(4)].map(() => k - Math.floor(r() * Math.min(n, k + 1)));
  const row = (y, title, ids, col) => { txt(c, title, 20, y, { a: 'left', s: 12, b: 1, c: col }); ids.forEach((id, j) => rr(c, 200 + j * 40, y - 13, 34, 26, 5, id >= 0 ? cols[Math.floor(id / 6) % 5] : C.panel)); };
  row(150, 'batch in order:', last, C.red); txt(c, 'all from the same moment ✗', 280, 172, { s: 11, c: C.red });
  row(205, 'random batch:', rnd, C.green); txt(c, 'a healthy mix ✓', 280, 227, { s: 11, c: C.green });
};
CAP.p2_buffer = 'Each tile is one stored experience, coloured by when it happened. A batch taken in order is all one colour, nearly identical. A random batch from the buffer mixes many moments.';

// ───────── target network: copy every N steps (or soft updates) ─────────
A.p2_chase = (c, t, S, p = {}) => {
  const soft = p.mode === 'soft', N = Math.round(ctl(S, 'N', 1)), tau = ctl(S, 'tau', .05), key = soft ? 's' + tau : 'h' + N;
  if (S.key !== key) { S.key = key; S.w = 0; S.wp = 0; S.wt = 0; S.k = 0; S.hist = []; S.r = mulberry(3); }
  if (every(S, t, 'ch', .09)) {
    if (S.k >= 160) { S.w = 0; S.wp = 0; S.wt = 0; S.k = 0; S.hist = []; S.r = mulberry(3); }
    const same = !soft && N === 1, y = 1 + .9 * (same ? S.w : S.wt) + (same ? 1.5 * (S.w - S.wp) : 0);
    S.wp = S.w; S.w += .5 * (y - S.w) + gauss(S.r) * .15;
    if (soft) S.wt = (1 - tau) * S.wt + tau * S.w; else if (S.k % N === N - 1) S.wt = S.w;
    S.hist.push([S.w, y]); S.k++;
  }
  const x0 = 34, x1 = 386, y0 = 210, sc = 11.5, X = i => x0 + i * (x1 - x0) / 160, Y = v => y0 - clamp(v, -1, 14) * sc;
  line(c, x0, y0, x1, y0, C.line, 1.5); line(c, x0, Y(10), x1, Y(10), alpha(C.green, .6), 1.5, [5, 5]); txt(c, 'goal 10', x1, Y(10) - 9, { a: 'right', s: 10, c: C.green });
  if (S.hist.length > 1) { poly(c, S.hist.map((h, i) => [X(i), Y(h[1])]), alpha(C.gold, .9), 2); poly(c, S.hist.map((h, i) => [X(i), Y(h[0])]), C.pink, 3); }
  line(c, 40, 40, 58, 40, C.gold, 2); txt(c, 'target it learns toward', 64, 40, { a: 'left', s: 11, c: C.dim });
  line(c, 220, 40, 238, 40, C.pink, 3); txt(c, 'network’s prediction', 244, 40, { a: 'left', s: 11, c: C.dim });
  label(c, soft ? `soft update: target moves ${Math.round(tau * 1000) / 10}% of the way each step` : N === 1 ? 'no frozen copy: the target jumps every step' : `frozen copy refreshed every ${N} steps`, 200, 16, C.fg, 12);
};
CAP.p2_chase = 'Pink is the network’s prediction, gold the target it trains toward. With no frozen copy the target moves with every update, so pink overshoots and wobbles. A frozen copy is slower but steady.';
})();
