// New-topic simulations, part 4 (HER, RND & Go-Explore, Curriculum, Meta-RL). Loaded after anim.js.
'use strict';
{
const CAP = RLAnim.CAP, TOUCH = RLAnim.TOUCH;
const tick = (S, t) => { const dt = clamp(t - (S.lt ?? t), 0, .05); S.lt = t; return dt; };
const sig = x => 1 / (1 + Math.exp(-x));

// ───────── HER: a miss becomes a lesson ─────────
const HER_G = [[300, 70], [250, 120], [340, 150], [290, 95], [220, 80], [330, 110]];
const HER_L = [[230, 150], [320, 80], [260, 90], [210, 140], [300, 160], [250, 60]];
A.n4_herRelabel = (c, t) => {
  const P = 6, k = Math.floor(t / P), u = t % P, i = k % HER_G.length;
  const [gx, gy] = HER_G[i], [lx, ly] = HER_L[i], sx = 50, sy = 190;
  rr(c, 12, 30, 376, 182, 12, alpha(C.panel, .6), C.line, 1);
  txt(c, '🦾', sx, sy, { s: 28 });
  const relab = u > 3.2, gpos = relab ? [lerp(gx, lx, ease(clamp((u - 3.2) / 1, 0, 1))), lerp(gy, ly, ease(clamp((u - 3.2) / 1, 0, 1)))] : [gx, gy];
  if (relab) txt(c, '★', gx, gy, { s: 24, c: C.gold, al: .25 });
  txt(c, '★', gpos[0], gpos[1], { s: 26, c: relab ? C.green : C.gold });
  const f = clamp(u / 1.6, 0, 1), bx = lerp(sx, lx, f), by = lerp(sy, ly, f) - Math.sin(f * Math.PI) * 60;
  dot(c, bx, by, 8, C.pink);
  let msg, col;
  if (u < 1.6) { msg = '1 · Aim for the gold star…'; col = C.fg; }
  else if (u < 3.2) { msg = '2 · Missed! Reward 0. On its own, this teaches almost nothing.'; col = C.red; }
  else if (u < 4.6) { msg = '3 · Hindsight: “what if the goal had been HERE?”'; col = C.green; }
  else { msg = '4 · Stored as a success: reward +1 for that goal.'; col = C.green; }
  rr(c, 12, 218, 376, 30, 9, alpha(C.blue, .12), C.line, 1); txt(c, msg, 200, 233, { s: 12.5, b: 1, c: col });
  const done = k + (u > 4.6 ? 1 : 0);
  txt(c, `real successes: 0   ·   hindsight successes: ${done}`, 200, 16, { s: 12, c: C.gold });
};
CAP.n4_herRelabel = 'The robot arm throws the pink ball at the gold star and misses. Then the star slides to where the ball landed: the same throw is saved as a success for that goal.';

A.n4_goalInput = (c, t, S) => {
  const dt = tick(S, t);
  if (!S.g) S.g = { x: 80, y: 190, gx: 320, gy: 70, n: 0, idle: 0, r: mulberry(11), reached: false };
  const G = S.g;
  while (S.taps.length) { const q = S.taps.shift(); if (q.y > 34 && q.y < 214) { G.gx = clamp(q.x, 30, 370); G.gy = q.y; G.reached = false; G.idle = 0; } }
  const dx = G.gx - G.x, dy = G.gy - G.y, d = Math.hypot(dx, dy);
  if (d > 4) { const v = Math.min(d, 70 * dt); G.x += dx / d * v; G.y += dy / d * v; }
  else if (!G.reached) { G.reached = true; G.n++; }
  else { G.idle += dt; if (G.idle > 2.5) { G.gx = 40 + G.r() * 320; G.gy = 50 + G.r() * 150; G.reached = false; G.idle = 0; } }
  rr(c, 12, 30, 376, 184, 12, alpha(C.panel, .6), C.line, 1);
  txt(c, '★', G.gx, G.gy, { s: 26, c: G.reached ? C.green : C.gold });
  line(c, G.x, G.y, G.gx, G.gy, alpha(C.dim, .35), 1.5, [4, 5]);
  txt(c, '🤖', G.x, G.y, { s: 24 });
  rr(c, 12, 220, 376, 30, 9, alpha(C.blue, .12), C.line, 1);
  txt(c, `one policy, input = (where I am, where the ★ is)  →  move`, 200, 235, { s: 12, b: 1 });
  txt(c, `goals reached: ${G.n}`, 200, 16, { s: 12, c: C.gold });
};
TOUCH.add('n4_goalInput');
CAP.n4_goalInput = 'The gold star is the goal. The same robot policy steers to any star, because the goal is fed in as part of its input. A new star appears after each success.';

// ───────── RND: novelty bonus ─────────
const GW_W = 16, GW_H = 8, CS = 22, OX = 24, OY = 34;
const wallRnd = (x, y) => (x === 5 && y !== 2) || (x === 10 && y !== 6);
const nbrs = (x, y) => [[1, 0], [-1, 0], [0, 1], [0, -1]].map(([a, b]) => [x + a, y + b]).filter(([a, b]) => a >= 0 && b >= 0 && a < GW_W && b < GW_H && !wallRnd(a, b));
function gridBase(c) {
  for (let y = 0; y < GW_H; y++) for (let x = 0; x < GW_W; x++) if (wallRnd(x, y)) rr(c, OX + x * CS + 1, OY + y * CS + 1, CS - 2, CS - 2, 3, C.wall, C.line, 1);
}
A.n4_rndNovelty = (c, t, S) => {
  const beta = S.ctl.beta ?? 1;
  if (!S.rn || S.rn.beta !== beta || t - S.rn.t0 > 45) S.rn = { beta, t0: t, x: 1, y: 4, n: new Float32Array(GW_W * GW_H), r: mulberry(5), next: t, seen: 1 };
  const R = S.rn;
  if (!R.n[R.y * GW_W + R.x]) R.n[R.y * GW_W + R.x] = 1;
  while (t >= R.next) {
    const nb = nbrs(R.x, R.y), w = nb.map(([a, b]) => Math.exp(beta * 6 / Math.sqrt(1 + R.n[b * GW_W + a])));
    let u = R.r() * w.reduce((p, q) => p + q, 0), k = 0; while (k < nb.length - 1 && u > w[k]) { u -= w[k]; k++; }
    [R.x, R.y] = nb[k]; const id = R.y * GW_W + R.x; if (!R.n[id]) R.seen++; R.n[id]++; R.next += .22;
  }
  for (let y = 0; y < GW_H; y++) for (let x = 0; x < GW_W; x++) {
    if (wallRnd(x, y)) continue; const n = R.n[y * GW_W + x], bonus = 1 / Math.sqrt(1 + n);
    rr(c, OX + x * CS + 1, OY + y * CS + 1, CS - 2, CS - 2, 3, n ? mix(C.panel, C.teal, .15 + .5 * (1 - bonus)) : mix(C.panel, C.gold, .55), null);
  }
  gridBase(c);
  txt(c, '🤖', OX + R.x * CS + CS / 2, OY + R.y * CS + CS / 2, { s: 15 });
  const open = GW_W * GW_H - 14;
  txt(c, `gold = never visited (big bonus)   ·   teal = familiar (small bonus)`, 200, 16, { s: 11.5, c: C.dim });
  txt(c, `explored ${Math.round(100 * R.seen / open)}% of the 3 rooms   ·   bonus weight ${beta.toFixed(1)}`, 200, 230, { s: 12.5, b: 1, c: C.gold });
};
CAP.n4_rndNovelty = 'Three rooms joined by narrow doors. Gold squares are new, so they pay a big novelty bonus; squares fade to teal as the robot visits them. With a strong bonus it finds the far rooms.';

A.n4_goExplore = (c, t, S) => {
  if (!S.ge) {
    const R = mulberry(9), arch = new Map([['1,4', [[1, 4]]]]);
    S.ge = { R, arch, phase: 0, pt: t, path: [[1, 4]], k: 0, x: 1, y: 4, target: '1,4', steps: 0, iter: 0 };
  }
  const G = S.ge, el = t - G.pt;
  if (G.iter > 26) { delete S.ge; return A.n4_goExplore(c, t, S); }
  if (G.phase === 0 && el > 1.2) { G.phase = 1; G.pt = t; G.k = 0; G.x = 1; G.y = 4; }
  else if (G.phase === 1) { G.k = Math.floor(el / .07); if (G.k >= G.path.length - 1) { [G.x, G.y] = G.path.at(-1); G.phase = 2; G.pt = t; G.steps = 0; G.cur = G.path.slice(); } else [G.x, G.y] = G.path[G.k]; }
  else if (G.phase === 2 && el > (G.steps + 1) * .3) {
    const nb = nbrs(G.x, G.y), [nx, ny] = nb[Math.floor(G.R() * nb.length)];
    G.x = nx; G.y = ny; G.cur.push([nx, ny]); G.steps++;
    const key = nx + ',' + ny; if (!G.arch.has(key) || G.arch.get(key).length > G.cur.length) G.arch.set(key, G.cur.slice());
    if (G.steps >= 6) { G.phase = 0; G.pt = t; G.iter++;
      const keys = [...G.arch.keys()], score = k => { const [a, b] = k.split(',').map(Number); return a + G.R() * 3; };
      G.target = keys.reduce((p, q) => score(q) > score(p) ? q : p); G.path = G.arch.get(G.target); }
  }
  for (let y = 0; y < GW_H; y++) for (let x = 0; x < GW_W; x++) {
    if (wallRnd(x, y)) continue; const inA = G.arch.has(x + ',' + y);
    rr(c, OX + x * CS + 1, OY + y * CS + 1, CS - 2, CS - 2, 3, inA ? mix(C.panel, C.teal, .45) : C.panel, null);
  }
  gridBase(c);
  if (G.phase >= 1) poly(c, (G.phase === 1 ? G.path : G.cur).map(([a, b]) => [OX + a * CS + CS / 2, OY + b * CS + CS / 2]), alpha(G.phase === 1 ? C.blue : C.pink, .8), 2.5);
  const [tx, ty] = G.target.split(',').map(Number);
  rr(c, OX + tx * CS + 1, OY + ty * CS + 1, CS - 2, CS - 2, 3, null, C.gold, 2.5);
  txt(c, '🤖', OX + G.x * CS + CS / 2, OY + G.y * CS + CS / 2, { s: 15 });
  const msg = ['1 · Remember: pick a promising square from the archive (gold box)', '2 · Return: replay the saved route straight there (blue)', '3 · Explore: a few random steps from there (pink)'][G.phase];
  txt(c, msg, 200, 230, { s: 12, b: 1, c: [C.gold, C.blue, C.pink][G.phase] });
  txt(c, `archive: ${G.arch.size} squares (teal)   ·   round ${G.iter + 1}`, 200, 16, { s: 11.5, c: C.dim });
};
CAP.n4_goExplore = 'Teal squares are the archive of places reached so far. Each round: pick a promising square (gold box), replay the saved route there in blue, then explore randomly in pink.';

// ───────── Curriculum ─────────
A.n4_curriculum = (c, t, S) => {
  const start = Math.round(S.ctl.start ?? 1);
  if (!S.cu || S.cu.start !== start || t - S.cu.t0 > 50) S.cu = { start, t0: t, L: start, s: .4, hist: [], next: t + .5, r: mulberry(3), wins: 0 };
  const U = S.cu;
  while (t >= U.next && U.L <= 6) {
    const p = sig((U.s - U.L + .6) * 3), ok = U.r() < p;
    U.hist.push(ok); if (U.hist.length > 12) U.hist.shift();
    if (ok) { U.s += .09; U.wins++; }
    if (U.hist.length >= 6 && U.hist.slice(-6).filter(Boolean).length >= 5) { U.L++; U.hist = []; }
    U.next += .7;
  }
  for (let l = 1; l <= 6; l++) {
    const x = 30 + (l - 1) * 58, h = 22 + l * 22, y = 205 - h;
    rr(c, x, y, 50, h, 6, l < U.L && l >= start ? alpha(C.green, .5) : l < start ? alpha(C.panel, .4) : l === U.L ? alpha(C.gold, .55) : C.panel, l === U.L ? C.gold : C.line, l === U.L ? 2.5 : 1);
    txt(c, 'level ' + l, x + 25, 216, { s: 11, c: C.dim });
    if (l === U.L && U.L <= 6) txt(c, '🤖', x + 25, y - 14, { s: 20 });
  }
  U.hist.forEach((ok, i) => dot(c, 30 + i * 14, 231, 5, ok ? C.green : C.red));
  const state = U.L > 6 ? 'Finished every level! 🎉' : U.wins === 0 && t - U.t0 > 6 ? 'Stuck: no wins, no learning' : '← last tries (green = success)';
  txt(c, state, 390, 231, { s: 11.5, b: 1, a: 'right', c: U.wins === 0 && t - U.t0 > 6 ? C.red : C.fg });
  txt(c, `start at level ${start}   ·   skill ${U.s.toFixed(1)}   ·   successes ${U.wins}`, 200, 16, { s: 12, c: C.gold });
};
CAP.n4_curriculum = 'Six levels, each taller than the last. Dots are recent tries, green for success. Five wins out of six moves the robot up a level. Its skill only grows when it succeeds.';

A.n4_autoCurric = (c, t, S) => {
  const step = .9, T = 30, k = Math.floor((t % T) / step);
  const sk = memo('ac', () => { const out = []; let s = 0; for (let i = 0; i < 40; i++) { out.push(s); s += .17; } return out; });
  const s = sk[k], rates = [1, 2, 3, 4, 5, 6].map(l => sig((s - l + 1.3) * 2.2));
  let pick = 0; rates.forEach((r, i) => { if (Math.abs(r - .5) < Math.abs(rates[pick] - .5)) pick = i; });
  line(c, 30, 205, 370, 205, C.line, 1.5);
  const y50 = 205 - .5 * 160; line(c, 30, y50, 370, y50, alpha(C.gold, .6), 1.5, [5, 5]); txt(c, '50%', 372, y50, { s: 10, c: C.gold, a: 'left' });
  rates.forEach((r, i) => {
    const x = 42 + i * 55; rr(c, x, 205 - r * 160, 40, r * 160, 6, i === pick ? C.gold : r > .8 ? alpha(C.green, .6) : r < .2 ? alpha(C.red, .5) : alpha(C.teal, .7));
    txt(c, Math.round(r * 100) + '%', x + 20, 197 - r * 160, { s: 11, b: 1 });
    txt(c, 'task ' + (i + 1), x + 20, 216, { s: 11, c: i === pick ? C.gold : C.dim, b: i === pick });
  });
  txt(c, `teacher picks task ${pick + 1}: success near 50%, not too easy, not too hard`, 200, 238, { s: 12, b: 1 });
  txt(c, 'bars = how often the agent succeeds at each task', 200, 16, { s: 12, c: C.dim });
};
CAP.n4_autoCurric = 'Each bar is how often the agent succeeds at a task, easy on the left. The teacher trains on the gold bar, the one nearest 50 percent. As skill grows, the gold pick moves right.';

// ───────── Meta-RL ─────────
const DOORS = 3, TRIALS = 6;
A.n4_metaAdapt = (c, t, S) => {
  if (!S.ma) S.ma = { r: mulberry(21), task: 1, k: 0, next: t + .8, meta: [], scr: [], Q: [0, 0, 0], tasks: 1, tapped: false };
  const M = S.ma;
  const newTask = () => { M.task = Math.floor(M.r() * DOORS); M.k = 0; M.meta = []; M.scr = []; M.Q = [0, 0, 0]; M.tasks++; M.next = t + .8; };
  while (S.taps.length) { S.taps.shift(); newTask(); }
  if (t >= M.next) {
    if (M.k < TRIALS) {
      const found = M.meta.find(m => m.win), tried = new Set(M.meta.map(m => m.a));
      const am = found ? found.a : [0, 1, 2].find(d => !tried.has(d));
      M.meta.push({ a: am, win: am === M.task });
      const as = M.k < 5 || M.r() < .6 ? Math.floor(M.r() * DOORS) : M.Q.indexOf(Math.max(...M.Q)); // too few tries to learn much: mostly guessing
      const w = as === M.task; M.Q[as] += .2 * ((w ? 1 : 0) - M.Q[as]); M.scr.push({ a: as, win: w });
      M.k++; M.next = t + 1.1;
    } else newTask();
  }
  const row = (y, lab, col, arr) => {
    txt(c, lab, 16, y, { s: 12, b: 1, a: 'left', c: col });
    for (let i = 0; i < TRIALS; i++) { const m = arr[i], x = 130 + i * 36;
      rr(c, x, y - 15, 32, 30, 7, m ? (m.win ? alpha(C.green, .55) : alpha(C.red, .3)) : C.panel, C.line, 1);
      if (m) txt(c, m.win ? '🎁' : '🚪' + (m.a + 1), x + 16, y, { s: m.win ? 15 : 11 }); }
    txt(c, arr.filter(m => m.win).length + ' 🎁', 384, y, { s: 12, b: 1, a: 'right' });
  };
  txt(c, `new task ${M.tasks}: the prize hides behind one of 3 doors`, 200, 18, { s: 12, c: C.gold });
  for (let d = 0; d < DOORS; d++) rr(c, 150 + d * 82, 38, 70, 26, 6, C.panel, C.line, 1), txt(c, '🚪 door ' + (d + 1), 185 + d * 82, 51, { s: 11.5 });
  row(110, 'meta-learned', C.teal, M.meta); row(165, 'from scratch', C.dim, M.scr);
  txt(c, 'trials →', 130, 82, { s: 10.5, c: C.dim, a: 'left' });
  txt(c, 'Meta-learned: try each door once, then stick with the prize.', 200, 210, { s: 12, b: 1, c: C.teal });
  txt(c, 'From scratch: relearns slowly on every new task.', 200, 232, { s: 12, c: C.dim });
};
TOUCH.add('n4_metaAdapt');
CAP.n4_metaAdapt = 'Each row is one agent trying doors on a fresh task; a gift box means it found the prize. The meta-learned agent checks each door once, then keeps winning. The other starts over.';

A.n4_mamlInit = (c, t) => {
  const P = 7, k = Math.floor(t / P), u = t % P, cx = 200, cy = 128;
  const opts = [0, 1, 2, 3, 4, 5].map(i => [cx + Math.cos(i * 1.05 + .3) * 80, cy + Math.sin(i * 1.05 + .3) * 62]);
  const nt = opts[k % 6], rnd = [40, 220];
  rr(c, 12, 30, 376, 196, 12, alpha(C.panel, .5), C.line, 1);
  opts.forEach((o, i) => txt(c, '★', o[0], o[1], { s: 16, c: alpha(C.gold, .45) }));
  txt(c, '★', nt[0], nt[1], { s: 26, c: C.pink });
  const steps = clamp(Math.floor((u - .8) / 1), 0, 3), frac = clamp((u - .8) / 1 - steps, 0, 1);
  const walk = (s0, rate) => { const pts = [s0]; let p = s0; for (let i = 0; i < 3; i++) { p = [lerp(p[0], nt[0], rate), lerp(p[1], nt[1], rate)]; pts.push(p); } return pts; };
  const draw = (pts, col) => { const shown = pts.slice(0, steps + 1); if (steps < 3) shown.push([lerp(pts[steps][0], pts[steps + 1][0], ease(frac)), lerp(pts[steps][1], pts[steps + 1][1], ease(frac))]); poly(c, shown, col, 2.5); dot(c, pts[0][0], pts[0][1], 6, col); dot(c, shown.at(-1)[0], shown.at(-1)[1], 5, col, '#fff'); };
  draw(walk([cx, cy], .62), C.teal); draw(walk(rnd, .3), C.dim);
  txt(c, 'meta start', cx, cy + 16, { s: 10.5, c: C.teal }); txt(c, 'random start', rnd[0] + 4, rnd[1] - 13, { s: 10.5, c: C.dim, a: 'left' });
  txt(c, `faint ★ training tasks  ·  pink ★ new task  ·  step ${steps} of 3`, 200, 16, { s: 11, c: C.dim });
  txt(c, 'A start in the middle of all tasks reaches a new one fast.', 200, 240, { s: 12, b: 1 });
};
CAP.n4_mamlInit = 'Each star is a task’s best setting. The teal start was learned to sit in the middle of the training tasks, so three small steps reach the new pink task. The grey random start is still far away.';
}
