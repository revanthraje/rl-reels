// New-topic simulations, part 1 (Rainbow, A2C/A3C, TRPO). Loaded after anim.js: uses its helpers, A and RLAnim.
'use strict';
{
const CAP = RLAnim.CAP;
const tick = (S, t) => { const dt = clamp(t - (S.lt ?? t), 0, .05); S.lt = t; return dt; };
const ctl = (S, k, d) => (S && S.ctl && S.ctl[k] != null && !isNaN(S.ctl[k])) ? +S.ctl[k] : d;
const ACOL = [C.pink, C.teal, C.gold, C.blue];

// ───────── Rainbow: dueling heads, Q = V + A ─────────
A.n1_dueling = (c, t, S) => {
  const auto = !(S && S.ctl && S.ctl.v != null);
  const ph = Math.floor(t / 6) % 2;                       // auto mode alternates two situations
  const V = auto ? lerp(ph ? 1 : 5, ph ? 5 : 1, ease(clamp((t % 6) / 1.2, 0, 1))) : ctl(S, 'v', 3);
  const Adv = [1.2, -.6, .3, -.9], acts = ['⬆️ gas', '⬅️ left', '➡️ right', '🛑 brake'];
  txt(c, 'Q  =  V  +  A', 200, 18, { s: 15, b: 1, c: C.gold });
  // V gauge
  rr(c, 18, 40, 70, 180, 12, C.panel, C.line);
  txt(c, 'V', 53, 56, { s: 15, b: 1, c: C.teal }); txt(c, 'how good', 53, 72, { s: 10, c: C.dim }); txt(c, 'is this state', 53, 84, { s: 10, c: C.dim });
  const vb = 205, vs = 15; bar(c, 38, vb, 30, Math.max(0, V) * vs, alpha(C.teal, .85));
  txt(c, V.toFixed(1), 53, vb - Math.max(0, V) * vs - 10, { s: 13, b: 1 });
  // A bars around a zero line
  rr(c, 100, 40, 128, 180, 12, C.panel, C.line);
  txt(c, 'A', 164, 56, { s: 15, b: 1, c: C.purple }); txt(c, 'extra for each action', 164, 72, { s: 10, c: C.dim });
  const z = 140; line(c, 108, z, 220, z, C.line, 1.5, [3, 3]);
  Adv.forEach((a, i) => { const x = 112 + i * 27; c.fillStyle = alpha(a >= 0 ? C.purple : C.red, .8); c.fillRect(x, a >= 0 ? z - a * 30 : z, 20, Math.abs(a) * 30); txt(c, (a > 0 ? '+' : '') + a.toFixed(1), x + 10, a >= 0 ? z - a * 30 - 8 : z - a * 30 + 9, { s: 9, c: C.fg }); txt(c, acts[i].split(' ')[0], x + 10, 205, { s: 12 }); });
  // Q bars
  rr(c, 240, 40, 145, 180, 12, C.panel, C.gold, 2);
  txt(c, 'Q for each action', 312, 56, { s: 12, b: 1, c: C.gold });
  const qb = 195, qs = 15, best = Adv.indexOf(Math.max(...Adv));
  Adv.forEach((a, i) => { const q = V + a, x = 250 + i * 33, h = q * qs;
    c.fillStyle = alpha(i === best ? C.gold : C.blue, .85); c.fillRect(x, h >= 0 ? qb - h : qb, 26, Math.abs(h));
    txt(c, q.toFixed(1), x + 13, (h >= 0 ? qb - h : qb - h) - 8 * Math.sign(h || 1), { s: 10, b: i === best });
    txt(c, acts[i].split(' ')[0], x + 13, 208, { s: 12 }); });
  line(c, 246, qb, 380, qb, C.line, 1.5);
  txt(c, auto ? (ph ? 'open road: the state is great, V is high' : 'about to crash: the state is bad, V is low') : 'V moves every Q together. The best action (gold) stays the same.', 200, 240, { s: 12, c: C.fg });
};
CAP.n1_dueling = 'Teal is V, how good the situation is. Purple and red bars are A, how much better or worse each action is. Each gold or blue Q bar is simply V plus that action’s A.';

// ───────── Rainbow: distributional returns ─────────
const DIST = memo('n1dist', () => { const r = mulberry(51), safe = [], risky = [];
  for (let i = 0; i < 40; i++) { safe.push(clamp(Math.round(5 + gauss(r) * 1.1), 0, 10)); risky.push(r() < .5 ? Math.round(r()) : 9 + Math.round(r())); }
  return { safe, risky }; });
A.n1_distrib = (c, t) => {
  const T = 18, tt = t % T, n = Math.min(40, Math.floor(Math.max(0, tt - 1) / .38));
  const panel = (x0, name, arr, col) => {
    rr(c, x0, 34, 184, 186, 12, C.panel, C.line);
    txt(c, name, x0 + 92, 50, { s: 13, b: 1, c: col });
    const h = Array(11).fill(0); let sum = 0; for (let i = 0; i < n; i++) { h[arr[i]]++; sum += arr[i]; }
    const yb = 196, bw = 14;
    for (let k = 0; k <= 10; k++) { const x = x0 + 12 + k * 15; c.fillStyle = alpha(col, .8); c.fillRect(x, yb - h[k] * 6.5, bw, h[k] * 6.5); if (k % 5 === 0) txt(c, String(k), x + bw / 2, 208, { s: 10, c: C.dim }); }
    line(c, x0 + 10, yb, x0 + 176, yb, C.line, 1.5);
    if (n) { const m = sum / n, mx = x0 + 12 + m * 15 + bw / 2; line(c, mx, 66, mx, yb, C.gold, 2, [4, 3]); txt(c, 'average ' + m.toFixed(1), mx, 74, { s: 11, b: 1, c: C.gold }); }
    if (n && n < 40 && (tt % .38) < .25) { const v = arr[n - 1]; dot(c, x0 + 12 + v * 15 + bw / 2, 92, 5, C.fg); }
  };
  panel(12, '🛣️ safe route', DIST.safe, C.teal); panel(204, '🎢 risky route', DIST.risky, C.pink);
  txt(c, 'outcome of each try (return) →', 200, 226, { s: 11, c: C.dim });
  txt(c, n < 40 ? `tries so far: ${n}` : 'same average, totally different shapes', 200, 18, { s: 13, b: 1 });
  txt(c, 'one number (the average) hides this; a distribution keeps it', 200, 246, { s: 11, c: C.fg });
};
CAP.n1_distrib = 'Each try drops a white dot into a bin by its return. The safe route piles up around 5; the risky one gives 0 or 10. The gold dashed averages are almost equal, but the shapes are not.';

// ───────── Rainbow: the six ingredients ─────────
A.n1_rainbowBars = (c, t) => {
  const items = [['Double Q-learning', C.pink], ['Prioritised replay', C.gold], ['Dueling heads', C.purple], ['Multi-step returns', C.teal], ['Distributional', C.blue], ['Noisy nets', C.green]];
  const T = 16, tt = t % T, k = clamp(Math.floor((tt - 1.2) / 1.7) + 1, 0, 6);
  rr(c, 14, 30, 176, 200, 12, C.panel, C.line);
  txt(c, 'DQN  +', 102, 48, { s: 13, b: 1 });
  items.forEach(([n, col], i) => { const on = i < k, y = 72 + i * 25;
    rr(c, 26, y - 10, 152, 21, 8, on ? alpha(col, .3) : C.bg, on ? col : C.line, on ? 2 : 1);
    txt(c, (on ? '✓ ' : '') + n, 102, y + .5, { s: 11.5, b: on, c: on ? C.fg : C.dim }); });
  // stylised score curve: each ingredient lifts it a bit
  rr(c, 202, 30, 184, 200, 12, C.panel, C.line);
  line(c, 218, 208, 376, 208, C.line, 1.5); line(c, 218, 208, 218, 46, C.line, 1.5);
  txt(c, 'training time →', 300, 220, { s: 10, c: C.dim }); txt(c, 'score', 232, 52, { s: 10, c: C.dim, a: 'left' });
  const lvl = j => 196 - 22 * j - 8 * Math.sqrt(j);
  const pts = []; for (let x = 0; x <= 150; x += 5) { const g = 1 - Math.exp(-x / 45); pts.push([220 + x, 208 - (208 - lvl(0)) * g]); }
  poly(c, pts, C.dim, 2, [4, 3]);
  if (k) { const p2 = []; for (let x = 0; x <= 150; x += 5) { const g = 1 - Math.exp(-x / 45); p2.push([220 + x, 208 - (208 - lvl(k)) * g]); } poly(c, p2, k === 6 ? C.gold : items[k - 1][1], 3); }
  txt(c, 'plain DQN', 372, lvl(0) + 12, { s: 10, c: C.dim, a: 'right' });
  if (k === 6) txt(c, '🌈 Rainbow', 372, lvl(6) - 10, { s: 12, b: 1, c: C.gold, a: 'right' });
  txt(c, k < 6 ? 'adding fixes one at a time…' : 'all six together: Rainbow (2018)', 200, 16, { s: 13, b: 1 });
  txt(c, 'shapes are illustrative, not real data', 200, 246, { s: 10, c: C.dim });
};
CAP.n1_rainbowBars = 'On the left, six known fixes switch on one by one. On the right, the dashed line is plain DQN and the coloured curve climbs as fixes are added. The curve shapes are illustrative.';

// ───────── A2C: parallel workers ─────────
A.n1_parallel = (c, t, S) => {
  const k = Math.round(clamp(ctl(S, 'k', 4), 1, 8));
  const cols = [C.pink, C.teal, C.gold, C.blue, C.purple, C.green, C.red, '#f0abfc'];
  const top = 30, H = 196, lh = H / k;
  txt(c, `${k} worker${k > 1 ? 's' : ''}, each with its own game`, 108, 16, { s: 11, b: 1 });
  for (let i = 0; i < k; i++) {
    const y = top + i * lh, cy = y + lh / 2;
    rr(c, 10, y + 2, 196, lh - 4, Math.min(8, lh / 3), alpha(cols[i], .12), alpha(cols[i], .6), 1);
    const ph = (t * .22 + i * .37) % 1; dot(c, 26 + ph * 150, cy, Math.min(6, lh / 4), cols[i]);
    if (lh > 18) txt(c, '🎮 ' + (i + 1), 196, cy, { s: Math.min(11, lh / 2.2), a: 'right', c: C.fg });
  }
  // experience tiles flowing to the batch: one tile every 0.5 s per round-robin worker
  const rate = .5, idx = Math.floor(t / rate), u = (t % rate) / rate;
  const src = j => j % k;
  const flyW = src(idx); const fy = top + flyW * lh + lh / 2;
  rr(c, lerp(206, 262, ease(u)) - 6, lerp(fy, 128, ease(u)) - 6, 12, 12, 3, cols[flyW]);
  // batch box: last 8 tiles (from 1 worker, consecutive moments are near copies)
  rr(c, 270, 30, 118, 100, 12, C.panel, C.line);
  txt(c, 'training batch', 329, 44, { s: 11, b: 1 });
  for (let j = 0; j < 8; j++) { const id = idx - 1 - j; if (id < 0) continue; const w = src(id);
    rr(c, 280 + (j % 4) * 26, 58 + Math.floor(j / 4) * 32, 22, 26, 4, cols[w]); txt(c, k === 1 ? 'm' + (id % 99) : '', 291 + (j % 4) * 26, 71 + Math.floor(j / 4) * 32, { s: 8, c: C.bg }); }
  rr(c, 270, 142, 118, 62, 12, C.panel, C.gold, 2);
  txt(c, '🧠 shared', 329, 160, { s: 12, b: 1 }); txt(c, 'actor + critic', 329, 176, { s: 11, c: C.dim });
  const pulse = (t % 2) < .3; if (pulse) rr(c, 270, 142, 118, 62, 12, null, C.gold, 4);
  txt(c, `${k * 50} steps / second`, 329, 194, { s: 10, c: C.gold });
  txt(c, k === 1 ? 'one worker: the batch is all one colour, near-identical moments' : 'many workers: the batch mixes different games and moments', 200, 244, { s: 11.5, c: C.fg });
};
CAP.n1_parallel = 'Each coloured lane is one worker playing its own copy of the game. Its experiences fly into the training batch on the right. More workers means more data and a more varied, mixed-colour batch.';

// ───────── A2C vs A3C timelines ─────────
A.n1_asyncSync = (c, t) => {
  const T = 12, tt = t % T, X = s => 70 + s / T * 310;
  const cols = [C.pink, C.teal, C.gold];
  // A3C
  txt(c, 'A3C: send whenever ready (asynchronous)', 14, 16, { s: 12, b: 1, a: 'left', c: C.pink });
  const sends = [[2, 5.6, 9.1], [3.1, 7.3, 10.6], [4.4, 6.4, 11.4]];
  const all = sends.flatMap((ts, w) => ts.map(s => [s, w])).sort((a, b) => a[0] - b[0]);
  let ver = 1; const lastV = [1, 1, 1];
  for (const [s, w] of all) if (s <= tt) { ver++; lastV[w] = ver; }
  for (let w = 0; w < 3; w++) { const y = 36 + w * 22; txt(c, 'worker ' + (w + 1), 14, y, { s: 10, a: 'left', c: cols[w] }); line(c, X(0), y, X(T), y, alpha(cols[w], .35), 6);
    sends[w].forEach(s => { if (s <= tt) { dot(c, X(s), y, 5, cols[w]); if (tt - s < .7) arrow(c, X(s), y - 6, X(s), 27, cols[w], 2, 6); } }); }
  rr(c, 300, 92, 88, 20, 6, C.panel, C.pink, 1.5); txt(c, 'shared model v' + ver, 344, 102, { s: 10, b: 1 });
  const stale = lastV.map(v => ver - v); const sw = stale.indexOf(Math.max(...stale));
  txt(c, `worker ${sw + 1} is still acting with v${lastV[sw]} (${Math.max(...stale)} behind)`, 14, 102, { s: 10, a: 'left', c: C.dim });
  // A2C
  txt(c, 'A2C: wait for everyone, update once (synchronous)', 14, 130, { s: 12, b: 1, a: 'left', c: C.teal });
  const work = [[2.6, 3.4, 2.9], [2.2, 3.7, 3.1], [3.3, 2.8, 3.6]]; const B = [4, 8, 12];
  let v2 = 1;
  for (let w = 0; w < 3; w++) { const y = 152 + w * 22; txt(c, 'worker ' + (w + 1), 14, y, { s: 10, a: 'left', c: cols[w] });
    B.forEach((b, j) => { const s0 = b - 4, e = s0 + work[j][w];
      if (tt > s0) { line(c, X(s0), y, X(Math.min(tt, e)), y, cols[w], 6); }
      if (tt > e) { line(c, X(e), y, X(Math.min(tt, b)), y, alpha(C.dim, .5), 2, [3, 3]); } }); }
  B.forEach(b => { if (tt >= b - .02 || b === T && tt > T - .3) { line(c, X(b) - 1, 142, X(b) - 1, 206, C.teal, 2); } });
  v2 = 1 + B.filter(b => tt >= b - .3).length;
  B.forEach(b => { if (tt > b - .9 && tt < b) { txt(c, '⏳ waiting', X(b) - 22, 216, { s: 9, c: C.dim }); } });
  rr(c, 300, 224, 88, 20, 6, C.panel, C.teal, 1.5); txt(c, 'shared model v' + v2, 344, 234, { s: 10, b: 1 });
  txt(c, 'dashed = idle, waiting at the barrier', 14, 234, { s: 10, a: 'left', c: C.dim });
  line(c, X(tt), 26, X(tt), 208, alpha(C.fg, .25), 1);
};
CAP.n1_asyncSync = 'Top: in A3C each worker sends its update the moment it finishes, so others keep acting with an older model. Bottom: in A2C workers wait at the line, then one combined update is made.';

// ───────── TRPO: the trust region on a performance landscape ─────────
const J = th => 6 * Math.exp(-(((th - 5) / 2) ** 2)) - (th > 5.7 ? 12 * (1 - Math.exp(-(th - 5.7) * 5)) : 0);
const hill = th => 6 * Math.exp(-(((th - 5) / 2) ** 2));
A.n1_trustRegion = (c, t, S) => {
  const d = clamp(ctl(S, 'd', .8), .2, 3);
  const st = S.tr && S.tr.d === d ? S.tr : (S.tr = { d, th: 1, from: 1, t0: t, n: 0, crash: 0 });
  const P = 2.4;
  if (t - st.t0 > P) {
    if (st.crash || st.n >= 7) { st.th = st.from = 1; st.n = 0; st.crash = 0; }
    else { const g = (hill(st.th + .01) - hill(st.th - .01)) / .02; st.from = st.th; st.th = clamp(st.th + clamp(20 * g, -d, d), 0, 9.8); st.n++; if (J(st.th) < 0) st.crash = 1; }
    st.t0 = t;
  }
  const u = ease(clamp((t - st.t0) / .9, 0, 1)), th = lerp(st.from, st.th, u);
  const X = v => 26 + v * 36, Y = v => clamp(200 - v * 24, 30, 226);
  // landscape
  const pts = []; for (let v = 0; v <= 10; v += .05) pts.push([X(v), Y(J(v))]); poly(c, pts, C.teal, 3);
  line(c, 20, 200, 390, 200, C.line, 1, [2, 4]);
  txt(c, 'policy settings →', 384, 238, { s: 10, c: C.dim, a: 'right' }); txt(c, 'performance', 14, 22, { s: 10, c: C.dim, a: 'left' });
  txt(c, '⚠️ cliff', X(6.6), 110, { s: 11, b: 1, c: C.red });
  // trust region around the start of this step
  const c0 = st.from; c.fillStyle = alpha(C.gold, .14); c.fillRect(X(c0 - d), 30, 36 * 2 * d, 196);
  line(c, X(c0 - d), 30, X(c0 - d), 226, alpha(C.gold, .6), 1.5, [4, 3]); line(c, X(c0 + d), 30, X(c0 + d), 226, alpha(C.gold, .6), 1.5, [4, 3]);
  // local guess = tangent line at the start point
  const g = (hill(c0 + .01) - hill(c0 - .01)) / .02; line(c, X(c0 - 1.6), Y(hill(c0) - 1.6 * g), X(c0 + 1.6), Y(hill(c0) + 1.6 * g), C.purple, 2, [5, 3]);
  dot(c, X(th), Y(J(th)), 8, st.crash && u > .9 ? C.red : C.pink, '#fff');
  txt(c, `trust region ±${d.toFixed(1)}  ·  step ${st.n}`, 200, 16, { s: 12, b: 1, c: C.gold });
  if (st.crash && u > .9) txt(c, '💥 the step jumped off the cliff: performance collapsed', 200, 252, { s: 12, b: 1, c: C.red });
  else txt(c, 'purple dashes = the local guess, only reliable inside gold', 200, 252, { s: 11, c: C.fg });
};
CAP.n1_trustRegion = 'The teal curve is how well each policy really does; the pink dot is the current policy. The purple dashes are the local guess. Each step may only move inside the gold band, the trust region.';

// ───────── TRPO: KL meter between old and new policy ─────────
const soft = l => { const m = Math.max(...l), e = l.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };
A.n1_klMeter = (c, t, S) => {
  const lim = clamp(ctl(S, 'kl', .05), .005, .5);
  const st = S.kl && S.kl.lim === lim ? S.kl : (S.kl = { lim, old: [0, 0, 0, 0], cur: [0, 0, 0, 0], hold: 0, n: 1 });
  const dt = tick(S, t);
  const po = soft(st.old), pn = soft(st.cur);
  const kl = po.reduce((a, p, i) => a + p * Math.log(p / pn[i]), 0);
  if (st.hold > 0) { st.hold -= dt; if (st.hold <= 0) { st.old = st.cur.slice(); st.n++; if (soft(st.old)[2] > .9) { st.old = [0, 0, 0, 0]; st.cur = [0, 0, 0, 0]; st.n = 1; } } }
  else if (kl < lim) st.cur[2] += dt * .7; else st.hold = 1.6;
  const names = ['⬆️', '⬇️', '➡️', '⬅️'];
  rr(c, 14, 30, 232, 200, 12, C.panel, C.line);
  txt(c, 'action probabilities', 130, 46, { s: 12, b: 1 });
  for (let i = 0; i < 4; i++) { const x = 34 + i * 52, yb = 196;
    c.fillStyle = alpha(C.teal, .8); c.fillRect(x, yb - pn[i] * 118, 30, pn[i] * 118);
    c.strokeStyle = C.fg; c.lineWidth = 2; c.setLineDash([4, 3]); c.strokeRect(x - 3, yb - po[i] * 118, 36, po[i] * 118); c.setLineDash([]);
    txt(c, names[i], x + 15, 212, { s: 14 }); txt(c, Math.round(pn[i] * 100) + '%', x + 15, yb - pn[i] * 118 - 9, { s: 10, b: 1 }); }
  txt(c, 'dashed = old policy · teal = new', 130, 62, { s: 10, c: C.dim });
  // KL meter
  rr(c, 262, 30, 124, 200, 12, C.panel, C.line);
  txt(c, 'KL distance', 324, 46, { s: 12, b: 1 }); txt(c, 'old → new', 324, 61, { s: 10, c: C.dim });
  const mx = Math.max(.12, lim * 1.6), H = 140, yb = 206, f = v => clamp(v / mx, 0, 1) * H;
  rr(c, 306, yb - H, 36, H, 6, C.bg, C.line, 1);
  c.fillStyle = kl >= lim * .98 ? C.red : C.teal; c.fillRect(308, yb - f(kl), 32, f(kl));
  line(c, 296, yb - f(lim), 352, yb - f(lim), C.gold, 3); txt(c, 'limit ' + lim.toFixed(2), 324, yb - f(lim) - 9, { s: 10, b: 1, c: C.gold });
  txt(c, kl.toFixed(3), 324, 220, { s: 11, b: 1 });
  txt(c, st.hold > 0 ? `🛑 limit hit: update ${st.n} done, new becomes old` : `update ${st.n}: pushing ➡️ up, KL growing…`, 200, 16, { s: 12, b: 1, c: st.hold > 0 ? C.gold : C.fg });
  txt(c, 'each update may only change behaviour this much', 200, 246, { s: 11, c: C.fg });
};
CAP.n1_klMeter = 'Teal bars are the new policy, dashed boxes the old one. The update keeps raising ➡️ until the KL meter hits the gold limit. Then it stops, and the new policy becomes the starting point.';
}
