// Part 3 animations (Double DQN → AlphaZero). Loaded after anim.js: uses its helpers, A and RLAnim.
// Interactive ones read S.ctl (sliders) or S.taps / S.ptr (canvas taps; registered in RLAnim.TOUCH).
'use strict';
(() => {
const CAP = RLAnim.CAP, TOUCH = RLAnim.TOUCH;
const ctl = (S, k, d) => (S.ctl && S.ctl[k] != null && !isNaN(S.ctl[k])) ? +S.ctl[k] : d;
const takeTaps = S => { const out = []; while (S.taps && S.taps.length) out.push(S.taps.shift()); return out; };
// run a discrete simulation: calls fn once per `period` seconds of t (catches up at most 40 steps per frame)
function ticks(S, key, t, period, fn) {
  const target = Math.floor(t / period); S[key] ??= 0;
  if (target - S[key] > 40) S[key] = target - 40;
  while (S[key] < target) { S[key]++; fn(S[key]); }
}
const softmax = (v, tau = 1) => { const m = Math.max(...v), e = v.map(x => Math.exp((x - m) / tau)), s = e.reduce((a, b) => a + b); return e.map(x => x / s); };
const entropy2 = p => -p.reduce((a, q) => a + (q > 1e-9 ? q * Math.log2(q) : 0), 0);
function meter(c, x, y, w, frac, col, label) {
  rr(c, x, y, w, 9, 4.5, C.panel); rr(c, x, y, Math.max(4, w * clamp(frac, 0, 1)), 9, 4.5, col);
  if (label) txt(c, label, x, y - 9, { s: 11, a: 'left', c: C.dim });
}

// ───────── Double DQN ─────────
A.p3_maxNoise = (c, t, S) => {
  const sig = ctl(S, 'sig', 1), n = Math.round(ctl(S, 'n', 6)), key = sig + '|' + n;
  if (S.mk !== key) { S.mk = key; S.hist = []; S.k0 = Math.floor(t / 1.2); S.mn = undefined; }
  const round = Math.floor(t / 1.2), r = mulberry(round * 7919 + 13), a = [], b = [];
  for (let i = 0; i < n; i++) { a.push(sig * gauss(r)); b.push(sig * gauss(r)); }
  const ia = a.indexOf(Math.max(...a));
  if (S.mn !== round) { S.mn = round; S.hist.push([a[ia], b[ia]]); if (S.hist.length > 60) S.hist.shift(); }
  const avgMax = S.hist.reduce((s, h) => s + h[0], 0) / S.hist.length, avgDbl = S.hist.reduce((s, h) => s + h[1], 0) / S.hist.length;
  const y0 = 140, sc = Math.min(38, 88 / (2.4 * Math.max(sig, .25))), x0 = 22, W = 250, bw = Math.min(30, W / n - 6);
  rr(c, 10, 28, 270, 196, 12, C.panel);
  txt(c, `${n} actions, all truly worth 0 · guesses = 0 + noise`, 145, 16, { s: 11, b: 1 });
  line(c, 16, y0, 274, y0, C.fg, 1.5, [5, 4]); txt(c, 'true 0', 18, y0 - 8, { s: 10, c: C.dim, a: 'left' });
  for (let i = 0; i < n; i++) {
    const x = x0 + i * (W / n) + (W / n - bw) / 2, h = clamp(a[i] * sc, -62, 80);
    bar(c, x, y0, bw, h, i === ia ? C.gold : alpha(C.teal, .55));
    if (i === ia) txt(c, 'max ' + (a[i] >= 0 ? '+' : '') + a[i].toFixed(1), x + bw / 2, y0 - h - 10 * Math.sign(h || 1), { s: 11, b: 1, c: C.gold });
  }
  txt(c, 'gold = the max guess, used as the learning target', 145, 214, { s: 10.5, c: C.dim });
  // right: running averages
  rr(c, 288, 28, 104, 196, 12, C.panel); txt(c, `avg over ${S.hist.length} tries`, 340, 42, { s: 10.5, c: C.dim });
  const my = 140, ms = 30; line(c, 296, my, 384, my, C.fg, 1, [4, 4]);
  bar(c, 304, my, 30, clamp(avgMax * ms, -80, 80), C.pink); bar(c, 346, my, 30, clamp(avgDbl * ms, -80, 80), C.teal);
  txt(c, (avgMax >= 0 ? '+' : '') + avgMax.toFixed(2), 319, my - clamp(avgMax * ms, -80, 80) - 9, { s: 11, b: 1, c: C.pink });
  txt(c, (avgDbl >= 0 ? '+' : '') + avgDbl.toFixed(2), 361, my - clamp(avgDbl * ms, -80, 80) - 9 * Math.sign(avgDbl || 1), { s: 11, b: 1, c: C.teal });
  txt(c, 'max', 319, 196, { s: 11, b: 1, c: C.pink }); txt(c, 'double', 361, 196, { s: 11, b: 1, c: C.teal });
  txt(c, 'pink stays above 0: optimism from noise alone', 200, 242, { s: 12, c: C.gold });
};
CAP.p3_maxNoise = 'Every action is truly worth zero, but each guess has random noise. The gold bar is the biggest guess. On the right, pink averages the max over many tries and teal the double estimate.';

A.p3_snowball = (c, t) => {
  const K = 9, sim = memo('p3snow', () => { // chain of 5 states, all true values 0; average of 200 runs
    const r = mulberry(21), runs = 200, out = [];
    let Vm = Array.from({ length: runs }, () => new Float32Array(6)), Vd = Array.from({ length: runs }, () => new Float32Array(6));
    out.push({ m: new Float32Array(6), d: new Float32Array(6) });
    for (let k = 0; k < K; k++) {
      const nm = [], nd = [];
      for (let j = 0; j < runs; j++) {
        const m = new Float32Array(6), d = new Float32Array(6);
        for (let i = 0; i < 5; i++) {
          const qa = [0, 1, 2, 3].map(() => Vm[j][i + 1] + .6 * gauss(r)); m[i] = Math.max(...qa);
          d[i] = Vd[j][i + 1] + .6 * gauss(r); // choose with net A, score with independent net B: no upward bias
        }
        nm.push(m); nd.push(d);
      }
      Vm = nm; Vd = nd;
      const am = new Float32Array(6), ad = new Float32Array(6);
      for (let i = 0; i < 6; i++) { am[i] = Vm.reduce((s, v) => s + v[i], 0) / runs; ad[i] = Vd.reduce((s, v) => s + v[i], 0) / runs; }
      out.push({ m: am, d: ad });
    }
    return out;
  });
  const dt = 1.6, k = Math.floor(t / dt) % (K + 2), u = (t % dt) / dt, cur = sim[Math.min(k, K)], prev = sim[Math.max(0, Math.min(k, K) - 1)];
  const rows = [[64, 'max target (DQN)', 'm', C.pink], [168, 'double target', 'd', C.teal]];
  txt(c, `update round ${Math.min(k, K)}  ·  all true values are 0`, 200, 14, { s: 12, b: 1, c: C.gold });
  for (const [y, lab, f, col] of rows) {
    txt(c, lab, 12, y - 40, { s: 11, b: 1, c: col, a: 'left' });
    for (let i = 0; i < 6; i++) {
      const x = 22 + i * 62, v = lerp(prev[f][i], cur[f][i], ease(clamp(u * 1.6, 0, 1)));
      rr(c, x, y - 28, 50, 56, 10, C.panel, i === 5 ? C.green : C.line, 1.5);
      if (i < 5) { bar(c, x + 12, y + 22, 26, clamp(v * 22, -40, 44), col); txt(c, (v >= 0 ? '+' : '') + v.toFixed(2), x + 25, y + 36, { s: 10.5, b: 1, c: col }); }
      else txt(c, 'end', x + 25, y, { s: 11, c: C.green });
      if (i < 5) arrow(c, x + 50, y, x + 61, y, C.line, 1.5, 5);
    }
    if (k > 0 && k <= K) { const px = 22 + (5 - Math.min(5, u * 6)) * 62 + 25; dot(c, px, y - 34, 4, col); }
  }
  txt(c, 'each box learns from the box to its right — errors flow backwards', 200, 246, { s: 11, c: C.dim });
};
CAP.p3_snowball = 'Five states in a row, each truly worth zero. Each round, every box copies its target from the box to its right. Pink max targets creep upward and spread backwards; teal double targets stay near zero.';

// ───────── Policy gradient ─────────
const ACTS4 = ['↑ up', '→ right', '↓ down', '← left'];
A.p3_dials = (c, t, S) => {
  S.th ||= [0, 0, 0, 0]; S.shown ||= [.25, .25, .25, .25]; S.lastTap ??= -99;
  for (const q of takeTaps(S)) {
    if (q.x > 320 && q.y < 34) { S.th = [0, 0, 0, 0]; S.lastTap = t; continue; }
    const i = clamp(Math.floor((q.x - 20) / 90), 0, 3); S.th[i] = Math.min(4, S.th[i] + .6); S.lastTap = t; S.flash = [i, t];
  }
  if (t - S.lastTap > 6) { // idle demo: nudge "right" a little every 2 s, reset when it dominates
    ticks(S, 'demo', t, 2, () => { if (S.th[1] > 2.6) S.th = [0, 0, 0, 0]; else { S.th[1] += .55; S.flash = [1, t]; } });
  }
  const p = softmax(S.th); S.shown = S.shown.map((v, i) => lerp(v, p[i], .12));
  txt(c, 'tap a column to turn its dial up', 160, 15, { s: 12, b: 1, c: C.gold });
  rr(c, 326, 4, 66, 24, 12, C.panel, C.line); txt(c, '↺ reset', 359, 16, { s: 11, b: 1 });
  for (let i = 0; i < 4; i++) {
    const x = 20 + i * 90, cx = x + 40, fl = S.flash && S.flash[0] === i && t - S.flash[1] < .6;
    rr(c, x, 32, 80, 196, 12, C.panel, fl ? C.gold : C.line, fl ? 3 : 1.5);
    // dial
    c.beginPath(); c.arc(cx, 72, 22, Math.PI * .75, Math.PI * 2.25); c.strokeStyle = C.line; c.lineWidth = 5; c.stroke();
    c.beginPath(); c.arc(cx, 72, 22, Math.PI * .75, Math.PI * .75 + (S.th[i] / 4) * Math.PI * 1.5 + .001); c.strokeStyle = C.purple; c.lineWidth = 5; c.stroke();
    const a2 = Math.PI * .75 + (S.th[i] / 4) * Math.PI * 1.5; line(c, cx, 72, cx + 16 * Math.cos(a2), 72 + 16 * Math.sin(a2), C.fg, 3); dot(c, cx, 72, 4, C.fg);
    txt(c, 'θ ' + S.th[i].toFixed(1), cx, 106, { s: 11, c: C.purple, b: 1 });
    // probability bar
    bar(c, x + 18, 206, 44, S.shown[i] * 90, i === 1 ? C.teal : alpha(C.blue, .8));
    txt(c, Math.round(S.shown[i] * 100) + '%', cx, 198 - S.shown[i] * 90, { s: 12, b: 1 });
    txt(c, ACTS4[i], cx, 218, { s: 11 });
    if (fl) popup(c, 'turn ↑', cx, 46, (t - S.flash[1]) / .6, C.gold);
  }
  txt(c, 'dials (purple) set the probabilities (bars) — they always add to 100%', 200, 246, { s: 11, c: C.dim });
};
TOUCH.add('p3_dials');
CAP.p3_dials = 'Each column is one action. The purple dial is a policy parameter, and the bar below is that action’s probability. Turning one dial up makes its action likelier and the others a bit less likely.';

A.p3_pgLR = (c, t, S) => {
  const lr = ctl(S, 'lr', .3);
  S.th ??= 0; S.hist ||= [.5]; S.base ??= .5; S.ev ||= null;
  ticks(S, 'k', t, .9, k => {
    const r = mulberry(k * 31 + 7), p = 1 / (1 + Math.exp(-S.th)), right = r() < p, rew = (right ? 1 : .3) + .7 * gauss(r);
    const adv = rew - S.base; S.base = lerp(S.base, rew, .1);
    S.th = clamp(S.th + lr * adv * ((right ? 1 : 0) - p), -5, 5);
    S.hist.push(1 / (1 + Math.exp(-S.th))); if (S.hist.length > 70) S.hist.shift();
    S.ev = { right, rew, adv, at: k };
  });
  const p = 1 / (1 + Math.exp(-S.th));
  // top: probability split
  txt(c, 'two actions: left pays about 0.3, right about 1.0 (noisy)', 200, 14, { s: 11.5, b: 1, c: C.gold });
  rr(c, 20, 28, 360, 30, 10, C.panel); rr(c, 20, 28, 360 * (1 - p), 30, 10, alpha(C.blue, .8)); rr(c, 20 + 360 * (1 - p), 28, 360 * p, 30, 10, C.teal);
  txt(c, `← left ${Math.round((1 - p) * 100)}%`, 30, 43, { s: 12, b: 1, a: 'left' }); txt(c, `right ${Math.round(p * 100)}% →`, 370, 43, { s: 12, b: 1, a: 'right' });
  // chart of p(right)
  const X0 = 40, Y0 = 200, Wc = 340, Hc = 110;
  line(c, X0, Y0, X0 + Wc, Y0, C.line); line(c, X0, Y0 - Hc, X0, Y0, C.line);
  txt(c, '100%', X0 - 4, Y0 - Hc, { s: 10, a: 'right', c: C.dim }); txt(c, '0%', X0 - 4, Y0, { s: 10, a: 'right', c: C.dim });
  line(c, X0, Y0 - Hc / 2, X0 + Wc, Y0 - Hc / 2, alpha(C.dim, .4), 1, [3, 3]);
  poly(c, S.hist.map((v, i) => [X0 + i * Wc / 70, Y0 - v * Hc]), C.teal, 2.5);
  txt(c, 'chance of picking right, over time', X0 + 6, Y0 - Hc - 8, { s: 10.5, c: C.teal, a: 'left' });
  if (S.ev) {
    const e = S.ev, col = e.adv > 0 ? C.green : C.red;
    txt(c, `picked ${e.right ? 'right' : 'left'}, reward ${e.rew.toFixed(1)} → ${e.adv > 0 ? 'nudge it up' : 'nudge it down'}`, 200, 222, { s: 12, b: 1, c: col });
  }
  txt(c, lr > .9 ? 'big steps: fast but jumpy' : lr < .12 ? 'tiny steps: steady but slow' : 'each try nudges the odds a little', 200, 246, { s: 11.5, c: C.dim });
};
CAP.p3_pgLR = 'The top bar splits the policy between left and right. Every second it tries one, sees a noisy reward, and nudges that action’s odds up or down. The teal line tracks the chance of picking right.';

// ───────── REINFORCE ─────────
A.p3_creditTiles = (c, t) => {
  const T = 9, e = Math.floor(t / T) % 2, tt = t % T, G = e ? -5 : 10, col = G > 0 ? C.green : C.red;
  const moves = e ? ['⬅', '⬆', '⬅'] : ['➡', '➡', '⬆'];
  txt(c, `episode ${e + 1}: three moves, then the final reward`, 200, 16, { s: 12, b: 1, c: C.gold });
  moves.forEach((m, i) => {
    const x = 40 + i * 90, on = tt > .8 + i * 1;
    rr(c, x, 50, 70, 70, 12, C.panel, on ? C.fg : C.line, on ? 2 : 1); txt(c, on ? m : '', x + 35, 85, { s: 26 });
    txt(c, 'move ' + (i + 1), x + 35, 132, { s: 11, c: C.dim });
    if (i < 2) arrow(c, x + 72, 85, x + 88, 85, C.line, 1.5, 6);
    if (tt > 5.2) { const u = clamp((tt - 5.2) / 1.2, 0, 1); txt(c, 'credit ' + (G > 0 ? '+' : '') + G, x + 35, 152, { s: 12, b: 1, c: col, al: u });
      const L = 26 * ease(u); arrow(c, x + 35, 196, x + 35, 196 - Math.sign(G) * L, col, 4, 9); }
  });
  if (tt > 3.8) { const u = clamp((tt - 3.8) / .8, 0, 1); rr(c, 310, 50, 76, 70, 12, alpha(col, .25 * u), col, 2.5); txt(c, (G > 0 ? '+' : '') + G, 348, 80, { s: 24, b: 1, c: col, al: u }); txt(c, 'final', 348, 106, { s: 10, c: C.dim, al: u }); }
  if (tt > 5.2) txt(c, G > 0 ? 'every move in it becomes a bit MORE likely' : 'every move in it becomes a bit LESS likely', 200, 226, { s: 12.5, b: 1, c: col });
  txt(c, 'REINFORCE gives each move the whole return that followed', 200, 248, { s: 11, c: C.dim });
};
CAP.p3_creditTiles = 'Three moves play out, then the episode’s final reward appears on the right. Every move gets that same total as its credit. A plus 10 pushes all three up, a minus 5 pushes all three down.';

A.p3_noisyG = (c, t, S) => {
  const n = Math.round(ctl(S, 'n', 1)), trueA = -Math.PI / 4.2, cx = 130, cy = 140, R = 85;
  if (S.nk !== n) { S.nk = n; S.est = []; }
  ticks(S, 'k', t, .8, k => {
    const r = mulberry(k * 97 + n); let gx = 0, gy = 0;
    for (let i = 0; i < n; i++) { gx += Math.cos(trueA) + 1.6 * gauss(r); gy += Math.sin(trueA) + 1.6 * gauss(r); }
    S.est.push([gx / n, gy / n]); if (S.est.length > 12) S.est.shift();
  });
  rr(c, 18, 28, 224, 224, 14, C.panel);
  c.beginPath(); c.arc(cx, cy, R, 0, 7); c.strokeStyle = C.line; c.lineWidth = 1; c.stroke();
  const sc = R / 2.2, clip = (x, y) => { const L = Math.hypot(x, y) * sc; const k = L > R ? R / L : 1; return [cx + x * sc * k, cy + y * sc * k]; };
  (S.est || []).forEach(([x, y], i) => { const last = i === S.est.length - 1, [ex, ey] = clip(x, y); arrow(c, cx, cy, ex, ey, last ? C.pink : alpha(C.pink, .12 + .03 * i), last ? 3.5 : 1.5, last ? 10 : 6); });
  arrow(c, cx, cy, cx + Math.cos(trueA) * R * .85, cy + Math.sin(trueA) * R * .85, C.gold, 3, 10, [6, 4]);
  txt(c, 'direction each update points', 130, 18, { s: 11, b: 1 });
  // spread
  const E = S.est || []; let sp = 0; if (E.length > 1) { const mx = E.reduce((s, e) => s + e[0], 0) / E.length, my = E.reduce((s, e) => s + e[1], 0) / E.length; sp = Math.sqrt(E.reduce((s, e) => s + (e[0] - mx) ** 2 + (e[1] - my) ** 2, 0) / E.length); }
  txt(c, `episodes averaged: ${n}`, 320, 56, { s: 13, b: 1, c: C.fg });
  meter(c, 258, 96, 124, sp / 2.4, C.pink, 'wobble of the estimate');
  txt(c, 'gold dashes:', 258, 136, { s: 11, a: 'left', c: C.gold }); txt(c, 'true best direction', 258, 151, { s: 11, a: 'left', c: C.dim });
  txt(c, 'pink: this update', 258, 174, { s: 11, a: 'left', c: C.pink }); txt(c, 'faded: earlier ones', 258, 189, { s: 11, a: 'left', c: C.dim });
  txt(c, n < 4 ? 'few episodes:' : 'many episodes:', 320, 214, { s: 11.5, b: 1, c: C.gold }); txt(c, n < 4 ? 'right on average, but wild' : 'much steadier', 320, 230, { s: 11.5, c: C.gold });
};
CAP.p3_noisyG = 'The gold dashed arrow is the truly best direction to change the policy. Each pink arrow is one update estimated from real episodes. With one episode they swing wildly; averaging more steadies them.';

// ───────── Baselines & advantage ─────────
A.p3_baseB = (c, t, S) => {
  const b = ctl(S, 'b', 0), round = Math.floor(t / 3.5), r = mulberry(round * 13 + 5);
  const eps = Array.from({ length: 8 }, (_, i) => { const good = i % 2 === 0; return { good, G: (good ? 104 : 96) + 4 * gauss(r) }; });
  const y0 = 120, sc = Math.min(6, 82 / Math.max(1, ...eps.map(e => Math.abs(e.G - b))));
  txt(c, `baseline b = ${b.toFixed(0)}   ·   push = return − b`, 200, 14, { s: 12, b: 1, c: C.gold });
  line(c, 20, y0, 380, y0, C.dim, 1);
  let netA = 0, netB = 0;
  eps.forEach((e, i) => {
    const x = 40 + i * 44, v = e.G - b, col = e.good ? C.teal : C.purple; if (e.good) netA += v; else netB += v;
    arrow(c, x, y0, x, y0 - v * sc, v >= 0 ? col : alpha(col, .9), 4, 9);
    txt(c, e.G.toFixed(0), x, 218, { s: 10.5, c: C.dim }); txt(c, e.good ? 'A' : 'B', x, 232, { s: 11, b: 1, c: col });
  });
  txt(c, 'return of each episode, and its action', 200, 204, { s: 10, c: C.dim });
  const verdict = Math.sign(netA) !== Math.sign(netB) ? 'A pushed up, B pushed down: clear signal!' : 'both actions pushed the same way: signal buried';
  txt(c, `A avg ${(netA / 4 + b).toFixed(0)} vs B avg ${(netB / 4 + b).toFixed(0)} — ${verdict}`, 200, 250, { s: 11, b: 1, c: Math.sign(netA) !== Math.sign(netB) ? C.green : C.red });
};
CAP.p3_baseB = 'Eight episodes. Teal ones used action A, which is slightly better; purple ones used B. Each arrow is how hard that action is pushed: its return minus the baseline. At baseline 0 everything goes up.';

A.p3_advantage = (c, t) => {
  const Q = [13, 9, 11, 6], V = 10, names = ['jump', 'duck', 'run', 'wait'], y0 = 210, sc = 11, k = Math.floor(t / 2) % 6;
  txt(c, 'one situation, four possible actions', 200, 14, { s: 12, b: 1, c: C.gold });
  line(c, 20, y0 - V * sc, 380, y0 - V * sc, C.gold, 2.5, [6, 4]); txt(c, 'gold dashes: V(s) = 10, what you usually get here', 200, 34, { s: 11, c: C.gold });
  Q.forEach((q, i) => {
    const x = 50 + i * 85, on = k > i, A_ = q - V, col = A_ >= 0 ? C.green : C.red;
    bar(c, x, y0, 50, q * sc, alpha(C.blue, on ? .75 : .25)); txt(c, names[i], x + 25, y0 + 12, { s: 11 });
    txt(c, 'Q ' + q, x + 25, y0 - q * sc + 12, { s: 11, b: 1, c: C.bg });
    if (on) { arrow(c, x + 64, y0 - V * sc, x + 64, y0 - q * sc, col, 3, 8); txt(c, (A_ >= 0 ? '+' : '') + A_, x + 64 + 4, y0 - (V + A_ / 2) * sc, { s: 13, b: 1, c: col, a: 'left' }); }
  });
  txt(c, 'advantage = Q − V: green arrows mean "better than usual"', 200, 246, { s: 11.5, c: C.dim });
};
CAP.p3_advantage = 'Blue bars show how much each action is worth here. The gold dashed line is the usual value of the situation. Each arrow is the advantage: green above the line means better than usual, red means worse.';

// ───────── Actor-critic ─────────
A.p3_tdStep = (c, t) => {
  const T = 11, e = Math.floor(t / T) % 2, tt = t % T, ph = Math.min(4, Math.floor(tt / 2.4));
  const d = e ? { V: 10, r: 0, Vn: 8 } : { V: 10, r: 2, Vn: 9 }, delta = d.r + d.Vn - d.V, col = delta > 0 ? C.green : C.red;
  const steps = ['1 · the critic predicts', '2 · the actor acts, gets a reward', '3 · compare: TD error δ', '4 · both learn from δ'];
  txt(c, steps[Math.min(3, ph)], 200, 15, { s: 13, b: 1, c: C.gold });
  dot(c, 70, 80, 30, C.panel, C.teal); txt(c, 's', 70, 72, { s: 13, b: 1 }); txt(c, 'V ' + d.V, 70, 90, { s: 11, b: 1, c: C.teal });
  if (ph >= 1) { arrow(c, 102, 80, 268, 80, C.pink, 3, 10); txt(c, 'action → reward ' + (d.r >= 0 ? '+' : '') + d.r, 185, 66, { s: 12, b: 1, c: C.gold });
    dot(c, 300, 80, 30, C.panel, C.teal); txt(c, 's′', 300, 72, { s: 13, b: 1 }); txt(c, 'V ' + d.Vn, 300, 90, { s: 11, b: 1, c: C.teal }); }
  if (ph >= 2) {
    rr(c, 40, 124, 320, 40, 12, C.panel, col, 2);
    txt(c, `δ = reward + V(s′) − V(s) = ${d.r} + ${d.Vn} − ${d.V} = ${delta > 0 ? '+' : ''}${delta}`, 200, 144, { s: 13, b: 1, c: col });
  }
  if (ph >= 3) {
    const u = ease(clamp((tt - 7.2) / 1.5, 0, 1));
    txt(c, 'critic: V(s)', 100, 184, { s: 11, c: C.teal }); meter(c, 50, 192, 100, (d.V + u * delta * .1) / 12, C.teal); txt(c, (d.V + u * delta * .1).toFixed(1), 160, 197, { s: 11, b: 1, c: C.teal, a: 'left' });
    const p0 = .4, p1 = p0 + delta * .03 * u;
    txt(c, 'actor: chance of that action', 290, 184, { s: 11, c: C.pink }); meter(c, 240, 192, 100, p1, C.pink); txt(c, Math.round(p1 * 100) + '%', 350, 197, { s: 11, b: 1, c: C.pink, a: 'left' });
    txt(c, delta > 0 ? 'better than expected → do it more' : 'worse than expected → do it less', 200, 226, { s: 12.5, b: 1, c: col });
  }
  txt(c, 'happens after every single step', 200, 248, { s: 11, c: C.dim });
};
CAP.p3_tdStep = 'One step in four stages. The critic guesses this state is worth 10. The actor acts and gets a reward. The gap is the TD error delta. Then the critic’s guess and the actor’s odds both shift.';

function dartboard(c, cx, cy, R, darts, n, col) {
  [1, .66, .33].forEach((f, i) => { c.beginPath(); c.arc(cx, cy, R * f, 0, 7); c.fillStyle = [C.panel, '#232a4d', '#2c3460'][i]; c.fill(); });
  dot(c, cx, cy, 5, C.gold);
  for (let i = 0; i < Math.min(n, darts.length); i++) { const [x, y] = darts[i]; dot(c, cx + x * R, cy + y * R, i === n - 1 ? 5.5 : 4, i === n - 1 ? C.fg : col); }
}
const DART = memo('p3darts', () => { const r = mulberry(77); return Array.from({ length: 14 }, () => [gauss(r), gauss(r)]); });
A.p3_darts = (c, t) => {
  const n = Math.floor(t / .6) % 18;
  const L = DART.map(([x, y]) => [clamp(x * .38, -.95, .95), clamp(y * .38, -.95, .95)]), Rr = DART.map(([x, y]) => [clamp(.42 + x * .08, -.95, .95), clamp(-.3 + y * .08, -.95, .95)]);
  txt(c, 'gold dot = the true value we want to hit', 200, 14, { s: 12, b: 1, c: C.gold });
  dartboard(c, 105, 118, 78, L, n, C.pink); dartboard(c, 295, 118, 78, Rr, n, C.teal);
  txt(c, 'real returns', 105, 210, { s: 12, b: 1, c: C.pink }); txt(c, 'centred but scattered (noisy)', 105, 226, { s: 11, c: C.dim });
  txt(c, 'critic’s guess', 295, 210, { s: 12, b: 1, c: C.teal }); txt(c, 'tight but off-centre (biased)', 295, 226, { s: 11, c: C.dim });
  txt(c, 'each dot = one estimate', 200, 248, { s: 11, c: C.dim });
};
CAP.p3_darts = 'Each dot is one estimate of how good an action was; the gold centre is the truth. Pink real returns land around the centre but scatter. Teal critic guesses cluster tightly, off to one side.';

// ───────── GAE / n-step ─────────
A.p3_nstepN = (c, t, S) => {
  const n = Math.round(ctl(S, 'n', 3)), R = [1, 0, 2, 0, 1, 0, 0, 3], trueRest = i => R.slice(i).reduce((a, b) => a + b, 0), guess = i => i >= 8 ? 0 : trueRest(i) + 1.5;
  if (S.nk !== n) { S.nk = n; S.t0 = t; }
  const tt = t - S.t0, walk = Math.min(n, Math.floor(tt / .7)), done = tt > n * .7 + .6;
  txt(c, `${n}-step return: ${n} real rewards, then the critic’s guess`, 200, 14, { s: 12, b: 1, c: C.gold });
  for (let i = 0; i < 8; i++) {
    const x = 34 + i * 44, real = i < n;
    dot(c, x, 64, 16, real && i < walk ? alpha(C.teal, .9) : C.panel, real ? C.teal : C.line);
    txt(c, '' + R[i], x, 64, { s: 13, b: 1, c: real && i < walk ? C.bg : C.fg }); txt(c, 'r' + (i + 1), x, 92, { s: 10, c: C.dim });
  }
  agent(c, 34 + Math.min(walk, 7.6) * 44 - (walk >= n ? 22 : 0), 30, C.pink, 9);
  const sumReal = R.slice(0, n).reduce((a, b) => a + b, 0), g = n < 8 ? guess(n) : 0;
  if (done && n < 8) { const x0 = 34 + n * 44 - 18; rr(c, x0, 46, 384 - x0, 36, 12, C.panel); rr(c, x0, 46, 384 - x0, 36, 12, alpha(C.purple, .35), C.purple, 2); txt(c, n > 5 ? `guess ${g.toFixed(1)}` : `critic guesses ${g.toFixed(1)} for the rest`, (x0 + 384) / 2, 64, { s: 11.5, b: 1 }); }
  if (done) txt(c, `estimate = ${sumReal}${n < 8 ? ' + ' + g.toFixed(1) : ''} = ${(sumReal + g).toFixed(1)}   (truth: ${trueRest(0)})`, 200, 124, { s: 13, b: 1, c: C.fg });
  meter(c, 40, 168, 140, n / 8, C.teal, 'noise from real rewards');
  meter(c, 220, 168, 140, (8 - n) / 8, C.purple, 'reliance on the critic (bias)');
  txt(c, 'here the critic overestimates by 1.5', 200, 204, { s: 11, c: C.dim });
  txt(c, n <= 2 ? 'small n: steady, but inherits the critic’s error' : n >= 7 ? 'large n: honest, but many random rewards pile up' : 'middle n: a bit of both', 200, 230, { s: 12, b: 1, c: C.gold });
};
CAP.p3_nstepN = 'The circles are the rewards that really came. The agent counts the first n in teal, then the purple block is the critic’s guess for everything after. The meters show the trade you make.';

A.p3_gaeKnob = (c, t, S) => {
  const lam = ctl(S, 'lam', .95), N = 8, w = [];
  for (let k = 1; k <= N; k++) w.push(k < N ? (1 - lam) * Math.pow(lam, k - 1) : Math.pow(lam, N - 1));
  txt(c, `λ = ${lam.toFixed(2)}`, 100, 16, { s: 15, b: 1, c: C.gold });
  rr(c, 10, 30, 190, 200, 12, C.panel); txt(c, 'weight given to each n-step', 105, 44, { s: 10.5, c: C.dim });
  w.forEach((v, i) => { const x = 22 + i * 22; bar(c, x, 196, 16, Math.min(1, v) * 130, mix(C.purple, C.teal, i / (N - 1))); txt(c, i < N - 1 ? '' + (i + 1) : 'all', x + 8, 208, { s: 10, c: C.dim }); });
  txt(c, 'n = 1 … full return', 105, 222, { s: 10, c: C.dim });
  const n = Math.floor(t / .5) % 18, spread = lerp(.09, .42, lam), off = lerp(.5, .02, lam);
  const D = DART.map(([x, y]) => [clamp(off + x * spread, -.95, .95), clamp(-off * .6 + y * spread, -.95, .95)]);
  dartboard(c, 300, 120, 80, D, n, mix(C.purple, C.teal, lam));
  txt(c, 'estimates vs truth (gold)', 300, 22, { s: 11, c: C.dim });
  txt(c, lam < .3 ? 'trusts the critic: tight, off-centre' : lam > .97 ? 'real returns: centred, scattered' : 'blend: near centre, fairly tight', 300, 216, { s: 11, b: 1, c: C.gold });
  txt(c, 'PPO’s usual default: 0.95', 200, 248, { s: 11, c: C.dim });
};
CAP.p3_gaeKnob = 'Left: how much weight each n-step return gets for the current lambda. Short ones are purple, long ones teal. Right: where the resulting estimates land compared with the true value at the gold centre.';

// ───────── PPO ─────────
const PERF = x => x < 2.6 ? .15 + .8 * Math.exp(-(((x - 2.4) / 1.2) ** 2)) : .12 + .03 * Math.sin(x * 2); // peak then a cliff
A.p3_cliffStep = (c, t, S) => {
  const step = ctl(S, 'step', .4), X = x => 30 + x * 85, Y = v => 210 - v * 170;
  if (S.sk !== step) { S.sk = step; S.x = .3; S.trail = [.3]; S.k = Math.floor(t / 1.1); S.stuck = 0; }
  ticks(S, 'k', t, 1.1, () => {
    if (S.stuck > 2 || S.trail.length > 12) { S.x = .3; S.trail = [.3]; S.stuck = 0; return; }
    const g = (PERF(S.x + .01) - PERF(S.x - .01)) / .02, nx = clamp(S.x + step * Math.sign(g) * Math.min(1.5, Math.abs(g) * 1.3 + .4), 0, 4.1);
    if (Math.abs(g) < .02) S.stuck++; S.x = nx; S.trail.push(nx);
  });
  const pts = []; for (let i = 0; i <= 120; i++) { const x = i / 120 * 4.1; pts.push([X(x), Y(PERF(x))]); }
  line(c, 30, 210, 380, 210, C.line); poly(c, pts, C.teal, 3);
  txt(c, 'policy settings →', 376, 224, { s: 10.5, c: C.dim, a: 'right' }); txt(c, 'how well it plays', 34, 30, { s: 10.5, c: C.teal, a: 'left' });
  const tr = S.trail || [.3];
  for (let i = 1; i < tr.length; i++) { const a = [X(tr[i - 1]), Y(PERF(tr[i - 1]))], b = [X(tr[i]), Y(PERF(tr[i]))]; curve(c, a, b, -18, alpha(C.gold, .7), 0, 2); }
  const x = tr.at(-1), fell = x > 2.6;
  agent(c, X(x), Y(PERF(x)) - 2, fell ? C.red : C.pink, 9);
  txt(c, fell ? 'fell off the cliff: bad policy → bad data → stuck' : 'climbing with steps of size ' + step.toFixed(2), 200, 14, { s: 12.5, b: 1, c: fell ? C.red : C.gold });
  txt(c, 'PPO keeps every step small so this never happens', 200, 246, { s: 11.5, c: C.dim });
};
CAP.p3_cliffStep = 'The teal curve is how well the policy plays at each setting, with a cliff just past the peak. Gold hops are update steps. Small steps climb safely; too big a step jumps off the cliff and gets stuck.';

A.p3_ppoEpochs = (c, t, S) => {
  const eps = ctl(S, 'eps', .2), T = 11, ep = Math.min(8, Math.floor((t % T) / 1.1)), old = .1;
  const raw = old * Math.pow(1.3, ep), clipped = Math.min(raw, old * (1 + eps));
  txt(c, `same batch of data, reused for epoch ${ep} of 8`, 200, 14, { s: 12, b: 1, c: C.gold });
  txt(c, 'a good action (advantage > 0), old chance 10%', 200, 32, { s: 11, c: C.dim });
  const y0 = 210, sc = 230;
  for (const [x, lab, v, col] of [[90, 'no clipping', raw, C.red], [290, 'PPO (clipped)', clipped, C.teal]]) {
    rr(c, x - 70, 44, 140, 180, 12, C.panel);
    rr(c, x - 60, y0 - old * (1 + eps) * sc, 120, (old * (1 + eps) - old * (1 - eps)) * sc, 4, alpha(C.gold, .18));
    line(c, x - 60, y0 - old * sc, x + 60, y0 - old * sc, C.dim, 1.5, [4, 3]);
    bar(c, x - 22, y0, 44, Math.min(v, .6) * sc, col); txt(c, Math.round(v * 100) + '%', x, y0 - Math.min(v, .6) * sc - 10, { s: 13, b: 1, c: col });
    txt(c, lab, x, 236, { s: 12, b: 1, c: col });
  }
  txt(c, 'gold band: 1 ± ε times the old chance', 200, 252, { s: 11, c: C.gold });
};
CAP.p3_ppoEpochs = 'The same experience is reused for several passes. Without clipping, the good action’s chance keeps inflating, in red. PPO, in teal, stops once it leaves the gold band around the old chance.';

// ───────── SAC / entropy ─────────
A.p3_softTemp = (c, t, S) => {
  const al = ctl(S, 'alpha', .5), Q = [5, 4.6, 3, 1, .5], names = ['route 1', 'route 2', 'route 3', 'route 4', 'route 5'], p = softmax(Q, Math.max(.03, al)), H = entropy2(p);
  txt(c, `α = ${al.toFixed(2)}   ·   how much randomness is worth`, 200, 14, { s: 12, b: 1, c: C.gold });
  const picks = memo('p3picks', () => { const r = mulberry(9); return Array.from({ length: 400 }, () => r()); });
  const n = Math.floor(t / .45) % 26;
  p.forEach((q, i) => {
    const x = 30 + i * 70; bar(c, x, 180, 50, q * 120, mix(C.blue, C.gold, q)); txt(c, Math.round(q * 100) + '%', x + 25, 170 - q * 120, { s: 12, b: 1 });
    txt(c, names[i], x + 25, 192, { s: 10.5 }); txt(c, 'Q ' + Q[i], x + 25, 206, { s: 10, c: C.dim });
  });
  // recent sampled choices as dots above bars
  const cum = p.reduce((a, q) => (a.push((a.at(-1) || 0) + q), a), []);
  for (let j = 0; j < n; j++) { const u = picks[(Math.floor(t / 11.7) * 26 + j) % 400]; const i = cum.findIndex(v => u <= v); dot(c, 30 + (i < 0 ? 4 : i) * 70 + 8 + (j % 5) * 8.5, 38 + Math.floor(j / 5) * 9, 3.5, C.pink); }
  meter(c, 110, 222, 180, H / Math.log2(5), C.purple); txt(c, `entropy ${H.toFixed(2)}`, 300, 227, { s: 11, b: 1, c: C.purple, a: 'left' });
  txt(c, al < .2 ? 'α small: nearly always the best route' : al > 1.5 ? 'α large: spreads its bets widely' : 'mostly the best, sometimes the others', 200, 248, { s: 11.5, c: C.dim });
};
CAP.p3_softTemp = 'Five routes with different values, Q. Bars show how often the agent picks each; pink dots are recent picks. Alpha sets how much spreading out is rewarded; the purple meter shows the entropy.';

A.p3_routes = (c, t) => {
  const T = 14, tt = t % T, blocked = tt > 5.5;
  const panel = (x0, title, known, col) => {
    rr(c, x0, 26, 188, 198, 12, C.panel); txt(c, title, x0 + 94, 40, { s: 11.5, b: 1, c: col });
    const H = [x0 + 22, 132], W = [x0 + 166, 132], ys = [70, 132, 194];
    ys.forEach((y, i) => { const pts = [H, [x0 + 70, y], [x0 + 118, y], W]; poly(c, pts, known[i] ? alpha(col, .25 + .75 * known[i]) : alpha(C.dim, .25), 2 + 6 * known[i], known[i] ? null : [4, 4]); });
    txt(c, '🏠', H[0], H[1], { s: 18 }); txt(c, '🏢', W[0], W[1], { s: 18 });
    if (blocked) txt(c, '🚧', x0 + 94, 70, { s: 18 });
    // which route does the agent take?
    let route = 0; if (blocked) route = known[1] > .1 ? 1 : known[2] > .1 ? 2 : -1;
    if (route >= 0) { const y = ys[route], pts = [H, [x0 + 70, y], [x0 + 118, y], W], u = ((tt * .4) % 1); const p = along(pts, u); agent(c, p[0], p[1], col, 8); }
    else { const p = along([H, [x0 + 70, 70], [x0 + 94, 70]], Math.min(1, (tt - 5.5) / 2)); agent(c, p[0], p[1], C.red, 8); if (tt > 7.5) txt(c, '😵 no backup known', x0 + 94, 162, { s: 11, b: 1, c: C.red }); }
  };
  panel(8, 'greedy: one route only', [1, 0, 0], C.pink);
  panel(204, 'max-entropy: knows backups', [1, .5, .3], C.teal);
  txt(c, blocked ? 'road closed! the soft agent switches instantly' : 'both mostly drive the top (fastest) route', 200, 14, { s: 12, b: 1, c: blocked ? C.gold : C.fg });
  txt(c, 'line thickness = how well each route is known', 200, 246, { s: 11, c: C.dim });
};
CAP.p3_routes = 'Two drivers, home to office. Thick lines are routes a driver knows well. Both use the fast top route, until it closes. The greedy driver has no backup; the max-entropy driver already knows the others.';

// ───────── DDPG / TD3 ─────────
const QH = a => .62 * Math.exp(-(((a - .45) / .26) ** 2)) + .32 * Math.exp(-(((a + .55) / .25) ** 2)) + .08;
A.p3_hillTap = (c, t, S) => {
  const X = a => 30 + (a + 1) * 170, Y = q => 205 - q * 230, dQ = a => (QH(a + 1e-3) - QH(a - 1e-3)) / 2e-3;
  S.a ??= -.2; S.trail ||= [];
  for (const q of takeTaps(S)) { S.a = clamp((q.x - 30) / 170 - 1, -1, 1); S.trail = []; S.tap = t; }
  if (S.tap == null && Math.floor(t / 9) !== S.round) { S.round = Math.floor(t / 9); S.a = [-.15, -.95, .95, -.35][S.round % 4]; S.trail = []; }
  ticks(S, 'k', t, .3, () => { const g = dQ(S.a); if (Math.abs(g) > .03) { S.trail.push(S.a); S.a = clamp(S.a + .045 * Math.sign(g) * Math.min(1, Math.abs(g)), -1, 1); if (S.trail.length > 30) S.trail.shift(); } });
  const pts = []; for (let i = 0; i <= 100; i++) { const a = -1 + i / 50; pts.push([X(a), Y(QH(a))]); }
  line(c, 30, 205, 370, 205, C.line); poly(c, pts, C.teal, 3);
  txt(c, 'steering angle →', 370, 218, { s: 10.5, c: C.dim, a: 'right' }); txt(c, "critic's score Q", 34, 36, { s: 10.5, c: C.teal, a: 'left' });
  S.trail.forEach((a, i) => dot(c, X(a), Y(QH(a)), 3, alpha(C.gold, .2 + .025 * i)));
  const g = dQ(S.a), stopped = Math.abs(g) < .03;
  if (!stopped) arrow(c, X(S.a), Y(QH(S.a)) - 16, X(S.a) + 26 * Math.sign(g), Y(QH(S.a)) - 16, C.gold, 3, 8);
  agent(c, X(S.a), Y(QH(S.a)), C.pink, 9);
  txt(c, `steer ${S.a.toFixed(2)} → score ${(QH(S.a) * 10).toFixed(1)}`, 200, 14, { s: 12.5, b: 1, c: C.gold });
  txt(c, stopped ? (QH(S.a) < .5 ? 'stuck on a smaller hill (a local best)' : 'reached the top: best action here') : 'stepping uphill…', 200, 232, { s: 12, b: 1, c: stopped ? (QH(S.a) < .5 ? C.red : C.green) : C.fg });
  txt(c, 'tap the curve to drop the actor somewhere else', 200, 250, { s: 11, c: C.dim });
};
TOUCH.add('p3_hillTap');
CAP.p3_hillTap = 'The teal curve is the critic’s score for every steering angle. The robot is the actor’s current choice; the gold arrow shows which way is uphill. It keeps stepping until it reaches a peak.';

A.p3_twinCritics = (c, t) => {
  const X = a => 30 + (a + 1) * 170, Y = q => 210 - q * 200, ph = t * .5;
  const truth = a => .55 * Math.exp(-(((a - .2) / .35) ** 2)) + .1;
  const c1 = a => truth(a) + .32 * Math.exp(-(((a + .6 + .1 * Math.sin(ph)) / .1) ** 2)) + .04 * Math.sin(a * 9 + ph);
  const c2 = a => truth(a) + .3 * Math.exp(-(((a - .75 - .1 * Math.cos(ph)) / .1) ** 2)) + .04 * Math.cos(a * 8 - ph);
  const mn = a => Math.min(c1(a), c2(a));
  const argmax = f => { let b = -1, ba = 0; for (let i = 0; i <= 200; i++) { const a = -1 + i / 100; if (f(a) > b) { b = f(a); ba = a; } } return ba; };
  const curveOf = (f, col, lw, dash) => { const p = []; for (let i = 0; i <= 120; i++) { const a = -1 + i / 60; p.push([X(a), Y(f(a))]); } poly(c, p, col, lw, dash); };
  line(c, 30, 210, 370, 210, C.line);
  curveOf(truth, C.fg, 1.5, [5, 4]); curveOf(c1, alpha(C.pink, .6), 2); curveOf(c2, alpha(C.purple, .6), 2); curveOf(mn, C.teal, 3.5);
  const a1 = argmax(c1), am = argmax(mn);
  dot(c, X(a1), Y(c1(a1)), 7, C.pink); line(c, X(a1), Y(c1(a1)) - 8, X(a1), 42, alpha(C.pink, .6), 1, [2, 2]); txt(c, 'one critic: fooled by a spike', clamp(X(a1), 100, 300), 36, { s: 10.5, b: 1, c: C.pink });
  dot(c, X(am), Y(mn(am)), 7, C.teal); line(c, X(am), Y(mn(am)) + 8, X(am), 214, alpha(C.teal, .6), 1, [2, 2]); txt(c, 'lower of two: near the real best', clamp(X(am), 100, 300), 220, { s: 10.5, b: 1, c: C.teal });
  txt(c, 'dashed = true value · pink, purple = two critics · teal = the lower of them', 200, 14, { s: 11, c: C.fg });
  txt(c, 'their spikes rarely line up, so the minimum cancels them', 200, 246, { s: 11.5, c: C.gold });
};
CAP.p3_twinCritics = 'The dashed line is each action’s true value. Pink and purple are two critics, each with its own false spike. The thick teal curve takes the lower of the two everywhere, so the spikes cancel.';

// ───────── Model-based ─────────
A.p3_modelLearn = (c, t) => {
  const o = { x0: 16, y0: 40, s: 42 }, walk = memo('p3walk', () => { const r = mulberry(4); let [x, y] = GW.start; const out = []; for (let i = 0; i < 60; i++) { const a = Math.floor(r() * 4); const [nx, ny, rw] = gwStep(x, y, a); out.push({ x, y, a, nx, ny, rw }); if (gwTerm(nx, ny)) { x = GW.start[0]; y = GW.start[1]; } else { x = nx; y = ny; } } return out; });
  const dt = 1.1, k = Math.floor(t / dt) % 40, u = (t % dt) / dt, st = walk[k];
  gwDraw(c, null, o);
  const seen = new Map(); for (let i = 0; i <= k; i++) { const w = walk[i]; seen.set(w.x + ',' + w.y + ',' + w.a, w); }
  for (const w of seen.values()) { if (w.x === w.nx && w.y === w.ny) continue; const p0 = gwC(w.x, w.y, o), p1 = gwC(w.nx, w.ny, o); arrow(c, lerp(p0[0], p1[0], .2), lerp(p0[1], p1[1], .2), lerp(p0[0], p1[0], .62), lerp(p0[1], p1[1], .62), alpha(C.purple, .7), 1.5, 5); }
  const pos = lerpPath([[st.x, st.y], [st.nx, st.ny]], u); agent(c, ...gwC(...pos, o), C.pink, 9);
  txt(c, 'explore, and note what each move does', 120, 18, { s: 11.5, b: 1, c: C.gold });
  rr(c, 236, 30, 156, 210, 12, C.panel); txt(c, '📒 the learned model', 314, 46, { s: 11.5, b: 1, c: C.purple });
  const dn = ['⬆', '➡', '⬇', '⬅'];
  for (let j = 0; j < 7; j++) { const w = walk[k - j]; if (!w) break; txt(c, `(${w.x},${w.y}) ${dn[w.a]} → (${w.nx},${w.ny}) ${w.rw ? (w.rw > 0 ? '+1' : '−1') : '0'}`, 314, 70 + j * 23, { s: 11.5, c: j ? C.dim : C.fg, b: !j }); }
  txt(c, `${seen.size} of 64 moves learned`, 314, 232, { s: 10.5, c: C.purple });
  txt(c, 'purple arrows: moves the model now knows', 120, 222, { s: 10.5, c: C.dim });
};
CAP.p3_modelLearn = 'The robot wanders the grid. After each move it notes where it was, which way it went, where it landed and the reward. That notebook on the right is its model; purple arrows mark known moves.';

A.p3_compound = (c, t, S) => {
  const err = ctl(S, 'err', .08), N = 20, k = Math.floor(t / .45) % (N + 6), shown = Math.min(N, k);
  const real = [], dream = []; let x = 0, y = 0, vx = 1, vy = 1.6, dx = 0, dy = 0, dvx = 1, dvy = 1.6, a = 0;
  for (let i = 0; i <= N; i++) { real.push([x, y]); dream.push([dx, dy]);
    x += vx; y += vy; vy -= .16;
    dx += dvx; dy += dvy; dvx *= 1 + err * .25; dvy -= .16 * (1 - err * 1.2); }
  const X = v => 30 + v * 16, Y = v => 200 - v * 17;
  line(c, 20, 200, 380, 200, C.line);
  c.save(); c.beginPath(); c.rect(14, 24, 374, 184); c.clip();
  poly(c, real.slice(0, shown + 1).map(p => [X(p[0]), Y(p[1])]), C.teal, 3); poly(c, dream.slice(0, shown + 1).map(p => [X(p[0]), Y(p[1])]), C.purple, 2.5, [6, 4]);
  for (let i = 0; i <= shown; i += 1) { dot(c, X(real[i][0]), Y(real[i][1]), 3, C.teal); dot(c, X(dream[i][0]), Y(dream[i][1]), 3, C.purple); }
  const d = Math.hypot(real[shown][0] - dream[shown][0], real[shown][1] - dream[shown][1]);
  line(c, X(real[shown][0]), Y(real[shown][1]), X(dream[shown][0]), Y(dream[shown][1]), C.red, 2, [2, 3]);
  c.restore();
  txt(c, `imagined step ${shown} · off by ${d.toFixed(1)}`, 200, 14, { s: 12.5, b: 1, c: C.gold });
  txt(c, 'teal: what really happens   ·   purple dashes: the model’s imagination', 200, 222, { s: 11, c: C.fg });
  txt(c, `each step the model is only ${Math.round(err * 100)}% off — but errors pile up`, 200, 244, { s: 11, c: C.dim });
};
CAP.p3_compound = 'A thrown ball. Teal is its real path; purple dashes are the path the model imagines, one step at a time. A small error each step adds up, so the red gap grows the further ahead it imagines.';

// ───────── MCTS ─────────
A.p3_mctsTap = (c, t, S) => {
  const P = [.6, .45, .3], names = ['A', 'B', 'C'];
  if (!S.st) S.st = { N: [10, 2, 1], W: [6, 1, 0], last: null, tot: 13 };
  const st = S.st, step = (seed) => {
    const ucb = st.N.map((n, i) => st.W[i] / n + 1.1 * Math.sqrt(Math.log(st.tot) / n)), i = ucb.indexOf(Math.max(...ucb));
    const win = mulberry(seed)() < P[i]; st.N[i]++; st.W[i] += win; st.tot++; st.last = { i, win, at: t };
    if (st.tot > 60) { st.N = [10, 2, 1]; st.W = [6, 1, 0]; st.tot = 13; }
  };
  const taps = takeTaps(S); taps.forEach((_, j) => step(Math.floor(t * 1000) + j));
  if (taps.length) S.lastTap = t;
  if (S.lastTap == null || t - S.lastTap > 5) ticks(S, 'k', t, 2.2, k => step(k * 17 + 3));
  dot(c, 200, 40, 18, C.panel, C.gold); txt(c, 'now', 200, 40, { s: 11, b: 1 });
  const ucb = st.N.map((n, i) => ({ wr: st.W[i] / n, bo: 1.1 * Math.sqrt(Math.log(st.tot) / n) }));
  names.forEach((nm, i) => {
    const x = 80 + i * 120, y = 108, sel = st.last && st.last.i === i && t - st.last.at < 1.6;
    line(c, 200, 58, x, y - 22, sel ? C.gold : C.line, sel ? 3 : 1.5);
    dot(c, x, y, 22, heat(ucb[i].wr * .9), sel ? C.gold : C.line); txt(c, 'move ' + nm, x, y - 4, { s: 11, b: 1, c: C.bg }); txt(c, `${st.W[i]}/${st.N[i]}`, x, y + 9, { s: 11, b: 1, c: C.bg });
    const y0 = 222, sc = 52; bar(c, x - 16, y0, 32, ucb[i].wr * sc, C.teal); bar(c, x - 16, y0 - ucb[i].wr * sc, 32, ucb[i].bo * sc, alpha(C.gold, .75));
    txt(c, (ucb[i].wr + ucb[i].bo).toFixed(2), x, y0 - (ucb[i].wr + ucb[i].bo) * sc - 9, { s: 11, b: 1 });
    if (sel) popup(c, st.last.win ? 'won! +1' : 'lost', x, y + 36, (t - st.last.at) / 1.6, st.last.win ? C.green : C.red);
  });
  txt(c, 'teal = win rate · gold = bonus for few tries · pick the tallest', 200, 238, { s: 11, c: C.dim });
  txt(c, 'tap to run one more playout', 200, 252, { s: 11, b: 1, c: C.gold });
};
TOUCH.add('p3_mctsTap');
CAP.p3_mctsTap = 'Three candidate moves with their wins out of playouts. Each bar is the move’s UCB score: teal win rate plus a gold bonus that is big when a move has few tries. The tallest bar gets the next playout.';

// ───────── AlphaZero ─────────
A.p3_twoHeads = (c, t) => {
  const boards = [[1, 0, 0, 0, 2, 0, 0, 0, 0], [1, 0, 2, 0, 1, 0, 0, 0, 0], [1, 2, 1, 0, 2, 0, 0, 0, 0]];
  const pol = [[0, .1, .15, .1, 0, .1, .15, .2, .2], [0, .05, 0, .05, 0, .05, .05, .05, .75], [0, 0, 0, .15, 0, .1, .1, .6, .05]], val = [.52, .78, .35];
  const k = Math.floor(t / 4.5) % 3, u = (t % 4.5) / 4.5, b = boards[k];
  const cell = (x, y, i, s) => [x + (i % 3) * s, y + Math.floor(i / 3) * s];
  for (let i = 0; i < 9; i++) { const [x, y] = cell(16, 70, i, 34); rr(c, x, y, 32, 32, 5, C.panel); if (b[i]) txt(c, b[i] === 1 ? 'X' : 'O', x + 16, y + 17, { s: 18, b: 1, c: b[i] === 1 ? C.pink : C.teal }); }
  txt(c, 'position', 66, 56, { s: 11, b: 1 });
  // network
  const layers = [3, 5, 5, 3];
  layers.forEach((n, li) => { for (let j = 0; j < n; j++) { const x = 140 + li * 30, y = 120 + (j - (n - 1) / 2) * 20; dot(c, x, y, 5, alpha(C.purple, .4 + .6 * (Math.abs(Math.sin(t * 3 - li + j)) * (u > li * .12 ? 1 : .2)))); } });
  txt(c, '🧠 one network', 185, 56, { s: 11, b: 1, c: C.purple });
  arrow(c, 122, 120, 134, 120, C.dim, 2, 6);
  arrow(c, 236, 110, 262, 74, C.dim, 2, 7); arrow(c, 236, 130, 262, 176, C.dim, 2, 7);
  const show = clamp((u - .3) / .3, 0, 1);
  for (let i = 0; i < 9; i++) { const [x, y] = cell(262, 36, i, 26); rr(c, x, y, 24, 24, 4, b[i] ? C.wall : alpha(C.gold, .1 + .9 * pol[k][i] * show)); if (b[i]) txt(c, b[i] === 1 ? 'X' : 'O', x + 12, y + 13, { s: 12, b: 1, c: C.dim }); }
  txt(c, 'policy head', 300, 24, { s: 11, b: 1, c: C.gold }); txt(c, 'bright =', 344, 60, { s: 10, c: C.dim, a: 'left' }); txt(c, 'search here', 344, 73, { s: 10, c: C.dim, a: 'left' });
  txt(c, 'value head: X wins?', 330, 152, { s: 10.5, b: 1, c: C.teal });
  meter(c, 272, 168, 116, val[k] * show, C.teal); txt(c, Math.round(val[k] * 100 * show) + '%', 330, 192, { s: 13, b: 1, c: C.teal });
  txt(c, 'one look at the board → where to look + who is winning', 200, 238, { s: 11.5, c: C.dim });
};
CAP.p3_twoHeads = 'A tic-tac-toe position goes into one network. The top output lights up the squares worth searching, the policy head. The bottom output, the value head, estimates X’s chance of winning.';

A.p3_searchTeacher = (c, t) => {
  const prior = [.3, .25, .2, .15, .1], visits = [.1, .62, .14, .09, .05], T = 13, tt = t % T;
  const ph = tt < 2 ? 0 : tt < 7 ? 1 : 2, sims = Math.round(clamp((tt - 2) / 5, 0, 1) * 200), learn = ease(clamp((tt - 7.5) / 3.5, 0, 1));
  const net = prior.map((p, i) => lerp(p, visits[i], learn)), vis = visits.map(v => v * clamp((tt - 2) / 5, 0, 1));
  txt(c, ['1 · the network suggests moves', `2 · tree search explores (${sims} playouts)`, '3 · the network learns to match the search'][ph], 200, 14, { s: 12.5, b: 1, c: C.gold });
  const y0 = 200, sc = 220;
  for (let i = 0; i < 5; i++) {
    const x = 40 + i * 72;
    bar(c, x, y0, 26, net[i] * sc, C.purple); bar(c, x + 28, y0, 26, vis[i] * sc, C.gold);
    txt(c, 'move ' + (i + 1), x + 27, y0 + 13, { s: 11 });
  }
  if (ph === 2) { const i = 1; arrow(c, 40 + i * 72 + 13, y0 - net[i] * sc - 6, 40 + i * 72 + 13, y0 - visits[i] * sc + 4, C.fg, 2, 7); }
  txt(c, 'purple: network’s guess', 110, 236, { s: 11.5, b: 1, c: C.purple }); txt(c, 'gold: search’s choice', 290, 236, { s: 11.5, b: 1, c: C.gold });
  txt(c, 'search found move 2 is best; next time the network sees it at a glance', 200, 252, { s: 11, c: C.dim });
};
CAP.p3_searchTeacher = 'Purple bars are the network’s first guess for five moves. Gold bars show where the tree search spent its playouts. Then the network is trained so its purple bars match the gold.';
})();
