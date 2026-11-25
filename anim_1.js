// Part 1 simulations (Foundations + Bandits). Loaded after anim.js: uses its helpers, A and RLAnim.
// Interactive ones read S.ctl (sliders), S.taps (taps on the canvas, consumed with shift) or S.ptr.
'use strict';
{
const CAP = RLAnim.CAP, TOUCH = RLAnim.TOUCH;
const tick = (S, t) => { const dt = clamp(t - (S.lt ?? t), 0, .05); S.lt = t; return dt; };
const inBox = (q, x, y, w, h) => q.x >= x && q.x <= x + w && q.y >= y && q.y <= y + h;
const pct = v => Math.round(v * 100) + '%';
function betaCurve(a, b, n = 60) { // normalised Beta(a, b) density samples on (0, 1)
  const ys = []; let mx = -1e9;
  for (let k = 1; k < n; k++) { const q = k / n, l = (a - 1) * Math.log(q) + (b - 1) * Math.log(1 - q); ys.push(l); mx = Math.max(mx, l); }
  return ys.map(l => Math.exp(l - mx));
}
function drawBelief(c, a, b, x0, x1, yb, h, col) {
  const ys = betaCurve(a, b), n = ys.length + 1;
  c.beginPath(); c.moveTo(x0, yb); ys.forEach((v, k) => c.lineTo(lerp(x0, x1, (k + 1) / n), yb - v * h)); c.lineTo(x1, yb); c.closePath();
  c.fillStyle = alpha(col, .35); c.fill(); c.strokeStyle = col; c.lineWidth = 2; c.stroke();
}
function betaSample(r, a, b) { // sum-of-exponentials for small integer a, b
  let x = 0, y = 0; for (let i = 0; i < a; i++) x -= Math.log(1 - r()); for (let i = 0; i < b; i++) y -= Math.log(1 - r()); return x / (x + y);
}
function axis(c, x0, y0, x1, y1t, lab) { line(c, x0, y0, x1, y0, C.line, 1.5); line(c, x0, y0, x0, y1t, C.line, 1.5); if (lab) txt(c, lab, x1, y0 + 12, { s: 11, c: C.dim, a: 'right' }); }

// ───────── loop: one turn, step by step ─────────
A.p1_turn = (c, t) => {
  const P = 2.6, ph = Math.floor(t / P) % 4, turn = Math.floor(t / (4 * P)), u = (t % P) / P;
  const pos = turn % 4, after = ph >= 2 ? pos + 1 : pos;
  rr(c, 14, 50, 120, 150, 14, C.panel, ph === 1 ? C.pink : C.line, ph === 1 ? 3 : 1.5);
  txt(c, '🤖', 74, 110, { s: 40 }); txt(c, 'Agent', 74, 150, { s: 14, b: 1 }); txt(c, 'the learner', 74, 168, { s: 11, c: C.dim });
  rr(c, 226, 50, 160, 150, 14, C.panel, ph === 2 ? C.teal : C.line, ph === 2 ? 3 : 1.5);
  txt(c, 'Environment', 306, 66, { s: 13, b: 1 });
  for (let i = 0; i < 5; i++) { rr(c, 238 + i * 28, 104, 25, 25, 5, i === 4 ? alpha(C.green, .4) : C.bg, C.line, 1); }
  txt(c, '🏁', 238 + 4 * 28 + 12.5, 117, { s: 13 });
  const ax = 238 + after * 28 + 12.5; dot(c, ax, 117, 8, C.pink);
  txt(c, 'square ' + (after + 1), 306, 146, { s: 11, c: C.dim });
  const on = k => ph === k, col = (k, cc) => on(k) ? cc : alpha(C.dim, .35);
  arrow(c, 222, 88, 140, 88, col(0, C.blue), on(0) ? 3 : 2); txt(c, 'state', 181, 78, { s: 12, b: on(0), c: col(0, C.blue) });
  arrow(c, 140, 178, 222, 178, col(1, C.pink), on(1) ? 3 : 2); txt(c, 'action →', 181, 168, { s: 12, b: on(1), c: col(1, C.pink) });
  arrow(c, 222, 132, 140, 132, col(3, C.gold), on(3) ? 3 : 2); txt(c, 'reward', 181, 122, { s: 12, b: on(3), c: col(3, C.gold) });
  if (on(0)) dot(c, lerp(222, 140, ease(u)), 88, 5, C.blue);
  if (on(1)) dot(c, lerp(140, 222, ease(u)), 178, 5, C.pink);
  if (on(3)) { const rw = after === 4 ? '+1' : '0'; txt(c, rw, lerp(222, 140, ease(u)), 145, { s: 14, b: 1, c: C.gold }); }
  const steps = ['1 · Agent sees the state: “I’m on square ' + (pos + 1) + '”', '2 · Agent picks an action: move right', '3 · World changes: now on square ' + (pos + 2), '4 · World sends a reward: ' + (pos + 1 === 4 ? '+1, reached the flag!' : '0, nothing yet')];
  rr(c, 14, 212, 372, 34, 10, alpha(C.blue, .12), C.line, 1); txt(c, steps[ph], 200, 229, { s: 13, b: 1 });
  txt(c, 'turn ' + (turn % 4 + 1) + ' of the loop', 200, 24, { s: 12, c: C.gold });
};
CAP.p1_turn = 'One turn, slowed down: the agent sees its square, picks “move right”, the world moves it and hands back a reward. The reward is 0 until the robot reaches the flag.';

// ───────── loop: train the puppy yourself ─────────
const TRICKS = [['🪑', 'sit'], ['🔄', 'roll over'], ['🗣️', 'bark'], ['🌀', 'spin']];
A.p1_puppy = (c, t, S) => {
  if (!S.pp) { S.pp = { w: [1, 1, 1, 1], r: mulberry(7), next: 0, a: 0, gotT: -9, lastTap: -99, n: 0 }; }
  const P = S.pp, prob = P.w.map(v => v / P.w.reduce((a, b) => a + b, 0));
  const treat = auto => { P.w[P.a] += 1.2; P.gotT = t; P.n++; P.auto = auto; };
  while (S.taps.length) { S.taps.shift(); if (t - P.gotT > .4) treat(false); P.lastTap = t; }
  if (t >= P.next) { let u = P.r(), k = 0; while (k < 3 && u > prob[k]) { u -= prob[k]; k++; } P.a = k; P.next = t + 2.4; P.start = t; }
  if (t - P.lastTap > 9 && P.a === 0 && t - P.start > 1 && t - P.gotT > 2) treat(true);
  const [ico, name] = TRICKS[P.a];
  txt(c, '🐶', 90, 82, { s: 54 }); txt(c, 'tries:', 170, 62, { s: 12, c: C.dim, a: 'left' });
  txt(c, ico + ' ' + name, 170, 88, { s: 20, b: 1, a: 'left' });
  const g = t - P.gotT; if (g < 1.2) { txt(c, '🦴', 130, 50 - g * 20, { s: 26, al: 1 - g / 1.2 }); txt(c, 'treat!', 310, 72, { s: 16, b: 1, c: C.gold, al: 1 - g / 1.2 }); }
  txt(c, 'chance of each trick', 200, 128, { s: 11, c: C.dim });
  TRICKS.forEach(([i, nm], k) => {
    const x = 30 + k * 88; rr(c, x, 140, 76, 64, 10, C.panel, k === P.a ? C.pink : C.line, k === P.a ? 2.5 : 1);
    bar(c, x + 8, 196, 60 * prob[k], 12, alpha(C.teal, .85)); txt(c, i + ' ' + nm, x + 38, 154, { s: 11, b: 1 }); txt(c, pct(prob[k]), x + 38, 176, { s: 14, b: 1, c: C.teal });
  });
  txt(c, P.auto && g < 30 ? 'trainer rewards “sit” · tap to reward any trick yourself' : 'tap the picture to give a treat for the current trick', 200, 228, { s: 12, c: C.gold });
  txt(c, 'treats given: ' + P.n, 200, 248, { s: 11, c: C.dim });
};
TOUCH.add('p1_puppy');
CAP.p1_puppy = 'The puppy tries a random trick every couple of seconds. The teal bars show how likely each trick is. Every treat makes the rewarded trick more likely next time.';

// ───────── returns: small now vs big later ─────────
A.p1_twoPaths = (c, t) => {
  const T = 13, tt = t % T, k = Math.min(4, Math.floor(tt / 1.6));
  const rows = [{ n: 'Path A: party tonight', r: [1, 0, 0, 0], col: C.blue, y: 70 }, { n: 'Path B: study tonight', r: [-1, 0, 0, 10], col: C.pink, y: 160 }];
  const tot = [];
  rows.forEach((R, j) => {
    txt(c, R.n, 20, R.y - 34, { s: 13, b: 1, a: 'left', c: R.col });
    let sum = 0;
    for (let i = 0; i < 4; i++) {
      const x = 30 + i * 70, done = i < k; if (done) sum += R.r[i];
      rr(c, x, R.y - 18, 56, 36, 8, done ? alpha(R.col, .25) : C.panel, C.line, 1);
      txt(c, (R.r[i] > 0 ? '+' : '') + R.r[i], x + 28, R.y, { s: 15, b: 1, c: R.r[i] > 0 ? C.green : R.r[i] < 0 ? C.red : C.dim, al: done ? 1 : .35 });
      txt(c, ['tonight', 'day 2', 'day 3', 'exam'][i], x + 28, R.y + 28, { s: 10, c: C.dim });
    }
    if (k < 4) dot(c, 30 + k * 70 + 28, R.y - 24, 5, R.col);
    tot.push(sum);
    rr(c, 316, R.y - 22, 70, 44, 10, C.bg, R.col, 2); txt(c, 'return', 351, R.y - 9, { s: 10, c: C.dim }); txt(c, String(sum), 351, R.y + 9, { s: 18, b: 1 });
  });
  txt(c, k < 4 ? 'adding up rewards, one day at a time…' : 'B wins: 9 vs 1, even though it started with −1', 200, 238, { s: 13, b: 1, c: k < 4 ? C.dim : C.gold });
};
CAP.p1_twoPaths = 'Two plans, one day at a time. The boxes are daily rewards and the box on the right is the running total, the return. Plan B starts worse but ends far ahead.';

// ───────── returns: maze with a step cost (slider) ─────────
const MZ_SHORT = [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]];
const MZ_LONG = [[0, 1], [0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [5, 1], [5, 0], [6, 0], [6, 1]];
A.p1_mazeReturn = (c, t, S) => {
  const cost = S.ctl.cost ?? -1, s = 36, x0 = 18, y0 = 40;
  for (let y = 0; y < 3; y++) for (let x = 0; x < 7; x++) rr(c, x0 + x * s + 1, y0 + y * s + 1, s - 2, s - 2, 5, x === 6 && y === 1 ? alpha(C.green, .4) : C.panel);
  txt(c, '🚪', x0 + 6 * s + s / 2, y0 + s * 1.5, { s: 16 }); txt(c, 'start', x0 + s / 2, y0 + s * 1.5 + 26, { s: 10, c: C.dim });
  const T = 12, tt = t % T, pc = (x, y) => [x0 + x * s + s / 2, y0 + y * s + s / 2];
  [[MZ_LONG, C.gold, 4], [MZ_SHORT, C.teal, -4]].forEach(([P, col, off]) => {
    poly(c, P.map(([x, y]) => { const q = pc(x, y); return [q[0] + off, q[1] + off]; }), alpha(col, .5), 3);
    const f = Math.min(P.length - 1, tt / .8), q = lerpPath(P.map(([x, y]) => pc(x, y)), f); dot(c, q[0] + off, q[1] + off, 7, col);
  });
  const ret = n => 10 + cost * n, rs = ret(MZ_SHORT.length - 1), rl = ret(MZ_LONG.length - 1);
  const bx = 290;
  [['short: 6 steps', rs, C.teal, 70], ['long: 10 steps', rl, C.gold, 140]].forEach(([n, v, col, y]) => {
    txt(c, n, bx, y - 18, { s: 12, b: 1, a: 'left', c: col });
    txt(c, `10 ${cost < 0 ? '−' : '+'} ${Math.abs(cost).toFixed(1)}×${n.match(/\d+/)[0]} = ${v.toFixed(1)}`, bx, y, { s: 12, a: 'left' });
    bar(c, bx, y + 26, Math.max(0, v) * 9, 10, col);
  });
  const msg = Math.abs(rs - rl) < 1e-6 ? 'steps are free, so both routes tie' : rs > rl ? 'short route has the higher return' : 'long route has the higher return';
  txt(c, msg, 200, 196, { s: 13, b: 1, c: C.gold });
  txt(c, 'return = +10 at the door, plus the cost of every step', 200, 216, { s: 11, c: C.dim });
};
CAP.p1_mazeReturn = 'Two robots walk to the door: teal takes the short route, gold the long one. Each step adds the step cost, the door adds plus 10. The numbers on the right are each route’s return.';

// ───────── discount: bars shrink with γ (slider) ─────────
A.p1_gammaBars = (c, t, S) => {
  const g = S.ctl.gamma ?? .9, n = 10, x0 = 30, w = 32, yb = 200, sc = 13;
  let sum = 0; const hl = Math.floor(t / 1.4) % n;
  for (let k = 0; k < n; k++) {
    const v = 10 * Math.pow(g, k), grow = clamp((t - k * .25) / .6, 0, 1); sum += v;
    bar(c, x0 + k * 35, yb, w, v * sc * ease(grow), k === hl ? C.gold : alpha(C.teal, .8));
    txt(c, k === 0 ? 'now' : '+' + k, x0 + k * 35 + w / 2, yb + 12, { s: 10, c: C.dim });
    if (k === hl) txt(c, v.toFixed(v < 1 ? 2 : 1), x0 + k * 35 + w / 2, yb - v * sc - 10, { s: 12, b: 1, c: C.gold });
  }
  const hv = 10 * Math.pow(g, hl);
  txt(c, `reward 10, ${hl} step${hl === 1 ? '' : 's'} away: 10 × ${g.toFixed(2)}${hl > 1 ? superN(hl) : hl === 1 ? '' : '⁰'} = ${hv.toFixed(2)}`, 200, 22, { s: 13, b: 1 });
  txt(c, `the same reward of 10 at each future step · total worth now: ${sum.toFixed(1)}`, 200, 244, { s: 11, c: C.dim });
};
function superN(k) { return String(k).split('').map(d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]).join(''); }
CAP.p1_gammaBars = 'Each bar is a reward of 10 placed further in the future. Its height is what it is worth today: 10 times gamma, once for every step of waiting. The gold bar shows the sum.';

// ───────── discount: +1 now or +10 later (slider) ─────────
A.p1_gammaChoice = (c, t, S) => {
  const g = S.ctl.gamma ?? .5, vNow = 1, vLater = 10 * Math.pow(g, 6), pick = vLater > vNow ? 1 : 0;
  const start = [60, 130], A1 = [[60, 130], [130, 70]], A2 = [[60, 130], [110, 190], [170, 190], [230, 190], [290, 190], [340, 190], [360, 150]];
  poly(c, A1, pick === 0 ? C.blue : alpha(C.dim, .4), pick === 0 ? 4 : 2);
  poly(c, A2, pick === 1 ? C.pink : alpha(C.dim, .4), pick === 1 ? 4 : 2);
  A2.slice(1, -1).forEach(q => dot(c, q[0], q[1], 4, C.line));
  txt(c, '🍬 +1', 150, 56, { s: 15, b: 1 }); txt(c, 'right now', 150, 76, { s: 10, c: C.dim });
  txt(c, '🎁 +10', 360, 132, { s: 15, b: 1 }); txt(c, 'after 6 more steps', 340, 214, { s: 10, c: C.dim });
  const T = 5, u = (t % T) / (T - 1.2), path = pick ? A2 : A1, q = lerpPath(path, clamp(u, 0, 1) * (path.length - 1));
  agent(c, q[0], q[1], C.pink, 10);
  rr(c, 200, 22, 186, 72, 10, C.panel, C.line, 1);
  txt(c, 'worth today', 293, 36, { s: 11, c: C.dim });
  txt(c, `🍬  1.00`, 293, 56, { s: 13, b: 1, c: pick === 0 ? C.gold : C.fg });
  txt(c, `🎁  10 × γ⁶ = ${vLater.toFixed(2)}`, 293, 77, { s: 13, b: 1, c: pick === 1 ? C.gold : C.fg });
  txt(c, pick ? 'patient: goes for the big gift' : 'impatient: grabs the candy', 200, 246, { s: 13, b: 1, c: C.gold });
};
CAP.p1_gammaChoice = 'The robot can grab a candy worth 1 now, or walk 6 more steps for a gift worth 10. The box shows what each is worth today with this gamma. The robot picks the bigger number.';

// ───────── discount: how far ahead does it look? (slider) ─────────
A.p1_horizon = (c, t, S) => {
  const g = S.ctl.gamma ?? .9, H = 1 / (1 - g), K = 120, x0 = 34, x1 = 380, yb = 200, h = 140, X = k => x0 + (x1 - x0) * k / K;
  axis(c, x0, yb, x1, yb - h - 10, 'steps into the future →');
  c.beginPath(); c.moveTo(x0, yb); for (let k = 0; k <= K; k++) c.lineTo(X(k), yb - h * Math.pow(g, k)); c.lineTo(x1, yb); c.closePath();
  c.fillStyle = alpha(C.teal, .3); c.fill();
  poly(c, Array.from({ length: K + 1 }, (_, k) => [X(k), yb - h * Math.pow(g, k)]), C.teal, 2.5);
  const hx = X(Math.min(K, H)); line(c, hx, yb, hx, yb - h - 4, C.gold, 2, [5, 4]);
  txt(c, H > K ? `≈ ${Math.round(H)} steps (off the chart)` : `≈ ${Math.round(H)} steps`, Math.min(hx + 6, 300), yb - h + 2, { s: 13, b: 1, c: C.gold, a: 'left' });
  const k = (t * 12) % K; dot(c, X(k), yb - h * Math.pow(g, k), 5, C.pink);
  txt(c, `weight at step ${Math.floor(k)}: ${Math.pow(g, Math.floor(k)).toFixed(3)}`, x0 + 4, 30, { s: 11, a: 'left', c: C.pink });
  txt(c, `how much a reward counts, by how far away it is · 1 ÷ (1 − ${g.toFixed(3)}) = ${H.toFixed(1)}`, 200, 236, { s: 11, c: C.dim });
};
CAP.p1_horizon = 'The teal curve shows how much a reward counts depending on how many steps away it is. The gold line marks one over one minus gamma: roughly how far ahead the agent really looks.';

// ───────── mdp: slippery ice (tap to move) ─────────
A.p1_slippery = (c, t, S) => {
  if (!S.ice) S.ice = { r: mulberry(11), n: [0, 0, 0], go: -1, out: 0, start: 0, auto: 3 };
  const I = S.ice, s = 46, x0 = 40, y0 = 42, home = [1, 1];
  const fire = () => { const u = I.r(); I.out = u < .8 ? 0 : u < .9 ? 1 : 2; I.go = t; I.n[I.out]++; };
  while (S.taps.length) { S.taps.shift(); if (I.go < 0 || t - I.go > 1.3) fire(); I.auto = t + 6; }
  if (I.go < 0 && t > I.auto || I.go >= 0 && t - I.go > 2.6 && t > I.auto) { fire(); I.auto = t + 2.6; }
  for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) rr(c, x0 + x * s + 1, y0 + y * s + 1, s - 2, s - 2, 6, alpha(C.blue, .18), alpha(C.blue, .4), 1);
  txt(c, '🧊 ice', x0 + 1.5 * s, y0 - 14, { s: 12, c: C.dim });
  const dest = [[2, 1], [1, 0], [1, 2]], u = I.go >= 0 ? clamp((t - I.go) / .9, 0, 1) : 0;
  const tg = I.go >= 0 && t - I.go < 2 ? dest[I.out] : home, from = home;
  const ax = x0 + lerp(from[0], tg[0], ease(u)) * s + s / 2, ay = y0 + lerp(from[1], tg[1], ease(u)) * s + s / 2;
  cellArrow(c, x0 + 1.5 * s + s * .5, y0 + 1.5 * s - 30, 1, 26, alpha(C.gold, .9));
  agent(c, ax, ay, C.pink, 12);
  const lab = ['went right ✓', 'slipped up', 'slipped down'], tot = I.n.reduce((a, b) => a + b, 0);
  txt(c, 'action: always “move right”', 300, 30, { s: 12, b: 1, c: C.gold });
  lab.forEach((l, k) => {
    const y = 66 + k * 44; txt(c, l, 214, y, { s: 12, a: 'left', b: I.go >= 0 && I.out === k && t - I.go < 2 });
    rr(c, 214, y + 8, 160, 12, 4, C.panel); bar(c, 214, y + 20, 160 * (tot ? I.n[k] / tot : 0), 12, [C.teal, C.purple, C.purple][k]);
    txt(c, tot ? pct(I.n[k] / tot) : '–', 380, y, { s: 12, a: 'right', c: C.dim });
  });
  txt(c, `tries: ${tot} · the rule is 80% right, 10% up, 10% down`, 200, 236, { s: 12, c: C.dim });
};
TOUCH.add('p1_slippery');
CAP.p1_slippery = 'The robot is on ice and always tries to move right. Usually it does, but sometimes it slips up or down. The bars count what actually happened; they drift towards 80, 10 and 10 percent.';

// ───────── mdp: one frame is not Markov ─────────
A.p1_markov = (c, t) => {
  const ph = (t % 4) / 4, bx = k => 40 + 110 * (0.5 + 0.45 * Math.sin((t - k * .18) * 1.6)), by = k => 110 + 62 * Math.cos((t - k * .18) * 1.1);
  [[10, 'one frame', false], [205, 'last 3 frames', true]].forEach(([ox, lab, trail]) => {
    rr(c, ox, 30, 185, 160, 12, C.bg, C.line, 1.5); txt(c, lab, ox + 92, 18, { s: 12, b: 1, c: trail ? C.teal : C.red });
    rr(c, ox + 8, 100, 6, 40, 3, C.fg); rr(c, ox + 171, 70, 6, 40, 3, C.fg);
    if (trail) for (let k = 3; k >= 1; k--) dot(c, ox + bx(k * 1.5), by(k * 1.5), 6, alpha(C.gold, .18 * (4 - k)));
    dot(c, ox + bx(0), by(0), 7, C.gold);
    if (trail) { const dx = bx(0) - bx(1.5), dy = by(0) - by(1.5), L = Math.hypot(dx, dy) || 1; arrow(c, ox + bx(0), by(0), ox + bx(0) + dx / L * 28, by(0) + dy / L * 28, C.teal, 2.5); }
    else txt(c, '❓', ox + bx(0) + 18, by(0) - 16, { s: 16, al: .6 + .4 * Math.sin(t * 4) });
  });
  txt(c, 'one picture: is the ball coming or going? Unknown, so not Markov', 200, 212, { s: 12, c: C.red });
  txt(c, 'stack a few pictures: direction is visible, so the state is Markov again', 200, 234, { s: 12, c: C.teal });
};
CAP.p1_markov = 'Two views of a Pong game. On the left, one still picture: you cannot tell which way the gold ball moves. On the right, the faded trail of recent frames shows its direction.';

// ───────── policy: a lookup table ─────────
A.p1_lookup = (c, t) => {
  const path = gwPolicyPath(), V = gwVI().at(-1), pi = gwGreedy(V), n = path.length, T = n * 1.6 + 2, k = Math.min(n - 1, Math.floor((t % T) / 1.6));
  const o = { x0: 14, y0: 44, s: 40 };
  gwDraw(c, null, o);
  for (const [x, y] of gwCells()) { const [cx, cy] = gwC(x, y, o); cellArrow(c, cx, cy, pi[y * 5 + x], 16, alpha(C.dim, .5), 1.5); }
  const [ax, ay] = gwC(path[k][0], path[k][1], o); agent(c, ax, ay, C.pink, 10);
  txt(c, 'cheat sheet: state → action', 312, 26, { s: 12, b: 1, c: C.gold });
  const NM = ['↑ up', '→ right', '↓ down', '← left'];
  const rows = path.slice(0, -1);
  rows.forEach(([x, y], i) => {
    const yy = 46 + i * 24, on = i === k; if (yy > 236) return;
    rr(c, 236, yy - 10, 150, 21, 6, on ? alpha(C.pink, .3) : C.panel, on ? C.pink : null);
    txt(c, `square (${x + 1}, ${4 - y})`, 246, yy, { s: 12, a: 'left', b: on });
    txt(c, NM[pi[y * 5 + x]], 376, yy, { s: 12, a: 'right', b: on, c: on ? C.fg : C.dim });
  });
  if (k === n - 1) txt(c, '🏁 reached the goal', 312, 46 + rows.length * 24, { s: 12, b: 1, c: C.green });
};
CAP.p1_lookup = 'The robot walks the grid. At each square it looks up its row in the cheat sheet on the right, highlighted in pink, and does what it says. That table is the policy.';

// ───────── policy: deterministic ↔ stochastic (slider) ─────────
A.p1_stoch = (c, t, S) => {
  const rnd = S.ctl.rand ?? .3, V = gwVI().at(-1), pi = gwGreedy(V), o = { x0: 20, y0: 30, s: 48 };
  const probs = (x, y) => [0, 1, 2, 3].map(a => (a === pi[y * 5 + x] ? 1 - rnd : 0) + rnd / 4);
  gwDraw(c, null, o);
  for (const [x, y] of gwCells()) {
    const [cx, cy] = gwC(x, y, o), pr = probs(x, y);
    pr.forEach((p, a) => { if (p > .02) { const d = DIRS[a]; arrow(c, cx, cy, cx + d[0] * (6 + 14 * p), cy + d[1] * (6 + 14 * p), alpha(C.teal, .25 + .75 * p), 1 + 3 * p, 6); } });
  }
  if (!S.st) S.st = { r: mulberry(3), pos: GW.start.slice(), prev: GW.start.slice(), at: 0, ep: 0, goal: 0, steps: 0 };
  const W = S.st;
  if (t - W.at > .7) {
    W.at = t; W.prev = W.pos.slice();
    if (gwTerm(...W.pos) || W.steps > 30) { if (gwGoal(...W.pos)) W.goal++; if (gwTerm(...W.pos) || W.steps > 30) { W.ep++; W.pos = GW.start.slice(); W.prev = W.pos.slice(); W.steps = 0; } }
    else { const pr = probs(...W.pos); let u = W.r(), a = 0; while (a < 3 && u > pr[a]) { u -= pr[a]; a++; } const [nx, ny] = gwStep(W.pos[0], W.pos[1], a); W.pos = [nx, ny]; W.steps++; }
  }
  const u = ease(clamp((t - W.at) / .5, 0, 1)), [px, py] = gwC(lerp(W.prev[0], W.pos[0], u), lerp(W.prev[1], W.pos[1], u), o);
  agent(c, px, py, C.pink, 10);
  txt(c, rnd < .05 ? 'deterministic' : rnd > .95 ? 'fully random' : 'stochastic', 330, 50, { s: 15, b: 1, c: C.gold });
  txt(c, `best arrow: ${pct(1 - rnd + rnd / 4)}`, 330, 78, { s: 12 }); txt(c, `each other: ${pct(rnd / 4)}`, 330, 98, { s: 12, c: C.dim });
  txt(c, `walks: ${W.ep}`, 330, 140, { s: 12 }); txt(c, `steps this walk: ${W.steps}`, 330, 160, { s: 12, c: C.dim });
  txt(c, 'arrow thickness = chance of that move', 200, 238, { s: 11, c: C.dim });
};
CAP.p1_stoch = 'Each square has four arrows. A thicker arrow means a higher chance of taking that move. The robot rolls the dice at every step, so with more randomness its walks wander more.';

// ───────── values: tap a square to see its value ─────────
A.p1_valueTap = (c, t, S) => {
  const V = gwVI().at(-1), pi = gwGreedy(V), o = { x0: 16, y0: 34, s: 46 };
  if (!S.vt) S.vt = { cell: [0, 3], at: 0, lastTap: -99, r: mulberry(5) };
  const Z = S.vt;
  while (S.taps.length) { const q = S.taps.shift(); const x = Math.floor((q.x - o.x0) / o.s), y = Math.floor((q.y - o.y0) / o.s); if (x >= 0 && x < 5 && y >= 0 && y < 4 && !gwWall(x, y) && !gwTerm(x, y)) { Z.cell = [x, y]; Z.at = t; Z.lastTap = t; } }
  if (t - Z.at > 7 && t - Z.lastTap > 10) { const cs = gwCells(); Z.cell = cs[Math.floor(Z.r() * cs.length)]; Z.at = t; }
  gwDraw(c, (x, y) => heat(V[y * 5 + x]), o);
  for (const [x, y] of gwCells()) { const [cx, cy] = gwC(x, y, o); txt(c, V[y * 5 + x].toFixed(2), cx, cy, { s: 12, b: 1, c: '#fff' }); }
  const path = [Z.cell]; let [x, y] = Z.cell; while (!gwTerm(x, y) && path.length < 15) { [x, y] = gwStep(x, y, pi[y * 5 + x]); path.push([x, y]); }
  const [sx, sy] = gwC(...Z.cell, o); rr(c, sx - o.s / 2 + 2, sy - o.s / 2 + 2, o.s - 4, o.s - 4, 6, null, C.pink, 3);
  const f = clamp((t - Z.at - .5) / .9, 0, path.length - 1), q = lerpPath(path.map(p => gwC(p[0], p[1], o)), f);
  poly(c, path.map(p => gwC(p[0], p[1], o)), alpha(C.pink, .45), 3); agent(c, q[0], q[1], C.pink, 9);
  const n = path.length - 1, v = V[Z.cell[1] * 5 + Z.cell[0]];
  txt(c, 'value of the', 316, 60, { s: 12, c: C.dim }); txt(c, 'pink square', 316, 76, { s: 12, c: C.dim });
  txt(c, v.toFixed(2), 316, 104, { s: 26, b: 1, c: C.gold });
  txt(c, `goal is ${n} step${n > 1 ? 's' : ''} away`, 316, 138, { s: 12 }); txt(c, n > 1 ? `1 × 0.9${superN(n - 1)} = ${Math.pow(.9, n - 1).toFixed(2)}` : 'reward 1 next step', 316, 158, { s: 12, c: C.dim });
  txt(c, 'brighter = more valuable · 🔥 = −1', 200, 238, { s: 11, c: C.dim });
};
TOUCH.add('p1_valueTap');
CAP.p1_valueTap = 'Every square shows its value: the reward you can expect from there, with gamma 0.9. Bright squares are close to the flag. The pink robot walks from the chosen square to show why.';

// ───────── values: Q for each action in one square ─────────
A.p1_qcell = (c, t) => {
  const V = gwVI().at(-1), path = gwPolicyPath(), k = Math.floor(t / 3.2) % (path.length - 1), [x, y] = path[k], o = { x0: 14, y0: 60, s: 32 };
  gwDraw(c, (xx, yy) => xx === x && yy === y ? alpha(C.pink, .5) : null, o); txt(c, 'where we are', 94, 46, { s: 11, c: C.dim });
  const q = [0, 1, 2, 3].map(a => { const [nx, ny, r] = gwStep(x, y, a); return r + .9 * V[ny * 5 + nx]; }), best = q.indexOf(Math.max(...q));
  const cx = 290, cy = 120, R = 74, NM = ['up', 'right', 'down', 'left'];
  const tri = [[[-1, -1], [1, -1]], [[1, -1], [1, 1]], [[1, 1], [-1, 1]], [[-1, 1], [-1, -1]]];
  tri.forEach(([p1, p2], a) => {
    c.beginPath(); c.moveTo(cx, cy); c.lineTo(cx + p1[0] * R, cy + p1[1] * R); c.lineTo(cx + p2[0] * R, cy + p2[1] * R); c.closePath();
    c.fillStyle = heat(q[a]); c.fill(); c.strokeStyle = a === best ? C.gold : C.bg; c.lineWidth = a === best ? 3 : 2; c.stroke();
    const d = DIRS[a]; txt(c, NM[a], cx + d[0] * R * .62, cy + d[1] * R * .62 - 8, { s: 11, c: '#fff' });
    txt(c, q[a].toFixed(2), cx + d[0] * R * .62, cy + d[1] * R * .62 + 8, { s: 13, b: 1, c: '#fff' });
  });
  txt(c, `best move here: ${NM[best]} (Q = ${q[best].toFixed(2)})`, 290, 218, { s: 13, b: 1, c: C.gold });
  txt(c, 'one Q value per move: “how good if I go this way, then play well”', 200, 246, { s: 11, c: C.dim });
};
CAP.p1_qcell = 'The big square on the right is the pink square from the small map, split into four triangles, one per move. Each shows that move’s Q value. The gold outline marks the best move.';

// ───────── bellman: tap to do one backup sweep ─────────
A.p1_backupStep = (c, t, S) => {
  const N = 6, g = .9;
  if (!S.bs) S.bs = { V: new Array(N).fill(0), k: 0, at: 0, changed: -1, lastTap: -99 };
  const B = S.bs, conv = B.k >= N - 1;
  const sweep = () => { if (conv) { B.V.fill(0); B.k = 0; B.changed = -1; } else { const old = B.V.slice(); for (let i = 0; i < N - 1; i++) B.V[i] = (i === N - 2 ? 1 : 0) + (i === N - 2 ? 0 : g * old[i + 1]); B.k++; B.changed = N - 1 - B.k; } B.at = t; };
  while (S.taps.length) { S.taps.shift(); sweep(); B.lastTap = t; }
  if (t - B.at > (conv ? 4 : 2.8) && t - B.lastTap > 8) sweep();
  const s = 56, x0 = 28, y = 90;
  for (let i = 0; i < N; i++) {
    const x = x0 + i * s, goal = i === N - 1, fresh = i === B.changed && t - B.at < 1.5;
    rr(c, x + 3, y, s - 6, s - 6, 8, goal ? alpha(C.green, .4) : heat(B.V[i]), fresh ? C.gold : C.line, fresh ? 3 : 1);
    txt(c, goal ? '🏁' : B.V[i].toFixed(2), x + s / 2, y + (s - 6) / 2, { s: goal ? 20 : 14, b: 1, c: '#fff' });
    if (i < N - 1) arrow(c, x + s - 2, y + s + 6, x + 6, y + s + 6, alpha(C.gold, i === B.changed && t - B.at < 1.5 ? 1 : .25), 2, 6);
  }
  txt(c, '+1 for stepping onto the flag', x0 + (N - 1.5) * s, y - 14, { s: 11, c: C.green });
  txt(c, `sweeps done: ${B.k}`, 200, 30, { s: 15, b: 1, c: C.gold });
  txt(c, 'rule: my value = reward + 0.9 × value of the next square', 200, 190, { s: 12 });
  txt(c, conv ? 'settled! tap to start over' : 'tap to do one sweep: the news moves one square left', 200, 222, { s: 12, b: 1, c: C.pink });
};
TOUCH.add('p1_backupStep');
CAP.p1_backupStep = 'A corridor ending at a flag. All values start at 0. Each sweep applies the Bellman rule to every square, so the value spreads one square left each time: 1, then 0.9, then 0.81.';

// ───────── explore: favourite restaurant ─────────
const RESTO = [['🍕', 3], ['🍜', 2], ['🍣', 5], ['🌮', 3]];
A.p1_restaurant = (c, t) => {
  const D = 1.4, T = 15 * D + 3, day = Math.min(14, Math.floor((t % T) / D)), seq = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2];
  const tried = new Set(seq.slice(0, day + 1)); let joy = 0; for (let i = 0; i <= day; i++) joy += RESTO[seq[i]][1];
  RESTO.forEach(([e, q], i) => {
    const x = 22 + i * 94, on = seq[day] === i;
    rr(c, x, 46, 80, 96, 12, C.panel, on ? C.pink : C.line, on ? 3 : 1.2);
    txt(c, e, x + 40, 80, { s: 30 });
    txt(c, tried.has(i) ? '★'.repeat(q) : '? ? ?', x + 40, 120, { s: tried.has(i) ? 13 : 12, c: tried.has(i) ? C.gold : C.dim });
    if (on) txt(c, '👤', x + 40, 34, { s: 16 });
  });
  const msg = day < 5 ? 'Days 1–5: always the favourite pizza place (3 stars)' : day === 5 ? 'Day 6: try something new… sushi is 5 stars!' : 'From now on: sushi. Exploring once paid off';
  txt(c, `day ${day + 1}`, 200, 168, { s: 14, b: 1, c: C.gold });
  txt(c, msg, 200, 192, { s: 13, b: 1 });
  rr(c, 40, 214, 320, 12, 6, C.panel); bar(c, 40, 226, Math.min(320, joy * 5), 12, C.teal);
  txt(c, `total happiness ${joy}`, 200, 244, { s: 11, c: C.dim });
};
CAP.p1_restaurant = 'Four restaurants; question marks mean you have never tried that one. You eat at your favourite for five days, then risk a new place and find a 5-star one. The teal bar is total happiness.';

// ───────── shared bandit simulations (averaged over many runs) ─────────
const K_ARMS = 5, MU = [.2, .45, .8, .55, .3], RUNS = 160, STEPS = 400;
function simBandit(kind, eps) {
  return memo('p1b' + kind + eps, () => {
    const avg = new Float32Array(STEPS), reg = new Float32Array(STEPS), r = mulberry(99);
    for (let run = 0; run < RUNS; run++) {
      const N = new Array(K_ARMS).fill(0), Sm = new Array(K_ARMS).fill(0); let cum = 0;
      for (let s = 0; s < STEPS; s++) {
        let a;
        if (kind === 'ucb') { a = N.indexOf(0); if (a < 0) { let b = -1; for (let i = 0; i < K_ARMS; i++) { const v = Sm[i] / N[i] + Math.sqrt(.5 * Math.log(s + 1) / N[i]); if (v > b) { b = v; a = i; } } } }
        else if (r() < eps) a = Math.floor(r() * K_ARMS);
        else { let b = -1, ties = []; for (let i = 0; i < K_ARMS; i++) { const v = N[i] ? Sm[i] / N[i] : 0; if (v > b + 1e-9) { b = v; ties = [i]; } else if (Math.abs(v - b) < 1e-9) ties.push(i); } a = ties[Math.floor(r() * ties.length)]; }
        const rw = r() < MU[a] ? 1 : 0; N[a]++; Sm[a] += rw; avg[s] += rw / RUNS; cum += .8 - MU[a]; reg[s] += cum / RUNS;
      }
    }
    const sm = new Float32Array(STEPS); for (let s = 0; s < STEPS; s++) { let a = 0, n = 0; for (let j = Math.max(0, s - 15); j <= s; j++) { a += avg[j]; n++; } sm[s] = a / n; }
    return { avg: sm, reg };
  });
}

A.p1_epsLive = (c, t, S) => {
  const eps = Math.round((S.ctl.eps ?? .1) * 20) / 20, x0 = 40, x1 = 380, yb = 200, h = 150, X = s => x0 + (x1 - x0) * s / STEPS, Y = v => yb - v * h;
  axis(c, x0, yb, x1, yb - h - 6, 'pulls →'); txt(c, 'avg payout', x0 + 4, yb - h - 4, { s: 10, c: C.dim, a: 'left' });
  line(c, x0, Y(.8), x1, Y(.8), C.gold, 1.5, [5, 4]); txt(c, 'best arm 0.8', x1, Y(.8) - 9, { s: 10, c: C.gold, a: 'right' });
  if (S.lastEps !== eps) { S.lastEps = eps; S.e0 = t; }
  const g = simBandit('eps', 0).avg; poly(c, Array.from(g, (v, s) => [X(s), Y(v)]), alpha(C.dim, .7), 1.5, [3, 3]);
  txt(c, 'greedy (ε = 0)', X(STEPS) - 2, Y(g[STEPS - 1]) + 12, { s: 10, c: C.dim, a: 'right' });
  const d = simBandit('eps', eps).avg, n = Math.min(STEPS, Math.floor((t - S.e0) / 5 * STEPS));
  if (n > 1) poly(c, Array.from(d.slice(0, n), (v, s) => [X(s), Y(v)]), C.teal, 2.5);
  if (n > 1) dot(c, X(n - 1), Y(d[n - 1]), 4, C.teal);
  txt(c, `ε = ${eps.toFixed(2)}: final payout ≈ ${d[STEPS - 1].toFixed(2)} per pull`, 200, 22, { s: 13, b: 1, c: C.teal });
  txt(c, 'average over 160 runs of a 5-armed bandit', 200, 236, { s: 11, c: C.dim });
};
CAP.p1_epsLive = 'Average payout per pull as the agent learns. The teal curve uses your epsilon, the grey dashed one never explores, and the gold line is the best arm. A little exploring wins; a lot wastes pulls.';

// ───────── bandits: pull the arms yourself ─────────
const MINE = [.3, .7, .5];
A.p1_pullYourself = (c, t, S) => {
  if (!S.py) S.py = { r: mulberry(21), N: [0, 0, 0], W: [0, 0, 0], last: -1, lastT: -9, demo: false, lastTap: -99 };
  const P = S.py, x0 = 20, w = 112;
  const pull = (i, demo) => { const win = P.r() < MINE[i] ? 1 : 0; P.N[i]++; P.W[i] += win; P.last = i; P.win = win; P.lastT = t; P.demo = demo; };
  while (S.taps.length) { const q = S.taps.shift(); for (let i = 0; i < 3; i++) if (inBox(q, x0 + i * 124, 30, w, 150)) pull(i, false); P.lastTap = t; }
  if (t - P.lastTap > 8 && t - P.lastT > 2.2 && t > 4) pull(Math.floor(P.r() * 3), true);
  const tot = P.N.reduce((a, b) => a + b, 0), coins = P.W.reduce((a, b) => a + b, 0), reveal = tot >= 15;
  for (let i = 0; i < 3; i++) {
    const x = x0 + i * 124, on = P.last === i && t - P.lastT < 1;
    rr(c, x, 30, w, 150, 14, C.panel, on ? C.pink : C.line, on ? 3 : 1.5);
    txt(c, '🎰', x + w / 2, 62, { s: 30 }); txt(c, 'machine ' + 'ABC'[i], x + w / 2, 94, { s: 12, b: 1 });
    txt(c, `pulls ${P.N[i]} · wins ${P.W[i]}`, x + w / 2, 116, { s: 11, c: C.dim });
    const avg = P.N[i] ? P.W[i] / P.N[i] : 0; rr(c, x + 12, 132, w - 24, 10, 4, C.bg); bar(c, x + 12, 142, (w - 24) * avg, 10, C.teal);
    txt(c, P.N[i] ? 'your avg ' + avg.toFixed(2) : 'not tried', x + w / 2, 158, { s: 11, c: C.teal });
    if (reveal) txt(c, 'true ' + MINE[i], x + w / 2, 172, { s: 11, b: 1, c: C.gold });
    if (on) popup(c, P.win ? '+1 🪙' : 'nothing', x + w / 2, 40, (t - P.lastT), P.win ? C.green : C.dim);
  }
  txt(c, `coins ${coins} from ${tot} pulls`, 200, 202, { s: 14, b: 1 });
  txt(c, reveal ? 'revealed: B pays best. Did you find it, or keep testing?' : P.demo ? 'demo pulling at random · tap a machine to play' : 'tap a machine to pull it · the true payouts appear after 15 pulls', 200, 228, { s: 12, c: C.gold });
};
TOUCH.add('p1_pullYourself');
CAP.p1_pullYourself = 'Three slot machines with hidden payout rates. Each pull either pays a coin or not. Teal bars are your average so far. After 15 pulls the gold numbers reveal the true rates.';

// ───────── bandits: regret curves ─────────
A.p1_regret = (c, t, S, p = {}) => {
  const x0 = 40, x1 = 360, yb = 200, h = 160, n = Math.min(STEPS, Math.floor((t % 14) / 9 * STEPS)), X = s => x0 + (x1 - x0) * s / STEPS;
  const L = [['greedy', simBandit('eps', 0).reg, C.red], ['ε-greedy 0.1', simBandit('eps', .1).reg, C.teal], ['UCB', simBandit('ucb', 0).reg, C.gold]];
  const mx = Math.max(...L.map(l => l[1][STEPS - 1])), Y = v => yb - v / mx * h;
  axis(c, x0, yb, x1, yb - h - 6, 'pulls →'); txt(c, 'total regret', x0 + 4, yb - h - 2, { s: 10, c: C.dim, a: 'left' });
  L.forEach(([nm, r, col]) => {
    if (n < 2) return; const hi = p.focus && nm.toLowerCase().startsWith(p.focus);
    poly(c, Array.from(r.slice(0, n), (v, s) => [X(s), Y(v)]), col, hi ? 4 : 2.5);
    txt(c, nm, Math.min(X(n), 372), Y(r[n - 1]) - 9, { s: 11, b: 1, c: col, a: 'right' });
  });
  txt(c, 'regret = coins lost vs always pulling the best arm', 200, 236, { s: 11, c: C.dim });
};
CAP.p1_regret = 'Regret piles up every time the agent pulls a worse arm. Red never explores and keeps losing, teal explores at random, gold uses UCB. A flatter curve means the agent has found the best arm.';

// ───────── ε-greedy: roll the dice (slider) ─────────
A.p1_coin = (c, t, S) => {
  const eps = S.ctl.eps ?? .1;
  if (!S.cn) S.cn = { r: mulberry(4), k: -1, n: [0, 0], u: 0 };
  const Z = S.cn, k = Math.floor(t / 1.3);
  if (k !== Z.k) { Z.k = k; Z.u = Z.r(); Z.n[Z.u < eps ? 0 : 1]++; Z.arm = Z.u < eps ? Math.floor(Z.r() * 4) : 2; }
  const x0 = 30, x1 = 370, y = 70, X = v => lerp(x0, x1, v);
  rr(c, x0, y, (x1 - x0) * eps, 30, 6, alpha(C.gold, .6)); rr(c, X(eps), y, (x1 - x0) * (1 - eps), 30, 6, alpha(C.teal, .45));
  if (eps > .08) txt(c, 'explore', X(eps / 2), y + 15, { s: 11, b: 1, c: C.bg });
  txt(c, 'exploit', X((1 + eps) / 2), y + 15, { s: 12, b: 1, c: C.bg });
  const ux = X(Z.u), fall = clamp((t % 1.3) / .4, 0, 1);
  arrow(c, ux, y - 34 + 20 * ease(fall), ux, y - 2, C.pink, 2.5, 8); txt(c, 'dice: ' + Z.u.toFixed(2), ux, y - 42, { s: 11, b: 1, c: C.pink });
  const ex = Z.u < eps;
  for (let i = 0; i < 4; i++) { const x = 70 + i * 70, on = Z.arm === i; rr(c, x, 124, 54, 48, 10, C.panel, on ? (ex ? C.gold : C.teal) : C.line, on ? 3 : 1); txt(c, '🎰', x + 27, 142, { s: 18 }); txt(c, i === 2 ? 'best so far' : 'arm ' + (i + 1), x + 27, 163, { s: 10, c: i === 2 ? C.teal : C.dim }); }
  txt(c, ex ? 'below ε → pick ANY arm at random' : 'above ε → pick the best-looking arm', 200, 196, { s: 13, b: 1, c: ex ? C.gold : C.teal });
  const tot = Z.n[0] + Z.n[1];
  txt(c, `explored ${Z.n[0]} of ${tot} times (${pct(tot ? Z.n[0] / tot : 0)})`, 200, 224, { s: 12 });
  txt(c, 'each pull: roll a number between 0 and 1', 200, 244, { s: 11, c: C.dim });
};
CAP.p1_coin = 'Before every pull the agent rolls a random number, the pink arrow. If it lands in the gold zone, smaller than epsilon, it picks any machine at random. Otherwise it picks the best one so far.';

// ───────── ε-greedy: decaying ε ─────────
A.p1_decay = (c, t) => {
  const N = 80, x0 = 40, x1 = 380, yb = 150, h = 110, X = k => x0 + (x1 - x0) * k / N, epsF = k => Math.max(.05, Math.pow(.95, k)), n = Math.min(N, Math.floor((t % 14) / 10 * N)), r = mulberry(8);
  axis(c, x0, yb, x1, yb - h - 6, 'pulls →'); txt(c, 'ε', x0 - 12, yb - h, { s: 14, b: 1, c: C.gold });
  poly(c, Array.from({ length: n + 1 }, (_, k) => [X(k), yb - h * epsF(k)]), C.gold, 2.5);
  for (let k = 0; k < N; k++) { const ex = r() < epsF(k); if (k < n) dot(c, X(k) + 2, 182 + (ex ? 0 : 22), 3.5, ex ? C.gold : C.teal); }
  txt(c, 'explore', x0 - 4, 182, { s: 10, c: C.gold, a: 'right' }); txt(c, 'exploit', x0 - 4, 204, { s: 10, c: C.teal, a: 'right' });
  if (n > 0) dot(c, X(n), yb - h * epsF(n), 5, C.pink);
  txt(c, n < N / 3 ? 'early: ε is big, lots of exploring' : n < N * .8 ? 'ε shrinks as the estimates get better' : 'late: ε stays small, mostly exploiting', 200, 22, { s: 13, b: 1 });
  txt(c, 'each dot is one pull', 200, 238, { s: 11, c: C.dim });
};
CAP.p1_decay = 'The gold curve is epsilon shrinking over time. Each dot below is one pull: gold dots explore, teal dots exploit. Early on there are many gold dots, later almost none.';

// ───────── UCB: estimate + bonus, tap to pull ─────────
const UMU = [.4, .65, .5];
A.p1_ucbBonus = (c, t, S) => {
  if (!S.ub) S.ub = { r: mulberry(31), N: [1, 1, 1], W: [0, 1, 0], last: -1, at: 0, lastTap: -99, who: '' };
  const U = S.ub, tot = U.N.reduce((a, b) => a + b, 0);
  const score = i => U.W[i] / U.N[i] + Math.sqrt(2 * Math.log(tot) / U.N[i]);
  const pull = (i, who) => { U.N[i]++; U.W[i] += U.r() < UMU[i] ? 1 : 0; U.last = i; U.at = t; U.who = who; };
  while (S.taps.length) { const q = S.taps.shift(); for (let i = 0; i < 3; i++) if (inBox(q, 40 + i * 120, 20, 90, 200)) pull(i, 'you'); U.lastTap = t; }
  if (t - U.at > 1.8 && t - U.lastTap > 3) { let b = -1, a = 0; for (let i = 0; i < 3; i++) if (score(i) > b) { b = score(i); a = i; } pull(a, 'UCB'); }
  if (tot > 120) { U.N = [1, 1, 1]; U.W = [0, 1, 0]; }
  const yb = 190, sc = 85;
  let best = 0; for (let i = 1; i < 3; i++) if (score(i) > score(best)) best = i;
  for (let i = 0; i < 3; i++) {
    const x = 40 + i * 120, est = U.W[i] / U.N[i], sco = score(i), on = U.last === i && t - U.at < .8;
    rr(c, x, 20, 90, 200, 12, C.panel, on ? C.pink : i === best ? alpha(C.gold, .7) : C.line, on ? 3 : 1.5);
    bar(c, x + 25, yb, 40, est * sc, alpha(C.teal, .85));
    rr(c, x + 25, yb - Math.min(1.9, sco) * sc, 40, (Math.min(1.9, sco) - est) * sc, 3, alpha(C.gold, .3), C.gold, 1.5);
    txt(c, 'score ' + sco.toFixed(2), x + 45, 34, { s: 11, b: 1, c: i === best ? C.gold : C.fg });
    txt(c, 'arm ' + (i + 1), x + 45, 204, { s: 11, b: 1 }); txt(c, `pulled ${U.N[i]}×`, x + 45, 214, { s: 9, c: C.dim });
  }
  txt(c, 'teal = estimate · gold = bonus · highest total gets pulled', 200, 236, { s: 11, c: C.dim });
  txt(c, U.who === 'you' ? 'you pulled · tap any arm' : 'UCB picks the top score · tap an arm to pull it yourself', 200, 250, { s: 11, c: C.gold });
};
TOUCH.add('p1_ucbBonus');
CAP.p1_ucbBonus = 'Each arm’s score is a teal estimate plus a gold bonus for uncertainty. The arm with the highest total gets pulled. Watch the gold part shrink on arms that are pulled often.';

// ───────── Thompson: sample, pick, update ─────────
const TMU = [.35, .6, .5];
function tsRounds() {
  return memo('p1ts', () => {
    const r = mulberry(17), a = [1, 1, 1], b = [1, 1, 1], R = [];
    for (let k = 0; k < 60; k++) {
      const s = [0, 1, 2].map(i => betaSample(r, a[i], b[i])), pick = s.indexOf(Math.max(...s)), win = r() < TMU[pick];
      R.push({ a: a.slice(), b: b.slice(), s, pick, win }); win ? a[pick]++ : b[pick]++;
    }
    return R;
  });
}
A.p1_tsSample = (c, t) => {
  const R = tsRounds(), P = 2.6, k = Math.floor(t / P) % R.length, u = (t % P) / P, Rd = R[k], x0 = 80, x1 = 370;
  for (let i = 0; i < 3; i++) {
    const yb = 72 + i * 62;
    line(c, x0, yb, x1, yb, C.line, 1); txt(c, 'arm ' + (i + 1), 14, yb - 14, { s: 12, b: 1, a: 'left', c: i === Rd.pick && u > .4 ? C.pink : C.fg });
    txt(c, `${Rd.a[i] - 1}W ${Rd.b[i] - 1}L`, 14, yb + 2, { s: 10, a: 'left', c: C.dim });
    drawBelief(c, Rd.a[i], Rd.b[i], x0, x1, yb, 44, C.purple);
    if (u > .1) { const sx = lerp(x0, x1, Rd.s[i]), d = clamp((u - .1) / .25, 0, 1); line(c, sx, yb - 50 + 50 * ease(d) - 4, sx, yb, alpha(C.pink, .9), 2); dot(c, sx, yb - 50 * (1 - ease(d)), 4.5, i === Rd.pick && u > .4 ? C.gold : C.pink); }
  }
  txt(c, '0', x0, 262 - 14, { s: 10, c: C.dim }); txt(c, 'payout rate →', x1, 248, { s: 10, c: C.dim, a: 'right' });
  const msg = u < .4 ? '1 · draw one random guess from each belief (pink pins)' : u < .7 ? `2 · highest guess wins: pull arm ${Rd.pick + 1}` : `3 · result: ${Rd.win ? 'win, so belief shifts right' : 'loss, so belief shifts left'}`;
  txt(c, msg, 200, 18, { s: 12, b: 1, c: u < .4 ? C.pink : u < .7 ? C.gold : Rd.win ? C.green : C.red });
  txt(c, `round ${k + 1}`, 392, 36, { s: 10, c: C.dim, a: 'right' });
};
CAP.p1_tsSample = 'Purple hills are the agent’s beliefs about each arm’s payout rate. Each round it draws one random guess from every hill, the pink pins, pulls the arm with the highest guess and updates that hill.';

// ───────── Thompson: add evidence, the belief narrows (tap) ─────────
A.p1_betaTap = (c, t, S) => {
  if (!S.bt) S.bt = { w: 0, l: 0, r: mulberry(2), at: 0, lastTap: -99, flash: '' };
  const B = S.bt, x0 = 30, x1 = 370, yb = 160;
  while (S.taps.length) {
    const q = S.taps.shift(); B.lastTap = t;
    if (inBox(q, 40, 196, 140, 44)) { B.w++; B.flash = 'w'; B.at = t; }
    else if (inBox(q, 220, 196, 140, 44)) { B.l++; B.flash = 'l'; B.at = t; }
    else { B.w = 0; B.l = 0; B.flash = ''; }
  }
  if (t - B.lastTap > 7 && t - B.at > 1.6) { if (B.w + B.l >= 40) { B.w = 0; B.l = 0; } else if (B.r() < .7) { B.w++; B.flash = 'w'; } else { B.l++; B.flash = 'l'; } B.at = t; }
  line(c, x0, yb, x1, yb, C.line, 1.5);
  [0, .25, .5, .75, 1].forEach(v => txt(c, String(v), lerp(x0, x1, v), yb + 12, { s: 10, c: C.dim }));
  drawBelief(c, B.w + 1, B.l + 1, x0, x1, yb, 110, C.purple);
  const n = B.w + B.l, m = (B.w + 1) / (n + 2), sd = Math.sqrt(m * (1 - m) / (n + 3)), lo = clamp(m - 1.64 * sd, 0, 1), hi = clamp(m + 1.64 * sd, 0, 1);
  line(c, lerp(x0, x1, lo), yb + 24, lerp(x0, x1, hi), yb + 24, C.gold, 3);
  txt(c, `likely between ${lo.toFixed(2)} and ${hi.toFixed(2)}`, 200, 22, { s: 13, b: 1, c: C.gold });
  txt(c, `wins ${B.w} · losses ${B.l}`, 200, 40, { s: 12, c: C.dim });
  const fl = t - B.at < .5;
  rr(c, 40, 196, 140, 44, 12, fl && B.flash === 'w' ? alpha(C.green, .5) : C.panel, C.green, 2); txt(c, '👍 add a win', 110, 218, { s: 13, b: 1 });
  rr(c, 220, 196, 140, 44, 12, fl && B.flash === 'l' ? alpha(C.red, .5) : C.panel, C.red, 2); txt(c, '👎 add a loss', 290, 218, { s: 13, b: 1 });
  txt(c, t - B.lastTap > 7 ? 'demo: results from an arm that pays 0.7' : 'tap elsewhere to reset', 200, 252, { s: 10, c: C.dim });
};
TOUCH.add('p1_betaTap');
CAP.p1_betaTap = 'The purple hill is the belief about one arm’s payout rate, between 0 and 1. Every win pushes it right, every loss left, and more results make it narrower. The gold line is the likely range.';
}
