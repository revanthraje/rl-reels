// Part 4 simulations: world models, rewards, learning from people, bigger worlds.
// Loaded after anim.js (uses its helpers and A). Names are prefixed p4_.
'use strict';
(() => {
const CAP = RLAnim.CAP, TOUCH = RLAnim.TOUCH;
const dtOf = (S, t) => { const d = clamp(t - (S.lt ?? t), 0, .05); S.lt = t; return d; };
const cellOf = (q, ox, oy, cs, W, H) => { const x = Math.floor((q.x - ox) / cs), y = Math.floor((q.y - oy) / cs); return x >= 0 && y >= 0 && x < W && y < H ? [x, y] : null; };
function wrap(c, s, x, y, w, o = {}) { // simple word wrap, returns lines drawn
  c.font = `${o.b ? 700 : 500} ${o.s || 12}px system-ui,-apple-system,"Segoe UI",sans-serif`;
  const words = s.split(' '), lines = []; let L = '';
  for (const wd of words) { const tst = L ? L + ' ' + wd : wd; if (c.measureText(tst).width > w && L) { lines.push(L); L = wd; } else L = tst; }
  if (L) lines.push(L);
  lines.forEach((l, i) => txt(c, l, x, y + i * ((o.s || 12) + 4), o));
  return lines.length;
}
function grid(c, ox, oy, cs, W, H, fill) {
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) rr(c, ox + x * cs + 1, oy + y * cs + 1, cs - 2, cs - 2, 6, fill ? fill(x, y) : C.panel, C.line, 1);
}
const toward = (p, q) => { const dx = Math.sign(q[0] - p[0]), dy = Math.sign(q[1] - p[1]); return dx ? [p[0] + dx, p[1]] : [p[0], p[1] + dy]; };

// ───────── World models ─────────
A.p4_dream = (c, t) => {
  rr(c, 8, 34, 128, 190, 14, C.panel, C.line); txt(c, '🌍 real world', 72, 50, { s: 12, b: 1 });
  const real = Math.floor(t / 2);
  line(c, 22, 140, 122, 140, C.line, 3);
  txt(c, '🚗', 26 + (real % 5) * 22, 128, { s: 18 });
  txt(c, 'slow: 1 step', 72, 180, { s: 11, c: C.dim }); txt(c, 'every 2 seconds', 72, 194, { s: 11, c: C.dim });
  txt(c, 'real steps: ' + real, 72, 212, { s: 12, b: 1, c: C.teal });
  arrow(c, 140, 130, 168, 130, C.gold, 2.5); txt(c, 'learns', 154, 116, { s: 10, c: C.gold });
  rr(c, 172, 34, 220, 190, 14, alpha(C.purple, .08), C.purple, 1.5); txt(c, '💭 dream (world model)', 282, 50, { s: 12, b: 1, c: C.purple });
  const n = Math.floor(t * 4);
  for (let k = 11; k >= 0; k--) {
    const id = n - k; if (id < 0) continue; const r = mulberry(id * 13 + 7); let y = 130; const pts = [[190, 130]];
    for (let i = 1; i <= 10; i++) { y = clamp(y + gauss(r) * 9, 66, 196); pts.push([190 + i * 18, y]); }
    const fade = 1 - k / 12; c.globalAlpha = fade * (k === 0 ? 1 : .55); poly(c, pts, k === 0 ? C.pink : C.purple, k === 0 ? 2.5 : 1.5); c.globalAlpha = 1;
    if (k === 0) dot(c, pts[10][0], pts[10][1], 4, C.pink);
  }
  dot(c, 190, 130, 7, C.gold); txt(c, 'now', 190, 148, { s: 10, c: C.gold });
  txt(c, 'imagined steps: ' + n * 10, 282, 212, { s: 12, b: 1, c: C.pink });
  txt(c, 'Practice in the dream is free: about 20× more steps than real life', 200, 244, { s: 11, c: C.fg });
};
CAP.p4_dream = 'Left: the **real world**, where each step is slow. Right: the **dream**. From the gold dot (now), the model imagines many possible futures, drawn as purple lines, and the policy practises on all of them.';

A.p4_modelError = (c, t, S) => {
  const err = S.ctl?.err ?? .08, k = Math.floor(t * 1.5) % 26, r = mulberry(5);
  const X = i => 30 + i * 13.6, tru = i => 130 + Math.sin(i * .35) * 55;
  const T = [], D = []; let drift = 0;
  for (let i = 0; i <= 25; i++) { T.push([X(i), tru(i)]); if (i) drift = drift * 1.18 + gauss(r) * err * 18 + err * 6; D.push([X(i), clamp(tru(i) + drift, 20, 230)]); }
  txt(c, 'true future vs the model’s dream', 200, 16, { s: 12, b: 1 });
  poly(c, T.slice(0, k + 1), C.teal, 3); poly(c, D.slice(0, k + 1), C.pink, 2.5, [6, 4]);
  dot(c, T[k][0], T[k][1], 6, C.teal); dot(c, D[k][0], D[k][1], 6, C.pink);
  line(c, T[k][0], T[k][1], D[k][0], D[k][1], C.red, 2);
  const gap = Math.abs(T[k][1] - D[k][1]);
  txt(c, `steps imagined ahead: ${k}    gap: ${gap.toFixed(0)}`, 200, 244, { s: 12, b: 1, c: gap > 40 ? C.red : C.fg });
  rr(c, 12, 28, 120, 38, 8, alpha(C.bg, .8)); line(c, 18, 40, 38, 40, C.teal, 3); txt(c, 'reality', 44, 40, { s: 11, a: 'left' });
  line(c, 18, 56, 38, 56, C.pink, 2.5, [6, 4]); txt(c, 'dream', 44, 56, { s: 11, a: 'left' });
};
CAP.p4_modelError = 'The teal line is what really happens; the dashed pink line is what the world model **imagines**. A tiny mistake each step adds up, so the red gap grows the further ahead it dreams.';

// ───────── Shaping ─────────
const HC = memo('p4hc', () => { const r = mulberry(31), path = [[0, 3]]; let p = [0, 3];
  for (let i = 0; i < 24; i++) { const a = r() < .55 ? (r() < .5 ? 0 : 1) : Math.floor(r() * 4), d = [[0, -1], [1, 0], [0, 1], [-1, 0]][a];
    const q = [clamp(p[0] + d[0], 0, 6), clamp(p[1] + d[1], 0, 3)]; p = q; path.push(q); } return path; });
A.p4_hotCold = (c, t) => {
  const ox = 60, oy = 36, cs = 40, k = Math.floor(t / 1.1) % 24, u = (t / 1.1) % 1;
  grid(c, ox, oy, cs, 7, 4); txt(c, '🏁', ox + 6 * cs + 20, oy + 20, { s: 20 });
  const dist = p => Math.abs(6 - p[0]) + Math.abs(p[1]);
  let tot = 0; for (let i = 1; i <= k; i++) tot += dist(HC[i]) < dist(HC[i - 1]) ? .1 : dist(HC[i]) > dist(HC[i - 1]) ? -.1 : 0;
  poly(c, HC.slice(Math.max(0, k - 5), k + 1).map(p => [ox + p[0] * cs + 20, oy + p[1] * cs + 20]), alpha(C.blue, .6), 3);
  const p = HC[k], q = HC[Math.max(0, k - 1)]; dot(c, ox + p[0] * cs + 20, oy + p[1] * cs + 20, 11, C.pink);
  if (k) { const dd = dist(p) - dist(q); popup(c, dd < 0 ? '+0.1 warmer' : dd > 0 ? '−0.1 colder' : '0', ox + p[0] * cs + 20, oy + p[1] * cs + 4, u, dd < 0 ? C.green : dd > 0 ? C.red : C.dim); }
  txt(c, `step ${k}   ·   hint points so far: ${tot >= 0 ? '+' : ''}${tot.toFixed(1)}`, 200, 214, { s: 12, b: 1 });
  txt(c, 'A small hint every step: green when it gets closer to the flag', 200, 238, { s: 11, c: C.dim });
};
CAP.p4_hotCold = 'The pink dot wanders on a grid. Each time it gets closer to the flag it gets a small green **+0.1**, and each step away a red **minus 0.1**: a hot-and-cold game.';

A.p4_farm = (c, t, S) => {
  const ox = 60, oy = 34, cs = 40, W = 7, H = 4, goal = [6, 0];
  if (!S.ag) { S.ag = [0, 3]; S.coins = []; S.hint = 0; S.goals = 0; S.next = 0; S.touched = false; }
  for (let q; (q = S.taps.shift());) { const cl = cellOf(q, ox, oy, cs, W, H); if (!cl || (cl[0] === goal[0] && cl[1] === goal[1])) continue; S.touched = true;
    const i = S.coins.findIndex(k => k.x === cl[0] && k.y === cl[1]); if (i >= 0) S.coins.splice(i, 1); else if (S.coins.length < 3) S.coins.push({ x: cl[0], y: cl[1], back: 0 }); }
  if (!S.touched && t > 5 && !S.coins.length) S.coins.push({ x: 3, y: 2, back: 0 });
  if (t >= S.next) { S.next = t + .6;
    const live = S.coins.filter(k => t >= k.back);
    let tgt = goal;
    if (S.coins.length) { const near = (live.length ? live : S.coins).slice().sort((a, b) => Math.abs(a.x - S.ag[0]) + Math.abs(a.y - S.ag[1]) - Math.abs(b.x - S.ag[0]) - Math.abs(b.y - S.ag[1]))[0]; tgt = [near.x, near.y]; }
    if (S.ag[0] !== tgt[0] || S.ag[1] !== tgt[1]) S.ag = toward(S.ag, tgt);
    else if (S.coins.length) { const k = S.coins.find(k => k.x === tgt[0] && k.y === tgt[1]); if (k && t >= k.back) { k.back = t + 2; S.hint += .5; S.pop = { s: '+0.5 hint', t, x: tgt[0], y: tgt[1] }; } else S.ag = [clamp(S.ag[0] + (S.ag[0] > 0 ? -1 : 1), 0, 6), S.ag[1]]; }
    if (S.ag[0] === goal[0] && S.ag[1] === goal[1]) { S.goals++; S.pop = { s: '+1 goal!', t, x: 6, y: 0 }; S.ag = [0, 3]; } }
  grid(c, ox, oy, cs, W, H); txt(c, '🏁', ox + 6 * cs + 20, oy + 20, { s: 20 });
  for (const k of S.coins) txt(c, '🪙', ox + k.x * cs + 20, oy + k.y * cs + 21, { s: 20, al: t >= k.back ? 1 : .25 });
  dot(c, ox + S.ag[0] * cs + 20, oy + S.ag[1] * cs + 20, 11, C.pink);
  if (S.pop) popup(c, S.pop.s, ox + S.pop.x * cs + 20, oy + S.pop.y * cs + 4, (t - S.pop.t) / 1.2, S.pop.s.includes('goal') ? C.green : C.gold);
  txt(c, S.coins.length ? 'It farms the respawning hint coin and never finishes!' : 'No hint coins: it heads straight for the flag', 200, 16, { s: 12, b: 1, c: S.coins.length ? C.gold : C.green });
  txt(c, `hint points ${S.hint.toFixed(1)}   ·   goals reached ${S.goals}`, 200, 214, { s: 12, b: 1 });
  txt(c, 'tap a cell to add or remove a coin', 200, 238, { s: 11, c: C.dim });
};
TOUCH.add('p4_farm');
CAP.p4_farm = 'The pink agent should walk to the flag. Each gold **hint coin** pays a little and comes back after a moment. Once a coin exists, the agent circles it forever instead of finishing.';

// ───────── Hacking ─────────
A.p4_boat = (c, t) => {
  const cx = 200, cy = 132;
  c.beginPath(); c.ellipse(cx, cy, 170, 100, 0, 0, 7); c.fillStyle = alpha(C.blue, .18); c.fill();
  c.beginPath(); c.ellipse(cx, cy, 115, 52, 0, 0, 7); c.fillStyle = '#0b0e1d'; c.fill(); c.strokeStyle = C.line; c.lineWidth = 2; c.stroke();
  line(c, cx, cy - 100, cx, cy - 52, C.fg, 3, [5, 4]); txt(c, '🏁 finish', cx, cy - 108 + 0, { s: 11, b: 1 });
  const a = -Math.PI / 2 + t * .45, bx = cx + 142 * Math.cos(a), by = cy + 76 * Math.sin(a); txt(c, '🚤', bx, by, { s: 18 });
  const lc = [92, 182], la = t * 1.4, px = lc[0] + 20 * Math.cos(la), py = lc[1] + 12 * Math.sin(la);
  for (let i = 0; i < 3; i++) { const ta = i * 2.1, hit = ((la - ta) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI); const on = hit > 1.2; dot(c, lc[0] + 20 * Math.cos(ta), lc[1] + 12 * Math.sin(ta), 5, on ? C.gold : alpha(C.gold, .2)); }
  dot(c, px, py, 8, C.pink); txt(c, 'stuck in a lagoon', lc[0] + 4, lc[1] + 30, { s: 10, c: C.pink });
  const laps = Math.floor(t * .45 / (2 * Math.PI)), pinkPts = Math.floor(t * 1.4 / 2.1) * 10, bluePts = laps * 30 + Math.floor(t * .45 / 2.1) * 10;
  rr(c, 248, 168, 140, 54, 10, alpha(C.bg, .85), C.line);
  txt(c, `🚤 racer: ${bluePts} pts, ${laps} laps`, 318, 184, { s: 11, b: 1, c: C.blue });
  txt(c, `● hacker: ${pinkPts} pts, 0 laps`, 318, 206, { s: 11, b: 1, c: C.pink });
  txt(c, 'Points were meant to measure racing. The pink boat just collects points.', 200, 248, { s: 11, c: C.dim });
};
CAP.p4_boat = 'The speedboat races around the track as intended. The pink boat found three gold targets that keep coming back, and circles them forever. It scores **more points** without ever finishing a lap.';

A.p4_goodhart = (c, t, S) => {
  const push = S.ctl?.push ?? .3, X = u => 46 + u * 320, Y = v => 210 - v * 160;
  const proxy = u => .1 + .85 * Math.sqrt(u), truth = u => .1 + 1.6 * u * Math.exp(-u * u * 4.5) * 1.2;
  line(c, 46, 210, 370, 210, C.line, 2); line(c, 46, 210, 46, 40, C.line, 2);
  txt(c, 'how hard we optimise the number →', 208, 226, { s: 11, c: C.dim });
  const P = [], T = []; for (let i = 0; i <= 50; i++) { const u = i / 50; P.push([X(u), Y(proxy(u))]); T.push([X(u), Y(truth(u))]); }
  poly(c, P, C.gold, 3); poly(c, T, C.teal, 3);
  line(c, X(push), 40, X(push), 210, alpha(C.fg, .5), 1.5, [4, 4]);
  const pulse = 7 + Math.sin(t * 4) * 1.5; dot(c, X(push), Y(proxy(push)), pulse, C.gold); dot(c, X(push), Y(truth(push)), pulse, C.teal);
  txt(c, 'score (the measure)', 360, Y(proxy(1)) - 14, { s: 11, b: 1, c: C.gold, a: 'right' });
  txt(c, 'what we really want', 60, 30, { s: 11, b: 1, c: C.teal, a: 'left' });
  const gap = proxy(push) - truth(push);
  txt(c, gap > .45 ? 'Score keeps rising, the real goal has collapsed' : push < .35 ? 'Early on, both rise together' : 'They start to come apart…', 200, 250, { s: 12, b: 1, c: gap > .45 ? C.red : C.fg });
};
CAP.p4_goodhart = 'The gold line is the score the agent optimises; the teal line is what we really want. With a little optimisation both rise. Push harder and the score keeps climbing while the real goal falls.';

// ───────── Curiosity ─────────
const CNT = memo('p4cnt', () => { const W = 8, H = 5, r = mulberry(9), path = [[0, 4]], n = Array(W * H).fill(0); let p = [0, 4]; n[4 * W] = 1;
  for (let i = 0; i < 140; i++) { const nb = [[1, 0], [-1, 0], [0, 1], [0, -1]].map(d => [p[0] + d[0], p[1] + d[1]]).filter(q => q[0] >= 0 && q[1] >= 0 && q[0] < W && q[1] < H);
    nb.sort((a, b) => n[a[1] * W + a[0]] - n[b[1] * W + b[0]] || r() - .5); p = nb[0]; n[p[1] * W + p[0]]++; path.push(p); } return path; });
A.p4_counts = (c, t) => {
  const W = 8, H = 5, ox = 48, oy = 26, cs = 38, k = Math.floor(t * 1.2) % 120, n = Array(W * H).fill(0);
  for (let i = 0; i <= k; i++) n[CNT[i][1] * W + CNT[i][0]]++;
  grid(c, ox, oy, cs, W, H, (x, y) => { const v = n[y * W + x]; return v ? mix(C.panel, C.blue, Math.min(1, v / 5)) : alpha(C.gold, .35); });
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const v = n[y * W + x]; txt(c, v ? String(v) : '✨', ox + x * cs + 19, oy + y * cs + 19, { s: v ? 11 : 13, c: v ? C.dim : C.gold }); }
  const p = CNT[k]; dot(c, ox + p[0] * cs + 19, oy + p[1] * cs + 19, 10, C.pink);
  const v = n[p[1] * W + p[0]]; txt(c, `bonus here: 1/√${v} = ${(1 / Math.sqrt(v)).toFixed(2)}`, 200, 222, { s: 12, b: 1, c: C.gold });
  txt(c, 'numbers = visits · ✨ never visited = biggest bonus', 200, 244, { s: 11, c: C.dim });
};
CAP.p4_counts = 'Each cell shows how often the pink agent has been there. Gold sparkle cells are **unvisited** and pay the biggest bonus, so the agent keeps heading for them and slowly covers the whole map.';

A.p4_noisyTV = (c, t, S) => {
  const Z = 5, zx = i => 30 + i * 70;
  if (!S.sur) { S.sur = [1, 1, 1, 1, 1]; S.tv = true; S.zone = 0; S.next = 0; }
  for (let q; (q = S.taps.shift());) if (q.x > 300 && q.y < 120) S.tv = !S.tv;
  if (t >= S.next) { S.next = t + 1.1;
    S.sur[S.zone] *= .55; if (S.tv) S.sur[4] = 1;
    for (let i = 0; i < Z; i++) if (i !== S.zone) S.sur[i] = Math.min(1, S.sur[i] + .01);
    let best = 0; for (let i = 1; i < Z; i++) if (S.sur[i] > S.sur[best]) best = i;
    if (S.sur[best] < .05) best = Math.floor(mulberry(Math.floor(t))() * Z);
    S.zone = clamp(S.zone + Math.sign(best - S.zone), 0, 4); }
  rr(c, 20, 30, 360, 110, 12, C.panel, C.line);
  for (let i = 1; i < Z; i++) line(c, zx(i) - 5, 34, zx(i) - 5, 136, alpha(C.line, .8), 1, [3, 4]);
  rr(c, 316, 42, 56, 42, 6, '#000', C.fg, 2);
  if (S.tv) { const r = mulberry(Math.floor(t * 12)); for (let i = 0; i < 70; i++) { c.fillStyle = r() < .5 ? '#ddd' : '#555'; c.fillRect(320 + r() * 48, 46 + r() * 34, 3, 3); } }
  else txt(c, 'off', 344, 63, { s: 11, c: C.dim });
  txt(c, '📺 tap me', 344, 98, { s: 11, b: 1, c: C.gold });
  dot(c, zx(S.zone) + 30, 118, 10, C.pink);
  for (let i = 0; i < Z; i++) { bar(c, zx(i) + 14, 222, 34, S.sur[i] * 60, i === 4 && S.tv ? C.red : C.purple); }
  txt(c, 'surprise left in each area (curiosity bonus)', 200, 236, { s: 11, c: C.dim });
  txt(c, S.tv ? 'Static never gets less surprising: the agent is hooked on the TV' : 'TV off: surprise fades everywhere and the agent explores', 200, 14, { s: 12, b: 1, c: S.tv ? C.red : C.green });
};
TOUCH.add('p4_noisyTV');
CAP.p4_noisyTV = 'The room has five areas; the bars show how surprising each still is. Visiting an area makes it boring. The TV shows random static, which **stays surprising forever**, so the curious pink agent gets stuck there.';

// ───────── Imitation ─────────
function drivePath(noise, band, seed) { const r = mulberry(seed), P = []; let e = 0; // inside the band it corrects itself; outside, errors snowball
  for (let i = 0; i <= 60; i++) { P.push(e); const a = Math.abs(e); e += (a < band ? -e * .05 : Math.sign(e) * ((a - band) * .14 + .6)) + gauss(r) * noise * 9; } return P; }
A.p4_bcDrift = (c, t, S) => {
  const noise = S.ctl?.noise ?? .5, loop = Math.floor(t / 10), u = (t % 10) / 9, k = Math.min(60, Math.floor(u * 60)), P = drivePath(noise, 14, 3 + loop % 3);
  rr(c, 10, 60, 380, 140, 0, '#1b1f30'); line(c, 10, 60, 390, 60, C.fg, 2); line(c, 10, 200, 390, 200, C.fg, 2);
  c.fillStyle = alpha(C.teal, .18); c.fillRect(10, 130 - 14, 380, 28); line(c, 10, 130, 390, 130, alpha(C.teal, .6), 1.5, [8, 6]);
  txt(c, 'expert data only here', 384, 110, { s: 10, c: C.teal, a: 'right' });
  const pts = []; let crash = -1; for (let i = 0; i <= k; i++) { const y = 130 + P[i]; if (Math.abs(P[i]) > 68 && crash < 0) crash = i; pts.push([16 + i * 6.1, clamp(y, 62, 198)]); if (crash >= 0) break; }
  poly(c, pts, C.pink, 2.5); const h = pts[pts.length - 1]; txt(c, crash >= 0 ? '💥' : '🚗', h[0], h[1], { s: 16 });
  txt(c, 'Copying the expert: small slips push the car where no example exists', 200, 22, { s: 11, b: 1 });
  txt(c, crash >= 0 ? 'Off the road: it never saw how to recover from out here' : Math.abs(P[k]) < 14 ? 'Inside the band: it knows what to do' : 'Outside the band: guessing, so it drifts further', 200, 228, { s: 12, b: 1, c: crash >= 0 ? C.red : Math.abs(P[k]) < 14 ? C.green : C.gold });
};
CAP.p4_bcDrift = 'The teal band is where the expert drove, the only places with examples. The pink line is the copied driver. Inside the band it corrects itself; once a slip takes it outside, it has no examples and **drifts off the road**.';

A.p4_dagger = (c, t) => {
  const r = Math.floor(t / 6) % 4, u = (t % 6) / 5, band = 14 + r * 16, k = Math.min(60, Math.floor(u * 60)), P = drivePath(.6, band, 4), prev = r ? drivePath(.6, 14 + (r - 1) * 16, 4) : null;
  rr(c, 10, 60, 380, 140, 0, '#1b1f30'); line(c, 10, 60, 390, 60, C.fg, 2); line(c, 10, 200, 390, 200, C.fg, 2);
  c.fillStyle = alpha(C.teal, .18); c.fillRect(10, 130 - 14, 380, 28);
  if (r) { c.fillStyle = alpha(C.gold, .16); c.fillRect(10, 130 - band, 380, band - 14); c.fillRect(10, 130 + 14, 380, band - 14); }
  if (prev) for (let i = 4; i < 60; i += 7) if (Math.abs(prev[i]) > 14 && Math.abs(prev[i]) < 68) { const x = 16 + i * 6.1, y = 130 + prev[i]; arrow(c, x, y, x + 12, y - Math.sign(prev[i]) * 12, C.gold, 2, 6); }
  const pts = []; for (let i = 0; i <= k; i++) { pts.push([16 + i * 6.1, clamp(130 + P[i], 62, 198)]); if (Math.abs(P[i]) > 68) break; }
  poly(c, pts, C.pink, 2.5); const h = pts[pts.length - 1]; txt(c, '🚗', h[0], h[1], { s: 16 });
  txt(c, `round ${r + 1}: the expert labels the places the learner drifted to`, 200, 22, { s: 12, b: 1 });
  txt(c, 'teal = first demos   gold = expert corrections added', 200, 228, { s: 11, c: C.dim });
  txt(c, 'Each round the known area grows, so drifting stops', 200, 248, { s: 11, b: 1, c: C.gold });
};
CAP.p4_dagger = 'Each round, the learner drives and the expert adds gold arrows saying **what to do** where it ended up. The gold band of known situations grows, and the pink line stays on the road.';

// ───────── Inverse RL ─────────
A.p4_twoRoads = (c, t) => {
  const H = [40, 140], Wk = [360, 140], trip = Math.floor(t / 4), u = (t % 4) / 3.2;
  line(c, H[0], H[1], Wk[0], Wk[1], C.line, 10); line(c, H[0], H[1], Wk[0], Wk[1], '#1b1f30', 6);
  for (let i = 0; i < 6; i++) { const x = 70 + ((i * 50 + t * 12) % 260); rr(c, x, 136, 12, 8, 2, C.red); }
  txt(c, 'short road, traffic jams', 200, 162, { s: 11, c: C.red });
  const arc = f => [lerp(H[0], Wk[0], f), H[1] - Math.sin(f * Math.PI) * 90];
  const A2 = []; for (let i = 0; i <= 30; i++) A2.push(arc(i / 30)); poly(c, A2, C.line, 10); poly(c, A2, '#1b1f30', 6);
  txt(c, 'long road, empty', 200, 38, { s: 11, c: C.teal });
  txt(c, '🏠', H[0], H[1], { s: 22 }); txt(c, '🏢', Wk[0], Wk[1], { s: 22 });
  const p = arc(clamp(u, 0, 1)); dot(c, p[0], p[1], 8, C.pink);
  const w = 1 - Math.exp(-(trip + clamp(u, 0, 1)) / 2);
  txt(c, `trip ${trip + 1}: took the long road again`, 200, 186, { s: 12, b: 1 });
  txt(c, 'Our guess of what they care about:', 200, 206, { s: 11, c: C.dim });
  txt(c, 'hates traffic', 92, 226, { s: 11, a: 'right' }); rr(c, 98, 220, 100 * (.2 + .8 * w), 12, 4, C.teal);
  txt(c, 'minds time', 262, 226, { s: 11, a: 'right' }); rr(c, 268, 220, 100 * (.5 - .3 * w), 12, 4, C.gold);
};
CAP.p4_twoRoads = 'The pink dot is a friend driving from home to work. They always take the long empty road rather than the short jammed one. With each trip, our guess of their reward shifts: they **hate traffic** more than they mind time.';

A.p4_ambig = (c, t) => {
  const g = Math.floor(t / 3.5) % 3, ox = 30, oy = 60, cs = 46, W = 5, H = 3;
  const R = [(x, y) => x === 4 && y === 1 ? 1 : 0, (x, y) => x === 4 && y === 1 ? 0 : -.5, () => 0][g];
  grid(c, ox, oy, cs, W, H, (x, y) => heat(R(x, y)));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const v = R(x, y); txt(c, v === 0 ? '0' : (v > 0 ? '+' : '') + v, ox + x * cs + 23, oy + y * cs + 35, { s: 10, c: C.dim }); }
  for (let x = 0; x < 4; x++) arrow(c, ox + x * cs + 14, oy + cs + 18, ox + x * cs + cs + 6, oy + cs + 18, C.pink, 3, 8);
  txt(c, 'expert’s path', ox + 2.5 * cs, oy - 12, { s: 11, b: 1, c: C.pink });
  const names = ['+1 at the goal', '−0.5 for every step', 'everything is 0'];
  names.forEach((n, i) => { const y = 82 + i * 38; rr(c, 266, y - 15, 126, 30, 8, i === g ? alpha(C.gold, .25) : C.panel, i === g ? C.gold : C.line);
    txt(c, n, 329, y - 3, { s: 11, b: i === g ? 1 : 0 }); txt(c, i <= g ? 'fits ✓' : '', 329, y + 9, { s: 9, c: C.green }); });
  txt(c, 'Reward guesses', 329, 50, { s: 12, b: 1 });
  txt(c, `guess ${g + 1} of 3 also explains the same path`, 140, 220, { s: 12, b: 1, c: C.gold });
  txt(c, 'Many rewards fit one behaviour, so we need a tie-break rule', 200, 246, { s: 11, c: C.dim });
};
CAP.p4_ambig = 'The pink arrows are what the expert did. The grid colours show a guessed reward. Three very different guesses, even "everything is zero", all **explain the same path**.';

// ───────── Offline ─────────
const doseData = memo('p4dose', () => { const r = mulberry(4), D = []; for (let i = 0; i < 10; i++) { const d = i < 5 ? 1 : 2; D.push([d + (r() - .5) * .3, 0]); } return D; });
const trueQ = d => .9 - .14 * (d - 2.4) * (d - 2.4), naiveQ = d => .55 + .13 * d;
A.p4_dose = (c, t, S, p = {}) => {
  const cql = p.mode === 'cql', X = d => 46 + d * 52, Y = v => 150 - v * 90;
  line(c, 46, 150, 370, 150, C.line, 1.5); line(c, 46, 220, 46, 30, C.line, 2);
  for (let d = 0; d <= 6; d++) txt(c, String(d), X(d), 232, { s: 11, c: C.dim }); txt(c, 'dose (units)', 330, 246, { s: 11, c: C.dim });
  c.fillStyle = alpha(C.teal, .12); c.fillRect(X(.75), 30, X(2.25) - X(.75), 190); txt(c, 'logged data', X(1.5), 40, { s: 10, c: C.teal });
  for (const [d] of doseData) dot(c, X(d), Y(trueQ(d) + (d - 1.5) * .02), 4, C.teal);
  const T = [], N = [], K = [], caution = S.ctl?.caution ?? .5;
  const consQ = d => naiveQ(d) - caution * 1.1 * Math.max(0, Math.abs(d - 1.5) - .75) ** 1.5;
  for (let i = 0; i <= 60; i++) { const d = i / 10; T.push([X(d), Y(trueQ(d))]); N.push([X(d), Y(naiveQ(d))]); K.push([X(d), Y(consQ(d))]); }
  poly(c, T, C.fg, 2, [5, 4]); poly(c, N, C.pink, cql ? 1.5 : 3, cql ? [3, 3] : null); if (cql) poly(c, K, C.gold, 3);
  if (!cql) {
    const d = S.ctl?.dose ?? 5; line(c, X(d), 30, X(d), 220, alpha(C.fg, .4), 1, [3, 3]);
    dot(c, X(d), Y(naiveQ(d)), 7, C.pink); dot(c, X(d), Y(trueQ(d)), 7, C.fg);
    const out = d < .75 || d > 2.25;
    txt(c, `dose ${(+d).toFixed(1)}: model says ${naiveQ(d).toFixed(2)}, reality ${trueQ(d).toFixed(2)}`, 200, 14, { s: 12, b: 1, c: out ? C.red : C.green });
  } else {
    let bd = 0; for (let i = 0; i <= 60; i++) if (consQ(i / 10) > consQ(bd)) bd = i / 10;
    txt(c, '★', X(bd), Y(consQ(bd)) - 2, { s: 20, c: C.gold });
    txt(c, `picks dose ${bd.toFixed(1)} → really worth ${trueQ(bd).toFixed(2)}`, 200, 14, { s: 12, b: 1, c: trueQ(bd) > .5 ? C.green : C.red });
  }
  txt(c, '- - truth', 360, 60, { s: 11, a: 'right' }); txt(c, 'naive guess', 360, 76, { s: 11, c: C.pink, a: 'right' }); if (cql) txt(c, 'cautious guess', 360, 92, { s: 11, c: C.gold, a: 'right' });
};
CAP['p4_dose:naive'] = 'Teal dots are the logged doses, only 1 and 2 units. The pink line is the learner’s guess of how good each dose is; the dashed line is the truth. Far from the data, the guess is **made up** and wildly optimistic.';
CAP['p4_dose:cql'] = 'The gold line is a **cautious** guess: values far from the logged data are pushed down. The gold star is the dose it would pick. More caution pulls the choice back to where there is evidence.';

// ───────── RLHF ─────────
const ANS = [['helpful', 'Put it in a sunny window and water when the soil feels dry.'], ['vague', 'Plants need some care and attention.'], ['rude', 'Just look it up yourself.'], ['wrong', 'Water it ten times a day.']];
const PAIRS = [[0, 1], [2, 0], [3, 1], [0, 3], [1, 2], [3, 2]];
A.p4_prefer = (c, t, S) => {
  if (!S.sc) { S.sc = [0, 0, 0, 0]; S.i = 0; S.since = t; S.n = 0; S.pick = -1; S.at = 0; }
  const pr = PAIRS[S.i % PAIRS.length];
  const choose = (w, who) => { const l = pr[1 - w], wi = pr[w], pw = 1 / (1 + Math.exp(S.sc[l] - S.sc[wi])); S.sc[wi] += .8 * (1 - pw); S.sc[l] -= .8 * (1 - pw); S.n++; S.pick = w; S.at = t; S.who = who; };
  if (S.pick < 0) {
    for (let q; (q = S.taps.shift());) if (q.y > 36 && q.y < 128) { choose(q.x < 200 ? 0 : 1, 'you'); break; }
    if (S.pick < 0 && t - S.since > 5) choose(pr[0] < pr[1] ? 0 : 1, 'a sample person');
  } else { S.taps.length = 0; if (t - S.at > 1.6) { S.i++; S.pick = -1; S.since = t; } }
  txt(c, 'Q: How do I keep my basil plant alive?', 200, 16, { s: 12, b: 1 });
  for (let w = 0; w < 2; w++) { const x = 12 + w * 192, a = ANS[pr[w]], sel = S.pick === w;
    rr(c, x, 34, 184, 94, 12, sel ? alpha(C.green, .2) : C.panel, sel ? C.green : C.line, sel ? 3 : 1.5);
    txt(c, w ? 'B' : 'A', x + 14, 48, { s: 12, b: 1, c: C.gold }); wrap(c, a[1], x + 92, 66, 160, { s: 12 });
    if (sel) txt(c, '👍 preferred', x + 92, 116, { s: 11, b: 1, c: C.green }); }
  txt(c, S.pick >= 0 ? `${S.who} picked ${S.pick ? 'B' : 'A'} · comparisons: ${S.n}` : 'tap the better answer', 200, 142, { s: 11, c: S.pick >= 0 ? C.fg : C.gold, b: 1 });
  txt(c, 'Reward model’s learned scores', 200, 162, { s: 11, c: C.dim });
  ANS.forEach((a, i) => { const x = 30 + i * 92, v = S.sc[i]; line(c, x, 214, x + 70, 214, C.line, 1); bar(c, x + 18, 214, 34, v * 22, v >= 0 ? C.teal : C.red); txt(c, a[0], x + 35, 246, { s: 11, b: 1 }); });
};
TOUCH.add('p4_prefer');
CAP.p4_prefer = 'Two answers to the same question. Each time someone picks the better one, the **reward model** below updates its scores: answer styles that win go up, styles that lose go down.';

const KLS = [['helpful', .3, .6, .8], ['clear', .3, .5, .7], ['short', .15, .2, .35], ['vague', .15, .1, .2], ['flattering & long', .1, 1, .15]];
A.p4_kl = (c, t, S) => {
  const beta = S.ctl?.beta ?? .3, tg = KLS.map(k => k[1] * Math.exp(k[2] / beta)), Z = tg.reduce((a, b) => a + b);
  S.p ||= KLS.map(k => k[1]); S.p = S.p.map((v, i) => lerp(v, tg[i] / Z, .08));
  const rm = S.p.reduce((a, v, i) => a + v * KLS[i][2], 0), tq = S.p.reduce((a, v, i) => a + v * KLS[i][3], 0);
  KLS.forEach((k, i) => { const x = 22 + i * 74; rr(c, x + 8, 196 - k[1] * 160, 50, k[1] * 160, 4, null, C.dim, 1.5); bar(c, x + 16, 196, 34, S.p[i] * 160, i === 4 ? C.red : C.pink);
    wrap(c, k[0], x + 33, 210, 70, { s: 10, b: 1 }); });
  txt(c, 'grey outline = original model   pink = tuned model', 200, 16, { s: 11, c: C.dim });
  rr(c, 250, 30, 140, 46, 8, alpha(C.bg, .85), C.line);
  txt(c, `reward model: ${rm.toFixed(2)}`, 320, 44, { s: 11, b: 1, c: C.gold }); txt(c, `real quality: ${tq.toFixed(2)}`, 320, 62, { s: 11, b: 1, c: tq < .4 ? C.red : C.green });
  txt(c, beta < .2 ? 'Weak penalty: it games the reward model' : 'Penalty keeps it near the original', 120, 46, { s: 11, b: 1, c: beta < .2 ? C.red : C.green });
};
CAP.p4_kl = 'Bars show how often the model gives each style of answer. The reward model wrongly loves **flattering, long** answers. With a weak penalty the tuned model piles onto that style: the judge’s score rises while real quality drops.';

// ───────── POMDP ─────────
A.p4_corridor = (c, t) => {
  const u = (t % 11) / 9, x = 40 + clamp(u, 0, 1) * 320, seen = x > 258;
  for (let r = 0; r < 2; r++) { const y = 52 + r * 72; rr(c, 30, y, 340, 46, 8, C.panel, C.line); txt(c, 'corridor ' + 'AB'[r], 36, y - 8, { s: 11, b: 1, a: 'left' });
    for (let d = 0; d < 4; d++) rr(c, 70 + d * 80, y + 4, 18, 26, 3, r === 1 && d === 2 ? C.red : C.line); }
  txt(c, '🤖', x, 52 + 72 + 32, { s: 18 }); txt(c, 'really here (it doesn’t know)', 200, 186, { s: 10, c: C.dim });
  if (seen) txt(c, '👀 a red door!', x, 112, { s: 11, b: 1, c: C.red });
  const pB = seen ? lerp(.5, .9, clamp((x - 258) / 30, 0, 1)) : .5;
  txt(c, 'belief:', 60, 214, { s: 11, b: 1 });
  rr(c, 90, 206, 260 * (1 - pB), 16, 4, C.blue); rr(c, 90 + 260 * (1 - pB), 206, 260 * pB, 16, 4, C.gold);
  txt(c, `A ${Math.round(100 * (1 - pB))}%`, 90 + 130 * (1 - pB), 214, { s: 11, b: 1 }); txt(c, `B ${Math.round(100 * pB)}%`, 90 + 260 * (1 - pB) + 130 * pB, 214, { s: 11, b: 1 });
  txt(c, seen ? 'Only corridor B has a red door, so the belief jumps' : 'Both corridors look the same: 50/50', 200, 244, { s: 12, b: 1, c: seen ? C.gold : C.fg });
};
CAP.p4_corridor = 'Two corridors look identical; the robot is really in B but cannot tell. The bar at the bottom is its **belief**: 50/50 until it sees the red door that only B has.';

const FOG = { W: 9, H: 5, walls: new Set(['2,1', '2,2', '2,3', '5,0', '5,1', '5,3', '7,3', '6,3']), gold: [7, 1] };
A.p4_fog = (c, t, S) => {
  const ox = 47, oy = 34, cs = 34, W = FOG.W, H = FOG.H, wall = (x, y) => FOG.walls.has(x + ',' + y);
  if (!S.pos) { S.pos = [0, 4]; S.seen = new Set(); S.next = 0; S.tgt = null; S.found = 0; S.auto = 0; }
  for (let q; (q = S.taps.shift());) { const cl = cellOf(q, ox, oy, cs, W, H); if (cl && !wall(...cl)) { S.tgt = cl; S.user = true; } }
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) S.seen.add((S.pos[0] + dx) + ',' + (S.pos[1] + dy));
  if (S.found && t - S.found > 2) { S.pos = [0, 4]; S.seen = new Set(); S.found = 0; S.tgt = null; }
  if (!S.found && t >= S.next) { S.next = t + .45;
    if (!S.tgt && !S.user && t > 3) { const r = mulberry(Math.floor(t * 7)), cand = []; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (!wall(x, y) && !S.seen.has(x + ',' + y)) cand.push([x, y]); if (cand.length) S.tgt = cand[Math.floor(r() * cand.length)]; }
    if (S.tgt) { const opts = [[1, 0], [-1, 0], [0, 1], [0, -1]].map(d => [S.pos[0] + d[0], S.pos[1] + d[1]]).filter(q => q[0] >= 0 && q[1] >= 0 && q[0] < W && q[1] < H && !wall(...q));
      opts.sort((a, b) => Math.abs(a[0] - S.tgt[0]) + Math.abs(a[1] - S.tgt[1]) - Math.abs(b[0] - S.tgt[0]) - Math.abs(b[1] - S.tgt[1]) + (mulberry(Math.floor(t * 3))() - .5) * .5);
      if (opts.length) S.pos = opts[0]; if (S.pos[0] === S.tgt[0] && S.pos[1] === S.tgt[1]) S.tgt = null; }
    if (S.pos[0] === FOG.gold[0] && S.pos[1] === FOG.gold[1]) S.found = t; }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const near = Math.abs(x - S.pos[0]) <= 1 && Math.abs(y - S.pos[1]) <= 1, mem = S.seen.has(x + ',' + y), X = ox + x * cs, Y = oy + y * cs;
    if (!mem) { rr(c, X + 1, Y + 1, cs - 2, cs - 2, 5, '#05060c', '#141830', 1); txt(c, '?', X + cs / 2, Y + cs / 2, { s: 11, c: '#2a3060' }); continue; }
    rr(c, X + 1, Y + 1, cs - 2, cs - 2, 5, wall(x, y) ? C.wall : C.panel, C.line, 1); c.globalAlpha = near ? 1 : .45;
    if (wall(x, y)) rr(c, X + 4, Y + 4, cs - 8, cs - 8, 4, '#3a3f5c');
    if (x === FOG.gold[0] && y === FOG.gold[1]) txt(c, '💎', X + cs / 2, Y + cs / 2, { s: 17 });
    c.globalAlpha = 1; if (!near) rr(c, X + 1, Y + 1, cs - 2, cs - 2, 5, alpha('#000000', .35));
  }
  if (S.tgt) rr(c, ox + S.tgt[0] * cs + 4, oy + S.tgt[1] * cs + 4, cs - 8, cs - 8, 5, null, C.gold, 2);
  dot(c, ox + S.pos[0] * cs + cs / 2, oy + S.pos[1] * cs + cs / 2, 10, C.pink);
  txt(c, S.found ? 'Found the gem! 💎' : 'It only sees the 3×3 square around it', 200, 16, { s: 12, b: 1, c: S.found ? C.green : C.fg });
  txt(c, 'bright = seeing now · dim = remembered · ? = unknown', 200, 222, { s: 11, c: C.dim });
  txt(c, 'tap a cell to send the agent there', 200, 244, { s: 11, c: C.gold });
};
TOUCH.add('p4_fog');
CAP.p4_fog = 'The pink agent only sees the cells right around it. Dim cells are places it **remembers**, and question marks are still unknown. The gem is hidden somewhere, so it has to explore and remember.';

// ───────── Hierarchical ─────────
const MG = memo('p4mg', () => { const wps = [[0, 4], [3, 2], [6, 2], [9, 0]], path = [[0, 4]], seg = [0]; let p = [0, 4];
  for (let w = 1; w < wps.length; w++) while (p[0] !== wps[w][0] || p[1] !== wps[w][1]) { p = toward(p, wps[w]); path.push(p); seg.push(w - 1); }
  return { wps, path, seg }; });
A.p4_manager = (c, t) => {
  const ox = 20, oy = 34, cs = 36, { wps, path, seg } = MG, n = path.length, k = Math.floor(t / .4) % (n + 5), i = Math.min(n - 1, k), w = seg[i] + 1;
  grid(c, ox, oy, cs, 10, 5);
  line(c, ox + 3 * cs - 18, oy, ox + 3 * cs - 18, oy + 5 * cs, alpha(C.blue, .25), 3, [4, 4]); line(c, ox + 6 * cs + 18, oy, ox + 6 * cs + 18, oy + 5 * cs, alpha(C.blue, .25), 3, [4, 4]);
  poly(c, path.slice(0, i + 1).map(p => [ox + p[0] * cs + 18, oy + p[1] * cs + 18]), alpha(C.teal, .7), 3);
  for (let j = 1; j < wps.length; j++) txt(c, j < w ? '✓' : '🚩', ox + wps[j][0] * cs + 18, oy + wps[j][1] * cs + 16, { s: j === w ? 20 : 14, c: C.green, al: j > w ? .35 : 1 });
  const p = path[i]; dot(c, ox + p[0] * cs + 18, oy + p[1] * cs + 18, 10, C.pink);
  const g = wps[Math.min(w, wps.length - 1)]; rr(c, ox + g[0] * cs + 2, oy + g[1] * cs + 2, cs - 4, cs - 4, 6, null, C.gold, 2.5);
  txt(c, `👔 manager: “go to flag ${Math.min(w, 3)}”`, 110, 16, { s: 12, b: 1, c: C.gold });
  txt(c, `manager decisions: ${Math.min(w, 3)}`, 120, 232, { s: 12, b: 1, c: C.gold }); txt(c, `worker steps: ${i}`, 300, 232, { s: 12, b: 1, c: C.teal });
  txt(c, 'Few big decisions on top, many small steps below', 200, 252, { s: 11, c: C.dim });
};
CAP.p4_manager = 'The **manager** only picks the next flag (gold box). The pink **worker** takes the small steps to reach it. Count them: 3 manager decisions versus 13 worker steps.';

A.p4_options = (c, t) => {
  const opts = [['walk to door', C.teal, 0, 7], ['open door', C.gold, 7, 9], ['walk to goal', C.purple, 9, 16]], T = 16, X = s => 30 + s * 21.5, ph = (t * 1.6) % (T + 3);
  txt(c, 'Skills (options) the agent can call', 200, 16, { s: 12, b: 1 });
  opts.forEach(([n, col, s0, s1], i) => { const y = 44 + i * 44; txt(c, n, X(s0), y - 12, { s: 11, b: 1, c: col, a: 'left' });
    rr(c, X(s0), y - 4, X(Math.min(s1, Math.max(s0, ph))) - X(s0), 16, 5, alpha(col, .7)); rr(c, X(s0), y - 4, X(s1) - X(s0), 16, 5, null, col, 1.5);
    if (ph > s0) txt(c, '▶ start', X(s0) + 2, y + 20, { s: 9, c: C.dim, a: 'left' }); if (ph > s1) txt(c, '■ stop', X(s1) + 4, y + 20, { s: 9, c: C.dim, a: 'left' }); });
  txt(c, 'tiny actions:', 30, 178, { s: 11, a: 'left', c: C.dim });
  for (let s = 0; s < T && s < ph; s++) { const o = opts.find(o => s >= o[2] && s < o[3]); const sym = o[0] === 'open door' ? '✋' : '→'; txt(c, sym, X(s) + 10, 202, { s: 14, b: 1, c: o[1] }); }
  line(c, X(Math.min(ph, T)), 30, X(Math.min(ph, T)), 214, alpha(C.fg, .6), 1.5, [3, 3]);
  txt(c, 'The agent makes 3 choices; each skill runs many tiny actions', 200, 240, { s: 11, b: 1 });
};
CAP.p4_options = 'Each coloured bar is a reusable **skill**: it starts, runs on its own for a while, then stops when its job is done. The arrows below are the tiny actions each skill takes.';

// ───────── Multi-agent ─────────
A.p4_penalty = (c, t) => {
  const X = s => 40 + s * 16, Y = v => 196 - v * 140, n = 20, tt = (t * 1.5) % (n + 4), S = Math.min(n, tt);
  const kr = s => .5 + .4 * Math.sin(s * .55), st = s => .5 + .4 * Math.sin(s * .55 - 1.6);
  line(c, 40, 196, 370, 196, C.line, 2); line(c, 40, 56, 370, 56, alpha(C.line, .5), 1, [3, 3]);
  txt(c, 'chance of choosing RIGHT', 44, 44, { s: 11, c: C.dim, a: 'left' }); txt(c, 'time →', 350, 210, { s: 11, c: C.dim });
  const K = [], Sx = []; for (let s = 0; s <= S; s += .25) { K.push([X(s), Y(kr(s))]); Sx.push([X(s), Y(st(s))]); }
  poly(c, Sx, C.teal, 3); poly(c, K, C.pink, 3);
  dot(c, X(S), Y(st(S)), 6, C.teal); dot(c, X(S), Y(kr(S)), 6, C.pink);
  txt(c, '⚽ striker shoots right', 250, 30, { s: 11, b: 1, c: C.teal }); txt(c, '🧤 keeper dives right', 250, 16, { s: 11, b: 1, c: C.pink });
  const msg = kr(S) > st(S) ? 'Keeper caught on: striker switches side' : 'Striker found a gap: keeper adapts';
  txt(c, msg, 200, 238, { s: 12, b: 1 }); txt(c, 'Each one’s best move keeps changing because the other learns', 200, 254, { s: 10, c: C.dim });
};
CAP.p4_penalty = 'The teal line is how often the striker shoots right; the pink line is how often the keeper dives right. Each keeps adapting to the other, so the lines **chase each other** instead of settling.';

A.p4_ctde = (c, t) => {
  const train = (t % 12) < 6, ag = [0, 1, 2].map(i => [90 + i * 110 + 30 * Math.sin(t * .8 + i * 2), 170 + 25 * Math.cos(t * .7 + i)]);
  rr(c, 10, 110, 380, 110, 14, alpha(C.panel, .6), C.line);
  for (const [x, y] of ag) { c.beginPath(); c.arc(x, y, 34, 0, 7); c.fillStyle = alpha(C.blue, .12); c.fill(); c.strokeStyle = alpha(C.blue, .4); c.lineWidth = 1; c.stroke(); }
  const coach = [200, 52];
  if (train) { rr(c, 140, 30, 120, 44, 12, alpha(C.gold, .2), C.gold, 2); txt(c, '🧑‍🏫 coach sees all', 200, 52, { s: 12, b: 1, c: C.gold });
    ag.forEach(([x, y], i) => { line(c, coach[0], coach[1] + 22, x, y - 12, alpha(C.gold, .6), 1.5, [4, 4]); const u = ((t * .8 + i * .33) % 1); dot(c, lerp(x, coach[0], u), lerp(y - 12, coach[1] + 22, u), 3, C.gold); }); }
  else { rr(c, 140, 30, 120, 44, 12, null, alpha(C.dim, .5), 1.5); txt(c, 'no coach at test time', 200, 52, { s: 11, c: C.dim }); }
  ag.forEach(([x, y], i) => { dot(c, x, y, 11, [C.pink, C.teal, C.purple][i]); txt(c, 'agent ' + (i + 1), x, y + 22, { s: 10, b: 1 }); });
  txt(c, train ? 'TRAINING: a coach uses everyone’s view to give feedback' : 'PLAYING: each agent acts only on what it sees', 200, 244, { s: 12, b: 1, c: train ? C.gold : C.teal });
};
CAP.p4_ctde = 'Blue circles show what each agent can see. During **training**, a coach (gold) collects everyone’s information to give better feedback. At **test time** the coach is gone and each agent decides from its own view.';

// ───────── Sim-to-real ─────────
const slide = (u, d) => d * (1 - (1 - u) * (1 - u));
A.p4_gap = (c, t, S) => {
  const mu = S.ctl?.mu ?? .45, x0 = 50, tg = 290, d = 240, u = clamp((t % 5) / 3, 0, 1), dr = d * .3 / mu;
  for (let r = 0; r < 2; r++) { const y = 60 + r * 90; rr(c, 20, y - 22, 370, 44, 8, r ? '#2a2318' : alpha(C.blue, .15), C.line);
    txt(c, r ? `🌍 real floor (friction ${mu.toFixed(2)})` : '🖥️ simulator (friction 0.30)', 24, y - 32, { s: 11, b: 1, a: 'left', c: r ? C.gold : C.blue });
    line(c, tg, y - 22, tg, y + 22, C.green, 3); txt(c, '🎯', tg, y, { s: 16 });
    const x = x0 + slide(u, r ? dr : d); dot(c, Math.min(x, 385), y, 11, C.pink); }
  const miss = dr - d;
  txt(c, 'Same push learned in the simulator, used on the real floor', 200, 14, { s: 11, b: 1 });
  txt(c, Math.abs(miss) < 12 ? 'Real world matches: it lands on target' : `Misses by ${Math.abs(miss).toFixed(0)} px: ${miss > 0 ? 'overshoots (slippery)' : 'stops short (sticky)'}`, 200, 230, { s: 12, b: 1, c: Math.abs(miss) < 12 ? C.green : C.red });
};
CAP.p4_gap = 'Top: in the simulator, a puck pushed with the learned force stops on the target. Bottom: the same push on a real floor with **different friction** overshoots or stops short.';

const WORLDS = [.22, .35, .41, .27, .5, .16];
A.p4_dr = (c, t, S) => {
  const R = S.ctl?.range ?? .1, X = f => 30 + (f - .05) / .6 * 340, lo = .3 - R, hi = .3 + R;
  line(c, 30, 120, 370, 120, C.line, 2); for (let f = .1; f <= .61; f += .1) txt(c, f.toFixed(1), X(f), 136, { s: 10, c: C.dim });
  txt(c, 'friction', 200, 152, { s: 11, c: C.dim });
  rr(c, X(lo), 84, X(hi) - X(lo), 30, 6, alpha(C.blue, .3), C.blue, 1.5); txt(c, 'trained on these', (X(lo) + X(hi)) / 2, 74, { s: 11, b: 1, c: C.blue });
  const r = mulberry(Math.floor(t * 3)); for (let i = 0; i < 5; i++) dot(c, X(lo + r() * (hi - lo)), 92 + r() * 16, 2.5, C.fg);
  let ok = 0; WORLDS.forEach((f, i) => { const good = f >= lo - .03 && f <= hi + .03; ok += good; const x = X(f), y = 180 + (i % 2) * 26;
    line(c, x, 122, x, y - 10, alpha(good ? C.green : C.red, .5), 1, [2, 3]); txt(c, (good ? '✓ ' : '✗ ') + '🌍', x, y, { s: 12, b: 1, c: good ? C.green : C.red }); });
  txt(c, `Real-world tests passed: ${ok} / ${WORLDS.length}`, 200, 18, { s: 13, b: 1, c: ok === WORLDS.length ? C.green : C.fg });
  txt(c, R < .04 ? 'One fixed simulator: reality usually falls outside' : 'Wider randomisation covers more real worlds', 200, 38, { s: 11, c: C.dim });
};
CAP.p4_dr = 'The blue band is the range of friction the policy saw during training; the white dots are random training worlds. Each 🌍 below is a real-world test. Widen the band and **more real worlds** fall inside.';
})();
