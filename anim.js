// Code-generated looping animations ("gifs") for RL Reels.
// Every animation is a pure-ish draw function A[name](ctx, t, S, p):
//   ctx is pre-scaled to a 400 x 260 logical canvas, t = seconds since the slide became visible,
//   S = per-canvas scratch object, p = params from the slide definition.
'use strict';

const AW = 400, AH = 260;
const C = {
  bg: '#0e1122', panel: '#1a1f3a', line: '#2e3560', fg: '#eef0ff', dim: '#8a90b8',
  pink: '#ff5c8a', teal: '#3dd6c6', gold: '#ffc94d', blue: '#7b8cff', green: '#4ade80',
  red: '#f87171', purple: '#c084fc', wall: '#0a0c18'
};

// ---------- helpers ----------
function mulberry(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const lerp = (a, b, u) => a + (b - a) * u;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const ease = u => u < .5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
const MEMO = {};
function memo(key, fn) { return MEMO[key] || (MEMO[key] = fn()); }
function gauss(r) { let u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

function hex(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
function mix(a, b, u) { const x = hex(a), y = hex(b); u = clamp(u, 0, 1); return `rgb(${x.map((v, i) => Math.round(lerp(v, y[i], u))).join(',')})`; }
function alpha(h, a) { const x = hex(h); return `rgba(${x[0]},${x[1]},${x[2]},${a})`; }
function heat(v) { // v in roughly [-1, 1]
  if (v >= 0) return v < .5 ? mix(C.panel, C.teal, v * 2) : mix(C.teal, C.gold, (v - .5) * 2);
  return mix(C.panel, C.red, -v);
}
function rr(c, x, y, w, h, r, fill, stroke, lw) {
  c.beginPath(); c.roundRect(x, y, w, h, r);
  if (fill) { c.fillStyle = fill; c.fill(); }
  if (stroke) { c.strokeStyle = stroke; c.lineWidth = lw || 1.5; c.stroke(); }
}
function txt(c, s, x, y, o = {}) {
  c.font = `${o.b ? 700 : 500} ${o.s || 13}px system-ui,-apple-system,"Segoe UI",sans-serif`;
  c.fillStyle = o.c || C.fg; c.textAlign = o.a || 'center'; c.textBaseline = o.bl || 'middle';
  c.globalAlpha = o.al == null ? 1 : o.al; c.fillText(s, x, y); c.globalAlpha = 1;
}
function dot(c, x, y, r, col, stroke) { c.beginPath(); c.arc(x, y, r, 0, 7); c.fillStyle = col; c.fill(); if (stroke) { c.strokeStyle = stroke; c.lineWidth = 2; c.stroke(); } }
function line(c, x1, y1, x2, y2, col, lw = 2, dash) { c.beginPath(); c.setLineDash(dash || []); c.moveTo(x1, y1); c.lineTo(x2, y2); c.strokeStyle = col; c.lineWidth = lw; c.stroke(); c.setLineDash([]); }
function head(c, x, y, ang, col, size = 8) {
  c.beginPath(); c.moveTo(x, y);
  c.lineTo(x - size * Math.cos(ang - .45), y - size * Math.sin(ang - .45));
  c.lineTo(x - size * Math.cos(ang + .45), y - size * Math.sin(ang + .45));
  c.closePath(); c.fillStyle = col; c.fill();
}
function arrow(c, x1, y1, x2, y2, col, lw = 2, hs = 8, dash) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  line(c, x1, y1, x2 - Math.cos(a) * hs * .6, y2 - Math.sin(a) * hs * .6, col, lw, dash); head(c, x2, y2, a, col, hs);
}
function along(pts, f) { // point at fraction f along a polyline
  let L = 0; const seg = [];
  for (let i = 1; i < pts.length; i++) { const d = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); seg.push(d); L += d; }
  let d = clamp(f, 0, 1) * L;
  for (let i = 0; i < seg.length; i++) {
    if (d <= seg[i] || i === seg.length - 1) { const u = seg[i] ? d / seg[i] : 0; return [lerp(pts[i][0], pts[i + 1][0], u), lerp(pts[i][1], pts[i + 1][1], u)]; }
    d -= seg[i];
  }
}
function poly(c, pts, col, lw = 2, dash) { c.beginPath(); c.setLineDash(dash || []); pts.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.strokeStyle = col; c.lineWidth = lw; c.stroke(); c.setLineDash([]); }
function qbez(p0, p1, p2, u) { return [(1 - u) * (1 - u) * p0[0] + 2 * (1 - u) * u * p1[0] + u * u * p2[0], (1 - u) * (1 - u) * p0[1] + 2 * (1 - u) * u * p1[1] + u * u * p2[1]]; }
function curve(c, a, b, bend, col, r = 0, lw = 2) { // curved arrow between centers a and b, shortened by radius r
  const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy);
  const ctrl = [mx - dy / L * bend, my + dx / L * bend];
  const u0 = r / L, u1 = 1 - r / L; const pts = [];
  for (let i = 0; i <= 20; i++) pts.push(qbez(a, ctrl, b, lerp(u0, u1, i / 20)));
  poly(c, pts, col, lw);
  const e = pts[20], f = pts[19]; head(c, e[0], e[1], Math.atan2(e[1] - f[1], e[0] - f[0]), col, 9);
  return { ctrl, mid: qbez(a, ctrl, b, .5), at: u => qbez(a, ctrl, b, lerp(u0, u1, u)) };
}
function bar(c, x, y0, w, h, col) { c.fillStyle = col; if (h >= 0) c.fillRect(x, y0 - h, w, h); else c.fillRect(x, y0, w, -h); }
function popup(c, s, x, y, u, col) { if (u < 0 || u > 1) return; txt(c, s, x, y - u * 18, { s: 15, b: 1, c: col || C.gold, al: 1 - u * u }); }
function cellArrow(c, cx, cy, a, len, col, lw = 2.5) {
  const d = [[0, -1], [1, 0], [0, 1], [-1, 0]][a];
  arrow(c, cx - d[0] * len / 2, cy - d[1] * len / 2, cx + d[0] * len / 2, cy + d[1] * len / 2, col, lw, 7);
}

// ---------- shared gridworld (5 x 4, goal +1 top-right, pit -1 below it, one wall) ----------
const GW = { W: 5, H: 4, start: [0, 3], walls: new Set(['1,1']) };
const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]];
const gwGoal = (x, y) => x === 4 && y === 0, gwPit = (x, y) => x === 4 && y === 1;
const gwTerm = (x, y) => gwGoal(x, y) || gwPit(x, y);
const gwWall = (x, y) => GW.walls.has(x + ',' + y);
function gwStep(x, y, a) {
  let nx = x + DIRS[a][0], ny = y + DIRS[a][1];
  if (nx < 0 || ny < 0 || nx >= GW.W || ny >= GW.H || gwWall(nx, ny)) { nx = x; ny = y; }
  return [nx, ny, gwGoal(nx, ny) ? 1 : gwPit(nx, ny) ? -1 : 0];
}
function gwCells() { const out = []; for (let y = 0; y < GW.H; y++) for (let x = 0; x < GW.W; x++) if (!gwWall(x, y) && !gwTerm(x, y)) out.push([x, y]); return out; }
function gwVI(g = .9, K = 14) {
  return memo('vi' + g, () => {
    let V = new Float32Array(20); const snaps = [V.slice()];
    for (let k = 0; k < K; k++) {
      const N = new Float32Array(20);
      for (const [x, y] of gwCells()) { let best = -9; for (let a = 0; a < 4; a++) { const [nx, ny, r] = gwStep(x, y, a); best = Math.max(best, r + g * V[ny * 5 + nx]); } N[y * 5 + x] = best; }
      V = N; snaps.push(V.slice());
    }
    return snaps;
  });
}
function gwGreedy(V, g = .9) {
  const pi = new Int8Array(20).fill(-1);
  for (const [x, y] of gwCells()) { let best = -9, ba = 0; for (let a = 0; a < 4; a++) { const [nx, ny, r] = gwStep(x, y, a); const q = r + g * V[ny * 5 + nx]; if (q > best + 1e-6) { best = q; ba = a; } } pi[y * 5 + x] = ba; }
  return pi;
}
const GX = 80, GY = 34, GS = 48; // default grid placement
function gwDraw(c, fill, o = {}) {
  const x0 = o.x0 ?? GX, y0 = o.y0 ?? GY, s = o.s ?? GS;
  for (let y = 0; y < GW.H; y++) for (let x = 0; x < GW.W; x++) {
    const px = x0 + x * s, py = y0 + y * s;
    let col = gwWall(x, y) ? C.wall : gwGoal(x, y) ? alpha(C.green, .35) : gwPit(x, y) ? alpha(C.red, .35) : (fill && fill(x, y)) || C.panel;
    rr(c, px + 1.5, py + 1.5, s - 3, s - 3, 6, col);
    if (gwGoal(x, y)) txt(c, '🏁 +1', px + s / 2, py + s / 2, { s: 12, b: 1 });
    if (gwPit(x, y)) txt(c, '🔥 −1', px + s / 2, py + s / 2, { s: 12, b: 1 });
  }
}
const gwC = (x, y, o = {}) => [(o.x0 ?? GX) + x * (o.s ?? GS) + (o.s ?? GS) / 2, (o.y0 ?? GY) + y * (o.s ?? GS) + (o.s ?? GS) / 2];
function agent(c, x, y, col = C.pink, r = 11) { dot(c, x, y, r + 3, alpha(col, .25)); dot(c, x, y, r, col); txt(c, '🤖', x, y + 1, { s: r * 1.3 }); }
function pathFromActions(start, acts) { const p = [start]; let [x, y] = start; for (const a of acts) { [x, y] = gwStep(x, y, a); p.push([x, y]); } return p; }
function lerpPath(path, f) { // f in [0, len-1]
  const i = Math.min(Math.floor(f), path.length - 1), j = Math.min(i + 1, path.length - 1), u = ease(clamp(f - i, 0, 1));
  return [lerp(path[i][0], path[j][0], u), lerp(path[i][1], path[j][1], u)];
}
function gwPolicyPath() {
  return memo('gwpath', () => { const V = gwVI().at(-1), pi = gwGreedy(V); const p = [GW.start]; let [x, y] = GW.start; while (!gwTerm(x, y) && p.length < 20) { [x, y] = gwStep(x, y, pi[y * 5 + x]); p.push([x, y]); } return p; });
}

const A = {};

// ---------- Foundations ----------
A.loop = (c, t) => {
  const T = 3, u = (t % T) / T, step = Math.floor(t / T);
  rr(c, 40, 85, 110, 90, 14, C.panel, C.pink, 2); txt(c, '🤖', 95, 115, { s: 30 }); txt(c, 'Agent', 95, 155, { b: 1 });
  rr(c, 250, 85, 110, 90, 14, C.panel, C.teal, 2); txt(c, '🌍', 305, 115, { s: 30 }); txt(c, 'Environment', 305, 155, { b: 1 });
  const top = [[95, 85], [95, 45], [305, 45], [305, 85]], bot = [[305, 175], [305, 215], [95, 215], [95, 175]];
  poly(c, top, C.line, 3); head(c, 305, 85, Math.PI / 2, C.line, 10);
  poly(c, bot, C.line, 3); head(c, 95, 175, -Math.PI / 2, C.line, 10);
  txt(c, 'action  aₜ', 200, 30, { c: C.pink, b: 1 });
  txt(c, 'new state sₜ₊₁ , reward rₜ₊₁', 200, 232, { c: C.teal, b: 1 });
  if (u < .5) { const p = along(top, u * 2); dot(c, p[0], p[1], 8, C.pink); }
  else { const p = along(bot, (u - .5) * 2); dot(c, p[0], p[1], 8, C.teal); const q = along(bot, Math.max(0, (u - .5) * 2 - .08)); dot(c, q[0], q[1], 5, C.gold); }
  popup(c, step % 3 === 2 ? '+1 🎉' : '0', 95, 75, (u - .9) * 10);
  txt(c, 't = ' + step, 200, 130, { s: 22, b: 1, c: C.dim });
};

A.returns = (c, t) => {
  const R = [0, 0, 1, 0, -1, 0, 0, 5], dt = .7, T = R.length * dt + 2.5, tt = t % T;
  const n = Math.min(R.length, Math.floor(tt / dt) + 1); let G = 0;
  line(c, 25, 160, 300, 160, C.line, 2);
  for (let i = 0; i < R.length; i++) {
    const x = 30 + i * 34;
    txt(c, 't' + i, x + 12, 228, { s: 11, c: C.dim });
    if (i >= n) { rr(c, x, 150, 24, 20, 5, null, C.line); continue; }
    G += R[i]; const h = R[i] * 13, col = R[i] > 0 ? C.green : R[i] < 0 ? C.red : C.dim;
    if (R[i]) bar(c, x, 160, 24, h, col); else rr(c, x + 8, 156, 8, 8, 4, col);
    txt(c, (R[i] > 0 ? '+' : '') + R[i], x + 12, R[i] >= 0 ? 160 - h - 10 : 160 - h + 10, { b: 1, c: col });
  }
  const ax = 30 + (n - 1) * 34 + 12; txt(c, '🤖', ax, 30, { s: 20 }); line(c, ax, 42, ax, 140, alpha(C.pink, .4), 1, [3, 3]);
  rr(c, 320, 40, 60, 180, 10, C.panel, C.line);
  bar(c, 335, 200, 30, G * 25, G >= 0 ? C.gold : C.red);
  txt(c, 'Return', 350, 212, { s: 11, c: C.dim }); txt(c, 'G = ' + G, 350, 55, { b: 1, c: C.gold });
};

A.discount = (c, t) => {
  const gs = [.5, .9, .99], i = Math.floor(t / 3.5) % 3, u = ease(clamp((t % 3.5) / .8, 0, 1));
  const g = lerp(gs[(i + 2) % 3], gs[i], u);
  for (let k = 0; k < 20; k++) {
    const h = 140 * Math.pow(g, k);
    bar(c, 22 + k * 18, 200, 13, h, mix(C.teal, C.gold, Math.pow(g, k)));
  }
  line(c, 18, 200, 382, 200, C.line);
  txt(c, 'now', 28, 214, { s: 10, c: C.dim }); txt(c, '20 steps later', 360, 214, { s: 10, c: C.dim, a: 'right' });
  txt(c, 'γ = ' + g.toFixed(2), 200, 30, { s: 22, b: 1, c: C.gold });
  txt(c, `same +1 reward each step, valued γᵗ today`, 200, 236, { s: 12, c: C.dim });
  txt(c, `effective horizon ≈ 1/(1−γ) = ${Math.round(1 / (1 - g))} steps`, 200, 55, { s: 13 });
};

A.mdp = (c, t) => {
  const N = [[70, 175, 'Cool', '😎'], [200, 70, 'Warm', '🥵'], [330, 175, 'Overheated', '💥']];
  const P = [[.5, .5, 0], [.3, .4, .3], [1, 0, 0]], R = [2, 2, -10];
  const seq = memo('mdpseq', () => { const r = mulberry(7); let s = 0; const q = [0]; for (let i = 0; i < 60; i++) { const u = r(); let acc = 0, n = 0; for (; n < 3; n++) { acc += P[s][n]; if (u < acc) break; } s = Math.min(n, 2); q.push(s); } return q; });
  const dt = 1.4, k = Math.floor(t / dt) % 59, u = (t % dt) / dt, s = seq[k], s2 = seq[k + 1];
  const E = [[0, 1, 22, '.5'], [1, 0, 22, '.3'], [1, 2, 22, '.3'], [2, 0, 30, '1.0']];
  for (const [a, b, bend, lab] of E) { const cv = curve(c, N[a], N[b], bend, C.line, 32); txt(c, lab, cv.mid[0], cv.mid[1], { s: 12, b: 1, c: C.dim }); }
  for (const [n, lab] of [[0, '.5'], [1, '.4']]) { c.beginPath(); c.arc(N[n][0], N[n][1] - 38, 14, 0, 7); c.strokeStyle = C.line; c.lineWidth = 2; c.stroke(); txt(c, lab, N[n][0] + 24, N[n][1] - 48, { s: 12, b: 1, c: C.dim }); }
  N.forEach(([x, y, name, e], i) => { dot(c, x, y, 30, i === s ? alpha(C.pink, .3) : C.panel, i === s ? C.pink : C.line); txt(c, e, x, y - 4, { s: 20 }); txt(c, name, x, y + 46, { s: 12, b: 1 }); });
  // token
  let pos;
  if (s === s2) { const a = u * Math.PI * 2; pos = [N[s][0] + 14 * Math.sin(a), N[s][1] - 38 + 14 * Math.cos(a)]; if (u < .05 || u > .95) pos = [N[s][0], N[s][1]]; }
  else { const bend = s === 2 ? 30 : 22; const mx = (N[s][0] + N[s2][0]) / 2, my = (N[s][1] + N[s2][1]) / 2, dx = N[s2][0] - N[s][0], dy = N[s2][1] - N[s][1], L = Math.hypot(dx, dy); pos = qbez(N[s], [mx - dy / L * bend, my + dx / L * bend], N[s2], ease(Math.min(1, u * 1.6))); }
  dot(c, pos[0], pos[1], 7, C.gold);
  popup(c, (R[s2] > 0 ? '+' : '') + R[s2], N[s2][0] + 30, N[s2][1] - 10, (u - .65) * 3, R[s2] > 0 ? C.green : C.red);
  txt(c, 'action: drive fast 🏎️', 200, 18, { s: 12, c: C.dim });
};

A.gridPolicy = (c, t, S, p = {}) => {
  const V = gwVI().at(-1), pi = gwGreedy(V), path = gwPolicyPath();
  const mx = Math.max(...V);
  gwDraw(c, p.heat ? (x, y) => heat(V[y * 5 + x] / mx * .9) : null);
  for (const [x, y] of gwCells()) { const [cx, cy] = gwC(x, y); if (p.heat) txt(c, V[y * 5 + x].toFixed(2), cx, cy + 15, { s: 10, c: C.fg }); cellArrow(c, cx, cy - (p.heat ? 4 : 0), pi[y * 5 + x], 22, p.heat ? C.bg : alpha(C.fg, .8)); }
  const dt = .5, T = path.length * dt + 1.2, f = Math.min((t % T) / dt, path.length - 1);
  const [ax, ay] = lerpPath(path, f); const [px, py] = gwC(ax, ay); agent(c, px, py);
  if (f >= path.length - 1) popup(c, '+1', px, py - 20, (t % T - (path.length - 1) * dt) / 1.2, C.green);
  txt(c, p.heat ? 'colour = value V(s): how good it is to be here' : 'arrows = policy π(s): what to do in each square', 200, 242, { s: 12, c: C.dim });
};

A.backup = (c, t) => {
  // one-step lookahead tree: V(s) = max_a [ r + γ Σ p V(s') ]
  const T = 7, u = t % T;
  const root = [200, 40], acts = [[110, 115], [290, 115]], kids = [[60, 200, .8, 5], [160, 200, .2, -10], [250, 200, .5, 4], [330, 200, .5, 2]];
  const g = .9; const q = [.8 * (0 + g * 5) + .2 * (0 + g * -10), .5 * (1 + g * 4) + .5 * (1 + g * 2)];
  line(c, root[0], root[1], acts[0][0], acts[0][1], C.line, 2); line(c, root[0], root[1], acts[1][0], acts[1][1], C.line, 2);
  kids.forEach((k, i) => { const a = acts[i < 2 ? 0 : 1]; line(c, a[0], a[1], k[0], k[1], C.line, 2); txt(c, 'p=' + k[2], (a[0] + k[0]) / 2 + (i % 2 ? 14 : -14), (a[1] + k[1]) / 2, { s: 11, c: C.dim }); });
  kids.forEach((k, i) => { const on = u > .3; dot(c, k[0], k[1], 22, C.panel, on ? C.teal : C.line); txt(c, on ? 'V=' + k[3] : '?', k[0], k[1], { s: 12, b: 1 }); });
  txt(c, 'r = 0', acts[0][0] - 45, acts[0][1], { s: 11, c: C.dim }); txt(c, 'r = 1', acts[1][0] + 45, acts[1][1], { s: 11, c: C.dim });
  acts.forEach((a, i) => { const on = u > 2; dot(c, a[0], a[1], 9, on ? C.gold : C.dim); if (on) txt(c, 'Q=' + q[i].toFixed(1), a[0], a[1] - 20, { s: 12, b: 1, c: C.gold }); });
  // flowing dots up
  if (u > 1 && u < 2) kids.forEach((k, i) => { const a = acts[i < 2 ? 0 : 1], f = u - 1; dot(c, lerp(k[0], a[0], f), lerp(k[1], a[1], f), 4, C.teal); });
  if (u > 3 && u < 4) acts.forEach(a => dot(c, lerp(a[0], root[0], u - 3), lerp(a[1], root[1], u - 3), 4, C.gold));
  const best = q[1] > q[0] ? 1 : 0;
  dot(c, root[0], root[1], 24, C.panel, u > 4 ? C.pink : C.line);
  txt(c, u > 4 ? 'V=' + Math.max(...q).toFixed(1) : 'V(s)?', root[0], root[1], { s: 12, b: 1 });
  if (u > 4) { line(c, root[0], root[1] + 24, acts[best][0], acts[best][1] - 9, C.pink, 4); txt(c, 'take the best action', 300, 30, { s: 12, c: C.pink }); }
  txt(c, 'value now = reward + γ × value of where you land', 200, 246, { s: 12, c: C.dim });
};

// ---------- Planning ----------
A.valueIter = (c, t, S, p = {}) => {
  const snaps = gwVI(), K = snaps.length - 1, dt = .75, T = K * dt + 3, k = Math.min(Math.floor((t % T) / dt), K), V = snaps[k];
  gwDraw(c, (x, y) => heat(V[y * 5 + x]));
  for (const [x, y] of gwCells()) { const [cx, cy] = gwC(x, y); txt(c, V[y * 5 + x].toFixed(2), cx, cy, { s: 12, b: 1, c: V[y * 5 + x] > .4 ? C.bg : C.fg }); }
  if (k === K) { const pi = gwGreedy(V); for (const [x, y] of gwCells()) { const [cx, cy] = gwC(x, y); cellArrow(c, cx, cy + 14, pi[y * 5 + x], 14, C.bg, 2); } }
  txt(c, `sweep ${k}${k === K ? ' — converged! read off the arrows' : ''}`, 200, 18, { s: 13, b: 1, c: C.gold });
  txt(c, 'every square: V ← max over moves [ r + 0.9 × V(next) ]', 200, 246, { s: 12, c: C.dim });
};

A.policyIter = (c, t) => {
  const it = memo('pi', () => {
    const g = .9; let pi = new Int8Array(20).fill(3); const out = [];
    for (let i = 0; i < 8; i++) {
      const V = new Float32Array(20);
      for (let s = 0; s < 200; s++) for (const [x, y] of gwCells()) { const [nx, ny, r] = gwStep(x, y, pi[y * 5 + x]); V[y * 5 + x] = r + g * V[ny * 5 + nx]; }
      out.push({ pi: pi.slice(), V: V.slice() });
      const np = gwGreedy(V, g); for (const [x, y] of gwCells()) if (np[y * 5 + x] !== pi[y * 5 + x]) { const [nx, ny, r] = gwStep(x, y, pi[y * 5 + x]); const [mx, my, mr] = gwStep(x, y, np[y * 5 + x]); if (mr + g * V[my * 5 + mx] <= r + g * V[ny * 5 + nx] + 1e-6) np[y * 5 + x] = pi[y * 5 + x]; }
      let same = true; for (let s = 0; s < 20; s++) if (np[s] !== pi[s]) same = false;
      if (same) break; pi = np;
    }
    return out;
  });
  const ph = 1.6, T = it.length * 2 * ph + 1.5, tt = t % T, i = Math.min(Math.floor(tt / (2 * ph)), it.length - 1), improving = (tt % (2 * ph)) > ph && i < it.length - 1;
  const cur = it[i], nxt = it[Math.min(i + 1, it.length - 1)];
  const pi = improving ? nxt.pi : cur.pi;
  gwDraw(c, (x, y) => heat(cur.V[y * 5 + x]));
  for (const [x, y] of gwCells()) { const [cx, cy] = gwC(x, y), s = y * 5 + x, ch = improving && cur.pi[s] !== nxt.pi[s]; cellArrow(c, cx, cy, pi[s], 22, ch ? C.pink : alpha(C.fg, .8), ch ? 4 : 2.5); }
  txt(c, `round ${i + 1}: ${improving ? 'IMPROVE → act greedily w.r.t. V' : 'EVALUATE → how good is this policy?'}`, 200, 18, { s: 13, b: 1, c: improving ? C.pink : C.teal });
  txt(c, i === it.length - 1 ? 'nothing changes any more → optimal policy ✓' : 'pink arrows = changed by improvement', 200, 246, { s: 12, c: C.dim });
};

// ---------- Bandits ----------
function betaSample(r, a, b) { const g = k => { let s = 0; for (let i = 0; i < k; i++) s -= Math.log(1 - r()); return s; }; const x = g(a), y = g(b); return x / (x + y); }
function banditSim(mode) {
  return memo('band' + mode, () => {
    const mu = [.25, .45, .75, .5], r = mulberry(mode === 'ts' ? 11 : mode === 'ucb' ? 5 : 3), N = [0, 0, 0, 0], S = [0, 0, 0, 0], H = [], forced = [0, 1, 0, 0];
    for (let n = 0; n < 240; n++) {
      let a;
      if (mode === 'ts') { let best = -1; for (let i = 0; i < 4; i++) { const v = betaSample(r, 1 + S[i], 1 + N[i] - S[i]); if (v > best) { best = v; a = i; } } }
      else if (n < 4) a = n;
      else if (mode === 'eps' && r() < .1) a = Math.floor(r() * 4);
      else if (mode === 'ucb') { let best = -1; for (let i = 0; i < 4; i++) { const v = S[i] / N[i] + Math.sqrt(2 * Math.log(n) / N[i]); if (v > best) { best = v; a = i; } } }
      else { let best = -1; for (let i = 0; i < 4; i++) { const v = S[i] / N[i] + r() * 1e-6; if (v > best) { best = v; a = i; } } }
      const rew = (mode === 'greedy' || mode === 'eps') && n < 4 ? forced[n] : (r() < mu[a] ? 1 : 0);
      N[a]++; S[a] += rew; H.push({ a, rew, N: N.slice(), S: S.slice() });
    }
    return { mu, H };
  });
}
A.bandit = (c, t, S, p = {}) => {
  const mode = p.mode || 'eps', { mu, H } = banditSim(mode), T = 21;
  const tt = t % T, n = Math.min(239, Math.floor(tt < 4 ? tt * 2 : 8 + (tt - 4) * 15));
  const h = H[n], y0 = 205, sc = 130;
  let tot = 0, best = 0; for (let i = 0; i <= n; i++) { tot += H[i].rew; best += H[i].a === 2; }
  for (let i = 0; i < 4; i++) {
    const x = 28 + i * 90, w = 74, pulled = h.a === i && (tt % (tt < 4 ? .5 : 1 / 15)) < .3;
    rr(c, x, 50, w, 170, 12, C.panel, i === h.a ? C.pink : C.line, i === h.a ? 3 : 1.5);
    txt(c, '🎰 arm ' + (i + 1), x + w / 2, 36, { s: 12, b: 1 });
    const est = h.N[i] ? h.S[i] / h.N[i] : 0;
    if (mode === 'ts') {
      const a = 1 + h.S[i], b = 1 + h.N[i] - h.S[i]; const pts = []; let mx = -1e9; const lp = [];
      for (let k = 1; k < 40; k++) { const q = k / 40; const l = (a - 1) * Math.log(q) + (b - 1) * Math.log(1 - q); lp.push(l); mx = Math.max(mx, l); }
      lp.forEach((l, k) => pts.push([x + w / 2 + Math.exp(l - mx) * 30, y0 - (k + 1) / 40 * sc]));
      c.beginPath(); c.moveTo(x + w / 2, y0); pts.forEach(q => c.lineTo(q[0], q[1])); c.lineTo(x + w / 2, y0 - sc);
      pts.slice().reverse().forEach(q => c.lineTo(2 * (x + w / 2) - q[0], q[1])); c.closePath(); c.fillStyle = alpha(C.purple, .45); c.fill();
    } else bar(c, x + 14, y0, w - 28, est * sc, alpha(C.teal, .8));
    if (mode === 'ucb' && h.N[i]) { const bo = Math.sqrt(2 * Math.log(n + 1) / h.N[i]), top = Math.min(1.15, est + bo); line(c, x + w / 2, y0 - est * sc, x + w / 2, y0 - top * sc, C.gold, 3); line(c, x + w / 2 - 10, y0 - top * sc, x + w / 2 + 10, y0 - top * sc, C.gold, 3); }
    line(c, x + 6, y0 - mu[i] * sc, x + w - 6, y0 - mu[i] * sc, C.gold, 2, [4, 4]);
    txt(c, 'pulled ' + h.N[i] + '×', x + w / 2, 212, { s: 11, c: C.dim });
    if (pulled) txt(c, h.rew ? '+1 🪙' : '0', x + w / 2, 66, { s: 13, b: 1, c: h.rew ? C.green : C.dim });
  }
  txt(c, `pull ${n + 1}   ·   coins ${tot}   ·   best arm (3) chosen ${Math.round(100 * best / (n + 1))}%`, 200, 240, { s: 12, c: C.fg });
  txt(c, { eps: 'ε-greedy (ε = 0.1)', greedy: 'pure greedy', ucb: 'UCB: estimate + uncertainty bonus', ts: 'Thompson: sample from beliefs' }[mode] + '   - - true payout', 200, 14, { s: 12, c: C.gold });
};

// ---------- Model-free prediction ----------
function mcEpisodes(seed, n) {
  return memo('ep' + seed, () => {
    const V = gwVI().at(-1), pi = gwGreedy(V), r = mulberry(seed), eps = [];
    for (let e = 0; e < n; e++) {
      let [x, y] = GW.start; const path = [[x, y]], rew = [];
      while (!gwTerm(x, y) && path.length < 16) { const a = r() < .45 ? Math.floor(r() * 4) : pi[y * 5 + x]; const s = gwStep(x, y, a); x = s[0]; y = s[1]; path.push([x, y]); rew.push(s[2]); }
      eps.push({ path, rew });
    }
    return eps;
  });
}
A.mc = (c, t) => {
  const data = memo('mcdata', () => {
    const eps = mcEpisodes(21, 10), V = new Float32Array(20), cnt = new Float32Array(20), out = [];
    for (const ep of eps) { let G = 0; const Gs = []; for (let i = ep.rew.length - 1; i >= 0; i--) { G = ep.rew[i] + .9 * G; Gs[i] = G; } ep.path.slice(0, -1).forEach(([x, y], i) => { const s = y * 5 + x; cnt[s]++; V[s] += (Gs[i] - V[s]) / cnt[s]; }); out.push({ ...ep, G: Gs[0], V: V.slice() }); }
    return out;
  });
  const sd = .18, upd = 1.3; let tt = t % data.reduce((s, e) => s + e.path.length * sd + upd, 0), e = 0;
  while (tt > data[e].path.length * sd + upd) { tt -= data[e].path.length * sd + upd; e++; }
  const ep = data[e], prevV = e ? data[e - 1].V : new Float32Array(20), walking = tt < ep.path.length * sd, V = walking ? prevV : ep.V;
  const f = walking ? tt / sd : ep.path.length - 1;
  gwDraw(c, (x, y) => heat(V[y * 5 + x]));
  const pts = ep.path.slice(0, Math.floor(f) + 1).map(([x, y]) => gwC(x, y)); poly(c, pts, alpha(C.gold, walking ? .8 : .4), 4);
  const [ax, ay] = lerpPath(ep.path, Math.min(f, ep.path.length - 1)); agent(c, ...gwC(ax, ay));
  txt(c, `episode ${e + 1}`, 120, 18, { s: 13, b: 1, c: C.gold });
  txt(c, walking ? 'playing… no learning yet' : `episode over! G = ${ep.G.toFixed(2)} → update every visited square`, 250, 18, { s: 12, c: walking ? C.dim : C.green });
  txt(c, 'Monte Carlo: wait for the real total, then average', 200, 246, { s: 12, c: C.dim });
};
A.td = (c, t) => {
  const data = memo('tddata', () => {
    const eps = mcEpisodes(21, 10), V = new Float32Array(20), out = [];
    eps.forEach((ep, e) => ep.rew.forEach((r, i) => { const [x, y] = ep.path[i], [nx, ny] = ep.path[i + 1], s = y * 5 + x; V[s] += .6 * (r + .9 * V[ny * 5 + nx] - V[s]); out.push({ e, s: [x, y], n: [nx, ny], V: V.slice() }); }));
    return out;
  });
  const dt = .42, k = Math.floor(t / dt) % data.length, u = (t % dt) / dt, d = data[k];
  gwDraw(c, (x, y) => heat(d.V[y * 5 + x]));
  const p0 = gwC(...d.s), p1 = gwC(...d.n);
  if (d.s[0] !== d.n[0] || d.s[1] !== d.n[1]) curve(c, p1, p0, 18, C.gold, 14, 2.5);
  const [ax, ay] = lerpPath([d.s, d.n], Math.min(1, u * 1.8)); agent(c, ...gwC(ax, ay));
  txt(c, `episode ${d.e + 1} · update after EVERY step, using the guess at the next square`, 200, 18, { s: 12, b: 1, c: C.gold });
  txt(c, 'V(s) ← V(s) + α [ r + γ V(s′) − V(s) ]', 200, 246, { s: 13, b: 1, c: C.gold });
};
A.traces = (c, t) => {
  const data = memo('trdata', () => {
    const ep = mcEpisodes(33, 1)[0], e = new Float32Array(20), out = [];
    ep.path.forEach(([x, y], i) => { for (let s = 0; s < 20; s++) e[s] *= .72; if (i < ep.path.length - 1) e[y * 5 + x] = 1; out.push({ p: [x, y], e: e.slice() }); });
    return { ep, out };
  });
  const { ep, out } = data, dt = .45, T = out.length * dt + 2.5, tt = t % T, k = Math.min(Math.floor(tt / dt), out.length - 1), fin = tt > out.length * dt;
  const e = out[k].e;
  gwDraw(c, (x, y) => fin ? heat(e[y * 5 + x] * (ep.rew.at(-1))) : mix(C.panel, C.gold, e[y * 5 + x] * .8));
  for (const [x, y] of gwCells()) if (e[y * 5 + x] > .02) txt(c, e[y * 5 + x].toFixed(2), ...gwC(x, y), { s: 11, b: 1, c: C.fg });
  const [ax, ay] = lerpPath(ep.path, Math.min(tt / dt, ep.path.length - 1)); agent(c, ...gwC(ax, ay));
  txt(c, fin ? `reward ${ep.rew.at(-1) > 0 ? '+1' : '−1'}! credit flows back in proportion to the trace` : 'glow = eligibility trace (fades by γλ each step)', 200, 18, { s: 12, b: 1, c: fin ? C.green : C.gold });
  txt(c, 'recent squares get more credit, old ones less', 200, 246, { s: 12, c: C.dim });
};

// ---------- Model-free control ----------
A.qgrid = (c, t) => {
  const data = memo('qdata', () => {
    const Q = new Float32Array(80), r = mulberry(4), out = []; let ep = 0;
    while (out.length < 420) {
      let [x, y] = GW.start; ep++;
      while (!gwTerm(x, y) && out.length < 420) {
        const s = y * 5 + x; let a; if (r() < .25) a = Math.floor(r() * 4); else { a = 0; for (let b = 1; b < 4; b++) if (Q[s * 4 + b] > Q[s * 4 + a] + r() * 1e-6) a = b; }
        const [nx, ny, rew] = gwStep(x, y, a), ns = ny * 5 + nx, mq = gwTerm(nx, ny) ? 0 : Math.max(Q[ns * 4], Q[ns * 4 + 1], Q[ns * 4 + 2], Q[ns * 4 + 3]);
        Q[s * 4 + a] += .5 * (rew + .9 * mq - Q[s * 4 + a]); out.push({ p: [x, y], n: [nx, ny], Q: Q.slice(), ep, a }); x = nx; y = ny;
      }
    }
    return out;
  });
  const tt = t % 34, k = Math.min(data.length - 1, Math.floor(tt < 5 ? tt * 3 : 15 + (tt - 5) * 14)), d = data[k];
  gwDraw(c, null);
  for (const [x, y] of gwCells()) {
    const [cx, cy] = gwC(x, y), s = y * 5 + x, h = GS / 2 - 3;
    const tri = [[[cx, cy], [cx - h, cy - h], [cx + h, cy - h]], [[cx, cy], [cx + h, cy - h], [cx + h, cy + h]], [[cx, cy], [cx + h, cy + h], [cx - h, cy + h]], [[cx, cy], [cx - h, cy + h], [cx - h, cy - h]]];
    tri.forEach((tr, a) => { c.beginPath(); tr.forEach((q, i) => i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1])); c.closePath(); c.fillStyle = heat(d.Q[s * 4 + a]); c.fill(); c.strokeStyle = C.bg; c.lineWidth = 1; c.stroke(); });
  }
  const [ax, ay] = lerpPath([d.p, d.n], .5); agent(c, ...gwC(ax, ay), C.pink, 9);
  txt(c, `episode ${d.ep}  ·  step ${k + 1}`, 200, 18, { s: 13, b: 1, c: C.gold });
  txt(c, 'each square has 4 triangles = Q(s, ↑→↓←); bright = good move', 200, 246, { s: 12, c: C.dim });
};
A.cliff = (c, t) => {
  const W = 10, s = 34, x0 = 30, y0 = 70, P = (x, y) => [x0 + x * s + s / 2, y0 + y * s + s / 2];
  for (let y = 0; y < 4; y++) for (let x = 0; x < W; x++) { const cl = y === 3 && x > 0 && x < W - 1; rr(c, x0 + x * s + 1, y0 + y * s + 1, s - 2, s - 2, 4, cl ? alpha(C.red, .45) : C.panel); }
  txt(c, 'C L I F F   (−100)', x0 + 5 * s, y0 + 3.5 * s, { s: 12, b: 1, c: C.red }); txt(c, 'S', ...P(0, 3), { b: 1 }); txt(c, 'G', ...P(9, 3), { b: 1, c: C.green });
  const qp = [[0, 3], [0, 2]]; for (let x = 1; x < W; x++) qp.push([x, 2]); qp.push([9, 3]);
  const sp = [[0, 3], [0, 2], [0, 1], [0, 0]]; for (let x = 1; x < W; x++) sp.push([x, 0]); sp.push([9, 1], [9, 2], [9, 3]);
  poly(c, qp.map(q => P(...q)), alpha(C.pink, .5), 3, [5, 4]); poly(c, sp.map(q => P(...q)), alpha(C.teal, .5), 3, [5, 4]);
  const dt = .3, T = sp.length * dt + 1, tt = t % T, loop = Math.floor(t / T), fall = loop % 2 === 1 && tt / dt > 5;
  let qf = Math.min(tt / dt, qp.length - 1);
  if (fall) { qf = 5; const q = P(...qp[5]); const u = clamp((tt / dt - 5) / 2, 0, 1); dot(c, q[0], q[1] + u * s, 10, C.pink); popup(c, '−100 😱 (random explore step)', q[0], q[1] + 40, u, C.red); }
  else { const q = lerpPath(qp, qf); dot(c, ...P(...q), 10, C.pink); }
  const sq = lerpPath(sp, Math.min(tt / dt, sp.length - 1)); dot(c, ...P(...sq), 10, C.teal);
  dot(c, 40, 22, 7, C.pink); txt(c, 'Q-learning: learns the shortest edge path', 52, 22, { a: 'left', s: 12 });
  dot(c, 40, 44, 7, C.teal); txt(c, 'SARSA: learns the safe path (it knows it explores)', 52, 44, { a: 'left', s: 12 });
};

// ---------- Function approximation / Deep RL ----------
A.funcApprox = (c, t) => {
  const f = x => .5 + .3 * Math.sin(x * 6) + .15 * Math.sin(x * 15);
  const xs = memo('fax', () => { const r = mulberry(9); return Array.from({ length: 16 }, () => .03 + r() * .94); });
  const n = Math.min(xs.length, Math.floor((t % 14) / .6) + 1), pts = xs.slice(0, n);
  for (const [px, title] of [[10, 'Table: one box per state'], [205, 'Function: generalise']]) {
    rr(c, px, 30, 185, 190, 10, C.panel); txt(c, title, px + 92, 18, { s: 12, b: 1 });
    pts.forEach(x => dot(c, px + 8 + x * 170, 210 - f(x) * 160, 4, C.gold));
  }
  const B = 17;
  for (let b = 0; b < B; b++) { const inb = pts.filter(x => Math.floor(x * B) === b); const bx = 18 + b * 10; if (inb.length) bar(c, bx, 210, 8, f(inb.reduce((a, q) => a + q, 0) / inb.length) * 160, alpha(C.teal, .7)); else txt(c, '?', bx + 4, 200, { s: 10, c: C.dim }); }
  const cur = []; for (let i = 0; i <= 60; i++) { const x = i / 60; let w = 0, s = 0; pts.forEach(q => { const k = Math.exp(-((x - q) ** 2) / .004); w += k; s += k * f(q); }); cur.push([213 + x * 170, 210 - (w > 1e-3 ? s / w : .5) * 160]); }
  poly(c, cur, C.teal, 3);
  txt(c, `${n} states seen · table: "?" = no idea · curve: a guess`, 200, 242, { s: 11, c: C.dim });
};
A.dqn = (c, t) => {
  const bx = 18 + 64 * (.5 + .5 * Math.sin(t * 1.3)), by = 50 + 100 * Math.abs(Math.sin(t * 1.9));
  const pad = memo('pad', () => ({ x: 50 }));
  const target = bx; const qL = .5 + clamp((pad.x - target) / 60, -.4, .4), qR = .5 + clamp((target - pad.x) / 60, -.4, .4), qS = .7 - Math.abs(target - pad.x) / 60;
  const Q = [qL, qS, qR], best = Q.indexOf(Math.max(...Q)); pad.x += [-1.2, 0, 1.2][best]; pad.x = clamp(pad.x, 30, 85);
  rr(c, 10, 40, 100, 150, 8, '#000', C.line); dot(c, bx, by, 6, C.fg); rr(c, pad.x - 15, 178, 30, 6, 3, C.teal);
  txt(c, 'screen pixels', 60, 24, { s: 11, c: C.dim }); arrow(c, 115, 115, 140, 115, C.dim, 2);
  const L = [[160, 4], [210, 6], [260, 3]], nodes = L.map(([x, n]) => Array.from({ length: n }, (_, i) => [x, 115 + (i - (n - 1) / 2) * 26]));
  for (let l = 0; l < 2; l++) for (const a of nodes[l]) for (const b of nodes[l + 1]) line(c, a[0], a[1], b[0], b[1], alpha(C.blue, .25 + .25 * Math.sin(t * 4 + a[1] + b[1])), 1);
  nodes.forEach((ns, l) => ns.forEach((q, i) => dot(c, q[0], q[1], 7, mix(C.panel, C.blue, .5 + .5 * Math.sin(t * 3 + i + l)))));
  txt(c, 'Q-network', 210, 24, { s: 11, c: C.dim });
  ['⬅', '⏸', '➡'].forEach((lab, i) => { const y = 89 + i * 26; bar(c, 300, y + 8, Q[i] * 70, -16, i === best ? C.gold : alpha(C.teal, .6)); txt(c, lab, 288, y, { s: 12 }); });
  txt(c, 'Q(s, a)', 335, 24, { s: 11, c: C.dim });
  txt(c, 'pick the biggest bar → move paddle → repeat', 200, 230, { s: 12, c: C.gold });
};
A.replay = (c, t) => {
  const n = 12, w = 27, x0 = 38, y0 = 95, k = Math.floor(t / .5), samp = Math.floor(t / 1.6), su = (t % 1.6) / 1.6;
  const col = i => [C.pink, C.teal, C.gold, C.blue, C.purple, C.green][i % 6];
  txt(c, '🤖 playing', 38, 40, { a: 'left', b: 1 }); txt(c, 'replay buffer (oldest gets overwritten)', 200, 75, { s: 11, c: C.dim });
  for (let j = 0; j < n; j++) {
    const idx = k - ((k - j) % n + n) % n; rr(c, x0 + j * w, y0, w - 4, 34, 5, idx >= 0 && idx <= k ? alpha(col(idx), .55) : C.panel, C.line);
    if (idx >= 0) txt(c, '' + (idx + 1), x0 + j * w + (w - 4) / 2, y0 + 17, { s: 10, b: 1 });
  }
  const wp = k % n; arrow(c, 110, 45, x0 + wp * w + 11, y0 - 4, C.pink, 2);
  const r = mulberry(samp + 1), pick = new Set(); while (pick.size < Math.min(4, k + 1)) pick.add(Math.floor(r() * Math.min(n, k + 1)));
  rr(c, 140, 190, 120, 44, 10, C.panel, C.teal, 2); txt(c, '🧠 learner', 200, 212, { b: 1 });
  if (k >= 4) pick.forEach(j => { const sx = x0 + j * w + 11, sy = y0 + 38; rr(c, x0 + j * w - 1, y0 - 2, w - 2, 38, 6, null, C.gold, 2.5); line(c, sx, sy, 200, 190, alpha(C.gold, .6), 1.5, [3, 3]); const p = [lerp(sx, 200, su), lerp(sy, 190, su)]; dot(c, p[0], p[1], 4, C.gold); });
  txt(c, 'learn from a random mix of old & new experience', 200, 250, { s: 12, c: C.dim });
};
A.target = (c, t) => {
  const sim = memo('tgt', () => { const r = mulberry(2); let q = 0, fr = 0; const out = []; for (let i = 0; i < 900; i++) { if (i % 45 === 0) fr = q; const y = 1 + .9 * fr; q += .09 * (y - q) + .12 * gauss(r); out.push([q, y]); } return out; });
  const dt = .05, T = 900 * dt, k = Math.floor((t % T) / dt), W = 200, x0 = 30, y0 = 220, sy = 18;
  line(c, x0, y0, 380, y0, C.line); line(c, x0, 30, x0, y0, C.line);
  const s = Math.max(0, k - W); const tg = [], on = [];
  for (let i = s; i <= k; i++) { const x = x0 + (i - s) / W * 350; on.push([x, y0 - sim[i][0] * sy]); tg.push([x, y0 - sim[i][1] * sy]); if (i % 45 === 0 && i > 0) { line(c, x, 30, x, y0, alpha(C.gold, .25), 1, [3, 3]); txt(c, 'copy', x, 40, { s: 10, c: C.gold }); } }
  poly(c, tg, C.gold, 2.5); poly(c, on, C.pink, 2);
  dot(c, 40, 16, 5, C.pink); txt(c, 'online net (updates every step)', 50, 16, { a: 'left', s: 11 });
  dot(c, 230, 16, 5, C.gold); txt(c, 'target (frozen, then copied)', 240, 16, { a: 'left', s: 11 });
  txt(c, 'chasing a target that stands still for a while = calmer learning', 200, 242, { s: 12, c: C.dim });
};
A.overest = (c, t) => {
  const k = Math.floor(t / .55), u = (t % .55) / .55;
  const stats = memo('ovs', () => { const r = mulberry(5), out = []; let s1 = 0, s2 = 0; for (let i = 0; i < 400; i++) { const a = Array.from({ length: 6 }, () => gauss(r)), b = Array.from({ length: 6 }, () => gauss(r)); const ia = a.indexOf(Math.max(...a)); s1 += a[ia]; s2 += b[ia]; out.push({ a, b, ia, m1: s1 / (i + 1), m2: s2 / (i + 1) }); } return out; });
  const d = stats[k % 400], y0 = 140, sc = 30;
  for (const [px, title, vals, pick, mean] of [[10, 'max of noisy guesses', d.a, d.ia, d.m1], [205, 'Double: choose with A, score with B', d.b, d.ia, d.m2]]) {
    rr(c, px, 30, 185, 190, 10, C.panel); txt(c, title, px + 92, 18, { s: 11, b: 1 });
    line(c, px + 8, y0, px + 177, y0, C.dim, 1.5, [4, 4]); txt(c, 'true value 0', px + 150, y0 + 10, { s: 9, c: C.dim });
    vals.forEach((v, i) => bar(c, px + 18 + i * 26, y0, 18, v * sc, i === pick ? C.gold : alpha(C.teal, .5)));
    line(c, px + 8, y0 - mean * sc, px + 177, y0 - mean * sc, C.pink, 2.5);
    txt(c, `running avg of picked: ${mean >= 0 ? '+' : ''}${mean.toFixed(2)}`, px + 92, 208, { s: 11, b: 1, c: C.pink });
  }
  txt(c, 'all 6 actions are truly worth 0 — max is fooled by noise', 200, 242, { s: 12, c: C.dim });
};

// ---------- Policy gradients ----------
A.pg = (c, t) => {
  const sim = memo('pg', () => { const r = mulberry(8); const th = [0, 0, 0, 0]; const out = []; const sm = th => { const e = th.map(Math.exp), s = e.reduce((a, b) => a + b); return e.map(x => x / s); };
    for (let i = 0; i < 40; i++) { const p = sm(th); let u = r(), a = 0; while (u > p[a] && a < 3) { u -= p[a]; a++; } const rew = a === 1 ? 1 : -.4; out.push({ p, a, rew }); for (let b = 0; b < 4; b++) th[b] += .5 * rew * ((b === a) - p[b]); } out.push({ p: sm(th), a: -1, rew: 0 }); return out; });
  const dt = 1, k = Math.floor(t / dt) % 40, u = ease(clamp((t % dt) / dt * 2 - .6, 0, 1)), d = sim[k], nx = sim[k + 1];
  const labels = ['⬆ up', '➡ right', '⬇ down', '⬅ left'];
  for (let i = 0; i < 4; i++) {
    const x = 40 + i * 85, p = lerp(d.p[i], nx.p[i], u);
    rr(c, x, 40, 66, 160, 10, C.panel, i === d.a ? (d.rew > 0 ? C.green : C.red) : C.line, i === d.a ? 3 : 1.5);
    bar(c, x + 10, 195, 46, p * 150, i === 1 ? C.teal : alpha(C.blue, .7)); txt(c, (p * 100).toFixed(0) + '%', x + 33, 185 - p * 150, { s: 12, b: 1 });
    txt(c, labels[i], x + 33, 215, { s: 12 });
    if (i === d.a) popup(c, d.rew > 0 ? 'reward! ↑' : 'meh ↓', x + 33, 55, (t % dt) / dt, d.rew > 0 ? C.green : C.red);
  }
  txt(c, `try ${k + 1}: sample an action, then nudge its probability by how it went`, 200, 18, { s: 12, c: C.gold });
  txt(c, '(here "right" is the good move)', 200, 242, { s: 11, c: C.dim });
};
A.reinforce = (c, t) => {
  const eps = [{ acts: [0, 0, 0, 1, 1, 1, 1], G: 1 }, { acts: [1, 1, 1, 0, 1, 0], G: -1 }];
  const dt = .3, ph = 8 * dt + 2.4, e = Math.floor(t / ph) % 2, tt = t % ph, ep = eps[e], path = pathFromActions(GW.start, ep.acts), done = tt > path.length * dt;
  gwDraw(c, null);
  const f = Math.min(tt / dt, path.length - 1), n = Math.floor(f);
  for (let i = 0; i < Math.min(n, ep.acts.length); i++) {
    const [cx, cy] = gwC(...path[i]); const pulse = done ? 1 + .25 * Math.sin((tt - path.length * dt) * 8) : 1;
    cellArrow(c, cx, cy, ep.acts[i], done ? (ep.G > 0 ? 30 * pulse : 14) : 22, done ? (ep.G > 0 ? C.green : alpha(C.red, .7)) : C.fg, done ? (ep.G > 0 ? 5 : 2) : 3);
  }
  agent(c, ...gwC(...lerpPath(path, f)));
  txt(c, done ? (ep.G > 0 ? 'G = +1 → make ALL these moves more likely' : 'G = −1 → make all these moves less likely') : `episode ${e + 1}: play it out…`, 200, 18, { s: 13, b: 1, c: done ? (ep.G > 0 ? C.green : C.red) : C.gold });
  txt(c, 'REINFORCE: credit is the whole episode’s return', 200, 246, { s: 12, c: C.dim });
};
A.baseline = (c, t) => {
  const k = Math.floor(t / 1.6), r = mulberry(k + 3), G = Array.from({ length: 7 }, () => 100 + 12 * gauss(r)), m = 100;
  for (const [px, title, vals, sc] of [[10, 'push = return G', G, .8], [205, 'push = G − baseline', G.map(g => g - m), 3]]) {
    rr(c, px, 30, 185, 190, 10, C.panel); txt(c, title, px + 92, 18, { s: 12, b: 1 });
    const y0 = px < 100 ? 205 : 125; line(c, px + 8, y0, px + 177, y0, C.dim, 1);
    vals.forEach((v, i) => { const x = px + 26 + i * 22; arrow(c, x, y0, x, y0 - v * sc, v >= 0 ? C.green : C.red, 3, 7); });
  }
  txt(c, 'left: every action pushed up · right: better/worse than usual', 200, 236, { s: 11, c: C.dim });
  txt(c, 'same average direction, far less noise', 200, 252, { s: 11, c: C.gold });
};
A.actorCritic = (c, t) => {
  const T = 4, u = (t % T) / T, k = Math.floor(t / T), good = k % 2 === 0;
  const Ac = [95, 75], Cr = [305, 75], En = [200, 200];
  rr(c, 35, 45, 120, 60, 12, C.panel, C.pink, 2); txt(c, '🎭 Actor', 95, 66, { b: 1 }); txt(c, 'chooses actions', 95, 86, { s: 10, c: C.dim });
  rr(c, 245, 45, 120, 60, 12, C.panel, C.teal, 2); txt(c, '🧐 Critic', 305, 66, { b: 1 }); txt(c, 'predicts value', 305, 86, { s: 10, c: C.dim });
  rr(c, 140, 175, 120, 50, 12, C.panel, C.gold, 2); txt(c, '🌍 Env', 200, 200, { b: 1 });
  const legs = [[[95, 105], [95, 200], [140, 200], 'action', C.pink], [[260, 200], [305, 200], [305, 105], 'reward', C.gold], [[245, 75], [155, 75], null, good ? 'δ > 0' : 'δ < 0', good ? C.green : C.red]];
  legs.forEach(([a, b, cc, lab, col], i) => { const pts = cc ? [a, b, cc] : [a, b]; poly(c, pts, C.line, 2); const e = pts.at(-1), f = pts.at(-2); head(c, e[0], e[1], Math.atan2(e[1] - f[1], e[0] - f[0]), C.line, 9);
    const lu = u * 3 - i; if (lu > 0 && lu < 1) { const p = along(pts, lu); dot(c, p[0], p[1], 7, col); }
    const lp = cc ? along(pts, .5) : [200, 62]; txt(c, lab, lp[0] + (i === 0 ? -30 : i === 1 ? 32 : 0), lp[1] - (i === 2 ? 0 : 0), { s: i === 2 ? 11 : 11, b: 1, c: col }); });
  txt(c, good ? '"better than I expected!"' : '"worse than I expected"', 200, 128, { s: 12, b: 1, c: good ? C.green : C.red });
  if (u > .67) popup(c, good ? 'do that more!' : 'do that less', 95, 30, (u - .67) * 3, good ? C.green : C.red);
  txt(c, 'critic grades every step, so the actor needn’t wait for the episode to end', 200, 246, { s: 11, c: C.dim });
};
A.ppoClip = (c, t) => {
  const T = 10, pos = (t % T) < T / 2, Av = pos ? 1 : -1, eps = .2, x0 = 50, y0 = 130, sx = 150, sy = 45;
  const X = r => x0 + r * sx, Y = l => y0 - l * sy;
  line(c, x0, 55, x0, 230, C.line); line(c, x0, y0, 370, y0, C.line);
  txt(c, 'ratio r = π_new(a|s) / π_old(a|s)', 230, 238, { s: 11, c: C.dim }); 
  [1 - eps, 1, 1 + eps].forEach(r => { line(c, X(r), 30, X(r), 230, alpha(C.dim, .4), 1, [3, 3]); txt(c, r.toFixed(1), X(r), 222, { s: 10, c: C.dim }); });
  const L = r => Math.min(r * Av, clamp(r, 1 - eps, 1 + eps) * Av), raw = [], cl = [];
  for (let i = 0; i <= 64; i++) { const r = i / 32; raw.push([X(r), Y(r * Av)]); cl.push([X(r), Y(L(r))]); }
  poly(c, raw, alpha(C.dim, .6), 1.5, [5, 4]); poly(c, cl, C.teal, 4);
  const r = 1 + .65 * Math.sin((t % (T / 2)) / (T / 2) * Math.PI * 2 - .3), ly = L(r);
  const flat = pos ? r > 1 + eps : r < 1 - eps; dot(c, X(r), Y(ly), 8, flat ? C.red : C.gold);
  if (!flat) arrow(c, X(r), Y(ly), X(r + .18 * (pos ? 1 : -1)), Y(L(r + .18 * (pos ? 1 : -1))), C.gold, 3, 9);
  txt(c, pos ? 'advantage A > 0 (good action): raise its probability…' : 'advantage A < 0 (bad action): lower its probability…', 210, 16, { s: 12, b: 1, c: pos ? C.green : C.red });
  txt(c, flat ? '…but not past the clip: flat = no more push' : '…gradient pushes this way', 210, 36, { s: 12, c: flat ? C.red : C.gold });
};
A.entropy = (c, t) => {
  const lg = [2, 1, .5, 0, -.5, -1], tau = Math.exp(Math.sin(t * .7) * 1.5), e = lg.map(l => Math.exp(l / tau)), s = e.reduce((a, b) => a + b), p = e.map(x => x / s);
  const H = -p.reduce((a, q) => a + (q > 0 ? q * Math.log2(q) : 0), 0);
  p.forEach((q, i) => { bar(c, 40 + i * 55, 200, 40, q * 160, mix(C.blue, C.gold, q)); txt(c, 'a' + (i + 1), 60 + i * 55, 214, { s: 11, c: C.dim }); });
  txt(c, `temperature ${tau.toFixed(2)}   ·   entropy ${H.toFixed(2)} / ${Math.log2(6).toFixed(2)} bits`, 200, 20, { s: 13, b: 1, c: C.gold });
  rr(c, 100, 32, 200, 8, 4, C.panel); rr(c, 100, 32, 200 * H / Math.log2(6), 8, 4, C.purple);
  txt(c, H > 1.8 ? 'spread out → keeps exploring' : H < 1 ? 'peaked → confident, exploiting' : 'in between', 200, 240, { s: 13, c: H > 1.8 ? C.purple : H < 1 ? C.teal : C.dim });
};
A.continuous = (c, t) => {
  const T = 9, tt = t % T, w = .15 * Math.sin(t * .4);
  const Q = a => .55 * Math.exp(-(((a - .45 - w) / .28) ** 2)) + .3 * Math.exp(-(((a + .55) / .3) ** 2)) + .1, dQ = a => (Q(a + 1e-3) - Q(a - 1e-3)) / 2e-3;
  const X = a => 40 + (a + 1) * 160, Y = q => 210 - q * 220;
  const pts = []; for (let i = 0; i <= 80; i++) { const a = -1 + i / 40; pts.push([X(a), Y(Q(a))]); }
  line(c, 40, 210, 360, 210, C.line); poly(c, pts, C.teal, 3);
  txt(c, 'action a (e.g. steering angle) — infinitely many choices', 200, 228, { s: 11, c: C.dim }); txt(c, "critic's Q(s, a)", 60, 30, { s: 11, c: C.teal, a: 'left' });
  let a = -.1; for (let i = 0; i < tt * 30; i++) a += .004 * dQ(a) / (Math.abs(dQ(a)) + .3) * 3; a = clamp(a, -1, 1);
  const r = mulberry(Math.floor(t * 6)); for (let i = 0; i < 4; i++) { const b = clamp(a + .12 * gauss(r), -1, 1); dot(c, X(b), Y(Q(b)), 3, alpha(C.gold, .6)); }
  const g = dQ(a); arrow(c, X(a), Y(Q(a)), X(a + .18 * Math.sign(g)), Y(Q(a) + .18 * g * Math.sign(g)), C.pink, 3, 9);
  dot(c, X(a), Y(Q(a)), 8, C.pink); txt(c, 'actor', X(a), Y(Q(a)) - 18, { s: 11, b: 1, c: C.pink });
  txt(c, 'actor slides its action uphill on the critic’s curve', 200, 248, { s: 12, c: C.gold });
};
A.nstep = (c, t) => {
  const lam = .5 + .5 * Math.sin(t * .6), R = [1, 0, 2, 0, 1, 0, 0, 3];
  R.forEach((r, i) => { dot(c, 70 + i * 30, 30, 11, C.panel, C.gold); txt(c, '' + r, 70 + i * 30, 30, { s: 11, b: 1 }); });
  txt(c, 'rewards →', 10, 30, { s: 11, c: C.dim, a: 'left' });
  const ns = [1, 2, 3, 5, 8]; let wsum = 0;
  ns.forEach((n, j) => {
    const w = n === 8 ? Math.pow(lam, 4) : (1 - lam) * Math.pow(lam, j), y = 62 + j * 34; wsum += w;
    const x1 = 59 + Math.min(n, 8) * 30; rr(c, 56, y - 9, x1 - 56, 18, 9, alpha(C.teal, .15 + .85 * Math.min(1, w * 1.6)));
    if (n < 8) { rr(c, x1 + 2, y - 9, 34, 18, 9, alpha(C.purple, .15 + .85 * Math.min(1, w * 1.6))); txt(c, '+V', x1 + 19, y, { s: 10, b: 1 }); }
    txt(c, n === 8 ? 'full (MC)' : n + '-step', 50, y, { s: 11, a: 'right', c: C.fg });
    bar(c, 330, y + 8, w * 60, -16, C.gold); txt(c, w.toFixed(2), 325, y, { s: 10, a: 'right', c: C.dim });
  });
  txt(c, `λ = ${lam.toFixed(2)}`, 360, 30, { s: 14, b: 1, c: C.gold });
  txt(c, 'λ → 0: trust the critic’s guess (low noise, biased)', 200, 232, { s: 11, c: C.purple });
  txt(c, 'λ → 1: trust the real rewards (unbiased, noisy)', 200, 249, { s: 11, c: C.teal });
};

// ---------- Model-based ----------
A.modelBased = (c, t) => {
  const V = gwVI().at(-1), path = gwPolicyPath(), dt = 2.2, k = Math.floor(t / dt) % (path.length - 1), u = (t % dt) / dt, s = path[k];
  gwDraw(c, null);
  const opts = [0, 1, 2, 3].map(a => { const [nx, ny, r] = gwStep(s[0], s[1], a); return { a, n: [nx, ny], q: r + .9 * V[ny * 5 + nx] }; });
  const best = opts.reduce((a, b) => b.q > a.q ? b : a);
  if (u < .7) opts.forEach((o, i) => { const f = clamp(u * 4 - i * .4, 0, 1); if (f <= 0) return; const p0 = gwC(...s), p1 = gwC(...o.n); const same = o.n[0] === s[0] && o.n[1] === s[1];
    if (!same) arrow(c, p0[0], p0[1], lerp(p0[0], p1[0], f), lerp(p0[1], p1[1], f), alpha(o === best && u > .5 ? C.gold : C.purple, .9), 2.5, 8, [4, 3]);
    if (f >= 1) txt(c, o.q.toFixed(2), same ? p0[0] + 18 : p1[0], (same ? p0[1] - 18 : p1[1]) + 14, { s: 11, b: 1, c: o === best && u > .5 ? C.gold : C.purple }); });
  const pos = u < .7 ? s : lerpPath([s, path[k + 1]], (u - .7) / .3); agent(c, ...gwC(...pos));
  txt(c, u < .7 ? '💭 imagining each move with its model of the world…' : 'then take the best one for real', 200, 18, { s: 12, b: 1, c: u < .7 ? C.purple : C.gold });
  txt(c, 'model-based: plan in your head before acting', 200, 246, { s: 12, c: C.dim });
};
A.mcts = (c, t) => {
  const sim = memo('mcts', () => {
    const r = mulberry(12), nodes = [{ id: 0, p: -1, d: 0, ch: [], N: 0, W: 0, v: .5 }], its = [];
    for (let it = 0; it < 30; it++) {
      let n = nodes[0]; const path = [0];
      while (n.ch.length === 3 && n.d < 3) { let best = -1, bc; for (const ci of n.ch) { const ch = nodes[ci]; const u = ch.W / ch.N + 1.2 * Math.sqrt(Math.log(n.N) / ch.N); if (u > best) { best = u; bc = ch; } } n = bc; path.push(n.id); }
      let nw = null; if (n.d < 3) { nw = { id: nodes.length, p: n.id, d: n.d + 1, ch: [], N: 0, W: 0, v: clamp(n.v + (r() - .5) * .7, 0, 1) }; nodes.push(nw); n.ch.push(nw.id); path.push(nw.id); n = nw; }
      const val = clamp(n.v + (r() - .5) * .3, 0, 1); path.forEach(id => { nodes[id].N++; nodes[id].W += val; });
      its.push({ path, nw: nw && nw.id, val, NW: nodes.map(q => [q.N, q.W]) });
    }
    const pos = {}; let leaf = 0; const lay = id => { const n = nodes[id]; if (!n.ch.length) { pos[id] = leaf++; return pos[id]; } const xs = n.ch.map(lay); pos[id] = (xs[0] + xs.at(-1)) / 2; return pos[id]; }; lay(0);
    return { nodes, its, pos, leaves: leaf };
  });
  const { nodes, its, pos, leaves } = sim, dt = 1.1, k = Math.floor(t / dt) % its.length, u = (t % dt) / dt, it = its[k];
  const P = id => [20 + pos[id] / Math.max(1, leaves - 1) * 360, 36 + nodes[id].d * 52];
  const alive = new Set(); for (let j = 0; j <= k; j++) its[j].path.forEach(id => alive.add(id));
  if (u < .3 && it.nw != null) alive.delete(it.nw);
  for (const id of alive) if (id) line(c, ...P(nodes[id].p), ...P(id), it.path.includes(id) ? (u > .7 ? C.green : C.gold) : C.line, it.path.includes(id) ? 3 : 1.5);
  const NW = u > .7 ? it.NW : (k ? its[k - 1].NW : nodes.map(() => [0, 0]));
  for (const id of alive) { const [x, y] = P(id), [N, W] = NW[id] || [0, 0]; dot(c, x, y, 10, N ? heat(W / N * .9) : C.panel, it.path.includes(id) ? C.gold : C.line); txt(c, '' + N, x, y, { s: 9, b: 1, c: C.bg }); }
  if (it.nw != null && u > .3 && u < .7) { const [x, y] = P(it.nw); const pts = []; for (let i = 0; i <= 10; i++) pts.push([x + Math.sin(i * 1.7) * 6, y + 12 + i * 5 * (u - .3) / .4]); poly(c, pts, C.purple, 2); }
  const ph = u < .3 ? '1 select: follow promising branches' : u < .5 ? '2 expand: add a new move' : u < .7 ? '3 simulate: quick random playout' : `4 back up: result ${it.val.toFixed(2)} → update counts`;
  txt(c, ph, 200, 236, { s: 12, b: 1, c: u < .3 ? C.gold : u < .7 ? C.purple : C.green });
  txt(c, `iteration ${k + 1}  ·  numbers = visit counts`, 200, 252, { s: 10, c: C.dim });
};
A.selfplay = (c, t) => {
  const st = [[200, 40, '🎲 self-play games'], [330, 130, '📦 game records'], [200, 220, '🧠 train policy + value net'], [70, 130, '🌳 MCTS guided by net']];
  const T = 6, u = (t % T) / T, gen = Math.floor(t / T) + 1;
  c.beginPath(); c.ellipse(200, 130, 130, 90, 0, 0, 7); c.strokeStyle = C.line; c.lineWidth = 2; c.stroke();
  const a = -Math.PI / 2 + u * Math.PI * 2; dot(c, 200 + 130 * Math.cos(a), 130 + 90 * Math.sin(a), 7, C.gold);
  st.forEach(([x, y, l], i) => { const on = Math.floor(u * 4) === i; rr(c, x - 62, y - 15, 124, 30, 15, C.panel, on ? C.gold : C.line, on ? 2.5 : 1.5); txt(c, l, x, y, { s: 11, b: 1 }); });
  const moves = [4, 0, 8, 2, 6, 3, 5], m = Math.floor((t % T) / T * 8);
  for (let i = 0; i < 9; i++) { const x = 176 + (i % 3) * 16, y = 106 + Math.floor(i / 3) * 16; rr(c, x, y, 15, 15, 3, C.bg); const mi = moves.indexOf(i); if (mi >= 0 && mi < m) txt(c, mi % 2 ? 'O' : 'X', x + 7.5, y + 8, { s: 11, b: 1, c: mi % 2 ? C.teal : C.pink }); }
  txt(c, `generation ${gen}`, 200, 168, { s: 11, b: 1, c: C.gold });
};
A.worldModel = (c, t) => {
  const T = 8, tt = t % T, dream = tt > 3.5;
  const ball = s => { const x = 25 + Math.abs(((s * 55) % 140) - 70) * 1.0, y = 25 + Math.abs(((s * 80) % 100) - 50) * 1.0; return [x, y]; };
  rr(c, 15, 40, 110, 110, 10, '#000', dream ? C.line : C.teal, 2); txt(c, dream ? 'real world (eyes closed)' : 'real world', 70, 26, { s: 11, b: 1, c: dream ? C.dim : C.teal });
  if (!dream) { const [x, y] = ball(tt); dot(c, 15 + x, 40 + y, 8, C.fg); } else txt(c, '😴', 70, 95, { s: 26 });
  arrow(c, 130, 95, 160, 95, C.dim, 2); txt(c, 'encode', 145, 80, { s: 9, c: C.dim });
  rr(c, 165, 60, 70, 70, 10, C.panel, C.purple, 2); txt(c, 'latent z', 200, 50, { s: 11, b: 1, c: C.purple });
  for (let i = 0; i < 6; i++) bar(c, 172 + i * 10, 122, 7, 25 + 25 * Math.sin(t * 2.2 + i * 1.3), alpha(C.purple, .8));
  arrow(c, 240, 95, 270, 95, C.dim, 2); txt(c, 'imagine', 255, 80, { s: 9, c: C.dim });
  rr(c, 275, 40, 110, 110, 10, '#000', dream ? C.purple : C.line, 2); txt(c, 'dream', 330, 26, { s: 11, b: 1, c: dream ? C.purple : C.dim });
  if (dream) { const [x, y] = ball(tt); c.filter = 'blur(1.5px)'; dot(c, 275 + x + 2 * Math.sin(t * 9), 40 + y, 8, C.purple); c.filter = 'none'; }
  txt(c, dream ? '🤖 practising inside the learned model — no real steps needed' : 'learn a compact model of how the world moves', 200, 185, { s: 12, b: 1, c: dream ? C.purple : C.teal });
  txt(c, '(Dreamer, World Models, MuZero use this idea)', 200, 210, { s: 11, c: C.dim });
};

// ---------- Rewards & exploration extras ----------
A.shaping = (c, t) => {
  const N = 6, s = 26, walk = (seed, bias) => memo('sh' + seed, () => { const r = mulberry(seed); let x = 0, y = 5; const p = [[x, y]]; for (let i = 0; i < 40; i++) { if (x === 5 && y === 0) { p.push([x, y]); continue; } let a; if (r() < bias) a = (r() < .5 && x < 5) ? 1 : (y > 0 ? 0 : 1); else a = Math.floor(r() * 4); const d = DIRS[a]; x = clamp(x + d[0], 0, 5); y = clamp(y + d[1], 0, 5); p.push([x, y]); } return p; });
  const T = 11, f = Math.min((t % T) / .25, 40);
  [[20, 'sparse: +1 only at 🏁', walk(3, 0), false], [215, 'shaped: warmer = closer', walk(4, .8), true]].forEach(([x0, title, path, sh]) => {
    txt(c, title, x0 + 82, 18, { s: 12, b: 1 });
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) rr(c, x0 + x * s + 4, 32 + y * s, s - 2, s - 2, 4, sh ? heat((x + (5 - y)) / 10 * .9) : C.panel);
    txt(c, '🏁', x0 + 5 * s + 4 + s / 2 - 1, 32 + s / 2, { s: 13 });
    const P = q => [x0 + q[0] * s + 4 + s / 2 - 1, 32 + q[1] * s + s / 2 - 1], seen = path.slice(0, Math.floor(f) + 1).map(P);
    poly(c, seen, alpha(C.pink, .5), 2); const q = lerpPath(path, f); dot(c, ...P(q), 8, C.pink);
  });
  txt(c, 'extra hints make the right direction obvious — but they can also be gamed', 200, 205, { s: 11, c: C.dim });
  txt(c, 'safe recipe: potential-based shaping', 200, 225, { s: 12, b: 1, c: C.gold });
};
A.hacking = (c, t) => {
  const cx = 200, cy = 125, rx = 160, ry = 85;
  c.beginPath(); c.ellipse(cx, cy, rx, ry, 0, 0, 7); c.strokeStyle = C.line; c.lineWidth = 38; c.stroke();
  c.beginPath(); c.ellipse(cx, cy, rx, ry, 0, 0, 7); c.strokeStyle = alpha(C.dim, .3); c.lineWidth = 1; c.setLineDash([6, 6]); c.stroke(); c.setLineDash([]);
  line(c, cx + rx - 20, cy, cx + rx + 20, cy, C.fg, 4); txt(c, '🏁 finish', cx + rx - 20, cy + 30, { s: 11, b: 1 });
  const tc = [80, 120], R = 26; const tg = [0, 1, 2].map(i => [tc[0] + R * Math.cos(i * 2.1), tc[1] + R * Math.sin(i * 2.1)]);
  const ang = t * 2.6, bx = tc[0] + R * Math.cos(ang), by = tc[1] + R * Math.sin(ang);
  tg.forEach((q, i) => { const since = ((ang - i * 2.1) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2); const vis = since > 1.2; if (vis) txt(c, '🎯', q[0], q[1], { s: 14 }); if (since < .6) popup(c, '+10', q[0], q[1] - 12, since / .6); });
  txt(c, '🚤', bx, by, { s: 18 });
  txt(c, 'score: ' + (Math.floor(t * 2.6 / 2.1) * 10), 330, 30, { s: 14, b: 1, c: C.gold });
  txt(c, 'reward = points for hitting targets  →  boat circles forever, never finishes', 200, 236, { s: 11, c: C.dim });
};
A.curiosity = (c, t) => {
  const W = 11, H = 6, s = 30, x0 = 35, y0 = 30;
  const sim = memo('cur', () => { const r = mulberry(6), nv = new Float32Array(W * H).fill(1); let x = 0, y = 5; const out = [];
    for (let i = 0; i < 140; i++) { nv[y * W + x] *= .25; out.push({ p: [x, y], nv: nv.slice() }); let best = -1, bx = x, by = y; for (const [dx, dy] of DIRS) { const nx = x + dx, ny = y + dy; if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue; let sc = nv[ny * W + nx] + r() * .05; for (const [ex, ey] of DIRS) { const mx = nx + ex, my = ny + ey; if (mx >= 0 && my >= 0 && mx < W && my < H) sc += .3 * nv[my * W + mx]; } if (sc > best) { best = sc; bx = nx; by = ny; } } x = bx; y = by; }
    return out; });
  const k = Math.floor(t / .18) % sim.length, d = sim[k];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) rr(c, x0 + x * s + 1, y0 + y * s + 1, s - 2, s - 2, 5, mix(C.panel, C.purple, d.nv[y * W + x] * .85));
  dot(c, x0 + d.p[0] * s + s / 2, y0 + d.p[1] * s + s / 2, 9, C.gold);
  txt(c, 'bright = new/surprising = intrinsic reward; it fades once seen', 200, 225, { s: 12, c: C.dim });
  txt(c, 'no external reward at all — still explores everything', 200, 245, { s: 12, b: 1, c: C.purple });
};

// ---------- Offline / imitation / human feedback ----------
A.offline = (c, t) => {
  const band = x => .3 + .25 * Math.sin(x * 5), X = x => 40 + x * 330, Y = y => 210 - y * 180;
  const Qe = (x, y) => .7 * Math.exp(-(((y - band(x)) / .18) ** 2)) * (.6 + .4 * x) + 1.1 * Math.exp(-(((y - .9) ** 2) / .01 + ((x - .7) ** 2) / .04));
  for (let i = 0; i < 22; i++) for (let j = 0; j < 12; j++) { const x = (i + .5) / 22, y = (j + .5) / 12; c.fillStyle = heat(clamp(Qe(x, y), 0, 1) * .9); c.globalAlpha = .55; c.fillRect(X(i / 22), Y((j + 1) / 12), 330 / 22 + .5, 180 / 12 + .5); c.globalAlpha = 1; }
  const pts = memo('offd', () => { const r = mulberry(3); return Array.from({ length: 60 }, () => { const x = r(); return [x, band(x) + .07 * gauss(r)]; }); });
  pts.forEach(([x, y]) => dot(c, X(x), Y(y), 2.5, C.fg));
  const sx = .5 + .45 * Math.sin(t * .5);
  let ny = 0, cy = 0, nb = -9, cb = -9; for (let j = 0; j <= 100; j++) { const y = j / 100, q = Qe(sx, y); if (q > nb) { nb = q; ny = y; } const qc = q - 6 * Math.max(0, Math.abs(y - band(sx)) - .1) ** 2 * 10; if (qc > cb) { cb = qc; cy = y; } }
  line(c, X(sx), 30, X(sx), 210, alpha(C.fg, .4), 1, [3, 3]);
  txt(c, '✖', X(sx), Y(ny), { s: 18, b: 1, c: C.red }); dot(c, X(sx), Y(cy), 7, C.teal, C.bg);
  txt(c, 'state →', 370, 224, { s: 10, c: C.dim, a: 'right' }); txt(c, 'action ↑', 36, 26, { s: 10, c: C.dim, a: 'left' });
  txt(c, '✖ naive: trusts a Q-value it never saw data for', 200, 236, { s: 11, b: 1, c: C.red });
  txt(c, '● conservative: stays near the logged data (dots)', 200, 252, { s: 11, b: 1, c: C.teal });
};
A.imitation = (c, t) => {
  const T = 9, tt = t % T, dag = tt > T / 2, u = (tt % (T / 2)) / (T / 2);
  const road = x => 130 + 40 * Math.sin(x / 55);
  const up = [], dn = []; for (let x = 20; x <= 380; x += 6) { up.push([x, road(x) - 32]); dn.push([x, road(x) + 32]); }
  c.beginPath(); up.forEach((p, i) => i ? c.lineTo(...p) : c.moveTo(...p)); dn.slice().reverse().forEach(p => c.lineTo(...p)); c.closePath(); c.fillStyle = C.panel; c.fill();
  const mid = []; for (let x = 20; x <= 380; x += 6) mid.push([x, road(x)]); poly(c, mid, alpha(C.gold, .6), 2, [8, 6]);
  const xe = 20 + u * 360, off = x => dag ? 7 * Math.sin(x / 18) : .0009 * (x - 20) ** 2;
  const lp = []; for (let x = 20; x <= xe; x += 4) lp.push([x, road(x) + off(x)]); poly(c, lp, C.pink, 3);
  if (dag) for (let x = 40; x < xe; x += 40) arrow(c, x, road(x) + off(x), x, road(x) + off(x) * .2, C.green, 2, 6);
  txt(c, '🚗', xe, road(xe) + off(xe), { s: 18 });
  txt(c, dag ? 'DAgger: expert labels the states the learner actually visits' : 'behaviour cloning: tiny errors → unfamiliar states → bigger errors', 200, 22, { s: 12, b: 1, c: dag ? C.green : C.pink });
  if (!dag && u > .75) txt(c, '"I never saw this in the demos!" 😵', 300, 235, { s: 12, c: C.red });
  txt(c, '- - expert’s path', 60, 245, { s: 11, c: C.gold });
};
A.irl = (c, t) => {
  const W = 7, H = 5, s = 40, x0 = 60, y0 = 30, goal = [5, 1];
  const demos = [[[0, 4], [1, 4], [2, 4], [2, 3], [3, 3], [3, 2], [4, 2], [4, 1], [5, 1]], [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4, 1], [5, 1]], [[6, 4], [6, 3], [6, 2], [5, 2], [5, 1]]];
  const dt = .35, lens = demos.map(d => d.length * dt + .5), T = lens.reduce((a, b) => a + b) + 3; let tt = t % T, k = 0;
  while (k < 3 && tt > lens[k]) { tt -= lens[k]; k++; }
  const R = new Float32Array(W * H);
  demos.forEach((d, j) => { const upto = j < k ? d.length : j === k ? Math.min(d.length, Math.floor(tt / dt) + 1) : 0; d.slice(0, upto).forEach(([x, y], i) => R[y * W + x] += Math.pow((i + 1) / d.length, 3)); });
  const mx = Math.max(.01, ...R);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) rr(c, x0 + x * s + 1, y0 + y * s + 1, s - 2, s - 2, 6, heat(R[y * W + x] / mx * .95));
  demos.forEach((d, j) => { if (j > k) return; const upto = j < k ? d.length : Math.min(d.length, Math.floor(tt / dt) + 1); poly(c, d.slice(0, upto).map(([x, y]) => [x0 + x * s + s / 2, y0 + y * s + s / 2]), alpha(C.pink, j === k ? .9 : .35), 3); });
  if (k === 3) txt(c, '⭐', x0 + goal[0] * s + s / 2, y0 + goal[1] * s + s / 2, { s: 22 });
  txt(c, k < 3 ? `watching expert demo ${k + 1}…` : 'inferred: "they must want THIS spot"', 200, 18, { s: 12, b: 1, c: k < 3 ? C.pink : C.gold });
  txt(c, 'colour = guessed reward, built from where experts go', 200, 246, { s: 12, c: C.dim });
};
A.rlhf = (c, t) => {
  const T = 5, u = (t % T) / T, k = Math.floor(t / T), pick = k % 2;
  rr(c, 120, 12, 160, 28, 14, C.panel, C.line); txt(c, '"explain rain to a kid"', 200, 26, { s: 11 });
  [[25, 'A'], [215, 'B']].forEach(([x, l], i) => { const ch = u > .3 && i === pick; rr(c, x, 55, 160, 62, 10, C.panel, ch ? C.green : C.line, ch ? 3 : 1.5); txt(c, 'answer ' + l, x + 80, 68, { s: 11, b: 1 });
    for (let j = 0; j < 3; j++) line(c, x + 14, 84 + j * 11, x + 14 + 130 - (j * 23 + i * 17) % 60, 84 + j * 11, C.dim, 3); if (ch) txt(c, '✅', x + 148, 66, { s: 13 }); });
  txt(c, '👤 human: "this one is better"', 200, 135, { s: 12, c: u > .3 ? C.green : C.dim });
  rr(c, 40, 165, 130, 50, 10, C.panel, u > .5 ? C.gold : C.line, 2); txt(c, '⚖️ reward model', 105, 182, { s: 11, b: 1 }); txt(c, 'learns to score like a human', 105, 200, { s: 9, c: C.dim });
  rr(c, 230, 165, 130, 50, 10, C.panel, u > .75 ? C.pink : C.line, 2); txt(c, '🤖 language model', 295, 182, { s: 11, b: 1 }); txt(c, 'RL (PPO) to raise score', 295, 200, { s: 9, c: C.dim });
  arrow(c, 105, 145, 105, 162, u > .4 ? C.gold : C.line, 2); arrow(c, 172, 190, 228, 190, u > .65 ? C.pink : C.line, 2); txt(c, 'reward', 200, 180, { s: 9, c: C.dim });
  if (u > .65) { const p = along([[172, 190], [228, 190]], (u - .65) / .35); dot(c, p[0], p[1], 5, C.gold); }
  txt(c, 'Reinforcement Learning from Human Feedback', 200, 242, { s: 12, b: 1, c: C.gold });
};

// ---------- Partial observability / structure / multi-agent / sim2real ----------
A.pomdp = (c, t) => {
  const N = 14, doors = new Set([1, 5, 10]), cw = 25, x0 = 25, cyy = 190;
  const sim = memo('pomdp', () => { let b = new Float32Array(N).fill(1 / N); const out = []; const start = 3;
    for (let k = 0; k < 9; k++) { const pos = start + k; if (k) { const nb = new Float32Array(N); for (let i = 0; i < N; i++) { nb[Math.min(N - 1, i + 1)] += .9 * b[i]; nb[i] += .1 * b[i]; } b = nb; } const obs = doors.has(pos);
      const prior = b.slice(); for (let i = 0; i < N; i++) b[i] *= (doors.has(i) === obs) ? .9 : .1; const z = b.reduce((a, q) => a + q); b = b.map(q => q / z); out.push({ pos, obs, prior, b: Float32Array.from(b) }); }
    return out; });
  const dt = 1.5, k = Math.floor(t / dt) % sim.length, u = (t % dt) / dt, d = sim[k], bb = u < .45 ? d.prior : d.b;
  for (let i = 0; i < N; i++) { rr(c, x0 + i * cw + 1, cyy, cw - 2, 30, 3, C.panel); if (doors.has(i)) rr(c, x0 + i * cw + 5, cyy + 4, cw - 10, 26, 2, '#8a5a2b'); }
  line(c, x0, cyy, x0 + N * cw, cyy, C.dim, 2);
  for (let i = 0; i < N; i++) bar(c, x0 + i * cw + 4, cyy - 10, cw - 8, Math.min(120, bb[i] * 230), mix(C.blue, C.gold, bb[i] * 2.5));
  txt(c, '🤖', x0 + d.pos * cw + cw / 2, cyy + 46, { s: 18 });
  txt(c, 'bars = belief: "where do I think I am?"', 200, 18, { s: 12, b: 1, c: C.gold });
  txt(c, u < .45 ? 'move right → belief shifts & blurs' : `sensor says: ${d.obs ? '🚪 door!' : 'plain wall'} → reweight`, 200, 40, { s: 12, c: d.obs && u >= .45 ? C.green : C.fg });
  txt(c, '(the robot itself can’t see this 🤖 marker)', 200, 252, { s: 10, c: C.dim });
};
A.hrl = (c, t) => {
  const N = 9, s = 22, x0 = 100, y0 = 22, wall = (x, y) => (x === 4 && y !== 2 && y !== 6) || (y === 4 && x !== 2 && x !== 6);
  const goals = [[4, 2], [6, 4], [7, 7]];
  const path = memo('hrl', () => { const bfs = (a, b) => { const prev = {}, q = [a], key = p => p.join(','); prev[key(a)] = null; while (q.length) { const p = q.shift(); if (key(p) === key(b)) break; for (const [dx, dy] of DIRS) { const n = [p[0] + dx, p[1] + dy]; if (n[0] < 0 || n[1] < 0 || n[0] >= N || n[1] >= N || wall(...n) || key(n) in prev) continue; prev[key(n)] = p; q.push(n); } } const out = []; let p = b; while (p) { out.unshift(p); p = prev[p.join(',')]; } return out; };
    let cur = [1, 1], full = [], seg = []; goals.forEach((g, i) => { const p = bfs(cur, g); (i ? p.slice(1) : p).forEach(q => { full.push(q); seg.push(i); }); cur = g; }); return { full, seg }; });
  const dt = .28, T = path.full.length * dt + 1.5, f = Math.min((t % T) / dt, path.full.length - 1), gi = path.seg[Math.floor(f)];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) rr(c, x0 + x * s + 1, y0 + y * s + 1, s - 2, s - 2, 3, wall(x, y) ? C.wall : C.panel);
  goals.forEach((g, i) => { if (i >= gi) txt(c, i === 2 ? '🏆' : '🚩', x0 + g[0] * s + s / 2, y0 + g[1] * s + s / 2, { s: i === gi ? 15 : 11, al: i === gi ? 1 : .4 }); });
  const q = lerpPath(path.full, f); dot(c, x0 + q[0] * s + s / 2, y0 + q[1] * s + s / 2, 7, C.pink);
  txt(c, '👔 manager', 50, 70, { s: 11, b: 1, c: C.gold }); txt(c, ['"go to door 1"', '"go to door 2"', '"get the trophy"'][gi], 50, 88, { s: 10, c: C.gold });
  txt(c, '👷 worker', 352, 70, { s: 11, b: 1, c: C.pink }); txt(c, 'steps ↑→↓←', 352, 88, { s: 10, c: C.pink });
  txt(c, 'high level picks sub-goals; low level handles the steps', 200, 232, { s: 12, c: C.dim });
};
A.marl = (c, t) => {
  const sim = memo('marl', () => { const r = mulberry(14); let A1 = [60, 120], A2 = [340, 120]; let coins = Array.from({ length: 4 }, () => [40 + r() * 320, 40 + r() * 160]); const sc = [0, 0], out = [];
    for (let i = 0; i < 900; i++) { [A1, A2].forEach((a, j) => { let best = 0, bd = 1e9; coins.forEach((q, k) => { const d = Math.hypot(q[0] - a[0], q[1] - a[1]); if (d < bd) { bd = d; best = k; } }); const q = coins[best]; const sp = j ? 2.0 : 2.2; a[0] += (q[0] - a[0]) / (bd + 1e-6) * Math.min(sp, bd); a[1] += (q[1] - a[1]) / (bd + 1e-6) * Math.min(sp, bd); if (bd < 8) { sc[j]++; coins[best] = [40 + r() * 320, 40 + r() * 160]; } });
      out.push({ a1: A1.slice(), a2: A2.slice(), coins: coins.map(q => q.slice()), sc: sc.slice() }); } return out; });
  const d = sim[Math.floor(t * 40) % sim.length];
  rr(c, 20, 25, 360, 190, 12, C.panel, C.line);
  d.coins.forEach(q => txt(c, '🪙', q[0], q[1], { s: 14 }));
  dot(c, ...d.a1, 10, C.pink); dot(c, ...d.a2, 10, C.teal);
  txt(c, `🩷 ${d.sc[0]}   vs   ${d.sc[1]} 🩵`, 200, 14, { s: 13, b: 1 });
  txt(c, 'each agent’s "environment" includes the other learning agent', 200, 232, { s: 12, c: C.dim });
  txt(c, '→ the world keeps changing under your feet', 200, 249, { s: 11, c: C.gold });
};
A.domainRand = (c, t) => {
  const k = Math.floor(t / 1.1), real = k % 6 === 5, r = mulberry(k * 7 + 1);
  const sky = real ? '#9aa4b2' : mix('#1b2a6b', '#e07a5f', r()), ground = real ? '#6b6b6b' : mix('#2d6a4f', '#c9a227', r()), fr = real ? .5 : .1 + r() * .8, mass = real ? 1 : .5 + r() * 1.5, light = real ? .5 : r();
  c.fillStyle = sky; c.fillRect(20, 20, 360, 150); c.fillStyle = ground; c.fillRect(20, 170, 360, 50);
  const u = clamp(((t % 1.1) / 1.1) * 1.4, 0, 1), dist = 140 * (1 - fr * .7) / mass * ease(u);
  const bx = 150 + dist, sz = 26 + mass * 10;
  c.globalAlpha = .35; c.fillStyle = '#000'; c.beginPath(); c.ellipse(bx + sz / 2 + (light - .5) * 30, 172, sz * .7, 5, 0, 0, 7); c.fill(); c.globalAlpha = 1;
  rr(c, bx, 170 - sz, sz, sz, 4, real ? '#c0c0c0' : mix(C.pink, C.teal, r()));
  txt(c, '🦾', 120 + Math.min(dist, 30), 155, { s: 30 });
  txt(c, real ? '🌍 REAL WORLD — looks like just another variation!' : `sim #${k + 1}`, 200, 36, { s: 13, b: 1, c: real ? C.bg : C.fg });
  txt(c, `friction ${fr.toFixed(2)}   mass ${mass.toFixed(2)}   light ${light.toFixed(2)}`, 200, 236, { s: 12, c: C.gold });
  txt(c, 'train across many randomised simulators', 200, 252, { s: 11, c: C.dim });
};

window.RLAnim = { A, AW, AH, C };
