// RL Reels story toolkit: hand-drawn canvas helpers, characters, Pip's grid world + simulation, and the reel registry.
// Canvas is 400 x 420 logical px. A reel file calls RLStory.register(id, { title, track, onCtx, steps: [...] }).
(function () {
'use strict';
const W = 400, H = 420;
const C = { ink: '#3a3632', soft: '#8a8178', gold: '#e9a93b', rose: '#e07a7a', teal: '#6fb7b4', card: '#fffaf1', bg: '#fbf3e8',
  grey: '#b9b0a6', pink: '#f2b8b8', line: '#b8ab9a', goldInk: '#8a5d08', roseInk: '#a54444', orange: '#ef8f5a', yellow: '#f6d58c', blue: '#8fb3d9' };
let ctx = null;
const lerp = (a, b, u) => a + (b - a) * u, clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const ease = u => (u = clamp(u), u < .5 ? 2 * u * u : 1 - (-2 * u + 2) ** 2 / 2);
// deterministic RNG for simulations (mulberry32): const R = rng(7); R() in [0, 1)
function rng(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gauss(R) { let u = 0, v = 0; while (!u) u = R(); while (!v) v = R(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
// jitter source for the sketchy look (stable per seed)
function rnd(seed) { let a = seed * 9301 + 49297; return () => { a = (a * 9301 + 49297) % 233280; return a / 233280 - .5; }; }
const num = (v, d) => { if (Math.abs(v) < .0005) return '0'; return String(+v.toFixed(d ?? (Math.abs(v) < .1 ? 3 : 2))); };

// ───── drawing primitives ─────
function rline(x1, y1, x2, y2, seed, o = {}) {
  const R = rnd(seed), j = o.j ?? 1.4, p = o.p ?? 1; x2 = lerp(x1, x2, p); y2 = lerp(y1, y2, p);
  ctx.strokeStyle = o.c || C.ink; ctx.lineWidth = o.w || 1.3; ctx.lineCap = 'round';
  for (let k = 0; k < (o.once ? 1 : 2); k++) {
    ctx.beginPath(); ctx.moveTo(x1 + R() * j, y1 + R() * j);
    ctx.quadraticCurveTo((x1 + x2) / 2 + R() * j * 2, (y1 + y2) / 2 + R() * j * 2, x2 + R() * j, y2 + R() * j); ctx.stroke();
  }
}
function rrect(x, y, w, h, seed, o = {}) {
  if (o.fill) { ctx.save(); ctx.globalAlpha = o.fa ?? 1; ctx.fillStyle = o.fill; ctx.fillRect(x, y, w, h); ctx.restore(); }
  rline(x, y, x + w, y, seed, o); rline(x + w, y, x + w, y + h, seed + 1, o); rline(x + w, y + h, x, y + h, seed + 2, o); rline(x, y + h, x, y, seed + 3, o);
}
function hachure(x, y, w, h, color, gap, seed, alpha = 1) {
  const R = rnd(seed); ctx.save(); ctx.beginPath(); ctx.rect(x + 2, y + 2, w - 4, h - 4); ctx.clip();
  ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = 1.6;
  for (let d = -h; d < w; d += gap) { ctx.beginPath(); ctx.moveTo(x + d + R() * 2, y + h + R() * 2); ctx.lineTo(x + d + h + R() * 2, y + R() * 2); ctx.stroke(); }
  ctx.restore();
}
function rcircle(x, y, r, seed, o = {}) {
  const R = rnd(seed); ctx.beginPath();
  for (let k = 0; k <= 22; k++) { const a = k / 22 * Math.PI * 2 + .2, rr = r + R() * (o.j ?? 1.2); k ? ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr) : ctx.moveTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr); }
  ctx.closePath(); if (o.fill) { ctx.fillStyle = o.fill; ctx.fill(); } ctx.strokeStyle = o.c || C.ink; ctx.lineWidth = o.w || 1.3; ctx.stroke();
}
function dot(x, y, r, col, a = 1) { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); ctx.restore(); }
function box(x, y, w, h, col, a = 1) { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = col; ctx.fillRect(x, y, w, h); ctx.restore(); }
// text: o = { s: size, c: colour, a: alpha, al: 'left'|'center'|'right' }
function txt(s, x, y, o = {}) { ctx.save(); ctx.fillStyle = o.c || C.ink; ctx.globalAlpha = o.a ?? 1; ctx.font = `${o.s || 16}px 'Patrick Hand', 'Didact Gothic', cursive`; ctx.textAlign = o.al || 'center'; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y); ctx.restore(); }
function arrow(x1, y1, x2, y2, seed, o = {}) {
  rline(x1, y1, x2, y2, seed, o); const a = Math.atan2(y2 - y1, x2 - x1), L = o.head || 7;
  rline(x2, y2, x2 - L * Math.cos(a - .5), y2 - L * Math.sin(a - .5), seed + 5, o); rline(x2, y2, x2 - L * Math.cos(a + .5), y2 - L * Math.sin(a + .5), seed + 6, o);
}
function alpha(a, fn) { ctx.save(); ctx.globalAlpha = a; fn(); ctx.restore(); }

// ───── note cards, bubbles, captions, charts ─────
// card(lines, t): a note card whose lines appear one by one (line = string or { s, c }). Default spot: under a 4-row grid.
function card(lines, t, o = {}) {
  const x = o.x ?? 30, y = o.y ?? 292, w = o.w ?? 340, lh = o.lh ?? 23, h = 18 + lines.length * lh, a = clamp(t * 3);
  if (a <= 0) return; box(x, y, w, h, C.card, a);
  rrect(x, y, w, h, 200 + lines.length, { c: C.line, w: 1.1 });
  lines.forEach((l, i) => { const u = clamp((t - (o.gap ?? 1.1) * i) * 2.5); if (u > 0) txt(l.s ?? l, x + 14, y + 20 + i * lh, { s: o.fs || 17, al: 'left', a: u, c: l.c }); });
}
function bubble(s, x, y, t, col) { if (t < 0) return; const u = clamp(t * 3), d = clamp(t - 1.2); txt(s, x, y - 14 * ease(u) - d * 6, { s: 20, c: col || '#b07d12', a: u * (1 - d) }); }
function caption(s, a = 1) { txt(s, W / 2, 404, { s: 15, c: C.soft, a }); }
// bars(vals, opt): simple hand-drawn bar chart in a box. opt = { x, y, w, h, max, colors|color, labels, n (how many shown), title }
function bars(vals, o = {}) {
  const x = o.x ?? 40, y = o.y ?? 300, w = o.w ?? 320, h = o.h ?? 84, n = o.n ?? vals.length, max = o.max ?? Math.max(...vals.map(Math.abs), 1e-9), bw = w / vals.length;
  rline(x, y + h, x + w, y + h, 900, { c: C.line, w: 1 }); if (o.title) txt(o.title, x, y - 8, { s: 14, al: 'left', c: C.soft });
  for (let i = 0; i < Math.min(n, vals.length); i++) { const v = vals[i], bh = clamp(Math.abs(v) / max) * (h - 4), col = o.colors ? o.colors[i] : (o.color || C.teal);
    box(x + i * bw + bw * .12, y + h - bh, bw * .76, bh, col); if (o.labels) txt(o.labels[i], x + i * bw + bw / 2, y + h + 11, { s: 13, c: C.soft }); }
}

// ───── characters ─────
function mouse(x, y, face = 1, s = 1, bob = 0) {
  ctx.save(); ctx.translate(x, y + Math.sin(bob * 9) * 1.2); ctx.scale(face * s, s);
  ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(-12, 4); ctx.bezierCurveTo(-24, 6, -22, -8, -28, -4); ctx.stroke();
  ctx.beginPath(); ctx.ellipse(0, 2, 13, 9, 0, 0, Math.PI * 2); ctx.fillStyle = C.grey; ctx.fill(); ctx.stroke();
  rcircle(4, -8, 5, 3, { fill: C.pink, j: .6 });
  ctx.beginPath(); ctx.ellipse(11, 0, 7, 6, -.3, 0, Math.PI * 2); ctx.fillStyle = C.grey; ctx.fill(); ctx.stroke();
  ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(13, -2, 1.3, 0, 7); ctx.fill(); ctx.beginPath(); ctx.arc(18, 1.5, 1.6, 0, 7); ctx.fillStyle = '#d97c8c'; ctx.fill();
  ctx.restore();
}
function cheese(x, y, s = 1) {
  ctx.save(); ctx.translate(x, y); ctx.scale(s, s);
  ctx.beginPath(); ctx.moveTo(-16, 9); ctx.lineTo(16, 9); ctx.lineTo(16, -6); ctx.closePath(); ctx.fillStyle = '#f7d36b'; ctx.fill(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3; ctx.stroke();
  ctx.fillStyle = '#e2b54a'; [[8, 3, 2.4], [12, -1, 1.5], [3, 6, 1.6]].forEach(([a, b, r]) => { ctx.beginPath(); ctx.arc(a, b, r, 0, 7); ctx.fill(); });
  ctx.restore();
}
function cat(x, y, t, s = 1) {
  ctx.save(); ctx.translate(x, y); ctx.scale(s, s);
  ctx.beginPath(); ctx.moveTo(-13, -4); ctx.lineTo(-11, -17); ctx.lineTo(-4, -9); ctx.moveTo(13, -4); ctx.lineTo(11, -17); ctx.lineTo(4, -9);
  ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3; ctx.fillStyle = '#c9b9a8'; ctx.fill(); ctx.stroke();
  rcircle(0, 0, 13, 41, { fill: '#c9b9a8', j: .8 });
  ctx.beginPath(); ctx.arc(-5, -1, 3, .2, Math.PI - .2); ctx.moveTo(8, -1); ctx.arc(5, -1, 3, .2, Math.PI - .2); ctx.stroke();
  [[-1, 5, -16, 3], [-1, 6, -16, 8], [1, 5, 16, 3], [1, 6, 16, 8]].forEach(([a, b, c, d]) => rline(a, b, c, d, a * 7 + d, { once: true, w: 1, j: .5 }));
  ctx.restore();
  const z = (t * .6) % 1; txt('Zz', x + 18 + z * 8, y - 16 - z * 12, { s: 12 + z * 4, a: (1 - z) * .9, c: C.soft });
}
function robot(x, y, s = 1, t = 0, col = '#9fc9c7') {
  ctx.save(); ctx.translate(x, y + Math.sin(t * 6) * 1); ctx.scale(s, s);
  rline(0, -16, 0, -22, 51, { once: true }); dot(0, -23, 2.2, C.rose);
  rrect(-11, -16, 22, 15, 52, { fill: col }); dot(-4, -9, 1.8, C.ink); dot(4, -9, 1.8, C.ink);
  rrect(-9, 0, 18, 13, 56, { fill: '#cfe3e2' }); rline(-9, 13, -9, 18, 60, { once: true }); rline(9, 13, 9, 18, 61, { once: true });
  ctx.restore();
}
function coin(x, y, r = 7, label = '') { rcircle(x, y, r, 70 + Math.round(x), { fill: '#f7d36b', j: .5 }); if (label) txt(label, x, y + 1, { s: r * 1.4, c: C.goldInk }); }

// ───── Pip's grid world: geometry + drawing + simulation ─────
// spec = { cols, rows, cheese: [c, r], cat: [c, r] or null, rocks: [[c, r], ...], cs: cell px, ox, oy }
function gridWorld(spec) {
  const G = Object.assign({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]], cs: 64, ox: 40, oy: 16, start: [0, 3] }, spec || {});
  const { cols, rows, cs, ox, oy } = G, MOVES = [[0, -1], [1, 0], [0, 1], [-1, 0]];
  const key = (c, r) => r * cols + c, cx = c => ox + c * cs + cs / 2, cy = r => oy + r * cs + cs / 2;
  const isRock = (c, r) => G.rocks.some(([x, y]) => x === c && y === r);
  const term = (c, r) => (G.cheese && c === G.cheese[0] && r === G.cheese[1]) ? 1 : (G.cat && c === G.cat[0] && r === G.cat[1]) ? -1 : 0;
  function step(c, r, m) { const nc = c + MOVES[m][0], nr = r + MOVES[m][1]; if (nc < 0 || nr < 0 || nc >= cols || nr >= rows || isRock(nc, nr)) return [c, r]; return [nc, nr]; }
  // score of a move = reward for stepping there + gamma * guess of that cell (0 for the ending cells)
  function score(V, c, r, m, g) { const [nc, nr] = step(c, r, m); const rw = term(nc, nr); return rw + (rw ? 0 : g * V[key(nc, nr)]); }
  function moves(c, r) { const out = []; for (let m = 0; m < 4; m++) { const [nc, nr] = step(c, r, m); if (nc !== c || nr !== r) out.push(m); } return out; }
  function greedy(V, c, r, g) { let best = -1e9, bm = 0; for (const m of moves(c, r)) { const q = score(V, c, r, m, g); if (q > best + 1e-12) { best = q; bm = m; } } return bm; }
  function pick(V, c, r, g, R) { let best = -1e9, ms = []; for (const m of moves(c, r)) { const q = score(V, c, r, m, g); if (q > best + 1e-9) { best = q; ms = [m]; } else if (q > best - 1e-9) ms.push(m); } return ms[Math.floor(R() * ms.length)]; }
  // TD(0) with Pip's habit (peek + score, eps random). Records every transition. Returns { eps: [{ tr, end, V }] }
  function runTD(o = {}) {
    const { gamma = .9, alpha = .2, episodes = 200, seed = 7, eps: ex = .2, maxSteps = 400 } = o, R = rng(seed), V = new Float64Array(cols * rows), eps = [];
    for (let e = 0; e < episodes; e++) {
      let [c, r] = G.start; const tr = [];
      for (let k = 0; k < maxSteps; k++) {
        const m = R() < ex ? Math.floor(R() * 4) : pick(V, c, r, gamma, R), [nc, nr] = step(c, r, m), rew = term(nc, nr), done = rew !== 0;
        const old = V[key(c, r)], next = done ? 0 : V[key(nc, nr)], target = rew + gamma * next, nw = old + alpha * (target - old);
        V[key(c, r)] = nw; tr.push({ c, r, nc, nr, m, rew, old, next, target, nw }); c = nc; r = nr; if (done) break;
      }
      eps.push({ tr, end: tr[tr.length - 1].rew, V: Array.from(V) });
    }
    return { eps, gamma, alpha };
  }
  function greedyPath(V, g, from) { let [c, r] = from || G.start; const P = [[c, r]]; for (let k = 0; k < 30 && !term(c, r); k++) { [c, r] = step(c, r, greedy(V, c, r, g)); P.push([c, r]); } return P; }
  // drawing: maze({ V, nums, t, hi: [[c, r, colour, alpha]], draw: 0..1 build-in, pop: { cheese, cat }, labels: alpha, numA })
  function maze(o = {}) {
    const p = o.draw ?? 1, V = o.V, t = o.t || 0;
    if (V) for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const v = V[key(c, r)]; if (term(c, r) || isRock(c, r) || Math.abs(v) < .004) continue;
      const m = clamp(Math.abs(v) / .8); hachure(ox + c * cs, oy + r * cs, cs, cs, v >= 0 ? C.gold : C.rose, lerp(13, 4.5, m), 100 + c * 7 + r, lerp(.35, .9, m));
    }
    if (o.hi) o.hi.forEach(([c, r, col, a]) => box(ox + c * cs + 3, oy + r * cs + 3, cs - 6, cs - 6, col, a ?? .35));
    for (let k = 0; k <= rows; k++) rline(ox, oy + k * cs, ox + cols * cs, oy + k * cs, 10 + k, { p: p >= 1 ? 1 : clamp(p * 1.6 - k * .12), w: k % rows ? 1 : 1.8, c: k % rows ? C.line : C.ink });
    for (let k = 0; k <= cols; k++) rline(ox + k * cs, oy, ox + k * cs, oy + rows * cs, 30 + k, { p: p >= 1 ? 1 : clamp(p * 1.6 - .3 - k * .1), w: k % cols ? 1 : 1.8, c: k % cols ? C.line : C.ink });
    if (p >= .8) G.rocks.forEach(([c, r], i) => { const x = cx(c), y = cy(r); rcircle(x, y + 4, cs * .3, 60 + i, { fill: '#d9d1c6', j: 4 }); hachure(x - cs * .28, y - cs * .2, cs * .56, cs * .53, '#9d9387', 5, 70 + i, .55); });
    if (G.cheese && o.pop?.cheese !== 0) cheese(cx(G.cheese[0]), cy(G.cheese[1]), (o.pop?.cheese ?? 1) * cs / 64);
    if (G.cat && o.pop?.cat !== 0) cat(cx(G.cat[0]), cy(G.cat[1]) + 4, t, (o.pop?.cat ?? 1) * cs / 64);
    if (o.labels) { const a = o.labels; if (G.cheese) txt('+1 on arrival', cx(G.cheese[0]) - 6, cy(G.cheese[1]) - cs * .34, { s: 13, c: '#b07d12', a }); if (G.cat) txt('−1', cx(G.cat[0]) - 20, cy(G.cat[1]) - 20, { s: 15, c: '#b54b4b', a }); txt('start', cx(G.start[0]), cy(G.start[1]) + 24, { s: 13, c: C.soft, a }); }
    if (V && o.nums) for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      if (term(c, r) || isRock(c, r)) continue; const k = key(c, r), a = typeof o.nums === 'function' ? o.nums(c, r) : 1; if (a <= 0) continue;
      const v = V[k]; txt(num(v), ox + c * cs + cs - 7, oy + r * cs + 12, { s: 14, al: 'right', a: a * (o.numA?.(c, r) ?? 1), c: v < -.004 ? C.roseInk : v > .004 ? C.goldInk : C.soft });
    }
  }
  function trail(pts, a = .5) { pts.forEach(([c, r], i) => dot(cx(c) + ((i * 7) % 13) - 6, cy(r) + ((i * 11) % 13) - 6, 2.2, C.teal, a * (.25 + .75 * i / pts.length))); }
  // position of a walker moving along transitions tr[from..to) with dur seconds per step
  function walkPos(tr, from, to, t, dur) {
    const k = Math.floor(t / dur), u = ease((t % dur) / dur * 1.4);
    if (k >= to - from) { const l = tr[to - 1]; return { c: l.nc, r: l.nr, x: cx(l.nc), y: cy(l.nr), done: true, k: to - from, face: Math.sign(l.nc - l.c) || 1 }; }
    const s = tr[from + k]; return { x: lerp(cx(s.c), cx(s.nc), u), y: lerp(cy(s.r), cy(s.nr), u), k, s, face: Math.sign(s.nc - s.c) || 0, u };
  }
  let lastFace = 1; function pip(x, y, face, t) { if (face) lastFace = face; mouse(x, y + 3, lastFace, .95 * cs / 64, t); }
  function arrowIn(c, r, m, seed, o = {}) { const [dx, dy] = MOVES[m]; arrow(cx(c) - dx * 9, cy(r) - dy * 9 + 4, cx(c) + dx * 15, cy(r) + dy * 15 + 4, seed, Object.assign({ c: C.goldInk, w: 1.5 }, o)); }
  return Object.assign(G, { MOVES, key, cx, cy, isRock, term, step, score, moves, greedy, pick, runTD, greedyPath, maze, trail, walkPos, pip, arrowIn });
}

const reels = {};
window.RLStory = { reels, register(id, r) { r.id = id; reels[id] = r; },
  lib: { W, H, C, lerp, clamp, ease, rng, gauss, rnd, num, rline, rrect, hachure, rcircle, dot, box, txt, arrow, alpha, card, bubble, caption, bars,
    mouse, cheese, cat, robot, coin, gridWorld, setCtx(c) { ctx = c; }, get ctx() { return ctx; } } };
})();
