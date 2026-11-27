// ===== bandit_kit.js (copy) =====
// Shared Bandits-track kit: Pip's cheese arcade (4 machines A-D, crumb or nothing), seeded games, strategies, drawing.
// See briefs/notes/bandit_world.md. Safe to load more than once.
(function () {
'use strict';
const L = RLStory.lib, { C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, dot, box, alpha } = L;
const cx = () => L.ctx;
const K = 4, P = [0.30, 0.55, 0.70, 0.15], NAMES = ['A', 'B', 'C', 'D'], BEST = 2, PBEST = 0.70;
const COL = ['#8fb3d9', '#ef8f5a', '#6fb7b4', '#e07a7a'], INK = ['#4f74a0', '#b8602c', '#3f8582', '#a54444'];
const X = [62, 154, 246, 338], Y0 = 70, MW = 64, MH = 110;

// ───── simulation ─────
// crumb tape of game g: the n-th pull of machine k pays rng(100000 * (k + 1) + g)'s n-th draw < P[k]
function tape(g, probs) { const pr = probs || P; return pr.map((p, k) => { const R = rng(100000 * (k + 1) + g); return () => (R() < p ? 1 : 0); }); }
// play one game. policy(st, R, t) -> arm, st = { n, q, t, K } before the pull. Returns per-pull records.
function play(policy, g, T, o = {}) {
  const pr = o.P || P, best = Math.max(...pr), tp = tape(g, pr), R = rng(7000000 + g), n = new Array(K).fill(0), q = new Array(K).fill(0);
  const arms = [], rews = [], est = [], cnt = [], regret = []; let reg = 0;
  for (let t = 0; t < T; t++) {
    const a = policy({ n, q, t, K }, R, t), r = tp[a](); n[a]++; q[a] += (r - q[a]) / n[a]; reg += best - pr[a];
    arms.push(a); rews.push(r); est.push(q.slice()); cnt.push(n.slice()); regret.push(reg);
  }
  return { arms, rews, est, n: cnt, regret };
}
function argmaxR(q, R) { let m = -1e9, c = []; for (let k = 0; k < q.length; k++) { if (q[k] > m + 1e-12) { m = q[k]; c = [k]; } else if (q[k] > m - 1e-12) c.push(k); } return c.length === 1 ? c[0] : c[Math.floor(R() * c.length)]; }
// strategies: first K pulls try each machine once (A, B, C, D), then their own rule
const greedy = (st, R) => (st.t < K ? st.t : argmaxR(st.q, R));
const eps = e => (st, R) => (st.t < K ? st.t : (R() < e ? Math.floor(R() * K) : argmaxR(st.q, R)));
const epsDecay = f => (st, R) => (st.t < K ? st.t : (R() < f(st.t) ? Math.floor(R() * K) : argmaxR(st.q, R)));
const ucb = c => (st, R) => (st.t < K ? st.t : argmaxR(st.q.map((v, k) => v + c * Math.sqrt(Math.log(st.t) / st.n[k])), R));
function betaS(R, a, b) { const x = gammaS(R, a); return x / (x + gammaS(R, b)); }
function gammaS(R, a) { if (a < 1) return gammaS(R, a + 1) * Math.pow(R(), 1 / a); const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) { let x, v; do { x = L.gauss(R); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = R(); if (u < 1 - .0331 * x ** 4 || Math.log(u) < .5 * x * x + d * (1 - v + Math.log(v))) return d * v; } }
const thompson = (st, R) => argmaxR(st.q.map((v, k) => { const w = Math.round(v * st.n[k]); return betaS(R, 1 + w, 1 + st.n[k] - w); }), R);
// mean cumulative regret over games 0..games-1 (and optional extra stats)
function avgRegret(policy, games, T, o = {}) {
  const out = new Float64Array(T), share = new Float64Array(K); let bestMost = 0;
  for (let g = 0; g < games; g++) { const G = play(policy, g, T, o);
    for (let t = 0; t < T; t++) out[t] += G.regret[t] / games; const nn = G.n[T - 1]; nn.forEach((v, k) => share[k] += v / T / games); if (nn[BEST] > T / 2) bestMost++; }
  out.share = Array.from(share); out.bestMost = bestMost / games; return out;
}

// ───── drawing ─────
function crumb(x, y, s = 1) { const c = cx(); c.save(); c.translate(x, y); c.scale(s, s);
  c.beginPath(); c.moveTo(-7, 4); c.lineTo(7, 4); c.lineTo(7, -3); c.closePath(); c.fillStyle = '#f7d36b'; c.fill(); c.strokeStyle = C.ink; c.lineWidth = 1.1; c.stroke();
  c.fillStyle = '#e2b54a'; c.beginPath(); c.arc(3.5, 1.5, 1.2, 0, 7); c.fill(); c.restore(); }
// one machine at centre x. o = { pull 0..1, crumb 0..1 (falling then resting), miss 0..1, glow 0..1, secret 0..1 (show hidden chance), a, label }
function machine(x, k, o = {}) {
  const a = o.a ?? 1; if (a <= 0) return;
  const l = x - MW / 2, top = Y0, s = 40 + k * 17;
  if (o.glow) box(l - 6, top - 6, MW + 12, MH + 12, C.yellow, .55 * o.glow);
  rrect(l, top, MW, MH, s, { fill: C.card, w: 1.5 });
  box(l + 2, top + 2, MW - 4, 16, COL[k], .9); rline(l, top + 18, l + MW, top + 18, s + 5, { w: 1.1 });
  txt(o.label ?? NAMES[k], x, top + 10, { s: 16, c: C.ink });
  // window with a cheese wedge inside
  rrect(l + 10, top + 26, MW - 20, 34, s + 7, { fill: '#fdf6e6', c: C.line, w: 1 }); L.cheese(x, top + 45, .62);
  // chute and tray
  rrect(x - 11, top + 70, 22, 12, s + 9, { fill: '#5b544d', w: 1 });
  rrect(l + 8, top + MH - 22, MW - 16, 16, s + 11, { fill: '#efe3cf', w: 1.1 });
  // lever on the right side: pivot at (l + MW, top + 52)
  const px = l + MW + 1, py = top + 52, ang = lerp(-1.15, 1.0, ease(o.pull || 0)), ex = px + Math.cos(ang) * 22 + 2, ey = py + Math.sin(ang) * 22;
  rline(px, py, ex, ey, s + 13, { w: 2.2 }); rcircle(ex, ey, 4.5, s + 14, { fill: C.rose, j: .4 }); dot(px, py, 2.5, C.ink);
  // crumb falls from the chute into the tray
  if (o.crumb > 0) { const u = clamp(o.crumb * 1.6); crumb(x, lerp(top + 78, top + MH - 15, ease(u)), .95); }
  if (o.miss > 0) { const u = clamp(o.miss); txt('nothing', x, top + MH - 14, { s: 13, c: C.soft, a: clamp(u * 3) * (1 - clamp((u - .8) * 5)) }); }
  if (o.secret > 0) { txt(Math.round(P[k] * 100) + '%', x, top + MH + 14, { s: 15, c: INK[k], a: o.secret }); }
  if (a < 1) box(l - 8, top - 8, MW + 34, MH + 16, C.bg, 1 - a);   // fade = cover with the background (lib text sets its own alpha)
}
function arcade(o = {}) {
  const c = cx(); rline(14, Y0 + MH + 2, 386, Y0 + MH + 2, 31, { c: C.line, w: 1.2 });
  if (o.sign !== false) txt(o.sign || 'cheese arcade', 200, 34, { s: 18, c: C.soft, a: o.signA ?? 1 });
  for (let k = 0; k < K; k++) machine(X[k], k, Object.assign({}, o.all || {}, o.per ? o.per(k) || {} : {}));
}
// wins / pulls under a machine
function tally(k, wins, pulls, y, o = {}) { txt(pulls ? `${wins} of ${pulls}` : 'untried', X[k], y, { s: o.s || 14, c: o.c || C.soft, a: o.a ?? 1 }); }
// estimate bars: one bar per machine at X[k], height = q * h (max 1), value label on top. o = { y (baseline), h, a, show: k => bool, truth: show true chance marks, hi: k or [k, ...] highlighted, fmt }
function estBars(q, o = {}) {
  const y = o.y ?? 388, h = o.h ?? 80, bw = o.bw ?? 30, a = o.a ?? 1;
  if (a <= 0) return; (() => { rline(24, y, 376, y, 77, { c: C.line, w: 1 });
    for (let k = 0; k < K; k++) { if (o.show && !o.show(k)) continue; const v = q[k], bh = clamp(v) * h;
      if (Array.isArray(o.hi) ? o.hi.includes(k) : o.hi === k) box(X[k] - bw / 2 - 4, y - h - 6, bw + 8, h + 6, C.yellow, .45);
      if (bh > 0) box(X[k] - bw / 2, y - bh, bw, bh, COL[k], .85);
      if (o.truth) { const ty = y - P[k] * h; rline(X[k] - bw / 2 - 5, ty, X[k] + bw / 2 + 5, ty, 80 + k, { c: C.ink, w: 1.3, once: true, j: .3 }); }
      if (o.labels !== false) txt(o.fmt ? o.fmt(v, k) : (+v.toFixed(2)).toString(), X[k], y - bh - 10, { s: 14, c: INK[k] });
      txt(NAMES[k], X[k], y + 10, { s: 13, c: C.soft }); } })();
  if (a < 1) box(20, y - h - 22, 360, h + 36, C.bg, 1 - a);
}
RLStory.bandit = { K, P, NAMES, COL, INK, BEST, PBEST, X, Y0, MW, MH, tape, play, argmaxR, greedy, eps, epsDecay, ucb, thompson, betaS, avgRegret, crumb, machine, arcade, tally, estBars };
})();
// ===== end bandit_kit =====

// Contextual bandits as a story: Pip's cheese arcade stays open late, and the best machine depends on the hour.
// World, dice and drawing from the shared bandit kit (copied above); this reel changes the rule: chances drift with the hour.
(function () {
'use strict';
const L = RLStory.lib, B = RLStory.bandit, { W, C, lerp, clamp, ease, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
const { K, NAMES, COL, INK, X, Y0, MH } = B;
let ctx;

// ───── the late arcade ─────
// Open 10:00 to 22:00; each pull happens in one of the 12 opening hours (10, 11, …, 21), picked at random.
// At 10:00 the chances are the plain arcade's; by 21:00 they are P1; in between they move in a straight line.
const P0 = [.30, .55, .70, .15], P1 = [.25, .45, .25, .80], HOURS = Array.from({ length: 12 }, (_, i) => 10 + i);
const zOf = h => (h - 10) / 11, chance = (k, h) => P0[k] + (P1[k] - P0[k]) * zOf(h);
const bestAt = h => { let b = 0; for (let k = 1; k < K; k++) if (chance(k, h) > chance(b, h)) b = k; return b; };
const ORACLE = HOURS.reduce((s, h) => s + chance(bestAt(h), h), 0) / 12, AVG = [0, 1, 2, 3].map(k => HOURS.reduce((s, h) => s + chance(k, h), 0) / 12);
// cost of lumping: a two-row player that always pulls its row's best-on-average machine, regret per 1000 pulls
const ROWBEST = [[10, 15], [16, 21]].map(([a, b]) => { let bk = 0, bv = -1; for (let k = 0; k < 4; k++) { let v = 0; for (let h = a; h <= b; h++) v += chance(k, h); if (v > bv) { bv = v; bk = k; } } return bk; });
const LUMP = 1000 * HOURS.reduce((s_, h) => { const k = ROWBEST[h < 16 ? 0 : 1]; let b = 0; for (let j = 1; j < 4; j++) if (chance(j, h) > chance(b, h)) b = j; return s_ + chance(b, h) - chance(k, h); }, 0) / 12;
const hh = h => (h < 10 ? '0' : '') + h + ':00';

// ───── strategies ─────
// 'blind': UCB1 ignoring the clock. 'row2': UCB1 run separately in two rows (10–15, 16–21). 'row12': one row per hour.
// UCB1 in a row: pull each machine once in that row (A, B, C, D), then estimate + √(2 ln t / n), t and n counted inside the row.
// 'lin': LinUCB. For each machine, features x = (1, z) with z = (hour − 10) / 11; M = I + Σ x xᵀ, v = Σ reward · x over its pulls;
// line = M⁻¹ v (a least-squares line pulled gently toward 0); score = line(z) + α √(xᵀ M⁻¹ x), α = 1. Ties at random (kit argmaxR).
const rowOf = (kind, h) => kind === 'blind' ? 0 : kind === 'row2' ? (h < 16 ? 0 : 1) : h - 10;
function linPred(m, z) { const [a11, a12, a22] = [m.M[0], m.M[1], m.M[2]], det = a11 * a22 - a12 * a12, i11 = a22 / det, i12 = -a12 / det, i22 = a11 / det;
  const th0 = i11 * m.v[0] + i12 * m.v[1], th1 = i12 * m.v[0] + i22 * m.v[1]; return { pred: th0 + th1 * z, bon: Math.sqrt(i11 + 2 * i12 * z + i22 * z * z), th0, th1 }; }
function play(kind, g, T, o = {}) {
  const raw = [0, 1, 2, 3].map(k => L.rng(100000 * (k + 1) + g)), Rc = L.rng(5000000 + g), R = L.rng(7000000 + g);
  const rows = kind === 'row12' ? 12 : kind === 'row2' ? 2 : 1, n = Array.from({ length: rows }, () => [0, 0, 0, 0]), q = Array.from({ length: rows }, () => [0, 0, 0, 0]), w = Array.from({ length: rows }, () => [0, 0, 0, 0]), tr = new Array(rows).fill(0);
  const lin = [0, 1, 2, 3].map(() => ({ M: [1, 0, 1], v: [0, 0] })), c = o.c ?? 1, al = o.alpha ?? 1;
  const reg = new Float64Array(T), rec = o.rec ? { hours: [], arms: [], rews: [], snaps: {} } : null; let rg = 0;
  for (let t = 0; t < T; t++) {
    const h = 10 + Math.floor(Rc() * 12), z = zOf(h); let a;
    if (kind === 'lin') a = B.argmaxR(lin.map(m => { const p = linPred(m, z); return p.pred + al * p.bon; }), R);
    else { const r = rowOf(kind, h), u = n[r].indexOf(0); a = u >= 0 ? u : B.argmaxR(q[r].map((v, k) => v + c * Math.sqrt(2 * Math.log(tr[r]) / n[r][k])), R); }
    const p = chance(a, h), rw = raw[a]() < p ? 1 : 0; rg += chance(bestAt(h), h) - p; reg[t] = rg;
    if (kind === 'lin') { const m = lin[a]; m.M[0] += 1; m.M[1] += z; m.M[2] += z * z; m.v[0] += rw; m.v[1] += rw * z; }
    else { const r = rowOf(kind, h); n[r][a]++; w[r][a] += rw; q[r][a] += (rw - q[r][a]) / n[r][a]; tr[r]++; }
    if (rec) { rec.hours.push(h); rec.arms.push(a); rec.rews.push(rw); if (o.snapAt && o.snapAt.includes(t + 1)) rec.snaps[t + 1] = { n: n.map(x => x.slice()), w: w.map(x => x.slice()), lin: lin.map(m => ({ M: m.M.slice(), v: m.v.slice() })) }; }
  }
  return { reg, rec, n, w, lin };
}
function avg(kind, games, T, o = {}) { const out = new Float64Array(T); for (let g = 0; g < games; g++) { const r = play(kind, g, T, o).reg; for (let t = 0; t < T; t++) out[t] += r[t] / games; } return out; }
const memo = {}, M = (key, f) => memo[key] || (memo[key] = f());
const GAMES = 2000, T = 1000, VISIT = 3;
const AV = kind => M(kind, () => avg(kind, GAMES, T));
const V2 = M('v2', () => play('row2', VISIT, T, { rec: true, snapAt: [40, 1000] })), VL = M('vl', () => play('lin', VISIT, T, { rec: true, snapAt: [100, 1000] }));
const r0 = v => String(Math.round(v)), P100 = v => Math.round(+(v * 100).toFixed(6)), pc = v => P100(v) + ' percent', pct = v => P100(v) + '%';

// ───── drawing helpers ─────
const pip = (x, t, face = 1) => L.mouse(x, 232, face, .9, t);
function clock(h, x = 34, y = 36, a = 1) { alpha(a, () => { rcircle(x, y, 17, 970, { fill: C.card, w: 1.4 }); const ang = ((h % 12) / 12) * Math.PI * 2 - Math.PI / 2;
  rline(x, y, x + Math.cos(ang) * 9, y + Math.sin(ang) * 9, 971, { w: 2, once: true }); rline(x, y, x, y - 13, 972, { w: 1.3, once: true }); dot(x, y, 2, C.ink);
  txt(hh(h), x + 24, y + 1, { s: 15, c: C.ink, al: 'left' }); }); }
// chance-vs-hour chart. o = { x0, x1, y0, y1, show: k => alpha, best: alpha, hour: marker, lines: [{ th0, th1, bonusFn, k }] }
function hourChart(o = {}) {
  const x0 = o.x0 ?? 60, x1 = o.x1 ?? 340, y0 = o.y0 ?? 250, y1 = o.y1 ?? 80, HX = h => x0 + (h - 10) / 11 * (x1 - x0), PY = p => y0 - p * (y0 - y1);
  rline(x0, y0, x1 + 8, y0, 980, { c: C.ink, w: 1.2 }); rline(x0, y0, x0, y1 - 8, 981, { c: C.ink, w: 1.2 });
  txt('chance of a crumb', x0 - 4, y1 - 18, { s: 13, c: C.soft, al: 'left' }); [0, .5, 1].forEach(v => txt(String(v), x0 - 6, PY(v), { s: 12, c: C.soft, al: 'right' }));
  [10, 13, 16, 19, 21].forEach(h => txt(hh(h), HX(h), y0 + 11, { s: 12, c: C.soft })); txt('hour', x1 + 8, y0 + 26, { s: 13, c: C.soft, al: 'right' });
  if (o.true !== false) for (let k = 0; k < K; k++) { const a = o.show ? o.show(k) : 1; if (a <= 0) continue;
    alpha(a * (o.trueA ?? 1), () => { if (o.dash) { ctx.save(); ctx.setLineDash([5, 4]); } ctx.strokeStyle = INK[k]; ctx.lineWidth = o.dash ? 1.4 : 2.4; ctx.beginPath(); ctx.moveTo(HX(10), PY(chance(k, 10))); ctx.lineTo(HX(21), PY(chance(k, 21))); ctx.stroke(); if (o.dash) ctx.restore();
      if (o.labels !== false) txt(NAMES[k], HX(21) + 10, PY(chance(k, 21)) + (k === 0 ? 6 : k === 2 ? -6 : 0), { s: 14, c: INK[k] }); }); }
  if (o.best > 0) alpha(o.best, () => { for (let h = 10; h < 22; h++) { const b = bestAt(h); dot(HX(h), PY(chance(b, h)), 4.5, INK[b]); } });
  if (o.lines) o.lines.forEach(ln => { const k = ln.k, a = ln.a ?? 1; ctx.save(); ctx.globalAlpha = a * (ln.bandA ?? .2); ctx.fillStyle = COL[k]; ctx.beginPath();
      for (let i = 0; i <= 22; i++) { const h = 10 + i / 2, p = linPred(ln.m, zOf(h)); i ? ctx.lineTo(HX(h), PY(clamp(p.pred + p.bon * (ln.al ?? 1), -.05, 1.1))) : ctx.moveTo(HX(h), PY(clamp(p.pred + p.bon * (ln.al ?? 1), -.05, 1.1))); }
      for (let i = 22; i >= 0; i--) { const h = 10 + i / 2, p = linPred(ln.m, zOf(h)); ctx.lineTo(HX(h), PY(clamp(p.pred - p.bon * (ln.al ?? 1), -.05, 1.1))); } ctx.closePath(); if (ln.band !== false) ctx.fill();
      ctx.globalAlpha = a; ctx.strokeStyle = INK[k]; ctx.lineWidth = 2.4; ctx.beginPath(); ctx.moveTo(HX(10), PY(linPred(ln.m, 0).pred)); ctx.lineTo(HX(21), PY(linPred(ln.m, 1).pred)); ctx.stroke(); ctx.restore(); });
  if (o.hour != null) { const xh = HX(o.hour); ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.soft; ctx.beginPath(); ctx.moveTo(xh, y0); ctx.lineTo(xh, y1 - 4); ctx.stroke(); ctx.restore(); }
  return { HX, PY };
}
function regChart(curves, o = {}) {
  const x0 = 58, x1 = 360, y0 = o.y0 ?? 250, y1 = o.y1 ?? 80, max = o.max ?? 160, X_ = i => x0 + (i + 1) / T * (x1 - x0), Y_ = v => y0 - v / max * (y0 - y1);
  rline(x0, y0, x1 + 6, y0, 760, { c: C.ink, w: 1.2 }); rline(x0, y0, x0, y1 - 8, 761, { c: C.ink, w: 1.2 });
  txt('total regret', x0 - 4, y1 - 18, { s: 13, c: C.soft, al: 'left' }); txt('pulls', x1 + 4, y0 + 22, { s: 13, c: C.soft, al: 'right' });
  [0, max / 2, max].forEach(v => txt(String(v), x0 - 6, Y_(v), { s: 12, c: C.soft, al: 'right' })); [0, 500, 1000].forEach(v => txt(v === 1000 ? '1,000' : String(v), x0 + v / T * (x1 - x0), y0 + 10, { s: 12, c: C.soft }));
  curves.forEach(cv => { const n = Math.floor(clamp(cv.n ?? 1) * T); if (n < 2) return; ctx.save(); ctx.strokeStyle = cv.col; ctx.lineWidth = 2.2; ctx.beginPath();
    for (let i = 0; i < n; i += 6) i ? ctx.lineTo(X_(i), Y_(cv.v[i])) : ctx.moveTo(X_(i), Y_(cv.v[i])); ctx.lineTo(X_(n - 1), Y_(cv.v[n - 1])); ctx.stroke(); ctx.restore();
    if (n >= T) { txt(cv.label, X_(T - 1) - 4, Y_(cv.v[T - 1]) + (cv.ly ?? -12), { s: 14, c: cv.col, al: 'right' }); txt(r0(cv.v[T - 1]), X_(T - 1) + 6, Y_(cv.v[T - 1]), { s: 13, c: cv.col, al: 'left' }); } });
}
const GREY = '#8a8178', GOLD = '#c48a1a', TEAL = '#3f8582', PLUM = '#7a5aa6';
// table of tallies: rows = [{ label, n: [..], w: [..] }]
function table(rows, o = {}) { const y0 = o.y ?? 300, rh = o.rh ?? 34;
  rows.forEach((r, i) => { const y = y0 + i * rh, a = o.a ? o.a(i) : 1; if (a <= 0) return;
    alpha(a, () => { if (o.hi === i) box(20, y - rh / 2 + 2, 362, rh - 4, C.yellow, .45); txt(r.label, 24, y - 9, { s: 13, c: C.soft, al: 'left' });
      for (let k = 0; k < K; k++) { const best = o.mark && r.n[k] === Math.max(...r.n) && r.n[k] > 0; txt(r.n[k] ? `${r.w[k]} of ${r.n[k]}` : '–', X[k], y + 6, { s: 15, c: INK[k] }); if (best) rrect(X[k] - 34, y - 6, 68, 24, 990 + i * 4 + k, { c: INK[k], w: 1.2 }); } }); }); }

// ───── numbers quoted in the text ─────
const sim = () => ({ lump: LUMP, rowBest: ROWBEST, yourPick: HOURS.map(h => NAMES[linBest(LIN1000, h)]).join(''), oracle: ORACLE, avg: AVG, best: HOURS.map(h => NAMES[bestAt(h)]).join(''), blind: AV('blind')[T - 1], row2: AV('row2')[T - 1], row12: AV('row12')[T - 1], lin: AV('lin')[T - 1],
  v2_40: V2.rec.snaps[40], v2_1000: V2.rec.snaps[1000], v2hours: V2.rec.hours.slice(0, 12), vl1000: VL.rec.snaps[1000].lin.map(m => { const a = linPred(m, 0), b = linPred(m, 1); return [a.pred, b.pred, m.M[0] - 1]; }) });
const ROWLAB = ['10:00 to 15:00', '16:00 to 21:00'];
const rowsAt = snap => snap.n.map((n, i) => ({ label: ROWLAB[i], n, w: snap.w[i] }));
const LIN1000 = VL.rec.snaps[1000].lin, LIN100 = VL.rec.snaps[100].lin;
const linBest = (lin, h) => { let b = 0, s = -1e9; lin.forEach((m, k) => { const p = linPred(m, zOf(h)), v = p.pred + p.bon; if (v > s) { s = v; b = k; } }); return b; };

// ───── the steps ─────
const STEPS = [
 { h: 'Open late', p: ['The **cheese arcade** now stays open from 10 in the morning until 10 at night. One new rule: **the chances drift with the hour**. Each pull happens in one of the 12 opening hours, 10:00 to 21:00.', 'At 10:00 they are the ones from the earlier reels: A 30, B 55, C 70, D 15 percent. Through the day C fades and D wakes up. By 21:00: A 25, B 45, C 25, D 80 percent.'],
   shot: [3, 10],
   draw(t) { const h = Math.min(21, 10 + Math.floor(Math.max(0, t - 2) / .6)); B.arcade({ sign: 'cheese arcade, open late', per: k => ({ a: 1 }) }); clock(h); pip(200, t);
     for (let k = 0; k < K; k++) txt(pct(chance(k, h)), X[k], Y0 + MH + 14, { s: 15, c: INK[k] });
     if (t > 9.5) caption('chances shown here for us; Pip can’t see them', clamp(t - 9.5)); } },

 { h: 'Who is best, and when', p: () => [`Here are the four chances through the day. Left to right: the hour. Upward: the chance of a crumb. Each line is one machine, moving in a straight line from morning to night.`, `The dots mark the best machine each hour: C until 14:00, then B at 15:00, then D from 16:00. Pulling the best machine for every hour would drop a crumb on ${pc(ORACLE)} of pulls.`],
   draw(t) { hourChart({ show: k => clamp((t - .8 - k * .6) * 2), best: clamp(t - 4.5), y0: 260, y1: 90 });
     if (t > 7) card([`best for the hour, every hour:  ${pc(ORACLE)}`, { s: 'C mornings · B at 15:00 · D evenings', c: C.soft }], t - 7, { y: 300 }); } },

 { h: 'One winner for everyone?', p: () => [`Each pull now happens at a random opening hour. A plain bandit ignores the clock and hunts for the single best machine **on average**: that is B, at ${pc(AVG[1])}, with C and D close behind at ${pc(AVG[2])}.`, `Even if it found B perfectly, it would get a crumb on ${pc(AVG[1])} of pulls, while the best-for-the-hour plan gets ${pc(ORACLE)}. One winner for everyone leaves crumbs on the table.`],
   draw(t) { const bx = [80, 160, 240, 320]; txt('average chance over the day', 200, 80, { s: 15, c: C.soft, a: clamp(t - .3) });
     for (let k = 0; k < K; k++) { const u = clamp((t - .8 - k * .4) * 2), hgt = AVG[k] * 150 * u; box(bx[k] - 22, 250 - hgt, 44, hgt, COL[k], .9); txt(NAMES[k], bx[k], 262, { s: 15, c: INK[k] }); if (u >= 1) txt(pct(AVG[k]), bx[k], 250 - hgt - 12, { s: 15, c: INK[k] }); }
     rline(40, 250, 360, 250, 995, { c: C.ink }); if (t > 4.5) { const y = 250 - ORACLE * 150; ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = C.ink; ctx.globalAlpha = clamp(t - 4.5); ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(360, y); ctx.stroke(); ctx.restore(); txt(`best for the hour: ${pct(ORACLE)}`, 356, y - 11, { s: 14, al: 'right', a: clamp(t - 4.5) }); } } },

 { h: 'Look at the clock first', p: ['Information you can see **before** choosing is called the **context**: here, the hour. It is like a state that your pulls cannot change. A **contextual bandit** sees the context, picks an arm, gets a reward, and learns which arm is best for each kind of situation.', 'One thing stays simple: Pip’s pull doesn’t change what time the next pull happens. When actions shape future situations, you need full reinforcement learning.'],
   draw(t) { const steps = [['see the context', 'it is 18:00'], ['pick a machine', 'D, say'], ['get a reward', 'crumb or nothing']];
     steps.forEach(([a, b], i) => { const u = clamp((t - .6 - i * 1.4) * 2), x = 70 + i * 130; if (u <= 0) return; alpha(u, () => { rrect(x - 52, 110, 104, 64, 1000 + i, { fill: C.card }); txt(a, x, 132, { s: 16 }); txt(b, x, 156, { s: 14, c: C.soft }); if (i) arrow(x - 78, 142, x - 56, 142, 1010 + i, { c: C.soft }); }); });
     if (t > 4.5) { clock(18, 70, 230, clamp(t - 4.5)); B.crumb(330, 236, 1.6); }
     if (t > 6) card(['the next hour does not depend on the pull', { s: 'that is what keeps it a bandit, not full RL', c: C.soft }], t - 6, { y: 300 }); } },

 { h: 'A row for each part of the day', p: ['The simplest fix: keep **separate tallies for each context**. Split the day into two rows, 10:00 to 15:00 and 16:00 to 21:00, and run UCB in each row on its own, as if each were its own arcade.', 'Each row starts from scratch: it first pulls every machine once, then follows estimate plus bonus, using only that row’s pulls.'],
   draw(t) { const dur = .9, i = clamp(Math.floor(t / dur), 0, 39), u = t / dur - i, h = V2.rec.hours[i], a = V2.rec.arms[i], r = V2.rec.rews[i];
     B.arcade({ per: k => k === a ? { pull: clamp(u * 3) * (1 - clamp((u - .7) * 3)), crumb: r ? clamp((u - .3) * 1.6) : 0, miss: r ? 0 : clamp((u - .3) * 1.6) } : {} }); clock(h); pip(X[a], t);
     const nn = [[0, 0, 0, 0], [0, 0, 0, 0]], ww = [[0, 0, 0, 0], [0, 0, 0, 0]]; for (let j = 0; j < i + (u > .6 ? 1 : 0); j++) { const rr = V2.rec.hours[j] < 16 ? 0 : 1; nn[rr][V2.rec.arms[j]]++; ww[rr][V2.rec.arms[j]] += V2.rec.rews[j]; }
     table(nn.map((n, j) => ({ label: ROWLAB[j], n, w: ww[j] })), { hi: h < 16 ? 0 : 1, y: 310, rh: 44 }); caption(`pull ${i + 1}, at ${hh(h)}`); } },

 { h: 'Each row settles', p: () => { const s = V2.rec.snaps[1000]; return [`After 1000 pulls in this visit, the morning row pulled C ${s.n[0][2]} times out of ${s.n[0].reduce((a, b) => a + b, 0)}, and the evening row pulled D ${s.n[1][3]} times out of ${s.n[1].reduce((a, b) => a + b, 0)}.`, 'Each row found its own winner. Within a row the hours still differ: at 15:00 B is best, but only by one percentage point over C, so lumping 15:00 in with the morning costs very little.']; },
   draw(t) { B.arcade({}); pip(200, t); table(rowsAt(V2.rec.snaps[1000]), { y: 300, rh: 44, mark: true, a: i => clamp(t - .5 - i) }); txt('crumbs out of pulls, per row, after 1000 pulls', 200, 268, { s: 14, c: C.soft, a: clamp(t - .3) }); } },

 { h: 'Keeping score', p: () => [`Regret now compares each pull with the **best machine at that hour**: at 10:00 a pull of D gives up 0.70 minus 0.15, at 21:00 it gives up nothing.`, `Averaged over ${GAMES} visits of 1000 pulls: the clock-blind UCB, in grey, ends at ${r0(AV('blind')[T - 1])}. Two rows of UCB, in gold: ${r0(AV('row2')[T - 1])}. A row player that already knew its best machine would lose under ${Math.ceil(LUMP)} crumb per 1000 pulls, so nearly all of that is the cost of learning.`],
   draw(t) { regChart([{ v: AV('blind'), col: GREY, label: 'clock-blind UCB', n: (t - 1) / 2.5 }, { v: AV('row2'), col: GOLD, label: 'two rows', ly: 12, n: (t - 4) / 2.5 }]);
     card(['regret per pull = best chance at that hour', '                         − chance of the machine pulled'], t - .4, { y: 300, gap: 1 }); } },

 { h: 'Finer rows?', p: () => [`Why not one row per hour? Rows that fine are exactly right about their hour. But learning is the real cost here, and each of twelve rows sees only about one pull in twelve, starting from scratch.`, `With twelve rows, regret after 1000 pulls is ${r0(AV('row12')[T - 1])}: worse than two rows. More rows means more to learn, with less data in each. Real contexts, like a visitor’s age, place and history, would need millions of rows.`],
   draw(t) { regChart([{ v: AV('blind'), col: GREY, label: 'clock-blind', ly: -12 }, { v: AV('row2'), col: GOLD, label: 'two rows', ly: 14 }, { v: AV('row12'), col: TEAL, label: '12 rows', ly: -12, n: (t - 1) / 2.5 }]);
     if (t > 4.5) card(['12 rows: each sees about 1 pull in 12', { s: 'nothing learned at 11:00 helps at 12:00', c: C.roseInk }], t - 4.5, { y: 300, gap: 1 }); } },

 { h: 'Share what you learn: a line', p: () => [`Instead of rows, give each machine a **straight line**: chance = start + slope × lateness, where lateness runs from 0 at 10:00 to 1 at 21:00. Every pull of D, at any hour, helps place D’s line.`, `Here are D’s 45 results from the first 100 pulls of one visit: a dot at the top for a crumb, at the bottom for nothing, placed at its hour. The line is the best fit through them.`],
   draw(t) { const rec = VL.rec, pts = []; for (let i = 0; i < 100; i++) if (rec.arms[i] === 3) pts.push([rec.hours[i], rec.rews[i], i]);
     const { HX, PY } = hourChart({ show: k => k === 3 ? .5 : 0, dash: true, labels: false, lines: t > 4 ? [{ k: 3, m: LIN100[3], band: false, a: clamp(t - 4) }] : null, y0: 250, y1: 90 });
     pts.forEach(([h, r, i], j) => { const a = clamp((t - .6 - j * .12) * 3); dot(HX(h) + ((i * 7) % 11) - 5, PY(r) + (r ? 6 : -6) + ((i * 5) % 7) - 3, 3.2, INK[3], a * .8); });
     if (t > 4) card([`D: ${pts.length} pulls so far`, { s: 'dashed: true chance (hidden)   solid: Pip’s line', c: C.soft }], t - 4, { y: 300, gap: 1 }); } },

 { h: 'LinUCB: a line plus a bonus', p: ['To explore, add a bonus, as in UCB. The shaded band shows how unsure each line is. It depends on how many pulls the machine has had near each hour: narrow where results pile up, wide where they are few, like UCB’s bonus shrinking with n. Score = line + half the band’s width.', 'This is **LinUCB** (Li, Chu, Langford and Schapire, 2010). Lines start flat at 0, so an untried machine gets tried only thanks to its wide band.'],
   draw(t) { const h = 18; hourChart({ true: false, lines: [0, 1, 2, 3].map(k => ({ k, m: LIN100[k], bandA: .13, a: clamp((t - .6 - k * .5) * 2) })), hour: t > 4 ? h : null, y0: 250, y1: 90 });
     if (t > 3) caption(`after 100 pulls: ` + LIN100.map((m, k) => `${NAMES[k]} ${Math.round(m.M[0] - 1)}`).join(', ') + ` pulls; A and B still flat at 0`, clamp(t - 3));
     if (t > 4.5) { const sc = LIN100.map(m => { const p = linPred(m, zOf(h)); return p.pred + p.bon; }), b = sc.indexOf(Math.max(...sc));
       card([`at ${hh(h)}:  ` + sc.map((v, k) => `${NAMES[k]} ${v.toFixed(2)}`).join('   '), { s: `highest score: ${NAMES[b]}  (after 100 pulls)`, c: INK[b] }], t - 4.5, { y: 300, gap: 1 }); } } },

 { h: 'Lines win', p: () => [`LinUCB, in purple, ends at ${r0(AV('lin')[T - 1])} after 1000 pulls, far below two rows (${r0(AV('row2')[T - 1])}) and twelve rows (${r0(AV('row12')[T - 1])}).`, 'Be fair to the rows, though: here the true chances really are straight lines, so the line model fits perfectly. When the real pattern is curved, a line can be wrong for some hours however much data it gets; then richer models, even neural networks, are used.'],
   draw(t) { regChart([{ v: AV('blind'), col: GREY, label: 'clock-blind', ly: -12 }, { v: AV('row2'), col: GOLD, label: 'two rows', ly: 14 }, { v: AV('row12'), col: TEAL, label: '12 rows', ly: -12 }, { v: AV('lin'), col: PLUM, label: 'LinUCB', ly: -12, n: (t - 1) / 2.5 }]);
     if (t > 5) card(['lines share data across hours', { s: 'but only fit if the truth is close to a line', c: C.roseInk }], t - 5, { y: 300, gap: 1 }); } },

 { h: 'Your turn', ctl: { label: 'hour', values: HOURS, def: 8, fmt: hh }, p: () => { const pk = linBest(LIN1000, 15); return [`Drag the **hour**: these are LinUCB’s lines after 1000 pulls of one visit, with its scores and pick at that hour. A and B were pulled only ${Math.round(LIN1000[0].M[0] - 1)} times each and dropped ${LIN1000[0].v[0] + LIN1000[1].v[0] ? 'almost ' : ''}nothing, so their lines sit at 0; only their wide bands keep them in the running.`, `Try 15:00. B is truly best there (${pct(chance(1, 15))}), but LinUCB picks ${NAMES[pk]} (${pct(chance(pk, 15))}): a miss of about ${P100(chance(1, 15) - chance(pk, 15))} percentage points at that hour.`]; },
   shot: [3, 8],
   draw(t, st) { const h = HOURS[st.ci ?? 8]; hourChart({ true: false, lines: [0, 1, 2, 3].map(k => ({ k, m: LIN1000[k] })), hour: h, y0: 250, y1: 90 }); clock(h, 300, 40);
     const sc = LIN1000.map(m => { const p = linPred(m, zOf(h)); return p.pred + p.bon; }), b = sc.indexOf(Math.max(...sc)), tb = bestAt(h);
     card([sc.map((v, k) => `${NAMES[k]} ${v.toFixed(2)}`).join('    '), { s: `LinUCB pulls ${NAMES[b]}     truly best at ${hh(h)}: ${NAMES[tb]}` + (b === tb ? '   (same)' : '   (wrong!)'), c: b === tb ? C.ink : C.roseInk }, { s: `pulls in this visit: ` + LIN1000.map((m, k) => `${NAMES[k]} ${Math.round(m.M[0] - 1)}`).join('  '), c: C.soft }], 3, { y: 296, gap: 0 }); } },

 { h: 'The fine print', p: ['A bandit only sees the result of the arm it pulled. With rows, if Pip never pulled A in the evening, it has no evening data on A at all. A line can guess from morning pulls, but only if a line is the right shape.', 'The same gap bites later: records of old pulls only show what the old strategy chose, so testing a new strategy on them fairly needs special care.'],
   draw(t) { B.arcade({ per: k => ({ a: k === 0 ? .35 : 1 }) }); clock(20); pip(X[3], t); txt('A at 20:00: no data?', X[0] + 10, 264, { s: 15, c: C.roseInk, a: clamp(t - 1.5) });
     if (t > 4) card(['you only see the pulled arm’s result', { s: 'logs can’t answer “what if I had pulled A?”', c: C.soft }], t - 4, { y: 300, gap: 1 }); } },

 { h: 'Where it is used', p: ['In 2010, LinUCB chose which news stories to show on the Yahoo! front page. In the authors’ tests, it earned about 12.5 percent more clicks than a bandit that ignored the context.', 'Contextual bandits now help choose ads, recommendations and notification times, and are studied for personalised treatments in medicine. The context is whatever you know about the user and the moment.'],
   draw(t) { const items = [['news story', 'reader’s interests'], ['ad', 'page and visitor'], ['song', 'time and mood'], ['treatment', 'patient data']];
     txt('arm', 120, 80, { s: 15, c: C.soft, a: clamp(t - .3) }); txt('context', 280, 80, { s: 15, c: C.soft, a: clamp(t - .3) });
     items.forEach(([a, b], i) => { const u = clamp(t - .8 - i * 1.1), y = 116 + i * 40; txt(a, 120, y, { s: 17, a: u }); txt(b, 280, y, { s: 17, a: u, c: C.ink }); if (u > 0) arrow(222, y, 170, y, 1100 + i, { c: C.soft }); });
     if (t > 6) card(['2010  LinUCB on Yahoo! front-page news', { s: 'about 12.5% more clicks than context-free', c: C.soft }], t - 6, { y: 300, gap: 1 }); } },

 { h: 'Recap', p: ['A **contextual bandit** looks at the context before choosing, because the best arm can depend on the situation.', 'Few contexts: keep a row of tallies for each and run a bandit per row. Many contexts: learn a model that shares data, like LinUCB’s line plus bonus. Either way, the pull does not change the next context.'],
   draw(t) { hourChart({ y0: 250, y1: 90, best: 1 }); caption('true chances (hidden from Pip); dots = best each hour');
     card(['see the context, then pick an arm', 'best arm can differ per context', 'few contexts: a row each;  many: a model', { s: 'LinUCB: line + bonus;  pulls don’t change contexts', c: C.soft }], t - .3, { y: 290, gap: 1, lh: 22 }); } },
];
STEPS.forEach(s => { if (typeof s.p === 'function') { const f = s.p; Object.defineProperty(s, 'p', { get: f, enumerable: true }); } });
RLStory.register('contextual', { title: 'Contextual Bandits', track: 'Bandits', onCtx: c => { ctx = c; }, steps: STEPS, get _sim() { return sim(); } });
})();
