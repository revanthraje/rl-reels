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

// Thompson sampling as a story: Pip keeps a belief curve for every cheese machine and acts on one random draw from each.
// World, dice and drawing from the shared bandit kit (copied above). All numbers come from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, B = RLStory.bandit, { W, C, lerp, clamp, ease, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
const { K, P, NAMES, COL, INK, X, Y0, MH } = B;
let ctx;

// ───── belief curves ─────
// After w crumbs and l nothings (starting from "every chance equally plausible"), the plausibility of chance p is
// proportional to p^w (1 − p)^l: the Beta(1 + w, 1 + l) curve. Drawn scaled so the most plausible chance has full height.
const NG = 120;
function curve(w, l) { const ys = new Float64Array(NG + 1); let m = -1e300;
  for (let i = 0; i <= NG; i++) { const p = i / NG, v = (w ? w * Math.log(Math.max(p, 1e-12)) : 0) + (l ? l * Math.log(Math.max(1 - p, 1e-12)) : 0); ys[i] = v; if (v > m) m = v; }
  for (let i = 0; i <= NG; i++) ys[i] = Math.exp(ys[i] - m); return ys; }
// draw a belief curve in the box [x0, x1] × [y - h, y]. o = { col, fill, a, p (0..1 build-in), lw }
function belief(w, l, x0, x1, y, h, o = {}) {
  const ys = curve(w, l), n = Math.round(NG * clamp(o.p ?? 1)), a = o.a ?? 1; if (n < 1) return;
  ctx.save(); ctx.globalAlpha = a; ctx.beginPath(); ctx.moveTo(x0, y);
  for (let i = 0; i <= n; i++) ctx.lineTo(x0 + i / NG * (x1 - x0), y - ys[i] * h + Math.sin(i * 1.7) * .5);
  ctx.lineTo(x0 + n / NG * (x1 - x0), y); ctx.closePath(); ctx.globalAlpha = a * .28; ctx.fillStyle = o.fill || o.col || C.teal; ctx.fill();
  ctx.globalAlpha = a; ctx.beginPath(); for (let i = 0; i <= n; i++) { const xx = x0 + i / NG * (x1 - x0), yy = y - ys[i] * h + Math.sin(i * 1.7) * .5; i ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy); }
  ctx.strokeStyle = o.col || C.ink; ctx.lineWidth = o.lw || 2; ctx.lineJoin = 'round'; ctx.stroke(); ctx.restore();
}
function axis01(x0, x1, y, o = {}) { rline(x0, y, x1, y, 500 + Math.round(x0), { c: C.ink, w: 1.1 }); if (o.ticks !== false) { txt('0', x0, y + 10, { s: 12, c: C.soft }); txt(o.one || '1', x1, y + 10, { s: 12, c: C.soft }); if (o.mid) txt('0.5', (x0 + x1) / 2, y + 10, { s: 12, c: C.soft }); } }
// four small panels, one under each machine
const PW = 40, PY = 388, PH = 66;
function panels(wl, o = {}) {
  for (let k = 0; k < K; k++) { const x0 = X[k] - PW, x1 = X[k] + PW, a = o.a ?? 1;
    alpha(a, () => { axis01(x0, x1, PY, { ticks: o.ticks !== false && k === 0 }); if (o.hi === k) box(x0 - 3, PY - PH - 6, 2 * PW + 6, PH + 6, C.yellow, .45); });
    belief(wl[k][0], wl[k][1], x0, x1, PY, PH, { col: INK[k], fill: COL[k], a });
    if (o.draws && o.draws[k] != null) { const u = clamp(o.du ?? 1), dx = x0 + o.draws[k] * (x1 - x0), dy = lerp(PY - PH - 10, PY - 4, ease(u));
      dot(dx, dy, 4.5, INK[k], clamp(u * 3)); if (u >= 1) txt(o.draws[k].toFixed(2), dx, PY - PH - 14 + (o.dlab ?? 0), { s: 14, c: INK[k], a: clamp((o.du - 1) * 3) }); }
  }
}

// ───── simulations ─────
// Thompson sampling exactly as the kit: every pull, one draw from each machine's Beta(1 + crumbs, 1 + nothings) in order A, B, C, D;
// pull the highest draw. No forced first pulls. We record the draws (identical dice to B.play(B.thompson, g, T)).
function tsPlay(g, T) { const tp = B.tape(g), R = L.rng(7000000 + g), w = [0, 0, 0, 0], n = [0, 0, 0, 0], arms = [], rews = [], draws = [], wl = [];
  for (let t = 0; t < T; t++) { wl.push(w.map((v, k) => [v, n[k] - v])); const d = []; for (let k = 0; k < K; k++) d.push(B.betaS(R, 1 + w[k], 1 + n[k] - w[k]));
    let a = 0; for (let k = 1; k < K; k++) if (d[k] > d[a]) a = k; const r = tp[a](); n[a]++; w[a] += r; arms.push(a); rews.push(r); draws.push(d); }
  wl.push(w.map((v, k) => [v, n[k] - v])); return { arms, rews, draws, wl }; }   // wl[t] = [crumbs, nothings] per machine after t pulls
const VISIT = 24, GV = tsPlay(VISIT, 1000);
const WL = t => GV.wl[t];
// how often each machine would win the draw, from the beliefs after t pulls (4000 extra draws, own dice)
function winShare(t, n = 4000) { const R = L.rng(4242 + t), wl = WL(t), c = [0, 0, 0, 0]; for (let i = 0; i < n; i++) { let a = 0, b = -1; for (let k = 0; k < K; k++) { const x = B.betaS(R, 1 + wl[k][0], 1 + wl[k][1]); if (x > b) { b = x; a = k; } } c[a]++; } return c.map(v => v / n); }
// fast averages over many visits, same dice as the kit
function fastSim(kind, p, games, T) {
  const best = Math.max(...P), out = new Float64Array(T), share = new Float64Array(K), n = new Float64Array(K), q = new Float64Array(K), w = new Float64Array(K), sc = new Float64Array(K);
  for (let g = 0; g < games; g++) {
    const tp = B.tape(g), R = L.rng(7000000 + g); n.fill(0); q.fill(0); w.fill(0); let reg = 0;
    for (let t = 0; t < T; t++) {
      let a = 0;
      if (kind === 'ts') { for (let k = 0; k < K; k++) sc[k] = B.betaS(R, 1 + w[k], 1 + n[k] - w[k]); for (let k = 1; k < K; k++) if (sc[k] > sc[a]) a = k; }
      else if (t < K) a = t;
      else if ((kind === 'eps' && R() < p) || (kind === 'epsd' && R() < Math.min(1, p / t))) a = Math.floor(R() * K);
      else {
        if (kind === 'ucb') { const l2 = 2 * Math.log(t); for (let k = 0; k < K; k++) sc[k] = q[k] + p * Math.sqrt(l2 / n[k]); } else for (let k = 0; k < K; k++) sc[k] = q[k];
        let m = -1e9, nc = 0; for (let k = 0; k < K; k++) { if (sc[k] > m + 1e-12) { m = sc[k]; nc = 1; a = k; } else if (sc[k] > m - 1e-12) nc++; }
        if (nc > 1) { let j = Math.floor(R() * nc); for (let k = 0; k < K; k++) if (sc[k] > m - 1e-12 && !j--) { a = k; break; } }
      }
      const r = tp[a](); n[a]++; w[a] += r; q[a] += (r - q[a]) / n[a]; reg += best - P[a]; out[t] += reg;
    }
    for (let k = 0; k < K; k++) share[k] += n[k];
  }
  for (let t = 0; t < T; t++) out[t] /= games;
  return { reg: out, pulls: Array.from(share, v => v / games), games, T };
}
const memo = {}, M = (key, f) => memo[key] || (memo[key] = f());
const GAMES = 2000;
const EPSD = () => M('epsd', () => fastSim('epsd', 10, GAMES, 1000)), UCB25 = () => M('ucb25', () => fastSim('ucb', .25, GAMES, 1000));
const EPS1 = () => M('eps', () => fastSim('eps', .1, GAMES, 1000)), UCB1 = () => M('ucb', () => fastSim('ucb', 1, GAMES, 1000)), TS = () => M('ts', () => fastSim('ts', 0, GAMES, 1000));
const r0 = v => String(Math.round(v)), pc = v => Math.round(v * 100) + ' percent';
// the belief example: machine B in visit 5 (its first pulls: crumb, nothing, …)
const EXV = 5, EX = (() => { const tp = B.tape(EXV), s = []; for (let i = 0; i < 100; i++) s.push(tp[1]()); const c = n => s.slice(0, n).reduce((a, b) => a + b, 0); return { first: s.slice(0, 2), w10: c(10), w100: c(100) }; })();
const SH40 = M('sh40', () => winShare(40));

// ───── helpers ─────
const pip = (x, t, face = 1) => L.mouse(x, 232, face, .9, t);
function tallies(t, o = {}) { const wl = WL(t); for (let k = 0; k < K; k++) B.tally(k, wl[k][0], wl[k][0] + wl[k][1], o.y ?? 264, { c: INK[k], a: o.a }); }
function pullAnim(i, u) { const a = GV.arms[i], r = GV.rews[i]; B.arcade({ per: k => k === a ? { pull: clamp(u * 3) * (1 - clamp((u - .7) * 3)), crumb: r ? clamp((u - .3) * 1.6) : 0, miss: r ? 0 : clamp((u - .3) * 1.6) } : {} }); }
function regChart(curves, o = {}) {
  const x0 = 58, x1 = 360, y0 = o.y0 ?? 250, y1 = o.y1 ?? 80, T = 1000, max = o.max ?? 80, X_ = i => x0 + (i + 1) / T * (x1 - x0), Y_ = v => y0 - v / max * (y0 - y1);
  rline(x0, y0, x1 + 6, y0, 760, { c: C.ink, w: 1.2 }); rline(x0, y0, x0, y1 - 8, 761, { c: C.ink, w: 1.2 });
  txt('total regret', x0 - 4, y1 - 18, { s: 13, c: C.soft, al: 'left' }); txt('pulls', x1 + 4, y0 + 22, { s: 13, c: C.soft, al: 'right' });
  [0, max / 2, max].forEach(v => txt(String(v), x0 - 6, Y_(v), { s: 12, c: C.soft, al: 'right' })); [0, 500, 1000].forEach(v => txt(v === 1000 ? '1,000' : String(v), x0 + v / T * (x1 - x0), y0 + 10, { s: 12, c: C.soft }));
  curves.forEach(cv => { const n = Math.floor(clamp(cv.n ?? 1) * T); if (n < 2) return; ctx.save(); ctx.strokeStyle = cv.col; ctx.lineWidth = 2.2; ctx.beginPath();
    for (let i = 0; i < n; i += 6) i ? ctx.lineTo(X_(i), Y_(cv.v[i])) : ctx.moveTo(X_(i), Y_(cv.v[i])); ctx.lineTo(X_(n - 1), Y_(cv.v[n - 1])); ctx.stroke(); ctx.restore();
    if (n >= T) { txt(cv.label, X_(T - 1) - 4, Y_(cv.v[T - 1]) + (cv.ly ?? -12), { s: 14, c: cv.col, al: 'right' }); txt(r0(cv.v[T - 1]), X_(T - 1) + 6, Y_(cv.v[T - 1]), { s: 13, c: cv.col, al: 'left' }); } });
}
const TEAL = '#3f8582', GOLD = '#c48a1a', PLUM = '#7a5aa6';
// big single belief panel used in the intro steps
const BX0 = 60, BX1 = 340, BY = 250, BH = 120;
function bigPanel(w, l, t, o = {}) { axis01(BX0, BX1, BY, { mid: true }); txt('possible chance of a crumb', 200, BY + 26, { s: 13, c: C.soft });
  txt('how plausible', BX0 - 8, BY - BH - 14, { s: 13, c: C.soft, al: 'left' }); rline(BX0, BY, BX0, BY - BH - 4, 540, { c: C.line, w: 1 });
  belief(w, l, BX0, BX1, BY, BH, { col: INK[1], fill: COL[1], p: o.p ?? 1, a: o.a }); }

// ───── the steps ─────
const STEPS = [
 { h: 'Pip’s hunch', p: ['Back at the **cheese arcade**: machines A to D, each dropping a crumb or nothing, each with a hidden chance.', 'Pip could keep one estimate per machine. But “1 crumb from 1 pull” and “50 crumbs from 100 pulls” are very different evidence, even though the first estimate looks better. This reel tracks that difference with a **belief**, and lets the belief do the exploring.'],
   shot: [3, 9],
   say: [[["Back at the cheese arcade.", [20, 20, 360, 250]], ["Four machines, each with a hidden chance of a crumb.", [20, 20, 360, 250]]], [["Now compare these two.", [60, 275, 280, 55]], ["One crumb from one pull, or fifty from a hundred.", [60, 275, 280, 55]], ["A lucky start, or solid evidence?", [60, 275, 280, 55]]], "This reel tracks that difference with a belief, and lets the belief do the exploring."],
   draw(t) { B.arcade({}); pip(lerp(-20, 200, ease(t / 2.5)), t);
     if (t > 3) { txt('1 of 1', 130, 290, { s: 20, c: INK[1], a: clamp(t - 3) }); txt('50 of 100', 270, 290, { s: 20, c: INK[1], a: clamp(t - 4) }); }
     if (t > 5.5) { txt('a lucky start?', 130, 318, { s: 15, c: C.soft, a: clamp(t - 5.5) }); txt('solid evidence', 270, 318, { s: 15, c: C.soft, a: clamp(t - 6.3) }); } } },

 { h: 'A curve of hunches', p: ['Here is how Pip pictures one machine, B, before pulling it. Left to right: every chance of a crumb the machine **could** have, from 0 to 1. Height: how **plausible** Pip finds that chance.', 'Pip knows nothing yet, so every chance is equally plausible: a flat line. The tallest point always marks the most plausible chance.'],
   say: [[["Here's how Pip pictures machine B before pulling it.", [40, 60, 320, 230]], ["Left to right, every chance the machine could have.", [40, 110, 320, 170]], ["Height is how plausible Pip finds it.", [40, 110, 320, 170]]], [["Pip knows nothing yet, so the line is flat.", [40, 110, 320, 170]]]],
   draw(t) { B.machine(200, 1, { a: .0 }); bigPanel(0, 0, t, { p: (t - 2) / 1.5 }); txt('machine B, before any pull', 200, 70, { s: 17, c: INK[1] });
     if (t > 5) caption('flat = no idea yet', clamp(t - 5)); } },

 { h: 'One crumb', p: ['B’s first pull drops a crumb. Which chances fit that? Chance 0.9 would drop a crumb 9 times in 10, chance 0.1 only once. So after one crumb, 0.9 is **nine times as plausible** as 0.1, and chance 0 is ruled out.', 'Height at each chance = how likely that chance makes what Pip saw. The flat line tilts into a ramp.'],
   say: [[["B's first pull drops a crumb.", [40, 60, 320, 230]], ["Watch the flat line tilt into a ramp.", [40, 110, 320, 170]]], [["A chance of zero point nine would drop a crumb nine times in ten.", [40, 110, 320, 170]], ["So it's now nine times as plausible as zero point one.", [40, 110, 320, 170]]], [["And chance zero is ruled out.", [30, 300, 340, 64]]]],
   draw(t) { txt('machine B: 1 crumb from 1 pull', 200, 70, { s: 17, c: INK[1] }); const u = ease((t - 1) / 1.5);
     if (u < 1) bigPanel(0, 0, t, { a: 1 - u }); bigPanel(1, 0, t, { a: u });
     if (t > 3.5) { const S = p => BX0 + p * (BX1 - BX0); [[.1, .1], [.9, .9]].forEach(([p, v], i) => { const a = clamp(t - 3.5 - i); dot(S(p), BY - v * BH, 4, C.ink, a); txt(String(p), S(p), BY - v * BH - 14, { s: 15, a }); }); }
     if (t > 6) card(['chance 0.9 makes a crumb 9 times as likely as 0.1', { s: 'chance 0: impossible, it never drops anything', c: C.soft }], t - 6, { y: 300 }); } },

 { h: 'Then nothing', p: ['The second pull drops nothing. Now a chance of 1 is ruled out too: such a machine never fails. Multiply the old heights by how likely each chance makes “nothing”, and the ramp bends into a hill around 0.5.', 'Then rescale, so the tallest point is full height again. This way of updating a belief with evidence is called **Bayes’ rule**.'],
   say: [[["The second pull drops nothing.", [40, 60, 320, 230]], ["Now a chance of one is ruled out too.", [40, 110, 320, 170]], ["Watch the ramp bend into a hill around one half.", [40, 110, 320, 170]]], [["Old height, times how likely each chance makes what happened.", [30, 300, 340, 64]], "That's Bayes' rule."]],
   draw(t) { txt('machine B: 1 crumb from 2 pulls', 200, 70, { s: 17, c: INK[1] }); const u = ease((t - 1) / 1.5);
     if (u < 1) bigPanel(1, 0, t, { a: 1 - u }); bigPanel(1, 1, t, { a: u });
     if (t > 4) card(['new height  =  old height  ×  chance of what happened', { s: 'crumb: × chance      nothing: × (1 − chance)', c: C.soft }], t - 4, { y: 300, gap: 1.2 }); } },

 { h: 'More pulls, narrower hill', p: () => [`Keep pulling B. After 10 pulls with ${EX.w10} crumbs, the hill narrows around 0.5. After 100 with ${EX.w100} crumbs, a thin spike near ${(EX.w100 / 100).toFixed(2)}, near B’s true 0.55.`, 'These curves are the **Beta distribution**. With w crumbs and l nothings, Pip’s belief is Beta(1 + w, 1 + l). Its peak is the estimate, crumbs ÷ pulls; its width, how unsure Pip is.'],
   shot: [4, 10],
   say: [[["Keep pulling B. Watch the hill narrow.", [40, 110, 320, 170]], ["After ten pulls, it's centred near one half.", [40, 110, 320, 170]], ["After a hundred, it's a thin spike, right by B's true chance.", [40, 110, 320, 170]]], [["These curves are Beta distributions.", [30, 300, 340, 64]], ["The peak is the estimate. The width is how unsure Pip is.", [30, 300, 340, 64]]]],
   draw(t) { const st = t < 3.5 ? 0 : t < 7 ? 1 : 2, wl = [[1, 1], [EX.w10, 10 - EX.w10], [EX.w100, 100 - EX.w100]][st], lab = ['1 crumb from 2 pulls', `${EX.w10} crumbs from 10 pulls`, `${EX.w100} crumbs from 100 pulls`][st];
     txt('machine B: ' + lab, 200, 70, { s: 17, c: INK[1] }); bigPanel(wl[0], wl[1], t);
     if (t > 8) { const S = p => BX0 + p * (BX1 - BX0); rline(S(.55), BY + 2, S(.55), BY - BH - 6, 560, { c: C.ink, w: 1, once: true }); txt('true 0.55', S(.55) + 6, BY - BH - 8, { s: 14, al: 'left', a: clamp(t - 8) }); }
     card([`Beta(1 + crumbs, 1 + nothings)`, { s: 'peak = estimate    width = how unsure', c: C.soft }], t - 1, { y: 300 }); } },

 { h: 'The rule: one draw each', p: ['**Thompson sampling** uses the curves like this. Every turn, Pip draws **one random chance** from each machine’s curve: draws land more often where the curve is tall. Then it pulls the machine whose draw came out highest, and updates that one curve.', 'At the start all four curves are flat, so the four draws are just random numbers.'],
   say: [[["Here's Thompson sampling.", [20, 20, 360, 385]], ["Every turn, Pip draws one random chance from each machine's curve.", [18, 300, 365, 105]]], [["Then it pulls whichever draw came out highest, and updates that one curve.", [18, 300, 365, 105]]], "At the start the curves are all flat, so the draws are just random."],
   draw(t) { B.arcade({ per: k => ({ glow: k === GV.arms[0] && t > 6 ? .7 : 0 }) }); pip(200, t); tallies(0);
     panels(WL(0), { draws: t > 2 ? GV.draws[0] : null, du: (t - 2) / .8, hi: t > 6 ? GV.arms[0] : -1 });
     txt('each machine’s belief: chance 0 to 1', 200, 286, { s: 14, c: C.soft, a: clamp(t - .5) });
     if (t > 6) caption(`highest draw: ${NAMES[GV.arms[0]]} (${GV.draws[0][GV.arms[0]].toFixed(2)})`, clamp(t - 6)); } },

 { h: 'The first pulls', p: () => [`We follow one visit. Pull 1: A’s draw, ${GV.draws[0][0].toFixed(2)}, is highest. A drops nothing, so its curve tilts toward low chances. Pull 2: A’s draw is just ${GV.draws[1][0].toFixed(2)}, and C wins with ${GV.draws[1][2].toFixed(2)}. Nothing again.`, 'Each pull shows the draws, one dot per machine; the highest decides. B’s draws happen not to win at first, so it stays untried for a while.'],
   shot: [5, 13],
   say: [[["Let's follow one visit. Watch the dots on each curve.", [18, 300, 365, 105]], ["Pull one, A's draw is highest. A gives nothing.", [20, 20, 360, 260]], ["Its curve tilts low.", [18, 300, 365, 105]]], [["Pull two, C wins the draw. Nothing again.", [20, 20, 360, 260]]], "The highest draw decides, every time."],
   draw(t) { const dur = 2.6, i = clamp(Math.floor(t / dur), 0, 5), u = (t - i * dur) / dur;
     if (u > .45) pullAnim(i, (u - .45) / .55); else B.arcade({}); pip(X[GV.arms[i]], t); tallies(i + (u > .8 ? 1 : 0));
     panels(WL(i), { draws: GV.draws[i], du: u / .25, hi: u > .35 ? GV.arms[i] : -1 }); caption(`pull ${i + 1}`); } },

 { h: 'B gets lucky', p: () => [`B drops crumbs on 8 of its first 9 pulls. Its hill moves high, so its draws usually win, and Pip pulls B again and again.`, `C, meanwhile, has 0 crumbs from 2 pulls. Its curve leans low but is still **wide**: by its curve, the chance that C is better than 0.5 is still 1 in ${Math.round(1 / Math.pow(.5, 3))}. That width keeps C in the game.`],
   shot: [4, 13],
   say: [[["B gets lucky. Crumb after crumb.", [20, 20, 360, 260]], ["Its hill moves high, so its draws usually win.", [18, 300, 365, 105]]], [["Now look at C. No crumbs from two pulls.", [18, 300, 365, 105]], ["Its curve leans low, but it's still wide.", [18, 300, 365, 105]], "That width keeps C in the game."]],
   draw(t) { const dur = .55, i = clamp(6 + Math.floor(t / dur), 6, 20), u = t / dur - Math.floor(t / dur), done = 6 + t / dur >= 21;
     if (!done) { pullAnim(i, u); pip(X[GV.arms[i]], t); } else { B.arcade({}); pip(X[1], t); }
     const sh = done ? 21 : i; tallies(sh + (!done && u > .6 ? 1 : 0)); panels(WL(sh), { draws: done ? null : GV.draws[i], du: u / .3 + 1, hi: done ? -1 : GV.arms[i] });
     caption(done ? 'after 21 pulls' : `pull ${i + 1}`); } },

 { h: 'C’s comeback', p: () => [`On pull 22, C’s wide curve throws a lucky draw, ${GV.draws[21][2].toFixed(2)}, just above B’s ${GV.draws[21][1].toFixed(2)}. C drops a crumb. More chances follow: by pull 60, C has ${WL(60)[2][0]} crumbs from ${WL(60)[2][0] + WL(60)[2][1]} pulls, and its hill has caught up with B’s.`, 'Nobody told Pip to explore. The width of C’s curve did it.'],
   shot: [4, 12],
   say: [[["Pull twenty two. C's wide curve throws a lucky high draw, just above B's.", [18, 300, 365, 105]], ["C drops a crumb!", [20, 20, 360, 260]]], [["Watch more chances follow. C's hill catches up with B's.", [18, 300, 365, 105]]], "Nobody told Pip to explore. The width of C's curve did it."],
   draw(t) { if (t < 5) { const u = t / 5; if (u > .5) pullAnim(21, (u - .5) * 2); else B.arcade({}); pip(X[2], t); tallies(21 + (u > .8 ? 1 : 0)); panels(WL(21), { draws: GV.draws[21], du: t / .8, hi: t > 1.5 ? 2 : -1 }); caption('pull 22'); }
     else { const tt = clamp(22 + Math.floor((t - 5) / .17), 22, 60); B.arcade({}); pip(X[2], t); tallies(tt); panels(WL(tt)); caption(tt >= 60 ? 'after 60 pulls' : `pull ${tt}`); } } },

 { h: 'Picked as often as it might be best', p: () => [`At pull 40, how often does each machine win? Over 4000 draws, B wins ${pc(SH40[1])}, C ${pc(SH40[2])}, D ${pc(SH40[3])}, A ${pc(SH40[0])}.`, `The secret: Thompson sampling pulls each machine **as often as Pip believes it could be the best**. A machine’s share shrinks only as Pip grows sure it is worse; D, 1 crumb in 3, is still uncertain: ${pc(SH40[3])}.`],
   say: [[["Back at pull forty. How often does each machine win the draw?", [18, 280, 365, 125]], ["B wins most of the time. C and D, about eight percent each.", [18, 280, 365, 125]]], ["That's the secret. Each machine gets pulled about as often as it might be the best.", "A share only shrinks as Pip grows sure that machine is worse."]],
   draw(t) { B.arcade({}); pip(X[1], t); tallies(40); panels(WL(40), { a: 1 }); caption('back at pull 40');
     const by = 304; txt('share of 4000 draws won', 200, by - 16, { s: 14, c: C.soft, a: clamp(t - 1) });
     for (let k = 0; k < K; k++) { const u = clamp((t - 1.5 - k * .4) * 2); txt(pc(SH40[k]).replace(' percent', '%'), X[k], by + 6, { s: 17, c: INK[k], a: u }); } } },

 { h: 'Narrowing down', p: () => [`Fast-forward to pull 1000 in this visit. C’s curve has become a spike near ${(WL(1000)[2][0] / (WL(1000)[2][0] + WL(1000)[2][1])).toFixed(2)}, from ${WL(1000)[2][0] + WL(1000)[2][1]} pulls. A and D were pulled only ${WL(1000)[0][0] + WL(1000)[0][1]} and ${WL(1000)[3][0] + WL(1000)[3][1]} times, so their curves stay wide.`, 'That is fine: Pip never needs to know exactly how bad D is, only that it is very unlikely to beat C.'],
   shot: [3, 12],
   say: [[["Fast forward to pull one thousand.", [20, 20, 360, 260]], ["C's curve has become a spike, near zero point seven.", [18, 300, 365, 105]]], [["A and D were barely pulled, so their curves stay wide.", [18, 300, 365, 105]], "That's fine. Pip only needs to know they're unlikely to beat C."]],
   draw(t) { const tt = Math.min(1000, Math.round(60 * Math.pow(1000 / 60, clamp(t / 8)))); B.arcade({}); pip(X[2], t); tallies(tt); panels(WL(tt)); caption(tt >= 1000 ? 'after 1000 pulls' : `pull ${tt}`); } },

 { h: 'Keeping score', p: () => [`Compare by **regret**: each pull gives up 0.70 minus the chance of the machine pulled. Chart: total regret over 1000 pulls, averaged over ${GAMES} visits, lower better.`, `ε-greedy with ε = 0.1 ends at ${r0(EPS1().reg[999])}, UCB1 at ${r0(UCB1().reg[999])}, Thompson sampling, purple, at ${r0(TS().reg[999])}. It also beats shrinking ε, ${r0(EPSD().reg[999])}. Only UCB with a bonus tuned to this arcade, c = 0.25, edges it: ${r0(UCB25().reg[999])}.`],
   say: [[["Now regret, averaged over two thousand visits.", [30, 60, 350, 205]], ["Teal, epsilon greedy. Gold, UCB one.", [30, 60, 350, 205]], ["And purple, Thompson sampling, the lowest. Eighteen.", [30, 60, 350, 205]]], [["It wastes far fewer pulls on D.", [30, 300, 340, 64]], "Only a specially tuned UCB edges it here."]],
   draw(t) { regChart([{ v: EPS1().reg, col: TEAL, label: 'ε-greedy', ly: 12, n: (t - 1) / 2 }, { v: UCB1().reg, col: GOLD, label: 'UCB1', n: (t - 3.5) / 2 }, { v: TS().reg, col: PLUM, label: 'Thompson', ly: -12, n: (t - 6) / 2 }]);
     if (t > 8.5) card([`average pulls of D:  Thompson ${r0(TS().pulls[3])}, ε-greedy ${r0(EPS1().pulls[3])}`, `average pulls of C:  Thompson ${r0(TS().pulls[2])} of 1000`], t - 8.5, { y: 300, gap: 1 }); } },

 { h: 'Your turn', tap: yourTap, p: ['Pull the machines yourself: tap a machine to pull it once, and watch its curve change. Or tap the button to let Thompson sampling choose: you will see its four draws first.', 'Try pulling one machine many times, and see how thin its curve gets.'],
   shot: [3],
   say: [[["Your turn. Tap a machine to pull it, and watch its curve change.", [20, 20, 360, 385]]], [["Or tap the button, and Thompson sampling chooses. You'll see its four draws first.", [110, 222, 180, 44]]], "Try pulling one machine many times, and see how thin its curve gets."],
   draw(t, st) { const s = yourState(st); if (s.pending && t > s.dt + 1.6) { s.pending = false; pullYour(s, s.pick, t); }
     B.arcade({ per: k => ({ glow: s.last === k ? .6 * (1 - clamp(t - s.lt - 1)) : 0, crumb: s.last === k && s.lr ? clamp((t - s.lt) * 1.5) : 0, miss: s.last === k && !s.lr ? clamp((t - s.lt) * 1.2) : 0 }) });
     for (let k = 0; k < K; k++) B.tally(k, s.wl[k][0], s.wl[k][0] + s.wl[k][1], 200, { c: INK[k] });
     rrect(120, 228, 160, 32, 880, { fill: C.card, c: C.ink, w: 1.4 }); txt('let Thompson choose', 200, 244, { s: 16 });
     panels(s.wl, { draws: s.draws, du: s.draws ? (t - s.dt) / .6 : 0, hi: s.draws ? s.pick : -1 });
     txt(s.n ? `your pulls: ${s.n}   crumbs: ${s.c}` : 'tap a machine, or the button', 200, 290, { s: 15, c: C.soft }); } },

 { h: 'The fine print', p: ['It needs a starting belief; here it was flat, every chance equally plausible. A strong wrong start can slow it.', 'Beta curves fit crumb-or-nothing rewards; other rewards need other belief shapes, often only approximate. Its randomness helps when many choices come before results return: a diceless rule like UCB would send every waiting visitor to the same machine; Thompson’s draws spread them out.'],
   say: [[["The fine print. It needs a starting belief.", [30, 90, 340, 150]], ["Flat is fine. But a strong wrong start, like this one on the right, can slow it down.", [210, 90, 160, 150]]], [["And its randomness is handy. When results come back slowly, its draws spread the waiting visitors out.", [30, 290, 340, 87]]]],
   draw(t) { const S = (p, x0, x1) => x0 + p * (x1 - x0);
     [['flat start', 0, 0, 1], ['strong wrong start', 2, 18, 2.5]].forEach(([lab, w, l, a0], j) => { const a = clamp(t - a0), x0 = 40 + j * 180, x1 = x0 + 140; if (a <= 0) return;
       alpha(a, () => { axis01(x0, x1, 200); txt(lab, (x0 + x1) / 2, 100, { s: 16, c: C.ink }); }); belief(w, l, x0, x1, 200, 80, { col: INK[2], fill: COL[2], a });
       if (j) alpha(a, () => { rline(S(.7, x0, x1), 202, S(.7, x0, x1), 116, 590, { c: C.ink, w: 1, once: true }); txt('C’s true 0.7', S(.7, x0, x1), 228, { s: 13, c: C.soft }); }); });
     if (t > 5) card(['needs: a starting belief', 'easiest for crumb-or-nothing rewards', { s: 'random: same history, different pulls', c: C.soft }], t - 5, { y: 290, gap: 1 }); } },

 { h: 'Where it is used', p: ['William Thompson proposed it in **1933**, to choose between two medical treatments, then it was largely forgotten.', 'Around 2010 it returned: Microsoft engineers used a version for Bing ad placement, and in 2011 Chapelle and Li at Yahoo found it excellent for ads and news recommendations. Proofs in 2012 showed its regret, like UCB1’s, grows only logarithmically. Today it is common in website tests.'],
   say: [[["William Thompson proposed it in nineteen thirty three, for medical treatments.", [40, 60, 340, 60]], "Then it was mostly forgotten."], [["Around twenty ten it came back, for Bing ads, then Yahoo's ads and news.", [40, 110, 340, 110]], ["Proofs soon showed its regret grows only slowly.", [40, 180, 340, 60]]], "Today it's common in website tests."],
   draw(t) { const items = [[1933, 'Thompson: which treatment?'], [2010, 'Bing ads (Graepel et al.)'], [2011, 'Chapelle & Li: ads, news'], [2012, 'regret proofs (Agrawal & Goyal)']];
     rline(60, 70, 60, 250, 900, { c: C.line, w: 1.5 }); items.forEach(([y, s], i) => { const a = clamp(t - .8 - i * 1.4), yy = 85 + i * 52; dot(60, yy, 5, i ? C.teal : C.gold, a); txt(String(y), 74, yy, { s: 17, al: 'left', a, c: C.ink }); txt(s, 124, yy, { s: 16, al: 'left', a, c: C.soft }); }); } },

 { h: 'Recap', p: ['Keep a **belief curve** for each machine: how plausible each chance is, given the crumbs so far.', 'Every turn, **draw one chance from each curve and pull the highest**. Wide curves sometimes draw high, so unsure machines get explored; narrow low curves almost never win, so bad machines get dropped. No ε schedule or bonus to tune, only a starting belief.'],
   say: [[["So, Pip keeps a belief curve for each machine.", [30, 290, 340, 87]], ["Each turn, draw one chance from every curve, and pull the highest.", [30, 290, 340, 87]]], ["Wide curves get explored. Narrow low ones get dropped. No schedule, no bonus to tune."]],
   draw(t) { B.arcade({}); pip(X[2], t);
     card(['belief: Beta(1 + crumbs, 1 + nothings)', 'draw one chance from each, pull the highest', { s: 'picked as often as it might be best', c: C.soft }], t - .3, { y: 290, gap: 1 }); } },
];

// ───── your turn: tap a machine or the button ─────
const YOUR_V = 77;
function yourState(st) {
  if (!st.ys) { st.ys = { pulls: [], wl: [[0, 0], [0, 0], [0, 0], [0, 0]], tapes: B.tape(YOUR_V), n: 0, c: 0, last: -1, lt: -9, lr: 0, draws: null, dt: 0, pick: -1 }; }
  return st.ys;
}
function pullYour(s, k, t) { const r = s.tapes[k](); s.wl[k][r ? 0 : 1]++; s.n++; s.c += r; s.last = k; s.lt = t; s.lr = r; }
function yourTap(x, y, st, t) {
  const s = yourState(st);
  if (s.pending) return;
  if (y > 222 && y < 266 && x > 110 && x < 290) { const R = L.rng(9100 + s.n); s.draws = s.wl.map(([w, l]) => B.betaS(R, 1 + w, 1 + l)); s.pick = s.draws.indexOf(Math.max(...s.draws)); s.dt = t; s.pending = true; return; }
  if (y < 190) { let k = 0; for (let j = 1; j < K; j++) if (Math.abs(x - X[j]) < Math.abs(x - X[k])) k = j; s.draws = null; pullYour(s, k, t); }
}

STEPS.forEach(s => { if (typeof s.p === 'function') { const f = s.p; Object.defineProperty(s, 'p', { get: f, enumerable: true }); } });
const sim = () => ({ firstDraws: GV.draws.slice(0, 2), arms: GV.arms.slice(0, 60).map(a => NAMES[a]).join(''), rews: GV.rews.slice(0, 60).join(''), wl21: WL(21), wl40: WL(40), wl60: WL(60), wl1000: WL(1000), d22: GV.draws[21], sh40: SH40,
  epsd: EPSD().reg[999], ucb25: UCB25().reg[999], ex: EX, eps: EPS1().reg[999], ucb: UCB1().reg[999], ts: TS().reg[999], tsPulls: TS().pulls, epsPulls: EPS1().pulls, match: (() => { const b = B.play(B.thompson, VISIT, 300); return b.arms.every((a, i) => a === GV.arms[i]); })() });
RLStory.register('thompson', { title: 'Thompson Sampling', track: 'Bandits', onCtx: c => { ctx = c; }, steps: STEPS, get _sim() { return sim(); } });
})();
