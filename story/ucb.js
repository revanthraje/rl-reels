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

// Upper Confidence Bound as a story: Pip at the cheese arcade gives every machine the benefit of the doubt.
// World, dice and drawing from the shared bandit kit (copied above). All numbers come from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, B = RLStory.bandit, { W, C, lerp, clamp, ease, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha, hachure } = L;
const { K, P, NAMES, COL, INK, X, Y0, MH } = B;
let ctx;

// ───── simulations ─────
// UCB1 as used in this reel: first pull A, B, C, D once each; then pull the highest  estimate + c·√(2 ln t / n),
// t = pulls so far, n = pulls of that machine. Ties broken at random (kit argmaxR).
const ucb2 = c => (st, R) => (st.t < K ? st.t : B.argmaxR(st.q.map((v, k) => v + c * Math.sqrt(2 * Math.log(st.t) / st.n[k])), R));
const bonus = (c, t, n) => c * Math.sqrt(2 * Math.log(t) / n);
const VISIT = 24;                                          // the visit we follow: C's first pull drops nothing
const GV = B.play(ucb2(1), VISIT, 1000), GG = B.play(B.greedy, VISIT, 1000);
const nAt = (G, t) => t ? G.n[t - 1] : [0, 0, 0, 0];      // counts after t pulls
const qAt = (G, t) => t ? G.est[t - 1] : [0, 0, 0, 0];
const wAt = (G, t) => nAt(G, t).map((n, k) => Math.round(qAt(G, t)[k] * n));
const scoresAt = (G, t, c = 1) => nAt(G, t).map((n, k) => qAt(G, t)[k] + bonus(c, t, n));   // scores used for pull t + 1
// fast averages over many visits, same dice as the kit (tape per machine, strategy dice rng(7000000 + g))
function fastSim(kind, p, games, T, marks = []) {
  let stuck = 0; const best = Math.max(...P), out = new Float64Array(T), share = new Float64Array(K), at = marks.map(() => new Float64Array(K)), n = new Float64Array(K), q = new Float64Array(K), sc = new Float64Array(K);
  for (let g = 0; g < games; g++) {
    const tp = B.tape(g), R = L.rng(7000000 + g); n.fill(0); q.fill(0); let reg = 0;
    for (let t = 0; t < T; t++) {
      let a;
      if (kind === 'try' && t < K * p) a = t % K;
      else if (t < K) a = t;
      else if ((kind === 'eps' && R() < p) || (kind === 'epsd' && R() < Math.min(1, p / t))) a = Math.floor(R() * K);
      else {
        if (kind === 'ucb') { const l2 = 2 * Math.log(t); for (let k = 0; k < K; k++) sc[k] = q[k] + p * Math.sqrt(l2 / n[k]); } else for (let k = 0; k < K; k++) sc[k] = q[k];
        let m = -1e9, nc = 0; for (let k = 0; k < K; k++) { if (sc[k] > m + 1e-12) { m = sc[k]; nc = 1; a = k; } else if (sc[k] > m - 1e-12) nc++; }
        if (nc > 1) { let j = Math.floor(R() * nc); for (let k = 0; k < K; k++) if (sc[k] > m - 1e-12 && !j--) { a = k; break; } }
      }
      const r = tp[a](); n[a]++; q[a] += (r - q[a]) / n[a]; reg += best - P[a]; out[t] += reg;
      const mi = marks.indexOf(t + 1); if (mi >= 0) for (let k = 0; k < K; k++) at[mi][k] += n[k];
    }
    for (let k = 0; k < K; k++) share[k] += n[k]; if (n[2] < T / 2) stuck++;
  }
  for (let t = 0; t < T; t++) out[t] /= games;
  return { reg: out, pulls: Array.from(share, v => v / games), at: at.map(a => Array.from(a, v => v / games)), stuck, games, T };
}
const memo = {}, M = (key, f) => memo[key] || (memo[key] = f());
const GAMES = 2000, LONG_GAMES = 500, LONG_T = 10000;
const EPS1 = () => M('eps', () => fastSim('eps', .1, GAMES, 1000));
const UCB = c => M('ucb' + c, () => fastSim('ucb', c, GAMES, 1000));
const LONG = kind => M('long' + kind, () => fastSim(kind, kind === 'eps' ? .1 : 1, LONG_GAMES, LONG_T, [1000]));
const DBL = () => { const u = LONG('ucb').reg; return [u[4999] - u[2499], u[LONG_T - 1] - u[4999]]; };
const CS = [0, .1, .25, .5, 1, 2];
// fair comparison with the earlier reels' strong rules: shrinking ε = min(1, 10 / t) (epsgreedy reel) and try each machine 10 times first (bandits reel)
const CMP = () => M('cmp', () => [['ε = 0.1', 'eps', .1], ['shrinking ε, 10 ÷ t', 'epsd', 10], ['try each 10 times first', 'try', 10], ['UCB1 (c = 1)', 'ucb', 1], ['UCB, c = 0.25', 'ucb', .25]]
  .map(([lab, kind, p]) => ({ lab, kind, short: fastSim(kind, p, GAMES, 1000).reg[999], long: fastSim(kind, p, LONG_GAMES, LONG_T).reg[LONG_T - 1] })));
const r0 = v => String(Math.round(v));
const fmtK = v => v >= 1000 ? String(v).replace(/(\d)(\d{3})$/, '$1,$2') : String(v);
const CROSS = () => M('cross', () => { const le = LONG('eps'), lu = LONG('ucb'); for (let t = 1000; t < LONG_T; t++) if (lu.reg[t] < le.reg[t]) return t + 1; return -1; });
// wobble: average of B's first 3 (and first 30) pulls in visits 0..29 (same crumb tapes)
const WOB = [3, 30].map(m => { const out = []; for (let g = 0; g < 30; g++) { const tp = B.tape(g); let s = 0; for (let i = 0; i < m; i++) s += tp[1](); out.push(s / m); } return out; });

// ───── helpers ─────
const pip = (x, t, face = 1) => L.mouse(x, 232, face, .9, t);
// pull animation of pull index i (0-based) of visit G, u = 0..1 progress within the pull
function pullScene(G, i, u, o = {}) {
  const a = G.arms[i], r = G.rews[i];
  B.arcade({ sign: o.sign, per: k => k === a ? { pull: clamp(u * 3) * (1 - clamp((u - .7) * 3)), crumb: r ? clamp((u - .3) * 1.6) : 0, miss: r ? 0 : clamp((u - .3) * 1.6), glow: o.glow ? .6 : 0 } : {} });
}
function tallies(G, t, o = {}) { const n = nAt(G, t), w = wAt(G, t); for (let k = 0; k < K; k++) B.tally(k, w[k], n[k], o.y ?? 264, { c: INK[k], a: o.a }); }
// score chart: bar = estimate, gold cap = bonus, label = score. o = { max, y, h, chosen, a, capA, ticks }
function scoreChart(q, bo, o = {}) {
  const y = o.y ?? 392, h = o.h ?? 84, max = o.max ?? 3, u = h / max, bw = 30, a = o.a ?? 1, capA = o.capA ?? 1;
  alpha(a, () => {
    rline(26, y, 378, y, 701, { c: C.line, w: 1 }); rline(26, y, 26, y - h - 4, 702, { c: C.line, w: 1 });
    (o.ticks || [0, 1, 2, 3]).forEach(v => { txt(String(v), 18, y - v * u, { s: 12, c: C.soft, al: 'right' }); if (v) rline(26, y - v * u, 378, y - v * u, 710 + v * 10, { c: '#ece2d4', w: 1, once: true }); });
    for (let k = 0; k < K; k++) {
      const eh = Math.max(0, q[k]) * u, bh = bo[k] * u * capA, x = X[k] - bw / 2;
      if (o.chosen === k) box(x - 5, y - eh - bh - 22, bw + 10, eh + bh + 22, C.yellow, .4);
      if (eh > 0) box(x, y - eh, bw, eh, COL[k], .9);
      if (bh > 0) { box(x, y - eh - bh, bw, bh, '#fbe7b5', .9); hachure(x, y - eh - bh, bw, bh, C.gold, 6, 720 + k, .7); rrect(x, y - eh - bh, bw, bh, 730 + k, { c: '#c99a3a', w: 1 }); }
      if (o.labels !== false) txt((+(q[k] + bo[k] * capA).toFixed(2)).toString(), X[k], y - eh - bh - 10, { s: 14, c: INK[k] });
      txt(NAMES[k], X[k] + bw / 2 + 9, y - 7, { s: 12, c: C.soft });
    }
  });
}
// regret chart. curves = [{ v: Float64Array, col, label, ly (label offset), n: fraction drawn }]
function regChart(curves, o = {}) {
  const x0 = 58, x1 = 360, y0 = o.y0 ?? 388, y1 = o.y1 ?? 300, T = o.T ?? 1000, max = o.max ?? 80;
  const X_ = i => x0 + (i + 1) / T * (x1 - x0), Y_ = v => y0 - v / max * (y0 - y1);
  rline(x0, y0, x1 + 6, y0, 760, { c: C.ink, w: 1.2 }); rline(x0, y0, x0, y1 - 8, 761, { c: C.ink, w: 1.2 });
  txt(o.ylab || 'total regret', x0 - 4, y1 - 18, { s: 13, c: C.soft, al: 'left' }); txt('pulls', x1 + 4, y0 + 22, { s: 13, c: C.soft, al: 'right' });
  (o.yt || [0, max / 2, max]).forEach(v => txt(String(v), x0 - 6, Y_(v), { s: 12, c: C.soft, al: 'right' }));
  (o.xt || [0, T / 2, T]).forEach(v => txt(fmtK(v), x0 + v / T * (x1 - x0), y0 + 10, { s: 12, c: C.soft }));
  curves.forEach((cv, j) => {
    const n = Math.floor(clamp(cv.n ?? 1) * T); if (n < 2) return; ctx.save(); ctx.strokeStyle = cv.col; ctx.lineWidth = 2.2; ctx.beginPath();
    const stp = Math.max(1, Math.floor(T / 150)); for (let i = 0; i < n; i += stp) { i ? ctx.lineTo(X_(i), Y_(cv.v[i])) : ctx.moveTo(X_(i), Y_(cv.v[i])); } ctx.lineTo(X_(n - 1), Y_(cv.v[n - 1])); ctx.stroke(); ctx.restore();
    if (cv.label && n >= T) { txt(cv.label, X_(T - 1) - 4, Y_(cv.v[T - 1]) + (cv.ly ?? -12), { s: 14, c: cv.lc || cv.col, al: 'right' }); txt(r0(cv.v[T - 1]), X_(T - 1) + 6, Y_(cv.v[T - 1]), { s: 13, c: cv.lc || cv.col, al: 'left' }); }
  });
}
const TEAL = '#3f8582', GOLD = '#c48a1a';

// ───── numbers quoted in the text ─────
const S5 = scoresAt(GV, 4), S9 = scoresAt(GV, 8), S20 = scoresAt(GV, 19);
const C40 = [wAt(GV, 40)[2], nAt(GV, 40)[2]];
const N1000 = nAt(GV, 1000), Q1000 = qAt(GV, 1000), B1000 = N1000.map(n => bonus(1, 1000, n));
const sim = () => { const e = EPS1(), u = UCB(1), le = LONG('eps'), lu = LONG('ucb');
  const cross = CROSS();
  return { S5, S9, S20, C40, N1000, Q1000, B1000, greedyN: nAt(GG, 1000), eps: e.reg[999], epsPulls: e.pulls, ucb: u.reg[999], ucbPulls: u.pulls,
    longEps: le.reg[LONG_T - 1], longUcb: lu.reg[LONG_T - 1], longEpsD: [le.at[0][3], le.pulls[3]], longUcbD: [lu.at[0][3], lu.pulls[3]], cross,
    cmp: CMP().map(r => [r.lab, r.short, r.long]), dbl: DBL(), sweep: CS.map(c => [c, UCB(c).reg[999], UCB(c).pulls, UCB(c).stuck]), wob: WOB.map(a => [Math.min(...a), Math.max(...a)]) }; };

// ───── the steps ─────
const STEPS = [
 { h: 'Back at the arcade', p: ['Pip is back at the **cheese arcade**: four machines, A to D. Each pull drops one crumb or nothing, and each machine has its own hidden chance of a crumb.', 'Pip keeps an **estimate** for each machine: crumbs divided by pulls. Greedy can get stuck. ε-greedy unsticks it with random pulls. This reel shows a smarter way to explore.'],
   shot: [3, 9],
   draw(t) { B.arcade({ per: k => ({ secret: clamp((t - 4 - k * .5) * 2) }) }); pip(lerp(-20, 200, ease(t / 2.5)), t);
     if (t > 4) txt('hidden chances: only we can see them', 200, 300, { s: 15, c: C.soft, a: clamp(t - 6) }); caption('C is the best machine; Pip doesn’t know that', clamp(t - 7.5)); } },

 { h: 'Random pulls are blind', p: () => [`With ε = 0.1, every tenth pull goes to a random machine, even one Pip knows is bad. Over 1000 pulls, averaged over 2000 visits, D gets ${r0(EPS1().pulls[3])} pulls, almost all dice rolls.`, 'The dice never stop: Pip pays for them on pull 10 and pull 10,000 alike. Yet in the first 1000 pulls, UCB checks D just as often; the difference comes after.'],
   draw(t) { const e = EPS1(); B.arcade({ per: k => ({ glow: k === 3 ? clamp(t - 4) * .8 : 0 }) }); pip(X[3], t);
     for (let k = 0; k < K; k++) txt(r0(e.pulls[k]), X[k], 264, { s: 18, c: INK[k], a: clamp(t - .8 - k * .6) });
     txt('average pulls of each machine, ε-greedy, 1000 pulls', 200, 292, { s: 14, c: C.soft, a: clamp(t - .5) });
     if (t > 4.5) card(['random pulls per machine:  about 25 each', { s: 'D is the worst machine, yet it gets its share', c: C.roseInk }], t - 4.5, { y: 312, gap: 1.2 }); } },

 { h: 'Two kinds of “looks bad”', p: ['Picture two machines. One gave nothing in its only pull. The other gave 6 crumbs in 40 pulls. Both look poor.', 'But one pull says almost nothing: that machine might still be great. Forty pulls say a lot. A smart explorer should spend its tries where it is **unsure**, not spread them evenly.'],
   draw(t) { const xs = [120, 280];
     [[0, 1, 0], [6, 40, 3]].forEach(([w, n, k], j) => { const x = xs[j], a = clamp(t - .5 - j * 1.2); alpha(a, () => { B.machine(x, k, { label: j ? 'Y' : 'X' }); txt(`${w} of ${n}`, x, 200, { s: 18, c: C.ink }); txt(`estimate ${+(w / n).toFixed(2)}`, x, 222, { s: 15, c: C.soft }); }); });
     if (t > 3.5) { const a = clamp(t - 3.5); alpha(a, () => { const y = 340; rline(40, y, 360, y, 600, { c: C.line });
       txt('0', 40, y + 14, { s: 12, c: C.soft }); txt('1', 360, y + 14, { s: 12, c: C.soft }); txt('where the true chance could be', 200, y + 14, { s: 13, c: C.soft });
       box(40, y - 26, 320 * .98, 8, COL[0], .5); txt('X: anywhere', 44, y - 36, { s: 14, c: INK[0], al: 'left' });
       box(40 + 320 * .06, y - 12, 320 * .2, 8, COL[3], .7); txt('Y: about here', 40 + 320 * .3, y - 8, { s: 14, c: INK[3], al: 'left' }); }); }
     caption('X and Y are made up for this picture', clamp(t - 6)); } },

 { h: 'How much does an average wobble?', p: () => [`Take machine B, true chance 55 percent, over 30 visits. Each dot is the average of B’s first 3 pulls in one visit. With 3 pulls only four averages are possible, and the dots use all of them, ${+Math.min(...WOB[0]).toFixed(2)} to ${+Math.max(...WOB[0]).toFixed(2)}.`, `Averages of the first 30 pulls: ${+Math.min(...WOB[1]).toFixed(2)} to ${+Math.max(...WOB[1]).toFixed(2)}. More pulls, less wobble; roughly, 4 times the pulls halves it.`],
   shot: [5, 12],
   draw(t) { B.machine(200, 1, {}); txt('true chance 55%', 200, 198, { s: 15, c: INK[1] });
     const row = (vals, y, lab, a0) => { const x0 = 50, x1 = 350, a = clamp(t - a0); if (a <= 0) return;
       alpha(a, () => { rline(x0, y, x1, y, 610 + y, { c: C.ink, w: 1.2 }); [0, .5, 1].forEach(v => txt(String(v), x0 + v * (x1 - x0), y + 14, { s: 12, c: C.soft }));
         rline(x0 + .55 * (x1 - x0), y - 44, x0 + .55 * (x1 - x0), y + 4, 620 + y, { c: INK[1], w: 1, once: true }); txt(lab, 200, y - (y > 350 ? 46 : 56), { s: 14, c: C.soft }); });
       const cnt = {}; vals.forEach((v, i) => { const d = clamp((t - a0 - .5 - i * .08) * 3); if (d <= 0) return; const key = v.toFixed(3), s = cnt[key] = (cnt[key] || 0) + 1; dot(x0 + v * (x1 - x0), y - 4 - (s - 1) * 3.6, 2.3, COL[1], d); }); };
     row(WOB[0], 298, 'each dot: average of B’s first 3 pulls in one visit', 1); row(WOB[1], 388, 'average of the first 30 pulls', 6.5); } },

 { h: 'The benefit of the doubt', p: ['**UCB** gives each machine a hopeful score: its estimate **plus a bonus**. The bonus is big when Pip has pulled the machine only a few times and small when it knows the machine well. Then Pip pulls the machine with the highest score.', 'A machine can win because it is good, or because Pip barely knows it. This is **optimism in the face of uncertainty**.'],
   draw(t) { B.arcade({}); pip(200, t);
     const q = [.3, .6, .5, .1], bo = [.25, .1, .45, .2], g = clamp((t - 2.5) / 1.5);
     scoreChart(q, bo, { max: 1, ticks: [0, 1], capA: g, labels: t > 4, a: clamp(t - .5), chosen: t > 6 ? 2 : -1 });
     if (t > 1) txt('bar = estimate', 160, 274, { s: 14, c: C.soft, a: clamp(t - 1), al: 'right' });
     if (t > 2.5) txt('gold cap = bonus', 175, 274, { s: 14, c: '#b07d12', a: clamp(t - 2.5), al: 'left' });
     caption('top = score. Highest score gets pulled (made-up numbers)', clamp(t - 6)); } },

 { h: 'Upper Confidence Bound', p: ['Think of the bonus as stretching each estimate up to the **top of the range** the true chance could plausibly reach. Few pulls, wide range, tall cap.', 'That top is an **upper confidence bound**, hence the name UCB. If a machine is truly worse than C, pulling it shrinks its cap until its top drops below C’s. After that, Pip checks it only rarely.'],
   draw(t) { const y = 250, x0 = 40, x1 = 360, S = v => x0 + v * (x1 - x0);
     B.arcade({ per: k => ({ a: .35 }) });
     txt('chance of a crumb', x0, y - 14, { s: 13, c: C.soft, al: 'left' }); rline(x0, y, x1, y, 640, { c: C.ink }); [0, .5, 1].forEach(v => txt(String(v), S(v), y + 10, { s: 12, c: C.soft }));
     const rows = [[0, 'A: 1 of 3', 1 / 3, .55, 1.5], [3, 'D: 6 of 40', .15, .17, 3.5]];
     rows.forEach(([k, lab, est, half, a0], j) => { const a = clamp(t - a0); if (a <= 0) return; const yy = 300 + j * 50;
       alpha(a, () => { const lo = Math.max(0, est - half), hi = Math.min(1, est + half); box(S(lo), yy - 5, S(hi) - S(lo), 10, COL[k], .45); dot(S(est), yy, 5, INK[k]);
         rline(S(hi), yy - 12, S(hi), yy + 12, 650 + j, { c: '#c48a1a', w: 2.2 }); txt(lab, 40, yy - 18, { s: 14, c: INK[k], al: 'left' }); txt('upper bound', S(hi) + 4, yy - 18, { s: 13, c: '#b07d12', al: 'left' }); }); });
     caption('dot = estimate, band = plausible range (sketch)', clamp(t - 5)); } },

 { h: 'The bonus, in words', p: ['UCB’s bonus: **c × square root of (2 × ln t ÷ n)**. **c** is a scale, 1 in the classic **UCB1**; **n**, this machine’s pulls: 4 times the pulls, half the bonus, like the wobble.', '**t**: total pulls. Its natural log, **ln t**, grows forever, slowly: 4.6 at 100, 6.9 at 1000. An ignored machine’s bonus thus creeps up: always another look, but rarely.'],
   shot: [4, 13],
   draw(t) { txt('bonus  =  c × √( 2 × ln t ÷ n )', 200, 44, { s: 24, c: C.ink, a: clamp(t * 2) });
     [['n', 'pulls of this machine', INK[1], 1.5], ['t', 'pulls so far, all machines', INK[0], 6], ['c', 'bonus scale: 1 in UCB1', '#b07d12', 9]].forEach(([s, d, col, a0], i) => { const a = clamp(t - a0); txt(s + ':', 70, 84 + i * 26, { s: 19, c: col, a, al: 'right' }); txt(d, 80, 84 + i * 26, { s: 17, c: C.ink, a, al: 'left' }); });
     // bonus vs n at t = 100
     const x0 = 60, x1 = 360, y0 = 272, y1 = 186, ns = [1, 4, 16, 100], bx = n => x0 + Math.log(n) / Math.log(100) * (x1 - x0), by = v => y0 - v / 3.2 * (y0 - y1), a = clamp(t - 2.5);
     alpha(a, () => { rline(x0, y0, x1, y0, 670, { c: C.line }); txt('bonus after 100 pulls in total, for a machine pulled n times', 200, y1 - 18, { s: 13, c: C.soft });
       ns.forEach((n, i) => { const v = bonus(1, 100, n), u = clamp((t - 3 - i * .6) * 2); box(bx(n) - 12, y0 - (y0 - by(v)) * u, 24, (y0 - by(v)) * u, C.yellow, .9); rrect(bx(n) - 12, by(v), 24, y0 - by(v), 680 + i, { c: '#c99a3a', w: 1, a: u });
         txt('n = ' + n, bx(n), y0 + 12, { s: 13, c: C.soft, a: u }); txt(v.toFixed(2), bx(n), by(v) - 10, { s: 14, c: '#8a5d08', a: u }); }); });
     if (t > 9.5) card(['ln 100 = 4.6,   ln 1000 = 6.9:  slow growth', { s: 'ignored machines’ bonuses creep up', c: C.soft }], t - 9.5, { y: 300, gap: 1 }); } },

 { h: 'Pull 5: the first scores', p: () => [`We follow one visit, number 24. As before, Pip first pulls each machine once. A, C and D give nothing; B gives a crumb. Now t = 4 and every n = 1, so every bonus is the same: ${S5[0].toFixed(2)}.`, `Scores: A ${S5[0].toFixed(2)}, B ${S5[1].toFixed(2)}, C ${S5[2].toFixed(2)}, D ${S5[3].toFixed(2)}. B wins. A score above 1 is fine: it is a hopeful score, not a chance.`],
   shot: [3, 11],
   draw(t) { const i0 = Math.min(3, Math.floor(t / 1.1)), u = (t % 1.1) / 1.1;
     if (t < 4.4) { pullScene(GV, i0, u); pip(X[GV.arms[i0]], t); tallies(GV, i0 + (u > .55 ? 1 : 0)); caption('first, one pull each'); }
     else { B.arcade({ per: k => ({ glow: k === 1 && t > 9 ? .7 : 0 }) }); pip(t > 9 ? lerp(X[3], X[1], ease((t - 9) * 1.5)) : X[3], t, t > 9 ? -1 : 1); tallies(GV, 4);
       scoreChart(qAt(GV, 4), nAt(GV, 4).map(n => bonus(1, 4, n)), { capA: clamp((t - 5.5) / 1.2), chosen: t > 8.5 ? 1 : -1, labels: t > 6.8 }); } } },

 { h: 'C gets a second look', p: () => [`Pulls 5 to 8 go to B twice, then A, then D. On pull 9, C, still pulled only once, has the biggest bonus. Scores: A ${S9[0].toFixed(2)}, B ${S9[1].toFixed(2)}, C ${S9[2].toFixed(2)}, D ${S9[3].toFixed(2)}.`, `Pip pulls C… nothing again: 0 of 2. Plain greedy, in this same visit 24 with the same dice, never went back to C; it pulled B ${GG.n[999][1]} times in 1000.`],
   draw(t) { const tt = 8, show = t < 6.5;
     if (show) { B.arcade({ per: k => ({ glow: k === 2 && t > 4.5 ? .7 : 0 }) }); pip(t > 4.5 ? lerp(X[3], X[2], ease((t - 4.5) * 1.5)) : X[3], t, -1); tallies(GV, tt);
       scoreChart(qAt(GV, tt), nAt(GV, tt).map(n => bonus(1, tt, n)), { capA: clamp((t - .8) / 1.2), chosen: t > 4 ? 2 : -1, labels: t > 2 }); }
     else { pullScene(GV, 8, clamp((t - 6.5) / 1.4)); pip(X[2], t); tallies(GV, t > 7.4 ? 9 : 8); if (t > 8) bubble('nothing', X[2], 205, t - 8, C.soft);
       if (t > 9) card([`greedy, same visit:  B pulled ${GG.n[999][1]} of 1000 times`, { s: 'C never got a second pull', c: C.roseInk }], t - 9, { y: 300 }); } } },

 { h: 'Patience pays', p: () => [`C now has 0 crumbs from 2 pulls. But every pull of another machine raises t, so C’s bonus slowly grows. On pull 20 its score is on top again (${S20[2].toFixed(2)}).`, `This time the crumbs come. By pull 40, C has ${C40[0]} of ${C40[1]}. The gold caps shrink as machines are pulled, and C’s bar rises toward its true 70 percent.`],
   shot: [4, 15],
   draw(t) { const dur = .42, i = clamp(9 + Math.floor(t / dur), 9, 39), u = t / dur - Math.floor(t / dur), done = 9 + t / dur >= 40;
     const tt = done ? 40 : i;
     if (!done) { pullScene(GV, i, u); pip(X[GV.arms[i]], t); } else { B.arcade({}); pip(X[2], t); }
     tallies(GV, done ? 40 : i + (u > .55 ? 1 : 0)); const sh = done ? 40 : i;
     scoreChart(qAt(GV, sh), nAt(GV, sh).map(n => bonus(1, sh, n)), { max: 3, chosen: done ? -1 : GV.arms[i], labels: true });
     caption(done ? 'after 40 pulls' : `pull ${i + 1}`); } },

 { h: 'Where the checks go', p: () => [`After 1000 pulls here: A ${N1000[0]}, B ${N1000[1]}, C ${N1000[2]}, D ${N1000[3]}. All four tops sit near one height, ${(Q1000.reduce((a, v, k) => a + v + B1000[k], 0) / 4).toFixed(2)}.`, 'UCB pulls the highest top, which levels the tops. D’s estimate is far below C’s, so it keeps a tall cap: few pulls. B is close to C, so its cap must be thin: many pulls. **Promising machines get checked more, hopeless ones less.**'],
   shot: [5, 12],
   draw(t) { B.arcade({}); pip(X[2], t); for (let k = 0; k < K; k++) txt(`${N1000[k]} pulls`, X[k], 264, { s: 15, c: INK[k], a: clamp(t - .5 - k * .4) });
     scoreChart(Q1000, B1000, { max: 1, ticks: [0, .5, 1], capA: clamp((t - 2.5) / 1.5), labels: t > 4, a: clamp(t - 1.5) });
     if (t > 5) { const yy = 392 - (Q1000[2] + B1000[2]) * 84; ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = C.soft; ctx.globalAlpha = clamp(t - 5); ctx.beginPath(); ctx.moveTo(30, yy); ctx.lineTo(376, yy); ctx.stroke(); ctx.restore(); } } },

 { h: 'Keeping score', p: () => [`Is UCB better than ε-greedy? Recall **regret**: every pull of C adds 0, B adds 0.15, A 0.40 and D 0.55, the crumbs given up compared with always pulling C.`, `The chart: left to right, pulls; upward, total regret so far, averaged over ${GAMES} visits. First, ε-greedy with ε = 0.1, the teal line: ${r0(EPS1().reg[999])} after 1000 pulls.`],
   draw(t) { regChart([{ v: EPS1().reg, col: TEAL, label: 'ε-greedy', n: (t - 3) / 4 }], { y0: 250, y1: 80, max: 80, yt: [0, 40, 80] });
     card(['regret per pull:  C 0,  B 0.15,  A 0.40,  D 0.55'], t - .5, { y: 300 }); } },

 { h: 'A surprise', p: () => [`Now UCB1, in gold. After 1000 pulls its regret is ${r0(UCB(1).reg[999])}: worse than ε-greedy! Big early bonuses make it check B a lot: ${r0(UCB(1).pulls[1])} pulls on average, against ${r0(EPS1().pulls[1])} for ε-greedy.`, 'Both lines still climb. But ε-greedy never stops paying: one pull in ten is a dice roll, forever. UCB’s checks get rarer as its caps shrink, so its climb keeps slowing.'],
   draw(t) { regChart([{ v: EPS1().reg, col: TEAL, label: 'ε-greedy', ly: 12 }, { v: UCB(1).reg, col: GOLD, label: 'UCB1', n: (t - 1) / 4 }], { y0: 250, y1: 80, max: 80, yt: [0, 40, 80] });
     if (t > 6) card([`pulls of B:  UCB1 ${r0(UCB(1).pulls[1])},  ε-greedy ${r0(EPS1().pulls[1])}`, { s: 'B is close to C, so UCB keeps checking it', c: C.soft }], t - 6, { y: 300 }); } },

 { h: 'The long game', p: () => [`So play longer: 10,000 pulls, averaged over ${LONG_GAMES} visits. UCB1 overtakes ε-greedy at about pull ${fmtK(Math.round(CROSS() / 100) * 100)} and finishes far lower: ${r0(LONG('ucb').reg[LONG_T - 1])} against ${r0(LONG('eps').reg[LONG_T - 1])}.`, `From pull 5000 to 10,000, ε-greedy’s regret grows by ${r0(LONG('eps').reg[LONG_T - 1] - LONG('eps').reg[4999])}, UCB1’s by only ${r0(LONG('ucb').reg[LONG_T - 1] - LONG('ucb').reg[4999])}. The card shows why: after pull 1000, ε-greedy pulls the hopeless D about ${r0(LONG('eps').pulls[3] - LONG('eps').at[0][3])} more times, UCB1 only about ${r0(LONG('ucb').pulls[3] - LONG('ucb').at[0][3])}.`],
   draw(t) { regChart([{ v: LONG('eps').reg, col: TEAL, label: 'ε-greedy', n: (t - .5) / 3, ly: -14 }, { v: LONG('ucb').reg, col: GOLD, label: 'UCB1', n: (t - .5) / 3, ly: -14 }], { y0: 250, y1: 80, T: LONG_T, max: 300, yt: [0, 150, 300], xt: [0, 5000, 10000] });
     if (t > 5) { const le = LONG('eps'), lu = LONG('ucb'); card(['pulls of D:   by 1000     by 10,000', { s: `ε-greedy        ${r0(le.at[0][3])}              ${r0(le.pulls[3])}`, c: TEAL }, { s: `UCB1              ${r0(lu.at[0][3])}              ${r0(lu.pulls[3])}`, c: GOLD }], t - 5, { y: 296, gap: 1 }); } } },

 { h: 'A fair comparison', p: () => { const c = CMP(); return [`From earlier reels: shrinking ε = 10 ÷ t ended 1000 pulls at ${r0(c[1].short)}; trying each machine 10 times first, ${r0(c[2].short)}. Both beat UCB1’s ${r0(c[3].short)}. Even at 10,000 pulls, shrinking ε wins: ${r0(c[1].long)} to ${r0(c[3].long)}.`, 'What does UCB buy? No schedule to pick; those 10s suit only this arcade. A guarantee, proven for any arcade. A tuned bonus, c = 0.25, beats them all.']; },
   shot: [6, 12],
   draw(t) { const c = CMP(), y0 = 96; txt('total regret, lower is better', 200, 62, { s: 15, c: C.soft, a: clamp(t - .3) });
     txt('1000 pulls', 268, y0 - 8, { s: 14, c: C.soft, a: clamp(t - .5) }); txt('10,000 pulls', 352, y0 - 8, { s: 14, c: C.soft, a: clamp(t - .5) });
     c.forEach((r, i) => { const a = clamp((t - 1 - i * .9) * 2), y = y0 + 22 + i * 34, col = r.kind === 'ucb' ? GOLD : TEAL; if (a <= 0) return;
       alpha(a, () => { if (i === 3) box(20, y - 15, 362, 30, C.yellow, .4); txt(r.lab, 26, y, { s: 16, c: col, al: 'left' }); txt(r0(r.short), 268, y, { s: 17, c: C.ink }); txt(r0(r.long), 352, y, { s: 17, c: C.ink }); }); });
     if (t > 6) card(['UCB1: no schedule, a guarantee anywhere', { s: `averages over 2000 visits (${LONG_GAMES} for 10,000)`, c: C.soft }], t - 6, { y: 300, gap: 1 }); } },

 { h: 'Your turn', ctl: { label: 'c', values: CS, def: 4, fmt: v => v }, p: () => [`Drag **c**, the bonus scale, to replay 2000 visits. At c = 0, no bonus: plain greedy, settling on the wrong machine in ${UCB(0).stuck} of 2000 visits.`, `Too big, and Pip re-checks machines it already knows. Here c = 0.25 does best, ending at ${r0(UCB(.25).reg[999])}, but that is tuned to this arcade; with c = 0.1, C gets under half the pulls in ${UCB(.1).stuck} visits.`],
   shot: [3, 8],
   draw(t, st) { const ci = st.ci ?? 4, c = CS[ci], u = UCB(c), e = EPS1();
     regChart([{ v: e.reg, col: TEAL, label: 'ε-greedy', ly: -12 }, { v: u.reg, col: GOLD, label: 'UCB, c = ' + c, n: (t - (st.changed ?? 0)) / 1.2, ly: u.reg[999] > e.reg[999] ? -14 : 14 }], { y0: 210, y1: 40, max: 140, yt: [0, 70, 140] });
     const sh = u.pulls; txt('average pulls of each machine, out of 1000', 200, 282, { s: 14, c: C.soft });
     for (let k = 0; k < K; k++) { const x = 50 + k * 82, w = 70, h = sh[k] / 1000 * 80; box(x, 392 - h, w, h, COL[k], .85); rrect(x, 312, w, 80, 790 + k, { c: C.line, w: 1 }); txt(`${NAMES[k]}  ${r0(sh[k])}`, x + w / 2, 300, { s: 14, c: INK[k] }); }
     caption(c >= 2 ? 'too much optimism: A, B and D keep getting checked' : (c === 0 ? 'no bonus = greedy: ' : c === 1 ? 'UCB1, slow to settle: ' : '') + (u.stuck ? `C got under half the pulls in ${u.stuck} of 2000 visits` : 'C got most pulls in every visit')); } },

 { h: 'The fine print', p: () => [`UCB1’s **guarantee**: for rewards from 0 to 1, regret grows at most logarithmically, so each doubling of the pulls adds about the same. Here: ${r0(DBL()[0])} from 2500 to 5000 pulls, ${r0(DBL()[1])} from 5000 to 10,000. The proof needs the bonus at least this big.`, 'It assumes the chances **never change**. If a much-pulled C got worse, its thin cap would make Pip slow to notice.'],
   draw(t) { B.arcade({ per: k => ({ secret: k === 2 ? 0 : 1 }) }); const a = clamp(t - 1.5), sw = clamp((t - 3) / 1.5);
     txt(sw < .5 ? '70%' : '40%?', X[2], Y0 + MH + 14, { s: 15, c: INK[2], a }); if (t > 3) txt('what if C changed?', X[2], 264, { s: 15, c: C.roseInk, a: clamp(t - 3), al: 'center' }); pip(X[2], t);
     if (t > 5) card(['guarantee: each doubling adds about the same regret', 'needs: rewards between 0 and 1  (or rescale c)', { s: 'and chances that stay fixed', c: C.roseInk }], t - 5, { y: 296, gap: 1.1 }); } },

 { h: 'Where it is used', p: ['Optimism goes back to Lai and Robbins, 1985; UCB1 and its guarantee to Auer, Cesa-Bianchi and Fischer, 2002.', 'A game program tries moves in its head before playing; each candidate is like an arm. In 2006 Kocsis and Szepesvári used the UCB score at every such choice: **UCT**. It powered strong Go programs; AlphaGo’s search used a related rule to pick moves to explore.'],
   draw(t) { // a tiny game tree whose branches show estimate + bonus
     const nodes = [[200, 50], [110, 130], [200, 130], [290, 130], [80, 210], [140, 210], [260, 210], [320, 210]], edges = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [3, 6], [3, 7]];
     edges.forEach(([a, b], i) => { const u = clamp((t - .5 - i * .3) * 2); if (u > 0) rline(nodes[a][0], nodes[a][1] + 10, lerp(nodes[a][0], nodes[b][0], u), lerp(nodes[a][1] + 10, nodes[b][1] - 10, u), 820 + i, { c: C.ink }); });
     nodes.forEach(([x, y], i) => { const u = clamp((t - .3 - i * .25) * 2); if (u > 0) L.rcircle(x, y, 10, 830 + i, { fill: i ? C.card : C.yellow }); });
     if (t > 4.5) L.rcircle(nodes[2][0], nodes[2][1], 14, 840, { c: '#c48a1a', w: 2.2 });
     if (t > 3.5) [[1, '0.4 + 0.3'], [2, '0.2 + 0.6'], [3, '0.5 + 0.2']].forEach(([i, s], j) => txt(s, nodes[i][0] + 15, nodes[i][1] + 1, { al: 'left', s: 14, c: '#8a5d08', a: clamp(t - 3.5 - j * .4) }));
     if (t > 5) caption('each branch: estimate + bonus, explore the highest', clamp(t - 5));
     if (t > 6.5) card(['1985  Lai & Robbins: optimism', '2002  UCB1  (Auer, Cesa-Bianchi, Fischer)', '2006  UCT: UCB inside game-tree search'], t - 6.5, { y: 290, gap: .9 }); } },

 { h: 'Recap', p: ['**UCB** scores each machine as its estimate plus a bonus for uncertainty, and pulls the top score.', 'The bonus shrinks as a machine is pulled and creeps up while it is ignored. So UCB spends its checks on machines that could still turn out best, and its regret keeps flattening while ε-greedy’s grows at a steady rate.'],
   draw(t) { B.arcade({}); scoreChart(Q1000, B1000, { max: 1, ticks: [0, 1], y: 270, h: 70, a: .9 });
     card(['score = estimate + bonus;  pull the highest', 'bonus = c × √(2 ln t ÷ n): shrinks with pulls', { s: 'optimism: unsure machines get checked', c: C.soft }, { s: 'checks get rarer: regret flattens', c: C.soft }], t - .3, { y: 296, gap: 1, lh: 22 }); } },
];
STEPS.forEach(s => { if (typeof s.p === 'function') { const f = s.p; Object.defineProperty(s, 'p', { get: f, enumerable: true }); } });
RLStory.register('ucb', { title: 'Upper Confidence Bound', track: 'Bandits', onCtx: c => { ctx = c; }, steps: STEPS, _fast: fastSim, get _sim() { return sim(); } });
})();
