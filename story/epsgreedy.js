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
// ε-greedy as a story: Pip spins a little wheel before each pull. Replays a visit where greedy froze, then regret charts. Numbers from the seeded games below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
const B = RLStory.bandit, { X, K, P, NAMES, COL, INK } = B;
let ctx;
const T = 1000, GAMES = 2000, STORY = 266, EPS = .1;
const f1 = v => (+v.toFixed(1)).toString(), f2 = v => (+v.toFixed(2)).toString(), f3 = v => (+v.toFixed(3)).toString();
// ε-greedy that also records, for every pull, whether the wheel said 'random' (r), 'best so far' (g), or a first try (f)
function epsRec(e, log) { return (st, R) => { if (st.t < K) { log.push('f'); return st.t; } if (R() < e) { log.push('r'); return Math.floor(R() * K); } log.push('g'); return B.argmaxR(st.q, R); }; }
// the story visit 266: greedy froze on A (first pulls A 1, B 0, C 0, D 0); ε = 0.1 with the same crumb tapes
const LOG = [], GE = B.play(epsRec(EPS, LOG), STORY, T), G0 = B.play(B.greedy, STORY, T);
const RANDS = LOG.map((x, i) => (x === 'r' ? i : -1)).filter(i => i >= 0), RPK = [0, 0, 0, 0]; RANDS.forEach(i => RPK[GE.arms[i]]++);
const SWITCH = GE.arms.findIndex((a, i) => a === 2 && LOG[i] === 'r');                         // the random pull that hit C (pull 32)
// averages over 2000 visits. ε = 0 is plain greedy (the bandits reel's player).
const EPSV = [0, .01, .05, .1, .2, .3, .5, 1];
const memo = {}; const AVG = e => memo[e] ||= B.avgRegret(e === 0 ? B.greedy : B.eps(e), GAMES, T);
EPSV.forEach(AVG);
const DEC = { slow: B.avgRegret(B.epsDecay(t => Math.min(1, 10 / t)), GAMES, T), fast: null };
// ε = 1 ÷ pulls so far, counting how many pulls the wheel made random (same coin use as B.epsDecay)
let FASTR = 0; DEC.fast = B.avgRegret((st, R) => { if (st.t < K) return st.t; if (R() < Math.min(1, 1 / st.t)) { FASTR++; return Math.floor(R() * K); } return B.argmaxR(st.q, R); }, GAMES, T); FASTR /= GAMES;
// what ε = 0.1 did in the visits greedy got stuck in, and in the visits greedy got right
const STUCKON = [0, 0, 0, 0]; let LOSTWON = 0;
// of the visits where greedy ended up pulling another machine most, how many did ε = 0.1 rescue (C pulled most)?
let FROZEN = 0, RESCUED = 0; for (let g = 0; g < GAMES; g++) { const n0 = B.play(B.greedy, g, T).n[T - 1]; if (n0.indexOf(Math.max(...n0)) === 2) continue; FROZEN++; const n1 = B.play(B.eps(EPS), g, T).n[T - 1]; const f1 = n1.indexOf(Math.max(...n1)); if (f1 === 2) RESCUED++; else STUCKON[f1]++; }
for (let g = 0; g < GAMES; g++) { const n0 = B.play(B.greedy, g, T).n[T - 1]; if (n0.indexOf(Math.max(...n0)) !== 2) continue; const n1 = B.play(B.eps(EPS), g, T).n[T - 1]; if (n1.indexOf(Math.max(...n1)) !== 2) LOSTWON++; }
const A1 = AVG(.1), SLOPE = (A1[T - 1] - A1[499]) / 500;
const sim = { story: GE.arms.slice(0, 40).map((a, i) => (i + 1) + NAMES[a] + GE.rews[i] + LOG[i]).join(' '), switchAt: SWITCH + 1, estAfter: GE.est[SWITCH], estAfter41: GE.est[40],
  fin: GE.est[T - 1], n: GE.n[T - 1], regret: GE.regret[T - 1], crumbs: GE.rews.reduce((a, b) => a + b, 0), randoms: RANDS.length, RPK, greedyRegret: G0.regret[T - 1], greedyN: G0.n[T - 1],
  avg: Object.fromEntries(EPSV.map(e => [e, AVG(e)[T - 1]])), best: Object.fromEntries(EPSV.map(e => [e, AVG(e).bestMost])), share01: A1.share, slope: SLOPE,
  decSlow: DEC.slow[T - 1], decSlowBest: DEC.slow.bestMost, decFast: DEC.fast[T - 1], decFastBest: DEC.fast.bestMost, fastRandomPulls: FASTR, FROZEN, RESCUED, STUCKON, LOSTWON };

// ───── helpers ─────
const wins = (G, i, k) => (i <= 0 ? 0 : Math.round(G.est[i - 1][k] * G.n[i - 1][k])), pulls = (G, i, k) => (i <= 0 ? 0 : G.n[i - 1][k]);
function pullOpt(u, r) { return { pull: u < .3 ? u / .3 : u < .58 ? 1 - (u - .3) / .28 : 0, crumb: r ? clamp((u - .28) / .45) : 0, miss: r ? 0 : clamp((u - .3) / .65) }; }
// the wheel: slice ε (teal, 'random') and 1 − ε (gold, 'best so far'); pointer angle ang (radians from top, clockwise)
function wheel(x, y, r, e, ang, o = {}) {
  const a0 = -Math.PI / 2;
  ctx.save(); ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, r, a0, a0 + e * Math.PI * 2); ctx.closePath(); ctx.fillStyle = C.teal; ctx.globalAlpha = .85; ctx.fill();
  ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, r, a0 + e * Math.PI * 2, a0 + Math.PI * 2); ctx.closePath(); ctx.fillStyle = C.yellow; ctx.fill(); ctx.restore();
  rcircle(x, y, r, 500, { w: 1.4 });
  if (ang != null) { const ex = x + Math.sin(ang) * r * .85, ey = y - Math.cos(ang) * r * .85; rline(x, y, ex, ey, 501, { w: 2.2, once: true, j: .2 }); dot(x, y, 3, C.ink); }
  if (o.labels) { txt(o.lr || 'random', x + (o.lrx ?? r * .55), y - r - 12, { s: o.ls || 14, c: '#3f8582' }); txt('best so far', x, y + r + 14, { s: o.ls || 14, c: C.goldInk }); }
}
// pointer angle for pull i: lands inside the slice that pull's wheel result names (deterministic per pull)
function landAng(i, res, e) { const h = ((i * 2654435761) >>> 0) / 4294967296; return res === 'r' ? (.15 + .7 * h) * e * Math.PI * 2 : (e + (.1 + .8 * h) * (1 - e)) * Math.PI * 2; }
// one pull of the ε-greedy story game: the wheel spins first (u 0..0.35), then the lever
function storyFrame(i, u, o = {}) {
  if (o.still) return stillFrame(o);
  const a = GE.arms[i], prev = i > 0 ? GE.arms[i - 1] : a, res = LOG[i], v = clamp((u - .35) / .65), landed = v > .62, n = landed ? i + 1 : i;
  B.arcade({ sign: false, per: k => Object.assign({ secret: o.secret || 0, glow: res === 'r' && k === a && u > .3 ? .5 : 0 }, k === a && u < .99 && u > .35 ? pullOpt(v, GE.rews[i]) : {}) });
  const target = landAng(i, res, EPS), spin = res === 'f' ? null : (u < .35 ? target + (1 - ease(u / .35)) * Math.PI * 4 : target);
  wheel(352, 34, 20, EPS, spin); txt(res === 'f' ? 'first try of each' : res === 'r' ? 'random!' : 'best so far', 318, 34, { s: 14, al: 'right', c: res === 'r' ? '#3f8582' : res === 'f' ? C.soft : C.goldInk, a: res === 'f' ? 1 : clamp((u - .3) * 6) });
  L.mouse(lerp(X[prev], X[a], ease(clamp((u - .3) / .2))), 224, 1, .9, u);
  for (let k = 0; k < K; k++) B.tally(k, wins(GE, n, k), pulls(GE, n, k), 254, { c: k === a && landed ? INK[k] : C.soft });
  const q = n ? GE.est[n - 1] : [0, 0, 0, 0], mx = Math.max(...q);
  if (o.bars !== false) B.estBars(q, { y: 372, h: 70, hi: n >= K ? q.map((_, k) => k).filter(k => q[k] > mx - 1e-12) : null, truth: o.truth, fmt: (val, k) => (pulls(GE, n, k) ? f2(val) : '0') });
  return n;
}
// end of the visit, no pull in progress: Pip stands at the favourite C
function stillFrame(o) {
  B.arcade({ sign: false, per: k => ({ secret: o.secret || 0 }) }); L.mouse(X[2], 224, 1, .9, 0);
  for (let k = 0; k < K; k++) B.tally(k, wins(GE, T, k), pulls(GE, T, k), 254, { c: k === 2 ? INK[2] : C.soft });
  if (o.bars !== false) B.estBars(GE.est[T - 1], { y: 372, h: 70, hi: 2, truth: o.truth });
}
function playAt(t, from, to, dur, o) { const k = Math.floor(Math.max(0, t) / dur), i = Math.min(to - 1, from + k), u = from + k >= to ? .99 : (Math.max(0, t) % dur) / dur; storyFrame(i, u, o); return i; }
// regret chart (y up to ymax), shared by several steps
function chart(ymax, o = {}) {
  const x0 = 62, x1 = o.x1 ?? 366, y0 = o.y0 ?? 244, y1 = o.y1 ?? 60, cX = t => x0 + t / T * (x1 - x0), cY = v => y0 - Math.min(v, ymax) / ymax * (y0 - y1);
  rline(x0, y0, x1 + 6, y0, 601, { c: C.ink, w: 1.2 }); rline(x0, y0, x0, y1 - 8, 602, { c: C.ink, w: 1.2 });
  (o.ticks || [0, ymax / 2, ymax]).forEach(v => { txt(String(v), x0 - 8, cY(v), { s: 12, c: C.soft, al: 'right' }); if (v) rline(x0, cY(v), x1, cY(v), 610 + v, { c: '#e9dfd0', w: 1, once: true, j: .3 }); });
  [0, 500, 1000].forEach(v => txt(String(v), cX(v), y0 + 12, { s: 12, c: C.soft })); txt('pulls', x1 - 12, y0 + 26, { s: 13, c: C.soft }); txt(o.title || 'total regret, average of 2000 visits', x0 - 4, y1 - 22, { s: 13, c: C.soft, al: 'left' });
  return { cX, cY, curve(A, col, p, w = 2.4, a = 1) { const n = Math.floor(clamp(p) * (T / 10)); if (n < 1) return; ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineJoin = 'round'; ctx.beginPath(); ctx.moveTo(cX(0), cY(0));
    for (let j = 1; j <= n; j++) { if (A[j * 10 - 1] > ymax) { ctx.lineTo(cX(j * 10 - 10 + 10 * (ymax - A[j * 10 - 11]) / (A[j * 10 - 1] - A[j * 10 - 11])), cY(ymax)); break; } ctx.lineTo(cX(j * 10), cY(A[j * 10 - 1])); } ctx.stroke(); ctx.restore(); } };
}
const LG = '#a54444', LE = '#3f8582', LD = '#8a5d08';

const STEPS = [
 { h: 'Greedy needs a nudge', p: ['Four arcade machines each drop a crumb with a hidden chance, shown here but not to Pip. **Greedy** Pip tries each once, then always pulls the best average so far.', `Visit ${STORY}, a new one, starts as badly: A paid first, the others didn’t, so greedy pulled A ${G0.n[T - 1][0]} times in 1000. Its **regret**, expected crumbs lost against always pulling C, is ${f1(G0.regret[T - 1])}.`],
   say: [[["Back at the arcade, with the secret chances showing. Pip can't see them.", [20, 20, 360, 250]], ["Greedy Pip had another bad visit.", [20, 20, 360, 250]]], [["A paid first, so A got nine hundred and ninety seven pulls.", [30, 292, 340, 64]], ["C, the best, was tried once.", [30, 292, 340, 64]]], "Its regret, three hundred and ninety nine point five."],
   draw(t) { B.arcade({ per: k => ({ secret: clamp(t - 1.5), glow: k === 0 ? .5 : 0 }) }); L.mouse(X[0], 224, 1, .9, t);
     for (let k = 0; k < K; k++) B.tally(k, wins(G0, T, k), pulls(G0, T, k), 254, { c: k === 0 ? INK[0] : C.soft, a: clamp(t - 1) });
     if (t > 3) card([`greedy, visit 266: A on ${G0.n[T - 1][0]} of 1000 pulls`, { s: 'C, the best machine, was tried once', c: C.roseInk }], t - 3, { gap: 1.1 }); }, shot: [7] },
 { h: 'Spin before you pull', p: ['The fix is almost silly. Before each pull, Pip spins a little wheel. A small slice says **random**: pull any machine, chosen at random. The rest says **best so far**: pull the greedy choice.', 'The size of the random slice is called **ε**, the Greek letter epsilon. This rule is **ε-greedy**.'],
   say: [[["The fix is almost silly. Before each pull, Pip spins a wheel.", [110, 50, 180, 170]], ["A small slice says random. Pull any machine.", [110, 50, 180, 170]], ["The rest says best so far.", [110, 50, 180, 170]]], [["The size of that slice is epsilon. This rule is epsilon greedy.", [30, 292, 340, 87]]]],
   draw(t) { const ang = t < 2 ? null : (t < 4 ? landAng(3, 'g', .1) + (1 - ease((t - 2) / 2)) * 9 : landAng(3, 'g', .1));
     wheel(200, 130, 70, .1, ang, { labels: true, ls: 17, lrx: 48 });
     if (t > 5) card(['slice ε: a random machine, any of the four', 'the rest: the machine with the best average', { s: 'here ε = 0.1, a tenth of the wheel', c: C.soft }], t - 5, { gap: 1.1 }); }, shot: [4, 10] },
 { h: 'What ε = 0.1 means', p: ['With ε = 0.1, nine pulls in ten go to the favourite and one in ten is random. A random pull lands on the favourite too, one time in four.', 'So the favourite gets 92.5 percent of pulls, each other machine 2.5 percent. Every machine keeps getting a few pulls, however bad it looks. This is the policy reel’s ε-random rule.'],
   say: [[["With epsilon at zero point one, nine pulls in ten go to the favourite.", [30, 50, 350, 170]], ["And a random pull can land on the favourite too.", [30, 50, 350, 170]]], [["So the favourite gets ninety two and a half percent.", [30, 292, 340, 64]], ["But every other machine still gets a few pulls, however bad it looks.", [30, 292, 340, 64]]]],
   draw(t) { wheel(84, 120, 48, .1, null, { labels: true, ls: 14, lrx: 30 });
     const sh = [.925, .025, .025, .025], lab = ['favourite', 'other', 'other', 'other'], cols = [C.gold, C.grey, C.grey, C.grey];
     sh.forEach((v, j) => { const y = 70 + j * 36, a = clamp((t - 1.5 - j * .6) * 2); if (a <= 0) return; txt(lab[j], 170, y, { s: 15, al: 'left', a });
       box(240, y - 10, Math.max(3, v * 130 * ease(clamp((t - 1.5 - j * .6) / .8))), 20, cols[j], .85 * a); txt((v * 100).toFixed(1) + '%', 240 + Math.max(3, v * 130) + 8, y, { s: 14, al: 'left', a, c: C.soft }); });
     if (t > 5) card(['favourite: 0.9 + 0.1 × 1/4 = 0.925', 'each other machine: 0.1 × 1/4 = 0.025'], t - 5, { gap: 1.1 }); }, shot: [9] },
 { h: 'Visit 266 again', p: ['Same visit, same crumbs in every machine, but now Pip spins first. The first four pulls try each machine once, as before: A pays, B, C and D don’t.', 'Then the wheel mostly says **best so far**, and best so far is A. The top corner shows each spin.'],
   say: [[["Same visit, same crumbs. But now Pip spins first.", [20, 20, 360, 250]], ["The first four pulls try each machine once. Only A pays.", [20, 20, 360, 250]]], [["Then the wheel in the corner mostly says best so far. And that's A.", [320, 8, 66, 54]]]],
   draw(t) { const i = playAt(t - .3, 0, 14, .9); caption(`pull ${i + 1}`); }, shot: [6, 12] },
 { h: 'Random pulls', shot: [5, 13], p: ['Pull 15: the wheel says random, and the random pick happens to be A itself. Pull 19: random again, B, nothing. Pull 25: random, B, a crumb.', 'Between them, the greedy pulls keep going to A. Its average sits near 0.3. C and D still show 0 from their single tries.'],
   say: [[["Now watch the wheel.", [320, 8, 66, 54]], ["Random! It picks A itself.", [20, 20, 360, 250]], ["Random again. B, nothing. Then B, a crumb.", [20, 20, 360, 250]]], [["In between, greedy pulls keep going to A.", [20, 20, 360, 250]], ["C and D still show zero.", [20, 295, 360, 105]]]],
   draw(t) { const i = playAt(t - .3, 14, 31, .75); caption(`pull ${i + 1}`); } },
 { h: 'Pull 32', p: [`Pull 32: random, and this time it picks C. A crumb! C now shows 1 crumb in 2 pulls, an average of 0.5. That beats A’s ${f2(GE.est[SWITCH][0])}.`, 'So from now on, best so far means C. C pays 6 of its next 8 pulls, and its average climbs. One lucky spin undid one unlucky first try.'],
   say: [[["Pull thirty two. Random, and this time it picks C.", [20, 20, 360, 250]], ["A crumb!", [20, 20, 360, 250]]], [["C's average jumps to zero point five, above A.", [20, 295, 360, 105]], ["From now on, best so far means C.", [20, 295, 360, 105]]], "One lucky spin undid one unlucky first try."],
   draw(t) { let i; if (t < 4.6) { i = SWITCH; storyFrame(SWITCH, clamp((t - .3) / 3.6) * .99); } else i = playAt(t - 4.6, SWITCH + 1, 40, 1.1); caption(`pull ${i + 1}`); }, shot: [4, 13] },
 { h: 'After 1000 pulls', p: [`C got ${GE.n[T - 1][2]} of the 1000 pulls, and its average settled at ${f2(GE.est[T - 1][2])}, close to the true 0.70. Regret for the visit: ${f1(GE.regret[T - 1])}, against ${f1(G0.regret[T - 1])} for greedy.`, 'Every estimate keeps getting fresh pulls, so a wrong first impression can always be corrected. That is the whole point of ε.'],
   say: [[["After a thousand pulls, C got nine hundred and one of them.", [20, 20, 360, 250]], ["Its average settled right next to its true chance.", [20, 295, 360, 105]]], [["Regret, about thirty four, against nearly four hundred for greedy.", [60, 22, 280, 28]]], "Every estimate keeps getting fresh pulls. That's the whole point of epsilon."],
   draw(t) { storyFrame(T - 1, .99, { still: true, secret: clamp(t - 1), truth: t > 1 }); caption('bars: Pip’s averages · black lines: the true chances', clamp(t - 1.5));
     if (t > 4) { txt(`regret ${f1(GE.regret[T - 1])}  (greedy: ${f1(G0.regret[T - 1])})`, 200, 36, { s: 17, c: LE, a: clamp(t - 4) }); } }, shot: [6] },
 { h: '2000 visits', p: ['The chart: total regret so far over 1000 pulls, averaged across 2000 visits. Lower is better.', `Greedy, red, ends at ${f1(AVG(0)[T - 1])}; ε-greedy with ε = 0.1, teal, at ${f1(A1[T - 1])}. Of the ${FROZEN} visits where greedy stuck on a worse machine, ε-greedy ended on C in ${RESCUED}; in the other ${FROZEN - RESCUED} it still trailed on B. It lost only ${LOSTWON} of the visits greedy got right.`],
   say: [[["Now average over two thousand visits.", [30, 40, 360, 225]], ["Red is greedy. Teal is epsilon greedy.", [30, 40, 360, 225]], ["Teal ends lower, at forty two.", [30, 40, 360, 225]]], [["In most of greedy's stuck visits, epsilon greedy found C.", [30, 292, 340, 87]], ["And it rarely lost the visits greedy got right.", [30, 292, 340, 87]]]],
   draw(t) { const c = chart(100, { ticks: [0, 50, 100] }); c.curve(AVG(0), LG, (t - 1) / 2.5); c.curve(A1, LE, (t - 3.5) / 2.5);
     if (t > 3.5) txt('greedy ' + f1(AVG(0)[T - 1]), c.cX(T) - 4, c.cY(AVG(0)[T - 1]) - 14, { s: 14, c: LG, al: 'right', a: clamp(t - 3.5) });
     if (t > 6) txt('ε = 0.1:  ' + f1(A1[T - 1]), c.cX(T) - 4, c.cY(A1[T - 1]) + 16, { s: 14, c: LE, al: 'right', a: clamp(t - 6) });
     if (t > 7) card([`greedy stuck on a worse machine: ${FROZEN} visits`, `ε-greedy ended up on C in ${RESCUED} of them`, { s: `and lost C in ${LOSTWON} of greedy’s ${GAMES - FROZEN} good visits`, c: C.soft }], t - 7, { gap: 1.1 }); }, shot: [10] },
 { h: 'The price: exploring forever', p: ['Look at the teal line near the end: it never goes flat. Even once Pip is sure C is best, one pull in ten is still random, and three in four of those hit worse machines.', `Over the last 500 pulls, it still rises about ${f3(SLOPE)} per pull. Greedy’s lucky visits pay nothing once C is found, but its unlucky ones pay forever.`],
   say: [[["But look at the teal line near the end. It never goes flat.", [30, 40, 360, 225]], ["Even when Pip is sure C is best, one pull in ten is still random.", [30, 292, 340, 64]]], [["And most of those hit worse machines.", [30, 292, 340, 64]]], "Epsilon explores forever."],
   draw(t) { const c = chart(100, { ticks: [0, 50, 100] }); c.curve(AVG(0), LG, 1, 2.4, .35); c.curve(A1, LE, 1);
     const a = clamp(t - 1.5); if (a > 0) { alpha(a, () => { rline(c.cX(500), c.cY(A1[499]), c.cX(1000), c.cY(A1[499]), 620, { c: C.soft, w: 1.2, once: true }); rline(c.cX(1000), c.cY(A1[499]), c.cX(1000), c.cY(A1[T - 1]), 621, { c: C.soft, w: 1.2, once: true }); });
       txt(`+${f1(A1[T - 1] - A1[499])} in 500 pulls`, c.cX(760), c.cY(A1[499]) + 14, { s: 13, c: C.soft, a }); }
     if (t > 4) card(['once C is known: 1 pull in 10 still random', { s: 'random pulls: B costs 0.15, A 0.40, D 0.55', c: C.soft }], t - 4, { gap: 1.1 }); }, shot: [9] },
 { h: 'Exploring blindly', p: [`Random pulls ignore all Pip has learned. In visit 266 the wheel said random ${RANDS.length} times. D, the worst, got ${RPK[3]}, A ${RPK[0]}, and B, the close runner-up, ${RPK[1]}. The wheel splits them evenly on average, about 23 each; the spread is luck.`, 'Smarter explorers, UCB and Thompson sampling in the next reels, check where Pip is unsure or a machine might be best.'],
   say: [[["And it explores blindly.", [20, 20, 360, 250]], ["Look at the random pulls in this visit.", [20, 276, 360, 124]], ["D, the worst, got sixteen. B, the close runner up, thirty one.", [20, 276, 360, 124]]], ["On average the wheel splits them evenly, hopeless or promising. Smarter explorers come in the next reels."]],
   draw(t) { storyFrame(T - 1, .99, { still: true, secret: 1, bars: false });
     const y0 = 392, h = 66, mx = Math.max(...RPK); txt('random pulls in visit 266, per machine', 200, 286, { s: 14, c: C.soft, a: clamp(t - .5) });
     for (let k = 0; k < K; k++) { const a = clamp((t - 1 - k * .5) * 2); if (a <= 0) continue; const bh = RPK[k] / mx * h * ease(clamp((t - 1 - k * .5) / .8)); box(X[k] - 15, y0 - bh, 30, bh, COL[k], .85 * a); txt(String(RPK[k]), X[k], y0 - bh - 10, { s: 14, c: INK[k], a }); txt(NAMES[k], X[k], y0 + 10, { s: 13, c: C.soft, a }); }
     rline(24, y0, 376, y0, 77, { c: C.line, w: 1 }); }, shot: [7] },
 { h: 'Your turn', ctl: { label: 'ε', values: EPSV, def: 3, fmt: v => v }, p: ['Drag **ε** to change the size of the random slice. Each line is one ε, labelled at its end, averaged over 2000 visits.', `At 0 Pip is plain greedy and sometimes freezes. Too much ε wastes pull after pull on bad machines. In this arcade the lowest line is ε = 0.05, at ${f1(AVG(.05)[T - 1])}, a little better than 0.1.`],
   say: [[["Your turn. Drag epsilon.", [30, 40, 360, 225]], ["Each line is one setting, averaged over two thousand visits.", [30, 40, 360, 225]]], [["At zero, Pip is plain greedy.", [30, 292, 340, 87]], ["Too much, and pulls get wasted.", [30, 292, 340, 87]], ["Here, the lowest line is zero point zero five.", [30, 40, 360, 225]]]],
   draw(t, st) { const ci = st.ci ?? 3, e = EPSV[ci], c = chart(150, { ticks: [0, 50, 100, 150], x1: 330 });
     EPSV.forEach((v, j) => { if (j !== ci) c.curve(AVG(v), '#cbbfae', 1, 1.4); }); c.curve(AVG(e), LE, (t - (st.changed ?? -9)) / .8, 3);
     // line-end labels, pushed apart so they don't overlap (ε = 1 leaves the chart at the top)
     const labs = EPSV.map((v, j) => ({ v, j, y: AVG(v)[T - 1] > 150 ? null : c.cY(AVG(v)[T - 1]) })).filter(o => o.y != null).sort((p, q) => q.y - p.y);
     for (let k = 1; k < labs.length; k++) labs[k].y = Math.min(labs[k].y, labs[k - 1].y - 12);
     labs.forEach(o => txt('ε ' + o.v, c.cX(T) + 6, o.y, { s: 12, al: 'left', c: o.j === ci ? LE : C.soft }));
     const top = AVG(1); const k1 = top.findIndex(v => v > 150); txt('ε 1', c.cX(k1) + 4, c.cY(150) + 2, { s: 12, al: 'left', c: ci === 7 ? LE : C.soft });
     const fin = AVG(e)[T - 1], best = AVG(e).bestMost;
     card([`ε = ${e}:  average regret ${f1(fin)}`, `C got over half the pulls in ${Math.round(best * 100)} percent of visits`, { s: e === 0 ? 'plain greedy' : e >= .5 ? 'mostly random: wasteful' + (fin > 150 ? ' (line leaves the chart)' : '') : e <= .01 ? 'rarely explores: slow to escape' : 'a little exploring', c: C.soft }], 2, { gap: 0, y: 290 }); }, shot: [3] },
 { h: 'Explore less as you learn', p: ['Early on, exploring is cheap and useful; later it mostly wastes pulls. So let ε **shrink**.', `Here ε = 10 divided by pulls so far, capped at 1: random for roughly ten pulls, 1 in 10 after 100, about 1 in 100 by the end. Average regret: ${f1(DEC.slow[T - 1])}, below ${f1(A1[T - 1])} for fixed ε = 0.1, and near the bandits reel’s best try-first number.`],
   say: [[["Early on, exploring is cheap and useful. Later, it's mostly waste.", [30, 40, 360, 225]], ["So let epsilon shrink.", [30, 40, 360, 225]]], [["Here epsilon is ten divided by the pulls so far.", [30, 292, 340, 64]], ["Watch the new brown line. It ends at twenty six point six, well below fixed epsilon.", [30, 40, 360, 225]]]],
   draw(t) { const c = chart(100, { ticks: [0, 50, 100] }); c.curve(A1, LE, 1, 2, .45); c.curve(DEC.slow, LD, (t - 1) / 2.5);
     txt('ε = 0.1 fixed', c.cX(T) - 4, c.cY(A1[T - 1]) - 12, { s: 13, c: LE, al: 'right' });
     if (t > 3.5) txt('shrinking ε: ' + f1(DEC.slow[T - 1]), c.cX(T) - 4, c.cY(DEC.slow[T - 1]) + 16, { s: 14, c: LD, al: 'right', a: clamp(t - 3.5) });
     if (t > 5) card(['ε = 10 ÷ pulls so far, at most 1', { s: 'after 10 pulls: 1  ·  after 100: 0.1  ·  after 1000: 0.01', c: C.soft }], t - 5, { gap: 1.1 }); }, shot: [9] },
 { h: 'But not too fast', p: [`Shrink ε too fast and Pip is greedy again. With ε = 1 divided by pulls so far, it is a quarter after the first four tries, and a visit averages only ${f1(FASTR)} random pulls. Average regret: ${f1(DEC.fast[T - 1])}; C got over half the pulls in only ${Math.round(DEC.fast.bestMost * 100)} percent of visits.`, 'So ε needs tuning, and closer machines or longer visits would favour other settings.'],
   say: [[["But shrink it too fast, and Pip is greedy again.", [30, 40, 360, 225]], ["With one divided by pulls, a visit gets only about six random pulls.", [30, 40, 360, 225]]], [["C won the visit only three times in four.", [30, 292, 340, 87]]], "So epsilon needs tuning."],
   draw(t) { const c = chart(100, { ticks: [0, 50, 100] }); c.curve(AVG(0), LG, 1, 2, .45); c.curve(DEC.slow, LD, 1, 2, .6); c.curve(DEC.fast, '#b8602c', (t - 1) / 2.5);
     txt('greedy', c.cX(T) - 4, c.cY(AVG(0)[T - 1]) - 12, { s: 13, c: LG, al: 'right' }); txt('ε = 10 ÷ pulls', c.cX(T) - 4, c.cY(DEC.slow[T - 1]) + 14, { s: 13, c: LD, al: 'right' });
     if (t > 3.5) txt('ε = 1 ÷ pulls: ' + f1(DEC.fast[T - 1]), c.cX(120), c.cY(80), { s: 14, c: '#b8602c', al: 'left', a: clamp(t - 3.5) });
     if (t > 5) card(['C got over half the pulls in:', `greedy ${Math.round(AVG(0).bestMost * 100)}%  ·  1 ÷ pulls ${Math.round(DEC.fast.bestMost * 100)}%  ·  10 ÷ pulls ${Math.round(DEC.slow.bestMost * 100)}%`, { s: 'of 2000 visits', c: C.soft }], t - 5, { gap: 1 }); }, shot: [9] },
 { h: 'Where you’ll see it', p: ['ε-greedy is everywhere because it is one line of code. Pip already used it in the TD learning reel: one move in five was random, so ε was 0.2.', 'DeepMind’s Atari player DQN, published in 2015, started with ε = 1, fully random, lowered it to 0.1 over its first million frames of play, and kept it there, so it never stopped trying new moves.'],
   say: [[["Epsilon greedy is everywhere, because it's one line of code.", [20, 60, 190, 180]], ["Pip used it in the TD reel, one random move in five.", [20, 60, 190, 180]]], [["The Atari player DQN started fully random.", [215, 80, 170, 160]], ["Watch its epsilon drop to zero point one over a million frames, then stay there.", [215, 80, 170, 160]]]],
   draw(t) { const M = L.gridWorld({ cs: 34, ox: 30, oy: 70 }); M.maze({ t }); M.pip(M.cx(0), M.cy(3) - 2, 1, t); txt('TD reel: ε = 0.2', 115, 228, { s: 15, c: C.soft });
     const a = clamp(t - 3); if (a > 0) { const x0 = 230, x1 = 370, y0 = 200, y1 = 90, Xf = f => x0 + f * (x1 - x0), Yf = v => y0 - v * (y0 - y1);
       alpha(a, () => { rline(x0, y0, x1, y0, 640, { c: C.ink, w: 1.1 }); rline(x0, y0, x0, y1 - 6, 641, { c: C.ink, w: 1.1 }); rline(Xf(0), Yf(1), Xf(.5), Yf(.1), 642, { c: C.teal, w: 2.2 }); rline(Xf(.5), Yf(.1), Xf(1), Yf(.1), 643, { c: C.teal, w: 2.2 }); });
       txt('1', x0 - 8, Yf(1), { s: 12, c: C.soft, al: 'right', a }); txt('0.1', x0 - 8, Yf(.1), { s: 12, c: C.soft, al: 'right', a }); txt('ε', x0 + 6, y1 - 14, { s: 15, a });
       txt('first million frames', Xf(.25), y0 + 12, { s: 12, c: C.soft, a }); txt('DQN (2015)', 300, 228, { s: 15, c: C.soft, a }); }
     if (t > 6) card(['TD reel: 1 random move in 5, so ε = 0.2', 'DQN: ε from 1 down to 0.1, then fixed'], t - 6, { gap: 1.1 }); }, shot: [9] },
 { h: 'Recap', p: ['**ε-greedy**: with chance ε pull a random arm, otherwise the arm with the best average. Every arm keeps being re-checked, so an unlucky start can be undone.', 'It is simple and often good enough. But it explores blindly, spending as many random pulls on hopeless arms as on promising ones, and ε has to be tuned. Shrinking ε over time helps.'],
   say: [[["So, with chance epsilon, pull a random arm. Otherwise, the best so far.", [30, 292, 340, 110]], ["An unlucky start can always be undone.", [30, 292, 340, 110]]], ["It's simple and often good enough. But it explores blindly, and epsilon needs tuning."]],
   draw(t) { wheel(200, 110, 56, .1, landAng(7, 'r', .1), { labels: true, ls: 15, lrx: 40 });
     card(['chance ε: a random arm', 'otherwise: the best average so far', 'fixes greedy’s frozen estimates', { s: 'blind and never-ending; shrink ε, tune it', c: C.soft }], t - .3, { gap: 1 }); }, shot: [6] },
];

RLStory.register('epsgreedy', { title: 'ε-Greedy', track: 'Bandits', onCtx: c => { ctx = c; }, steps: STEPS, sim });
})();
