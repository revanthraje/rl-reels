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
// Multi-armed bandits as a story: the cheese arcade as a formal problem, estimates by averaging, and greedy locking in. Numbers from the seeded games below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
const B = RLStory.bandit, { X, K, P, NAMES, COL, INK } = B;
let ctx;
const T = 1000, GAMES = 2000, STORY = 94;
const f2 = v => (+v.toFixed(2)).toString(), f1 = v => (+v.toFixed(1)).toString(), r0 = v => Math.round(v).toString(), r10 = v => String(Math.round(v / 10) * 10), pct = v => (v < 1 && v >= .995 ? (v * 100).toFixed(1) : String(Math.round(v * 100)));
// the story game: greedy in visit 94. Its first pulls: A 0, B 1, C 1, D 0, then B 0, C 0, C 0, B 1 ... and B forever after
const G = B.play(B.greedy, STORY, T);
// machine B's crumb tape in visit 94 (the same crumbs B gives in the greedy game): first five pulls for the averaging demo
const TB = (() => { const tp = B.tape(STORY), r = []; for (let i = 0; i < 5; i++) r.push(tp[1]()); return r; })();
const RUNB = (() => { const out = []; let q = 0; TB.forEach((r, i) => { const old = q; q = old + (r - old) / (i + 1); out.push({ n: i + 1, r, old, q }); }); return out; })();
// 2000 visits of greedy: which machine it ended up pulling most, and the typical regret of such a visit
const CAT = [[], [], [], []]; for (let g = 0; g < GAMES; g++) { const H = B.play(B.greedy, g, T), n = H.n[T - 1]; CAT[n.indexOf(Math.max(...n))].push(H.regret[T - 1]); }
const CN = CAT.map(c => c.length), CMED = CAT.map(c => { const s = c.slice().sort((a, b) => a - b); return s[s.length >> 1]; });
const AVG_G = CAT.flat().reduce((a, b) => a + b, 0) / GAMES;
// your turn: try every machine m times (A, B, C, D, A, B, ...) before going greedy
const MS = [1, 2, 3, 5, 10, 20, 50, 100];
const efirst = m => (st, R) => (st.t < K * m ? st.t % K : B.argmaxR(st.q, R));
const EF = MS.map(m => { const a = B.avgRegret(efirst(m), GAMES, T); return { m, reg: a[T - 1], best: a.bestMost }; });
const fin = G.est[T - 1], nn = G.n[T - 1];
const sim = { first8: G.arms.slice(0, 8).map((a, i) => NAMES[a] + G.rews[i]).join(' '), est7: G.est[6], est8: G.est[7], fin, n: nn, regret: G.regret[T - 1], crumbs: G.rews.reduce((a, b) => a + b, 0),
  B200: G.est[199][1], minBafter7: Math.min(...G.est.slice(7).map(q => q[1])), TB, RUNB, CN, CMED, AVG_G, EF };

// ───── helpers ─────
const wins = (i, k) => (i <= 0 ? 0 : Math.round(G.est[i - 1][k] * G.n[i - 1][k])), pulls = (i, k) => (i <= 0 ? 0 : G.n[i - 1][k]);
// a lever pull on machine k at phase u (0..1) with reward r: per-machine options
function pullOpt(k, u, r) { return { pull: u < .3 ? u / .3 : u < .58 ? 1 - (u - .3) / .28 : 0, crumb: r ? clamp((u - .28) / .45) : 0, miss: r ? 0 : clamp((u - .3) / .65) }; }
// greedy game frame: pull i at phase u; estimate bars under the machines show estimates after the pulls that have landed
function greedyFrame(i, u, o = {}) {
  const a = G.arms[i], prev = i > 0 ? G.arms[i - 1] : a, landed = u > .62, n = landed ? i + 1 : i;
  B.arcade({ per: k => Object.assign({ secret: o.secret || 0 }, k === a && u < .99 ? pullOpt(k, u, G.rews[i]) : {}) });
  L.mouse(lerp(X[prev], X[a], ease(clamp(u / .25))), 224, 1, .9, u);
  for (let k = 0; k < K; k++) B.tally(k, wins(n, k), pulls(n, k), 254, { c: k === a && landed ? INK[k] : C.soft });
  const q = n ? G.est[n - 1] : [0, 0, 0, 0];
  if (o.bars !== false) B.estBars(q, { y: 372, h: 70, hi: o.hi ?? (n >= K ? q.map((v, k) => k).filter(k => q[k] > Math.max(...q) - 1e-12) : null), truth: o.truth, fmt: (v, k) => (pulls(n, k) ? f2(v) : '0') });
  return n;
}
function playAt(t, from, to, dur, o) { const k = Math.floor(Math.max(0, t) / dur), i = Math.min(to - 1, from + k), u = from + k >= to ? .99 : (Math.max(0, t) % dur) / dur; greedyFrame(i, u, o); return i; }
// tiny machine for the "no states" picture
function mini(x, y, k) { rrect(x - 15, y - 22, 30, 44, 300 + k, { fill: C.card, w: 1.2 }); box(x - 14, y - 21, 28, 9, COL[k], .9); txt(NAMES[k], x, y + 4, { s: 14 }); }
// estimate chart for the frozen step: B's and C's estimates over the 1000 pulls of visit 94
const EC = { x0: 60, x1: 360, y0: 392, y1: 296 }, ecX = t => EC.x0 + t / T * (EC.x1 - EC.x0), ecY = v => EC.y0 - v * (EC.y0 - EC.y1);
function estChart(p) {
  rline(EC.x0, EC.y0, EC.x1 + 4, EC.y0, 700, { c: C.ink, w: 1.1 }); rline(EC.x0, EC.y0, EC.x0, EC.y1 - 4, 701, { c: C.ink, w: 1.1 });
  txt('1', EC.x0 - 8, EC.y1, { s: 12, c: C.soft, al: 'right' }); txt('0', EC.x0 - 8, EC.y0, { s: 12, c: C.soft, al: 'right' });
  txt('pull 1', EC.x0, EC.y0 + 10, { s: 12, c: C.soft }); txt('1000', EC.x1, EC.y0 + 10, { s: 12, c: C.soft });
  [1, 2].forEach(k => { ctx.save(); ctx.setLineDash([5, 5]); ctx.strokeStyle = INK[k]; ctx.globalAlpha = .55; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(EC.x0, ecY(P[k])); ctx.lineTo(EC.x1, ecY(P[k])); ctx.stroke(); ctx.restore(); });
  [1, 2].forEach(k => txt(`true chance of ${NAMES[k]}: ${P[k].toFixed(2)}`, EC.x0 + 150, ecY(P[k]) - 9, { s: 12, c: INK[k], a: .8 }));
  const n = Math.floor(clamp(p) * T);
  [1, 2].forEach(k => { ctx.save(); ctx.strokeStyle = COL[k]; ctx.lineWidth = 2.4; ctx.beginPath(); for (let i = 0; i < n; i++) { const x = ecX(i + 1), y = ecY(G.est[i][k]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); });
}

const STEPS = [
 { h: 'One-armed bandits', p: ['Old slot machines had a single lever, an arm, and they slowly took your money. People called them **one-armed bandits**.', 'Pip’s cheese arcade has four machines, so it is a **multi-armed bandit**: a row of options, each paying out by its own hidden rule. It is the simplest puzzle in reinforcement learning, and one of the most useful.'],
   draw(t) { B.arcade({ sign: 'a four-armed bandit', signA: clamp(t - 4), per: k => ({ a: clamp((t - .5 - k * .4) * 2), pull: k === 1 ? clamp((t - 2.6) * 3) * (1 - clamp((t - 3.2) * 3)) : 0 }) });
     L.mouse(200, 224, 1, .9, t); if (t > 2.4) { alpha(clamp(t - 2.4), () => arrow(X[1] + 52, 196, X[1] + 46, 134, 9, { c: C.goldInk, w: 1.6 })); txt('an arm', X[1] + 54, 208, { s: 15, c: C.goldInk, a: clamp(t - 2.4) }); }
     if (t > 5) card(['one lever: a one-armed bandit', 'many levers: a multi-armed bandit', { s: 'each lever pays by its own hidden rule', c: C.soft }], t - 5, { gap: 1.1 }); }, shot: [4, 9] },
 { h: 'The pieces', p: ['Each machine is an **arm**. Each turn Pip pulls one arm: that is its action. The reward is a crumb, worth 1, or nothing, worth 0.', 'Each arm has its own **reward distribution**, the hidden rule for what it pays. Here the rule is simple: a crumb with some fixed chance. So an arm’s **true average** reward per pull equals that chance. Pip never sees it.'],
   draw(t) { B.arcade({ per: k => ({ secret: clamp((t - 6 - k * .3) * 2) }) }); L.mouse(200, 224, 1, .9, t);
     card(['arm = a machine   ·   action = which arm to pull', 'reward = 1 for a crumb, 0 for nothing', { s: 'true average of an arm = its hidden chance', c: C.soft }], t - 1, { gap: 1.5 }); }, shot: [4, 10] },
 { h: 'No map to plan over', p: ['In Pip’s maze, every move changed where Pip stood, and the next choice depended on it. In the arcade nothing changes: after any pull, the same four machines wait, with the same chances.', 'There is no situation, or **state**, to keep track of. One visit of 1000 pulls plays the part of an episode, and its return is simply the total crumbs, with no discounting. The only question, every turn, is which arm to pull.'],
   draw(t) { const M = L.gridWorld({ cs: 30, ox: 26, oy: 70 }); M.maze({ t }); M.pip(M.cx(0), M.cy(3) - 2, 1, t);
     if (t > 1) { arrow(M.cx(0), M.cy(3) - 14, M.cx(0), M.cy(2) + 8, 21, { c: C.goldInk, w: 1.5 }); }
     txt('maze: a move changes', 100, 216, { s: 15, a: clamp(t - 1) }); txt('where Pip is', 100, 236, { s: 15, a: clamp(t - 1) });
     const a2 = clamp(t - 3); if (a2 > 0) { for (let k = 0; k < K; k++) mini(232 + k * 40, 130, k); txt('arcade: after a pull,', 292, 216, { s: 15, a: a2 }); txt('the same 4 machines', 292, 236, { s: 15, a: a2 }); }
     rline(200, 60, 200, 250, 22, { c: C.line, w: 1 });
     if (t > 6) card(['bandit: actions and rewards, but no states', { s: 'each turn is a fresh choice among the same arms', c: C.soft }], t - 6, { gap: 1.1 }); }, shot: [8] },
 { h: 'Estimate by averaging', p: [`Pip can’t see B’s chance, but it can estimate it. Pip pulls B five times: crumb, nothing, crumb, nothing, crumb.`, `Its **estimate** of B is the average reward so far: 3 crumbs in 5 pulls, so 0.6. The true chance is 0.55. Close, but five pulls can’t pin it down exactly.`],
   draw(t) { const dur = 1.3, k = Math.min(4, Math.floor(Math.max(0, t - .5) / dur)), u = t < .5 ? 0 : t - .5 >= 5 * dur ? .99 : ((t - .5) % dur) / dur;
     B.arcade({ per: j => (j === 1 ? Object.assign({ glow: .6 }, u < .99 ? pullOpt(1, u, TB[k]) : {}) : { a: .4 }) }); L.mouse(X[1], 224, 1, .9, t);
     const shown = u > .62 || u >= .99 ? k + 1 : k;
     for (let i = 0; i < shown; i++) { const x = 120 + i * 40; if (TB[i]) B.crumb(x, 268, 1.2); else txt('0', x, 268, { s: 18, c: C.soft }); txt(String(i + 1), x, 290, { s: 12, c: C.soft }); }
     if (shown) txt('pull', 86, 290, { s: 12, c: C.soft });
     if (t > 7.5) card([`estimate of B = 3 crumbs ÷ 5 pulls = ${f2(RUNB[4].q)}`, { s: 'true chance (hidden): 0.55', c: C.soft }], t - 7.5, { y: 312, gap: 1.1 }); }, shot: [5, 10] },
 { h: 'Update as you go', p: ['Pip doesn’t need to keep a list of every reward. After each pull of a machine it updates one number: **new estimate = old + (reward − old) divided by n**, where n counts the pulls of that machine so far.', 'Here are B’s five pulls again. Each line gives exactly the average so far. Why? The new average is (old × (n − 1) + reward) ÷ n, which rearranges to the same rule.'],
   draw(t) { B.arcade({ per: j => (j === 1 ? { glow: .6 } : { a: .4 }) }); L.mouse(X[1], 224, 1, .9, t);
     const lines = RUNB.map(s => `n = ${s.n}:  ${f2(s.old)} + (${s.r} − ${f2(s.old)}) ÷ ${s.n}  =  ${f2(s.q)}`);
     card(lines.map((s, i) => (i === 4 ? { s, c: C.goldInk } : s)), t - .8, { y: 252, lh: 27, gap: 1.4, fs: 17 }); }, shot: [9] },
 { h: 'Why divide by n?', p: ['The bracket, reward minus old estimate, is the **surprise**: how far this pull landed from what Pip expected. Dividing by n decides how far to move.', 'The first pull moves the estimate all the way, the tenth a tenth of the way. Every pull ends up counting equally, which suits machines that never change. TD learning uses the same shape with a fixed step instead, because what it averages keeps changing as Pip learns.'],
   draw(t) { const x0 = 70, x1 = 350, y0 = 230, y1 = 80, nmax = 20, X1 = n => x0 + (n - 1) / (nmax - 1) * (x1 - x0), Y1 = v => y0 - v * (y0 - y1);
     rline(x0 - 6, y0, x1 + 6, y0, 50, { c: C.ink, w: 1.1 }); rline(x0 - 6, y0, x0 - 6, y1 - 6, 51, { c: C.ink, w: 1.1 });
     txt('how much of the surprise is used', x0 - 6, y1 - 20, { s: 14, c: C.soft, al: 'left' }); txt('1', x0 - 14, y1, { s: 12, c: C.soft, al: 'right' }); txt('0', x0 - 14, y0, { s: 12, c: C.soft, al: 'right' });
     txt('n = 1', x0, y0 + 13, { s: 12, c: C.soft }); txt('n = 20', x1, y0 + 13, { s: 12, c: C.soft });
     const m = Math.min(nmax, Math.floor(t * 3)); for (let n = 1; n <= m; n++) dot(X1(n), Y1(1 / n), 4, C.gold);
     if (t > 2.5) txt('1', X1(1) + 12, Y1(1), { s: 13, c: C.goldInk, al: 'left' }); if (t > 3) txt('1 ÷ 2', X1(2) + 8, Y1(.5) - 10, { s: 13, c: C.goldInk, al: 'left' }); if (t > 4) txt('1 ÷ 10', X1(10), Y1(.1) - 14, { s: 13, c: C.goldInk });
     if (t > 6) card(['new = old + (reward − old) ÷ n', { s: 'reward − old: the surprise', c: C.soft }, { s: '÷ n: smaller steps as evidence piles up', c: C.soft }], t - 6, { gap: 1.1 }); }, shot: [9] },
 { h: 'The greedy player', p: ['Now a full visit of 1000 pulls. We picked **visit 94** from our numbered, seeded visits. Pip plays **greedy**: it pulls each machine once, then always the arm with the highest estimate, ties broken at random.', 'Under the machines: crumbs out of pulls so far. The bars are Pip’s estimates, and the yellow box marks the current favourite. After four pulls, B and C each show 1 crumb in 1 pull: a tie.'],
   draw(t) { const i = playAt(t - .5, 0, 4, 1.4); caption(`pull ${i + 1}`); }, shot: [7] },
 { h: 'A lucky start for B, an unlucky one for C', p: ['Pull 5 goes to B, by the coin flip: nothing, so B drops to 0.5. Pull 6, C: nothing, 0.5. Another tie; pull 7 goes to C: nothing again, 0.33.', 'Now B leads, so pull 8 goes to B: a crumb, 0.67. B’s average never dips below C’s 0.33 again, so C is never pulled again. Two misses in a row hit a 70 percent machine about one time in eleven.'],
   draw(t) { const i = playAt(t - .5, 4, 8, 2); caption(`pull ${i + 1}`); }, shot: [9] },
 { h: 'Frozen', p: [`After 1000 pulls, B got ${nn[1]} pulls and its estimate settled at ${f2(fin[1])}, near its true 0.55. C’s is still ${f2(fin[2])}, from three pulls long ago, though its true chance is 0.70.`, `The chart follows both estimates through the visit: solid lines are Pip’s estimates, dashed lines the true chances. B’s line wanders and settles. C’s line never moves again.`],
   draw(t) { greedyFrame(T - 1, .99, { secret: clamp(t - 1), bars: false }); estChart((t - 2) / 5);
     if (t > 7) { txt('B', EC.x1 + 10, ecY(fin[1]), { s: 14, c: INK[1] }); txt('C', EC.x1 + 10, ecY(fin[2]), { s: 14, c: INK[2] }); } }, shot: [4, 10] },
 { h: 'Why greedy can’t recover', p: ['An estimate only changes when its arm is pulled. If an estimate is **too high**, greedy keeps pulling that arm, and the extra pulls bring it back down. Self-correcting.', `If an estimate is **too low**, greedy stops pulling the arm, so the mistake is never corrected. Here A, C and D are all stuck below their true chances. Greedy fixes over-optimism, never pessimism. Over 2000 visits it froze like this in ${GAMES - CN[2]}, as the explore reel showed.`],
   draw(t) { greedyFrame(T - 1, .99, { secret: 1, truth: true, hi: 1 });
     if (t > 1.5) { const a = clamp(t - 1.5); txt('B began too high (1): pulled, came down', 30, 284, { s: 13, c: INK[1], a, al: 'left' }); }
     if (t > 4.5) { const a = clamp(t - 4.5); txt('A, C, D too low: never re-checked', 370, 304, { s: 13, c: INK[2], a, al: 'right' }); }
     caption('bars: estimates · black lines: true chances', clamp(t - .5)); }, shot: [8] },
 { h: 'Your turn', ctl: { label: 'tries each', values: MS, def: 0, fmt: v => v }, p: ['A simple fix: try **every machine several times** before going greedy. Drag to choose how many tries.', 'Too few, and an unlucky start still freezes a good machine out. Too many, and Pip wastes pulls on A and D. The best number depends on the visit length and on how close the machines are, which Pip doesn’t know in advance.'],
   draw(t, st) { const ci = st.ci ?? 0, x0 = 60, x1 = 370, y0 = 250, y1 = 70, ymax = 120, bw = (x1 - x0) / MS.length, Yb = v => y0 - Math.min(v, ymax) / ymax * (y0 - y1);
     rline(x0, y0, x1, y0, 800, { c: C.ink, w: 1.1 }); rline(x0, y0, x0, y1 - 6, 801, { c: C.ink, w: 1.1 }); [0, 50, 100].forEach(v => txt(String(v), x0 - 8, Yb(v), { s: 12, c: C.soft, al: 'right' }));
     txt('average regret after 1000 pulls, 2000 visits', x0, y1 - 22, { s: 14, c: C.soft, al: 'left' }); txt('tries of each machine before going greedy (uneven steps)', (x0 + x1) / 2, y0 + 30, { s: 13, c: C.soft });
     EF.forEach((e, i) => { const x = x0 + i * bw + bw * .15, hi = i === ci; if (hi) box(x - 4, y1 - 4, bw * .7 + 8, y0 - y1 + 4, C.yellow, .4);
       box(x, Yb(e.reg), bw * .7, y0 - Yb(e.reg), hi ? C.teal : '#cfc4b5', .9); txt(String(e.m), x + bw * .35, y0 + 12, { s: 12, c: hi ? C.ink : C.soft });
       if (hi) txt(f1(e.reg), x + bw * .35, Yb(e.reg) - 10, { s: 14, c: '#3f8582' }); });
     const e = EF[ci]; card([`${e.m} ${e.m === 1 ? 'try' : 'tries'} each:  average regret ${f1(e.reg)}`, `C got over half the pulls in ${pct(e.best)} percent of games`, { s: ci === 0 ? 'this is plain greedy' : e.m >= 50 ? `${K * e.m} pulls spent before greedy starts` : 'more tries: fewer frozen games', c: C.soft }], 2, { gap: 0 }); }, shot: [3] },
 { h: 'Where bandits live', p: ['The first paper on this problem, by William Thompson in 1933, asked how to choose between two medical treatments. Herbert Robbins framed the general problem in 1952.', 'Today bandit methods choose which headline, ad or page design to show each visitor. Each click is a reward, and each option is an arm.'],
   draw(t) {
     const items = [['1933', 'Thompson: two treatments, which is better?'], ['1952', 'Robbins: the general bandit problem'], ['today', 'headlines, ads, page designs: a click = 1']];
     items.forEach(([a, b], i) => { const y = 70 + i * 80, al = clamp((t - .5 - i * 2) * 2); if (al <= 0) return; box(26, y - 28, 348, 56, C.card, al); rrect(26, y - 28, 348, 56, 900 + i, { c: C.line });
       txt(a, 44, y, { s: 18, al: 'left', a: al, c: C.goldInk }); txt(b, 106, y, { s: 15, al: 'left', a: al }); }); }, shot: [7] },
 { h: 'Recap', p: ['A **multi-armed bandit**: several arms, each with an unknown reward distribution, and no states. Pull one arm per turn, collect as much as you can.', 'Estimate each arm by its average, updated as you go. Pure greedy can freeze out a good arm after an unlucky start, because low estimates never get re-checked. The next reels show smarter ways to explore.'],
   draw(t) { greedyFrame(T - 1, .99, { secret: 1, bars: false });
     card(['arms, unknown rewards, no states', 'estimate = average reward of that arm', 'new = old + (reward − old) ÷ n', { s: 'greedy never re-checks a low estimate', c: C.soft }], t - .3, { gap: 1, y: 286 }); }, shot: [6] },
];

RLStory.register('bandits', { title: 'Multi-Armed Bandits', track: 'Bandits', onCtx: c => { ctx = c; }, steps: STEPS, sim });
})();
