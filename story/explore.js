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
// Exploration vs exploitation as a story: Pip's dinner dilemma, then the cheese arcade. All numbers come from the seeded games below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rline, rrect, rcircle, hachure, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
const B = RLStory.bandit, { X, K, P, NAMES, COL, INK } = B;
let ctx;
const T = 1000, GAMES = 2000, STORY = 104, YOURS = 0, YOUR_N = 20;
const RND = (st, R) => Math.floor(R() * K);                    // always explore: a machine at random every pull
const MIX = B.eps(.1);                                          // mostly exploit: 1 pull in 10 at random
// the story visit (game 104): the exploiter and the explorer, same crumb tapes
const GX = B.play(B.greedy, STORY, T), GR = B.play(RND, STORY, T);
const sum = a => a.reduce((s, v) => s + v, 0);
const crumbsX = sum(GX.rews), crumbsR = sum(GR.rews), regX = GX.regret[T - 1], regR = GR.regret[T - 1];
// averages over 2000 visits (memoised once at load)
const AV = { x: B.avgRegret(B.greedy, GAMES, T), r: B.avgRegret(RND, GAMES, T), m: B.avgRegret(MIX, GAMES, T) };
const FAV = [0, 0, 0, 0]; for (let g = 0; g < GAMES; g++) { const n = B.play(B.greedy, g, T).n[T - 1]; FAV[n.indexOf(Math.max(...n))]++; }
const r0 = v => Math.round(v).toString(), f1 = v => (+v.toFixed(1)).toString(), f2 = v => (+v.toFixed(2)).toString();
const sim = { crumbsX, crumbsR, regX, regR, nX: GX.n[T - 1], nR: GR.n[T - 1], estR: GR.est[T - 1], avX: AV.x[T - 1], avR: AV.r[T - 1], avM: AV.m[T - 1], avX100: AV.x[99], avM100: AV.m[99], FAV, first4: GX.rews.slice(0, 4), shareM: AV.m.share, shareX: AV.x.share };

// ───── helpers ─────
function dim(a = .72, y0 = 0, y1 = 285) { box(0, y0, W, y1 - y0, C.bg, a); }
const wins = (G, i, k) => (i <= 0 ? 0 : Math.round(G.est[i - 1][k] * G.n[i - 1][k])), pulls = (G, i, k) => (i <= 0 ? 0 : G.n[i - 1][k]);
// one frame of a game: pull index i (0-based) in progress with phase u in [0, 1). Lever down, crumb or puff, tallies update.
function gameFrame(G, i, u, o = {}) {
  const a = G.arms[i], prev = i > 0 ? G.arms[i - 1] : a, landed = u > .62, n = landed ? i + 1 : i;
  B.arcade({ per: k => k !== a ? { glow: o.glow?.(k), secret: o.secret } : { pull: u < .32 ? u / .32 : u < .6 ? 1 - (u - .32) / .28 : 0, crumb: G.rews[i] ? clamp((u - .3) / .45) : 0, miss: G.rews[i] ? 0 : clamp((u - .32) / .6), glow: o.glow?.(k), secret: o.secret }, sign: o.sign });
  const px = lerp(X[prev], X[a], ease(clamp(u / .25)));
  L.mouse(px, 224, 1, .9, u);
  if (o.tally !== false) for (let k = 0; k < K; k++) B.tally(k, wins(G, n, k), pulls(G, n, k), 254, { c: k === a && landed ? INK[k] : C.soft });
  return n;
}
// animate pulls [from, to) starting at t0 with dur seconds each; returns the pull index reached
function playAt(G, t, from, to, dur, o) { const k = Math.floor(Math.max(0, t) / dur), i = Math.min(to - 1, from + k), u = from + k >= to ? .99 : (Math.max(0, t) % dur) / dur; gameFrame(G, i, u, o); return { i, done: from + k >= to }; }

// shops for the dinner frame
function shop(x, name, seed, col, o = {}) {
  const w = 120, l = x - w / 2, top = 80, h = 120;
  rrect(l, top, w, h, seed, { fill: C.card, w: 1.5 }); hachure(l, top, w, 24, col, 7, seed + 3, .9); rline(l, top + 24, l + w, top + 24, seed + 4);
  txt(name, x, top + 40, { s: 17 }); rrect(x - 16, top + 70, 32, 50, seed + 6, { fill: '#efe3cf', w: 1.2 }); dot(x + 9, top + 96, 2, C.ink);
  rrect(l + 10, top + 58, 26, 22, seed + 8, { fill: '#fdf6e6', c: C.line, w: 1 }); rrect(l + w - 36, top + 58, 26, 22, seed + 9, { fill: '#fdf6e6', c: C.line, w: 1 });
  if (o.tag) txt(o.tag, x, top - 14, { s: 15, c: o.tagC || C.soft, a: o.tagA ?? 1 });
}
function town(t, o = {}) { rline(10, 200, 390, 200, 3, { c: C.line, w: 1.2 });
  shop(100, 'The Usual', 11, C.gold, { tag: o.tags ? 'always good' : '', tagA: clamp(t - 2), tagC: C.goldInk });
  shop(300, 'New Place', 21, C.teal, { tag: o.tags ? 'great?  awful?' : '', tagA: clamp(t - 3), tagC: '#3f8582' }); }
// regret chart: x = pull 0..1000, y = total regret 0..300 (averaged over 2000 visits)
const CH = { x0: 62, x1: 368, y0: 244, y1: 56, ymax: 300 };
const cX = t => CH.x0 + t / T * (CH.x1 - CH.x0), cY = v => CH.y0 - v / CH.ymax * (CH.y0 - CH.y1);
function axes(a = 1) { alpha(a, () => {
  rline(CH.x0, CH.y0, CH.x1 + 6, CH.y0, 601, { c: C.ink, w: 1.2 }); rline(CH.x0, CH.y0, CH.x0, CH.y1 - 8, 602, { c: C.ink, w: 1.2 });
  [0, 100, 200, 300].forEach(v => { txt(String(v), CH.x0 - 8, cY(v), { s: 12, c: C.soft, al: 'right' }); if (v) rline(CH.x0, cY(v), CH.x1, cY(v), 610 + v, { c: '#e9dfd0', w: 1, once: true, j: .3 }); });
  [0, 500, 1000].forEach(v => txt(String(v), cX(v), CH.y0 + 12, { s: 12, c: C.soft }));
  txt('pulls', CH.x1 - 10, CH.y0 + 26, { s: 13, c: C.soft }); txt('total regret', CH.x0 - 4, CH.y1 - 20, { s: 13, c: C.soft, al: 'left' }); }); }
function curve(A, col, p, seed, label, lx, ly) { const n = Math.floor(clamp(p) * (T / 10)); if (n < 1) return;
  ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.lineJoin = 'round'; ctx.beginPath(); ctx.moveTo(cX(0), cY(0));
  for (let j = 1; j <= n; j++) ctx.lineTo(cX(j * 10), cY(A[j * 10 - 1])); ctx.stroke(); ctx.restore();
  if (p >= 1 && label) txt(label, lx, ly, { s: 14, c: col, al: 'left' }); }
const LX = '#a54444', LR = '#8a5d08', LM = '#3f8582';

// your turn: tap a machine to pull it (crumb tape of visit 0)
function yourTap(x, y, st, t) {
  if (st.done) { Object.keys(st).forEach(k => { if (k !== 'ci') delete st[k]; }); return; }
  if (y < 60 || y > 250) return; const k = X.findIndex(v => Math.abs(x - v) < 44); if (k < 0) return;
  if (!st.tp) { st.tp = B.tape(YOURS); st.log = []; st.n = [0, 0, 0, 0]; st.w = [0, 0, 0, 0]; st.reg = 0; }
  if (st.log.length && t - st.log[st.log.length - 1].t < .5) return;
  const r = st.tp[k](); st.n[k]++; st.w[k] += r; st.reg += B.PBEST - P[k]; st.log.push({ k, r, t }); if (st.log.length >= YOUR_N) st.done = t;
}

const STEPS = [
 { h: 'Tonight’s dinner', p: ['Meet Pip, a mouse with a favourite café: **The Usual**. The food there is always good.', 'Last week a **New Place** opened down the road. It might be wonderful. It might be awful. Pip can only eat in one place tonight. Which one?'],
   draw(t) { town(t, { tags: true }); L.mouse(200, 232, 1, 1, t); if (t > 4) { txt('?', 214 + Math.sin(t * 3) * 2, 204, { s: 26, c: C.soft, a: clamp(t - 4) }); }
     if (t > 5) card(['The Usual: Pip knows it is good', 'New Place: Pip knows nothing about it'], t - 5, { gap: 1.2 }); }, shot: [3, 9] },
 { h: 'Two ways to choose', p: ['Going back to The Usual is **exploiting**: using what you already know works.', 'Trying the New Place is **exploring**: taking a choice you know less about, to learn something. Exploring might cost Pip a good dinner tonight. It might also find a better café for every night after.'],
   draw(t) { town(t); L.mouse(200, 232, 1, 1, t); const a1 = clamp(t - .8), a2 = clamp(t - 3);
     alpha(a1, () => { arrow(180, 232, 130, 214, 41, { c: C.goldInk, w: 2 }); txt('exploit', 120, 236, { s: 18, c: C.goldInk }); });
     alpha(a2, () => { arrow(222, 232, 270, 214, 42, { c: '#3f8582', w: 2 }); txt('explore', 284, 236, { s: 18, c: '#3f8582' }); });
     if (t > 5) card([{ s: 'exploit: use what you know works', c: C.goldInk }, { s: 'explore: try something to learn more', c: '#3f8582' }, { s: 'you can’t do both in the same evening', c: C.soft }], t - 5, { gap: 1.1 }); }, shot: [4, 11] },
 { h: 'It depends on what’s ahead', p: ['If Pip leaves town tomorrow, tonight is the last dinner. A great discovery would have no later dinners to improve, so The Usual is the safe bet.', 'If Pip just moved here, a great find pays off for hundreds of dinners. **Exploring is an investment**: you pay now, and the reward comes later, from better choices.'],
   draw(t) { const rowY = [90, 190];
     [['leaving tomorrow', 1], ['just moved here', 60]].forEach(([s, n], j) => { const a = clamp(t - .6 - j * 2.4); if (a <= 0) return;
       txt(s, 30, rowY[j] - 34, { s: 17, al: 'left', a });
       for (let i = 0; i < n; i++) { const xx = 36 + (i % 30) * 11.4, yy = rowY[j] + Math.floor(i / 30) * 16; dot(xx, yy, 4, i === 0 ? C.teal : '#d8ccbb', a * clamp((t - .8 - j * 2.4) * 8 - i * .15)); }
       txt(n === 1 ? 'dinners left: just tonight' : 'dinners left (and many more)', 30, rowY[j] + (n > 30 ? 44 : 22), { s: 13, al: 'left', c: C.soft, a }); });
     if (t > 6) card(['teal dot: tonight, when exploring costs something', 'grey dots: the dinners a good find improves'], t - 6, { gap: 1.2 }); }, shot: [4, 11] },
 { h: 'The cheese arcade', p: ['To study this, Pip visits an arcade with four **cheese machines**, A, B, C and D. Each lever pull drops a crumb, worth 1, or nothing, worth 0.', 'Each machine has its own hidden chance of dropping a crumb. Pip gets 1000 pulls tonight and wants as many crumbs as possible. Unlike the maze, nothing moves: after every pull the same four machines wait.'],
   draw(t) { B.arcade({ signA: clamp(t - .3), per: k => ({ a: clamp((t - .6 - k * .5) * 2), pull: k === 2 ? clamp((t - 4) * 3) * (1 - clamp((t - 4.6) * 3)) : 0, crumb: k === 2 ? clamp((t - 4.4) / 1.2) : 0 }) });
     L.mouse(lerp(200, X[2], ease(t - 3)), 224, 1, .9, t);
     if (t > 6) card(['one pull: crumb (1) or nothing (0)', 'each machine: its own hidden chance', { s: '1000 pulls: collect as many crumbs as you can', c: C.soft }], t - 6, { gap: 1.1 }); }, shot: [5, 10] },
 { h: 'A peek at the secret', p: ['We will peek inside the machines, so we can judge Pip’s choices. **Pip cannot see this.**', 'A drops a crumb 30 percent of the time, B 55 percent, C 70 percent, and D only 15 percent. So C is the best machine. Pip has to find that out by pulling.'],
   draw(t) { B.arcade({ per: k => ({ secret: clamp((t - 1 - k * .7) * 2), glow: k === 2 ? clamp(t - 4.5) : 0 }) }); L.mouse(200, 224, 1, .9, t);
     if (t > 1) txt('only we see these', 200, 254, { s: 13, c: C.soft, a: clamp(t - 3.6) });
     if (t > 5.5) card(['best machine: C, a crumb 70 times in 100 pulls', { s: 'Pip sees only what its own pulls give', c: C.soft }], t - 5.5, { gap: 1.2 }); }, shot: [5, 9] },
 { h: 'Pip the exploiter', p: ['First Pip tries each machine once. A drops a crumb. B, C and D drop nothing.', 'Then Pip follows a simple rule: **always pull the machine that has paid best so far**. That is A, with 1 crumb in 1 pull. This rule is called **greedy**.'],
   draw(t) { const r = playAt(GX, t - .4, 0, 12, 1.15); if (t > 5) card([{ s: 'after 4 pulls:  A 1 of 1,  B, C, D 0 of 1', c: C.ink }, { s: 'greedy rule: pull the best so far, so A', c: C.goldInk }], t - 5, { gap: 1.1 });
     caption(`pull ${r.i + 1}`); }, shot: [5, 13] },
 { h: 'Stuck', p: [`After 1000 pulls, Pip pulled A ${GX.n[T - 1][0]} times and collected ${crumbsX} crumbs. It never went back to C.`, 'Why? B, C and D still show 0 crumbs from their single try, and A’s average never drops to 0. One unlucky pull of C, the best machine, decided the whole evening.'],
   draw(t) { const tt = t * 120, i = clamp(Math.floor(tt), 4, T - 1); gameFrame(GX, i, .99, {});
     caption(i >= T - 1 ? 'pull 1000 of 1000' : `pull ${i + 1} (sped up)`);
     if (t > 9) { B.arcade({ per: k => ({ secret: 1, glow: k === 2 ? 1 : 0 }) }); L.mouse(X[0], 224, 1, .9, t); for (let k = 0; k < K; k++) B.tally(k, wins(GX, T, k), pulls(GX, T, k), 254, { c: INK[k] }); }
     if (t > 9.4) card([`exploiter: ${crumbsX} crumbs`, { s: 'C, the best machine, was pulled once', c: C.roseInk }], t - 9.4, { gap: 1.1 }); }, shot: [5, 12] },
 { h: 'Pip the explorer', p: ['Same visit, same luck: each machine drops its crumbs in the same order as before, so C’s first pull misses again. This time Pip **always explores**: every pull goes to a random machine.', `Pip learns a lot: its averages end close to the secret chances. But it collects only ${crumbsR} crumbs, because three pulls in four go to machines worse than C.`],
   draw(t) { const tt = t * 120, i = clamp(Math.floor(tt), 0, T - 1); gameFrame(GR, i, .99, { secret: clamp(t - 9) });
     const n = i + 1; B.estBars(GR.est[i], { y: 372, h: 66, a: clamp(t - 1), truth: t > 9 });
     caption(i >= T - 1 ? (t > 9 ? 'bars: Pip’s averages · black lines: the secret chances' : 'pull 1000 of 1000') : `pull ${n} (sped up)`); }, shot: [5, 12] },
 { h: 'Each pull has a hidden cost', p: ['To compare the two, imagine someone who knew the secrets. They would pull C every time and expect 70 crumbs per 100 pulls.', 'Every pull of another machine gives up some of that. A pull of B gives up 0.70 minus 0.55, which is 0.15 of a crumb. A gives up 0.40, D gives up 0.55, and C gives up nothing.'],
   draw(t) { B.arcade({ per: k => ({ secret: 1, glow: k === 2 ? clamp(t - 1) : 0 }) }); L.mouse(X[2], 224, 1, .9, t);
     const cost = [.4, .15, 0, .55]; for (let k = 0; k < K; k++) txt(k === 2 ? 'gives up 0' : 'gives up ' + cost[k].toFixed(2), X[k], 254, { s: 14, c: INK[k], a: clamp((t - 3.5 - k * .6) * 2) });
     if (t > 6.5) card(['cost of a pull  =  0.70  −  chance of that machine', { s: 'what Pip gave up by not knowing', c: C.soft }], t - 6.5, { gap: 1.1 }); }, shot: [5, 10] },
 { h: 'Regret', p: [`Add up those costs over the evening and you get Pip’s **regret**: the price of not knowing. The exploiter’s regret is ${f1(regX)}. The explorer’s is ${f1(regR)}.`, 'Regret counts chances, not luck: a pull of A costs 0.40 whether or not this pull happened to drop a crumb. So regret measures the choices, not the dice.'],
   draw(t) {
     const bx = [120, 280], vals = [regX, regR], nm = ['exploiter', 'explorer'], cols = [LX, LR];
     bx.forEach((x, j) => { const a = clamp(t - .8 - j * 1.6); if (a <= 0) return; const h = vals[j] / 400 * 170 * ease(clamp((t - .8 - j * 1.6) / 1.2));
       box(x - 26, 250 - h, 52, h, cols[j], .8 * a); txt(nm[j], x, 266, { s: 15, c: cols[j], a }); txt(f1(vals[j] * clamp((t - .8 - j * 1.6) / 1.2)), x, 238 - h, { s: 17, c: cols[j], a }); });
     rline(60, 250, 340, 250, 620, { c: C.line, w: 1 });
     if (t > 4.5) card([`exploiter: ${GX.n[T - 1][0]} pulls of A × 0.40, plus the other 3`, `explorer: about 250 pulls each of A, B and D`, { s: 'regret = total cost of all 1000 pulls', c: C.soft }], t - 4.5, { gap: 1.1 }); }, shot: [4, 11] },
 { h: 'One visit is not enough', p: ['That was one visit. With different luck, the exploiter might have found C. So let’s run **2000 visits** of 1000 pulls each and average.', 'Here is the chart. Left to right: pulls during the evening. Upward: total regret so far, averaged over all visits. Lower is better. A line that goes flat has stopped losing.'],
   draw(t) { axes(clamp(t - .5)); const a = clamp(t - 5);
     if (a > 0) { alpha(a, () => { rline(cX(0), cY(0), cX(750), cY(300), 630, { c: '#d9cdbc', w: 2 }); txt('stuck on A: 0.40 lost per pull', cX(520), cY(150), { s: 13, c: C.soft, al: 'left' }); }); }
     if (t > 7.5) card(['across: pull 1 to pull 1000', 'up: total regret, average of 2000 visits'], t - 7.5, { gap: 1.1 }); }, shot: [4, 10] },
 { h: 'Explore always, or never?', p: [`Gold, always explore: straight, steep, ending at ${r0(AV.r[T - 1])}. Red, always exploit: only ${r0(AV.x[T - 1])}. A **twist**: in the visit we watched, the exploiter lost more. That was bad luck; on average it does far better.`, `But red never flattens. In ${FAV[2]} of 2000 visits greedy **settled on** C, its most-pulled machine. In the other ${GAMES - FAV[2]} it stuck to B, A or D and kept losing.`],
   draw(t) { axes(); curve(AV.r, LR, (t - .5) / 3, 1, 'always explore', cX(380), cY(215));
     curve(AV.x, LX, (t - 4) / 3, 2, 'always exploit (greedy)', cX(470), cY(AV.x[T - 1]) - 14);
     if (t > 7.5) card([`greedy settled on:  C ${FAV[2]},  B ${FAV[1]},  A ${FAV[0]},  D ${FAV[3]}`, { s: 'settled on = pulled most, out of 2000 visits', c: C.soft }], t - 7.5, { gap: 1.1 }); }, shot: [4, 11] },
 { h: 'A little of both', p: [`A mixed Pip: **one pull in ten goes to a random machine**, any of the four, **the rest to the best so far**. Its teal line ends at ${r0(AV.m[T - 1])}, lower than both.`, 'Random pulls rescue an unlucky start: C gets more tries, and its true quality shows. Yet it pays for random pulls forever. How much to explore, and smarter ways, come next.'],
   draw(t) { axes(); curve(AV.r, LR, 1, 1, 'always explore', cX(380), cY(215)); curve(AV.x, LX, 1, 2, 'always exploit', cX(560), cY(AV.x[T - 1]) - 14);
     curve(AV.m, LM, (t - 1) / 3, 3, 'mix: 1 in 10 random', cX(560), cY(AV.m[T - 1]) + 16);
     if (t > 5) txt(r0(AV.m[T - 1]), cX(T) + 4, cY(AV.m[T - 1]), { s: 13, c: LM, al: 'left' }); if (t > 5) txt(r0(AV.x[T - 1]), cX(T) + 4, cY(AV.x[T - 1]) - 4, { s: 13, c: LX, al: 'left' });
     if (t > 5.5) card([`pulls of C:  mix ${Math.round(AV.m.share[2] * 100)} percent,  greedy ${Math.round(AV.x.share[2] * 100)} percent`, { s: 'averaged over 2000 visits of 1000 pulls', c: C.soft }], t - 5.5, { gap: 1.1 }); }, shot: [5, 11] },
 { h: 'Your turn', tap: yourTap, p: [`Tap a machine to pull its lever. You get ${YOUR_N} pulls. Pull the ones you believe in, but don’t forget the ones you haven’t tested enough.`, 'After your last pull, the secret chances and your regret appear. Tap again to start over.'],
   draw(t, st) { const lg = st.log || [], last = lg[lg.length - 1], u = last ? clamp((t - last.t) / 1) : 1;
     B.arcade({ per: k => ({ pull: last && last.k === k && u < .6 ? (u < .3 ? u / .3 : 1 - (u - .3) / .3) : 0, crumb: last && last.k === k && last.r ? clamp((u - .25) / .5) : 0, miss: last && last.k === k && !last.r ? clamp((u - .3) / .7) : 0, secret: st.done ? clamp(t - st.done - .8) : 0 }) });
     L.mouse(last ? lerp(lg.length > 1 ? X[lg[lg.length - 2].k] : 200, X[last.k], ease(u * 4)) : 200, 224, 1, .9, t);
     for (let k = 0; k < K; k++) B.tally(k, st.w ? st.w[k] : 0, st.n ? st.n[k] : 0, 254, { c: last && last.k === k ? INK[k] : C.soft });
     const nP = lg.length, crumbs = st.w ? sum(st.w) : 0;
     if (!st.done) { txt(`pulls ${nP} of ${YOUR_N}     crumbs ${crumbs}`, 200, 312, { s: 18 }); caption(nP ? 'tap a machine to pull again' : 'tap a machine to pull its lever'); }
     else card([`your crumbs: ${crumbs} in ${YOUR_N} pulls`, `your regret: ${f2(st.reg)}   (C every time: 0)`, { s: 'tap to play again', c: C.soft }], t - st.done - 1.2, { gap: 1 }); }, shot: [3] },
 { h: 'Where you meet this', p: ['A music app plays songs you love, but also slips in new ones to learn your taste. A website shows its usual page to most visitors and a new design to some, to see which works better.', 'Some medical trials also adapt as results come in, sending more patients to the treatment that is doing better. Each is the arcade in disguise.'],
   draw(t) { const items = [['music app', 'a favourite song, or a new one?'], ['website', 'the usual page, or a new design?'], ['medical trial', 'the treatment that works so far, or the new one?']];
     items.forEach(([a, b], i) => { const y = 70 + i * 70, al = clamp((t - .6 - i * 2) * 2); if (al <= 0) return; rrect(28, y - 26, 344, 54, 700 + i, { fill: C.card, c: C.line, fa: al });
       txt(a, 44, y - 8, { s: 18, al: 'left', a: al }); txt(b, 44, y + 13, { s: 15, al: 'left', c: C.soft, a: al }); });
     if (t > 7) card([{ s: 'exploit: the option that worked so far', c: C.goldInk }, { s: 'explore: the option you know less about', c: '#3f8582' }], t - 7, { gap: 1.1 }); }, shot: [4, 10] },
 { h: 'Recap', p: ['**Exploit** what you know, or **explore** to learn more: every choice under uncertainty faces this.', 'Never exploring can lock you onto a worse option after one unlucky try. Always exploring wastes most of your pulls. **Regret** measures the price of not knowing. The next reels show smarter ways to keep it small.'],
   draw(t) { B.arcade({ per: k => ({ secret: 1, glow: k === 2 ? .6 : 0 }) }); L.mouse(X[2], 224, 1, .9, t);
     card(['exploit: use the best so far', 'explore: try what you know less about', 'regret: what not knowing cost you', { s: 'never explore: stuck · always explore: wasteful', c: C.soft }], t - .3, { gap: 1 }); }, shot: [6] },
];

RLStory.register('explore', { title: 'Exploration vs Exploitation', track: 'Bandits', onCtx: c => { ctx = c; }, steps: STEPS, sim });
})();
