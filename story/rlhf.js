// RLHF as a story: Bot, a tiny stand-in for a language model, answers Pip's question "How do I get to the cheese?" by picking one of
// eight whole answers. A good answer is hard to write down as a rule, so simulated people compare pairs of Bot's answers; a three-knob
// reward model is fitted to those comparisons (Bradley–Terry); then PPO tunes Bot's chances toward high reward-model scores.
// Without a KL leash, PPO finds the answer the reward model over-scores (a repeated route); with the leash, people's verdict improves.
// Every number comes from the seeded simulation below (reel.sim).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;

// ───── the world: eight answers, the people's hidden taste, Bot's starting chances (after imitation)
const ANS = ['Up three, then right four.', 'Up three, right four. Mind the cat!', 'Follow your nose.', 'Down, then left.', 'I don’t know.',
  'Cheese is nice!', 'Good luck, little mouse!', 'Up three, right four, up three, right four.'];
const TASTE = [2.5, 3.0, -0.5, -1.5, -0.5, -1.0, -0.5, -1.0];       // people's hidden taste (only the simulation knows it)
const REF = [0.10, 0.07, 0.20, 0.14, 0.16, 0.14, 0.18, 0.01];      // Bot's chances after copying example answers
const NA = 8, SPAM = 7;
const FEAT = ANS.map(s => [s.split(/\s+/).length, (s.match(/\b(up|right)\b/gi) || []).length, (s.match(/\b(down|left)\b/gi) || []).length]);
const sig = x => 1 / (1 + Math.exp(-x));
function samp(P, R) { let u = R(), i = 0; while (i < P.length - 1 && u >= P[i]) { u -= P[i]; i++; } return i; }
const softmax = th => { const m = Math.max(...th), e = th.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };

// ───── 1. comparisons: Bot answers twice (skip identical pairs), a simulated person picks with chance sigmoid(taste gap)
const NC = 400, CMP = [];
{ const R = rng(12); while (CMP.length < NC) { const a = samp(REF, R), b = samp(REF, R); if (a === b) continue; const aw = R() < sig(TASTE[a] - TASTE[b]); CMP.push(aw ? [a, b] : [b, a]); } }
// ───── 2. reward model: score = w · (words, up-or-right words, down-or-left words), fitted by gradient ascent on the Bradley–Terry likelihood
const WHIST = [];
const WFIT = (() => { let w = [0, 0, 0];
  for (let it = 0; it < 4000; it++) { const g = [0, 0, 0];
    for (const [x, y] of CMP) { const d = FEAT[x].map((f, j) => f - FEAT[y][j]), p = sig(d[0] * w[0] + d[1] * w[1] + d[2] * w[2]); d.forEach((v, j) => g[j] += (1 - p) * v); }
    w = w.map((v, j) => v + 0.05 * g[j] / NC); if (it % 40 === 0) WHIST.push(w.slice()); }
  WHIST.push(w.slice()); return w; })();
const RMS = FEAT.map(f => f[0] * WFIT[0] + f[1] * WFIT[1] + f[2] * WFIT[2]);
const AGREE = CMP.filter(([x, y]) => RMS[x] > RMS[y]).length;
const SPAMC = CMP.filter(c => c.includes(SPAM)), SPAMW = SPAMC.filter(c => c[0] === SPAM).length;
const CNT = new Array(NA).fill(0); CMP.forEach(([x, y]) => { CNT[x]++; CNT[y]++; });
// ───── 3. PPO on Bot's chances: batch of 32 answers, reward = RM score − β log(chance ÷ original chance), advantage = reward − batch average,
// 4 passes, ratio clipped to 0.8–1.2, step size 0.5. People's verdict = chance a person prefers new Bot's answer over the original Bot's.
const NU = 100, NB = 32, NPASS = 4, LR = 0.5, EPS = 0.2;
const verdict = P => { let s = 0; for (let a = 0; a < NA; a++) for (let b = 0; b < NA; b++) s += P[a] * REF[b] * sig(TASTE[a] - TASTE[b]); return s; };
const klRef = P => P.reduce((s, p, i) => s + (p > 1e-12 ? p * Math.log(p / REF[i]) : 0), 0);
const rmAvg = P => P.reduce((s, p, i) => s + p * RMS[i], 0);
const memo = {};
function ppo(beta) { if (memo[beta]) return memo[beta];
  const R = rng(3); let th = REF.map(Math.log); const H = [{ P: REF.slice(), rm: rmAvg(REF), win: verdict(REF), kl: 0 }];
  for (let u = 0; u < NU; u++) { const P = softmax(th), S = [];
    for (let i = 0; i < NB; i++) { const a = samp(P, R); S.push({ a, p: P[a], r: RMS[a] - beta * Math.log(P[a] / REF[a]) }); }
    const mu = S.reduce((s, x) => s + x.r, 0) / NB; S.forEach(x => { x.A = x.r - mu; });
    for (let e = 0; e < NPASS; e++) { const Pn = softmax(th), g = new Array(NA).fill(0);
      for (const x of S) { const ratio = Pn[x.a] / x.p, cl = clamp(ratio, 1 - EPS, 1 + EPS); if (ratio * x.A > cl * x.A) continue;
        for (let b = 0; b < NA; b++) g[b] += x.A * ratio * ((b === x.a) - Pn[b]) / NB; }
      th = th.map((v, i) => v + LR * g[i]); }
    const Pn = softmax(th); H.push({ P: Pn, rm: rmAvg(Pn), win: verdict(Pn), kl: klRef(Pn) }); }
  return (memo[beta] = H); }
const OFF = ppo(0), ON = ppo(1), FO = OFF[NU], FN = ON[NU];
const BETAS = [0, 0.25, 0.5, 1, 2, 4];
const PEAK = OFF.reduce((b, h, i) => h.win > OFF[b].win ? i : b, 0), FLIP = OFF.findIndex(h => h.P[SPAM] > .5);
const BEST = verdict([0, 1, 0, 0, 0, 0, 0, 0]);
// the leash example: spam said with chance 0.5 instead of 0.01
const LEASH_EX = Math.log(0.5 / REF[SPAM]);

// ───── formatting
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−'), f1 = v => (Math.abs(v) < .05 ? 0 : v).toFixed(1).replace('-', '−');
const f3 = v => (Math.abs(v) < .0005 ? 0 : v).toFixed(3).replace('-', '−');
const pc = v => Math.round(v * 100) + '%';
const GREEN = '#5d9a5b', BLUE = '#4f7fb0';

// ───── drawing helpers
function bot(x, y, s, t) { L.robot(x, y, s, t); }
const MO = '#9b7fc0', MOINK = '#5f4689';   // Mo's colour (fill) and ink (people_world.md)
function teacher(x, y, face = 1, s = 1, t = 0) {            // Mo: Pip's mouse with glasses and a plum scarf
  L.mouse(x, y, face, s * 1.08, t); const c = L.ctx; c.save(); c.translate(x, y + Math.sin(t * 9) * 1.2); c.scale(face * s * 1.08, s * 1.08);
  c.strokeStyle = MO; c.lineWidth = 3.2; c.lineCap = 'round'; c.beginPath(); c.moveTo(5, -7); c.quadraticCurveTo(8, 1, 6, 8); c.stroke();
  c.beginPath(); c.moveTo(6, 6); c.lineTo(2, 12); c.stroke();
  c.strokeStyle = L.C.ink; c.lineWidth = 1.1; c.beginPath(); c.arc(13.5, -2, 3.3, 0, 7); c.stroke(); c.beginPath(); c.moveTo(10.2, -2.5); c.lineTo(7, -4); c.stroke();
  c.restore();
}
function say(s, x, y, w, o = {}) {           // speech box with a tail toward (tx, ty)
  const h = o.h ?? 26, a = o.a ?? 1; if (a <= 0) return;
  alpha(a, () => { box(x, y, w, h, o.fill || C.card); rrect(x, y, w, h, o.seed || 500, { c: o.c || C.line, w: o.lw || 1.2 });
    if (o.tx != null) { const bx = clamp(o.tx, x + 10, x + w - 10), by = o.ty < y ? y : y + h; rline(bx - 5, by, o.tx, o.ty, (o.seed || 500) + 9, { c: o.c || C.line, w: 1.2, once: true }); rline(bx + 5, by, o.tx, o.ty, (o.seed || 500) + 10, { c: o.c || C.line, w: 1.2, once: true }); }
    txt(s, x + w / 2, y + h / 2 + 1, { s: o.s || 14, c: o.tc }); });
}
// the answer table: rows of answers with optional columns. cols = [{ v: [...], fmt, x, col, bar: { x, w, max } }]
const ROWH = 18;
function table(y0, o = {}) {
  const rows = o.rows || [...Array(NA).keys()];
  if (o.head) o.head.forEach(([s, x, al]) => txt(s, x, y0 - 12, { s: 12, c: C.soft, al: al || 'right' }));
  rows.forEach((i, k) => { const y = y0 + k * ROWH + 4, a = o.rowA ? o.rowA(i, k) : 1; if (a <= 0) return;
    if (o.hi && o.hi(i)) box(16, y - ROWH / 2 + 1, 368, ROWH - 2, o.hiCol || C.yellow, .35 * a);
    txt(ANS[i], 22, y, { s: 13, al: 'left', a, c: o.tc ? o.tc(i) : C.ink });
    (o.cols || []).forEach(cl => { const v = cl.v[i];
      if (cl.bar) { const bw = clamp(v / cl.bar.max) * cl.bar.w; box(cl.bar.x, y - 5, bw, 10, cl.col ? (typeof cl.col === 'function' ? cl.col(i) : cl.col) : C.teal, .8 * a); }
      txt(cl.fmt ? cl.fmt(v) : String(v), cl.x, y, { s: 13, al: 'right', a, c: cl.tc ? cl.tc(i, v) : C.ink }); }); });
}
// line chart over updates: o = { x0, x1, y0 (bottom), y1 (top), lo, hi, n, series: [{ v, col, w, label }], ref: [{ v, label }], title, ticks }
function chart(o) {
  const lo = o.lo, hi = o.hi, X = i => o.x0 + i / NU * (o.x1 - o.x0), Y = v => o.y0 - (v - lo) / (hi - lo) * (o.y0 - o.y1), a = o.a ?? 1;
  alpha(a, () => {
    rline(o.x0 - 4, o.y0, o.x1 + 4, o.y0, 960 + o.y0, { c: C.line, w: 1, once: true }); rline(o.x0 - 4, o.y0, o.x0 - 4, o.y1 - 2, 961 + o.y0, { c: C.line, w: 1, once: true });
    (o.ticks || []).forEach(v => txt(String(v).replace('-', '−'), o.x0 - 9, Y(v), { s: 12, c: C.soft, al: 'right' }));
    if (o.xl) { txt('update 0', o.x0, o.y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt('update ' + NU, o.x1, o.y0 + 11, { s: 12, c: C.soft, al: 'right' }); }
    if (o.title) txt(o.title, o.x0 - 8, o.y1 - 16, { s: 13, c: C.soft, al: 'left' });
    (o.ref || []).forEach(e => { ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.grey; ctx.lineWidth = 1.1; ctx.beginPath(); ctx.moveTo(o.x0, Y(e.v)); ctx.lineTo(o.x1, Y(e.v)); ctx.stroke(); ctx.restore();
      if (e.label) txt(e.label, o.x0 + 30, Y(e.v) + 9, { s: 12, c: C.soft, al: 'left' }); });
  });
  (o.series || []).forEach(s => { const n = Math.min(s.n ?? NU, NU); ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 2; ctx.lineJoin = 'round'; ctx.beginPath();
    for (let i = 0; i <= n; i++) { const x = X(i), y = Y(s.v[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore();
    if (s.label && n >= NU) txt(s.label, o.x1 + 6, Y(s.v[NU]), { s: 13, c: s.col, al: 'left', a }); else if (s.dot) dot(X(n), Y(s.v[n]), 3, s.col, a); });
  return { X, Y };
}
const RMV = H => H.map(h => h.rm), WINV = H => H.map(h => h.win);
function twoCharts(t, runs, o = {}) {        // top: reward-model score; bottom: people's verdict
  const n = o.n ?? NU, a1 = o.a1 ?? 1, a2 = o.a2 ?? 1;
  chart({ x0: 60, x1: 330, y0: 136, y1: 50, lo: 0, hi: 5, ticks: [0, 5], title: 'reward-model score of Bot’s answers (average)', a: a1,
    series: runs.map(r => ({ v: RMV(r.H), col: r.col, n, label: n >= NU ? f2(r.H[NU].rm) : '', dot: true })) });
  chart({ x0: 60, x1: 330, y0: 274, y1: 190, lo: 0, hi: 1, ticks: [0, 0.5, 1], xl: true, title: 'people’s verdict: new Bot preferred over original', a: a2,
    ref: [{ v: .5, label: '0.5 = no better' }], series: runs.map(r => ({ v: WINV(r.H), col: r.col, n, label: n >= NU ? f2(r.H[NU].win) : '', dot: true })) });
}
function scene(t, o = {}) {                   // Pip asks, Bot answers
  L.mouse(60, 92, 1, 1.1, t); bot(330, 88, 2, t);
  say('How do I get to the cheese?', 18, 22, 200, { tx: 70, ty: 70, a: o.qa ?? 1, seed: 510 });
  if (o.ans != null) say(ANS[o.ans], o.ax ?? 120, 124, o.aw ?? 250, { tx: 316, ty: 110, a: o.aa ?? 1, seed: 520, tc: o.atc });
}
function miniMaze(x0, y0, cs, t, o = {}) {
  const G = L.gridWorld({ cs, ox: x0, oy: y0 }); G.maze({ t });
  if (o.route) { const P = [[0, 3], [0, 2], [0, 1], [0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], u = o.route;
    for (let i = 0; i < P.length - 1; i++) { const v = clamp(u * 7 - i); if (v <= 0) break; arrow(G.cx(P[i][0]), G.cy(P[i][1]), lerp(G.cx(P[i][0]), G.cx(P[i + 1][0]), v), lerp(G.cy(P[i][1]), G.cy(P[i + 1][1]), v), 700 + i, { c: GREEN, w: 2, head: v > .9 ? 5 : 0.01 }); } }
  G.pip(G.cx(0), G.cy(3), 1, t); return G;
}
const chanceCol = { v: null, x: 378, fmt: f2, bar: { x: 290, w: 64, max: 1 }, col: C.teal };
const withCh = P => [Object.assign({}, chanceCol, { v: P })];

const STEPS = [
 { h: 'Bot, the answer machine', p: ['Pip asks Bot a question: how do I get to the cheese? Bot is a tiny stand-in for a **language model**. A real one writes word by word; Bot just picks one of eight whole answers, each with a chance.',
   'Bot learned these chances by copying example answers that people wrote, as in the imitation reel. Some are good, many are not.'],
   shot: [4, 11],
   draw(t) { scene(t, { ans: t > 2 ? 0 : null, aa: clamp(t - 2) });
     if (t > 4) table(180, { head: [['Bot’s answers', 22, 'left'], ['chance', 378]], cols: withCh(REF), rowA: (i, k) => clamp((t - 4 - k * .25) * 3) }); } },
 { h: 'What is a good answer?', p: ['From the start, the way is up three, then right four, just above the cat. So some answers help, some do not.',
   'Could a rule in code judge them? Counting words fails: the longest answer repeats itself, and people dislike it. Direction words fail too: Down, then left, has them and is wrong. For real questions, coding helpful and honest is far harder.'],
   shot: [4, 11],
   draw(t) { miniMaze(100, 22, 40, t, { route: clamp((t - .8) / 3) });
     card([{ s: 'more words? the longest one is wrong  ✗', c: C.roseInk }, { s: 'direction words? Down, then left, has two  ✗', c: C.roseInk }, 'nobody can write the rule for a good answer'], t - 4.5, { y: 296, gap: 1.6, fs: 16 });
     table(206, { rows: [0, 1, 3, 7], head: [['four of Bot’s answers', 22, 'left']], rowA: (i, k) => clamp((t - 1 - k * .3) * 3), tc: i => i <= 1 ? GREEN : C.ink }); } },
 { h: 'Easy to compare', p: ['But people can easily say which of two answers they **prefer**. Bot answers twice, a person picks one.',
   'Here the people are simulated; Mo, the old mouse with glasses, stands in for them. Each answer has a hidden **taste** score that drives their choices. Bot never sees it, only the comparisons.'],
   shot: [4, 11],
   draw(t) { say('A: ' + ANS[0], 20, 30, 220, { seed: 530, a: clamp(t - .5), c: t > 3 ? GREEN : C.line, lw: t > 3 ? 2 : 1.2 }); say('B: ' + ANS[2], 20, 70, 220, { seed: 540, a: clamp(t - 1) });
     teacher(325, 78, -1, 1.7, t); if (t > 3) { txt('A ✓', 320, 120, { s: 18, c: GREEN, a: clamp(t - 3) }); }
     if (t > 5) table(176, { head: [['hidden taste (only the simulation knows)', 22, 'left']], cols: [{ v: TASTE, x: 378, fmt: f1, tc: (i, v) => v > 0 ? GREEN : C.roseInk }], rowA: (i, k) => clamp((t - 5 - k * .2) * 3) }); } },
 { h: 'How people choose', p: ['People are noisy. Our simulated person picks answer A with chance 1 ÷ (1 + e to the minus gap), where the gap is A’s taste minus B’s: the **Bradley–Terry** model.',
   `The chart shows this curve: across, the taste gap; up, the chance of picking A. Equal taste gives 0.5. The two answers above differ by 3, so A wins with chance ${f2(sig(3))}, not always.`],
   shot: [4, 11],
   draw(t) { const x0 = 60, x1 = 350, y0 = 250, y1 = 60, X = g => x0 + (g + 5) / 10 * (x1 - x0), Y = p => y0 - p * (y0 - y1), u = clamp((t - 1.5) / 2.5);
     rline(x0, y0, x1, y0, 1001, { c: C.line, w: 1, once: true }); rline(X(0), y0, X(0), y1 - 4, 1002, { c: C.line, w: 1, once: true });
     [-4, -2, 0, 2, 4].forEach(g => txt(String(g).replace('-', '−'), X(g), y0 + 12, { s: 12, c: C.soft })); [0, .5, 1].forEach(p => txt(String(p), x0 - 6, Y(p), { s: 12, c: C.soft, al: 'right' }));
     txt('taste gap, A minus B', x1, y0 + 28, { s: 13, c: C.soft, al: 'right' }); txt('chance A is picked', X(0) + 6, y1 - 10, { s: 13, c: C.soft, al: 'left' });
     ctx.save(); ctx.strokeStyle = BLUE; ctx.lineWidth = 2.2; ctx.beginPath(); for (let k = 0; k <= 100 * u; k++) { const g = -5 + k / 10; k ? ctx.lineTo(X(g), Y(sig(g))) : ctx.moveTo(X(g), Y(sig(g))); } ctx.stroke(); ctx.restore();
     if (t > 5) { const a = clamp(t - 5); alpha(a, () => { ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = C.goldInk; ctx.beginPath(); ctx.moveTo(X(3), y0); ctx.lineTo(X(3), Y(sig(3))); ctx.lineTo(X(0), Y(sig(3))); ctx.stroke(); ctx.restore(); });
       dot(X(3), Y(sig(3)), 4, C.goldInk, a); txt(`gap 3 → ${f2(sig(3))}`, X(3) + 8, Y(sig(3)) + 16, { s: 14, c: C.goldInk, a, al: 'left' }); }
     card(['equal taste → 0.5', 'bigger gap → closer to always', { s: 'never quite certain', c: C.soft }], t - 7, { y: 300, gap: 1 }); } },
 { h: `${NC} comparisons`, p: [`Bot answers twice, ${NC} times over, and the simulated people pick. Most pairs are two of Bot’s common answers.`,
   `Bot’s rare repeated answer, the last row, turns up in only ${SPAMC.length} of the ${NC} pairs. It wins ${SPAMW} and loses ${SPAMC.length - SPAMW}. Keep an eye on it.`],
   shot: [3, 11],
   draw(t) { const n = Math.min(NC, Math.floor(Math.max(0, t - .5) ** 2 * 18)), win = new Array(NA).fill(0), seen = new Array(NA).fill(0);
     for (let k = 0; k < n; k++) { const [x, y] = CMP[k]; win[x]++; seen[x]++; seen[y]++; }
     const cur = CMP[Math.max(0, n - 1)];
     if (n < NC) { say(ANS[cur[0]], 20, 30, 250, { seed: 550, c: GREEN, lw: 2, s: 13 }); say(ANS[cur[1]], 20, 64, 250, { seed: 560, s: 13 }); }
     teacher(345, 74, -1, 1.5, t); txt(`pair ${n} of ${NC}`, 340, 112, { s: 14, c: C.soft });
     table(150, { head: [['answer', 22, 'left'], ['in pairs', 318], ['won', 378]], cols: [{ v: seen, x: 318 }, { v: win, x: 378 }], hi: i => i === SPAM && t > 9, hiCol: C.pink });
     card([{ s: 'green box: the answer Mo picked', c: GREEN }, 'in pairs: times shown,  won: times picked'], t - 1, { y: 314, gap: 1, fs: 15 }); } },
 { h: 'A reward model', p: ['Now we build a **reward model**: a program that scores any answer, trained to agree with the comparisons. Real ones are large networks reading the whole text. Ours counts three things and gives each a weight: every word, every up or right, every down or left.',
   'Score = the sum. Training picks the weights that make the people’s choices most likely under the same curve.'],
   shot: [4, 11],
   draw(t) { const k = Math.min(WHIST.length - 1, Math.floor(clamp((t - 3) / 6) * (WHIST.length - 1))), w = WHIST[k];
     const lab = ['per word', 'per up or right', 'per down or left'];
     lab.forEach((s, j) => { const y = 50 + j * 46; txt(s, 30, y, { s: 15, al: 'left' }); rline(170, y, 350, y, 1100 + j, { c: C.line, w: 1, once: true }); rline(260, y - 6, 260, y + 6, 1110 + j, { c: C.soft, w: 1, once: true });
       const x = 260 + clamp(w[j], -1, 1) * 90; rcircle(x, y, 7, 1120 + j, { fill: w[j] >= 0 ? C.yellow : C.pink, j: .4 }); txt(f3(w[j]), x, y - 17, { s: 13, c: w[j] >= 0 ? C.goldInk : C.roseInk }); });
     txt('−1', 170, 192, { s: 12, c: C.soft }); txt('0', 260, 192, { s: 12, c: C.soft }); txt('+1', 350, 192, { s: 12, c: C.soft });
     caption(k < WHIST.length - 1 ? 'training: weights moving' : 'trained weights');
     card(['each comparison nudges the weights', 'so the chosen answer’s score rises', { s: 'above the rejected one’s', c: C.soft }], t - 1, { y: 230, gap: .8 }); } },
 { h: 'Scores for every answer', p: [`Trained weights: ${f3(WFIT[0])} per word, ${f3(WFIT[1])} per up or right, ${f3(WFIT[2])} per down or left. The good answers score ${f2(RMS[0])} and ${f2(RMS[1])}, above the rest. It matches ${pc(AGREE / NC)} of choices: low, but people’s own hidden taste matches only ${pc(CMP.filter(([x, y]) => TASTE[x] > TASTE[y]).length / NC)}, since in ${CMP.filter(([x, y]) => TASTE[x] === TASTE[y]).length} pairs both tasted the same.`,
   `The repeated answer scores highest, ${f2(RMS[SPAM])}: it says up and right four times, and the scorer just counts.`],
   shot: [4, 11],
   draw(t) { scene(t, { qa: .4 });
     table(150, { head: [['answer', 22, 'left'], ['RM score', 318], ['taste', 378]], cols: [{ v: RMS, x: 318, fmt: f2, tc: (i, v) => i === SPAM && t > 7 ? C.roseInk : C.goldInk }, { v: TASTE, x: 378, fmt: f1, tc: () => C.soft }],
       rowA: (i, k) => clamp((t - .5 - k * .3) * 3), hi: i => i === SPAM && t > 7, hiCol: C.pink });
     caption('RM = reward model; taste = the hidden truth, for comparison', clamp(t - 3));
     card([`first row: 5 words × ${f3(WFIT[0])} + 2 × ${f3(WFIT[1])} = ${f2(RMS[0])}`, { s: `last row: 8 words × ${f3(WFIT[0])} + 4 × ${f3(WFIT[1])} = ${f2(RMS[SPAM])}`, c: t > 7 ? C.roseInk : C.ink }], t - 4, { y: 316, gap: 1.5, fs: 15 }); } },
 { h: 'Practice with PPO', p: ['Now the reward model is the judge, scoring thousands of answers at no cost. Bot improves with **PPO**, from the PPO reel.',
   `Each update Bot answers ${NB} times. An answer’s advantage is its reward minus the batch average. Bot takes ${NPASS} passes over the batch, ratio clipped to 0.8 to 1.2. Answers are one step long, so the batch average serves as the critic.`],
   shot: [4, 11],
   draw(t) { bot(200, 70, 2, t); const R = rng(77);
     for (let i = 0; i < 32; i++) { const a = samp(REF, R), u = clamp((t - 1 - i * .12) * 2); if (u <= 0) continue; const x = 30 + (i % 8) * 44, y = 120 + Math.floor(i / 8) * 26;
       alpha(u, () => { box(x, y, 38, 20, C.card); rrect(x, y, 38, 20, 1200 + i, { c: C.line, w: 1 }); txt(f2(RMS[a]), x + 19, y + 11, { s: 12, c: RMS[a] > 1 ? C.goldInk : C.soft }); }); }
     txt('32 answers, each with its reward-model score', 200, 236, { s: 13, c: C.soft, a: clamp(t - 3) });
     { const R2 = rng(77); let sm = 0; for (let i = 0; i < 32; i++) sm += RMS[samp(REF, R2)]; txt(`batch average ${f2(sm / 32)}: above it goes up, below it goes down`, 200, 254, { s: 13, c: C.goldInk, a: clamp(t - 4) }); }
     card(['advantage = reward − batch average', `${NPASS} passes, ratio clipped to 0.8 to 1.2`, { s: 'raise chances of above-average answers', c: C.goldInk }], t - 5, { y: 280, gap: 1 }); } },
 { h: 'Two ways to keep score', p: ['Two charts will track Bot. Top: the average reward-model score of Bot’s answers, what PPO is pushing up.',
   'Bottom: **people’s verdict**, the chance a person prefers the new Bot’s answer over the original Bot’s. 0.5 means no better. We can compute it exactly from the hidden taste. Bot cannot. Both start at the dot.'],
   shot: [3, 9],
   draw(t) { twoCharts(t, [{ H: OFF, col: C.rose }], { n: 0, a1: clamp(t - .5), a2: clamp(t - 3) });
     txt(`start ${f2(OFF[0].rm)}`, 70, 136 - OFF[0].rm / 5 * 86 - 12, { s: 13, c: C.roseInk, al: 'left', a: clamp(t - 1.5) });
     card(['top: what the reward model thinks', { s: 'bottom: what people think (the truth here)', c: GREEN }], t - 5, { y: 316, gap: 1.2, fs: 16 }); } },
 { h: 'Leash off', p: [`Run PPO with only the reward-model score as reward. At first both lines rise: Bot learns the good answers, and by update ${PEAK} people’s verdict is ${f2(OFF[PEAK].win)}. Rarely, Bot gives its repeated answer, which scores higher still, and PPO piles on.`,
   `By update ${NU} the score is ${f2(FO.rm)}, the repeated answer comes ${(FO.P[SPAM] * 100).toFixed(1)} times in 100, and people’s verdict is ${f2(FO.win)}: worse than the original.`],
   shot: [5, 12],
   draw(t) { const n = Math.round(clamp((t - .5) / 6) * NU); twoCharts(t, [{ H: OFF, col: C.rose }], { n });
     if (t > 7) { const X = 60 + PEAK / NU * 270, Y = 274 - OFF[PEAK].win * 84; dot(X, Y, 4, C.goldInk, clamp(t - 7)); txt(`update ${PEAK}: ${f2(OFF[PEAK].win)}`, X + 4, Y + 17, { s: 13, c: C.goldInk, al: 'left', a: clamp(t - 7) }); }
     if (t > 8) card([{ s: `Bot: “${ANS[SPAM]}”`, c: C.roseInk }, `${(FO.P[SPAM] * 100).toFixed(1)} times in 100`], t - 8, { y: 312, gap: .8, fs: 15 }); } },
 { h: 'Reward hacking', p: [`The reward model learned that up and right are good, and that is true in Bot’s usual answers. Repeating them is not, but in its ${SPAMC.length} comparisons there was too little to learn from, and three weights cannot say repeating is bad.`,
   'Maximising a learned score finds exactly such gaps. This is **reward hacking**: a high score, a worse answer.'],
   shot: [4, 11],
   draw(t) { scene(t, { ans: SPAM, ax: 30, aw: 340, aa: clamp(t - .5), atc: C.roseInk });
     table(170, { rows: [1, SPAM], head: [['answer', 22, 'left'], ['RM score', 318], ['taste', 378]], cols: [{ v: RMS, x: 318, fmt: f2, tc: () => C.goldInk }, { v: TASTE, x: 378, fmt: f1, tc: () => C.soft }], rowA: (i, k) => clamp((t - 2 - k) * 3) });
     card([`scorer: 4 × up or right = 4 × ${f3(WFIT[1])} = ${f2(4 * WFIT[1])}`, { s: 'people: it repeats, the route is wrong', c: C.roseInk }], t - 5, { y: 250, gap: 1.2 }); } },
 { h: 'A leash', p: ['The fix, a **leash** to the original Bot: each reward loses β (beta) × log of new over old chance. Averaged over new answers, that is the **KL divergence**: how far Bot has moved.',
   `Bot settles where leashed rewards tie: original chance × e to the score over β, rescaled to sum to 1. At β 1, the repeated answer grows only from 1 in 100 to ${f2(FN.P[SPAM])}.`],
   shot: [4, 11],
   draw(t) { bot(110, 110, 2, t); alpha(.45, () => bot(290, 110, 2, t)); txt('tuned Bot', 110, 160, { s: 14, c: C.soft }); txt('original Bot', 290, 160, { s: 14, c: C.soft });
     const stretch = 1 + .5 * Math.sin(t * 1.4) * clamp(t - 1); ctx.save(); ctx.strokeStyle = C.goldInk; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(126, 104);
     for (let k = 1; k <= 12; k++) ctx.lineTo(lerp(126, 274, k / 12), 104 + Math.sin(k * 1.6) * 6 / stretch); ctx.stroke(); ctx.restore();
     card(['reward = RM score − β × log(new ÷ original)', { s: 'drift far from the original: pay more', c: C.goldInk }, 'Bot settles at: original × e^(score ÷ β)', { s: 'then rescaled to add up to 1', c: C.soft }], t - 2.5, { y: 200, gap: 1.4, fs: 16 }); } },
 { h: 'Leash on', p: [`Same run with β = 1. The score rises less, to ${f2(FN.rm)}. People’s verdict rises to ${f2(FN.win)}: people now prefer the tuned Bot’s answer about ${Math.round(FN.win * 100)} times in 100.`,
   `The good answers climb to ${f2(FN.P[0])} and ${f2(FN.P[1])}. The repeated one still grows, from 0.01 to ${f2(FN.P[SPAM])}: the leash holds it back, it does not erase the blind spot.`],
   shot: [5, 12],
   draw(t) { const n = Math.round(clamp((t - .5) / 6) * NU); twoCharts(t, [{ H: OFF, col: '#e9b8b8' }, { H: ON, col: BLUE }], { n });
     card([{ s: 'pale rose: leash off', c: C.roseInk }, { s: 'blue: leash on, β = 1', c: BLUE }], t - 1, { y: 310, gap: .6, fs: 15 }); } },
 { h: 'Your turn', ctl: { label: 'β', values: BETAS, def: 3, fmt: v => String(v) },
   p: ['Drag **β**, the strength of the leash. Each setting reruns PPO for 100 updates. The bars show Bot’s final chances, the card the reward-model score and people’s verdict.',
   `Too loose, and Bot hacks the scorer. Too tight, and Bot barely changes. Of these settings, β = 1 does best with people, ${f2(FN.win)}.`],
   shot: [4, 10],
   draw(t, st) { const b = BETAS[st.ci ?? 3], H = ppo(b), F = H[NU];
     table(36, { head: [['Bot’s final chances', 22, 'left'], ['chance', 378]], cols: withCh(F.P), hi: i => i === SPAM && F.P[SPAM] > .3, hiCol: C.pink });
     REF.forEach((r, i) => { const x = 290 + r * 64, y = 36 + i * ROWH + 4; rline(x, y - 7, x, y + 7, 1250 + i, { c: C.ink, w: 1.2, once: true, j: .2 }); });
     txt('black tick: the original Bot’s chance', 378, 190, { s: 12, c: C.soft, al: 'right' });
     card([`β = ${b}${b === 0 ? ', no leash' : ''}`, `reward-model score ${f2(F.rm)}`, { s: `people’s verdict ${f2(F.win)}`, c: F.win > .5 ? GREEN : C.roseInk }, { s: `KL from the original ${f2(F.kl)}`, c: C.soft }], 1, { y: 200, gap: 0 }); } },
 { h: 'Not a cure', p: [`The best Bot here, always the good answer with the cat warning, would earn ${f2(BEST)}. The leash got ${f2(FN.win)}: safer, not perfect. A reward model is only as good as its comparisons.`,
   'So teams collect fresh comparisons on the tuned model’s answers and retrain the reward model. Unable to see hidden taste, real teams pick β, and when to stop, from people’s ratings.'],
   shot: [4, 11],
   draw(t) { const x0 = 60, x1 = 340, y = 120, X = v => x0 + v * (x1 - x0);
     rline(x0, y, x1, y, 1300, { c: C.line }); [0, .5, 1].forEach(v => { rline(X(v), y - 5, X(v), y + 5, 1301 + v * 4, { c: C.line, once: true }); txt(String(v), X(v), y + 16, { s: 12, c: C.soft }); });
     txt('people’s verdict', x0, y - 50, { s: 13, c: C.soft, al: 'left' });
     [[FO.win, C.roseInk, 'leash off'], [.5, C.soft, 'original'], [FN.win, BLUE, 'leash on'], [BEST, GREEN, 'best possible']].forEach(([v, col, s], k) => { const a = clamp(t - .8 - k * .9);
       dot(X(v), y, 5, col, a); txt(s, X(v), y + (k % 2 ? 34 : -22), { s: 13, c: col, a }); txt(f2(v), X(v), y + (k % 2 ? 50 : -38), { s: 12, c: col, a }); });
     card(['the scorer is learned, so it has blind spots', 'the leash limits how far Bot chases them', { s: 'fresh comparisons keep the scorer honest', c: C.goldInk }], t - 5, { y: 230, gap: 1.1 }); } },
 { h: 'Where it is used', p: ['Christiano and colleagues, in 2017, taught simulated robots and Atari players from people comparing short video clips; a backflip took about 900 comparisons.',
   'In 2022, **InstructGPT** (Ouyang and colleagues) tuned GPT-3 with comparisons, a reward model, and PPO with a KL penalty. People preferred a 1.3 billion parameter InstructGPT to the 175 billion parameter GPT-3. This recipe, **RLHF**, shaped later chat assistants.'],
   shot: [3, 10],
   draw(t) { alpha(clamp(t - .5), () => { rrect(40, 40, 140, 90, 1401, { fill: '#e8e0d4' }); rrect(52, 52, 52, 66, 1402, { fill: C.card }); rrect(116, 52, 52, 66, 1403, { fill: C.card });
       rcircle(78, 85, 12, 1404, { j: 1.5 }); rcircle(142, 85, 12, 1405, { j: 1.5 }); txt('A or B?', 110, 146, { s: 15 }); txt('2017: clips', 110, 166, { s: 13, c: C.soft }); });
     alpha(clamp(t - 3), () => { say('Q: …', 230, 40, 130, { seed: 1410 }); say('A ✓', 230, 74, 60, { seed: 1411, tc: GREEN }); say('B', 300, 74, 60, { seed: 1412 }); txt('2022: InstructGPT', 295, 146, { s: 15 }); txt('comparisons → RM → PPO + KL', 295, 166, { s: 13, c: C.soft }); });
     card(['2017: preferences between clips (Christiano et al.)', '2022: InstructGPT, RLHF on a language model', { s: 'later: DPO learns from comparisons directly', c: C.soft }], t - 5, { y: 230, gap: 1.1, fs: 16 }); } },
 { h: 'Recap', p: ['When a good answer is hard to write down, ask people to **compare** two answers. Fit a **reward model** so that bigger score gaps mean more reliable preferences.',
   'Then tune the model with PPO against that score, on a **KL leash** to the original, because a learned scorer has blind spots that an unleashed optimiser will find.'],
   shot: [3, 9],
   draw(t) { scene(t, { ans: 1, aa: 1, atc: GREEN });
     card(['1. people compare pairs of answers', '2. reward model: chance A wins grows with gap', '3. PPO raises the reward-model score', { s: '4. KL leash: stay near the original', c: C.goldInk }], t - .3, { y: 170, gap: 1 }); } },
];

RLStory.register('rlhf', { title: 'RLHF', track: 'Learning from people', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { REF, TASTE, FEAT, W: WFIT, RMS, AGREE, NC, SPAMC: SPAMC.length, SPAMW, CNT, off: { rm0: OFF[0].rm, rm: FO.rm, win: FO.win, P: FO.P, kl: FO.kl }, on: { rm: FN.rm, win: FN.win, P: FN.P, kl: FN.kl },
    betas: BETAS.map(b => { const F = ppo(b)[NU]; return { b, rm: F.rm, win: F.win, kl: F.kl, spam: F.P[SPAM] }; }), BEST, LEASH_EX, sig3: sig(3) } });
})();
