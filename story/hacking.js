// Reward hacking as a story: we ask Pip to tidy up crumbs on its way to the cheese, reward it for touching crumbs,
// and Q-learning finds that bumping a wall while standing on one crumb pays best. All numbers come from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, key, maze, trail, pip } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const CR = [[0, 1], [2, 3], [4, 2]], GAM = 0.9, ALPHA = 0.2, EPS = 0.2, CAP = 50, NW = 100, SEED = 1, PSEED = 1, PNW = 300;
const crIdx = (c, r) => CR.findIndex(([a, b]) => a === c && b === r);
// Q-learning as in the qlearning reel. mode 'touch': +B for every step that ends on a crumb (crumbs stay).
// mode 'once': a crumb pays +B once, then is gone; Pip's state = cell + which crumbs are left (8 combinations).
function run(mode, B, seed, nw) {
  const R = rng(seed), Q = new Float64Array(20 * 8 * 4), E = [], sk = (c, r, left) => (mode === 'once' ? left : 0) * 20 + key(c, r);
  for (let e = 0; e < nw; e++) {
    let [c, r] = G.start, left = 7, pts = 0, end = 0; const got = new Set(), tr = [];
    for (let k = 0; k < CAP; k++) {
      const s = sk(c, r, left); let m;
      if (R() < EPS) m = Math.floor(R() * 4); else { let b = -1e9, ms = []; for (let a = 0; a < 4; a++) { const q = Q[s * 4 + a]; if (q > b + 1e-9) { b = q; ms = [a]; } else if (q > b - 1e-9) ms.push(a); } m = ms[Math.floor(R() * ms.length)]; }
      const [nc, nr] = G.step(c, r, m), t = G.term(nc, nr), done = t !== 0, ci = crIdx(nc, nr); let rew = t, nl = left;
      if (ci >= 0 && (mode === 'touch' || left & (1 << ci))) { rew += B; got.add(ci); if (mode === 'once') nl = left & ~(1 << ci); }
      const s2 = sk(nc, nr, nl), nx = done ? 0 : Math.max(Q[s2 * 4], Q[s2 * 4 + 1], Q[s2 * 4 + 2], Q[s2 * 4 + 3]);
      Q[s * 4 + m] += ALPHA * (rew + GAM * nx - Q[s * 4 + m]); tr.push({ c, r, m, nc, nr });
      pts += rew; c = nc; r = nr; left = nl; if (done) { end = t; break; }
    }
    E.push({ pts, end, crumbs: got.size, n: tr.length, tr });
  }
  return { E, Q, sk, mode };
}
// Pip follows its biggest numbers from the start, no random moves, 16 steps at most
function follow(o) {
  let [c, r] = G.start, left = 7; const P = [[c, r]];
  for (let k = 0; k < 16 && !G.term(c, r); k++) { const s = o.sk(c, r, left); let b = -1e9, bm = 0; for (let a = 0; a < 4; a++) if (o.Q[s * 4 + a] > b + 1e-12) { b = o.Q[s * 4 + a]; bm = a; }
    const [nc, nr] = G.step(c, r, bm), ci = crIdx(nc, nr); if (ci >= 0 && o.mode === 'once') left &= ~(1 << ci); P.push([nc, nr, bm]); c = nc; r = nr; }
  return { P, ok: G.term(c, r) > 0 };
}
// exact values for the 'touch' reward (value iteration; knows the rules, used only to check claims)
function VI(B) {
  const V = new Float64Array(20), q = (c, r, m) => { const [nc, nr] = G.step(c, r, m), t = G.term(nc, nr); return t + (crIdx(nc, nr) >= 0 ? B : 0) + (t ? 0 : GAM * V[key(nc, nr)]); };
  for (let it = 0; it < 500; it++) for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) V[key(c, r)] = Math.max(q(c, r, 0), q(c, r, 1), q(c, r, 2), q(c, r, 3));
  return { V, q };
}
const RUNS = {}, runT = B => RUNS[B] ||= run('touch', B, SEED, NW);
const H = runT(0.2), HF = follow(H), HE = H.E;
const FARM = HF.P[HF.P.length - 1], BUMPM = FARM[2], FSTEPS = HF.P.findIndex((p, i) => i > 0 && p[0] === HF.P[i - 1][0] && p[1] === HF.P[i - 1][1]) - 1;                         // the cell Pip ends on and the move it repeats
const V2 = VI(0.2), FARMV = V2.V[key(0, 1)], STARTV = V2.V[key(0, 3)];
// value of a fixed path (cells) under a reward: 'touch' or 'once' crumbs (+0.2), cheese +1, discount 0.9 per step
function pathValue(P, mode) { let v = 0, left = 7; for (let i = 1; i < P.length; i++) { const [c, r] = P[i], ci = crIdx(c, r); let rw = G.term(c, r); if (ci >= 0 && (mode === 'touch' || left & (1 << ci))) { rw += .2; left &= ~(1 << ci); } v += GAM ** (i - 1) * rw; } return v; }
const BOTTOM = [[0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [4, 2], [4, 1], [4, 0]], LEFTR = [[0, 3], [0, 2], [0, 1], [0, 0], [1, 0], [2, 0], [3, 0], [4, 0]];
const ALL3 = [[0, 3], [0, 2], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [4, 2], [4, 1], [4, 0]];
// best plan that heads straight for the cheese (no lingering): the bottom route over two crumbs
const HONEST = pathValue(BOTTOM, 'touch'), HONESTL = pathValue(LEFTR, 'touch');
const EATV = V2.q(3, 0, 1), TURNV = V2.q(3, 0, 3), BELOW_EAT = V2.q(4, 1, 0), BELOW_CR = V2.q(4, 1, 2);   // left of / below the cheese: eat vs go to a crumb
// exact values under the patched reward (crumbs pay once; state = cell + crumbs left)
function VIP() { const V = new Float64Array(160), q = (c, r, l, m) => { const [nc, nr] = G.step(c, r, m), t = G.term(nc, nr), ci = crIdx(nc, nr); let rw = t, nl = l; if (ci >= 0 && l & (1 << ci)) { rw += .2; nl = l & ~(1 << ci); } return rw + (t ? 0 : GAM * V[nl * 20 + key(nc, nr)]); };
  for (let it = 0; it < 300; it++) for (let l = 0; l < 8; l++) for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) V[l * 20 + key(c, r)] = Math.max(q(c, r, l, 0), q(c, r, l, 1), q(c, r, l, 2), q(c, r, l, 3));
  let [c, r] = G.start, l = 7; const P = [[c, r]]; for (let k = 0; k < 20 && !G.term(c, r); k++) { let b = -1e9, bm = 0; for (let m = 0; m < 4; m++) { const v = q(c, r, l, m); if (v > b + 1e-12) { b = v; bm = m; } } const [nc, nr] = G.step(c, r, bm), ci = crIdx(nc, nr); if (ci >= 0) l &= ~(1 << ci); c = nc; r = nr; P.push([c, r]); }
  return { v: V[7 * 20 + key(0, 3)], P }; }
const VP = VIP(), PBOT = pathValue(BOTTOM, 'once'), PALL3 = pathValue(ALL3, 'once');
const firstN = n => HE.slice(0, n).filter(e => e.end > 0).length, lastN = n => HE.slice(NW - n).filter(e => e.end > 0).length;
const LASTPTS = HE.slice(NW - 50).reduce((a, e) => a + e.pts, 0) / 50, LASTCHEESE = lastN(50), FIRSTCHEESE = firstN(10);
const LASTHONEST = Math.max(...HE.map((e, i) => (e.end > 0 ? i : -1))) + 1;   // last walk (1-based) that reached the cheese
const P3 = run('once', 0.2, PSEED, PNW), PF = follow(P3), PL = P3.E.slice(PNW - 100);
const PCHEESE = PL.filter(e => e.end > 0).length, PCRUMBS = PL.reduce((a, e) => a + e.crumbs, 0) / 100;
const BV = [0, .02, .05, .1, .2];
const SUMM = {}; const summ = B => SUMM[B] ||= (() => { const o = runT(B), last = o.E.slice(NW - 50), F = follow(o), V = VI(B);
  return { F, pts: last.reduce((a, e) => a + e.pts, 0) / 50, cheese: last.filter(e => e.end > 0).length, farm: !F.ok, vStart: V.V[key(0, 3)], bestFarm: farmBest(B) }; })();
// is bumping on a crumb really the best plan from the start? (follow the exact values)
function farmBest(B) { const V = VI(B); let [c, r] = G.start; for (let k = 0; k < 20 && !G.term(c, r); k++) { let b = -1e9, bm = 0; for (let m = 0; m < 4; m++) { const v = V.q(c, r, m); if (v > b + 1e-12) { b = v; bm = m; } } [c, r] = G.step(c, r, bm); } return !G.term(c, r); }
let THR = null; const thr = () => { if (THR != null) return THR; let lo = 0, hi = .2; for (let i = 0; i < 14; i++) { const mid = (lo + hi) / 2; if (farmBest(mid)) hi = mid; else lo = mid; } return (THR = hi); };
const f2 = v => L.num(v, 2), f1 = v => (Math.round(v * 10) / 10).toFixed(1);

// ───── drawing
function crumb(x, y, s = 1, a = 1) { if (a <= 0) return; ctx.save(); ctx.globalAlpha = a; ctx.translate(x, y); ctx.scale(s, s);
  ctx.beginPath(); ctx.moveTo(-8, 5); ctx.lineTo(8, 5); ctx.lineTo(8, -4); ctx.closePath(); ctx.fillStyle = '#f7d36b'; ctx.fill(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.1; ctx.stroke();
  ctx.fillStyle = '#e2b54a'; ctx.beginPath(); ctx.arc(4, 2, 1.3, 0, 7); ctx.fill(); ctx.restore(); }
function crumbs(a = 1, gone) { CR.forEach(([c, r], i) => { if (gone && gone.has(i)) return; crumb(cx(c) + 14, cy(r) + 16, 1, a); }); }
function dim(a = .62) { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, 285); ctx.restore(); }
// walk along a list of cells [c, r, move]; a repeated cell is a bump in direction 'move'
function walker(P, t, dur) {
  t = Math.max(0, t); const k = Math.min(P.length - 1, Math.floor(t / dur)), u = (t % dur) / dur;
  if (k >= P.length - 1) { const l = P[P.length - 1]; return { x: cx(l[0]), y: cy(l[1]), k, done: true }; }
  const a = P[k], b = P[k + 1];
  if (a[0] === b[0] && a[1] === b[1]) { const [dx, dy] = G.MOVES[b[2]], s = Math.sin(clamp(u * 1.6) * Math.PI) * 13; return { x: cx(a[0]) + dx * s, y: cy(a[1]) + dy * s, k, face: dx || 0, bump: true }; }
  const e = ease(u * 1.3); return { x: lerp(cx(a[0]), cx(b[0]), e), y: lerp(cy(a[1]), cy(b[1]), e), k, face: b[0] - a[0] };
}
// bump forever at the farm cell
function bumpLoop(t, c, r, m, per = .6) { const [dx, dy] = G.MOVES[m], s = Math.sin(clamp(((t % per) / per) * 1.6) * Math.PI) * 13; return { x: cx(c) + dx * s, y: cy(r) + dy * s, face: dx || 0 }; }
// chart: one bar per walk = points that walk, gold = reached the cheese
function pointsChart(n, t) {
  const x0 = 50, x1 = 366, y0 = 388, y1 = 308, bw = (x1 - x0) / NW, Y = v => y0 - clamp(v / 10) * (y0 - y1);
  rline(x0 - 3, y0, x1 + 3, y0, 950, { c: C.line, w: 1 }); rline(x0 - 3, y0, x0 - 3, y1 - 4, 951, { c: C.line, w: 1 });
  [0, 5, 10].forEach(v => txt(String(v), x0 - 8, Y(v), { s: 12, c: C.soft, al: 'right' }));
  txt('points in each walk (plain sum)', x0, y1 - 12, { s: 13, c: C.soft, al: 'left' });
  box(218, y1 - 16, 10, 8, C.gold); txt('cheese', 232, y1 - 12, { s: 13, al: 'left', c: C.soft });
  box(290, y1 - 16, 10, 8, C.grey); txt('no cheese', 304, y1 - 12, { s: 13, al: 'left', c: C.soft });
  for (let i = 0; i < Math.min(n, NW); i++) { const e = HE[i], h = y0 - Y(Math.max(.15, e.pts)); box(x0 + i * bw + .3, y0 - h, Math.max(1, bw - .8), h, e.end > 0 ? C.gold : C.grey); }
  txt('walk 1', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NW}`, x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
}
function flame(x, y, t) { const f = 1 + Math.sin(t * 17) * .15; ctx.save(); ctx.translate(x, y); ctx.scale(f, f); ctx.beginPath(); ctx.moveTo(-5, 4); ctx.quadraticCurveTo(-6, -4, 0, -12); ctx.quadraticCurveTo(6, -4, 5, 4); ctx.closePath(); ctx.fillStyle = C.orange; ctx.fill(); ctx.beginPath(); ctx.moveTo(-2, 4); ctx.quadraticCurveTo(-2, -2, 0, -6); ctx.quadraticCurveTo(2, -2, 2, 4); ctx.closePath(); ctx.fillStyle = C.yellow; ctx.fill(); ctx.restore(); }
function boat(x, y, a) { ctx.save(); ctx.translate(x, y); ctx.rotate(a); ctx.beginPath(); ctx.moveTo(-12, -5); ctx.lineTo(10, -5); ctx.lineTo(15, 0); ctx.lineTo(10, 5); ctx.lineTo(-12, 5); ctx.closePath(); ctx.fillStyle = '#e9e1d6'; ctx.fill(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3; ctx.stroke(); ctx.fillStyle = C.rose; ctx.fillRect(-4, -3, 7, 6); ctx.restore(); }
const MV = ['up', 'right', 'down', 'left'];

const STEPS = [
 { h: 'A chore for Pip', p: ['Back in Pip’s maze: cheese +1, cat −1, both end the walk, γ (gamma) = 0.9. Today there are also **three crumbs** lying around.', 'We want Pip to tidy up: pick up the crumbs, then go to the cheese. Pip, as always, knows nothing of what we want. It only feels the rewards we hand out.'],
   shot: [3, 9],
   say: [["Welcome back to Pip's maze. Today there's a chore."], [["Three crumbs are lying around.", [40, 16, 320, 256]], "We want Pip to tidy up, then go to the cheese."], ["But Pip knows nothing of what we want. It only feels the rewards we hand out."]],
   draw(t) { maze({ t, labels: clamp(t - 1) }); CR.forEach(([c, r], i) => crumb(cx(c) + 14, cy(r) + 16, 1, clamp(t - 1.8 - i * .4))); pip(cx(0), cy(3), 1, t);
     card(['what we want: collect the crumbs,', 'then eat the cheese', { s: 'what Pip knows: only the rewards it gets', c: C.soft }], t - 3.5, { gap: 1.2 }); } },
 { h: 'Writing the reward', p: ['We have a crumb sensor, so we write: **+0.2 for every step that ends on a crumb**. Cheese and cat stay as before, and a walk stops after 50 steps.', 'The sensor only checks where Pip stands, so a crumb never disappears. Pip learns with Q-learning, as in the Q-learning reel: one move in five random.'],
   shot: [3, 10],
   say: [["So we write a reward."], [["Every step that ends on a crumb pays a little bonus.", [30, 292, 340, 87]], "Cheese and cat stay the same."], ["Our sensor only checks where Pip stands. So a crumb never disappears."]],
   draw(t) { maze({ t }); crumbs(); pip(cx(0), cy(3), 1, t); CR.forEach(([c, r], i) => txt('+0.2', cx(c) - 8, cy(r) - 14, { s: 15, c: C.goldInk, a: clamp(t - 1 - i * .4) }));
     card(['+0.2 for each step that ends on a crumb', '+1 cheese, −1 cat: walk over', { s: 'walks stop after 50 steps; crumbs stay', c: C.soft }], t - 3, { gap: 1.2 }); } },
 { h: 'What Pip learned', p: [`After ${NW} walks, Pip follows its biggest numbers from the start, no random moves. It walks ${FSTEPS} steps to the crumb on the left, and then **bumps into the rock** beside it, again and again.`, 'Each bump is a step that ends on a crumb: +0.2, every single step. Pip never goes near the cheese.'],
   shot: [5, 12],
   say: [["After training, Pip follows its biggest numbers from the start."], [["It walks to the crumb on the left.", [40, 16, 320, 256]], ["And then it bumps into the rock beside it, again, and again.", [40, 16, 320, 256]]], ["Each bump ends on the crumb. Bonus, every single step. Pip never goes near the cheese."]],
   draw(t) { maze({ t }); crumbs(); const P = HF.P.slice(0, HF.P.findIndex((p, i) => i > 0 && p[0] === HF.P[i - 1][0] && p[1] === HF.P[i - 1][1]));
     const tw = (P.length - 1) * .6, m = t < tw + .5 ? walker(P, t - .5, .6) : bumpLoop(t - tw - .5, FARM[0], FARM[1], BUMPM);
     trail(P.slice(0, (m.k ?? P.length) + 1), .6); pip(m.x, m.y, m.face, t);
     if (t > tw + 1) { const n = Math.floor((t - tw - .5) / .6); for (let i = 0; i < Math.min(4, n); i++) bubble('+0.2', cx(FARM[0]) + 4, cy(FARM[1]) - 18, t - tw - .5 - (n - i) * .6 - .2); }
     if (t > tw + 2) card([`walks to the crumb: ${P.length - 1} steps`, `then: bump ${MV[BUMPM]}, forever`, { s: '+0.2 on every step', c: C.goldInk }], t - tw - 2, { gap: 1.1 }); } },
 { h: 'Why bumping wins', p: [`Bump forever on the crumb: 0.2 every step. With the 0.9 discount, worth ${f2(FARMV)}. From the start: ${f2(STARTV)}. Going straight to the cheese along the bottom, over two crumbs: only ${f2(HONEST)}.`, `Even beside the cheese, the best move is to turn away. Left of it, eating is worth ${f2(EATV)}, returning to the left crumb ${f2(TURNV)}. Below it, the right crumb is next door: ${f2(BELOW_CR)}.`],
   shot: [4, 11],
   say: [["Why does bumping win? A bonus every step, forever, adds up."], [["Even with the discount, it's worth more than the cheese.", [30, 292, 340, 87]]], [["Even right beside the cheese, the best move is to turn away.", [40, 16, 320, 256]]]],
   draw(t) { maze({ t, hi: t > 4 ? [[3, 0, C.rose, .25 * clamp(t - 4)]] : null }); crumbs(); const m = bumpLoop(t, FARM[0], FARM[1], BUMPM); pip(m.x, m.y, m.face, t);
     if (t > 4) txt('left of the cheese', cx(3), cy(0) + 18, { s: 12, c: C.roseInk, a: clamp(t - 4) });
     card([`bump forever on the crumb: 0.2 + 0.9 × 0.2 + … = ${f2(FARMV)}`, `from the start, bumping plan: ${f2(STARTV)}`, `from the start, straight to the cheese: ${f2(HONEST)}`, { s: `left of the cheese: eat ${f2(EATV)}, go back ${f2(TURNV)}`, c: C.roseInk }], t - 1, { gap: 1.3, fs: 16 }); } },
 { h: 'Reward hacking', p: ['Pip broke no rule. It found the best way to collect **exactly what we rewarded**: steps that end on a crumb. What we meant, tidy up and then eat, was never in the numbers.', 'When an agent scores well by a route its designers did not intend, it is called **reward hacking**, or **specification gaming**.'],
   say: [["Pip broke no rule. It found the best way to collect exactly what we rewarded."], ["What we meant, tidy up and then eat, was never in the numbers."], [["Scoring well by a route the designers didn't intend is called reward hacking, or specification gaming.", [30, 292, 340, 87]]]],
   draw(t) { maze({ t }); crumbs(); const m = bumpLoop(t, FARM[0], FARM[1], BUMPM); pip(m.x, m.y, m.face, t);
     card([{ s: 'we rewarded: steps that end on a crumb', c: C.goldInk }, { s: 'we meant: tidy up, then eat the cheese', c: C.ink }, { s: 'the agent optimises the first', c: C.roseInk }], t - 1, { gap: 1.3 }); } },
 { h: 'High score from the first walk', p: [`Each bar is one training walk, as high as its points, the plain sum of rewards; gold if it reached cheese. Scores are high from walk 1: once the crumb paid +0.2, that move had the only number above 0, so Pip repeated it.`, `Only ${FIRSTCHEESE} of the first 10 reached cheese, the last on walk ${LASTHONEST}. Last 50: ${f1(LASTPTS)} points per walk, ${LASTCHEESE} cheeses.`],
   shot: [4, 12],
   say: [["Here's every training walk."], [["Each bar shows its points. Gold means it reached the cheese.", [40, 296, 340, 104]], ["The scores are high from the very first walk.", [40, 296, 340, 104]]], ["Watch the gold bars. Only a few early walks reach the cheese, then hardly any."]],
   draw(t) { maze({ t }); crumbs(); dim(.45); const m = bumpLoop(t, FARM[0], FARM[1], BUMPM); pip(m.x, m.y, m.face, t); pointsChart(Math.floor(clamp((t - 1) / 7) * NW), t); } },
 { h: 'Goodhart’s law', p: ['Economists know this as **Goodhart’s law**. In Marilyn Strathern’s words: when a measure becomes a target, it ceases to be a good measure.', 'Touching crumbs was a fine sign of tidying, as long as nobody optimised it. Once Pip optimised it, the sign and the goal came apart.'],
   say: [["Economists know this as Goodhart's law. When a measure becomes a target, it stops being a good measure."], [["Touching crumbs was a fine sign of tidying, as long as nobody pushed on it.", [40, 16, 320, 256]]], ["Once Pip optimised it, the sign and the goal came apart."]],
   draw(t) { maze({ t }); crumbs(); dim(.7); txt('“When a measure becomes a target,', W / 2, 110, { s: 21, a: clamp(t - .5) }); txt('it ceases to be a good measure.”', W / 2, 140, { s: 21, a: clamp(t - 1.2) });
     txt('Goodhart’s law, as phrased by Marilyn Strathern', W / 2, 178, { s: 14, c: C.soft, a: clamp(t - 2) });
     card(['measure: steps on a crumb', 'goal: a tidy maze and a fed Pip', { s: 'optimise the measure hard → they come apart', c: C.roseInk }], t - 3, { gap: 1.2 }); } },
 { h: 'Patch the reward', p: ['Fix the loophole: each crumb pays +0.2 **once**, then is gone. Pip’s state now includes which crumbs are left, since that decides whether a cell pays.', `A longer run, ${PNW} walks: eight times as many states. Pip takes ${PF.P.filter(([c, r]) => crIdx(c, r) >= 0).length} crumbs en route, skips the left one, eats in ${PF.P.length - 1} steps. That is best here: all three are worth ${f2(PALL3)}, against ${f2(PBOT)}; the detour delays cheese.`],
   shot: [5, 12],
   say: [["Let's patch the loophole. Each crumb pays once, then it's gone."], [["Now watch Pip. It picks up crumbs along the way, and eats.", [40, 16, 320, 256]]], [["But it skips the left one. The detour would delay the cheese too much.", [30, 292, 340, 87]]]],
   draw(t) { maze({ t }); const m = walker(PF.P, t - .5, .55), gone = new Set(); PF.P.slice(0, m.k + 1).forEach(([c, r]) => { const i = crIdx(c, r); if (i >= 0) gone.add(i); });
     crumbs(1, gone); trail(PF.P.slice(0, m.k + 1), .6); pip(m.x, m.y, m.face, t);
     if (m.done) bubble('+1', cx(4), cy(0) - 18, t - .5 - (PF.P.length - 1) * .55);
     card([`crumbs collected: ${gone.size} of 3`, m.done ? `cheese, after ${PF.P.length - 1} steps` : `step ${m.k}`, { s: `all three crumbs: ${f2(PALL3)};  this plan: ${f2(PBOT)}`, c: C.roseInk }], t - 1, { gap: .2 }); } },
 { h: 'Patches don’t scale', p: ['Patching worked because we watched Pip and saw the trick. In a big world we cannot list every loophole ahead, and a hard-searching agent finds the ones we missed.', 'A patch also changes the problem: Pip had to track which crumbs were left. Each fix is a new reward with its own gaps: ours still does not say “collect every crumb”, so Pip skipped one.'],
   say: [["Patching worked because we watched Pip, and saw the trick."], [["In a big world, we can't list every loophole ahead of time.", [40, 16, 320, 256]], "A hard-searching agent finds the ones we missed."], ["And each patch is a new reward, with its own gaps."]],
   draw(t) { maze({ t }); pip(cx(4), cy(0) + 4, 1, t); dim(.5);
     card(['patch: works for the gap you saw', 'gaps you did not see stay open', { s: 'a strong optimiser finds them', c: C.roseInk }], t - .8, { gap: 1.2 }); } },
 { h: 'A boat that never finished', p: ['This happens with real agents. In 2016, OpenAI trained an agent on the boat racing game CoastRunners, where points come from hitting targets along the course.', 'It found a lagoon where three targets kept coming back, and circled there, crashing and catching fire, never finishing the race. It still scored more than finishing would. Like Pip, it was rewarded for points, not for the race.'],
   shot: [4, 10],
   say: [["This happens with real agents, too."], [["In twenty sixteen, an agent learned the boat racing game CoastRunners.", [40, 30, 320, 230]], ["It found a lagoon where targets kept coming back, and circled there forever, crashing and catching fire.", [40, 30, 320, 230]]], ["It never finished the race, but it scored more than finishing would."]],
   draw(t) { const X = 200, Y = 140; rrect(40, 30, 320, 230, 800, { fill: '#dbe9f2', c: C.line });
     [[0, 1], [2.1, 1], [4.2, 1]].forEach(([a], i) => { const tx = X + Math.cos(a) * 62, ty = Y + Math.sin(a) * 48; L.rcircle(tx, ty, 8, 810 + i, { fill: C.yellow, j: .5 }); });
     const a = t * 1.5; boat(X + Math.cos(a) * 62, Y + Math.sin(a) * 48, a + Math.PI / 2);
     if (Math.sin(t * 3) > -.2) flame(X + Math.cos(a) * 62, Y + Math.sin(a) * 48 - 10, t);
     rline(330, 40, 330, 250, 820, { c: C.ink, w: 1, once: true }); txt('finish', 330, 270, { s: 13, c: C.soft });
     card(['2016, CoastRunners: points for hitting targets', 'agent: circles three targets that keep coming back', { s: 'more points than finishing; never finishes', c: C.roseInk }], t - 1.5, { gap: 1.2, fs: 16 }); } },
 { h: 'More from the list', p: ['Victoria Krakovna and colleagues at DeepMind keep a public list of such cases. A simulated robot arm, rewarded for the height of a block’s bottom face, **flipped the block over** instead of stacking it.', 'A simulated hand, judged by people watching through a camera, hovered between camera and ball so it **looked** like grabbing. A Tetris program paused the game forever, so it never lost.'],
   shot: [4, 10],
   say: [["Researchers at DeepMind keep a public list of cases like this."], [["A robot arm, rewarded for the height of a block's bottom face, just flipped it over.", [30, 220, 340, 110]], ["A simulated hand only looked like it was grabbing, to the camera.", [30, 220, 340, 110]], ["And a Tetris program paused the game forever, so it never lost.", [30, 220, 340, 110]]]],
   draw(t) { ctx.save(); ctx.translate(200, 128); ctx.scale(1.15, 1.15); ctx.translate(-200, -128);
     // block flip
     const a1 = clamp(t - .5); alpha(a1, () => { rrect(40, 140, 90, 6, 830, { fill: C.line }); const fl = ease(clamp((t - 1.2) / 1.2)); ctx.save(); ctx.translate(85, 140 - 20); ctx.rotate(fl * Math.PI); L.rrect(-16, -16, 32, 32, 831, { fill: C.pink }); ctx.restore(); if (fl > .99) box(69, 102, 32, 4, C.roseInk); txt('was the bottom', 85, 92, { s: 12, c: C.roseInk, a: clamp(t - 2.4) }); txt('flipped block', 85, 166, { s: 13, c: C.soft }); });
     // hand and camera
     const a2 = clamp(t - 3); alpha(a2, () => { L.rcircle(245, 128, 10, 840, { fill: C.orange }); rrect(158, 112, 22, 16, 841, { fill: C.grey }); txt('camera', 169, 100, { s: 12, c: C.soft }); txt('ball', 245, 146, { s: 12, c: C.soft }); const hx = lerp(232, 206, ease(clamp((t - 3.5) / 1.5))); L.rcircle(hx, 120, 9, 842, { fill: '#e9e1d6' }); txt('hand', hx, 100, { s: 12, c: C.soft }); txt('hand hides the ball', 210, 166, { s: 13, c: C.soft }); });
     // tetris pause
     const a3 = clamp(t - 5.5); alpha(a3, () => { rrect(300, 70, 60, 84, 850, { fill: C.card }); [[0, 3], [1, 3], [2, 3], [3, 3], [1, 2], [2, 2], [0, 2]].forEach(([c, r], i) => box(306 + c * 12, 100 + r * 12, 11, 11, C.blue)); txt('PAUSED', 330, 86, { s: 13, c: C.roseInk }); txt('paused forever', 330, 166, { s: 13, c: C.soft }); });
     ctx.restore();
     card(['2017  arm flips the block (Popov and colleagues)', '2017  hand fools the camera (Christiano and colleagues)', '2013  Tetris paused forever (Tom Murphy)'], t - 7, { gap: 1.1, fs: 16, y: 220 }); } },
 { h: 'Why it matters more for strong agents', p: ['Pip found its trick in 20 cells. A stronger agent searches more, finding gaps weaker ones miss. In 2022, Alexander Pan, Kush Bhatia and Jacob Steinhardt found more capable agents exploited badly specified rewards more.', 'Chatbots tuned with a **reward model**, a network predicting which answers people prefer, face the same pull: an answer can score well with it yet be no better for people.'],
   say: [["Pip found its trick in a tiny maze. Stronger agents search harder, and find gaps weaker ones miss."], [["Researchers found more capable agents exploit bad rewards more.", [30, 292, 340, 87]]], ["Chatbots tuned on a reward model feel the same pull. An answer can score well, yet be no better for people."]],
   draw(t) { maze({ t }); crumbs(); dim(.55); const m = bumpLoop(t, FARM[0], FARM[1], BUMPM); pip(m.x, m.y, m.face, t);
     card(['stronger search → finds more loopholes', '2022: more capable agents hacked more', { s: 'learned reward models can be gamed too', c: C.soft }], t - 1, { gap: 1.2 }); } },
 { h: 'What helps', p: ['No complete cure exists. Watch what the agent **does**, not just its score. Test it in situations it never trained on. Patch loopholes you find.', 'Don’t push too hard on a reward you do not fully trust. In chatbot tuning, a penalty usually keeps the new model’s answers close to the untuned model’s, so they cannot drift toward odd answers only the reward model likes.'],
   say: [["There's no complete cure."], [["Watch what the agent does, not just its score. Test it in new situations. Patch the loopholes you find.", [30, 292, 340, 87]]], ["And don't push too hard on a reward you don't fully trust."]],
   draw(t) { maze({ t }); dim(.5);
     card(['watch behaviour, not only the score', 'test in new situations', 'patch loopholes you find', { s: 'don’t over-optimise an imperfect reward', c: C.soft }], t - .5, { gap: 1.1 }); } },
 { h: 'Your turn', ctl: { label: 'crumb reward', values: BV, def: 4, fmt: v => '+' + v },
   get p() { return ['Choose what each crumb step pays. After 100 training walks, Pip follows its biggest numbers from the start. How small before Pip goes for the cheese?', `Bumping is best only above about ${f2(thr())}. Below that Pip still bumps: the crumb came long before the cheese, and rare random moves never carry it far enough to learn better. Not hacking: unfinished **exploration**, from the bandit reels.`]; },
   shot: [3, 10],
   say: [["Your turn. Choose what each crumb step pays."], [["Then watch where trained Pip goes.", [40, 16, 320, 256]]], ["How small does the bonus have to be before Pip heads for the cheese? Below a point, Pip still bumps, but that's unfinished exploration, not hacking."]],
   draw(t, st) { const B = BV[st.ci ?? 4], S = summ(B), t0 = t - (st.changed ?? 0); maze({ t }); if (B > 0) crumbs();
     const P = S.F.P; let m; const loopAt = P.findIndex((p, i) => i > 0 && p[0] === P[i - 1][0] && p[1] === P[i - 1][1]);
     if (!S.F.ok && loopAt > 0) { const tw = (loopAt - 1) * .5; m = t0 < tw + .5 ? walker(P.slice(0, loopAt), t0 - .5, .5) : bumpLoop(t0 - tw, P[loopAt][0], P[loopAt][1], P[loopAt][2]); }
     else m = walker(P, t0 - .5, .5);
     trail(P.slice(0, Math.min(P.length, (m.k ?? P.length) + 1)), .5); pip(m.x, m.y, m.face, t);
     card([`crumb reward: +${B} per step on a crumb`, S.farm ? 'Pip: stays on a crumb, bumping' : `Pip: goes to the cheese, ${P.length - 1} steps`, `walks reaching the cheese, last 50: ${S.cheese}`, { s: `best plan, if Pip knew the maze: ${S.bestFarm ? 'bump' : 'cheese'}`, c: C.soft }], 10, { lh: 21, fs: 16, y: 292 }); } },
 { h: 'Recap', p: ['An agent optimises **exactly what you reward**, not what you meant. Gaps between the two get found: Pip bumped a rock on a crumb, a boat circled for points.', 'When a measure becomes a target, it stops being a good measure. The stronger the optimiser, the more this matters: watch behaviour, test widely, and patch.'],
   say: [["So, an agent optimises exactly what you reward, not what you meant."], [["Pip bumped a rock on a crumb. A boat circled for points.", [30, 292, 340, 87]]], ["The stronger the optimiser, the more this matters. Watch behaviour, test widely, and patch."]],
   draw(t) { maze({ t }); crumbs(); const m = bumpLoop(t, FARM[0], FARM[1], BUMPM); pip(m.x, m.y, m.face, t);
     card(['agents optimise the reward, not the intent', 'loopholes get found (Pip, CoastRunners)', 'Goodhart: measure as target stops measuring', 'stronger agents find more gaps'], t - .3, { gap: 1 }); } },
];
RLStory.register('hacking', { title: 'Reward Hacking', track: 'Rewards', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { VP, PBOT, PALL3, HONESTL, BELOW_EAT, BELOW_CR, H, HF, FARM, BUMPM, FARMV, STARTV, HONEST, EATV, TURNV, FIRSTCHEESE, LASTHONEST, LASTPTS, LASTCHEESE, PF, PCHEESE, PCRUMBS, summ, BV, VI } });
})();
