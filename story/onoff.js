// On-policy vs off-policy as a story: whose behaviour do the numbers describe, and can Pip learn from someone else's diary?
// All numbers come from the seeded runs below (Wanderer diary seed 501, Pip's own run seed 54 as in the Q-learning reel).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, txt, arrow, card, bubble, caption, dot, box, num } = L;
let ctx;
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, key, maze, trail, pip } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GAM = .9, ALPHA = .2, qk = (c, r, m) => key(c, r) * 4 + m;
const sgn = x => x.replace('-', '−'), f2 = v => sgn(num(v));
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);
// ───── diaries: lists of walks, each step { c, r, m, rew, nc, nr, m2 (the next move the walker made) }
function diary(n, seed, pol) { const R = rng(seed), D = []; for (let w = 0; w < n; w++) { let [c, r] = G.start, m = pol(c, r, R); const walk = [];
  for (let k = 0; k < 3000; k++) { const [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr), done = rew !== 0, m2 = done ? null : pol(nc, nr, R); walk.push({ c, r, m, rew, nc, nr, m2 }); if (done) break; c = nc; r = nr; m = m2; } D.push(walk); } return D; }
// replay a diary in order, several passes; 'q' = Q-learning target (max), 's' = SARSA target (the walker's own next move)
function replay(D, algo, passes) { const Q = new Float64Array(80), snaps = [];
  for (let p = 0; p < passes; p++) { D.forEach(w => w.forEach(t => { const nx = t.rew ? 0 : algo === 'q' ? Math.max(Q[qk(t.nc, t.nr, 0)], Q[qk(t.nc, t.nr, 1)], Q[qk(t.nc, t.nr, 2)], Q[qk(t.nc, t.nr, 3)]) : Q[qk(t.nc, t.nr, t.m2)]; Q[qk(t.c, t.r, t.m)] += ALPHA * (t.rew + GAM * nx - Q[qk(t.c, t.r, t.m)]); })); snaps.push(Array.from(Q)); }
  return { Q: Array.from(Q), snaps }; }
// the Wanderer: every move random. The careful mouse: the arrows from the values reel, never distracted.
const ARR = new Array(20).fill(-1);
[[0, 3, 0], [0, 2, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [2, 1, 0], [1, 2, 3], [1, 3, 1], [2, 3, 1], [3, 3, 1], [4, 3, 0], [4, 2, 0], [4, 1, 0], [3, 2, 1]].forEach(([c, r, m]) => { ARR[key(c, r)] = m; });
const NWD = 100, PASSES = 10, WD = diary(NWD, 501, (c, r, R) => Math.floor(R() * 4)), XD = diary(NWD, 777, (c, r) => ARR[key(c, r)]);
const WSTEPS = WD.reduce((s, w) => s + w.length, 0), WCHEESE = WD.filter(w => w[w.length - 1].rew > 0).length, WCAT = NWD - WCHEESE;
const WLEN = WSTEPS / NWD, WRET = WD.reduce((s, w) => s + w[w.length - 1].rew * GAM ** (w.length - 1), 0) / NWD;
const WSHORT = Math.min(...WD.filter(w => w[w.length - 1].rew > 0).map(w => w.length));
const QW = replay(WD, 'q', PASSES), SW = replay(WD, 's', PASSES), QX = replay(XD, 'q', PASSES);
// Pip's own online Q-learning (same as the Q-learning reel: seed 54, 300 walks, eps 0.2), recorded as a diary too
const online = algo => { const R = rng(54), Q = new Float64Array(80), D = []; const best = (c, r) => G.term(c, r) ? 0 : Math.max(Q[qk(c, r, 0)], Q[qk(c, r, 1)], Q[qk(c, r, 2)], Q[qk(c, r, 3)]);
  const act = (c, r) => { if (R() < .2) return Math.floor(R() * 4); let b = -1e9, ms = []; for (let m = 0; m < 4; m++) { const q = Q[qk(c, r, m)]; if (q > b + 1e-9) { b = q; ms = [m]; } else if (q > b - 1e-9) ms.push(m); } return ms[Math.floor(R() * ms.length)]; };
  let rets = []; for (let e = 0; e < 300; e++) { let [c, r] = G.start, m = algo === 's' ? act(c, r) : 0; const walk = [];
    for (let k = 0; k < 2000; k++) { if (algo === 'q') m = act(c, r); const [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr), m2 = (algo === 's' && !rew) ? act(nc, nr) : 0;
      const nx = rew ? 0 : algo === 's' ? Q[qk(nc, nr, m2)] : best(nc, nr); Q[qk(c, r, m)] += ALPHA * (rew + GAM * nx - Q[qk(c, r, m)]); walk.push({ c, r, m, rew, nc, nr }); c = nc; r = nr; m = m2; if (rew) break; }
    D.push(walk); rets.push(walk[walk.length - 1].rew * GAM ** (walk.length - 1)); }
  return { Q: Array.from(Q), D, last100: rets.slice(200).reduce((a, b) => a + b, 0) / 100 }; };
const OWN = online('q'), OWNS = online('s');  // OWN reproduces the Q-learning reel's run exactly
// exact values of the Wanderer (repeated sweeps), and its Q at the start
const VW = (() => { let V = new Float64Array(20); for (let it = 0; it < 2000; it++) { const Nw = new Float64Array(20); CELLS.forEach(([c, r]) => { let v = 0; for (let m = 0; m < 4; m++) { const [nc, nr] = G.step(c, r, m), rw = G.term(nc, nr); v += .25 * (rw + (rw ? 0 : GAM * V[key(nc, nr)])); } Nw[key(c, r)] = v; }); V = Nw; } return V; })();
const QWEX = m => { const [nc, nr] = G.step(0, 3, m), rw = G.term(nc, nr); return rw + (rw ? 0 : GAM * VW[key(nc, nr)]); };
const maxQ = (Q, c, r) => Math.max(Q[qk(c, r, 0)], Q[qk(c, r, 1)], Q[qk(c, r, 2)], Q[qk(c, r, 3)]);
const argQ = (Q, c, r) => { let b = -1e9, bm = 0; for (let m = 0; m < 4; m++) if (Q[qk(c, r, m)] > b + 1e-12) { b = Q[qk(c, r, m)]; bm = m; } return bm; };
const flat = (Q, c, r) => { const v = [0, 1, 2, 3].map(m => Q[qk(c, r, m)]); return Math.max(...v) - Math.min(...v) < 1e-9; };
const blank = Q => CELLS.filter(([c, r]) => flat(Q, c, r)).length;
const PERF = [[0, 2], [0, 1], [0, 0], [1, 0], [2, 0], [3, 0]].every(([c, r], i) => Math.abs(maxQ(QW.Q, c, r) - GAM ** (5 - i)) < 1e-3) && Math.abs(maxQ(QW.Q, 0, 3) - GAM ** 6) < 1e-3;
const VST = (() => { let V = new Array(20).fill(0); for (let it = 0; it < 300; it++) { const N2 = V.slice(); CELLS.forEach(([c, r]) => { let b = -9; for (let m = 0; m < 4; m++) { const [nc, nr] = G.step(c, r, m), rw = G.term(nc, nr); b = Math.max(b, rw + (rw ? 0 : GAM * V[key(nc, nr)])); } N2[key(c, r)] = b; }); V = N2; } return V; })();
const QST = (c, r, m) => { const [nc, nr] = G.step(c, r, m), rw = G.term(nc, nr); return rw + (rw ? 0 : GAM * VST[key(nc, nr)]); };
const bestArrows = Q => CELLS.filter(([c, r]) => !flat(Q, c, r) && QST(c, r, argQ(Q, c, r)) > Math.max(...[0, 1, 2, 3].map(m => QST(c, r, m))) - 1e-9).length;
const STARTP = QW.snaps.map(Q => maxQ(Q, 0, 3));
// your-turn data sources
const SRC = [{ name: 'Pip’s own walks', Q: OWN.Q, steps: OWN.D.reduce((s, w) => s + w.length, 0), walks: 300 }, { name: 'the Wanderer’s diary', Q: QW.Q, steps: WSTEPS, walks: NWD }, { name: 'the careful mouse’s diary', Q: QX.Q, steps: XD.reduce((s, w) => s + w.length, 0), walks: NWD }];

// ───── drawing (four triangles per cell, as in the Q-learning reel)
function tri(xm, ym, x0, y0, x1, y1, m, col, a) { const P = [[[x0, y0], [x1, y0]], [[x1, y0], [x1, y1]], [[x1, y1], [x0, y1]], [[x0, y1], [x0, y0]]][m];
  ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(xm, ym); ctx.lineTo(...P[0]); ctx.lineTo(...P[1]); ctx.closePath(); ctx.fill(); ctx.restore(); }
function wedges(Q, c, r, x0, y0, x1, y1) { const xm = (x0 + x1) / 2, ym = (y0 + y1) / 2;
  for (let m = 0; m < 4; m++) { const v = Q[qk(c, r, m)]; if (Math.abs(v) >= .004) tri(xm, ym, x0 + 2, y0 + 2, x1 - 2, y1 - 2, m, v > 0 ? C.gold : C.rose, lerp(.14, .62, clamp(Math.abs(v) / .9))); }
  ctx.save(); ctx.globalAlpha = .55; rline(x0 + 3, y0 + 3, x1 - 3, y1 - 3, 300 + c * 9 + r, { c: C.line, w: .9, once: true, j: .6 }); rline(x1 - 3, y0 + 3, x0 + 3, y1 - 3, 330 + c * 9 + r, { c: C.line, w: .9, once: true, j: .6 }); ctx.restore(); }
function qmaze(Q, o = {}) { CELLS.forEach(([c, r]) => { const x0 = OX + c * CS, y0 = OY + r * CS; wedges(Q, c, r, x0, y0, x0 + CS, y0 + CS); }); maze({ t: o.t || 0 }); }
function arrowsQ(Q, a = 1, col = C.goldInk) { if (a <= 0) return; CELLS.forEach(([c, r]) => { if (flat(Q, c, r)) { txt('?', cx(c), cy(r) + 4, { s: 18, c: '#b9ab9a', a }); return; } ctx.save(); ctx.globalAlpha = a; G.arrowIn(c, r, argQ(Q, c, r), 400 + key(c, r), { c: col, w: 1.5 }); ctx.restore(); }); }
function zoom(Q, c, r, x, y, s, o = {}) { const a = o.a ?? 1; if (a <= 0) return; ctx.save(); ctx.globalAlpha = a; box(x, y, s, s, C.card); wedges(Q, c, r, x, y, x + s, y + s); L.rrect(x, y, s, s, 640, { w: 1.4 });
  const xm = x + s / 2, ym = y + s / 2, P = [[xm, y + s * .17], [x + s * .8, ym + 1], [xm, y + s * .85], [x + s * .2, ym + 1]];
  for (let m = 0; m < 4; m++) { const v = Q[qk(c, r, m)]; txt(sgn(Math.abs(v) < .0005 ? '0' : String(+v.toFixed(2))), P[m][0], P[m][1], { s: 14, c: v < -.0005 ? C.roseInk : v > .0005 ? C.goldInk : C.soft }); }
  if (o.label) txt(o.label, xm, y + s + 11, { s: 13, c: o.lc || C.soft }); ctx.restore(); }
function dim(a = .62) { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, 285); ctx.restore(); }
function wanderer(x, y, t, a = 1) { ctx.save(); ctx.globalAlpha = a; L.mouse(x, y, 1, .9, t); ctx.restore(); txt('?', x + 4, y - 22, { s: 16, c: C.soft, a: a * (.5 + .5 * Math.sin(t * 3)) }); }
// two labelled boxes: who acts, who is learned about
function roles(t, beh, tgt, same) { const y = 300, h = 58;
  [[30, 'behaviour policy', 'the rule that picks Pip’s moves', beh, C.teal], [210, 'target policy', 'the rule the numbers describe', tgt, C.gold]].forEach(([x, a, b, c, col], i) => { const u = clamp((t - .4 - i * 1.4) * 2); if (u <= 0) return; ctx.save(); ctx.globalAlpha = u;
    L.rrect(x, y, 160, h, 860 + i, { fill: C.card, c: col, w: 1.6 }); txt(a, x + 80, y + 14, { s: 16, c: col === C.gold ? C.goldInk : '#2f6f6c' }); txt(b, x + 80, y + 32, { s: 12, c: C.soft }); txt(c, x + 80, y + 48, { s: 15 }); ctx.restore(); });
  if (same != null && t > 3.4) txt(same ? '=  the same rule: on-policy' : '≠  different rules: off-policy', W / 2, y + h + 22, { s: 17, c: same ? '#2f6f6c' : C.goldInk, a: clamp(t - 3.4) }); }
// follow arrows from a cell (your turn)
function follow(Q, from) { let [c, r] = from; const P = [[c, r]], seen = new Set([key(c, r)]);
  while (!G.term(c, r) && P.length < 30) { if (flat(Q, c, r)) return { P, why: (P.length > 1 ? `after ${P.length - 1} steps: ` : '') + 'no clue here, the diary never came by' };
    const [nc, nr] = G.step(c, r, argQ(Q, c, r)); if (nc === c && nr === r) return { P, why: 'its best move bumps a wall' }; c = nc; r = nr; P.push([c, r]); if (seen.has(key(c, r))) return { P, why: 'going in circles' }; seen.add(key(c, r)); }
  return { P, why: null }; }

const STEPS = [
 { h: 'Two jobs', p: ['In the last two reels, Pip learned four numbers per cell while moving with its ε-greedy habit: the biggest number, or one time in five a random move.', 'Any such learner has two rules in play. One picks the moves that create the experience: the **behaviour policy**. The other is the rule whose values the numbers describe: the **target policy**. Careful: that is not the update target, the number each update moves toward. A policy is just a rule for picking moves.'],
   draw(t) { qmaze(OWN.Q, { t }); arrowsQ(OWN.Q, .5); pip(cx(0), cy(3), 1, t); roles(t, 'ε-greedy', '?', null); } },
 { h: 'SARSA: one rule', p: [`SARSA’s update target uses the next move that its ε-greedy habit really picks, random moves included. So its numbers describe ε-greedy: the very rule that is acting. Run in this maze, its biggest number at the start is ${f2(maxQ(OWNS.Q, 0, 3))}, close to what its Pip really earned, ${f2(OWNS.last100)}.`, 'When the two rules are the same, the learning is **on-policy**: you learn about exactly what you do.'],
   draw(t) { qmaze(OWNS.Q, { t }); arrowsQ(OWNS.Q, .5, '#2f6f6c'); pip(cx(0), cy(3), 1, t); dim(.3); txt('SARSA’s own run in this maze', W / 2, 286, { s: 16, c: '#2f6f6c', a: clamp(t * 2) }); roles(t, 'ε-greedy', 'ε-greedy', true); } },
 { h: 'Q-learning: two rules', p: [`Q-learning also acts ε-greedy, but its update target uses the biggest number: it assumes the best next move every time. Its numbers describe the **greedy** rule, a Pip that never moves at random. That is why they said ${f2(maxQ(OWN.Q, 0, 3))} at the start while Pip really earned ${f2(OWN.last100)}.`, 'Learning about one policy while following another is **off-policy** learning.'],
   draw(t) { qmaze(OWN.Q, { t }); arrowsQ(OWN.Q, .5); pip(cx(0), cy(3), 1, t); dim(.3); txt('Q-learning’s run (the Q-learning reel)', W / 2, 286, { s: 16, c: C.goldInk, a: clamp(t * 2) }); roles(t, 'ε-greedy', 'always the biggest', false); } },
 { h: 'Why Q-learning can do it', p: ['Q(s, a) asks: what happens after move a in cell s? The maze answers with a reward and a landing cell, and the answer is the same whoever chose the move. Then the max fills in the best way to carry on.', 'So nothing in Q-learning’s update target depends on who was acting. SARSA’s needs the next move of the walker, so it describes that walker.'],
   draw(t) { qmaze(OWN.Q, { t }); dim(.82); const u = clamp(t - .5);
     txt('update target  =  r  +  γ × max Q(s′, ·)', W / 2, 90, { s: 20, a: u }); txt('the maze’s reply     the best continuation', W / 2 + 40, 118, { s: 14, c: C.soft, a: u });
     txt('update target  =  r  +  γ × Q(s′, a′)', W / 2, 180, { s: 20, a: clamp(t - 2.5) }); txt('a′: the walker’s own next move', W / 2 + 40, 208, { s: 14, c: C.roseInk, a: clamp(t - 3) });
     txt('Q-learning', 60, 64, { s: 15, c: C.goldInk, al: 'left', a: u }); txt('SARSA', 60, 154, { s: 15, c: '#2f6f6c', al: 'left', a: clamp(t - 2.5) });
     card(['who chose move a?  doesn’t matter', { s: 'Q-learning: no part of the target is the walker’s', c: C.goldInk }, { s: 'SARSA: a′ comes from the walker', c: '#2f6f6c' }], t - 4.5, { gap: 1.1 }); } },
 { h: 'Someone else’s diary', p: [`Remember the **Wanderer**, the mouse whose every move is random? Pip finds a diary of ${NWD} of its walks from the start: every cell, move, reward and landing, ${WSTEPS.toLocaleString('en-US')} steps in all.`, `Only ${WCHEESE} of those walks reached the cheese; ${WCAT} ended on the cat. The quickest took ${WSHORT} steps. Can Pip learn the best route from this, without taking a single step itself?`],
   draw(t) { const w = Math.min(NWD - 1, Math.floor(t / .25)), walk = WD[w]; maze({ t }); trail(walk.map(s => [s.nc, s.nr]), .9);
     const l = walk[walk.length - 1]; wanderer(cx(l.nc), cy(l.nr), t); if (l.rew > 0) bubble('+1', cx(4), cy(0) - 18, (t % .25) * 4);
     card([`diary: walk ${w + 1} of ${NWD}, ${walk.length} steps, ${l.rew > 0 ? 'cheese' : 'cat'}`, `so far: ${WD.slice(0, w + 1).filter(x => x[x.length - 1].rew > 0).length} cheese, ${WD.slice(0, w + 1).filter(x => x[x.length - 1].rew < 0).length} cat`, { s: 'every move random', c: C.soft }], 10, { lh: 23 }); caption('teal dots: footprints of the walk shown'); } },
 { h: 'Replaying the diary', p: [`Pip reads the diary step by step and applies Q-learning’s update to each step, as if it had made that move itself. It reads the whole diary ${PASSES} times.`, `The biggest number at the start: ${f2(STARTP[0])} after one reading, ${f2(STARTP[1])} after two, ${f2(STARTP[PASSES - 1])} after ${PASSES}. So fast, because one reading is ${WSTEPS.toLocaleString('en-US')} updates spread over every cell. ${PERF ? `Both shortest routes, up the left side or along the bottom, get the perfect numbers, 0.53 up to 1, though the Wanderer never walked either.` : ''}`],
   shot: [5, 13],
   draw(t) { const p = clamp(Math.floor((t - .5) / 1), 0, PASSES - 1), Q = QW.snaps[p]; qmaze(Q, { t }); if (t > PASSES + .8) arrowsQ(QW.Q, clamp(t - PASSES - .8)); pip(cx(0), cy(3), 1, t);
     zoom(Q, 0, 3, 30, 294, 92, { label: 'the start cell' });
     card([`reading ${p + 1} of ${PASSES}`, `biggest at the start: ${f2(STARTP[p])}`, { s: 'perfect: 0.53', c: C.soft }], 10, { x: 140, w: 230, fs: 17 }); } },
 { h: 'SARSA, same diary', p: [`Replayed with SARSA, the same diary gives the Wanderer’s own values, since its update target uses the Wanderer’s next move. At the start, all four are close to ${sgn(VW[key(0, 3)].toFixed(2))}.`, `Why near 0, when ${WCAT} walks in 100 met the cat? The walks average ${Math.round(WLEN)} steps, and 0.9 multiplied ${Math.round(WLEN) - 1} times is ${(GAM ** (Math.round(WLEN) - 1)).toFixed(4)}: late endings barely count. Correct, but useless for Pip.`],
   draw(t) { qmaze(SW.Q, { t }); pip(cx(0), cy(3), 1, t);
     zoom(QW.Q, 0, 3, 40, 294, 92, { label: 'Q-learning', lc: C.goldInk, a: clamp(t - .6) }); zoom(SW.Q, 0, 3, 268, 294, 92, { label: 'SARSA', lc: '#2f6f6c', a: clamp(t - 1.6) });
     txt('start cell,', W / 2, 318, { s: 16, c: C.soft, a: clamp(t - 2.4) }); txt('same diary', W / 2, 338, { s: 16, c: C.soft, a: clamp(t - 2.4) }); txt('big grid: SARSA', W / 2, 364, { s: 14, c: '#2f6f6c', a: clamp(t - 3) }); } },
 { h: 'Why off-policy matters', p: [`Learning about one rule while following another opens doors. Pip can **reuse old experience**, replaying it many times instead of walking again. It can learn from other walkers: a random one, a human, an older version of itself. And it can explore as wildly as it likes, while still learning the best way to act. The random diary even got the bottom-row arrows right, which Pip’s own walks did not: ${bestArrows(QW.Q)} of ${CELLS.length} arrows are best moves.`],
   draw(t) { qmaze(QW.Q, { t }); arrowsQ(QW.Q, .6); pip(cx(0), cy(3), 1, t); dim(.4);
     card(['reuse old experience, again and again', 'learn from others: logs, demonstrations', 'explore freely, still learn the best rule'], t - .6, { gap: 1.3 }); } },
 { h: 'Risk: only what the data shows', p: [`Another diary, from a careful mouse that follows the habit’s arrows and never moves at random: ${NWD} walks, all the same 7 steps. Q-learning learns that route perfectly, and nothing else. In ${blank(QX.Q)} of ${CELLS.length} cells the four numbers never moved.`, 'Drop Pip in one of those cells and it has no clue. Off-policy data can only teach about moves someone actually tried.'],
   draw(t) { qmaze(QX.Q, { t }); arrowsQ(QX.Q, clamp(t - 1)); const P = XD[0].map(s => [s.c, s.r]).concat([[4, 0]]); trail(P, .7 * clamp(t - .4));
     const t2 = t - 4, P2 = follow(QX.Q, [2, 3]).P; pip(cx(2), cy(3), 1, t); if (t2 > 0) txt('no clue', cx(2), cy(3) - 22, { s: 15, c: C.roseInk, a: clamp(t2) });
     card([`careful mouse: ${NWD} walks, all 7 steps`, `cells never visited: ${blank(QX.Q)} of ${CELLS.length}`, { s: 'no data, no numbers', c: C.roseInk }], t - 2, { gap: 1.1 }); } },
 { h: 'Risk: a rule that never acted', p: ['Off-policy numbers describe a rule that never acted. In the cat room, Q-learning’s numbers were about a greedy Pip; the Pip that really walked was ε-greedy, and met more cats than SARSA’s.', 'With large problems a neural network replaces the table, and changing one guess shifts others. A guess chasing a guess can then run away, and off-policy data brings no walk of that rule to correct it. Sutton and Barto call this mix the **deadly triad**; more in a later topic.'],
   draw(t) { qmaze(QW.Q, { t }); arrowsQ(QW.Q, .5); pip(cx(0), cy(3), 1, t); dim(.4);
     card(['the deadly triad:', '   a network instead of a table', '   bootstrapping (guesses from guesses)', '   off-policy data'], t - .6, { gap: 1 }); } },
 { h: 'Your turn', ctl: { label: 'data', values: [0, 1, 2], def: 1, fmt: v => ['own', 'Wanderer', 'careful'][v], onChange: st => { st.path = null; } }, tap: tapCell,
   p: ['Pick whose experience Q-learning learns from: Pip’s own ε-greedy walks, the Wanderer’s random diary, or the careful mouse’s diary. The diaries are replayed ten times.', 'Tap a cell to drop Pip there and follow the arrows. Where the data never went, Pip has no clue.'],
   draw(t, st) { const S = SRC[st.ci ?? 1]; qmaze(S.Q, { t }); arrowsQ(S.Q, .7);
     const P = st.path; if (P) { const t2 = t - st.pt, kk = clamp(Math.floor(t2 / .45), 0, P.length - 1), uu = ease((t2 % .45) / .45 * 1.3), a = P[kk], b = P[Math.min(P.length - 1, kk + 1)];
       trail(P.slice(0, kk + 1), .7); pip(lerp(cx(a[0]), cx(b[0]), uu), lerp(cy(a[1]), cy(b[1]), uu), b[0] - a[0], t);
       caption(kk >= P.length - 1 ? (st.why ? st.why : `reached the ${G.term(...P[P.length - 1]) > 0 ? 'cheese' : 'cat'} in ${P.length - 1} steps`) : `step ${kk}`);
     } else { pip(cx(0), cy(3), 1, t); caption('tap a cell to drop Pip there'); }
     card([`learning from ${S.name}`, `${S.walks} walks, ${S.steps.toLocaleString('en-US')} steps`, `biggest number at the start: ${f2(maxQ(S.Q, 0, 3))}`, { s: `cells with no clue: ${blank(S.Q)} of ${CELLS.length}`, c: blank(S.Q) ? C.roseInk : C.soft }], 10, { y: 290, lh: 21, fs: 16 }); } },
 { h: 'Where you meet them', p: ['Off-policy: Q-learning, and **DQN**, which in 2015 kept a memory of past Atari moves and replayed them in random order. SAC, a 2018 method for robot control, does the same. Learning purely from logged data is called offline RL.', 'On-policy: SARSA, and **PPO** (2017), used to fine-tune chatbots. PPO learns a policy directly, without a Q table (a later topic). It gathers fresh experience with its current policy, learns from it briefly, then throws it away.'],
   draw(t) { qmaze(QW.Q, { t }); arrowsQ(QW.Q, .5); pip(cx(0), cy(3), 1, t); dim(.4);
     card([{ s: 'off-policy: Q-learning, DQN (replay memory), SAC', c: C.goldInk }, { s: 'on-policy: SARSA, PPO', c: '#2f6f6c' }, { s: 'on: learns only from its current rule’s data', c: C.soft }, { s: 'off: reuses any data, harder to keep stable', c: C.soft }], t - .5, { gap: 1.1 }); } },
 { h: 'Recap', p: ['The **behaviour policy** acts; the **target policy** is what the numbers describe. Same rule: on-policy, like SARSA. Different rules: off-policy, like Q-learning.', 'Off-policy learners can reuse old experience and learn from others, but only about moves the data contains, and they are harder to keep stable.'],
   draw(t) { qmaze(QW.Q, { t }); arrowsQ(QW.Q, .7); pip(cx(0), cy(3), 1, t);
     card(['behaviour policy: picks the moves', 'target policy: the rule being learned about', 'same: on-policy (SARSA)', 'different: off-policy (Q-learning)'], t - .3, { gap: 1 }); } },
];
function tapCell(x, y, st, t) { const c = Math.floor((x - OX) / CS), r = Math.floor((y - OY) / CS); if (c < 0 || c >= 5 || r < 0 || r >= 4 || G.isRock(c, r) || G.term(c, r)) return;
  const f = follow(SRC[st.ci ?? 1].Q, [c, r]); st.path = f.P; st.why = f.why; st.pt = t; }
RLStory.register('onoff', { title: 'On-Policy vs Off-Policy', track: 'Model-free', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { OWNS, WLEN, WRET, bestArrows, WD, XD, QW, SW, QX, OWN, VW, QWEX, WSTEPS, WCHEESE, WCAT, WSHORT, PERF, STARTP, SRC, maxQ, blank, qk } });
})();
