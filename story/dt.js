// Decision Transformer as a story: Pip finds boxes of the Wanderer's diaries (random walks, as in the values and on/off reels).
// Each diary step is written as (return-to-go, cell, move). A stand-in for the transformer, honestly labelled (a lookup table:
// in this cell, copy the move whose return-to-go is closest to what is still wanted), is asked for a total at the start.
// It finds the 7-move route from diaries whose best walk is far worse, by joining pieces through the countdown; but not always,
// and never from pieces that are missing. Offline value learning on the same diaries is the comparison.
// Rules for this reel: dry floor, each move costs 0.05, +1 onto the cheese, −1 onto the cat, no discount, stop after 30 moves.
// Every number comes from the seeded simulation below (reel.sim).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({});
const { cx, cy, maze, walkPos, pip } = G;
const COST = 0.05, CAP = 30, NW = 40;
const no = (c, r) => r * 5 + c + 1;                                     // loop reel numbering: cells 1..20 row by row
const MV = ['up', 'right', 'down', 'left'];
const rew = (nc, nr) => -COST + G.term(nc, nr);

// ───── a box of diaries: NW Wanderer walks from the start (every move random, bumps allowed), stopped after CAP moves
function makeBox(seed, n = NW) { const R = rng(seed), D = [];
  for (let i = 0; i < n; i++) { let c = 0, r = 3; const tr = [];
    for (let k = 0; k < CAP; k++) { const m = Math.floor(R() * 4), [nc, nr] = G.step(c, r, m), rw = rew(nc, nr); tr.push({ c, r, m, nc, nr, rw, w: i }); c = nc; r = nr; if (G.term(c, r)) break; }
    let g = 0; for (let j = tr.length - 1; j >= 0; j--) { g += tr[j].rw; tr[j].rtg = g; } D.push(tr); }
  const by = {}; D.forEach(tr => tr.forEach(s => { (by[s.r * 5 + s.c] ||= []).push(s); }));
  const ret = D.map(tr => tr[0].rtg), end = D.map(tr => { const l = tr[tr.length - 1]; return G.term(l.nc, l.nr); });
  return { D, by, ret, best: Math.max(...ret), cheese: end.filter(e => e > 0).length, cat: end.filter(e => e < 0).length }; }
// ───── the stand-in for the transformer: in this cell, the diary steps whose return-to-go is closest to the wanted total; copy their most common move
function runDT(B, target) { let c = 0, r = 3, want = target, tot = 0; const tr = [];
  for (let k = 0; k < CAP; k++) { const ex = B.by[r * 5 + c]; if (!ex) { tot -= COST * (CAP - k); break; }
    let best = Infinity, ms = [0, 0, 0, 0], src = [-1, -1, -1, -1];
    for (const s of ex) { const d = Math.abs(s.rtg - want); if (d < best - 1e-9) { best = d; ms = [0, 0, 0, 0]; src = [-1, -1, -1, -1]; } if (d < best + 1e-9) { ms[s.m]++; if (src[s.m] < 0) src[s.m] = s; } }
    const m = ms.indexOf(Math.max(...ms)), [nc, nr] = G.step(c, r, m), rw = rew(nc, nr);
    tr.push({ c, r, m, nc, nr, rw, want, ex: src[m], n: ms[m] }); tot += rw; want -= rw; c = nc; r = nr; if (G.term(c, r)) break; }
  const l = tr[tr.length - 1]; return { tr, tot, end: l ? G.term(l.nc, l.nr) : 0 }; }
// ───── offline value learning on the same diaries: value of a logged move = its reward + best logged value where it lands (no discount)
function runQ(B) { const T = {}; B.D.forEach(tr => tr.forEach(s => { T[(s.r * 5 + s.c) * 4 + s.m] = s; })); const Q = {};
  for (let it = 0; it < 60; it++) for (const k in T) { const s = T[k], nk = s.nr * 5 + s.nc; let nx = 0;
    if (!G.term(s.nc, s.nr)) { nx = -9; for (let m = 0; m < 4; m++) if (T[nk * 4 + m]) nx = Math.max(nx, Q[nk * 4 + m] ?? -9); } Q[k] = s.rw + nx; }
  let c = 0, r = 3, tot = 0; const tr = [];
  for (let k = 0; k < CAP; k++) { const sk = r * 5 + c; let bm = -1, bq = -1e9; for (let m = 0; m < 4; m++) if (T[sk * 4 + m] && Q[sk * 4 + m] > bq) { bq = Q[sk * 4 + m]; bm = m; }
    if (bm < 0) { tot -= COST * (CAP - k); break; } const s = T[sk * 4 + bm]; tr.push(s); tot += s.rw; c = s.nc; r = s.nr; if (G.term(c, r)) break; }
  return { tr, tot, Q }; }

// ───── the story boxes (seeds 1000 + k; the same seeds feed the 100-box count)
const B1 = makeBox(1004), B2 = makeBox(1026), B0 = makeBox(1000);
const BESTW = B1.ret.indexOf(B1.best), BW = B1.D[BESTW], BWB = BW.filter(s => s.c === s.nc && s.r === s.nr).length;                 // the best diary in box 1
const D65 = runDT(B1, .65), DLO = runDT(B1, -1.5), D0 = runDT(B1, 0);
const ASKS = []; for (let a = -1.5; a <= 1.0001; a += .05) ASKS.push(+a.toFixed(2));
const GOT = ASKS.map(a => runDT(B1, a).tot);
const SL = [-1.5, -0.5, 0, 0.2, 0.4, 0.65, 1];
const dtMemo = {}; const dtOf = (B, k, a) => dtMemo[k + ':' + a] || (dtMemo[k + ':' + a] = runDT(B, a));
const E2 = runDT(B2, .65), Q2 = runQ(B2), Q1 = runQ(B1), E0 = runDT(B0, .65), Q0 = runQ(B0);
const SRC = [...new Set(D65.tr.map(s => s.ex.w))];                      // diaries the 0.65 walk copied from
// where box 2's walk went wrong: the first step that differs from the value-learning walk
const DIV2 = E2.tr.findIndex((s, i) => !Q2.tr[i] || s.m !== Q2.tr[i].m);
// ───── 100 boxes of each size, computed a few at a time when the chart step is shown
const SIZES = [5, 10, 20, 40, 80], NBOX = 100;
const many = { k: 0, dt: SIZES.map(() => 0), q: SIZES.map(() => 0), whole: SIZES.map(() => 0) };
function manyStep(budget) { while (many.k < NBOX * SIZES.length && budget-- > 0) { const si = Math.floor(many.k / NBOX), b = many.k % NBOX, B = makeBox(1000 + b, SIZES[si]);
    if (runDT(B, .65).tot > .649) many.dt[si]++; if (runQ(B).tot > .649) many.q[si]++; if (B.best > .649) many.whole[si]++; many.k++; } return many.k >= NBOX * SIZES.length; }

// ───── formatting and drawing helpers
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−');
const BLUE = '#4f7fb0', GREEN = '#5d9a5b', PLUM = '#9b7fc0';
function diaryTrails(B, a = .5, only) { B.D.forEach((tr, i) => { if (only && !only(i)) return; tr.forEach((s, j) => dot(cx(s.c) + ((i * 7 + j * 5) % 23) - 11, cy(s.r) + ((i * 11 + j * 3) % 23) - 11, 1.6, C.soft, a)); }); }
function pathLine(tr, col, w = 2.4, n = tr.length, off = 0) {
  ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.beginPath(); if (!tr.length) { ctx.restore(); return; }
  ctx.moveTo(cx(tr[0].c) + off, cy(tr[0].r) + off); for (let i = 0; i < Math.min(n, tr.length); i++) { const s = tr[i]; ctx.lineTo(cx(s.nc) + off + (s.nc === s.c && s.nr === s.r ? 0 : 0), cy(s.nr) + off); } ctx.stroke(); ctx.restore(); }
function notebook(x, y, s = 1, col = '#e8d9c0') { for (let k = 2; k >= 0; k--) rrect(x + k * 4 * s, y - k * 4 * s, 34 * s, 24 * s, 1700 + k, { fill: col, w: 1.1 }); rline(x + 6 * s, y + 8 * s, x + 26 * s, y + 8 * s, 1710, { c: C.soft, w: 1, once: true }); rline(x + 6 * s, y + 14 * s, x + 22 * s, y + 14 * s, 1711, { c: C.soft, w: 1, once: true }); }
function cellNums(a = 1) { for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.isRock(c, r)) txt(String(no(c, r)), G.ox + c * G.cs + 9, G.oy + r * G.cs + 11, { s: 12, c: C.soft, a }); }
function bumps(tr, n) { for (let i = 0; i < Math.min(n, tr.length); i++) { const s = tr[i]; if (s.c === s.nc && s.r === s.nr) { const [dx, dy] = G.MOVES[s.m]; txt('bump', cx(s.c) + dx * 18, cy(s.r) + dy * 18 + 4, { s: 12, c: C.roseInk }); } } }
// play a walk: Pip moves along tr; returns the step index shown
function play(tr, t, dur, col, o = {}) { const m = walkPos(tr, 0, tr.length, Math.max(0, t), dur); const k = m.done ? tr.length : m.k;
  pathLine(tr, col, 2.6, k); if (!m.done && m.k < tr.length) pathLine(tr.slice(m.k, m.k + 1), col, 2.6, 0); pip(m.x, m.y, m.face, t); return { k, m }; }
const endWord = e => e > 0 ? 'cheese' : e < 0 ? 'cat' : 'stopped';

const STEPS = [
 { h: 'A box of old diaries', p: [`Pip finds a box of ${NW} diaries. Each records one walk by the **Wanderer**, the mouse from earlier reels who moves at random, bumping into walls. Pip may not walk itself; it can only learn from these pages, as in the offline RL reel.`,
   `In this box, ${B1.cheese} walks reached the cheese, ${B1.cat} met the cat, and the other ${NW - B1.cheese - B1.cat} were stopped after 30 moves.`],
   shot: [4, 11],
   draw(t) { maze({ t }); diaryTrails(B1, .55 * clamp(t - 1.5)); pip(cx(0), cy(3), 1, t);
     card([`${NW} diaries of random walks`, `cheese ${B1.cheese},  cat ${B1.cat},  stopped ${NW - B1.cheese - B1.cat}`, { s: 'no new walks allowed', c: C.soft }], t - 3.5, { y: 296, gap: 1.1 }); notebook(330, 330, 1); } },
 { h: 'Rules for this reel', p: ['Same maze, dry floor, new scoring: each move costs 0.05; the cheese pays +1, the cat −1, either ends the walk. Rewards are summed, undiscounted: a Decision Transformer works with plain totals, so the total still wanted drops by each reward.',
   `So shorter is better. The best walk takes 7 moves: 1 − 7 × 0.05 = 0.65. The box’s best diary scores only ${f2(B1.best)}.`],
   shot: [4, 11],
   draw(t) { maze({ t, labels: 1 }); const route = [[0, 3, 0, 2], [0, 2, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 2, 0], [2, 0, 3, 0], [3, 0, 4, 0]].map(([c, r, nc, nr]) => ({ c, r, nc, nr }));
     if (t > 4) alpha(clamp(t - 4), () => pathLine(route, GREEN, 2.4, Math.min(7, Math.floor((t - 4) * 3))));
     card(['each move: −0.05', 'onto the cheese: +1,  onto the cat: −1', { s: 'total = all rewards added, no discount', c: C.goldInk }, { s: `best possible 0.65 (7 moves), best diary ${f2(B1.best)}`, c: GREEN }], t - 1.5, { y: 290, gap: 1.2, fs: 16, lh: 22 }); } },
 { h: 'Return-to-go', p: [`The key number is the **return-to-go**: the total still to come. Here is the box’s best diary: ${BW.length} moves, ${BWB} of them bumps into walls or rocks. At the start it is ${f2(BW[0].rtg)}, the whole total.`,
   'Each paid move cost is no longer to come, so it creeps up by 0.05. On the last move the cheese arrives, leaving 0.'],
   shot: [6, 13],
   draw(t) { maze({ t }); const dur = .45, m = walkPos(BW, 0, BW.length, Math.max(0, t - .5), dur), k = m.done ? BW.length : m.k;
     pathLine(BW, C.teal, 2, k); pip(m.x, m.y, m.face, t); if (!m.done && m.s && m.s.c === m.s.nc && m.s.r === m.s.nr) bumps([m.s], 1); const cur = m.done ? 0 : BW[Math.min(k, BW.length - 1)].rtg;
     card([`move ${Math.min(k + 1, BW.length)} of ${BW.length}`, { s: `return-to-go here: ${f2(cur)}`, c: C.goldInk }, { s: m.done ? 'walk over: nothing left to come' : 'each move adds 0.05 to it', c: C.soft }], 1, { y: 296, gap: 0 }); } },
 { h: 'A walk as a sentence', p: ['Write each diary step as three tokens: the return-to-go, the cell, the move. The return-to-go enters as a plain number; each token becomes numbers the network reads. A walk becomes a sentence. Cells run 1 to 20, as in the loop reel; the start is 16.',
   'A **transformer**, the network behind chatbots, predicts the next move token, as a chatbot predicts the next word.'],
   shot: [4, 11],
   draw(t) { maze({ t }); cellNums(clamp(t - .5)); pathLine(BW, C.teal, 2, 4);
     const toks = BW.slice(0, 4).flatMap(s => [[f2(s.rtg), C.goldInk], [String(no(s.c, s.r)), C.ink], [MV[s.m], BLUE]]);
     toks.forEach(([s, col], i) => { const u = clamp((t - 2 - i * .35) * 3); if (u <= 0) return; const x = 22 + (i % 6) * 60, y = 296 + Math.floor(i / 6) * 34;
       alpha(u, () => { box(x, y, 54, 26, C.card); rrect(x, y, 54, 26, 1800 + i, { c: i % 3 === 0 ? C.gold : C.line, w: 1.1 }); txt(s, x + 27, y + 14, { s: 15, c: col }); }); });
     if (t > 7) txt('to-go · cell · move, then the next step …', W / 2, 374, { s: 13, c: C.soft, a: clamp(t - 7) }); } },
 { h: 'Training is plain prediction', p: ['Training needs no values or Bellman updates: the model just predicts the logged move from what came before. Simple and stable: no values feeding on their own guesses, which inflated numbers in the offline RL reel.',
   'Ours is a stand-in **lookup**: in this cell, copy the most common move of diary steps whose return-to-go is closest to the total still wanted, up on ties. A real network also reads recent steps and generalises.'],
   shot: [4, 11],
   draw(t) { maze({ t, hi: [[0, 3, C.yellow, .4]] }); diaryTrails(B1, .25); pip(cx(0), cy(3), 1, t);
     const ex = B1.by[no(0, 3) - 1].slice().sort((a, b) => b.rtg - a.rtg), seen = new Set(), rows = [];
     for (const s of ex) { const k = f2(s.rtg) + s.m; if (seen.has(k)) continue; seen.add(k); rows.push(s); if (rows.length >= 4) break; }
     card(['diary steps in cell 16 (largest return-to-go first):', ...rows.map(s => ({ s: `   to-go ${f2(s.rtg)}  →  ${MV[s.m]}`, c: C.ink }))], t - 2, { y: 286, gap: .7, fs: 15, lh: 21 }); } },
 { h: 'Ask for 0.65', p: [`Now ask for the best total, 0.65. In cell 16 the closest return-to-go is ${f2(D65.tr[0].ex.rtg)}, from a diary back in cell 16 late in its walk, so Pip copies ${MV[D65.tr[0].m]}. After each move Pip subtracts its reward from the total still wanted, and looks again.`,
   `Pip reaches the cheese in ${D65.tr.length} moves: ${f2(D65.tot)}, beating every diary in the box, whose best scored ${f2(B1.best)}.`],
   shot: [5, 12],
   draw(t) { maze({ t }); diaryTrails(B1, .2); const dur = .9, { k, m } = play(D65.tr, t - .8, dur, BLUE);
     const s = D65.tr[Math.min(k, D65.tr.length - 1)];
     card(m.done ? [{ s: `reached the cheese: total ${f2(D65.tot)}`, c: BLUE }, `best single diary: ${f2(B1.best)}`] : [`wanted from here: ${f2(s.want)}`, `cell ${no(s.c, s.r)}: closest diary to-go ${f2(s.ex.rtg)} → ${MV[s.m]}`], 1, { y: 300, gap: 0 }); } },
 { h: 'Pieces of different walks', p: [`How can it beat every diary? At each cell it copies a step from whichever diary fits the total still wanted, and those steps came from ${SRC.length} different diaries, the box’s only ${B1.cheese} that reached the cheese. Pieces of different walks make a better one: **stitching**.`,
   'The Decision Transformer paper saw this too: trained only on random walks, asking a high return gave short paths.'],
   shot: [4, 11],
   draw(t) { maze({ t }); const cols = [BLUE, C.orange, PLUM, GREEN, C.rose, C.teal];
     D65.tr.forEach((s, i) => { const u = clamp((t - .5 - i * .6) * 2); if (u <= 0) return; const col = cols[SRC.indexOf(s.ex.w) % cols.length]; alpha(u, () => { pathLine([s], col, 3.2); }); });
     pip(cx(0), cy(3), 1, t);
     card(SRC.map((w, j) => ({ s: `diary ${w + 1}: ${D65.tr.filter(s => s.ex.w === w).length} of the ${D65.tr.length} moves`, c: cols[j % cols.length] })), t - 5, { y: 292, gap: .7, fs: 15, lh: 22 }); } },
 { h: 'Ask for less', p: [`The number is a dial. Ask for 0, and Pip reaches the cheese the long way, in ${D0.tr.length} moves: ${f2(D0.tot)}. Ask for −1.5, and it wanders until it is stopped after 30 moves: ${f2(DLO.tot)}.`,
   'Nobody wants a bad walk, but it shows the model really listens to the total it is given.'],
   shot: [6, 13],
   draw(t) { maze({ t }); const tr = t < 7.5 ? D0.tr : DLO.tr, t0 = t < 7.5 ? .5 : 7.5; play(tr, t - t0, .3, t < 7.5 ? GREEN : C.rose);
     card([{ s: `asked 0 → got ${f2(D0.tot)} (${D0.tr.length} moves)`, c: GREEN }, { s: `asked −1.5 → got ${f2(DLO.tot)} (${endWord(DLO.end)})`, c: C.roseInk, a: 1 }], t - 1, { y: 300, gap: 6.5 }); } },
 { h: 'Asked versus got', p: [`Requests run from −1.5 to 1 in 0.05 steps. Across: the total asked. Up: the total got; on the dashed diagonal they match. Below −0.5 Pip always wanders until stopped, −1.50. No diary made a quick trip to the cat: the quickest of ${B1.cat} cat walks took ${Math.min(...B1.D.filter(tr => { const l = tr[tr.length - 1]; return G.term(l.nc, l.nr) < 0; }).map(tr => tr.length))} moves.`,
   `From −0.5 to 0.65 the dots sit on the diagonal. Above, they stay flat: asking 1 still gives ${f2(GOT[ASKS.indexOf(1)])}. No better walk exists to copy.`],
   shot: [4, 11],
   draw(t) { const x0 = 70, x1 = 360, y0 = 260, y1 = 40, lo = -1.6, hi = 1.05, X = v => x0 + (v - lo) / (hi - lo) * (x1 - x0), Y = v => y0 - (v - lo) / (hi - lo) * (y0 - y1);
     rline(x0, y0, x1, y0, 1900, { c: C.line, w: 1, once: true }); rline(x0, y0, x0, y1, 1901, { c: C.line, w: 1, once: true });
     [-1.5, -1, -0.5, 0, 0.5, 1].forEach(v => { txt(String(v).replace('-', '−'), X(v), y0 + 12, { s: 12, c: C.soft }); txt(String(v).replace('-', '−'), x0 - 8, Y(v), { s: 12, c: C.soft, al: 'right' }); });
     txt('total asked for', x1, y0 + 28, { s: 13, c: C.soft, al: 'right' }); txt('total got', x0 + 6, y1 - 10, { s: 13, c: C.soft, al: 'left' });
     alpha(clamp(t - 1), () => { ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.grey; ctx.beginPath(); ctx.moveTo(X(-1.5), Y(-1.5)); ctx.lineTo(X(1), Y(1)); ctx.stroke(); ctx.restore(); });
     ctx.save(); ctx.setLineDash([2, 4]); ctx.strokeStyle = GREEN; ctx.globalAlpha = clamp(t - 6); ctx.beginPath(); ctx.moveTo(x0, Y(.65)); ctx.lineTo(x1, Y(.65)); ctx.stroke(); ctx.restore();
     txt('best possible 0.65', x0 + 6, Y(.65) - 9, { s: 12, c: GREEN, al: 'left', a: clamp(t - 6) });
     const n = Math.floor(clamp((t - 2) / 4) * ASKS.length); for (let i = 0; i < n; i++) dot(X(ASKS[i]), Y(GOT[i]), 3, BLUE, .85);
     card(['dot: one request, played from the start', { s: 'flat at the top: no better walk to copy', c: C.goldInk }], t - 8, { y: 300, gap: 1 }); } },
 { h: 'Only what is in the box', p: [`Another box, ${B0.cheese === 0 ? 'where no diary ever reached the cheese' : ''}. Ask for 0.65 and Pip wanders until stopped: ${f2(E0.tot)}. No total, however high, can produce cheese steps never written down.`,
   `A copier is limited by its data’s pieces. Value learning on that box reaches no cheese either: it heads for the cat in ${Q0.tr.length} moves, ${f2(Q0.tot)}, the least bad ending it has seen.`],
   shot: [4, 11],
   draw(t) { maze({ t }); diaryTrails(B0, .4); play(E0.tr, t - 1, .2, C.rose);
     card([`this box: cheese ${B0.cheese}, cat ${B0.cat}`, { s: `asked 0.65 → got ${f2(E0.tot)} (${endWord(E0.end)})`, c: C.roseInk }, { s: `value learning: ${f2(Q0.tot)} (${endWord(G.term(Q0.tr[Q0.tr.length - 1].nc, Q0.tr[Q0.tr.length - 1].nr))})`, c: C.soft }], t - 4, { y: 296, gap: 1.2 }); } },
 { h: 'When pieces do not join', p: [`A third box holds every piece of the 7-move route. Asked for 0.65, Pip still takes ${E2.tr.length} moves: ${f2(E2.tot)}. In cell ${no(E2.tr[DIV2].c, E2.tr[DIV2].r)} the closest return-to-go came from a diary going ${MV[E2.tr[DIV2].m]}, the long way.`,
   `Value learning, here a table over logged moves only, so nothing is invented, finds ${f2(Q2.tot)}: a move’s value is its reward plus the best value where it lands, so good endings flow back through shared cells. The lookup joins pieces only when the countdown lines up.`],
   shot: [5, 12],
   draw(t) { maze({ t }); diaryTrails(B2, .2); alpha(clamp(t - 6), () => pathLine(Q2.tr, GREEN, 2.4, Q2.tr.length, -4)); play(E2.tr, t - .5, .6, BLUE);
     card([{ s: `lookup, asked 0.65: ${f2(E2.tot)} in ${E2.tr.length} moves`, c: BLUE }, { s: `value learning: ${f2(Q2.tot)} in ${Q2.tr.length} moves`, c: GREEN }], t - 6, { y: 300, gap: 1 }); } },
 { h: 'A hundred boxes', p: ['Now 100 boxes per size: 5, 10, 20, 40 and 80 diaries. Across: diaries per box. Up: boxes of 100 giving the 7-move walk. Grey: a single diary already was it. Blue: the lookup asked for 0.65. Green: value learning.',
   'Both join pieces far beyond any single diary. The lookup is less reliable: with 40 diaries, 78 boxes against 85.'],
   shot: [5, 12],
   draw(t) { const done = manyStep(t > .3 ? 12 : 0), x0 = 70, x1 = 350, y0 = 250, y1 = 60, X = i => x0 + i / (SIZES.length - 1) * (x1 - x0), Y = v => y0 - v / NBOX * (y0 - y1);
     rline(x0, y0, x1, y0, 2000, { c: C.line, w: 1, once: true }); rline(x0, y0, x0, y1, 2001, { c: C.line, w: 1, once: true });
     SIZES.forEach((s, i) => txt(String(s), X(i), y0 + 12, { s: 12, c: C.soft })); [0, 50, 100].forEach(v => txt(String(v), x0 - 8, Y(v), { s: 12, c: C.soft, al: 'right' }));
     txt('diaries per box', x1, y0 + 28, { s: 13, c: C.soft, al: 'right' }); txt('boxes (of 100) giving the 7-move walk', x0 - 20, y1 - 16, { s: 13, c: C.soft, al: 'left' });
     const ns = Math.min(SIZES.length, Math.floor(many.k / NBOX)), show = done ? SIZES.length : ns;
     [[many.whole, C.grey], [many.dt, BLUE], [many.q, GREEN]].forEach(([v, col], j) => { const a = clamp(t - 1 - j * 1.5); if (a <= 0) return;
       ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.beginPath(); for (let i = 0; i < show; i++) i ? ctx.lineTo(X(i), Y(v[i])) : ctx.moveTo(X(i), Y(v[i])); ctx.stroke(); ctx.restore();
       for (let i = 0; i < show; i++) dot(X(i), Y(v[i]), 3, col, a); });
     card(done ? [{ s: `40 diaries: single ${many.whole[3]},  lookup ${many.dt[3]},  values ${many.q[3]}`, c: C.ink }, { s: `80 diaries: single ${many.whole[4]},  lookup ${many.dt[4]},  values ${many.q[4]}`, c: C.ink }] : [`counting … ${many.k} of ${NBOX * SIZES.length}`], 1, { y: 300, gap: 0, fs: 16 }); } },
 { h: 'Your turn', ctl: { label: 'ask', values: SL, def: 5, fmt: v => String(v).replace('-', '−') },
   p: ['Drag the total to ask for, and Pip plays it from the start, in the first box. Tap the maze to switch to the third box, where the pieces do not line up.',
   'Watch where it lands below 0.65, and what happens when you ask for more than the maze allows.'],
   shot: [4, 10],
   tap(x, y, st) { st.b = st.b ? 0 : 1; st.changed = performance.now() / 1000; st.t0 = null; },
   draw(t, st) { const a = SL[st.ci ?? 5], B = st.b ? B2 : B1, R = dtOf(B, st.b ? 2 : 1, a); if (st.key !== a + ':' + (st.b | 0)) { st.key = a + ':' + (st.b | 0); st.t0 = t; }
     maze({ t }); diaryTrails(B, .2); play(R.tr, t - st.t0 - .3, R.tr.length > 14 ? .22 : .45, BLUE);
     card([`${st.b ? 'third' : 'first'} box, asked ${String(a).replace('-', '−')}`, { s: `got ${f2(R.tot)} in ${R.tr.length} moves (${endWord(R.end)})`, c: BLUE }, { s: 'tap the maze to switch box', c: C.soft }], 1, { y: 296, gap: 0 }); } },
 { h: 'Where it came from', p: ['Lili Chen and colleagues introduced the **Decision Transformer** in 2021, alongside the similar Trajectory Transformer. On offline Atari and robot benchmarks, it matched that era’s offline RL methods.',
   'Later studies found stitching unreliable, as here, and that in worlds with chance it copies merely lucky moves: on a wet floor, a lucky slide carries a high return-to-go, so a high request copies it. Its appeal: training is plain, stable prediction.'],
   shot: [3, 10],
   draw(t) { notebook(60, 80, 1.6); txt('logged games', 90, 140, { s: 14, c: C.soft });
     alpha(clamp(t - 1.5), () => { arrow(150, 80, 220, 80, 2100, { c: C.ink }); rrect(230, 50, 120, 60, 2101, { fill: '#e3eef8' }); txt('transformer', 290, 74, { s: 16 }); txt('predicts the move', 290, 94, { s: 13, c: C.soft }); });
     card(['2021: Decision Transformer (Chen et al.)', '2021: Trajectory Transformer (Janner et al.)', { s: 'later: stitching and luck found to be weak spots', c: C.soft }], t - 4, { y: 200, gap: 1.1 }); } },
 { h: 'Recap', p: ['A **Decision Transformer** treats RL as sequence prediction: write each logged step as return-to-go, state and action, and learn to predict the action.',
   'At play time, ask for a high return and subtract rewards as they arrive. It can join pieces of walks, but less reliably than value learning, and it cannot use pieces the data never had.'],
   shot: [3, 9],
   draw(t) { maze({ t }); pathLine(D65.tr, BLUE, 2.6); pip(cx(4) - 18, cy(0) + 4, 1, t);
     card(['tokens: return-to-go, cell, move', 'train: predict the logged move', 'play: ask for a total, subtract rewards', { s: 'weak at stitching; limited by its data', c: C.goldInk }], t - .3, { y: 290, gap: 1 }); } },
];

RLStory.register('dt', { title: 'Decision Transformer', track: 'Learning from people', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { box1: { bumps: BWB, cheese: B1.cheese, cat: B1.cat, best: B1.best, bestLen: BW.length }, d65: { tot: D65.tot, len: D65.tr.length, first: D65.tr[0].ex.rtg, m0: MV[D65.tr[0].m], src: SRC.length },
    d0: { tot: D0.tot, len: D0.tr.length }, dlo: { tot: DLO.tot, end: DLO.end }, ask1: GOT[ASKS.indexOf(1)],
    box0: { cheese: B0.cheese, cat: B0.cat, dt: E0.tot, q: Q0.tot }, box2: { best: B2.best, dt: E2.tot, len: E2.tr.length, q: Q2.tot, qlen: Q2.tr.length, div: DIV2, divCell: E2.tr[DIV2] && no(E2.tr[DIV2].c, E2.tr[DIV2].r), divMove: E2.tr[DIV2] && MV[E2.tr[DIV2].m] },
    q1: Q1.tot, slider: SL.map(a => [a, runDT(B1, a).tot, runDT(B2, a).tot]), offDiag: ASKS.filter((a, i) => a >= -0.5 && a <= .65 && Math.abs(GOT[i] - a) > .001), manyFn: () => { while (!manyStep(1000)); return many; } } });
})();
