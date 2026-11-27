// Curriculum learning as a story: a bigger maze where the cheese is 12 steps away and a walk is stopped after 30 steps.
// Pip learns by Q-learning; without help it rarely finds the cheese, so it rarely learns. With a curriculum it starts close
// to the cheese and moves farther away lesson by lesson. All numbers come from the seeded runs below (seeds 1..50).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const ROCKS = [[2, 0], [2, 1], [2, 2], [2, 3], [5, 2], [5, 3], [5, 4], [5, 5]];
const G = L.gridWorld({ cols: 8, rows: 6, cs: 44, ox: 24, oy: 12, cheese: [7, 0], cat: null, rocks: ROCKS, start: [0, 5] });
const { cx, cy, key, maze, trail, pip } = G, NCELL = 48;
const GAM = 0.9, ALPHA = 0.2, EPS = 0.2, MAXS = 30, NW = 300, NS = 50;
// moves table and distances (steps to the cheese, worked out by us from the map; Pip does not know them)
const NEXT = new Int16Array(NCELL * 4); for (let r = 0; r < 6; r++) for (let c = 0; c < 8; c++) for (let m = 0; m < 4; m++) { const [nc, nr] = G.step(c, r, m); NEXT[key(c, r) * 4 + m] = key(nc, nr); }
const CHK = key(7, 0), STK = key(0, 5);
const DIST = (() => { const d = new Array(NCELL).fill(-1); d[CHK] = 0; const q = [CHK]; while (q.length) { const k = q.shift(); for (let m = 0; m < 4; m++) { const n = NEXT[k * 4 + m]; if (d[n] < 0) { d[n] = d[k] + 1; q.push(n); } } } return d; })();
const LESSON = [[4, 0], [3, 2], [3, 5]];                              // lesson starts: 3, 6 and 9 steps from the cheese, then the real start
// plan: list of [start key, number of walks]; after the list, the real start
const plan = (n, starts = LESSON) => starts.map(([c, r]) => [key(c, r), n]);
const FREE = []; for (let k = 0; k < NCELL; k++) if (k !== CHK && !G.isRock(k % 8, Math.floor(k / 8))) FREE.push(k);
function startOf(P, e, RS) { if (P === 'rand') return FREE[Math.floor(RS() * FREE.length)]; let acc = 0; for (const [k, n] of P) { acc += n; if (e < acc) return k; } return STK; }
// Q-learning: eps-greedy, ties at random; update on landing with the biggest number where Pip landed (0 at the cheese).
// After every walk: test from the real start, following the biggest numbers, no random moves (fails if all equal or > 30 steps).
function run(seed, P, total = NW, rec = false) {
  const R = rng(seed), RS = rng(seed + 7000), Q = new Float64Array(NCELL * 4), out = [];   // RS: start cells for the random-start baseline only
  for (let e = 0; e < total; e++) {
    let s = startOf(P, e, RS), ok = 0, k; const s0 = s; const tr = rec ? [] : null;
    for (k = 0; k < MAXS; k++) {
      let m; if (R() < EPS) m = Math.floor(R() * 4); else { let b = -1e9, ms = []; for (let j = 0; j < 4; j++) { const q = Q[s * 4 + j]; if (q > b + 1e-12) { b = q; ms = [j]; } else if (q > b - 1e-12) ms.push(j); } m = ms[Math.floor(R() * ms.length)]; }
      const n = NEXT[s * 4 + m], rew = n === CHK ? 1 : 0, best = rew ? 0 : Math.max(Q[n * 4], Q[n * 4 + 1], Q[n * 4 + 2], Q[n * 4 + 3]);
      Q[s * 4 + m] += ALPHA * (rew + GAM * best - Q[s * 4 + m]);
      if (rec) tr.push({ c: s % 8, r: Math.floor(s / 8), nc: n % 8, nr: Math.floor(n / 8) }); s = n; if (rew) { ok = 1; k++; break; }
    }
    let g = STK, pass = 0; for (let i = 0; i < 30; i++) { let b = -1e9, bm = 0; for (let j = 0; j < 4; j++) if (Q[g * 4 + j] > b + 1e-12) { b = Q[g * 4 + j]; bm = j; } if (b <= 0) break; g = NEXT[g * 4 + bm]; if (g === CHK) { pass = 1; break; } }
    const w = { ok, pass, start: s0, len: k }; if (rec) { w.tr = tr; w.V = Array.from({ length: NCELL }, (_, i) => Math.max(Q[i * 4], Q[i * 4 + 1], Q[i * 4 + 2], Q[i * 4 + 3])); w.Q = Array.from(Q); } out.push(w);
  }
  return out;
}
// learning curves (share of the 50 runs passing the test after each walk), memoised per plan
const CURVES = {}; function curve(id, P) { if (CURVES[id]) return CURVES[id]; const cv = new Float64Array(NW), solved = [];
  for (let s = 1; s <= NS; s++) { const o = run(s, P); o.forEach((w, i) => { cv[i] += w.pass / NS; }); } return (CURVES[id] = cv); }
const P_NONE = [], P_CUR = plan(20), P_JUMP = [[key(4, 0), 60]], P_SLOW = [[key(4, 0), 120], [key(3, 2), 20], [key(3, 5), 20]];
const cRand = () => curve('rand', 'rand'), allBy = cv => { for (let i = 0; i < NW; i++) if (Array.prototype.every.call(cv.slice(i), v => v > .999)) return i + 1; return null; };   // first walk from which all 50 runs pass
const cNone = () => curve('none', P_NONE), cCur = () => curve('cur', P_CUR), cJump = () => curve('jump', P_JUMP), cSlow = () => curve('slow', P_SLOW);
const pct = v => Math.round(v * NS);                                // runs out of 50
// story runs (seed STORY): without and with the curriculum
const STORY = 25, SN = run(STORY, P_NONE, NW, true), SC = run(STORY, P_CUR, NW, true);
const SN1 = SN.findIndex(w => w.ok), SN50 = SN.slice(0, 50).filter(w => w.ok).length, SNOK = SN.filter(w => w.ok).length, SNPASS = SN.findIndex((w, i) => SN.slice(i).every(x => x.pass));
const SC1 = SC[0], SCL1 = SC.slice(0, 20).filter(w => w.ok).length, SCPASS = SC.findIndex((w, i) => SC.slice(i).every(x => x.pass));
const SCL = [0, 1, 2].map(l => SC.slice(l * 20, l * 20 + 20).filter(w => w.ok).length), SCREAL = SC.slice(60, 100).filter(w => w.ok).length;
const f2 = v => (Math.abs(v) < .005 ? '0' : String(+v.toFixed(2)));
const ZV = new Array(NCELL).fill(0);

// ───── drawing
function startFlag(c, r, label, a = 1, col = C.roseInk) { if (a <= 0) return; alpha(a, () => { L.rcircle(cx(c), cy(r), 15, 900 + c * 7 + r, { c: col, w: 1.6 }); txt(label, cx(c), cy(r) + (r === 5 ? -20 : 19), { s: 12, c: col }); }); }
function bestArrows(Q, a = 1) { for (let r = 0; r < 6; r++) for (let c = 0; c < 8; c++) { const k = key(c, r); if (k === CHK || G.isRock(c, r)) continue; let b = -1e9, bm = 0; for (let j = 0; j < 4; j++) if (Q[k * 4 + j] > b + 1e-12) { b = Q[k * 4 + j]; bm = j; } if (b <= 0) continue; alpha(a * .8, () => G.arrowIn(c, r, bm, 400 + k)); } }
// a strip: one bar per walk (teal = reached the cheese, light = stopped after 30 steps)
function strip(W, n, label, o = {}) {
  const x0 = 40, x1 = 360, y0 = 388, y1 = 318, w = (x1 - x0) / NW;
  txt(label, x0, y1 - 14, { s: 14, al: 'left', c: C.soft }); rline(x0, y0, x1, y0, 900, { c: C.line, w: 1 });
  for (let i = 0; i < Math.min(n, NW); i++) box(x0 + i * w, y1, Math.max(1, w - .15), y0 - y1, W[i].ok ? C.teal : '#ece3d6');
  txt('walk 1', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NW}`, x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
  if (o.lessons) [20, 40, 60].forEach((e, i) => { const x = x0 + e * w; rline(x, y1 - 4, x, y0, 910 + i, { c: C.ink, w: 1.2, once: true }); });
  if (o.lessons) ['3', '6', '9'].forEach((s, i) => txt(s, x0 + (i * 20 + 10) * w, y1 - 2, { s: 11, c: C.ink }));
}
// learning-curve chart: share of 50 runs whose best-number route from the real start reaches the cheese
function curveChart(lines, n, o = {}) {
  const x0 = 56, x1 = 356, y0 = 390, y1 = 310, X = i => x0 + i * (x1 - x0) / (NW - 1), Y = v => lerp(y0, y1, v);
  rline(x0 - 4, y0, x1 + 4, y0, 950, { c: C.line, w: 1 }); rline(x0 - 4, Y(1), x1 + 4, Y(1), 951, { c: '#e4d8c7', w: 1, once: true });
  txt('50', x0 - 14, Y(1), { s: 12, c: C.soft }); txt('0', x0 - 12, y0, { s: 12, c: C.soft }); txt('25', x0 - 14, Y(.5), { s: 12, c: C.soft });
  txt(o.title || 'runs (of 50) that can do the real task', x0 - 34, y1 - 14, { s: 14, al: 'left', c: C.soft });
  txt('walk 1', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NW}`, x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
  lines.forEach(([cv, col, lab, ly], li) => { ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); for (let i = 0; i < Math.min(n, NW); i++) i ? ctx.lineTo(X(i), Y(cv[i])) : ctx.moveTo(X(0), Y(cv[0])); ctx.stroke(); ctx.restore();
    if (lab && n >= NW) txt(lab, x1 - 2, ly ?? Y(cv[NW - 1]) - 9, { s: 13, al: 'right', c: col }); });
  if (o.mark != null) { rline(X(o.mark), y0, X(o.mark), y1, 960, { c: C.ink, w: 1, once: true }); }
}
const LT = [[4, 0, '3 away'], [3, 2, '6 away'], [3, 5, '9 away']];

const STEPS = [
 { h: 'A long way to the cheese', p: ['A bigger maze: 8 cells wide, 6 high, two walls of rocks, no cat. Cheese top right, Pip bottom left, shortest route **12 steps**.', '+1 for the step onto the cheese, which ends the walk; γ (gamma) 0.9. A walk without cheese after 30 steps is stopped and pays nothing. That is not a real ending: the last update still uses the landing number.'],
   draw(t) { maze({ t, draw: clamp(t / 1.5) }); if (t > 1) pip(cx(0), cy(5), 1, t); txt('+1', cx(7) - 10, cy(0) - 12, { s: 14, c: '#b07d12', a: clamp(t - 1.5) }); txt('start', cx(0), cy(5) + 16, { s: 12, c: C.soft, a: clamp(t - 1.5) });
     if (t > 3) for (let k = 0; k < NCELL; k++) { const c = k % 8, r = Math.floor(k / 8); if (DIST[k] > 0 && !G.isRock(c, r)) txt(String(DIST[k]), G.ox + c * 44 + 39, G.oy + r * 44 + 10, { s: 12, al: 'right', c: C.soft, a: clamp(t - 3) }); }
     card(['+1 on arrival at the cheese: walk over', 'no cheese after 30 steps: stopped, 0', { s: 'small numbers: steps to cheese, unknown to Pip', c: C.soft }], t - 4.5, { y: 290, gap: 1.2 }); } },
 { h: 'How Pip learns', p: ['Pip uses **Q-learning**, as in its reel: a number per move per cell, nudged after each step toward the reward plus 0.9 times the best number where Pip landed. It mostly takes the biggest; one move in five is random.', 'All numbers start at 0, so until Pip first reaches the cheese, every nudge is 0 toward 0: it learns **nothing** and just wanders.'],
   draw(t) { maze({ t, V: ZV }); pip(cx(0), cy(5), 1, t);
     card(['target = reward + 0.9 × best number where Pip lands', 'number moves 0.2 of the way to the target', { s: 'no reward yet → all numbers stay 0', c: C.roseInk }], t - 1, { y: 290, gap: 1.3 }); } },
 { h: 'Wandering in the dark', p: [`Walk 1, from the real start. With every number 0, each move is a random pick among four. After 30 steps Pip has not found the cheese: stopped, reward 0, and nothing learned.`, `It goes on like that. ${SN50 === 0 ? 'In its first 50 walks, Pip never reached the cheese' : `In its first 50 walks, Pip reached the cheese ${SN50 === 1 ? 'once' : SN50 + ' times'}`}; the first time was walk ${SN1 + 1}.`],
   shot: [5, 13],
   draw(t) { const tr = SN[0].tr, dur = .32, m = G.walkPos(tr, 0, tr.length, t, dur); maze({ t, V: ZV }); trail(tr.slice(0, Math.min(tr.length, m.k + 1)).map(s => [s.c, s.r]), .55); pip(m.x, m.y, m.face, t);
     caption(m.done ? 'walk 1: 30 steps, no cheese, stopped' : `walk 1, step ${m.k + 1} of 30`); } },
 { h: 'Rare rewards, slow learning', p: ['The strip shows Pip’s first 300 walks, one bar each: teal if it reached the cheese, pale if stopped. Gold shading shows each cell’s best number.', `Each success pushes the good news only about one move further back, so the next is still mostly luck. Pip reached the cheese ${SNOK} times in 300 walks; only from walk ${SNPASS + 1} did its best numbers lead home.`],
   draw(t) { const n = clamp(Math.floor(t / 8 * NW), 1, NW), w = SN[n - 1]; maze({ t, V: w.V }); bestArrows(w.Q, .9); pip(cx(0), cy(5), 1, t);
     strip(SN, n, 'no curriculum: each walk, did Pip reach the cheese?'); caption(`after walk ${n}`, 0); } },
 { h: 'Start easy', p: ['A **curriculum** is a list of lessons, from easy to hard. Lesson 1: Pip starts only 3 steps from the cheese, for 20 walks. Lesson 2: 6 steps away, 20 walks. Lesson 3: 9 away, 20 walks. Then the real start, 12 away.', 'Only the starting cell changes. The maze, the rules and the way Pip learns stay exactly the same.'],
   draw(t) { maze({ t, V: ZV }); LT.forEach(([c, r, s], i) => startFlag(c, r, `lesson ${i + 1}`, clamp((t - .6 - i * 1.2) * 2)));
     startFlag(0, 5, 'then here', clamp((t - 4.2) * 2), C.goldInk); pip(cx(4), cy(0), 1, t);
     card(['lesson 1: 3 steps away · 20 walks', 'lesson 2: 6 away · lesson 3: 9 away', { s: 'then the real start, 12 away', c: C.soft }], t - 5.5, { y: 290, gap: 1.2 }); } },
 { h: 'Lesson 1', p: [`Curriculum walk 1 starts 3 steps from the cheese, same seed as before. Even at random the cheese is close: Pip finds it in ${SC1.len} steps, and the last move gets the first non-zero number.`, `In lesson 1, ${SCL1} of 20 walks reached the cheese. The gold shading shows a trail of good numbers growing back from the cheese.`],
   shot: [4, 13],
   draw(t) { const tr = SC[0].tr, dur = .5, m = G.walkPos(tr, 0, tr.length, t, dur), tEnd = tr.length * dur;
     if (t < tEnd + 1.5) { maze({ t, V: ZV }); trail(tr.slice(0, m.k + 1).map(s => [s.c, s.r]), .55); pip(m.x, m.y, m.face, t); if (m.done) bubble('+1', cx(7), cy(0) - 14, t - tEnd); caption(`lesson 1, walk 1, step ${Math.min(tr.length, m.k + 1)}`); }
     else { const n = clamp(Math.floor((t - tEnd - 1.5) / .4) + 1, 1, 20), w = SC[n - 1]; maze({ t, V: w.V }); bestArrows(w.Q, .9); pip(cx(4), cy(0), 1, t); caption(`lesson 1, after walk ${n} of 20`); } } },
 { h: 'Each lesson builds on the last', p: [`Lesson 2 starts 6 steps away. Pip still wanders, but only has to stumble into the gold trail; then the big numbers lead it home. ${SCL[1]} of 20 walks succeeded, then ${SCL[2]} of 20 in lesson 3.`, `Then the real start: in its next 40 walks Pip reached the cheese ${SCREAL} times. From walk ${SCPASS + 1}, its best numbers led home from the real start.`],
   draw(t) { const n = clamp(Math.floor(t / 10 * 100) + 1, 1, 100), w = SC[n - 1]; maze({ t, V: w.V }); bestArrows(w.Q, .9);
     const l = n <= 20 ? 0 : n <= 40 ? 1 : n <= 60 ? 2 : 3, sc = l < 3 ? LT[l] : [0, 5]; startFlag(sc[0], sc[1], l < 3 ? `lesson ${l + 1}` : 'real start', 1, l < 3 ? C.roseInk : C.goldInk); pip(cx(sc[0]), cy(sc[1]), 1, t);
     caption(`curriculum, after walk ${n}`); } },
 { h: 'The same strip, with lessons', p: ['The same kind of strip, for the curriculum run. The black ticks mark the ends of lessons 1, 2 and 3; after that, every walk starts at the real start.', `Teal from the very beginning: the reward is no longer rare. Without the curriculum, the same seed gave ${SNOK} cheese walks in 300; with it, ${SC.filter(w => w.ok).length}.`],
   draw(t) { const n = clamp(Math.floor(t / 8 * NW), 1, NW), w = SC[n - 1]; maze({ t, V: w.V }); bestArrows(w.Q, .9); pip(cx(0), cy(5), 1, t);
     strip(SC, n, 'curriculum: each walk, did Pip reach the cheese?', { lessons: 1 }); } },
 { h: 'Fifty runs, same seeds', get p() { const a = cNone(), b = cCur(), c = cRand(); return [`One seed could be luck, so we repeat with seeds 1 to 50, testing after every walk: from the real start, following its biggest numbers, does Pip reach the cheese? Lines count passing runs; lesson walks count too.`,
   `After 100 walks: ${pct(a[99])} of 50 without, ${pct(b[99])} with the curriculum; after 200, ${pct(a[199])} against ${pct(b[199])}. Grey, another easy fix, starts each walk in a random cell: ${pct(c[99])}, then ${pct(c[199])}.`]; },
   shot: [5, 12],
   draw(t) { const n = clamp(Math.floor((t - 1) / 7 * NW), 1, NW); maze({ t, V: SC[NW - 1].V }); bestArrows(SC[NW - 1].Q, .5);
     curveChart([[cNone(), C.rose, 'no curriculum'], [cRand(), '#8a8174', 'random start', 334], [cCur(), C.teal, 'curriculum']], n); } },
 { h: 'Why it works', p: ['Learning needs rewards. On the hard task, rewards were so rare that most walks taught nothing. Easy lessons pay off almost every time, and what Pip learns there is exactly what it needs later: the end of the route.', 'Compare reward shaping: shaping changes the **rewards** to give hints. A curriculum keeps the rewards and changes the **tasks**.'],
   draw(t) { maze({ t, V: SC[60].V }); bestArrows(SC[60].Q, .8); startFlag(0, 5, 'real start', 1, C.goldInk); pip(cx(0), cy(5), 1, t);
     card(['hard task: rewards rare, little to learn from', 'easy lessons: rewards often, a trail home', { s: 'curriculum: change the tasks, not the rewards', c: C.soft }], t - 1, { y: 290, gap: 1.2 }); } },
 { h: 'Order matters', get p() { const a = cNone(), b = cCur(), j = cJump(), s = cSlow(); return [`Bad pacing is slower. **Too big a jump**: 60 walks 3 steps away, then straight to the real start. After 100 walks ${pct(j[99])} runs pass, against ${pct(b[99])}; all 50 pass from walk ${allBy(j)}, against ${allBy(b)}.`,
   `**Too slow**: 120 walks on lesson 1. No run can pass before meeting the real start, so the fair measure is when all 50 pass: walk ${allBy(s)}. The extra walks went on a skill Pip already had.`]; },
   draw(t) { const n = clamp(Math.floor((t - .5) / 6 * NW), 1, NW); maze({ t, V: ZV, draw: 1 }); dimTop();
     curveChart([[cNone(), C.rose], [cCur(), C.teal], [cJump(), C.orange], [cSlow(), C.blue]], n, { title: 'runs (of 50) that can do the real task' });
     card([{ s: 'gradual: 3, 6, 9 away, 20 walks each', c: '#3f7f7c' }, { s: 'jump: 60 walks 3 away, then the real start', c: C.orange }, { s: 'too slow: 120 walks 3 away, then 6, 9', c: '#4f74a0' }, { s: 'none: always the real start', c: C.roseInk }], t - .3, { y: 40, x: 40, w: 320, gap: 0, fs: 16, lh: 24 }); } },
 { h: 'Your turn', ctl: { label: 'walks per lesson', values: [0, 5, 10, 20, 40, 80], def: 3, fmt: v => String(v), onChange: () => {} },
   p: ['Drag the number of **walks per lesson**: the three lessons, 3, 6 and 9 steps away, each get that many walks before the real start. 0 means no curriculum. All 50 runs are redone; the rose line is always no curriculum.', 'Too few walks per lesson and the trail is not ready; too many and walks are wasted.'],
   draw(t, st) { const v = [0, 5, 10, 20, 40, 80][st.ci ?? 3], cv = curve('L' + v, v ? plan(v) : P_NONE), a = cNone();
     maze({ t, V: ZV }); LT.forEach(([c, r, s], i) => startFlag(c, r, '', v ? 1 : .25)); startFlag(0, 5, '', 1, C.goldInk); pip(cx(0), cy(5), 1, t); dimTop();
     const at = i => pct(cv[i]);
     card([v ? `curriculum: ${v} walks in each of 3 lessons` : 'no curriculum: always the real start', `runs passing after 100 walks: ${at(99)} of 50`, `after 200 walks: ${at(199)} of 50`, { s: `no curriculum: ${pct(a[99])} and ${pct(a[199])}`, c: C.roseInk }], 10, { y: 70, gap: 0 });
     curveChart([[a, C.rose], [cv, C.teal]], NW, { mark: v ? 3 * v : null }); } },
 { h: 'Automatic curricula', p: ['Hand-written lessons take knowledge and trial and error. Instead, a **teacher** program can pick the next lesson, like start cells where Pip now succeeds about half the time: not hopeless, not boring. Pip’s recent walks from each start tell it, without extra tests.', 'Self-play, from the multi-agent reel, is a curriculum too: the opponent gets harder exactly as fast as the learner improves.'],
   draw(t) { maze({ t, V: SC[50].V }); bestArrows(SC[50].Q, .6); L.robot(352, 250, 1.2, t);
     card(['teacher: pick tasks with about 50% success', 'harder as the learner improves', { s: 'self-play does this by itself', c: C.soft }], t - 2, { y: 290, gap: 1.2 }); } },
 { h: 'Where it is used', p: ['In 2009 Yoshua Bengio and colleagues named machine **curriculum learning**, inspired by how people are taught. In 2017 Carlos Florensa and colleagues trained simulated robots by starting them near the goal and moving the start farther away, as here.', 'OpenAI’s 2019 Rubik’s cube hand widened its randomised simulation as it improved: automatic domain randomisation, an automatic curriculum.'],
   draw(t) { maze({ t, V: SC[NW - 1].V }); bestArrows(SC[NW - 1].Q, .5);
     card(['2009  Bengio et al.: “curriculum learning”', '2017  Florensa et al.: start near the goal', '2019  OpenAI: harder simulation as skill grows'], t - .5, { y: 292, gap: 1.2 }); } },
 { h: 'Recap', p: ['When rewards are rare, an agent must stumble on success before any learning starts. A **curriculum** starts with easy versions of the task and makes them harder step by step.', 'Easy lessons pay often and build skills the hard task needs. Order and pace matter: too big a jump or too slow both waste practice. A teacher can choose the lessons automatically.'],
   draw(t) { maze({ t, V: SC[NW - 1].V }); bestArrows(SC[NW - 1].Q, .8); pip(cx(0), cy(5), 1, t);
     card(['rare rewards: little to learn from', 'curriculum: easy tasks first, then harder', 'each lesson builds on the last', 'pace matters; teachers can pick lessons'], t - .3, { y: 284, gap: 1, lh: 22 }); } },
];
function dimTop() { box(0, 0, W, 285, C.bg, .6); }
RLStory.register('curriculum', { title: 'Curriculum Learning', track: 'Bigger worlds', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { run, P_CUR, DIST, SN, SC, SN1, SN50, SNOK, SNPASS, SC1, SCL1, SCL, SCREAL, SCPASS, cNone, cCur, cJump, cSlow, curve, plan, P_NONE } });
})();
