// Sim-to-real as a story: Pip practises on a computer copy of a running track (dry floor), then meets the real, polished
// floor, where a run sometimes skids one cell further. Domain randomisation: practise on many floors with random slip chances.
// All numbers come from the exact scores and the seeded test walks below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
// ───── the track: cells 0..7, Pip starts at 0, cheese on 6 (+1), cat on 7 (−1; a move past the cheese ends there). γ = 0.9.
// Each step Pip walks (1 cell) or runs (2 cells). On a slippery floor a run skids one cell further with chance s, and the skid
// costs one extra step (Pip slides, then gets back on its feet). Walking never skids.
const GAM = 0.9, CH = 6, CAT = 7, NC = 6;
function value(h, s, c0) {                       // exact average return from cell c0 with habit h (h[c] = 1 run, 0 walk)
  const v = new Array(8).fill(0);
  for (let c = 5; c >= 0; c--) { const outs = h[c] ? [[c + 2, 1 - s, 1], [c + 3, s, 2]] : [[c + 1, 1, 1]]; let x = 0;
    for (let [n, p, k] of outs) { n = Math.min(n, CAT); const r = n === CH ? 1 : n >= CAT ? -1 : 0; x += p * GAM ** (k - 1) * (r + (r ? 0 : GAM * v[n])); } v[c] = x; }
  return v[c0];
}
const HAB = []; for (let m = 0; m < 64; m++) HAB.push([0, 1, 2, 3, 4, 5].map(i => (m >> i) & 1));
const nRuns = h => h.reduce((a, b) => a + b, 0);
// practice: walks start in random cells; score = average return over start cells 0..5 and over the floors practised on.
// Keep the best habit; when two are equally good, the one with more runs (Pip runs when walking is no better).
function practise(floors) {
  let best = null, bs = -9;
  for (const h of HAB) { const sc = floors.reduce((a, s) => a + [0, 1, 2, 3, 4, 5].reduce((b, c) => b + value(h, s, c), 0) / 6, 0) / floors.length;
    if (sc > bs + 1e-9 || (Math.abs(sc - bs) < 1e-9 && nRuns(h) > nRuns(best))) { bs = sc; best = h; } }
  return best;
}
const DRY = practise([0]), RANGE = [0, .1, .2, .3, .4], DR = practise(RANGE);
const THR = (() => { for (let i = 1; i <= 400; i++) if (practise([i / 1000]).join('') === DR.join('')) return i / 1000; return null; })();   // smallest single slip chance (0.001 grid) whose best habit is the safe one
const WIDE = practise([0, .1, .2, .3, .4, .5, .6, .7, .8, .9]), WIDE_SAME = WIDE.join('') === DR.join('');
// a walk on a floor with slip chance s. Returns transitions [{ c, n, run, skid }] and the return
function walk(h, s, R) { let c = 0, k = 0; const tr = [];
  while (c < CH) { const run = !!h[c], skid = run && R() < s, n = Math.min(CAT, c + (run ? 2 : 1) + (skid ? 1 : 0)); k += skid ? 2 : 1; tr.push({ c, n, run, skid }); c = n; }
  return { tr, end: c === CH ? 1 : -1, ret: (c === CH ? 1 : -1) * GAM ** (k - 1), steps: k }; }
// real-floor tests: 1000 walks per habit per slip chance, dice rng(5000 + 10 s)
const SL = [0, .1, .2, .3, .4, .5, .6], NT = 1000;
const TESTS = {}; function test(h, s) { const id = h.join('') + s; if (TESTS[id]) return TESTS[id]; const R = rng(5000 + Math.round(s * 10)), W = [];
  for (let i = 0; i < NT; i++) W.push(walk(h, s, R)); const cat = W.filter(w => w.end < 0).length, avg = W.reduce((a, w) => a + w.ret, 0) / NT;
  return (TESTS[id] = { W, cat, cheese: NT - cat, avg }); }
const TD2 = test(DRY, .2), TR2 = test(DR, .2), TD0 = test(DRY, 0), TR0 = test(DR, 0);
const SKIDW = TD2.W.findIndex(w => w.end < 0 && w.tr.length === 3 && w.tr[2].skid);     // run, run, then a skid onto the cat
const RW2 = TR2.W.findIndex(w => w.tr.some(t => t.skid));                                  // a randomised-habit walk with a skid
const EX = { dry0: value(DRY, 0, 0), dr0: value(DR, 0, 0), dry2: value(DRY, .2, 0), dr2: value(DR, .2, 0) };
const f2 = v => (Math.abs(v) < .005 ? '0' : String(+v.toFixed(2)));

// ───── drawing
const X0 = 28, CW = 43, cxx = c => X0 + c * CW + CW / 2;
function chevron(x, y, n, col, a = 1) { alpha(a, () => { for (let k = 0; k < n; k++) { const dx = (k - (n - 1) / 2) * 8; rline(x + dx - 3, y - 5, x + dx + 2, y, 500 + k, { c: col, w: 1.8, once: true, j: .3 }); rline(x + dx + 2, y, x + dx - 3, y + 5, 510 + k, { c: col, w: 1.8, once: true, j: .3 }); } }); }
// track at height y (cells y..y+44). o = { slip: sheen 0..1, sim: true for the computer copy, habit, habA, label, a, nums }
function track(y, o = {}) {
  const a = o.a ?? 1; if (a <= 0) return;
  alpha(a, () => {
    const ly = o.habit ? y - 30 : y - 16;
    if (o.sim) { box(X0 - 12, ly - 12, 8 * CW + 24, y + 58 - ly, '#e3eef6'); rrect(X0 - 12, ly - 12, 8 * CW + 24, y + 58 - ly, 600 + y, { c: '#7d9bb8', w: 1.6 }); txt(o.label || 'simulator', X0 - 4, ly, { s: 13, al: 'left', c: '#4f74a0' }); }
    else if (o.label) txt(o.label, X0 - 4, ly, { s: 13, al: 'left', c: C.soft });
    for (let c = 0; c < 8; c++) { const x = X0 + c * CW; box(x, y, CW, 44, c === CAT ? '#f3dede' : c === CH ? '#fbefcf' : '#f6eee2');
      if (o.slip) L.hachure(x, y, CW, 44, '#8fb3d9', lerp(14, 6, o.slip), 700 + c, .25 + .5 * o.slip); }
    for (let k = 0; k <= 8; k++) rline(X0 + k * CW, y, X0 + k * CW, y + 44, 620 + k, { c: k % 8 ? C.line : C.ink, w: k % 8 ? 1 : 1.6 });
    rline(X0, y, X0 + 8 * CW, y, 640, { w: 1.6 }); rline(X0, y + 44, X0 + 8 * CW, y + 44, 641, { w: 1.6 });
    L.cheese(cxx(CH), y + 26, .62); L.cat(cxx(CAT), y + 26, o.t || 0, .66);
    if (o.nums !== 0) for (let c = 0; c < 8; c++) txt(String(c), X0 + c * CW + 6, y + 9, { s: 12, c: C.soft, al: 'left' });
    if (o.habit) o.habit.forEach((r, c) => chevron(cxx(c), y - 9, r ? 2 : 1, r ? C.goldInk : '#3f7f7c', o.habA ? o.habA(c) : 1));
  });
}
// Pip's position along transitions tr: each move takes 0.9 s, a skid another 0.9 s
const DUR = .9;
function walkPos(tr, t) {
  let tt = t; for (let i = 0; i < tr.length; i++) { const s = tr[i], base = s.c + (s.run ? 2 : 1), len = DUR * (s.skid ? 2 : 1);
    if (tt < len) { if (tt < DUR) { const u = ease(clamp(tt / (DUR * .8))); return { x: cxx(lerp(s.c, Math.min(base, CAT), u)), hop: s.run ? Math.sin(Math.PI * u) * 16 : 0, i, skidding: false }; }
      const u = ease(clamp((tt - DUR) / (DUR * .8))); return { x: cxx(lerp(Math.min(base, CAT), s.n, u)), hop: 0, i, skidding: true, wob: Math.sin(tt * 30) * (1 - u) * .3 }; }
    tt -= len; }
  const l = tr[tr.length - 1]; return { x: cxx(l.n), hop: 0, i: tr.length, done: true };
}
function pipOn(y, P, t) { ctx.save(); ctx.translate(P.x, y + 28 - P.hop); if (P.wob) ctx.rotate(P.wob); L.mouse(0, 0, 1, .62, t); ctx.restore(); }
function walkDur(tr) { return tr.reduce((a, s) => a + DUR * (s.skid ? 2 : 1), 0); }
// 100 dots: the first 100 test walks (gold = cheese, rose = cat)
function dots100(W, x0, y0, n, a = 1) { for (let i = 0; i < Math.min(n, 100); i++) dot(x0 + (i % 20) * 14, y0 + Math.floor(i / 20) * 14, 5, W[i].end > 0 ? C.gold : C.rose, a); }
function habitLegend(y, a) { alpha(a, () => { chevron(60, y, 1, '#3f7f7c'); txt('walk: 1 cell', 72, y, { s: 14, al: 'left' }); chevron(190, y, 2, C.goldInk); txt('run: 2 cells', 206, y, { s: 14, al: 'left' }); }); }
// chart of the average real return for both habits across slip chances (1000 test walks each)
function sweepChart(n, sel, a = 1) {
  const x0 = 62, x1 = 352, y0 = 392, y1 = 306, X = i => x0 + i * (x1 - x0) / (SL.length - 1), Y = v => lerp(y0, y1, (v - .4) / .5);
  alpha(a, () => {
    rline(x0 - 6, y0, x1 + 6, y0, 960, { c: C.line, w: 1 }); [.5, .9].forEach(v => txt(String(v), x0 - 18, Y(v), { s: 12, c: C.soft }));
    rline(x0 - 6, Y(.9), x1 + 6, Y(.9), 961, { c: '#e4d8c7', w: 1, once: true });
    SL.forEach((s, i) => txt(String(s), X(i), y0 + 10, { s: 12, c: C.soft }));
    box(X(0) - 4, y1 - 4, X(4) - X(0) + 8, y0 - y1 + 4, C.teal, .1); txt('practised range', X(2), y0 - 8, { s: 12, c: '#3f7f7c' });
    [[DRY, '#c99a3a', 'dry habit'], [DR, C.teal, 'randomised habit']].forEach(([h, col], k) => { for (let i = 0; i < Math.min(n, SL.length); i++) { const v = test(h, SL[i]).avg; if (i) rline(X(i - 1), Y(test(h, SL[i - 1]).avg), X(i), Y(v), 970 + i + k * 20, { c: col, w: 1.6, once: true }); dot(X(i), Y(v), 3.4, col); } });
    if (sel != null) rline(X(sel), y0, X(sel), y1 - 6, 990, { c: C.ink, w: 1.2, once: true });
    txt('average return, 1000 real walks', x0 - 40, y1 - 14, { s: 14, al: 'left', c: C.soft });
    txt('dry', x1 - 104, y1 - 14, { s: 14, al: 'left', c: C.goldInk }); txt('randomised', x1 - 74, y1 - 14, { s: 14, al: 'left', c: '#3f7f7c' });
  });
}

const STEPS = [
 { h: 'Practice is expensive', p: ['Pip has a new job: a real running track, with cheese at the end. Practising on the real track is slow, and a mistake can be painful: the cat sleeps just past the cheese.', 'So the engineers build a **simulator**: a computer copy of the track. In it, Pip can practise thousands of walks in seconds, many copies at once, and nobody gets hurt.'],
   draw(t) { track(70, { t, label: 'the real track', a: clamp(t - .3) }); L.mouse(cxx(0), 98, 1, .62, t);
     track(196, { t, sim: true, a: clamp(t - 2) }); if (t > 2) L.mouse(cxx(0), 224, 1, .62, t);
     card(['real track: slow, risky, one at a time', 'simulator: fast, safe, thousands at once', { s: 'idea: practise in the copy, then go real', c: C.soft }], t - 3.5, { gap: 1.2 }); } },
 { h: 'The rules of the track', p: ['Cells are numbered 0 to 7; Pip starts on 0. Each step it either **walks** one cell or **runs** two. Arriving on the cheese, cell 6, pays +1. Going past it means landing on the cat, cell 7: −1. Either ends the walk.', 'Sooner is better: each step of delay multiplies the reward by 0.9, the discount γ (gamma). The result is the walk’s **return**: three runs give 0.9 × 0.9 = 0.81, six walks only 0.59.'],
   draw(t) { track(110, { t, nums: 1 }); habitLegend(70, clamp(t - 1));
     const tr = [{ c: 0, n: 2, run: 1 }, { c: 2, n: 4, run: 1 }, { c: 4, n: 6, run: 1 }], P = walkPos(tr, ((t - 2) % 4.5 + 4.5) % 4.5); if (t > 2) pipOn(110, P, t); else L.mouse(cxx(0), 138, 1, .62, t);
     if (t > 2) txt('+1', cxx(CH), 170, { s: 15, c: '#b07d12' }); if (t > 2.5) txt('−1', cxx(CAT), 170, { s: 15, c: C.roseInk });
     card(['run, run, run: 3 steps, return 0.81', 'walk six times: 6 steps, return 0.59', { s: 'past the cheese = the cat: −1, walk over', c: C.roseInk }], t - 4, { gap: 1.2 }); } },
 { h: 'A dry simulator', p: ['In the simulator the floor is perfectly dry: a run always goes exactly two cells. Practice walks start in random cells, so Pip learns what to do in every cell. Its **habit** is one choice per cell.', `The best habit for this floor: run everywhere, except next to the cheese, where a run would overshoot. When walking is no better, Pip runs. From cell 0 that is run, run, run: ${f2(EX.dry0)}.`],
   shot: [4, 11],
   draw(t) { track(110, { t, sim: true, habit: DRY, habA: c => clamp((t - 1.5 - c * .3) * 3) }); habitLegend(208, clamp(t - 1));
     const tr = walk(DRY, 0, rng(1)).tr, d = walkDur(tr) + 1.5, P = walkPos(tr, Math.max(0, (t - 4) % d)); pipOn(110, P, t);
     card(['the dry habit: run, run, run, run, run, walk', `from cell 0: 3 runs, return ${f2(EX.dry0)}`, { s: 'perfect, for this exact floor', c: C.soft }], t - 5, { gap: 1.2 }); } },
 { h: 'The real floor is polished', p: ['The real track is polished. When Pip runs, it sometimes **skids** one cell further and needs an extra step to get back on its feet. The lost step counts even when the skid lands on the cheese or the cat. Walking never skids. The chance of a skid, the **slip chance**, is not known exactly: the engineers only guess it is somewhere between 0 and 0.4.', 'The difference between the copy and the real thing is called the **reality gap**.'],
   draw(t) { track(70, { t, sim: true, nums: 0, label: 'simulator: dry' }); track(196, { t, slip: .7, label: 'real track: polished', nums: 0 });
     const tr = [{ c: 1, n: 4, run: 1, skid: 1 }], P = walkPos(tr, ((t - 1) % 4 + 4) % 4); pipOn(196, P, t); if (P.skidding) txt('skid!', P.x, 186, { s: 15, c: '#4f74a0' });
     const tr2 = [{ c: 1, n: 3, run: 1 }], P2 = walkPos(tr2, ((t - 1) % 4 + 4) % 4); pipOn(70, P2, t);
     card(['simulator: a run is always 2 cells', 'real: sometimes 3, and a step lost', { s: 'real robots: slipperier floors, delays, noisy sensors', c: C.soft }], t - 4, { gap: 1.2 }); } },
 { h: 'One skid, at the worst moment', p: [`Pip takes its dry habit onto the real track. Say the real slip chance is 0.2. Run, run: fine. From cell 4 it runs again, which in the simulator always landed exactly on the cheese.`, 'This time it skids one cell further, past the cheese, onto the cat. The habit never had to plan for a skid, because the simulator never made one.'],
   shot: [4.2, 11],
   draw(t) { const w = TD2.W[SKIDW], P = walkPos(w.tr, Math.max(0, t - 1)); track(140, { t, slip: .45, habit: DRY, label: 'real track, slip chance 0.2' }); pipOn(140, P, t);
     if (P.skidding) txt('skid!', P.x, 200, { s: 16, c: '#4f74a0' }); if (P.done) bubble('−1', cxx(CAT), 210, t - 1 - walkDur(w.tr), C.roseInk);
     card(['run: 0 → 2,   run: 2 → 4', 'run from 4: skid → 7, the cat', { s: 'in the simulator this run never failed', c: C.soft }], t - 2, { gap: 2 }); } },
 { h: 'A thousand real walks', p: [`One walk could be bad luck, so the engineers test 1000 walks at slip chance 0.2. Each dot is one of the first 100: gold reached the cheese, rose met the cat.`, `${TD2.cat} of the 1000 walks met the cat. The average return is ${f2(TD2.avg)}, far below the ${f2(EX.dry0)} the simulator promised.`],
   draw(t) { track(70, { t, slip: .45, habit: DRY, nums: 0, label: 'dry habit, real slip chance 0.2' }); const n = Math.floor((t - 1) * 14);
     dots100(TD2.W, 67, 160, n);
     card([`met the cat: ${TD2.cat} of ${NT}`, `average return: ${f2(TD2.avg)}`, { s: `the simulator promised ${f2(EX.dry0)}`, c: C.soft }], t - 8.5, { gap: 1.2 }); },
   shot: [5, 13] },
 { h: 'Tuned to one exact world', p: ['The dry habit is not stupid. It is perfect for the floor it practised on, and it squeezes every step out of it: it counts on each run landing exactly two cells on.', 'That is the trouble with a single, exact simulator. Whatever the copy gets slightly wrong, the habit may rely on. Engineers can try to make the copy more accurate, but it is never perfect.'],
   draw(t) { track(110, { t, sim: true, habit: DRY, label: 'simulator: dry' }); alpha(clamp(t - 1.5), () => { box(cxx(4) - 18, 104, 36, 56, C.yellow, .45); });
     card(['the run from cell 4: right only on a dry floor', 'squeezing out steps = relying on details', { s: 'details the real world may not share', c: C.soft }], t - 2.5, { gap: 1.2 }); } },
 { h: 'Randomise the simulator', p: ['A different idea: **domain randomisation**. Before each practice walk, the simulator picks a new slip chance at random: 0, 0.1, 0.2, 0.3 or 0.4. Pip is not told which floor it has.', 'Now the best habit is the one with the best average over all those floors, so it cannot rely on any single one. Here we find it by scoring all 64 possible habits on those floors, standing in for a learner.'],
   shot: [4, 11],
   draw(t) { RANGE.forEach((s, i) => { const y = 34 + i * 50, a = clamp((t - .4 - i * .5) * 3); if (a <= 0) return;
       alpha(a, () => { for (let c = 0; c < 8; c++) { const x = 92 + c * 32; box(x, y, 32, 32, c === CAT ? '#f3dede' : c === CH ? '#fbefcf' : '#f6eee2'); if (s) L.hachure(x, y, 32, 32, '#8fb3d9', lerp(14, 6, s / .4), 700 + c + i, .2 + s); }
         rrect(92, y, 256, 32, 800 + i, { c: '#7d9bb8', w: 1.3 }); txt(`slip ${s}`, 82, y + 16, { s: 14, al: 'right', c: '#4f74a0' }); L.cheese(92 + 6 * 32 + 16, y + 19, .45); }); });
     card(['each practice walk: a random floor', 'keep the habit with the best average', { s: 'the real floor is then just one more variation', c: C.soft }], t - 3.2, { gap: 1.2 }); } },
 { h: 'The randomised habit', p: [`The habit that wins: run until two cells from the cheese, then walk. From cell 4, a run might skid past the cheese, so it walks; from cell 3, a skid lands on the cheese, so running is safe.`, `On the dry floor it is slower, ${f2(EX.dr0)} instead of ${f2(EX.dry0)}: one extra step. That is the price of caution. Pip can’t see the floor, as in the fog reel; remembering skids would not help here, since even after two clean runs the floor may still be slippery.`],
   draw(t) { track(110, { t, sim: true, habit: DR, label: 'simulator: random floor each walk', slip: .3, habA: c => clamp((t - .8 - c * .3) * 3) }); habitLegend(208, 1);
     alpha(clamp(t - 3), () => { box(cxx(4) - 18, 104, 36, 56, C.yellow, .4); box(cxx(5) - 18, 104, 36, 56, C.yellow, .4); });
     card(['run, run, run, run, walk, walk', `dry floor, from cell 0: ${f2(EX.dr0)} (dry habit ${f2(EX.dry0)})`, { s: 'never lands past the cheese, whatever the slip', c: C.soft }], t - 4, { gap: 1.2 }); } },
 { h: 'Back on the real track', p: [`The same test: 1000 real walks at slip chance 0.2. None meets the cat. The average return is ${f2(TR2.avg)}, against ${f2(TD2.avg)} for the dry habit.`, `Honest note: in this toy, practising on any single slip chance above about ${THR.toFixed(2)} would give the same safe habit; below that, a run from cell 4 is still worth the risk. Randomising matters because nobody knows which single value matches reality, and real worlds differ in many ways at once.`],
   draw(t) { track(70, { t, slip: .45, habit: DR, nums: 0, label: 'randomised habit, real slip chance 0.2' });
     const w = TR2.W[RW2], P = walkPos(w.tr, Math.max(0, (t - .5) % (walkDur(w.tr) + 2))); pipOn(70, P, t); if (P.skidding) txt('skid!', P.x, 130, { s: 15, c: '#4f74a0' });
     const n = Math.floor((t - 1) * 14); dots100(TR2.W, 67, 160, n);
     card([`met the cat: ${TR2.cat} of ${NT}`, `average return: ${f2(TR2.avg)}`, { s: `dry habit, same floor: ${f2(TD2.avg)}`, c: C.soft }], t - 8.5, { gap: 1.2 }); },
   shot: [5, 13] },
 { h: 'Your turn', ctl: { label: 'real slip', values: SL, def: 2, fmt: v => String(v), onChange: () => {} },
   p: ['Drag the **real slip chance**; both habits are tested on 1000 real walks. Across: the slip chance; up: the average return; shaded: the practised range.', `On a dry floor the dry habit wins, ${f2(TD0.avg)} to ${f2(TR0.avg)}; from 0.1 on, the randomised one wins. The dry habit recovers a little on slick floors: early skids often carry it past cell 4, the only cell where a skid hits the cat.`],
   draw(t, st) { const i = st.ci ?? 2, s = SL[i], a = test(DRY, s), b = test(DR, s);
     track(40, { t, slip: s / .6, habit: DRY, nums: 0, label: `dry habit, real slip ${s}` }); track(138, { t, slip: s / .6, habit: DR, nums: 0, label: `randomised habit, real slip ${s}` });
     const k = Math.floor(t / 3.2) % 40, wa = a.W[k], wb = b.W[k], ta = t % 3.2 * 1.6;
     pipOn(40, walkPos(wa.tr, ta), t); pipOn(138, walkPos(wb.tr, ta), t);
     txt(`dry: ${a.cat} cat walks, average ${f2(a.avg)}`, 200, 216, { s: 15, c: C.goldInk }); txt(`randomised: ${b.cat} cat walks, average ${f2(b.avg)}`, 200, 240, { s: 15, c: '#3f7f7c' });
     sweepChart(SL.length, i); } },
 { h: 'Not a free lunch', p: ['Why is the randomised line flat, even beyond 0.4? Walking never skids, and a skid moves Pip three cells in two steps, like a run then a walk, so every walk takes four steps. Real robots are not so lucky: if reality lies outside the randomised range, the policy can break.', `Too wide a range can cost too, though not here: practising on slips up to 0.9 ${WIDE_SAME ? 'gives the same habit' : 'changes the habit'}. In real tasks it can make a policy cautious and slow everywhere, or too hard to learn.`],
   draw(t) { sweepChart(SL.length, null);
     card(['too narrow: reality falls outside', 'too wide: cautious, or too hard to learn', { s: 'randomise what really differs', c: C.soft }], t - 1, { y: 120, gap: 1.2 }); } },
 { h: 'Real robots', p: ['In 2017, Josh Tobin and colleagues trained a network to find objects in camera images using only simulated pictures, with random colours, textures and lighting: not an RL policy, but the same idea. It worked on real camera images it had never seen, well enough for a robot to grasp the objects.', 'In 2018 and 2019, OpenAI trained a robot hand in heavily randomised simulation to turn a block, and later a Rubik’s cube. It worked on the real hand, though not every time.'],
   draw(t) { const a = clamp(t - .5); alpha(a, () => { for (let i = 0; i < 4; i++) { const x = 52 + i * 82, cols = ['#e9b3b3', '#9fc9c7', '#f6d58c', '#b8c7e6'], bgs = ['#f3e6d8', '#e2ecf3', '#efe3f1', '#e8f1e2'];
       box(x, 40, 68, 60, bgs[i]); rrect(x, 40, 68, 60, 840 + i, { c: C.line }); box(x + 20 + (i % 2) * 8, 60 + (i % 3) * 4, 22, 22, cols[i]); rrect(x + 20 + (i % 2) * 8, 60 + (i % 3) * 4, 22, 22, 850 + i, { w: 1.1 }); }
       txt('random colours, textures, lighting', 200, 116, { s: 14, c: C.soft }); });
     alpha(clamp(t - 3), () => { hand(200, 196, t); txt('robot hand, randomised physics', 200, 268, { s: 14, c: C.soft }); });
     card(['2017  Tobin et al.: random textures, real images', '2018  OpenAI: a robot hand turns a block', '2019  the same idea, a Rubik’s cube'], t - 5, { gap: 1.2 }); } },
 { h: 'Recap', p: ['A **simulator** makes practice fast, cheap and safe, but it never matches the real world exactly. A policy tuned to one exact copy can rely on details reality does not share: the **reality gap**.', '**Domain randomisation** practises on many randomised versions, so the real world is just one more of them. The price is some caution, and the range must cover reality.'],
   draw(t) { track(70, { t, sim: true, habit: DR, slip: .3, label: 'simulator, randomised' }); track(196, { t, slip: .45, habit: DR, nums: 0, label: 'real track' }); L.mouse(cxx(0), 224, 1, .62, t);
     card(['practise in a simulator: fast and safe', 'one exact copy: habits that break in reality', 'randomise the copy: reality is one more version', 'price: caution; the range must cover reality'], t - .3, { y: 284, gap: 1, lh: 22 }); } },
];
function hand(x, y, t) {   // a simple robot hand holding a cube
  rrect(x - 30, y + 20, 60, 34, 870, { fill: '#cfd8df', w: 1.4 });
  [-22, -8, 6, 20].forEach((dx, i) => { const a = Math.sin(t * 2 + i) * .08; ctx.save(); ctx.translate(x + dx, y + 20); ctx.rotate(a); rrect(-4, -34, 9, 34, 880 + i, { fill: '#cfd8df', w: 1.2 }); ctx.restore(); });
  ctx.save(); ctx.translate(x, y + 2); ctx.rotate(Math.sin(t) * .25); const cs = ['#e07a7a', '#f6d58c', '#6fb7b4', '#8fb3d9'];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { box(-15 + i * 10, -15 + j * 10, 10, 10, cs[(i + 2 * j) % 4]); } rrect(-15, -15, 30, 30, 890, { w: 1.3 }); ctx.restore();
}
RLStory.register('sim2real', { title: 'Sim-to-Real & Domain Randomisation', track: 'Bigger worlds', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { DRY, DR, EX, TD2, TR2, TD0, TR0, SKIDW, RW2, test, SL, value } });
})();
