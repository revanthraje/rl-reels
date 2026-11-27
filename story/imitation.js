// Imitation learning as a story: Pip copies Mo, an old mouse who knows the maze (behaviour cloning), drifts into cells
// Mo never showed on the wet floor (compounding errors), and fixes it by asking Mo about its own cells (DAgger).
// All numbers come from the seeded runs below (test dice rng(1000 + i), i < 1000, as in the policy reel; DAgger walks rng(36)).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({});
const { cx, cy, key, maze, MOVES } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const SLIP = .2, GAMMA = .9, CAP = 100, NEVAL = 1000, NAMES = ['up', 'right', 'down', 'left'];
const MO = '#9b7fc0', MOINK = '#5f4689';
const no = (c, r) => r * 5 + c + 1;                          // loop-reel cell numbers 1..20
function mk(s) { const pi = []; s.split('/').forEach((row, r) => row.split('').forEach((ch, c) => { pi[r * 5 + c] = { U: 0, R: 1, D: 2, L: 3 }[ch] ?? 0; })); return pi; }
const PA = mk('RRRR./U.U.U/UL.RU/URRRU');                   // Mo's rule card = policy A of the policy reel
const OPEN = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.isRock(c, r) && !G.term(c, r)) OPEN.push([c, r]);
// Pip's paws on the wet floor: as intended 8 in 10, sideways 1 in 10 each way (relative to the tried move)
function slipMove(c, r, m, R) { const u = R(); let mm = m; if (u >= 1 - SLIP) mm = u < 1 - SLIP / 2 ? (m + 3) % 4 : (m + 1) % 4; const [nc, nr] = G.step(c, r, mm); return { c, r, m, mm, nc, nr }; }
function walk(pi, R, from = [0, 3]) { let [c, r] = from; const tr = [];
  for (let k = 1; k <= CAP; k++) { const s = slipMove(c, r, pi[r * 5 + c], R); tr.push(s); c = s.nc; r = s.nr; const e = G.term(c, r); if (e) return { tr, end: e, ret: e * GAMMA ** (k - 1) }; }
  return { tr, end: 0, ret: 0 }; }
// Mo's route (Mo never slides): 7 steps up the left side and along the top
const ROUTE = []; { let c = 0, r = 3; while (!G.term(c, r)) { ROUTE.push([c, r, PA[r * 5 + c]]); [c, r] = G.step(c, r, PA[r * 5 + c]); } }
// behaviour cloning: known cells copy the most common example; empty cells copy the nearest known cell (grid distance, lowest cell number on ties)
function clone(ex) { const cnt = {}; ex.forEach(([c, r, m]) => { const k = r * 5 + c; (cnt[k] = cnt[k] || [0, 0, 0, 0])[m]++; });
  const K = new Map(Object.keys(cnt).map(k => [+k, cnt[k].indexOf(Math.max(...cnt[k]))])), pi = new Array(20).fill(0), guess = new Set(), from = {};
  OPEN.forEach(([c, r]) => { const k = r * 5 + c; if (K.has(k)) { pi[k] = K.get(k); return; } let best = 1e9, bk = -1;
    [...K.keys()].sort((a, b) => a - b).forEach(q => { const d = Math.abs(q % 5 - c) + Math.abs(Math.floor(q / 5) - r); if (d < best) { best = d; bk = q; } });
    pi[k] = K.get(bk); guess.add(k); from[k] = bk; });
  return { K, pi, guess, from, n: ex.length }; }
// 1000 test walks from the start, same dice for every rule; 'lost' = walks that stood in a guessed cell
const jm = {};
function judge(P) { const id = P.pi.join('') + [...P.guess].join(','); if (jm[id]) return jm[id]; let s = 0, ch = 0, ct = 0, to = 0, lost = 0, lostCh = 0;
  for (let i = 0; i < NEVAL; i++) { const w = walk(P.pi, rng(1000 + i)); s += w.ret; if (w.end > 0) ch++; else if (w.end < 0) ct++; else to++;
    if (w.tr.some(t => P.guess.has(t.r * 5 + t.c))) { lost++; if (w.end > 0) lostCh++; } }
  return (jm[id] = { avg: s / NEVAL, ch, ct, to, lost, lostCh }); }
const rep = n => { let e = []; for (let i = 0; i < n; i++) e = e.concat(ROUTE); return e; };
const BC = clone(rep(3)), JBC = judge(BC), FULL = { pi: PA, guess: new Set(), K: new Map(OPEN.map(([c, r]) => [r * 5 + c, PA[r * 5 + c]])) }, JF = judge(FULL);
const BC30 = clone(rep(30)), J30 = judge(BC30), SAME30 = BC30.pi.join('') === BC.pi.join('');
const GUESSES = OPEN.filter(([c, r]) => BC.guess.has(r * 5 + c)), NBUMP = GUESSES.filter(([c, r]) => { const [a, b] = G.step(c, r, BC.pi[r * 5 + c]); return a === c && b === r; }).length;
// the story walks: the first test walk that copies Mo perfectly, and test walk 194 (slides off the route)
const GOODI = (() => { for (let i = 0; i < NEVAL; i++) { const w = walk(BC.pi, rng(1000 + i)); if (w.end > 0 && w.tr.length === 7 && w.tr.every(t => t.m === t.mm)) return i; } return 0; })();
const GOOD = walk(BC.pi, rng(1000 + GOODI)), STI = 193, STORY = walk(BC.pi, rng(1000 + STI));
const STUNK = STORY.tr.filter(t => BC.guess.has(t.r * 5 + t.c)).length;
const BACKK = STORY.tr.findIndex((t, i) => i > 0 && !BC.guess.has(t.nr * 5 + t.nc) && BC.guess.has(t.r * 5 + t.c));   // the step that slides back onto the route
const SPLIT = (() => { for (let i = 2; i < STORY.tr.length; i++) { const t = STORY.tr[i]; if (t.c === 1 && t.r === 3 && STORY.tr[i - 1].c === 2) return i; } return 5; })();  // back in cell 17
// DAgger: start from plain cloning; each round Pip walks 5 times with its copy (dice rng(36)), Mo labels every cell Pip stood in
const DAG = (() => { let ex = rep(3); const R = rng(36), out = [];
  for (let k = 0; k <= 4; k++) { const P = clone(ex), J = judge(P), walks = [];
    if (k < 4) for (let i = 0; i < 5; i++) { const w = walk(P.pi, R); walks.push(w); w.tr.forEach(t => ex.push([t.c, t.r, PA[t.r * 5 + t.c]])); }
    out.push({ P, J, walks, cells: P.K.size }); }
  out.forEach((d, k) => { d.newCells = k ? [...d.P.K.keys()].filter(q => !out[k - 1].P.K.has(q)) : []; d.added = k ? d.P.n - out[k - 1].P.n : 0; });
  return out; })();
const R1STEPS = DAG[0].walks.reduce((s, w) => s + w.tr.length, 0);
const fmtN = n => n.toLocaleString('en-US');

// ───── drawing helpers
function teacher(x, y, face = 1, s = 1, t = 0) {            // Mo: Pip's mouse with glasses and a plum scarf (people_world.md)
  L.mouse(x, y, face, s * 1.08, t); const c = ctx; c.save(); c.translate(x, y + Math.sin(t * 9) * 1.2); c.scale(face * s * 1.08, s * 1.08);
  c.strokeStyle = MO; c.lineWidth = 3.2; c.lineCap = 'round'; c.beginPath(); c.moveTo(5, -7); c.quadraticCurveTo(8, 1, 6, 8); c.stroke();
  c.beginPath(); c.moveTo(6, 6); c.lineTo(2, 12); c.stroke();
  c.strokeStyle = C.ink; c.lineWidth = 1.1; c.beginPath(); c.arc(13.5, -2, 3.3, 0, 7); c.stroke(); c.beginPath(); c.moveTo(10.2, -2.5); c.lineTo(7, -4); c.stroke();
  c.restore(); }
function wet(a = 1) { for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.isRock(c, r)) continue; L.hachure(OX + c * CS, OY + r * CS, CS, CS, C.blue, 15, 300 + c * 5 + r, .22 * a); } }
function scene(t, o = {}) { wet(o.wet ?? 1); maze({ t, labels: o.labels }); }
function mArrow(c, r, m, seed, col, a = 1, w = 1.6) { if (a <= 0) return; alpha(a, () => G.arrowIn(c, r, m, seed, { c: col, w })); }
// Mo's examples (plum), the copy's rule in known cells (gold), guesses (gold with a dot), grey '?' for cells with no example
function exArrows(K, a = 1, o = {}) { K.forEach((m, k) => { const c = k % 5, r = Math.floor(k / 5), u = o.only ? (o.only.includes(k) ? a : 0) : a; mArrow(c, r, m, 500 + k, MOINK, u, 1.8); }); }
function qMarks(K, a = 1) { if (a <= 0) return; OPEN.forEach(([c, r]) => { if (!K.has(r * 5 + c)) txt('?', cx(c), cy(r) + 4, { s: 20, c: '#b0a596', a }); }); }
function guessArrows(P, a = 1, o = {}) { if (a <= 0) return; let i = 0; OPEN.forEach(([c, r]) => { const k = r * 5 + c; if (!P.guess.has(k)) return; const u = (o.t != null ? clamp((o.t - i * (o.gap ?? 0)) * 2) : 1) * a; i++; if (u <= 0) return;
  const bad = o.hiCat && c === 2 && r === 1; mArrow(c, r, P.pi[k], 600 + k, bad ? C.roseInk : C.goldInk, u, bad ? 2.2 : 1.6); dot(OX + (c + 1) * CS - 11, OY + r * CS + 11, 3.4, bad ? C.rose : C.gold, u); }); }
function nums(a = 1) { if (a <= 0) return; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.isRock(c, r)) continue; txt(String(no(c, r)), OX + c * CS + 8, OY + r * CS + 11, { s: 12, al: 'left', c: C.soft, a }); } }
// where a walker is during transitions tr at time t (dur seconds per step); bumps go a little toward the wall and back
function walkAt(tr, t, dur) { const k = Math.floor(t / dur);
  if (k >= tr.length) { const l = tr[tr.length - 1]; return { x: cx(l.nc), y: cy(l.nr), k: tr.length, done: true, s: l }; }
  if (k < 0) return { x: cx(tr[0].c), y: cy(tr[0].r), k: 0, s: null };
  const s = tr[k], u = clamp((t - k * dur) / (dur * .7));
  if (s.nc === s.c && s.nr === s.r) { const [dx, dy] = MOVES[s.mm], b = Math.sin(Math.PI * u) * CS * .28; return { x: cx(s.c) + dx * b, y: cy(s.r) + dy * b, k, s, u }; }
  const e = ease(u); return { x: lerp(cx(s.c), cx(s.nc), e), y: lerp(cy(s.r), cy(s.nr), e), k, s, u }; }
function trailOf(tr, k, col = C.teal, a = .6, k0 = 0) { for (let i = k0; i < Math.min(k, tr.length); i++) { const s = tr[i]; dot(cx(s.nc) + ((i * 7) % 11) - 5, cy(s.nr) + ((i * 5) % 11) - 5, 2.6, col, a); } }
let face = 1; function pipAt(p, t) { const f = p.s ? Math.sign(MOVES[p.s.mm][0]) : 0; if (f) face = f; L.mouse(p.x, p.y + 3, face, .95, t); }
// one walk, animated: slides and bumps labelled; returns the walker
function play(tr, t, dur, o = {}) { const p = walkAt(tr, t, dur); trailOf(tr, p.k, o.col, .6, o.k0 || 0);
  if (p.s && !p.done && p.u < .9) { if (p.s.mm !== p.s.m) txt('slide!', p.x, p.y - 28, { s: 15, c: '#41678f' }); else if (p.s.nc === p.s.c && p.s.nr === p.s.r) txt('bump', p.x, p.y - 28, { s: 15, c: C.soft }); }
  if (o.mo) teacher(p.x, p.y + 3, face = p.s ? (Math.sign(MOVES[p.s.mm][0]) || face) : face, .95, t); else pipAt(p, t);
  return p; }
// stacked bar of 1000 walks: gold cheese, rose cat, grey stopped
function stack(x, y, w, h, J, a = 1, label = '', sub = '') { if (a <= 0) return; alpha(a, () => { const s = w / NEVAL;
  box(x, y, J.ch * s, h, C.gold, .8); box(x + J.ch * s, y, J.ct * s, h, C.rose, .8); box(x + (J.ch + J.ct) * s, y, J.to * s, h, C.grey, .9); rrect(x, y, w, h, 880 + y, { c: C.line, w: 1 });
  txt(label, x, y - 10, { s: 14, al: 'left', c: C.ink }); txt(sub, x + w, y - 10, { s: 14, al: 'right', c: C.soft });
  txt(String(J.ch), x + J.ch * s / 2, y + h / 2 + 1, { s: 15, c: C.goldInk }); txt(String(J.ct), x + (J.ch + J.ct / 2) * s, y + h / 2 + 1, { s: 14, c: C.roseInk }); }); }
function dim(a = .6) { box(0, 0, W, 285, C.bg, a); }

const STEPS = [
 { h: 'Meet Mo', p: ['This is **Mo**, an old mouse who has lived in Pip’s maze for years. In every cell, Mo knows a good move.', 'Pip could learn by trial and error, as in earlier reels, but that takes many walks and many meetings with the cat. Here is a shortcut: watch Mo, and copy. That is **imitation learning**.'],
   draw(t) { scene(t, { labels: clamp(t - .5), wet: 0 }); teacher(cx(0), cy(3) + 2, 1, 1, t); L.mouse(cx(1) + 4, cy(3) + 5, -1, .85, t * .7); if (t > 2) { txt('Mo', cx(0), cy(3) - 22, { s: 16, c: MOINK, a: clamp(t - 2) }); txt('Pip', cx(1) + 4, cy(3) - 20, { s: 16, c: C.soft, a: clamp(t - 2) }); }
     card([{ s: 'Mo: knows a good move in every cell', c: MOINK }, 'Pip: wants to learn it', { s: 'idea: copy what Mo does', c: C.soft }], t - 2.5); } },
 { h: 'Two kinds of paws', p: ['The floor is wet, as in the MDP and policy reels. Pip’s moves go as intended 8 times in 10 and slide sideways 1 in 10 each way. Mo’s old paws grip: **Mo never slides**.', 'Rewards are as usual: +1 for the step onto the cheese, −1 onto the cat, and either ends the walk. But a copier never looks at rewards. It only sees what Mo does.'],
   draw(t) { scene(t, { labels: 1 }); const x = cx(0), y = cy(3); L.mouse(x, y + 3, 1, .95, t);
     const u = clamp(t - 1); alpha(u, () => { arrow(x, y - 14, x, y - 52, 71, { c: C.goldInk, w: 1.8 }); arrow(x + 10, y - 6, x + 46, y - 6, 72, { c: '#41678f', w: 1.3 }); });
     txt('0.8', x - 16, y - 40, { s: 14, c: C.goldInk, a: u }); txt('0.1', x + 30, y - 18, { s: 14, c: '#41678f', a: u });
     card(['Pip tries up: up 8, right 1, left 1 (a bump) in 10', { s: 'Mo tries up: always up', c: MOINK }, { s: 'rewards: +1 cheese, −1 cat (unseen by a copier)', c: C.soft }], t - 3); } },
 { h: 'Recording a demonstration', p: ['Mo walks from the start to the cheese while we write down every step as a pair: the cell Mo was in, and the move Mo made. Cell 16, up. Cell 11, up. And so on.', `One recorded walk is a **demonstration**. Mo’s takes ${ROUTE.length} steps: up the left side, then along the top.`],
   shot: [4, 11],
   draw(t) { scene(t); nums(.8); const tr = ROUTE.map(([c, r, m]) => { const [nc, nr] = G.step(c, r, m); return { c, r, m, mm: m, nc, nr }; }), p = play(tr, t - 1, 1.1, { mo: 1, col: MO });
     for (let i = 0; i < Math.min(p.k, ROUTE.length); i++) mArrow(ROUTE[i][0], ROUTE[i][1], ROUTE[i][2], 500 + i, MOINK, 1, 1.8);
     const lines = ROUTE.map(([c, r, m]) => `cell ${no(c, r)} → ${NAMES[m]}`); box(30, 292, 340, 110, C.card); rrect(30, 292, 340, 110, 210, { c: C.line, w: 1.1 });
     txt('the record', 44, 305, { s: 14, al: 'left', c: C.soft });
     lines.forEach((s, i) => { if (i < p.k) txt(s, 50 + (i >= 4 ? 165 : 0), 326 + (i % 4) * 20, { s: 17, al: 'left', c: MOINK, a: clamp((t - 1 - (i + .6) * 1.1) * 3) }); }); } },
 { h: 'Three demonstrations', p: [`Mo walks the route 3 times. Mo never slides, so every walk is the same: ${BC.n} examples, all in the same ${BC.K.size} cells.`, `In the other ${OPEN.length - BC.K.size} cells, nobody has ever seen Mo. Those cells get a grey question mark.`],
   draw(t) { scene(t); for (let w = 0; w < 3; w++) { const u = clamp((t - .4 - w * 1.2) * 1.5); if (u > 0) ROUTE.forEach(([c, r], i) => dot(cx(c) - 12 + w * 12, cy(r) + 18, 2.6, MO, u * .8)); }
     exArrows(BC.K, clamp(t - .5)); qMarks(BC.K, clamp(t - 4)); teacher(cx(4) - 20, cy(0) + 18, -1, .8, t);
     card([{ s: `3 demonstrations: ${BC.n} examples`, c: MOINK }, `cells with examples: ${BC.K.size}`, { s: `cells with none: ${OPEN.length - BC.K.size}`, c: C.soft }], t - 2.5); } },
 { h: 'Behaviour cloning', p: ['Each example is like a quiz question with its answer: “in cell 16, which move?” “Up.” Learning a rule from questions with known answers is called **supervised learning**.', 'Used on demonstrations, it is **behaviour cloning**: in each cell, give Mo’s answer. Had Mo answered differently on different visits, Pip would take the most common answer.'],
   draw(t) { scene(t); exArrows(BC.K, .35); qMarks(BC.K, .6); let i = 0; BC.K.forEach((m, k) => { const u = clamp((t - 3 - i * .45) * 2); i++; mArrow(k % 5, Math.floor(k / 5), m, 700 + k, C.goldInk, u, 1.8); });
     L.mouse(cx(0), cy(3) + 3, 1, .95, t);
     card([{ s: 'question: in cell 16, which move?', c: C.ink }, { s: 'answer (Mo): up', c: MOINK }, { s: 'Pip’s rule: the same answer, cell by cell', c: C.goldInk }], t - .5, { gap: 1.2 }); } },
 { h: 'Cells without examples', p: ['But Pip’s rule must answer in every cell. Our copier uses a simple stand-in for what a trained network does: an empty cell copies the **nearest** cell with an example, counting steps across the grid. On a tie, the lower-numbered cell wins.', `Those answers are guesses. ${NBUMP} of the ${GUESSES.length} push into a rock or a wall. In cell 8, right beside the cat, the guess is “right”: onto the cat.`],
   shot: [4, 10],
   draw(t) { scene(t); exArrows(BC.K, .35); BC.K.forEach((m, k) => mArrow(k % 5, Math.floor(k / 5), m, 700 + k, C.goldInk, 1, 1.8)); qMarks(BC.K, clamp(1.5 - t));
     guessArrows(BC, 1, { t: t - 1.5, gap: .5, hiCat: t > 7 }); if (t > 7) txt('onto the cat!', cx(2), cy(1) + 20, { s: 14, c: C.roseInk, a: clamp(t - 7) });
     card([{ s: 'arrow: Mo’s example, copied', c: C.goldInk }, { s: 'arrow with a dot: a guess from the nearest example', c: C.goldInk }, { s: `${NBUMP} guesses bump, 1 walks onto the cat`, c: C.roseInk }], t - 2, { gap: 1.4, fs: 16 }); } },
 { h: 'Looks just like Mo', p: [`Pip walks with its copy. Walk ${GOODI + 1}: no slides, ${GOOD.tr.length} steps, cheese.`, 'On Mo’s route the copy is perfect, because every cell there holds Mo’s example. If Pip never slid, cloning would be all it needs.'],
   draw(t) { scene(t); BC.K.forEach((m, k) => mArrow(k % 5, Math.floor(k / 5), m, 700 + k, C.goldInk, .8, 1.6)); guessArrows(BC, .35); const p = play(GOOD.tr, t - .8, .9);
     if (p.done) bubble('+1', cx(4), cy(0) - 16, t - .8 - GOOD.tr.length * .9);
     card([`walk ${GOODI + 1} of ${NEVAL}`, `step ${Math.min(p.k, GOOD.tr.length)} of ${GOOD.tr.length}`, { s: 'every cell on the way has an example', c: C.soft }], 10); } },
 { h: 'One slide', p: [`Walk ${STI + 1}. On its very first move Pip tries up and slides right, into cell 17. Mo never stood there, so the copy guesses: “up”.`, 'Up leads to cell 12? No: Pip slides again, into cell 18. There the guess “up” pushes into the rock. Bump, bump. Then a slide back into cell 17.'],
   shot: [4, 9],
   draw(t) { scene(t); nums(.6); BC.K.forEach((m, k) => mArrow(k % 5, Math.floor(k / 5), m, 700 + k, C.goldInk, .45, 1.6)); guessArrows(BC, .85);
     const p = play(STORY.tr.slice(0, SPLIT), t - .8, 1.2); card([`walk ${STI + 1}: step ${Math.min(p.k, SPLIT)}`, { s: 'dot arrows: guesses, never shown by Mo', c: C.goldInk }, { s: 'blue: wet floor', c: '#41678f' }], 10); } },
 { h: 'Errors pile up', p: [`Pip climbs to cell 12 and bumps another rock, until a lucky slide puts it back on Mo’s route. Near the end it slides down into cell 8, and the guess says right: onto the cat.`, `That is **compounding error**: a slip leads to a cell Mo never showed, a guess there leads further off, and no example shows the way back. ${STUNK} of Pip’s ${STORY.tr.length} steps were in cells without an example.`],
   shot: [5, 15],
   draw(t) { scene(t); nums(.6); BC.K.forEach((m, k) => mArrow(k % 5, Math.floor(k / 5), m, 700 + k, C.goldInk, .45, 1.6)); guessArrows(BC, .85, { hiCat: true });
     trailOf(STORY.tr, SPLIT, C.teal, .35); const p = play(STORY.tr, t - .5 + SPLIT * 1, 1, { k0: SPLIT });
     const kk = Math.min(p.k, STORY.tr.length), inG = STORY.tr.slice(0, kk).filter(s => BC.guess.has(s.r * 5 + s.c)).length;
     if (p.done) bubble('−1', cx(3), cy(1) - 16, t - .5 - (STORY.tr.length - SPLIT));
     card([`walk ${STI + 1}: step ${kk} of ${STORY.tr.length}`, { s: `steps taken in cells without an example: ${inG}`, c: C.goldInk }, { s: p.done ? 'ended on the cat' : ' ', c: C.roseInk }], 10); } },
 { h: 'A thousand walks', p: ['Now 1000 walks from the start, same dice for both rules. Each bar is 1000 walks: gold reached the cheese, rose the cat, grey was stopped after 100 steps. Top: Pip with Mo’s whole rule card, a perfect copy in every cell. Bottom: the clone.', `The card reaches the cheese ${JF.ch} times, the clone ${JBC.ch}. ${JBC.lost} clone walks entered a cell without an example; only ${JBC.lostCh} of those reached the cheese.`],
   shot: [4, 12],
   draw(t) { scene(t, { wet: .5 }); guessArrows(BC, .5); dim(.55); L.mouse(cx(0), cy(3) + 3, 1, .95, t);
     stack(40, 312, 320, 24, JF, clamp(t - 1), 'Pip with Mo’s whole card', 'of 1000'); stack(40, 366, 320, 24, JBC, clamp(t - 3), 'Pip’s clone', `${JBC.to} stopped`);
     if (t > 6) { alpha(clamp(t - 6), () => { box(50, 92, 300, 68, C.card); rrect(50, 92, 300, 68, 970, { c: C.line, w: 1.1 }); }); txt(`clone walks that entered a guessed cell: ${JBC.lost}`, W / 2, 112, { s: 17, a: clamp(t - 6) }); txt(`of those, reached the cheese: ${JBC.lostCh}`, W / 2, 140, { s: 17, c: C.goldInk, a: clamp(t - 7) }); }
     caption('Mo itself never slides: 7 steps, every time', clamp(t - 9)); } },
 { h: 'More of Mo won’t help', p: [`Would more demonstrations help? Thirty walks by Mo give ${BC30.n} examples, but still the same ${BC30.K.size} cells, because Mo never leaves the route. The clone ${SAME30 ? 'is exactly the same' : 'barely changes'}, and so is its score: ${J30.ch}.`, 'The trouble is not how much data, but where it comes from. Mo’s examples cover Mo’s cells; Pip’s slides take it to its own. This mismatch is called **distribution shift**.'],
   draw(t) { scene(t); for (let w = 0; w < 30; w++) { const u = clamp((t - .3 - w * .12) * 2); if (u > 0) ROUTE.forEach(([c, r]) => dot(cx(c) - 20 + (w % 10) * 4.4, cy(r) + 16 + Math.floor(w / 10) * 5, 1.8, MO, u * .7)); }
     exArrows(BC30.K, .9); qMarks(BC30.K, clamp(t - 4));
     card([{ s: `30 demonstrations: ${BC30.n} examples`, c: MOINK }, `cells with examples: still ${BC30.K.size}`, { s: `clone reaches the cheese: ${J30.ch} of 1000`, c: C.goldInk }], t - 2); } },
 { h: 'DAgger: ask about Pip’s cells', p: ['In 2011, Stéphane Ross, Geoff Gordon and Drew Bagnell proposed a fix: let the learner walk, and ask the expert about the cells the learner actually reaches. Mo watches Pip and says, for every cell Pip stood in, “here, I would go …”.', 'Add those answers to the examples, clone again, repeat. The method is called **DAgger**, short for Dataset Aggregation.'],
   draw(t) { scene(t); nums(.5); BC.K.forEach((m, k) => mArrow(k % 5, Math.floor(k / 5), m, 700 + k, C.goldInk, .45, 1.6)); guessArrows(BC, .45); box(OX + CS + 3, OY + 2 * CS + 3, CS - 6, CS - 6, C.bg, .85); L.mouse(cx(1) + 4, cy(2) + 14, 1, .8, t); teacher(cx(0) - 2, cy(3) + 4, 1, .9, t);
     if (t > 1.5) { alpha(clamp(t - 1.5), () => { L.rrect(cx(1) - 18, cy(3) - 2, 150, 28, 940, { fill: C.card, c: MO, w: 1.4 }); rline(cx(1) - 18, cy(3) + 8, cx(0) + 18, cy(3) - 4, 941, { c: MO, w: 1.4 }); }); txt('cell 12: I’d go left', cx(1) + 57, cy(3) + 12, { s: 15, c: MOINK, a: clamp(t - 1.5) }); }
     if (t > 3) alpha(clamp(t - 3), () => arrow(cx(1) + 14, cy(2) - 14, cx(1) - 16, cy(2) - 14, 777, { c: MOINK, w: 2 }));
     card(['1. Pip walks with its clone', { s: '2. Mo labels every cell Pip stood in', c: MOINK }, '3. add the labels, clone again', { s: '4. repeat', c: C.soft }], t - 4, { gap: 1.2 }); } },
 { h: 'Round 1', p: [`Pip walks 5 times with its clone, slides and guesses included: ${R1STEPS} steps. Mo labels every one, so there are now ${DAG[1].P.n} examples.`, `${DAG[1].newCells.length} cells get Mo’s answer for the first time: cells ${DAG[1].newCells.sort((a, b) => a - b).map(k => k + 1).join(', ').replace(/, (\d+)$/, ' and $1')}. With them, the clone reaches the cheese ${DAG[1].J.ch} times in 1000, up from ${DAG[0].J.ch}.`],
   shot: [5, 12],
   draw(t) { scene(t); const ws = DAG[0].walks, wi = clamp(Math.floor((t - .5) / 1.3), 0, 4);
     for (let i = 0; i <= wi; i++) trailOf(ws[i].tr, i < wi ? 1e9 : Math.floor(((t - .5) - i * 1.3) / 1.3 * ws[i].tr.length) + 1, C.teal, i < wi ? .25 : .7);
     exArrows(DAG[0].P.K, .8); if (t < 7.5) guessArrows(DAG[0].P, .6); else guessArrows(DAG[1].P, .6);
     DAG[1].newCells.forEach((k, i) => { const u = clamp((t - 7.5 - i * .5) * 2); mArrow(k % 5, Math.floor(k / 5), PA[k], 800 + k, MOINK, u, 2.2); if (u > 0) L.rrect(OX + (k % 5) * CS + 3, OY + Math.floor(k / 5) * CS + 3, CS - 6, CS - 6, 820 + k, { c: MO, w: 1.6 }); });
     card([`Pip’s walks this round: ${Math.min(wi + 1, 5)} of 5`, { s: `examples: ${t < 7 ? DAG[0].P.n : DAG[1].P.n}, cells: ${t < 7.5 ? DAG[0].cells : DAG[1].cells}`, c: MOINK }, { s: t > 9 ? `clone reaches the cheese: ${DAG[1].J.ch} of 1000` : ' ', c: C.goldInk }], 10); } },
 { h: 'Round by round', p: ['This chart follows the rounds. Across: round 0, plain cloning, to round 4. Up: how many of the 1000 test walks reach the cheese. The dashed line is Pip with Mo’s whole card.', `Rounds 2 and 3 add examples but no new cells, so nothing changes. In round 4 all ${OPEN.length} cells hold Mo’s answer, and the clone scores ${DAG[4].J.ch}, ${DAG[4].J.ch === JF.ch ? 'exactly the card’s score' : 'close to the card'}. That is the ceiling: a copier matches its teacher, never beats it. And Mo’s card suits paws that grip: from cells 4, 10 and 14, one sideways slide lands on the cat.`],
   shot: [5, 13],
   draw(t) { const x0 = 70, x1 = 350, y0 = 230, y1 = 50, lo = 760, hi = 880, Y = v => lerp(y0, y1, (v - lo) / (hi - lo)), X = k => lerp(x0, x1, k / 4);
     rline(x0, y0, x1 + 10, y0, 950, { c: C.line }); rline(x0, y0, x0, y1 - 10, 951, { c: C.line }); [800, 840, 880].forEach(v => txt(String(v), x0 - 8, Y(v), { s: 13, al: 'right', c: C.soft }));
     for (let k = 0; k <= 4; k++) txt(String(k), X(k), y0 + 14, { s: 14, c: C.soft }); txt('round', x1, y0 + 34, { s: 14, al: 'right', c: C.soft }); txt('cheese, of 1000', x0 - 30, y1 - 22, { s: 14, al: 'left', c: C.soft });
     if (t > 1.5) { ctx.save(); ctx.setLineDash([5, 5]); ctx.globalAlpha = clamp(t - 1.5); ctx.strokeStyle = MOINK; ctx.beginPath(); ctx.moveTo(x0, Y(JF.ch)); ctx.lineTo(x1, Y(JF.ch)); ctx.stroke(); ctx.restore(); txt(`Mo’s whole card ${JF.ch}`, x1, Y(JF.ch) - 12, { s: 13, al: 'right', c: MOINK, a: clamp(t - 1.5) }); }
     for (let k = 0; k <= 4; k++) { const u = clamp(t - 3 - k * 1.2); if (u <= 0) continue; if (k) rline(X(k - 1), Y(DAG[k - 1].J.ch), lerp(X(k - 1), X(k), u), lerp(Y(DAG[k - 1].J.ch), Y(DAG[k].J.ch), u), 960 + k, { c: C.goldInk, w: 1.6 });
       if (u >= 1) { dot(X(k), Y(DAG[k].J.ch), 5, C.gold); txt(String(DAG[k].J.ch), X(k) + (k ? 0 : 22), Y(DAG[k].J.ch) + (k ? 18 : 0), { s: 14, c: C.goldInk }); } }
     card([`cells with Mo’s answer: ${DAG.map(d => d.cells).join(' → ')}`, `examples: ${DAG.map(d => d.P.n).join(' → ')}`, { s: 'gold dots: the clone after each round', c: C.goldInk }], t - 9, { gap: 1, fs: 16 }); } },
 { h: 'The price of asking', p: ['DAgger needs the expert on call, answering about awkward places it would never get into itself. For driving, that means labelling recordings of a learner’s drives, which is slow. And the learner must practise for real, mistakes included.', 'In return, the guarantee improves. Count the mistakes beyond the expert’s over a walk of many steps. One early slip can leave a clone lost for every step that follows, so in the worst case its extra mistakes grow with the square of the walk’s length. DAgger trains on the learner’s own cells, so they grow only in proportion to it.'],
   draw(t) { scene(t, { wet: .6 }); exArrows(DAG[4].P.K, .5); dim(.85); teacher(130, 170, 1, 1.5, t); L.mouse(280, 178, -1, 1.2, t);
     if (t > .8) { alpha(clamp(t - .8), () => { L.rrect(240, 72, 120, 34, 992, { fill: C.card, c: C.line, w: 1.3 }); rline(290, 106, 282, 150, 993, { c: C.line, w: 1.2 }); }); txt('and here?', 300, 89, { s: 16, a: clamp(t - .8) }); }
     if (t > 2) { alpha(clamp(t - 2), () => { L.rrect(50, 72, 120, 34, 990, { fill: C.card, c: MO, w: 1.4 }); rline(110, 106, 125, 140, 991, { c: MO, w: 1.3 }); }); txt('go left', 110, 89, { s: 16, c: MOINK, a: clamp(t - 2) }); }
     card(['DAgger needs: an expert who answers on call', '   labels for states the expert never visits', '   real practice walks by the learner', { s: 'it buys: errors that no longer pile up', c: C.goldInk }], t - 1, { gap: 1.2 }); } },
 { h: 'Your turn', ctl: { label: 'round', values: [0, 1, 2, 3, 4], def: 0, fmt: v => String(v), onChange: st => { st.w = null; } }, tap: tapCell,
   p: ['Slide through the DAgger rounds, from 0, plain cloning, to 4. Plum arrows are Mo’s examples; gold arrows with a dot are the clone’s guesses.', 'Tap a cell to drop Pip there and watch one walk with that round’s clone, slides and all. Each tap rolls new dice.'],
   draw(t, st) { const d = DAG[st.ci ?? 0]; scene(t); exArrows(d.P.K, .9); guessArrows(d.P, .9, { hiCat: true });
     if (st.w) { const p = play(st.w.tr, t - st.wt, .55); caption(p.done ? (st.w.end > 0 ? `cheese after ${st.w.tr.length} steps` : st.w.end < 0 ? `cat after ${st.w.tr.length} steps` : 'stopped after 100 steps') : `step ${p.k + 1}`); }
     else { L.mouse(cx(0), cy(3) + 3, 1, .95, t); caption('tap a cell to drop Pip there'); }
     card([`round ${st.ci ?? 0}: ${d.P.n} examples, ${d.cells} of ${OPEN.length} cells`, { s: `clone: cheese ${d.J.ch}, cat ${d.J.ct}${d.J.to ? `, stopped ${d.J.to}` : ''} of 1000`, c: C.goldInk }, { s: `Mo’s whole card: cheese ${JF.ch}`, c: MOINK }], 10, { y: 292, lh: 22, fs: 16 }); } },
 { h: 'Where it is used', p: ['Around 1990, ALVINN at Carnegie Mellon learned to steer a van from camera images by copying a human driver. In 2016, NVIDIA trained a car to steer from about 72 hours of human driving. To teach recovery, it added camera views shifted off-centre, labelled with the steering that leads back: DAgger’s idea, without an expert on call.', 'Robots often learn from demonstrations given by remote control, and chatbots begin by imitating human-written text.'],
   draw(t) { scene(t, { wet: .5 }); exArrows(DAG[4].P.K, .6); dim(.45); teacher(cx(0), cy(3) + 2, 1, .9, t); L.mouse(cx(1) + 10, cy(3) + 3, 1, .9, t);
     card(['ALVINN, about 1990: steering from a camera', 'NVIDIA 2016: about 72 hours of driving', { s: '   plus shifted views to learn recovery', c: C.soft }, 'robots: demonstrations by remote control'], t - .6, { gap: 1.2 }); } },
 { h: 'Recap', p: ['**Behaviour cloning** turns an expert’s demonstrations into examples and copies the expert’s move in each state: plain supervised learning, no rewards needed.', 'Its weak spot: small mistakes lead to states the expert never showed, and the errors **compound**. More expert walks don’t reach those states. **DAgger** asks the expert about the states the learner actually visits.'],
   draw(t) { scene(t, { wet: .6 }); exArrows(DAG[4].P.K, .8); teacher(cx(4) - 20, cy(0) + 18, -1, .8, t); L.mouse(cx(0), cy(3) + 3, 1, .95, t);
     card(['cloning: copy the expert, state by state', 'off the expert’s path, errors compound', 'more expert walks don’t go there', { s: 'DAgger: ask about the learner’s states', c: MOINK }], t - .3, { gap: 1 }); } },
];
let taps = 0;
function tapCell(x, y, st, t) { const c = Math.floor((x - OX) / CS), r = Math.floor((y - OY) / CS); if (c < 0 || c >= 5 || r < 0 || r >= 4 || G.isRock(c, r) || G.term(c, r)) return;
  st.w = walk(DAG[st.ci ?? 0].P.pi, rng(5000 + taps++), [c, r]); st.wt = t; }
RLStory.register('imitation', { title: 'Imitation Learning', track: 'Learning from people', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { ROUTE, BC, JBC, JF, BC30, J30, SAME30, NBUMP, GUESSES, GOODI, GOOD, STORY, STUNK, SPLIT, BACKK, DAG, R1STEPS } });
})();
