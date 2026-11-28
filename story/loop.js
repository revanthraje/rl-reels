// The RL loop as a story: Pip meets the maze, and we name every piece of the loop. Pip moves at random (no learning yet).
// All numbers come from the seeded random walks below (seed 769).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, maze, trail, pip, ox: OX, oy: OY, cs: CS } = G;
const NAMES = ['up', 'right', 'down', 'left'];
const no = (c, r) => r * G.cols + c + 1;                       // cell number 1..20, left to right, top to bottom

// ───── simulation: Pip picks each of the 4 actions with probability 1/4 (it knows nothing yet) ─────
const SEED = 769;
function walks(seed, n) {
  const R = L.rng(seed), out = [];
  for (let e = 0; e < n; e++) {
    let [c, r] = G.start; const tr = [];
    for (let k = 0; k < 5000; k++) { const m = Math.floor(R() * 4), [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr); tr.push({ c, r, m, nc, nr, rew, bump: nc === c && nr === r }); c = nc; r = nr; if (rew) break; }
    out.push(tr);
  }
  return out;
}
const EPS = walks(SEED, 100), E1 = EPS[0], N1 = E1.length;
const LENS = EPS.map(e => e.length), ENDS = EPS.map(e => e[e.length - 1].rew);
const NCHEESE = ENDS.filter(v => v > 0).length, NCAT = 100 - NCHEESE;
const MEANLEN = LENS.reduce((a, b) => a + b, 0) / 100, BUMPS = EPS.reduce((a, e) => a + e.filter(q => q.bump).length, 0) / LENS.reduce((a, b) => a + b, 0);
const AVGR = (NCHEESE - NCAT) / 100;
const BUMP1 = E1.findIndex(q => q.bump);                         // first bump of episode 1 (index 2 = step 3)

// ───── helpers ─────
const F = s => `${s}px 'Patrick Hand', cursive`;
function width(s, size) { ctx.save(); ctx.font = F(size); const w = ctx.measureText(s).width; ctx.restore(); return w; }
// rich text: parts = strings or [base, subscript]; returns width. o = { s, c, a, al }
function rich(parts, x, y, o = {}) {
  const s = o.s || 17, ss = Math.round(s * .66);
  const ws = parts.map(p => typeof p === 'string' ? width(p, s) : width(p[0], s) + width(p[1], ss) + 1), tot = ws.reduce((a, b) => a + b, 0);
  let X = o.al === 'center' ? x - tot / 2 : o.al === 'right' ? x - tot : x;
  parts.forEach((p, i) => {
    if (typeof p === 'string') txt(p, X, y, { s, c: o.c, a: o.a, al: 'left' });
    else { const wb = width(p[0], s); txt(p[0], X, y, { s, c: o.c, a: o.a, al: 'left' }); txt(p[1], X + wb + 1, y + s * .3, { s: ss, c: o.c, a: o.a, al: 'left' }); }
    X += ws[i];
  });
  return tot;
}
// a note card whose rows may contain subscripts (row = parts array, or { parts, c })
function rcard(rows, t, o = {}) {
  const x = o.x ?? 30, y = o.y ?? 292, w = o.w ?? 340, lh = o.lh ?? 23, h = 18 + rows.length * lh, a = clamp(t * 3);
  if (a <= 0) return; box(x, y, w, h, C.card, a); rrect(x, y, w, h, 200 + rows.length, { c: C.line, w: 1.1 });
  rows.forEach((row, i) => { const u = clamp((t - (o.gap ?? 1.1) * i) * 2.5); if (u <= 0) return; const parts = typeof row === 'string' ? [row] : Array.isArray(row) ? row : row.parts; rich(parts, x + 14, y + 20 + i * lh, { s: o.fs || 17, a: u, c: row.c }); });
}
function nums(a = 1, hiStart) { for (let r = 0; r < G.rows; r++) for (let c = 0; c < G.cols; c++) { if (G.isRock(c, r)) box(OX + c * CS + 3, OY + r * CS + 3, 16, 15, C.bg, .85 * a); txt(String(no(c, r)), OX + c * CS + 8, OY + r * CS + 11, { s: 12, al: 'left', c: hiStart && c === 0 && r === 3 ? C.ink : C.soft, a }); } }
function dim(a = .7) { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, 286); ctx.restore(); }
// Pip walking along transitions tr[from..to), dur s per step; a bump wobbles toward the wall and back
function walkAt(tr, from, to, t, dur) {
  const k = Math.floor(t / dur);
  if (k >= to - from) { const l = tr[to - 1]; return { x: cx(l.nc), y: cy(l.nr), k: to - from, done: true, face: Math.sign(l.nc - l.c) || 0 }; }
  const s = tr[from + k], u = ease((t % dur) / dur * 1.4), [dx, dy] = G.MOVES[s.m];
  if (s.bump) { const b = Math.sin(clamp((t % dur) / dur * 1.6) * Math.PI) * 11; return { x: cx(s.c) + dx * b, y: cy(s.r) + dy * b, k, s, face: dx || 0, bumping: (t % dur) / dur < .7 }; }
  return { x: lerp(cx(s.c), cx(s.nc), u), y: lerp(cy(s.r), cy(s.nr), u), k, s, face: dx || 0 };
}
function bumpMark(c, r, m, a) { if (a <= 0) return; const [dx, dy] = G.MOVES[m]; const x = clamp(cx(c) + dx * 14, OX + 22, OX + 5 * CS - 22), y = dy < 0 ? cy(r) + 24 : cy(r) - 24; box(x - 20, y - 8, 40, 16, C.bg, .85 * a); txt('bump!', x, y, { s: 15, c: C.roseInk, a });
  alpha(a, () => rline(cx(c) + dx * 31 - dy * 10, cy(r) + dy * 31 - dx * 10, cx(c) + dx * 31 + dy * 10, cy(r) + dy * 31 + dx * 10, 990, { c: C.rose, w: 2.5 })); }
const sign = v => v > 0 ? '+' + v : v < 0 ? '−' + (-v) : '0';
function logLine(k, tr) { const q = tr[k]; return `t = ${k}:  cell ${no(q.c, q.r)},  ${NAMES[q.m]}  →  cell ${no(q.nc, q.nr)},  reward ${sign(q.rew)}`; }
// bar chart of episode lengths: one thin bar per episode, teal = cheese, rose = cat, capped at 150 steps
function epChart(n, o = {}) {
  const S = o.slots || 100, x0 = 46, x1 = 362, y0 = 378, y1 = 308, cap = 150, w = (x1 - x0) / S;
  rline(x0, y0, x1, y0, 900, { c: C.line, w: 1 }); rline(x0, y0, x0, y1, 901, { c: C.line, w: 1 });
  txt('steps', x0 - 4, y1 - 10, { s: 13, c: C.soft, al: 'left' }); txt('150', x0 - 5, y1, { s: 12, c: C.soft, al: 'right' }); txt('0', x0 - 5, y0, { s: 12, c: C.soft, al: 'right' });
  txt('episode 1', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt('episode ' + S, x1, y0 + 11, { s: 12, c: C.soft, al: 'right' });
  for (let i = 0; i < Math.min(n, S); i++) {
    const v = LENS[i], h = Math.min(v, cap) / cap * (y0 - y1), grow = o.grow ? clamp(o.grow(i)) : 1;
    box(x0 + i * w + (S > 20 ? .4 : w * .2), y0 - h * grow, Math.max(1.2, S > 20 ? w - .8 : w * .6), h * grow, ENDS[i] > 0 ? C.teal : C.rose);
    if (v > cap && grow >= 1) txt('↑', x0 + i * w + w / 2, y1 - 6, { s: 11, c: C.rose });
    if (o.label && o.label(i) && grow >= 1) txt(String(v), x0 + i * w + w / 2, y0 - h - 9, { s: 12, c: ENDS[i] > 0 ? '#2f7f7b' : C.roseInk });
  }
}
// small sketch of the maze for the loop diagram
function miniMaze(x, y, s) {
  rrect(x, y, 5 * s, 4 * s, 610, { w: 1.4 }); for (let k = 1; k < 5; k++) rline(x + k * s, y, x + k * s, y + 4 * s, 611 + k, { c: C.line, w: .9, once: true }); for (let k = 1; k < 4; k++) rline(x, y + k * s, x + 5 * s, y + k * s, 620 + k, { c: C.line, w: .9, once: true });
  L.cheese(x + 4.5 * s, y + .5 * s, s / 64 * 1.3); L.cat(x + 3.5 * s, y + 1.5 * s + 2, 0, s / 64 * 1.3);
  [[1, 1], [2, 2]].forEach(([c, r], i) => rcircle(x + (c + .5) * s, y + (r + .5) * s, s * .3, 630 + i, { fill: '#d9d1c6', j: 2 }));
}
// arrow-key pad for "your turn"
const PAD = { x: 92, y: 338, d: 31, r: 14 };
const padPos = m => [PAD.x + G.MOVES[m][0] * PAD.d, PAD.y + G.MOVES[m][1] * PAD.d];
function drawPad(t, st) {
  for (let m = 0; m < 4; m++) {
    const [x, y] = padPos(m), [dx, dy] = G.MOVES[m], hot = st.last && st.last.m === m && t - st.last.t < .35;
    rrect(x - PAD.r, y - PAD.r, PAD.r * 2, PAD.r * 2, 700 + m, { fill: hot ? C.yellow : C.card, c: C.ink, w: 1.2 });
    arrow(x - dx * 7, y - dy * 7, x + dx * 7, y + dy * 7, 710 + m, { c: C.ink, w: 1.6, head: 6 });
  }
}

// ───── the steps ─────
const STEPS = [
 { h: 'Meet Pip', p: ['This is Pip, a mouse in a small maze. In the top corner sits a piece of **cheese**. Close by sleeps a **cat**. Two rocks block the middle.', 'Pip has never been here. Nobody gives it a map or tells it what to do. Whatever it learns, it must find out by trying things. That is **reinforcement learning**.'],
   shot: [3, 8],
   say: [[["Meet Pip, a little mouse in a little maze.", [40, 16, 320, 256]], ["Watch the cheese appear in the top corner. And right next to it, a sleeping cat.", [40, 16, 320, 256]]], [["Pip has no map and no instructions.", [40, 16, 320, 256]], "Whatever it learns, it learns by trying things. That's reinforcement learning."]],
   draw(t) { maze({ draw: t / 1.6, pop: { cheese: ease((t - 1.8) * 3), cat: ease((t - 2.4) * 3) }, t }); if (t > 3) pip(cx(0), cy(3), 1, t);
     if (t > 3.6) bubble('?', cx(0) + 14, cy(3) - 22, (t - 3.6) % 2.4);
     card(['cheese: top right', 'cat: asleep just below and left of it', { s: 'Pip: bottom left, knows nothing', c: C.soft }], t - 4.2, { gap: 1 }); } },

 { h: 'Two sides: agent and environment', p: ['Every RL problem splits the world in two. The **agent** is the one that makes choices: here, Pip.', 'The **environment** is everything else: the walls, the rocks, the cheese, the cat, and the rules for what happens next. Pip controls only its own choices. It cannot move a rock or wake the cat.'],
   shot: [3, 8],
   say: [[["First, the one making choices. Pip, circled in teal. That's the agent.", [40, 16, 320, 256]]], [["Then the gold box. Everything else is the environment.", [40, 16, 320, 256]], ["Walls, rocks, cheese, cat, and the rules.", [40, 16, 320, 256]]], [["Pip controls its own moves, nothing more. It can't push a rock or wake the cat.", [30, 292, 340, 87]]]],
   draw(t) { const a1 = clamp(t - .6), a2 = clamp(t - 2.6);
     maze({ t }); pip(cx(0), cy(3), 1, t);
     alpha(a1, () => { rcircle(cx(0), cy(3) + 2, 27, 801, { c: C.teal, w: 2.4 }); });
     alpha(a2, () => { rrect(OX - 7, OY - 7, 5 * CS + 14, 4 * CS + 14, 805, { c: C.gold, w: 2.4 }); });
     card([{ s: 'agent = Pip, the one who chooses', c: '#2f7f7b' }, { s: 'environment = the maze and all its rules', c: C.goldInk }, { s: 'they take turns, step after step', c: C.soft }], t - .8, { gap: 2 }); } },

 { h: 'State: where things stand', p: ['The **state** describes the situation now. Since the cheese, cat and rocks never move, it is just Pip’s cell. Cells are numbered 1 to 20; Pip starts in 16.', 'What the agent senses is its **observation**. Pip always knows its cell, so here they match. A camera robot sees only part of a room; then they differ, and it must act on what it observes.'],
   shot: [3, 9],
   say: [[["Watch the numbers fill the cells.", [40, 16, 320, 256]], ["Since nothing else ever moves, the situation is just where Pip stands.", [40, 16, 320, 256]], ["That's the state. Pip starts in cell sixteen.", [40, 16, 320, 256]]], ["Here, Pip always knows its cell. A robot with a camera might only see part of the room, and that's harder."]],
   draw(t) { maze({ t, hi: t > 3.5 ? [[0, 3, C.teal, .25]] : null }); nums(clamp((t - .6) * 1.5), t > 3.5); pip(cx(0), cy(3), 1, t);
     card(['state = which cell Pip is in', 'Pip starts in cell 16', { s: 'cells 7 and 13 are rocks: 18 cells Pip can be in', c: C.soft }], t - 3.6, { gap: 1.2 }); } },

 { h: 'Actions: four moves', p: ['In every state Pip picks an **action**: up, right, down or left. That is the whole menu.', 'The environment decides what an action does. A move goes one cell over, but if a wall or a rock is in the way, Pip **bumps** into it and stays where it is. From cell 16, down and left both bump.'],
   shot: [4, 10],
   say: [[["Four arrows, four moves. Up, right, down, left. That's the whole menu.", [40, 16, 320, 256]]], [["Now watch Pip try down.", [40, 16, 320, 256]], ["Bump. A wall. It stays put.", [40, 16, 320, 256]], ["Left bumps too.", [30, 300, 340, 64]]], ["The environment decides what each move really does."]],
   draw(t) { maze({ t }); nums(.5);
     for (let m = 0; m < 4; m++) { const a = clamp((t - .5 - m * .7) * 2); if (a <= 0) continue; const [dx, dy] = G.MOVES[m]; alpha(a, () => arrow(cx(0) + dx * 22, cy(3) + dy * 22, cx(0) + dx * 40, cy(3) + dy * 40, 820 + m, { c: C.goldInk, w: 1.8 }));
       txt(NAMES[m], cx(0) + dx * 44 + (dy ? 22 : 0), cy(3) + dy * 50 + (dx ? -12 : 0), { s: 14, c: C.goldInk, a }); }
     // demo: Pip tries down, then left: both bump
     let x = cx(0), y = cy(3);
     [[5, 2], [7.5, 3]].forEach(([t0, m]) => { const u = (t - t0) / 1.2; if (u > 0 && u < 1) { const b = Math.sin(clamp(u * 1.6) * Math.PI) * 11; x += G.MOVES[m][0] * b; y += G.MOVES[m][1] * b; } });
     pip(x, y, 1, t);
     if (t > 5) rcard([`down from 16: wall  →  still in 16`], t - 5, { y: 300 });
     if (t > 7.5) rcard([`down from 16: wall  →  still in 16`, `left from 16: wall  →  still in 16`], 1 + t - 7.5, { y: 300, gap: 0 });
     if (t > 5.6 && t < 6.6) bumpMark(0, 3, 2, 1); if (t > 8.1 && t < 9.1) bumpMark(0, 3, 3, 1); } },

 { h: 'Reward: one number per step', p: ['After every action the environment hands Pip a **reward**, a single number. Stepping onto the cheese pays +1, onto the cat −1, and every other step 0, bumps included. Cheese or cat ends the walk.', 'The task’s designer chose these numbers. Pip cannot change them and does not know them in advance. Built into Pip is only this: more reward is better.'],
   shot: [4, 10],
   say: [[["Look at the number in each cell.", [40, 16, 320, 256]], ["Mostly zeros. The cheese pays plus one, the cat minus one.", [40, 16, 320, 256]]], [["Either one ends the walk.", [30, 292, 340, 87]]], ["Pip doesn't know these numbers in advance. All it wants is more reward."]],
   draw(t) { maze({ t }); pip(cx(0), cy(3), 1, t);
     for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.isRock(c, r)) continue; const v = G.term(c, r), a = clamp((t - .8 - (v ? 1.5 : 0)) * 2);
       txt(v > 0 ? '+1' : v < 0 ? '−1' : '0', cx(c), cy(r) + (v ? -22 : 24), { s: v ? 20 : 15, c: v > 0 ? C.goldInk : v < 0 ? C.roseInk : C.soft, a }); }
     card(['number in a cell = reward for stepping onto it', 'bump into a wall or rock: reward 0', { s: 'cheese or cat: the walk is over', c: C.soft }], t - 3.4, { gap: 1.2 }); } },

 { h: 'One turn of the loop', p: ['Put it together. The agent looks at the state and picks an action. The environment applies its rules and answers with two things: the **next state** and a **reward**.', 'Then the same thing happens again from the new state. That cycle is the **RL loop**. Every method in this course runs on it.'],
   shot: [5, 11],
   say: [[["Here's the whole cycle in one picture.", [20, 50, 360, 180]], ["The agent sends an action across.", [20, 50, 360, 180]], ["The environment sends back two things, the next state and a reward.", [20, 50, 360, 180]]], [["Then round it goes again. Watch the dot circle.", [20, 50, 360, 180]], ["That's the RL loop. Every method in this course runs on it.", [30, 292, 340, 106]]]],
   draw(t) { maze({ t }); dim(.82);
     const A = { x: 92, y: 140 }, E = { x: 296, y: 140 };
     rrect(A.x - 62, A.y - 48, 124, 96, 840, { fill: C.card, c: '#2f7f7b', w: 1.8 }); L.mouse(A.x, A.y + 4, 1, 1.2, t); txt('agent', A.x, A.y - 34, { s: 17, c: '#2f7f7b' });
     rrect(E.x - 72, E.y - 48, 144, 96, 841, { fill: C.card, c: C.goldInk, w: 1.8 }); miniMaze(E.x - 45, E.y - 22, 18); txt('environment', E.x, E.y - 34, { s: 17, c: C.goldInk });
     const p1 = clamp((t - .8) / 1.2), p2 = clamp((t - 2.6) / 1.2);
     if (p1 > 0) arrow(A.x + 20, A.y - 50, E.x - 20, E.y - 50, 850, { c: C.ink, w: 1.6, p: p1 });
     if (p1 >= 1) txt('action', W / 2, A.y - 72, { s: 17, a: clamp((t - 2) * 2) });
     if (p2 > 0) arrow(E.x - 20, E.y + 52, A.x + 20, A.y + 52, 852, { c: C.ink, w: 1.6, p: p2 });
     if (p2 >= 1) txt('next state  +  reward', W / 2, A.y + 74, { s: 17, a: clamp((t - 3.8) * 2) });
     if (t > 4.5) { const u = ((t - 4.5) / 4) % 1, P = u < .5 ? [lerp(A.x + 20, E.x - 20, u * 2), A.y - 50] : [lerp(E.x - 20, A.x + 20, (u - .5) * 2), A.y + 52]; dot(P[0], P[1], 5, C.rose); }
     card(['1. agent sees the state', '2. agent picks an action', '3. environment answers: next state + reward', { s: '4. back to 1, from the new state', c: C.soft }], t - 4.6, { gap: 1, lh: 22 }); } },

 { h: 'Counting time', p: ['Time counts t = 0, 1, 2, and on. At time t, Pip is in state S t and picks action A t.', 'The answer comes one tick later, so R t+1 is the reward for A t. A walk reads S0, A0, R1, S1, A1, R2, and on. The third move, A2, pays R3; a 14-move walk ends at t = 14.'],
   shot: [6, 12],
   say: [[["Time ticks along this axis, zero, one, two, three.", [14, 70, 376, 160]]], [["Watch the letters line up. A state, an action, then a reward.", [14, 70, 376, 160]], ["Notice the reward carries the next tick's number. The answer always comes one tick later.", [14, 70, 376, 160]]], [["Teal for states, gold for actions, rose for rewards.", [30, 292, 340, 87]]]],
   draw(t) {
     const items = [['S', '0', 0], ['A', '0', 0], ['R', '1', 1], ['S', '1', 1], ['A', '1', 1], ['R', '2', 2], ['S', '2', 2], ['A', '2', 2], ['R', '3', 3], ['S', '3', 3]];
     const X = i => 34 + i * 37, Y = 120, cols = { S: '#2f7f7b', A: C.goldInk, R: C.roseInk };
     // time axis
     rline(24, 196, 380, 196, 860, { c: C.line, w: 1.2 }); [0, 1, 2, 3].forEach(k => { const x = X(k * 3) - 0; const a = clamp((t - .4 - k * .8) * 2); alpha(a, () => rline(x, 190, x, 202, 861 + k, { c: C.soft })); rich(['t = ' + k], x, 216, { s: 15, c: C.soft, a, al: 'center' }); });
     items.forEach(([b, s, k], i) => { const a = clamp((t - 2.4 - i * .45) * 2); if (a <= 0) return; rich([[b, s]], X(i), Y + (k * 3 === i ? 0 : 0), { s: 24, c: cols[b], a, al: 'center' }); if (b === 'S') rline(X(i), Y + 18, X(i), 186, 870 + i, { c: '#cfe3e2', w: 1, once: true }); });
     // bracket over one turn: S0 A0 then R1 S1
     const b = clamp((t - 7.2) * 2); if (b > 0) { alpha(b, () => { rline(X(0) - 10, Y - 26, X(3) + 10, Y - 26, 880, { c: C.ink, w: 1.2 }); rline(X(0) - 10, Y - 26, X(0) - 10, Y - 18, 881, { w: 1.2 }); rline(X(3) + 10, Y - 26, X(3) + 10, Y - 18, 882, { w: 1.2 }); });
       txt('one turn of the loop', (X(0) + X(3)) / 2, Y - 40, { s: 15, a: b }); }
     rcard([[['S', 't'], ' = state at time t   (teal)'], [['A', 't'], ' = action Pip picks there   (gold)'], [['R', 't+1'], ', ', ['S', 't+1'], ' = reward (rose) + next state, a tick later']], t - 8.2, { gap: 1.1 }); } },

 { h: 'Pip’s first move', p: ['Pip knows nothing yet, so it picks each action at random, as if rolling a four-sided die. It has no memory either, so it may repeat a bump. Its first roll, at t = 0: right.', 'The environment moves Pip from cell 16 to cell 17 and pays 0. In symbols: S0 = 16, A0 = right, R1 = 0, S1 = 17.'],
   shot: [3, 9],
   say: [[["Pip knows nothing yet, so it rolls a four sided die.", [40, 16, 320, 256]], ["The die says right.", [40, 16, 320, 256]]], [["Watch Pip slide from cell sixteen to seventeen.", [40, 16, 320, 256]], ["Reward zero.", [30, 292, 340, 87]], ["That's one full turn of the loop, written in symbols on the card.", [30, 292, 340, 87]]]],
   draw(t) { maze({ t, hi: [[0, 3, C.teal, .22 * clamp(t - .5)], [1, 3, C.teal, .22 * clamp(t - 3.2)]] }); nums(.6);
     const m = walkAt(E1, 0, 1, clamp(t - 2.2, 0, .99), 1); pip(m.x, m.y, 1, t);
     if (t > .8 && t < 2.2) bubble('right!', cx(0) + 6, cy(3) - 26, t - .8, C.goldInk);
     rcard([[['S', '0'], ' = 16     start'], [['A', '0'], ' = right     a random pick'], [['R', '1'], ' = 0,   ', ['S', '1'], ' = 17     the environment’s answer']], t - 1, { gap: 1.6 }); } },

 { h: 'A bump', p: [`Pip’s third move, at t = 2: it is in cell 18 and rolls down. There is a wall below. The environment’s rule says Pip stays put, so S3 = 18 again, and R3 = 0.`, 'Pip’s wishes don’t matter here: the environment decides what an action does. A wasted move still uses up a time step.'],
   shot: [4, 9],
   say: [[["Pip walks a couple of steps, then rolls down.", [40, 16, 320, 256]], ["Bump. There's a wall below.", [40, 16, 320, 256]]], [["Pip stays in cell eighteen, and the reward is zero.", [30, 292, 340, 64]]], ["A wasted move still uses up a time step. The teal dots are Pip's footprints."]],
   draw(t) { trail(E1.slice(0, BUMP1 + 1).map(q => [q.c, q.r]), 1); maze({ t, hi: [[2, 3, C.teal, .22]] }); nums(.6);
     let m = t < 1.4 ? walkAt(E1, 0, BUMP1, t, .7) : walkAt(E1, BUMP1, BUMP1 + 1, clamp(t - 2.2, 0, 1.19), 1.2);
     if (t >= 1.4 && t < 2.2) m = { x: cx(2), y: cy(3), face: 1 };
     pip(m.x, m.y, m.face, t);
     if (t > 2.6 && t < 4) bumpMark(2, 3, 2, 1);
     rcard([[['S', '2'], ' = 18,   ', ['A', '2'], ' = down'], [['R', '3'], ' = 0,   ', ['S', '3'], ' = 18     bumped: same cell']], t - 3.6, { gap: 1.4 });
     caption('teal dots = Pip’s footprints', clamp(t - 5.5)); } },

 { h: 'Round and round', p: ['The loop keeps turning: state, action, reward, next state. Pip bumps the wall again, steps back and forth, and walks into a rock.', 'Every one of these moves pays 0. After 13 moves, Pip has collected no reward at all. Each log line shows t, the cell, the action, where Pip lands and the reward.'],
   shot: [5, 12],
   say: [[["Round and round it goes. Watch Pip wander.", [40, 16, 320, 256]], ["Bumping, stepping back and forth, walking into a rock.", [40, 16, 320, 256]]], [["Each line of the log is one turn of the loop.", [30, 292, 340, 96]], ["And every single reward is zero.", [30, 292, 340, 96]]], [["Thirteen moves, and nothing to show for it.", [60, 392, 280, 24]]]],
   draw(t) { const d = .95, m = walkAt(E1, BUMP1 + 1, N1 - 1, t, d), k = Math.min(BUMP1 + 1 + m.k, N1 - 1);
     trail(E1.slice(0, k + 1).map(q => [q.c, q.r]), .9); maze({ t }); nums(.45); pip(m.x, m.y, m.face, t);
     if (m.bumping) bumpMark(m.s.c, m.s.r, m.s.m, 1);
     const shown = m.done ? k : k + ((t % d) / d > .55 ? 1 : 0), from = Math.max(0, shown - 4);
     box(30, 292, 340, 96, C.card); rrect(30, 292, 340, 96, 890, { c: C.line, w: 1.1 });
     for (let i = from; i < shown; i++) txt(logLine(i, E1), 44, 310 + (i - from) * 22, { s: 15, al: 'left', c: i === shown - 1 ? C.ink : C.soft });
     caption(`rewards so far: ${E1.slice(0, shown).reduce((a, q) => a + q.rew, 0)}  after ${shown} moves`); } },

 { h: 'The walk ends', p: [`On its ${N1}th move, at t = ${N1 - 1}, Pip steps up onto the cheese: R${N1} = +1. Cheese and cat are **terminal** states: when Pip reaches one, the walk stops, here at t = ${N1}.`, 'One walk from the start to a terminal state is called an **episode**. Afterwards Pip is put back in cell 16, and a fresh episode begins.'],
   shot: [3, 10],
   say: [[["Watch the last move. Pip steps up onto the cheese.", [40, 16, 320, 256]], ["Plus one!", [40, 16, 320, 256]]], [["Cheese and cat are terminal. Reaching one ends the walk.", [30, 292, 340, 87]], ["That whole walk is called an episode.", [30, 292, 340, 87]]], [["Then Pip is put back at the start, and a new episode begins.", [40, 16, 320, 256]]]],
   draw(t) { const back = clamp((t - 9) / .6); trail(E1.map(q => [q.c, q.r]), .35 * (1 - back)); maze({ t, hi: t > 1.6 && back < 1 ? [[4, 0, C.gold, .3]] : null }); nums(.45);
     const m = walkAt(E1, N1 - 1, N1, clamp(t - .4, 0, 1.2), 1.2); if (back <= 0) pip(m.x, m.y, 1, t); else alpha(back, () => pip(cx(0), cy(3), 1, t));
     bubble('+1', cx(4), cy(0) - 18, t - 1.4);
     if (t > 9.6) caption('Pip is back in cell 16: episode 2 starts', clamp(t - 9.6));
     rcard([[['S', String(N1 - 1)], ' = 10,   ', ['A', String(N1 - 1)], ' = up'], [['R', String(N1)], ' = +1,   ', ['S', String(N1)], ' = 5: cheese, walk over'], { parts: [`episode 1: ${N1} moves, total reward 1`], c: C.goldInk }], t + .2, { gap: 1.3 }); } },

 { h: 'Episode after episode', p: ['Let random Pip walk episode after episode. On this chart each bar is **one episode**: further right means walked later. Its height is how many steps it took.', `Teal ended at the cheese, rose at the cat. The first ten: 14 steps to the cheese, 13 to the cat, 67 to the cat, and so on. Only ${ENDS.slice(0, 10).filter(v => v > 0).length} of these 10 reach the cheese.`],
   shot: [5, 12],
   say: [[["Now let random Pip walk again and again.", [40, 16, 320, 256]], ["Each bar below is one episode. Taller means more steps.", [30, 290, 345, 110]]], [["Teal bars found the cheese. Rose bars found the cat.", [30, 290, 345, 110]], ["Out of these first ten, only three reach the cheese.", [30, 290, 345, 110]]]],
   draw(t) { const n = clamp(Math.floor((t - 1.2) / .9) + 1, 0, 10); maze({ t }); nums(.35);
     if (n > 0) { const e = n - 1, l = EPS[e][LENS[e] - 1]; trail(EPS[e].map(q => [q.c, q.r]).slice(-40), .45); pip(cx(l.nc), cy(l.nr) + (ENDS[e] > 0 ? 0 : 14), 1, t); caption(`episode ${n}: ${LENS[e]} steps, ${ENDS[e] > 0 ? 'cheese' : 'cat'} · dots: its last steps`); }
     epChart(n, { slots: 10, label: () => true }); } },

 { h: 'What blind wandering gets', p: [`After 100 episodes only ${NCHEESE} reached the cheese. One clear reason: the cat can be entered from four open cells, the cheese from only two, and the cat sits on the way to the cheese.`, `Episodes average ${Math.round(MEANLEN)} steps; about ${Math.round(BUMPS * 10)} moves in 10 are bumps. A small arrow marks bars too tall to draw; those episodes still ended at the cheese or cat.`],
   shot: [4, 11],
   say: [[["A hundred episodes, and only twenty six reached the cheese.", [30, 290, 345, 110]]], [["Why so few? Watch the rose arrows.", [40, 16, 320, 256]], ["The cat can be entered from four cells.", [40, 16, 320, 256]], ["The cheese, from only two.", [40, 16, 320, 256]]], ["And blind wandering is slow, about sixty three steps per episode."]],
   draw(t) { maze({ t });
     const into = (from, to, col, a, sd) => { if (a <= 0) return; const dx = to[0] - from[0], dy = to[1] - from[1]; alpha(a, () => arrow(cx(from[0]) + dx * 8, cy(from[1]) + dy * 8, cx(to[0]) - dx * 22, cy(to[1]) - dy * 22, sd, { c: col, w: 2 })); };
     [[3, 0], [2, 1], [4, 1], [3, 2]].forEach((f, i) => into(f, [3, 1], C.rose, clamp((t - 4 - i * .3) * 2), 980 + i));
     [[3, 0], [4, 1]].forEach((f, i) => into(f, [4, 0], C.teal, clamp((t - 6 - i * .3) * 2), 986 + i));
     txt('4 ways in', cx(2), cy(1) + 22, { s: 14, c: C.roseInk, a: clamp(t - 4.5) });
     txt('2 ways in', cx(4), cy(0) - 22, { s: 14, c: '#2f7f7b', a: clamp(t - 6.5) });
     epChart(100, { grow: i => (t - .4 - i * .03) * 3 });
     caption(`cheese ${NCHEESE}   ·   cat ${NCAT}`, clamp(t - 3.5)); } },

 { h: 'No answer key', p: [`Notice what Pip is never told. Nobody says “up was the right move in cell 10”. It only gets rewards, and in episode 1, ${N1 - 1} steps paid 0 before one paid +1.`, 'Which of those steps earned the cheese? Pip must work that out itself. This is the big difference from **supervised learning**, where every example comes with the correct answer.'],
   shot: [4, 11],
   say: [[["Here's episode one again, move by move.", [20, 20, 360, 130]], ["Every reward is zero, until the very last.", [20, 20, 360, 130]]], [["So which moves earned the cheese? Nobody says.", [20, 20, 360, 130]]], [["That's the big difference from supervised learning.", [30, 194, 340, 84]], ["There, every example comes with the right answer. Here, only a score, and often a late one.", [30, 194, 340, 84]]]],
   draw(t) {
     txt('episode 1: action (arrow, grey = bump) and reward', 30, 34, { s: 15, al: 'left', c: C.soft });
     for (let i = 0; i < N1; i++) { const q = E1[i], x = 34 + i * 24.5, a = clamp((t - .4 - i * .2) * 3); if (a <= 0) continue; const [dx, dy] = G.MOVES[q.m];
       alpha(a, () => arrow(x - dx * 7, 80 - dy * 7, x + dx * 7, 80 + dy * 7, 900 + i, { c: q.bump ? C.grey : C.goldInk, w: 1.5, head: 5 }));
       txt(q.rew ? '+1' : '0', x, 112, { s: 16, c: q.rew ? C.goldInk : C.soft, a });
       if (t > 4.2 && i < N1 - 1) txt('?', x, 140, { s: 16, c: C.soft, a: clamp((t - 4.2 - i * .05) * 2) }); }
     if (t > 4.2) txt('which steps deserve credit for the +1?', W / 2, 168, { s: 16, a: clamp(t - 5) });
     // comparison
     const a2 = clamp(t - 6.5);
     if (a2 > 0) { rrect(30, 194, 340, 84, 940, { fill: C.card, c: C.line, w: 1.1 });
       txt('supervised:  picture  →  “cat”   (here is the answer)', 44, 216, { s: 15, al: 'left', a: a2 });
       txt('RL:  action  →  0   (only a score, often late)', 44, 246, { s: 15, al: 'left', a: clamp(t - 7.5) });
       txt('grey arrows = bumps', 44, 268, { s: 12, al: 'left', c: C.soft, a: clamp(t - 8) }); }
     card([{ s: 'the reward never says which move was right', c: C.soft }, { s: 'it only says how things turned out', c: C.soft }], t - 9, { y: 300, gap: 1 }); } },

 { h: 'The goal', p: ['What is Pip trying to do? Collect the most reward **in total** over an episode, not just on the next step. Every move but the last pays 0, so the next reward alone says almost nothing.', `A rule for choosing actions is a **policy**. Rolling the die is one, the random policy; it averages ${AVGR.toFixed(2).replace('-', '−')} per episode. Later reels find much better ones.`],
   shot: [4, 10],
   say: [["So what does Pip want? The most reward in total, over a whole episode.", "The next reward alone says almost nothing."], [["Twenty six cheeses, seventy four cats.", [40, 16, 320, 256]], ["The random policy averages minus zero point four eight per episode.", [30, 292, 340, 87]]], ["A policy is just a rule for picking moves. Later reels find much better ones."]],
   draw(t) { maze({ t }); pip(cx(0), cy(3), 1, t);
     const a = clamp(t - .8); txt(`× ${NCHEESE}`, cx(4) - 34, cy(0) + 2, { s: 18, c: C.goldInk, a, al: 'right' }); txt(`× ${NCAT}`, cx(3) - 30, cy(1) + 4, { s: 18, c: C.roseInk, a, al: 'right' });
     card([`${NCHEESE} cheeses × (+1)  +  ${NCAT} cats × (−1)  =  ${sign(NCHEESE - NCAT)}`, `average per episode:  ${sign(NCHEESE - NCAT)} ÷ 100  =  ${AVGR.toFixed(2).replace('-', '−')}`, { s: 'goal: a policy with a higher total', c: C.goldInk }], t - 1.8, { gap: 1.5 }); } },

 { h: 'Your turn: be the agent', ctl: null, tap: tapAct, p: ['Now you are the agent. Tap an arrow to pick an action. The environment answers with the next state and the reward, and the log records each turn.', 'Try bumping into a rock, then reach the cheese. When the episode ends, tap anywhere to start a new one.'],
   shot: [3],
   say: [[["Now you're the agent.", [40, 16, 320, 256]], ["Tap an arrow on the pad.", [50, 296, 84, 84]]], [["Each tap is one turn, and the log records it.", [150, 292, 222, 96]]], ["Try bumping into a rock. Then go find the cheese."]],
   draw(t, st) {
     if (!st.pos) { st.pos = G.start.slice(); st.log = []; st.total = 0; st.ep = 1; }
     const lt = st.last ? t - st.last.t : 9; let x = cx(st.pos[0]), y = cy(st.pos[1]);
     if (st.last && lt < .4) { const u = ease(lt / .4), [dx, dy] = G.MOVES[st.last.m]; if (st.last.bump) { const b = Math.sin(clamp(lt / .4) * Math.PI) * 11; x += dx * b; y += dy * b; } else { x = lerp(cx(st.last.from[0]), x, u); y = lerp(cy(st.last.from[1]), y, u); } }
     trail(st.log.map(q => [q.c, q.r]), .5); maze({ t }); nums(.45); pip(x, y, st.last ? (G.MOVES[st.last.m][0] || 0) : 1, t);
     if (st.last && st.last.bump && lt < 1) bumpMark(st.last.from[0], st.last.from[1], st.last.m, 1 - clamp(lt - .5) * 2);
     if (st.last && st.last.rew && lt > .3) bubble(sign(st.last.rew), x, y - 18, lt - .3, st.last.rew > 0 ? '#b07d12' : C.roseInk);
     drawPad(t, st);
     const lx = 150; box(lx, 292, 222, 96, C.card); rrect(lx, 292, 222, 96, 960, { c: C.line, w: 1.1 });
     const from = Math.max(0, st.log.length - 3);
     if (!st.log.length) txt('your log is empty: pick a move', lx + 12, 312, { s: 14, al: 'left', c: C.soft });
     st.log.slice(from).forEach((q, i) => txt(`${from + i + 1}. cell ${no(q.c, q.r)}, ${NAMES[q.m]} → ${no(q.nc, q.nr)}, r ${sign(q.rew)}`, lx + 12, 312 + i * 22, { s: 14, al: 'left', c: i === st.log.length - from - 1 ? C.ink : C.soft }));
     txt(st.done ? `episode over: ${st.log.length} steps, total ${sign(st.total)}` : `episode ${st.ep}, state = cell ${no(st.pos[0], st.pos[1])}`, lx + 12, 376, { s: 14, al: 'left', c: st.done ? (st.total > 0 ? C.goldInk : C.roseInk) : '#2f7f7b' });
     caption(st.done ? 'tap anywhere for a new episode' : '');
   } },

 { h: 'The same loop, everywhere', p: ['Bigger problems use the same loop. DeepMind’s Atari player DQN, in 2015, saw screen pixels, chose joystick moves, and was rewarded by score changes.', 'AlphaGo, in 2016, was rewarded only at a game’s end: win or lose. Chat assistants are tuned with RL from human feedback, RLHF: often the whole reply is one action, scored once by a model of which answers people prefer.'],
   shot: [4, 10],
   say: [[["The same loop runs much bigger problems.", [10, 20, 380, 190]], ["DQN watched screen pixels and played Atari for points.", [10, 20, 380, 190]], ["AlphaGo was rewarded only at the end, win or lose.", [10, 20, 380, 190]]], [["Chat models too. A whole reply can be one action, scored once.", [10, 20, 380, 190]], ["Only the sizes change.", [30, 292, 340, 87]]]],
   draw(t) {
     const xs = [16, 108, 200, 288], head = ['agent', 'observes', 'acts', 'reward'];
     const rows = [['Pip', 'its cell', 'up, right, …', 'cheese +1, cat −1'], ['DQN, 2015', 'screen pixels', 'joystick move', 'score change'], ['AlphaGo, 2016', 'the board', 'places a stone', 'win +1, loss −1'], ['chat model', 'the prompt', 'writes a reply', 'preference score']];
     head.forEach((h, i) => txt(h, xs[i], 40, { s: 16, al: 'left', c: C.soft, a: clamp(t * 2) })); rline(18, 54, 384, 54, 970, { c: C.line, w: 1 });
     rows.forEach((row, j) => { const a = clamp((t - .8 - j * 1.6) * 2); if (a <= 0) return; row.forEach((s, i) => txt(s, xs[i], 82 + j * 50, { s: 14, al: 'left', a, c: i === 0 ? C.ink : '#5b544d' })); if (j < 3) rline(18, 104 + j * 50, 384, 104 + j * 50, 971 + j, { c: '#e4d8c7', w: 1, once: true }); });
     card(['same loop, every time:', 'observe  →  act  →  reward + next observation', { s: 'only the sizes change', c: C.soft }], t - 7.6, { gap: 1 }); } },

 { h: 'Recap', p: ['The **agent** picks actions; the **environment** answers each one with a reward and the next state. That loop repeats until a terminal state ends the **episode**.', 'Nobody tells the agent the right move. Its goal is the most reward in total. Adding rewards up sounds simple, but the next reel finds a catch.'],
   shot: [6],
   say: [[["So that's the loop.", [40, 16, 320, 256]], ["The agent acts, the environment answers with a reward and the next state, again and again until the episode ends.", [30, 292, 340, 110]]], ["Nobody tells Pip the right move. It just wants the most reward in total. And adding rewards up hides a catch, coming next."]],
   draw(t) { maze({ t }); pip(cx(0), cy(3), 1, t);
     card(['agent sees the state, picks an action', 'environment answers: reward + next state', 'repeat until a terminal state: one episode', { s: 'goal: the most total reward per episode', c: C.goldInk }], t - .3, { gap: 1 }); } },
];

function tapAct(x, y, st, t) {
  if (!st.pos) return;
  if (st.done) { st.pos = G.start.slice(); st.log = []; st.total = 0; st.done = false; st.last = null; st.ep = (st.ep || 1) + 1; return; }
  for (let m = 0; m < 4; m++) { const [px, py] = padPos(m); if (Math.abs(x - px) <= PAD.r + 3 && Math.abs(y - py) <= PAD.r + 3) {
    const [c, r] = st.pos, [nc, nr] = G.step(c, r, m), rew = G.term(nc, nr);
    st.log.push({ c, r, m, nc, nr, rew }); st.last = { m, t, bump: nc === c && nr === r, from: [c, r], rew }; st.pos = [nc, nr]; st.total += rew; if (rew) st.done = true; return; } }
}

RLStory.register('loop', { title: 'The RL Loop', track: 'Foundations', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { SEED, E1: E1.map(q => `${no(q.c, q.r)} ${NAMES[q.m]}${q.bump ? '*' : ''} -> ${no(q.nc, q.nr)} r${q.rew}`), N1, BUMP1, NCHEESE, NCAT, MEANLEN, BUMPS, AVGR, LENS: LENS.slice(0, 12), MAX: Math.max(...LENS), MIN: Math.min(...LENS) } });
})();
