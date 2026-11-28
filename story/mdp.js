// Markov Decision Process as a story: Pip's maze written down as a rulebook, then made slippery. All numbers come from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, hachure, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
// the same maze as the other Pip reels: cheese (4,0), cat (3,1), rocks (2,2) and (1,1), start (0,3)
const G = L.gridWorld({});
const { cx, cy, maze, pip, MOVES } = G, CS = G.cs, OX = G.ox, OY = G.oy;
const UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, SLIP = .2;
const P1 = v => String(+v.toFixed(2));
const fmtN = n => n.toLocaleString('en-US');

// ───── the slippery world ─────
// Pip tries move m. With chance 1 − p it goes that way; otherwise it slides sideways, p/2 to each side (never backwards).
// Moving into the outer wall or a rock = Pip stays put. Rewards: +1 landing on cheese, −1 landing on cat (both end the walk), else 0.
function slipMove(c, r, m, p, R) {
  const u = R(); let mm = m;
  if (u >= 1 - p) mm = u < 1 - p / 2 ? (m + 3) % 4 : (m + 1) % 4;
  const [nc, nr] = G.step(c, r, mm); return { c, r, m, mm, nc, nr };
}
// exact outcome list for (cell, move): [{ nc, nr, prob, mm }], outcomes landing in the same cell merged
function outcomes(c, r, m, p) {
  const out = [];
  for (const [mm, q] of [[m, 1 - p], [(m + 3) % 4, p / 2], [(m + 1) % 4, p / 2]]) {
    if (q <= 0) continue; const [nc, nr] = G.step(c, r, mm), o = out.find(o => o.nc === nc && o.nr === nr);
    if (o) o.prob += q; else out.push({ nc, nr, prob: q, mm });
  }
  return out;
}

// Sim A: Pip tries RIGHT from the start 1000 times (seed 2). Landing cells: (1,3) moved right, (0,2) slid up, (0,3) slid down into the wall = stayed.
const TRIES = (() => { const R = rng(2), out = []; for (let i = 0; i < 1000; i++) out.push(slipMove(0, 3, RIGHT, SLIP, R)); return out; })();
const kindOf = s => s.nc === 1 ? 0 : s.nr === 2 ? 1 : 2;            // 0 right, 1 up, 2 stayed
const CUM = [[0, 0, 0]]; TRIES.forEach(s => { const a = CUM[CUM.length - 1].slice(); a[kindOf(s)]++; CUM.push(a); });
const N1000 = CUM[1000], N10 = CUM[10];

// Sim B: 2000 walks of random moves on the slippery floor (seed 1), at most 200 steps each.
const WALKS = (() => { const R = rng(1), out = [];
  for (let e = 0; e < 2000; e++) { let c = 0, r = 3; const h = [[c, r]], tr = [];
    for (let k = 0; k < 200; k++) { const s = slipMove(c, r, Math.floor(R() * 4), SLIP, R); s.prev = h.length > 1 ? h[h.length - 2] : null; tr.push(s); c = s.nc; r = s.nr; h.push([c, r]); if (G.term(c, r)) break; }
    out.push({ h, tr }); }
  return out; })();
// Markov check at cell (0,2): every RIGHT attempt there, split by the cell Pip came from (below = (0,3), above = (0,1))
const MK = (() => { const t = { below: [0, 0, 0], right: [0, 0, 0], above: [0, 0, 0] };
  WALKS.forEach(w => w.tr.forEach(s => { if (s.c !== 0 || s.r !== 2 || s.m !== RIGHT || !s.prev) return;
    const g = s.prev[0] === 1 && s.prev[1] === 2 ? 'right' : s.prev[0] !== 0 ? null : s.prev[1] === 3 ? 'below' : s.prev[1] === 1 ? 'above' : null; if (!g) return; t[g][s.nc === 1 ? 0 : s.nr === 1 ? 1 : 2]++; }));
  const fr = a => { const n = a[0] + a[1] + a[2]; return { n, f: a.map(x => x / n) }; };
  return { below: fr(t.below), right: fr(t.right), above: fr(t.above) }; })();
// pooling: visits to (0,2) over all walks, and how many arrived by a path (cell sequence from the start) never walked before
const POOL = (() => { let visits = 0; const seen = new Set();
  WALKS.forEach(w => w.h.forEach((p, i) => { if (p[0] === 0 && p[1] === 2) { visits++; seen.add(w.h.slice(0, i + 1).join('|')); } }));
  return { visits, fresh: seen.size }; })();
// Sim C: number of different cell-by-cell paths of exactly k steps from the start (walk not ended), with the slippery floor
function pathCount(k) { let cur = new Map([['0,3', 1]]);
  for (let i = 0; i < k; i++) { const nx = new Map();
    for (const [key, n] of cur) { const [c, r] = key.split(',').map(Number); if (G.term(c, r)) continue; const outs = new Set();
      for (let m = 0; m < 4; m++) { const [a, b] = G.step(c, r, m); outs.add(a + ',' + b); } for (const o of outs) nx.set(o, (nx.get(o) || 0) + n); }
    cur = nx; }
  let s = 0; for (const n of cur.values()) s += n; return s; }
const PATHS = [1, 2, 3, 10].map(k => [k, pathCount(k)]);
// Sim D: Pip runs a fixed list of moves (the shortest route) 1000 times per slip chance (seed 5)
const PLAN = [UP, UP, UP, RIGHT, RIGHT, RIGHT, RIGHT], SLIPS = [0, .1, .2, .3, .5];
const PLANRUNS = {};
function planRuns(p) {
  if (PLANRUNS[p]) return PLANRUNS[p];
  const R = rng(5), cnt = [0, 0, 0], samples = [];
  for (let i = 0; i < 1000; i++) { let c = 0, r = 3; const tr = [];
    for (const m of PLAN) { const s = slipMove(c, r, m, p, R); tr.push(s); c = s.nc; r = s.nr; if (G.term(c, r)) break; }
    const e = G.term(c, r); cnt[e > 0 ? 0 : e < 0 ? 1 : 2]++; if (samples.length < 40) samples.push(tr); }
  return (PLANRUNS[p] = { cnt, samples });
}

// ───── drawing helpers ─────
function dim(a = .7, y0 = 0, y1 = 285) { box(0, y0, W, y1 - y0, C.bg, a); }
function ghost(c, r, s, a = 1, o = {}) { txt(s, cx(c) + (o.dx ?? 0), cy(r) + (o.dy ?? 20), { s: o.s || 14, c: o.c || '#9a8f84', a }); }
// a little U-turn arrow inside cell (c, r) on the side of the blocked direction mm
function stayLoop(c, r, mm, seed, col) {
  const [dx, dy] = MOVES[mm], x = cx(c) + dx * 21, y = cy(r) + dy * 21 + 4, a0 = Math.atan2(dy, dx);
  ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 7, a0 - Math.PI / 2 - .3, a0 + Math.PI / 2 + .3); ctx.stroke(); ctx.restore();
  const ae = a0 + Math.PI / 2 + .3, ex = x + 7 * Math.cos(ae), ey = y + 7 * Math.sin(ae), ta = ae + Math.PI / 2;
  rline(ex, ey, ex - 6 * Math.cos(ta - .5), ey - 6 * Math.sin(ta - .5), seed, { c: col, w: 1.4, once: true }); rline(ex, ey, ex - 6 * Math.cos(ta + .5), ey - 6 * Math.sin(ta + .5), seed + 1, { c: col, w: 1.4, once: true });
}
// arrows from (c, r) to every landing cell of move m, labelled with their chances; items appear one by one
function fan(c, r, m, p, t, o = {}) {
  outcomes(c, r, m, p).forEach((q, i) => {
    const a = clamp((t - i * (o.gap ?? .6)) * 2) * (o.a ?? 1); if (a <= 0) return;
    const col = q.mm === m ? (o.main || C.goldInk) : (o.side || '#3f7f7c'), lab = o.lab ? o.lab(q, i) : P1(q.prob);
    alpha(a, () => {
      if (q.nc === c && q.nr === r) { stayLoop(c, r, q.mm, 620 + i * 3, col); const [dx, dy] = MOVES[q.mm]; if (lab) txt(lab, cx(c) + dx * 21 + (dy ? 19 : 0), cy(r) + dy * 21 + 4 + (dx ? -15 : 0), { s: 15, c: col, al: dy ? 'left' : 'center' }); }
      else { const dx = q.nc - c, dy = q.nr - r; arrow(cx(c) + dx * (dx ? 24 : 16), cy(r) + dy * 16 + 4, cx(q.nc) - dx * 18, cy(q.nr) - dy * 18 + 4, 610 + i * 7, { c: col, w: q.mm === m ? 2.2 : 1.5 });
        if (lab) txt(lab, (cx(c) + cx(q.nc)) / 2 + (dy ? 7 : 0), (cy(r) + cy(q.nr)) / 2 + 4 + (dx ? -13 : 0), { s: 15, c: col, al: dy ? 'left' : 'center' }); }
    });
  });
}
// where Pip is drawn during one attempt: it travels to the landing cell, or bumps toward the blocked side and comes back
function tryPos(s, u) {
  if (s.nc === s.c && s.nr === s.r) { const [dx, dy] = MOVES[s.mm], k = Math.sin(Math.PI * clamp(u)) * CS * .28; return { x: cx(s.c) + dx * k, y: cy(s.r) + dy * k }; }
  const e = ease(u); return { x: lerp(cx(s.c), cx(s.nc), e), y: lerp(cy(s.r), cy(s.nr), e) };
}
// tally under the maze: three bars (moved right, slid up, stayed put), heights = count / max
const TLAB = ['moved right', 'slid up', 'stayed put'], TCOL = [C.gold, C.teal, C.blue];
function tally(cnt, max, title, a = 1, guide) {
  const base = 380, hmax = 54, xs = [100, 200, 300];
  alpha(a, () => {
    txt(title, 40, 299, { s: 14, al: 'left', c: C.soft }); rline(50, base, 350, base, 905, { c: C.line, w: 1 });
    if (guide) guide.forEach(([v, s]) => { const y = base - v / max * hmax; ctx.save(); ctx.setLineDash([3, 4]); ctx.strokeStyle = '#cdbfae'; ctx.beginPath(); ctx.moveTo(56, y); ctx.lineTo(346, y); ctx.stroke(); ctx.restore(); txt(s, 352, y, { s: 12, al: 'left', c: C.soft }); });
    xs.forEach((x, i) => { const h = clamp(cnt[i] / max) * hmax; box(x - 26, base - h, 52, h, TCOL[i]); txt(TLAB[i], x, base + 12, { s: 14, c: C.soft }); txt(fmtN(cnt[i]), x, base - h - 9, { s: 15 }); });
  });
}
// a polyline path through cell centres, shifted by off px so two paths can sit side by side
function pathLine(P, off, col, p, seed) {
  const n = (P.length - 1) * clamp(p);
  for (let i = 0; i < Math.ceil(n); i++) { const u = clamp(n - i), [a, b] = P[i], [c2, d] = P[i + 1];
    rline(cx(a) + off, cy(b) + off, lerp(cx(a), cx(c2), u) + off, lerp(cy(b), cy(d), u) + off, seed + i, { c: col, w: 2.2, j: .8 }); }
}
function awakeCat(x, y, s = 1) {
  ctx.save(); ctx.translate(x, y); ctx.scale(s, s);
  ctx.beginPath(); ctx.moveTo(-13, -4); ctx.lineTo(-11, -17); ctx.lineTo(-4, -9); ctx.moveTo(13, -4); ctx.lineTo(11, -17); ctx.lineTo(4, -9);
  ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3; ctx.fillStyle = '#c9b9a8'; ctx.fill(); ctx.stroke(); ctx.restore();
  rcircle(x, y, 13 * s, 41, { fill: '#c9b9a8', j: .8 });
  dot(x - 5 * s, y - 2 * s, 2.6 * s, '#5d7a3a'); dot(x + 5 * s, y - 2 * s, 2.6 * s, '#5d7a3a'); dot(x - 5 * s, y - 2 * s, 1 * s, C.ink); dot(x + 5 * s, y - 2 * s, 1 * s, C.ink);
  rline(x - 3 * s, y + 6 * s, x + 3 * s, y + 6 * s, 47, { once: true, w: 1 });
}
// a small copy of the maze (no rocks drawn by gridWorld at this size look fine) for side-by-side pictures
function mini(ox, oy) { return L.gridWorld({ cs: 34, ox, oy }); }
const MA = mini(22, 52), MB = mini(208, 52);
function wet(a = 1) { for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.isRock(c, r)) continue; hachure(OX + c * CS, OY + r * CS, CS, CS, C.blue, 11, 300 + c * 5 + r, .28 * a); } }

// the path pictures for the Markov step: two different pasts that both end on (0,2)
const PAST_A = [[0, 3], [1, 3], [2, 3], [1, 3], [1, 2], [0, 2]], PAST_B = [[0, 3], [0, 2]];

const STEPS = [
 { h: 'Writing the world down', p: ['Pip the mouse is back in its maze. Every step, Pip notes which cell it is in, its **state**, picks a move, its **action**, and gets a **reward**.', 'This reel writes the maze’s rules down so precisely that a computer could run them. That rulebook is called a **Markov Decision Process**, or MDP.'],
   shot: [3, 9.5],
   say: [[["Watch the maze draw itself, cheese, then cat.", [40, 16, 320, 256]], ["Pip notes its cell, picks a move, and gets a reward.", [40, 16, 320, 256]]], [["State, action, reward. Then round again.", [30, 292, 340, 110]]], "This reel writes the maze's rules down so precisely that a computer could run them. That rulebook is called an MDP."],
   draw(t) { maze({ draw: t / 1.4, pop: { cheese: ease((t - 1.5) * 3), cat: ease((t - 1.8) * 3) }, t });
     const u = clamp((t - 5.5) / .8), x = lerp(cx(0), cx(1), ease(u)); if (t > 2.2) pip(x, cy(3), 1, t);
     if (t > 3 && t < 5.6) box(OX + 3, OY + 3 * CS + 3, CS - 6, CS - 6, C.teal, .22 * clamp(t - 3) * (1 - clamp((t - 5.2) * 3)));
     if (t > 4.2 && t < 5.6) alpha(clamp(t - 4.2) * (1 - clamp((t - 5.2) * 3)), () => arrow(cx(0) + 22, cy(3) + 4, cx(0) + 46, cy(3) + 4, 701, { c: C.goldInk, w: 2 }));
     if (t > 6.4) bubble('0', cx(1), cy(3) - 26, t - 6.4, C.soft);
     if (t > 3) card([{ s: 'state: the cell Pip is in (shaded)', c: '#3f7f7c' }, { s: 'action: the move it picks, here right', c: C.goldInk }, 'reward: 0 for this step', { s: 'and again from the new state, until the walk ends', c: C.soft }], t - 3, { gap: 1.4 }); } },

 { h: 'Ingredient one: states', p: ['First, list every situation Pip can be in. Here a state is simply Pip’s cell, numbered as in the first reel, 1 to 20. Cells 7 and 13 are rocks, so there are **18 states**.', 'Two are special: landing on the cheese or the cat ends the walk. These are **terminal states**. That leaves 16 cells where Pip still has to choose a move.'],
   shot: [4, 10],
   say: [[["First ingredient, the states. Watch the cell numbers appear.", [40, 16, 320, 256]], ["Take away the two rocks, and eighteen cells are left.", [40, 16, 320, 256]]], [["Two are circled in rose, the cheese and the cat.", [40, 16, 320, 256]], ["Land on either and the walk ends. They're terminal.", [30, 292, 340, 87]]], "That leaves sixteen cells where Pip still has to choose."],
   draw(t) { maze({ t }); let n = 0;
     for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.isRock(c, r)) continue; const k = n++, a = clamp((t - .6 - k * .2) * 3); if (a <= 0) continue;
       const term = G.term(c, r); txt(String(r * 5 + c + 1), OX + c * CS + 10, OY + r * CS + 12, { s: 14, al: 'left', a, c: term ? C.roseInk : C.soft }); }
     pip(cx(0), cy(3), 1, t);
     if (t > 5) { const a = clamp(t - 5); alpha(a, () => { rrect(OX + 4 * CS + 2, OY + 2, CS - 4, CS - 4, 801, { c: C.rose, w: 1.6 }); rrect(OX + 3 * CS + 2, OY + CS + 2, CS - 4, CS - 4, 802, { c: C.rose, w: 1.6 }); }); }
     if (t > 6) card(['18 states: cells 1 to 20, minus rocks 7 and 13', { s: 'terminal (walk ends): cheese 5 and cat 9', c: C.roseInk }, { s: '16 cells where Pip still chooses a move', c: C.soft }], t - 6, { gap: 1 }); } },

 { h: 'Ingredient two: actions', p: ['Next, list the moves. Pip has the same four **actions** in every cell: up, right, down and left.', 'One convention matters: if a move points into the outer wall or a rock, Pip bumps and **stays put**. That still counts as a step.'],
   shot: [3, 9],
   say: [[["Second ingredient, the actions.", [40, 16, 320, 256]], ["Up, right, down, left. The same four in every cell.", [40, 16, 320, 256]]], [["Now watch Pip try left at the wall.", [40, 16, 320, 256]], ["Bump. It stays put, and that still counts as a step.", [30, 292, 340, 64]]]],
   draw(t) { maze({ t }); const c0 = 3, r0 = 3;
     if (t < 4.6) { pip(cx(c0), cy(r0), 1, t); ['up', 'right', 'down', 'left'].forEach((s, i) => { const [dx, dy] = MOVES[i], a = clamp((t - .5 - i * .5) * 3) * (1 - clamp((t - 4.1) * 3)); if (a <= 0) return;
         alpha(a, () => arrow(cx(c0) + dx * 26, cy(r0) + dy * 18 + 4, cx(c0) + dx * 50, cy(r0) + dy * 42 + 4, 720 + i * 9, { c: C.goldInk, w: 2 }));
         txt(s, cx(c0) + dx * 40 + (dy ? 18 : 0), cy(r0) + dy * 36 + 4 + (dx ? -12 : 0), { s: 14, c: C.goldInk, a }); }); }
     else { const u = clamp((t - 5.6) / .9), s = { c: 0, r: 3, mm: LEFT, nc: 0, nr: 3 }, P = tryPos(s, u); pip(P.x, P.y, -1, t);
       alpha(clamp(t - 4.8), () => arrow(cx(0) - 2, cy(3) - 26, cx(0) - 26, cy(3) - 26, 731, { c: C.goldInk, w: 1.8 })); ghost(0, 3, 'left?', clamp(t - 4.8), { dy: -42, s: 14, c: C.goldInk });
       if (t > 6.6) bubble('bump: stays', cx(0) + 10, cy(3) + 40, t - 6.6, C.soft); }
     if (t > 7.4) card(['actions: up, right, down, left, in every cell', { s: 'into a wall or rock: stay put, still one step', c: C.soft }], t - 7.4, { gap: 1 }); } },

 { h: 'Ingredient three: rewards', p: ['Then the rewards. A reward is paid for the step that **lands** on a cell: +1 for the cheese, −1 for the cat, and the walk ends there.', 'Every other step pays 0, bumps included. In general a reward may depend on the cell, the move and the landing cell, written R(s, a, s′). Here only the landing cell matters.'],
   shot: [3, 8],
   say: [[["Third ingredient, the rewards.", [40, 16, 320, 256]], ["Plus one for landing on the cheese, minus one for the cat.", [40, 16, 320, 256]], ["And zero everywhere else.", [40, 16, 320, 256]]], ["In general, a reward could depend on the cell, the move, and where Pip lands. Here, only the landing cell matters."]],
   draw(t) { maze({ t, labels: 0 }); pip(cx(0), cy(3), 1, t); let n = 0;
     for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) { if (G.isRock(c, r) || G.term(c, r)) continue; ghost(c, r, '0', clamp((t - 2.5 - n++ * .08) * 3), { dy: 22, dx: 18, s: 14 }); }
     ghost(4, 0, '+1', clamp(t - .6), { dy: -20, dx: -14, s: 18, c: '#b07d12' }); ghost(3, 1, '−1', clamp(t - 1.4), { dy: 22, dx: 18, s: 18, c: '#b54b4b' });
     if (t > 4.5) card(['land on the cheese: +1, walk ends', 'land on the cat: −1, walk ends', { s: 'any other step, bumps too: 0', c: C.soft }], t - 4.5, { gap: 1 }); } },

 { h: 'A wet floor', p: ['Now a twist: someone mopped the floor. When Pip tries a move, it goes that way **8 times in 10**. The other 2 times it slides sideways, relative to the direction it tried: 1 in 10 to its left, 1 in 10 to its right. It never slides backwards.', 'So the same action can lead to different cells. The world has become **random**.'],
   shot: [3, 9],
   say: [[["Now a twist. Someone mopped the floor.", [40, 16, 320, 256]], ["Watch Pip try to go up.", [40, 16, 320, 256]]], [["Eight times in ten, it goes up.", [40, 16, 320, 256]], ["Otherwise it slides sideways, one in ten each way. Never backwards.", [30, 292, 340, 87]]], "The same action can now lead to different cells. The world has become random."],
   draw(t) { wet(clamp(t * 1.5)); maze({ t }); const c0 = 3, r0 = 3; pip(cx(c0), cy(r0), 1, t);
     ghost(c0, r0, 'Pip tries up', clamp(t - 1), { dy: 40, s: 14, c: C.goldInk });
     if (t > 2) fan(c0, r0, UP, SLIP, t - 2, { gap: 1 });
     if (t > 5.5) card(['tries up: goes up 8 times in 10', { s: 'slides left 1 in 10, slides right 1 in 10', c: '#3f7f7c' }, { s: 'never backwards', c: C.soft }], t - 5.5, { gap: 1 }); } },

 { h: 'Try it at the start', p: ['Pip stands at the start and tries **right**, again and again. Right leads on to the next cell. A slide up lands one cell higher. A slide down hits the outer wall, so Pip stays put.', 'The bars count where each try ended, one bar per outcome. In the first 10 tries: 9 moved right and 1 slid up.'],
   shot: [2.6, 8, 14],
   say: [[["Pip stands at the start and tries right, again and again.", [40, 16, 320, 256]], ["It might move right, slide up, or slide into the wall and stay.", [40, 16, 320, 256]]], [["Watch the bars count where each try ended.", [40, 295, 330, 105]], ["After ten tries, nine went right and one slid up.", [40, 295, 330, 105]]]],
   draw(t) { wet(); maze({ t }); fan(0, 3, RIGHT, SLIP, t, { gap: .4, a: .55 });
     const k = clamp(Math.floor((t - 3) / 1.1), -1, 9), u = clamp(((t - 3) % 1.1) / .6);
     let P = { x: cx(0), y: cy(3) }; if (k >= 0 && t < 3 + 10 * 1.1) P = tryPos(TRIES[k], u); else if (k >= 9) P = tryPos(TRIES[9], 1);
     pip(P.x, P.y, 1, t);
     const done = clamp(Math.floor((t - 3.6) / 1.1) + 1, 0, 10); tally(CUM[Math.min(10, done)], 10, `where each try ended, after ${Math.min(10, done)} of 10 tries`, clamp(t - .5));
   } },

 { h: 'A thousand tries', p: ['Keep going to 1000 tries. The bars settle: **795** moved right, **102** slid up and 103 stayed put.', 'That is 0.795, 0.102 and 0.103 of the tries, close to the 0.8, 0.1 and 0.1 built into the floor. Each single try is a surprise, but the chances behind them are fixed. The dashed lines mark 800 and 100.'],
   shot: [3, 9],
   say: [[["Now keep going, to a thousand tries.", [40, 295, 330, 105]], ["Watch the bars settle.", [40, 295, 330, 105]], ["Seven hundred and ninety five went right. About a hundred each way otherwise.", [40, 295, 330, 105]]], ["Each single try is a surprise. But the chances behind them are fixed, eight in ten and one in ten."]],
   draw(t) { wet(); maze({ t }); pip(cx(0), cy(3), 1, t); const n = Math.round(lerp(10, 1000, ease(clamp((t - .6) / 6)))), cnt = CUM[n];
     [[1, 3, 0], [0, 2, 1]].forEach(([c, r, i]) => ghost(c, r, fmtN(cnt[i]), 1, { dy: 22, s: 16, c: i ? '#3f7f7c' : C.goldInk })); ghost(0, 3, fmtN(cnt[2]) + ' stayed', 1, { dy: -22, s: 14, c: '#4f74a0' });
     tally(cnt, 1000, `where each try ended, after ${fmtN(n)} tries`, 1, [[800, '800'], [100, '100']]); } },

 { h: 'Transition probabilities', p: ['These chances are written **P(s′ | s, a)**, read “P of s prime given s and a”: the probability that Pip lands in cell s′, the next state, after taking action a in state s.', 'If Pip bumps and stays put, s′ is s. For each cell and move, the landing chances add up to 1. Sixteen cells times four moves: 64 such lists.'],
   shot: [3, 10],
   say: [[["These chances get a name, transition probabilities.", [40, 16, 320, 256]], ["The chance of landing in each next cell, given where Pip is and what it tries.", [40, 16, 320, 256]]], [["From the start, trying right, they're zero point eight, zero point one, and zero point one.", [30, 292, 340, 87]], "They always add up to one."]],
   draw(t) { wet(); maze({ t }); pip(cx(0), cy(3), 1, t); fan(0, 3, RIGHT, SLIP, t - .3, { gap: .5 });
     ghost(0, 3, 's', clamp(t - .3), { dy: -20, dx: -22, s: 17, c: C.ink }); ghost(1, 3, 's′', clamp(t - .8), { dy: -22, dx: 18, s: 16, c: C.goldInk }); ghost(0, 2, 's′', clamp(t - 1.3), { dy: -22, dx: 18, s: 16, c: '#3f7f7c' });
     if (t > 2.4) card(['P( s′ | s, a ): chance of landing in s′', { s: 'from s = start with a = right (total 1):', c: C.soft }, { s: '0.8 next cell · 0.1 slid up · 0.1 stayed, s′ = s', c: C.ink }], t - 2.4, { gap: 1.2 }); } },

 { h: 'The whole rulebook', p: ['The pieces together: states, actions, transition probabilities, rewards, and the discount **γ (gamma)**, here 0.9. Like rewards, the designer picks γ: it says what we want, not how the floor behaves. That list is an MDP.', 'Pip does **not** read this rulebook; it learns by trying, like those 1000 tries. Computing the best plan from the rulebook, as the discount reel did, is called **planning**.'],
   shot: [5, 11],
   say: [[["Here's the whole rulebook.", [30, 22, 340, 168]], ["States, actions, transition chances, rewards, and gamma, the discount, at zero point nine.", [30, 22, 340, 168]]], [["That list is an MDP.", [30, 22, 340, 168]]], [["But Pip never reads it. It learns by trying.", [30, 262, 340, 64]], "Working out the best plan from the rulebook itself is called planning."]],
   draw(t) { wet(); maze({ t }); pip(cx(0), cy(3), 1, t); dim(.8, 0, 420);
     card(['S   states: 18 cells, 2 of them terminal', 'A   actions: up, right, down, left', 'P   transition probabilities P(s′ | s, a)', 'R   rewards R(s, a, s′): +1 cheese, −1 cat, else 0', 'γ   discount 0.9, chosen by the designer'], t - .3, { y: 22, gap: .9, lh: 30, fs: 17 });
     if (t > 6) card(['Pip never sees this rulebook.', { s: 'It only sees what happens when it tries.', c: C.soft }], t - 6, { y: 262, gap: 1 }); } },

 { h: 'Why “Markov”?', p: ['Here is the key assumption. The chances of what happens next depend only on the **current state and action**, not on the path that led there. This is the **Markov property**.', 'Pip can reach this cell straight from the start, or after a detour along the bottom row. Either way, trying right gives the same 0.8, 0.1 and 0.1.'],
   shot: [4, 10],
   say: [[["Here's the key assumption.", [40, 16, 320, 256]], ["Watch two paths reach the same cell. A gold detour, and a teal path straight up.", [40, 16, 320, 256]]], [["Either way, trying right gives the same chances.", [40, 16, 320, 256]], ["The past adds nothing, once you know the cell.", [30, 292, 340, 87]]], "That's the Markov property."],
   draw(t) { wet(); maze({ t }); alpha(1 - .88 * clamp((t - 5) * 2), () => { pathLine(PAST_A, 5, C.gold, (t - .5) / 3, 840); pathLine(PAST_B, -5, '#3f7f7c', (t - 3.8) / .8, 860); });
     ghost(2, 3, 'detour', clamp(t - 2), { dy: -20, s: 14, c: C.goldInk }); ghost(0, 3, 'straight up', clamp(t - 4.2), { dy: 24, dx: 6, s: 14, c: '#3f7f7c' });
     pip(cx(0), cy(2), 1, t); if (t > 5.2) fan(0, 2, RIGHT, SLIP, t - 5.2, { gap: .5 });
     if (t > 7) card(['two different pasts, same cell', { s: 'same chances next: 0.8 · 0.1 · 0.1', c: C.goldInk }, { s: 'the past adds nothing once you know the cell', c: C.soft }], t - 7, { gap: 1 }); } },

 { h: 'Checking it in data', p: ['Over 2000 random walks, whenever Pip tries right here, we log where it came from and where it lands, skipping bumps in place.', 'From below: 0.799, 0.093, 0.108. From the right or above: nearly the same. This is the test when the rules are hidden: rows further apart than luck allows mean the cell is not enough. Built Markov, our maze passes.'],
   shot: [3, 10],
   say: [[["Can we check that in data?", [40, 16, 320, 256]], ["Pip arrives at this cell from below, or from above.", [40, 16, 320, 256]]], [["Each row of the table counts where it landed after trying right.", [30, 290, 340, 110]], ["The rows nearly match. Same eight in ten, whatever the past.", [30, 290, 340, 110]]], "When the rules are hidden, this is how you'd test it."],
   draw(t) { wet(); maze({ t, hi: [[0, 2, C.gold, .25]] }); pip(cx(0), cy(2), 1, t);
     alpha(clamp(t - .5), () => arrow(cx(0) - 22, cy(3) + 2, cx(0) - 22, cy(2) + 20, 870, { c: '#3f7f7c', w: 2 })); ghost(0, 3, 'from below', clamp(t - .5), { dy: 22, s: 13, c: '#3f7f7c' });
     alpha(clamp(t - 1.2), () => arrow(cx(0) - 22, cy(1) + 2, cx(0) - 22, cy(2) - 18, 871, { c: C.goldInk, w: 2 })); ghost(0, 1, 'from above', clamp(t - 1.2), { dy: -18, s: 13, c: C.goldInk });
     if (t > 1.8) fan(0, 2, RIGHT, SLIP, 9, { a: .45 * clamp(t - 1.8), lab: () => '' });
     const a = clamp(t - 2.6); if (a <= 0) return; box(30, 290, 340, 110, C.card, a); rrect(30, 290, 340, 110, 880, { c: C.line, w: 1.1 });
     const cols = [190, 255, 320]; alpha(a, () => { txt('tried right here', 42, 304, { s: 14, al: 'left', c: C.soft }); ['moved right', 'slid up', 'slid down'].forEach((s, i) => txt(s, cols[i], 304, { s: 14, c: C.soft })); });
     [['below', '#3f7f7c', 330], ['right', '#6d645b', 356], ['above', C.goldInk, 382]].forEach(([g, col, y], j) => { const b = clamp(t - 3.6 - j * 1.2); if (b <= 0) return; const d = MK[g];
       txt(`from ${g} (${fmtN(d.n)}×)`, 42, y, { s: 15, al: 'left', c: col, a: b }); d.f.forEach((f, i) => txt(f.toFixed(3), cols[i], y, { s: 17, a: b })); }); } },

 { h: 'When the cell is not enough', p: ['New rule: the cat wakes after Pip’s 10th step, and pounces if Pip is beside it. Next to the cat, step 4 is safe, step 12 is not. The cell alone no longer predicts what comes next: **not Markov**.', 'Adding “is the cat awake?” fails: a sleeping cat might wake next step. The state must hold the clock: the cell plus steps taken so far.'],
   shot: [4, 10],
   say: [[["Now a rule that breaks it.", [16, 20, 370, 230]], ["The cat wakes after Pip's tenth step.", [16, 20, 370, 230]]], [["On the left, step four, the cat sleeps. Safe.", [16, 20, 184, 200]], ["On the right, step twelve, same cell. Pounce!", [202, 20, 184, 200]]], [["Same cell, different futures. Not Markov.", [30, 292, 340, 87]], ["The fix is to count the steps in the state too.", [30, 292, 340, 87]]]],
   draw(t) { const A = clamp(t * 2);
     alpha(A, () => { MA.maze({ t, pop: { cat: 0 } }); MB.maze({ t, pop: { cat: 0 } }); });
     L.cat(MA.cx(3), MA.cy(1) + 3, t, 34 / 64); awakeCat(MB.cx(3), MB.cy(1) + 2, 34 / 64);
     [MA, MB].forEach(M => M.pip(M.cx(3), M.cy(2), 1, t)); if (t > 2.3) txt('pounce!', MB.cx(3) + 26, MB.cy(1) + 18, { s: 14, c: '#b54b4b', a: clamp(t - 2.3) });
     txt('Pip’s step 4', 107, 34, { s: 16 }); txt('Pip’s step 12', 293, 34, { s: 16 });
     txt('cat asleep: safe', 107, 210, { s: 15, c: '#3f7f7c', a: clamp(t - 1.5) }); txt('cat awake: danger', 293, 210, { s: 15, c: '#b54b4b', a: clamp(t - 2.3) });
     txt('same cell', 200, 240, { s: 15, c: C.soft, a: clamp(t - 3.2) });
     if (t > 4.6) card(['same cell, different futures: not Markov', { s: '“awake?” alone is not enough: when will it wake?', c: C.soft }, { s: 'fix: state = (cell, steps taken so far)', c: C.goldInk }], t - 4.6, { gap: 1.1 }); } },

 { h: 'Cells, not paths', p: ['Why does Markov matter so much? Because of it, anything Pip works out about what to do or what to expect can be stored **per cell**, for just 16 cells.', 'If the past mattered, Pip would need an entry for every **path** instead. Counting paths from the start, a bump counting as a step: 3 after one step, 37 after three, 202,435 after ten.'],
   shot: [4, 11],
   say: ["So why does Markov matter so much?", [["Because Pip can file everything it learns per cell. Just sixteen of them.", [30, 276, 340, 124]]], [["If the past mattered, it would need a page for every path.", [30, 276, 340, 124]], ["After ten steps, that's over two hundred thousand paths.", [30, 276, 340, 124]]]],
   draw(t) { wet(.6); maze({ t }); pip(cx(0), cy(3), 1, t); dim(.35);
     const lines = PATHS.map(([k, n]) => ({ s: `paths of ${k} step${k > 1 ? 's' : ''} from the start:  ${fmtN(n)}`, c: k === 10 ? '#b54b4b' : C.ink }));
     card([{ s: 'cells where Pip chooses a move:  16', c: C.goldInk }, ...lines], t - .8, { y: 276, lh: 22, gap: 1.3, fs: 17 });
   } },

 { h: 'Every visit counts', p: ['In those 2000 random walks, Pip stood on this cell 16,047 times. **13,887** of those arrivals came by a path it had never walked before.', 'Filed by path, almost every page of Pip’s notebook would hold one lonely visit. Filed by cell, all 16,047 share one page. Markov alone allows that pooling: what happens next does not depend on how Pip got here.'],
   shot: [4, 10],
   say: [[["Look at this cell.", [40, 16, 320, 256]], ["In two thousand random walks, Pip stood here about sixteen thousand times.", [30, 292, 340, 87]], ["Most arrivals came by a brand new path.", [30, 292, 340, 87]]], ["Filed by path, each page would hold one lonely visit. Filed by cell, they all share one page. Markov makes that pooling fair."]],
   draw(t) { wet(); maze({ t, hi: [[0, 2, C.gold, .3]] }); pip(cx(0), cy(2), 1, t);
     const u = ease(clamp((t - .5) / 4)), v = Math.round(POOL.visits * u), f = Math.round(POOL.fresh * u);
     ghost(0, 2, fmtN(v), 1, { dy: 22, s: 15, c: C.goldInk });
     card([`visits to this cell:  ${fmtN(v)}`, { s: `arrived by a brand-new path:  ${fmtN(f)}`, c: '#b54b4b' }, { s: 'Markov: every visit tells the same story', c: C.soft }], t - .2, { gap: 1.2 }); } },

 { h: 'Your turn', ctl: { label: 'slip', values: SLIPS, def: 2, fmt: v => String(v), onChange: st => { st.run = null; } }, tap: tapRun,
   p: ['Pip memorises a shortest route as a fixed list: up three, then right four. The **slip chance**, the total chance of sliding sideways, was 0.2 so far. Drag it: Pip runs the list 1000 times. Bars count endings: cheese, cat, or nowhere.', 'At 0.2 only 229 reach the cheese: a list cannot react to slips. A rule for every cell can: a **policy**, coming next.'],
   shot: [3, 8],
   say: [[["Your turn. Pip memorised a route as a fixed list. Up three, right four.", [40, 16, 320, 256]], ["Drag the slip chance, and the list runs a thousand times.", [40, 290, 330, 100]]], [["At zero point two, only two hundred and twenty nine runs reach the cheese.", [40, 290, 330, 100]], "A list can't react to slips. A rule for every cell can. That's a policy, coming next."]],
   draw(t, st) { const p = SLIPS[st.ci ?? 2], R1 = planRuns(p), R0 = st.prevCi != null ? planRuns(SLIPS[st.prevCi]) : R1, u = ease((t - (st.changed ?? -9)) / .8);
     if (p > 0) wet(p / .2 > 1 ? 1 : p / .2); maze({ t });
     // the memorised list, drawn as faint numbered arrows along the route
     let c = 0, r = 3; PLAN.forEach((m, i) => { alpha(.45, () => G.arrowIn(c, r, m, 900 + i * 7, { c: C.goldInk, w: 1.4 })); [c, r] = G.step(c, r, m); });
     const run = st.run; let P = { x: cx(0), y: cy(3) }, face = 1;
     if (run) { const d = .7, k = Math.floor((t - st.rt) / d), uu = clamp(((t - st.rt) % d) / .55);
       const kk = Math.min(k, run.length - 1), s = run[kk]; G.trail(run.slice(0, kk + (k >= run.length ? 1 : 0)).map(q => [q.c, q.r]).concat(k >= run.length ? [[s.nc, s.nr]] : []), .7); P = k >= run.length ? tryPos(s, 1) : tryPos(s, uu); face = s.nc !== s.c ? Math.sign(s.nc - s.c) : 0;
       if (k < run.length && s.mm !== s.m && uu > .3) bubble('slip!', cx(s.c), cy(s.r) - 24, (t - st.rt) % d - .2, '#3f7f7c');
       if (k >= run.length) { const e = G.term(run[run.length - 1].nc, run[run.length - 1].nr); caption(e > 0 ? 'this run: cheese!' : e < 0 ? 'this run: the cat' : 'this run: the list ran out before the cheese'); }
       else caption(`move ${kk + 1} of 7: list says ${['up', 'right', 'down', 'left'][s.m]}`); }
     else caption(`slip ${p}: ${p ? P1(p / 2) + ' to each side' : 'no sliding at all'}  ·  tap to watch one run`);
     pip(P.x, P.y, face, t);
     const cnt = R1.cnt.map((v, i) => lerp(R0.cnt[i], v, u)), xs = [100, 200, 300], base = 374, hm = 58, lab = ['cheese', 'cat', 'nowhere'], col = [C.gold, C.rose, C.grey];
     txt('1000 runs of the list end on …', 40, 299, { s: 14, al: 'left', c: C.soft }); rline(50, base, 350, base, 906, { c: C.line, w: 1 });
     xs.forEach((x, i) => { const h = cnt[i] / 1000 * hm; box(x - 26, base - h, 52, h, col[i]); txt(lab[i], x, base + 11, { s: 14, c: C.soft }); txt(fmtN(Math.round(cnt[i])), x, base - h - 9, { s: 15 }); }); } },

 { h: 'Where MDPs are used', p: ['The name honours Andrey Markov, who studied chains of random events in the early 1900s. In the 1950s Richard Bellman used MDPs for planning under uncertainty.', 'Robot navigation, stock and inventory planning and board games are often MDPs. When one snapshot is not Markov, people widen the state: DeepMind’s Atari player fed in the last **four screen images**, so a ball’s motion showed.'],
   shot: [3, 10],
   say: [[["The name honours Andrey Markov, who studied chains of random events over a century ago.", [30, 292, 340, 87]], ["Richard Bellman put decisions into them in the nineteen fifties.", [30, 292, 340, 87]]], [["Robots, board games, and more are often written as MDPs.", [30, 60, 350, 140]], ["And when one snapshot isn't enough, people widen the state. The Atari player looked at the last four frames, so it could see motion.", [270, 70, 120, 130]]]],
   draw(t) { const a1 = clamp(t - .5), a2 = clamp(t - 2), a3 = clamp(t - 3.5);
     alpha(a1, () => { L.robot(80, 120, 2, t); txt('robots', 80, 190, { s: 16 }); });
     alpha(a2, () => { for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) box(160 + i * 20, 80 + j * 20, 20, 20, (i + j) % 2 ? '#d9c8b0' : C.card); rrect(160, 80, 80, 80, 950, { c: C.ink }); dot(190, 110, 6, C.ink); dot(210, 130, 6, '#f5efe6'); rcircle(210, 130, 6, 951, { j: .3 }); txt('board games', 200, 190, { s: 16 }); });
     alpha(a3, () => { for (let k = 0; k < 4; k++) { const x = 282 + (k % 2) * 56, y = 82 + Math.floor(k / 2) * 42; box(x, y, 50, 36, C.card); rrect(x, y, 50, 36, 960 + k, { c: C.line, w: 1 }); dot(x + 10 + k * 9, y + 26 - k * 5, 3.2, C.ink); txt(String(k + 1), x + 44, y + 9, { s: 12, c: C.soft }); }
       txt('last 4 frames', 330, 190, { s: 16 }); });
     if (t > 5) card(['Markov: Andrey Markov, early 1900s', 'decision processes: Richard Bellman, 1950s', { s: 'not Markov? put more into the state', c: C.soft }], t - 5, { gap: 1 }); } },

 { h: 'Recap', p: ['An **MDP** writes a world down as states, actions, transition probabilities, rewards and γ. The same action can lead to different cells, but the chances are fixed.', 'The **Markov property** says the next step depends only on the current state and action. That is what lets everything Pip gathers be filed per cell instead of per path.'],
   shot: [4, 9],
   say: [[["So, an MDP writes a world down. States, actions, chances, rewards, and gamma.", [30, 292, 340, 110]], ["Outcomes can be random, but the chances are fixed.", [30, 292, 340, 110]]], ["And because of Markov, Pip can learn cell by cell, not path by path."]],
   draw(t) { wet(); maze({ t }); pip(cx(0), cy(2), 1, t); fan(0, 2, RIGHT, SLIP, 9, { a: .6 });
     card(['MDP = states, actions, P(s′ | s, a), rewards, γ', 'same action, random outcome, fixed chances', 'Markov: next depends only on state and action', { s: 'so experience can be filed per cell, not per path', c: C.soft }], t - .3, { gap: 1 }); } },
];

function tapRun(x, y, st, t) { const p = SLIPS[st.ci ?? 2], R1 = planRuns(p); st.k = ((st.k ?? -1) + 1) % R1.samples.length; st.run = R1.samples[st.k]; st.rt = t; st.changed = -9; st.prevCi = null; }

RLStory.register('mdp', { title: 'Markov Decision Process', track: 'Foundations', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { N10, N1000, MK, POOL, PATHS, planRuns, outcomes } });
})();
