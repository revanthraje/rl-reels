// POMDPs as a story: a corridor maze built for this reel, in fog. Pip feels only the walls of its own cell, so four corridor
// cells feel identical. A memoryless rule must treat them alike and fails; memory (the last move) or a belief (chances over
// cells, needs the map) fixes it. Every number comes from the seeded runs and exact computations below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha, hachure } = L;
let ctx;
// ───── the fog maze: corridor cells 0..6 (shown as 1..7), rooms below cells 0, 3, 6: cat, cheese, cat
const NC = 7, GAM = 0.9, MAXS = 50, EPS = 0.2, ALPHA = 0.2, NWALK = 300, NRUN = 20;
const UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, MV = [[0, -1], [1, 0], [0, 1], [-1, 0]], MN = ['up', 'right', 'down', 'left'];
const ROOM = x => x === 3 ? 1 : (x === 0 || x === 6) ? -1 : 0;        // reward for stepping down into the room below x (0 = rock, bump)
// one move from corridor cell x: returns [nx, r, done] (nx = x on a bump; done = stepped into a room)
function step(x, m) {
  if (m === UP) return [x, 0, false];
  if (m === DOWN) { const r = ROOM(x); return r ? [x, r, true] : [x, 0, false]; }
  const nx = x + MV[m][0]; return (nx < 0 || nx >= NC) ? [x, 0, false] : [nx, 0, false];
}
// what Pip feels: 0 left end (cell 1), 1 corridor (2, 3, 5, 6), 2 junction (4), 3 right end (7)
const OBS = x => x === 0 ? 0 : x === 3 ? 2 : x === 6 ? 3 : 1, ON = ['left end', 'corridor', 'junction', 'right end'];
const OCELLS = [0, 1, 2, 3].map(o => [0, 1, 2, 3, 4, 5, 6].filter(x => OBS(x) === o));
// ───── exact values of fixed rules (rule: x, last move → list of [move, prob]); a walk is cut after MAXS steps
function evalRule(rule) {          // run every start (deterministic rules) or evaluate exactly (coin flip, memoryless)
  return [0, 1, 2, 3, 4, 5, 6].map(x0 => { let x = x0, last = -1, P = [];
    for (let k = 0; k < MAXS; k++) { const m = rule(x, last); const [nx, r, d] = step(x, m); P.push({ x, m, nx, r, d }); if (d) return { ret: r * GAM ** k, n: k + 1, P, r }; x = nx; last = m; }
    return { ret: 0, n: MAXS, P, r: 0 }; });
}
const SEE = x => x < 3 ? RIGHT : x > 3 ? LEFT : DOWN;                                  // Pip could see its cell
const ML_R = (x) => [RIGHT, RIGHT, DOWN, LEFT][OBS(x)], ML_L = (x) => [RIGHT, LEFT, DOWN, LEFT][OBS(x)];   // memoryless: corridor means right / left
const MEMR = (x, last) => { const o = OBS(x); if (o !== 1) return [RIGHT, 0, DOWN, LEFT][o]; return last === LEFT ? LEFT : RIGHT; }; // keep going; no memory yet: right
const R_SEE = evalRule(SEE), R_R = evalRule(ML_R), R_L = evalRule(ML_L), R_MEM = evalRule(MEMR);
const avg = a => a.reduce((s, v) => s + v, 0) / a.length;
const AV = { see: avg(R_SEE.map(r => r.ret)), right: avg(R_R.map(r => r.ret)), mem: avg(R_MEM.map(r => r.ret)) };
const NOK = R => R.filter(r => r.r > 0).length;
// coin flip in the corridor (memoryless, stochastic): exact values by sweeps
const COIN = (() => { let V = new Array(NC).fill(0); for (let it = 0; it < 3000; it++) { const N = V.slice();
  for (let x = 0; x < NC; x++) { const o = OBS(x), pol = o === 0 ? [[RIGHT, 1]] : o === 2 ? [[DOWN, 1]] : o === 3 ? [[LEFT, 1]] : [[RIGHT, .5], [LEFT, .5]]; let v = 0;
    for (const [m, p] of pol) { const [nx, r, d] = step(x, m); v += p * (r + (d ? 0 : GAM * V[nx])); } N[x] = v; } V = N; } return V; })();
AV.coin = avg(COIN);
// ───── Q-learning (as in the Q-learning reel: ε = 0.2, α = 0.2, ties at random, bumps are moves) on three kinds of input
const REP = { see: (x) => 'c' + x, feel: (x) => 'o' + OBS(x), mem: (x, l) => 'o' + OBS(x) + 'l' + l };
function runQ(rep, seed) {
  const R = rng(seed), Q = {}, q = (s, m) => Q[s + ':' + m] || 0, rets = [];
  for (let e = 0; e < NWALK; e++) { let x = Math.floor(R() * NC), last = -1, ret = 0;
    for (let k = 0; k < MAXS; k++) { const s = rep(x, last); let m;
      if (R() < EPS) m = Math.floor(R() * 4); else { let b = -1e9, ms = []; for (let a = 0; a < 4; a++) { const v = q(s, a); if (v > b + 1e-9) { b = v; ms = [a]; } else if (v > b - 1e-9) ms.push(a); } m = ms[Math.floor(R() * ms.length)]; }
      const [nx, r, d] = step(x, m), s2 = rep(nx, m), tg = r + (d ? 0 : GAM * Math.max(q(s2, 0), q(s2, 1), q(s2, 2), q(s2, 3)));
      Q[s + ':' + m] = q(s, m) + ALPHA * (tg - q(s, m)); if (d) { ret = r * GAM ** k; break; } x = nx; last = m; }
    rets.push(ret); }
  const greedy = (x, l) => { const s = rep(x, l); let b = -1e9, bm = 0; for (let a = 0; a < 4; a++) if (q(s, a) > b + 1e-12) { b = q(s, a); bm = a; } return bm; };
  const ok = NOK(evalRule(greedy));
  return { rets, ok, corr: greedy(2, -1) };
}
const LEARN = (() => { const out = {};
  for (const k in REP) { const curve = new Array(NWALK).fill(0); let ok = 0, all7 = 0, last100 = 0;
    for (let s = 1; s <= NRUN; s++) { const r = runQ(REP[k], 100 + s); r.rets.forEach((v, i) => { curve[i] += v / NRUN; }); ok += r.ok / NRUN; if (r.ok === 7) all7++; last100 += avg(r.rets.slice(-100)) / NRUN; }
    const sm = curve.map((_, i) => avg(curve.slice(Math.max(0, i - 19), i + 1)));      // trailing average over 20 walks
    out[k] = { sm, ok, all7, last100 }; }
  return out; })();
const CLO = Math.floor(Math.min(...Object.values(LEARN).flatMap(l => l.sm)) * 10) / 10;   // bottom of the learning chart
// ───── beliefs: chances over the 7 corridor cells; Pip has the map. acc = chance a look reads the true kind of cell
function predict(b, m) { const n = new Array(NC).fill(0); b.forEach((p, x) => { n[step(x, m)[0]] += p; }); return n; }
function look(b, o, acc = 1) { const n = b.map((p, x) => p * (OBS(x) === o ? acc : (1 - acc) / 3)), s = n.reduce((a, v) => a + v, 0); return n.map(v => v / s); }
const B0 = new Array(NC).fill(1 / NC);
// the story walk: Pip really starts in cell 5 (index 4), feels, moves right, feels, moves right, feels
const SW = (acc, reads) => { const b = [B0]; let cur = look(B0, reads[0], acc); b.push(cur); cur = predict(cur, RIGHT); b.push(cur); cur = look(cur, reads[1], acc); b.push(cur);
  cur = predict(cur, RIGHT); b.push(cur); cur = look(cur, reads[2], acc); b.push(cur); return b; };
const BX = SW(1, [1, 1, 3]);                       // exact whiskers
const BS = SW(0.9, [1, 1, 3]), BW = SW(0.9, [1, 2, 3]);   // shaky whiskers: right readings / second reading wrong (junction)
// ───── formatting
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−');
const pc = v => v >= .995 ? '1' : v < .005 ? '0' : v.toFixed(2).replace(/^0/, '0');

// ───── drawing: geometry
const CS = 50, OX = 25, OY = 40;
const cx = x => OX + x * CS + CS / 2, cy = y => OY + y * CS + CS / 2;
function mazeDraw(o = {}) {
  const t = o.t || 0, p = o.draw ?? 1, ROOMS = [0, 3, 6];
  if (o.bel) o.bel.forEach((b, x) => { if (b > .004) box(OX + x * CS + 2, OY + 2, CS - 4, CS - 4, o.belCol || C.teal, lerp(.12, .75, clamp(b))); });
  if (o.hi) o.hi.forEach(([x, y, col, a]) => box(OX + x * CS + 3, OY + y * CS + 3, CS - 6, CS - 6, col, a ?? .3));
  // corridor: thin cell lines, thick outer walls
  for (let x = 1; x < NC; x++) rline(OX + x * CS, OY, OX + x * CS, OY + CS, 20 + x, { c: C.line, w: 1, p: clamp(p * 2 - .2) });
  rline(OX, OY, OX + NC * CS, OY, 11, { w: 2.2, p: clamp(p * 2) }); rline(OX, OY, OX, OY + CS, 12, { w: 2.2, p: clamp(p * 2) }); rline(OX + NC * CS, OY, OX + NC * CS, OY + CS, 13, { w: 2.2, p: clamp(p * 2) });
  for (let x = 0; x < NC; x++) { const a = OX + x * CS, b = OY + CS;
    if (ROOMS.includes(x)) { rline(a, b, a + CS, b, 40 + x, { c: C.line, w: 1, p: clamp(p * 2 - .5) }); rline(a, b, a, b + CS, 50 + x, { w: 2.2, p: clamp(p * 2 - .6) }); rline(a + CS, b, a + CS, b + CS, 60 + x, { w: 2.2, p: clamp(p * 2 - .6) }); rline(a, b + CS, a + CS, b + CS, 70 + x, { w: 2.2, p: clamp(p * 2 - .7) }); }
    else rline(a, b, a + CS, b, 80 + x, { w: 2.2, p: clamp(p * 2 - .4) }); }
  const pop = o.pop ?? 1;
  if (pop > 0) { L.cheese(cx(3), cy(1) + 2, .62 * pop); L.cat(cx(0), cy(1) + 4, t, .58 * pop); L.cat(cx(6), cy(1) + 4, t + .4, .58 * pop); }
  if (o.nums) for (let x = 0; x < NC; x++) txt(String(x + 1), OX + x * CS + 9, OY + 10, { s: 13, c: C.soft, a: typeof o.nums === 'function' ? o.nums(x) : o.nums });
  if (o.fog) { for (let x = 0; x < NC; x++) for (let y = 0; y < 2; y++) { if (y === 1 && !ROOMS.includes(x)) continue; if (o.see === x && y === 0) continue;
    box(OX + x * CS + 1, OY + y * CS + 1, CS - 2, CS - 2, '#d9d2c9', .88 * o.fog); hachure(OX + x * CS, OY + y * CS, CS, CS, '#bfb6ab', 9, 300 + x * 3 + y, .45 * o.fog); } }
  if (o.bel && o.belNums) o.bel.forEach((b, x) => txt(pc(b), cx(x), OY + CS - 9, { s: 13, c: b > .004 ? '#2f6f6c' : C.soft, a: o.belNums }));
}
function pip(x, y, face, t, a = 1) { alpha(a, () => L.mouse(x, y + 3, face || 1, .68, t)); }
// little icon of what Pip feels: a square, thick lines where walls are
function feel(o, x, y, s = 30, a = 1, col = C.ink) {
  const wall = [[1, 1, 0, 1], [1, 0, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0]][o];       // up, right, down, left (1 = wall)
  const P = [[x - s / 2, y - s / 2, x + s / 2, y - s / 2], [x + s / 2, y - s / 2, x + s / 2, y + s / 2], [x + s / 2, y + s / 2, x - s / 2, y + s / 2], [x - s / 2, y + s / 2, x - s / 2, y - s / 2]];
  alpha(a, () => { box(x - s / 2, y - s / 2, s, s, C.card, 1); P.forEach((q, i) => { if (wall[i]) rline(q[0], q[1], q[2], q[3], 500 + i + o * 4, { w: 2.6, c: col }); else { ctx.save(); ctx.setLineDash([3, 4]); ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(q[0], q[1]); ctx.lineTo(q[2], q[3]); ctx.stroke(); ctx.restore(); } }); dot(x, y + 2, 3, C.grey); });
}
// a walk along a list of steps [{ x, m, nx, d }]: position at time t
function walkAt(P, t, dur) {
  const k = Math.floor(t / dur), u = ease(clamp((t % dur) / dur * 1.5));
  if (k >= P.length) { const l = P[P.length - 1]; return { x: cx(l.nx), y: l.d ? cy(1) : cy(0), k: P.length, done: true, face: l.m === LEFT ? -1 : 1 }; }
  const s = P[k]; if (s.d) return { x: cx(s.x), y: lerp(cy(0), cy(1), u), k, s, u, face: 1 };
  if (s.nx === s.x) { const dy = s.m === UP ? -1 : s.m === DOWN ? 1 : 0, dx = s.m === LEFT ? -1 : s.m === RIGHT ? 1 : 0, b = Math.sin(Math.PI * u) * 9; return { x: cx(s.x) + dx * b, y: cy(0) + dy * b, k, s, u, face: dx || 0 }; }
  return { x: lerp(cx(s.x), cx(s.nx), u), y: cy(0), k, s, u, face: s.nx > s.x ? 1 : -1 };
}
function dim(a = .7, y0 = 0, y1 = 285) { box(0, y0, W, y1 - y0, C.bg, a); }
function rarrow(x, y, m, col, seed, len = 13) { const [dx, dy] = MV[m]; arrow(x - dx * len * .5, y - dy * len * .5, x + dx * len, y + dy * len, seed, { c: col, w: 1.8, head: 6 }); }
// the learning chart: walks 1..300 across, return (trailing average of 20 walks, mean of 20 runs) up
function chart(o) {
  const x0 = 58, x1 = 352, y0 = o.y0 ?? 388, y1 = o.y1 ?? 312, LO = CLO, X = i => x0 + i / (NWALK - 1) * (x1 - x0), Y = v => y0 - (v - LO) / (1 - LO) * (y0 - y1);
  rline(x0 - 4, Y(0), x1 + 4, Y(0), 960, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1 - 4, 961, { c: C.line, w: 1, once: true });
  txt('0', x0 - 13, Y(0), { s: 12, c: C.soft }); txt('1', x0 - 13, y1, { s: 12, c: C.soft }); txt(String(CLO).replace('-', '−'), x0 - 17, y0, { s: 12, c: C.soft }); txt('walk 1', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt('walk ' + NWALK, x1, y0 + 11, { s: 12, c: C.soft, al: 'right' });
  if (o.title) txt(o.title, x0 - 10, y1 - 14, { s: 13, c: C.soft, al: 'left' });
  (o.lines || []).forEach(s => { const n = Math.min(NWALK, s.n ?? NWALK); if (n < 2) return; ctx.save(); ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 1.8; ctx.lineJoin = 'round'; if (s.dash) ctx.setLineDash([4, 4]); ctx.beginPath();
    for (let i = 0; i < n; i++) { const x = X(i), y = Y(s.v[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore();
    if (s.label && n >= NWALK) txt(s.label, x1 + 2, Y(s.v[NWALK - 1]) + (s.dy || 0), { s: 12, c: s.lc || s.col, al: 'right' }); });
  return { X, Y };
}
// a belief bar row under the maze (same numbers as in the cells)
function belCaption(s, a = 1) { txt(s, W / 2, 158, { s: 15, c: '#2f6f6c', a }); }
// your-turn buttons
const BTN = [{ s: '◀ left', x: 70, m: LEFT }, { s: 'down ▼', x: 200, m: DOWN }, { s: 'right ▶', x: 330, m: RIGHT }], BY = 214;
function buttons(a = 1) { BTN.forEach((b, i) => alpha(a, () => { rrect(b.x - 52, BY - 17, 104, 34, 870 + i, { fill: C.card, c: C.line }); txt(b.s, b.x, BY, { s: 17 }); })); }
const ACCS = [1, .9, .75];
function yt(st) { if (st.x != null && st.acc === (st.ci ?? 0)) return st;
  st.acc = st.ci ?? 0; st.walk = (st.walk ?? 0) + 1; const R = rng(4100 + st.walk * 13 + st.acc); st.R = R; st.x = Math.floor(R() * NC);   // a random corridor cell, as the rules say
  st.ended = 0; st.n = 0; st.log = []; const o = readFeel(st); st.b = look(B0, o, ACCS[st.acc]); st.log.push(o); st.at = -9; return st; }
function readFeel(st) { const tr = OBS(st.x), acc = ACCS[st.acc]; if (st.R() < acc) return tr; const others = [0, 1, 2, 3].filter(o => o !== tr); return others[Math.floor(st.R() * 3)]; }

// the memoryless loop (corridor means right) from cell 5, shown for 8 steps
const LOOP = R_R[4].P.slice(0, 8);
const MEMW = R_MEM[4].P;                                          // memory rule from cell 5
const KEEP = (b, i) => b.map((v, x) => v > .004 ? x + 1 : 0).filter(Boolean);

const STEPS = [
 { h: 'A maze built for fog', p: ['A new maze, built to show one problem: a corridor of seven cells, numbered 1 to 7, with three rooms below: the cheese under cell 4, cats under cells 1 and 7.', 'The usual rules: +1 for stepping onto the cheese, −1 onto a cat, either ends the walk, γ (gamma) 0.9. Each walk starts in one of the seven cells, picked at random.'],
   shot: [3, 9],
   draw(t) { mazeDraw({ t, draw: t / 1.5, pop: ease((t - 1.6) * 2), nums: clamp(t - 2.4) }); pip(cx(1), cy(0), 1, t, clamp(t - 3));
     if (t > 4) card(['corridor cells 1 to 7, three rooms below', { s: 'onto the cheese +1, onto a cat −1: walk over', c: C.goldInk }, 'up, or down into a wall: a bump, Pip stays', { s: 'γ = 0.9; start: any of cells 1 to 7, at random', c: C.soft }], t - 4, { gap: 1.2 }); } },
 { h: 'Fog', p: ['Now the fog comes in. Pip sees only its own cell, and even there only which sides are walls and which are open. No cheese, no cats, no cell numbers.', 'What Pip senses is its **observation**. Where it really is, the cell, is still the **state**, but Pip cannot see it.'],
   shot: [3, 9],
   draw(t) { const f = clamp((t - .8) / 1.5); mazeDraw({ t, nums: 1 - f, fog: f, see: 1 }); pip(cx(1), cy(0), 1, t);
     if (t > 3) { feel(1, 200, 210, 54, clamp(t - 3)); txt('what Pip feels in cell 2', 200, 168, { s: 15, c: C.soft, a: clamp(t - 3) }); txt('walls above and below, open left and right', 200, 258, { s: 14, c: C.soft, a: clamp(t - 3.6) }); }
     if (t > 5) card([{ s: 'state: the cell Pip is in (hidden)', c: '#3f7f7c' }, { s: 'observation: what Pip can sense right now', c: C.goldInk }, { s: 'here: walls or openings on its four sides', c: C.soft }], t - 5, { gap: 1.2 }); } },
 { h: 'Only four feelings', p: ['Walking the corridor, Pip can feel only four different things. Cell 1: a wall on the left, an opening below. Cell 7: the mirror image. Cell 4: open left, right and below.', 'And cells 2, 3, 5 and 6 all feel exactly the same: walls above and below, open on both sides. Call that feeling **corridor**.'],
   shot: [4, 11],
   draw(t) { mazeDraw({ t, nums: 1, fog: .35 }); const xs = [62, 160, 250, 338];
     [0, 3, 2, 1].forEach((o, i) => { const u = clamp(t - .8 - i * 1.4); if (u <= 0) return; feel(o, xs[i], 196, 34, u); txt(ON[o], xs[i], 228, { s: 14, a: u, c: o === 1 ? C.roseInk : C.ink });
       txt('cell' + (OCELLS[o].length > 1 ? 's ' : ' ') + OCELLS[o].map(x => x + 1).join(', '), xs[i], 248, { s: 13, a: u, c: C.soft });
       OCELLS[o].forEach(x => alpha(u * .9, () => box(OX + x * CS + 3, OY + 3, CS - 6, CS - 6, o === 1 ? C.rose : C.gold, .28))); });
     if (t > 6.5) card([{ s: '7 cells, but only 4 different feelings', c: C.ink }, { s: 'corridor = cell 2, 3, 5 or 6: Pip cannot tell which', c: C.roseInk }], t - 6.5, { gap: 1.2 }); } },
 { h: 'Twins that need opposite moves', p: ['Cells 3 and 5 feel identical, yet from cell 3 the cheese is to the right, and from cell 5 to the left. Likewise cells 2 and 6.', 'So what Pip feels does not tell it what to do. In the MDP reel’s words, the observation is **not Markov**: what happens next depends on more than what Pip can sense.'],
   shot: [3, 9],
   draw(t) { mazeDraw({ t, nums: 1, hi: [[2, 0, C.rose, .25 * clamp(t - .5)], [4, 0, C.rose, .25 * clamp(t - .5)]] });
     if (t > 1.5) { const u = clamp(t - 1.5); alpha(u, () => { rarrow(cx(2), cy(0) + 6, RIGHT, C.goldInk, 610); rarrow(cx(4), cy(0) + 6, LEFT, C.goldInk, 615); rarrow(cx(1), cy(0) + 6, RIGHT, C.goldInk, 620); rarrow(cx(5), cy(0) + 6, LEFT, C.goldInk, 625); }); }
     if (t > 3) { feel(1, 130, 200, 40, clamp(t - 3)); feel(1, 270, 200, 40, clamp(t - 3)); txt('cell 3', 130, 236, { s: 14, a: clamp(t - 3) }); txt('cell 5', 270, 236, { s: 14, a: clamp(t - 3) }); txt('=', 200, 200, { s: 28, a: clamp(t - 3.6) });
       txt('best: right', 130, 258, { s: 14, c: C.goldInk, a: clamp(t - 4.2) }); txt('best: left', 270, 258, { s: 14, c: C.goldInk, a: clamp(t - 4.2) }); }
     if (t > 6) card(['same feeling, opposite best moves', { s: 'the observation alone is not Markov', c: C.roseInk }], t - 6, { gap: 1.2 }); } },
 { h: 'A name for it: POMDP', p: ['Underneath, nothing changed: states, moves, rewards and γ are still an MDP. Add one thing, which observation each state gives, and you have a **partially observable** MDP: a **POMDP**, said “pom-dee-pee”.', 'The true state is still Markov. It is just hidden behind the observations.'],
   shot: [3, 9],
   draw(t) { mazeDraw({ t, nums: 1, fog: .2 });
     card(['MDP: states, moves, P(s′ | s, a), rewards, γ', { s: '+ observations: what each state lets Pip sense', c: C.goldInk }, { s: '= POMDP: the state is there, but hidden', c: '#3f7f7c' }], t - .6, { y: 170, gap: 1.5 });
     if (t > 5) card([{ s: 'here: cell 3 → “corridor”, cell 4 → “junction”', c: C.soft }, { s: 'Pip gets “corridor”, never “cell 3”', c: C.soft }], t - 5, { y: 300, gap: 1.2 }); } },
 { h: 'A rule from what Pip feels', p: ['The simplest policy maps the current observation straight to a move. It has no memory, so it is called **memoryless**.', 'Three entries are easy: left end, go right; junction, go down; right end, go left. But a memoryless rule has one entry for “corridor”, so all four corridor cells must get the same move.'],
   shot: [3, 10],
   draw(t) { mazeDraw({ t, fog: .6 }); const rows = [[0, 'go right'], [2, 'go down'], [3, 'go left'], [1, '?']];
     rows.forEach(([o, s], i) => { const u = clamp(t - .8 - i * 1.1); if (u <= 0) return; const x = 50 + i * 92; feel(o, x + 18, 196, 30, u); txt(ON[o], x + 18, 226, { s: 13, a: u, c: C.soft }); txt(s, x + 18, 250, { s: 16, a: u, c: o === 1 ? C.roseInk : C.goldInk }); });
     if (t > 5.5) card(['memoryless policy: observation → move', { s: 'one entry for “corridor”, shared by cells 2, 3, 5, 6', c: C.roseInk }], t - 5.5, { gap: 1.2 }); } },
 { h: 'Corridor means right', p: [`Try “corridor: go right”. Cells 1 to 4 work. But start in cell 5: right to 6, right to 7, the right end says left, back to 6, which feels like corridor: right again. Pip shuttles **forever**.`, `So it reaches the cheese from ${NOK(R_R)} of 7 starts. “Corridor: go left” fails on the other side, also ${NOK(R_L)} of 7. Up and down only bump.`],
   shot: [4, 11],
   draw(t) { const D = .9, m = walkAt(LOOP, Math.max(0, t - .8), D), k = Math.min(LOOP.length - 1, m.k); mazeDraw({ t, nums: 1, fog: .45, see: m.done ? 5 : LOOP[k].x }); pip(m.x, m.y, m.face, t);
     if (t > .8) { const s = LOOP[k]; feel(OBS(s.x), 200, 196, 34, 1); txt(`feels ${ON[OBS(s.x)]}  →  ${MN[s.m]}`, 200, 236, { s: 16, c: OBS(s.x) === 1 ? C.roseInk : C.goldInk }); }
     caption(m.done ? 'and so on, forever (we stop a walk after 50 steps)' : `step ${Math.min(LOOP.length, m.k + 1)}`);
     if (t > 9) card([`corridor → right:  cheese from ${NOK(R_R)} of 7 starts (1 to 4)`, `corridor → left:  cheese from ${NOK(R_L)} of 7 starts (4 to 7)`], t - 9, { gap: 1.1 }); } },
 { h: 'Roll a die?', p: [`One trick helps a little: in the corridor, flip a coin: left or right. Now every start reaches the cheese, with dithering. Average return over the 7 starts: ${f2(AV.coin)}, against ${f2(AV.right)} for “always right”.`, `Seeing its cell, Pip would go straight: ${f2(AV.see)}. No memoryless rule can. With the full state a fixed rule always suffices; under fog, a random one can beat every fixed one.`],
   shot: [3, 10],
   draw(t) { mazeDraw({ t, nums: 1, fog: .3 });
     const items = [['always right', AV.right, C.rose], ['coin flip in the corridor', AV.coin, C.orange], ['if Pip could see its cell', AV.see, C.teal]];
     txt('average return over the 7 starts', 40, 162, { s: 14, c: C.soft, al: 'left' });
     items.forEach(([s, v, col], i) => { const u = clamp(t - .8 - i * 1.4); if (u <= 0) return; const y = 192 + i * 30; txt(s, 40, y, { s: 15, al: 'left', a: u });
       box(210, y - 9, 150 * v * ease(u), 18, col, .8); txt(f2(v), 214 + 150 * v * ease(u), y, { s: 15, al: 'left', a: u }); });
     if (t > 6) card([{ s: 'a stochastic rule can beat every fixed one here', c: C.ink }, { s: 'but no rule on observations alone sees the cell', c: C.soft }], t - 6, { gap: 1.2 }); } },
 { h: 'Fix one: remember', p: ['Give Pip a one-item memory: **the move it just made**. Now “corridor, going right” and “corridor, going left” are different situations, each with its own entry.', `Rule: in the corridor keep going the way you were; with no memory yet, go right. From cell 5: right, right, the end turns it around, then left, left, left, down. Cheese in ${MEMW.length} steps.`],
   shot: [4, 12],
   draw(t) { const D = .9, m = walkAt(MEMW, Math.max(0, t - 1.2), D), k = Math.min(MEMW.length - 1, m.k); mazeDraw({ t, nums: 1, fog: .45, see: m.done ? -1 : MEMW[k].x }); pip(m.x, m.y, m.face, t);
     if (t > 1.2) { const s = MEMW[k], last = k ? MN[MEMW[k - 1].m] : 'none yet'; feel(OBS(s.x), 120, 200, 34, 1); txt(ON[OBS(s.x)], 120, 232, { s: 14, c: C.soft });
       txt('+', 170, 200, { s: 22 }); txt(`last move: ${last}`, 260, 200, { s: 16, c: '#3f7f7c' }); txt(`→ ${MN[s.m]}`, 260, 228, { s: 17, c: C.goldInk }); }
     if (m.done) bubble('+1', cx(3), cy(1) - 14, t - 1.2 - MEMW.length * D);
     if (t > 8.5) card([`memory rule: cheese from ${NOK(R_MEM)} of 7 starts`, { s: `average return ${f2(AV.mem)}   (seeing the cell: ${f2(AV.see)})`, c: C.soft }, { s: 'no memory on move 1: cells 5 and 6 detour', c: C.soft }], t - 8.5, { gap: 1.2 }); } },
 { h: 'Learning in the fog', p: [`Can Q-learning find them? Same settings: one move in five random, α (alpha) 0.2. Across: walks. Up: return, averaged over the last 20 walks and ${NRUN} runs.`, `Grey dashed: Pip seeing its cell. Lines start below 0: early walks often meet a cat. Rose, learning from feelings, stalls near ${f2(LEARN.feel.last100)}: its one corridor entry mixes opposite-move twins, and a greedy rule cannot flip a coin.`],
   shot: [4, 12],
   draw(t) { mazeDraw({ t, nums: 1, fog: .45 }); const n1 = clamp(Math.floor((t - 2.5) * 80), 0, NWALK), n2 = clamp(Math.floor((t - 6.5) * 80), 0, NWALK);
     txt('Q-learning, ε 0.2, α 0.2, γ 0.9', 200, 170, { s: 15, c: C.soft });
     chart({ y0: 388, y1: 222, title: `return, last 20 walks, mean of ${NRUN} runs`, lines: [{ v: LEARN.see.sm, col: '#9a9187', dash: 1, n: n1, label: 'sees its cell', lc: C.soft, dy: -9 }, { v: LEARN.feel.sm, col: C.rose, n: n2, label: 'feels only', lc: C.roseInk, dy: 10 }] }); } },
 { h: 'Memory pays, slowly', p: [`Teal learns from what Pip feels plus its last move. At first no better, with more situations to fill; then it pulls ahead: ${f2(LEARN.mem.last100)} against ${f2(LEARN.feel.last100)} over the last 100 walks.`, `After ${NWALK} walks, the rose runs’ best moves reach the cheese from ${LEARN.feel.ok.toFixed(1)} of 7 starts on average; the teal runs’, ${LEARN.mem.ok.toFixed(2)}. All lines include random exploring moves, so even grey stays below ${f2(AV.see)}.`],
   shot: [4, 12],
   draw(t) { const n3 = clamp(Math.floor((t - 1) * 70), 0, NWALK);
     chart({ y0: 388, y1: 222, title: `return, last 20 walks, mean of ${NRUN} runs`, lines: [{ v: LEARN.see.sm, col: '#9a9187', dash: 1, label: 'sees its cell', lc: C.soft, dy: -9 }, { v: LEARN.feel.sm, col: C.rose, label: 'feels only', lc: C.roseInk, dy: 10 }, { v: LEARN.mem.sm, col: '#3f9a96', w: 2.2, n: n3, label: 'feels + last move', lc: '#2f6f6c', dy: -9 }] });
     if (t > 6) card([`best moves after ${NWALK} walks, cheese from:`, { s: `feels only: ${LEARN.feel.ok.toFixed(1)} of 7 starts on average`, c: C.roseInk }, { s: `feels + last move: ${LEARN.mem.ok.toFixed(2)} of 7 (${LEARN.mem.all7} of ${NRUN} runs: all 7)`, c: '#2f6f6c' }], t - 6, { y: 40, gap: 1.2 }); } },
 { h: 'Fix two: a belief', p: ['Fix two needs more knowledge. Give Pip **the map**: where each move leads and what each cell feels like, but not its own position. Then it can keep a **belief**: its chance of each cell, shown as shading.', `At first, 1 in 7 per cell, about ${f2(1 / 7)}. Then Pip feels “corridor”, which only cells 2, 3, 5 and 6 feel like: 1 in 4 each.`],
   shot: [3, 10],
   draw(t) { const b = t < 5 ? BX[0] : BX[1], u = t < 5 ? 1 : ease((t - 5) / .8), bel = BX[0].map((v, x) => lerp(v, BX[1][x], t < 5 ? 0 : u));
     mazeDraw({ t, nums: 1, fog: .25, bel, belNums: 1 }); pip(cx(4), cy(0) - 2, 1, t, .25); txt('true cell (Pip can’t tell)', cx(4), OY - 12, { s: 12, c: C.soft, a: .8 });
     belCaption(t < 5 ? 'start: every cell equally likely' : 'feels “corridor”: only 2, 3, 5, 6 fit', clamp(t - 1));
     if (t > 4.6) { feel(1, 200, 210, 34, clamp(t - 4.6)); }
     card(['belief = a chance for every cell, adding up to 1', { s: 'needs the map: what each cell feels like', c: C.soft }], t - 7, { gap: 1.2 }); } },
 { h: 'Move, then look', p: ['Pip moves right. Not knowing from where, it shifts **every** chance one cell right: cell 2’s to 3, cell 6’s to 7, and so on.', 'Then it feels corridor again. Cells 4 and 7 do not feel like that, so their chances are wiped out. Cells 3 and 6 remain, scaled back up to add to 1: one half each.'],
   shot: [5.5, 8.5, 12],
   draw(t) { let bel, cap; if (t < 3) { bel = BX[1]; cap = 'before: 2, 3, 5, 6'; } else if (t < 7) { const u = ease((t - 3) / 1.2); bel = BX[1].map((v, x) => lerp(v, BX[2][x], u)); cap = 'after moving right: 3, 4, 6, 7'; } else { const u = ease((t - 7) / 1.2); bel = BX[2].map((v, x) => lerp(v, BX[3][x], u)); cap = 'feels “corridor”: only 3 and 6 fit'; }
     const pm = walkAt([{ x: 4, m: RIGHT, nx: 5 }], clamp(t - 3, 0, 1), 1); mazeDraw({ t, nums: 1, fog: .25, bel, belNums: 1 }); pip(pm.x, cy(0) - 2, 1, t, .25);
     if (t > 3 && t < 6.5) for (let x = 1; x < 6; x++) if (BX[1][x] > .01) alpha(clamp((t - 3) * 2) * (1 - clamp((t - 5.8) * 2)), () => arrow(cx(x) - 4, OY + CS + 10, cx(x + 1) - 8, OY + CS + 10, 640 + x, { c: '#2f6f6c', w: 1.5, head: 5 }));
     belCaption(cap); if (t > 6.6) feel(1, 200, 210, 34, clamp(t - 6.6));
     card(['1. move: shift every chance the way the move goes', '2. look: keep the cells that feel like this, rescale'], t - 1, { gap: 5.5 }); } },
 { h: 'Certain', p: ['Pip moves right again: chances shift to 4 and 7. It feels the **right end**, unique to cell 7: now Pip is certain.', `The map’s plan: left, left, left, down. Moving right led away from the cheese, but bought **information**: ${2 + 4} moves in all, as many as the memory rule. “Go right until sure” was hand-made; generally, planning from a belief is its own problem.`],
   shot: [3.5, 7, 13],
   draw(t) { let bel; if (t < 2.5) { const u = ease((t - .6) / 1.2); bel = BX[3].map((v, x) => lerp(v, BX[4][x], u)); } else { const u = ease((t - 3) / 1); bel = BX[4].map((v, x) => lerp(v, BX[5][x], u)); }
     const P = [{ x: 5, m: RIGHT, nx: 6 }], pm = walkAt(P, clamp(t - .6, 0, 1), 1);
     const plan = [{ x: 6, m: LEFT, nx: 5 }, { x: 5, m: LEFT, nx: 4 }, { x: 4, m: LEFT, nx: 3 }, { x: 3, m: DOWN, nx: 3, d: true }], q = t > 7 ? walkAt(plan, t - 7, .9) : null;
     const pbel = q ? bel.map((_, x) => x === Math.round((q.x - OX - CS / 2) / CS) ? 1 : 0) : bel;
     mazeDraw({ t, nums: 1, fog: .25, bel: q && q.done ? null : pbel, belNums: q && q.done ? 0 : 1, pop: q ? 1 : 0 }); if (!q) pip(pm.x, cy(0) - 2, 1, t, .25); else { pip(q.x, q.y - 2, q.face, t, 1); if (q.done) bubble('+1', cx(3), cy(1) - 14, t - 7 - 3.6); }
     belCaption(t < 2.5 ? 'after moving right: 4 or 7' : t < 7 ? 'feels the right end: cell 7, chance 1' : 'certain: follow the map', 1); if (t > 2.5 && t < 7) feel(3, 200, 210, 34, clamp(t - 2.5));
     if (t > 4.5 && t < 7) txt('belief collapsed: Pip knows its cell', 200, 250, { s: 15, c: C.goldInk, a: clamp(t - 4.5) });
     if (t > 7.5) card(['moving right earned nothing directly', { s: 'but it told Pip where it was: information', c: C.goldInk }], t - 7.5, { gap: 1.2 }); } },
 { h: 'Shaky whiskers', p: [`Say each feeling is right 9 times in 10, else a random other one; moves never skid. Looks now lower chances, not wipe them out. Same readings: ${f2(BS[5][6])} sure of cell 7.`, `Had the middle reading wrongly said “junction”, cell 7 still gets ${f2(BW[5][6])}: starts in 5 or 6 explain it, as right from 7 bumps. Cell 5 gets ${f2(BW[5][4])}: a start in 3, only the last reading wrong.`],
   shot: [4, 11],
   draw(t) { const A = t < 6.5, B = A ? BS[5] : BW[5]; mazeDraw({ t, nums: 1, fog: .25, bel: B, belNums: 1 }); pip(cx(6), cy(0) - 2, -1, t, .25);
     const rd = A ? [1, 1, 3] : [1, 2, 3]; txt('readings:', 46, 176, { s: 15, c: C.soft, al: 'left' });
     rd.forEach((o, i) => { feel(o, 150 + i * 70, 178, 28, clamp(t - .5 - i * .5), !A && i === 1 ? C.roseInk : C.ink); txt(ON[o], 150 + i * 70, 204, { s: 12, c: !A && i === 1 ? C.roseInk : C.soft, a: clamp(t - .5 - i * .5) }); });
     txt(A ? 'right, right in between; whiskers right 9 in 10' : 'middle reading wrong (it was corridor)', 200, 232, { s: 14, c: A ? C.soft : C.roseInk });
     card(['look: chance × (0.9 if it feels like this, else 0.033)', { s: 'then rescale so the chances add up to 1', c: C.soft }], t - 1.5, { gap: 1.2 }); } },
 { h: 'Your turn', ctl: { label: 'whiskers right', values: ACCS, def: 0, fmt: v => Math.round(v * 100) + '%', onChange: st => { st.x = null; } },
   p: ['Pip starts in a hidden random cell, any of 1 to 7. Tap **left**, **right** or **down** to move it, and watch the belief after each move and feel. Down ends the walk where there is a room. Tap the card for a new walk.', 'Use the slider to make the whiskers unreliable, and see how much longer it takes to be sure.'],
   tap(x, y, st, t) { yt(st); if (y > 290) { st.x = null; return; }
     if (st.ended || y < BY - 22 || y > BY + 22) return; const b = BTN.find(b => Math.abs(x - b.x) < 56); if (!b) return;
     const [nx, r, d] = step(st.x, b.m); st.n++; st.at = t; st.lastM = b.m;
     if (d) { st.ended = r; return; } st.x = nx; st.b = predict(st.b, b.m); const o = readFeel(st); st.log.push(o); st.b = look(st.b, o, ACCS[st.acc]); },
   shot: [3, 8],
   draw(t, st) { yt(st); const bm = st.b.indexOf(Math.max(...st.b));
     mazeDraw({ t, nums: 1, fog: .25, bel: st.ended ? null : st.b, belNums: st.ended ? 0 : 1 }); pip(cx(st.x), (st.ended ? cy(1) : cy(0)) - 2, st.lastM === LEFT ? -1 : 1, t, st.ended ? 1 : .3);
     if (st.ended) { txt(st.ended > 0 ? 'cheese! +1' : 'a cat! −1', 200, 168, { s: 20, c: st.ended > 0 ? C.goldInk : C.roseInk }); }
     else { const o = st.log[st.log.length - 1]; feel(o, 120, 166, 26, 1); txt(`Pip felt: ${ON[o]}`, 145, 166, { s: 15, al: 'left', c: o === OBS(st.x) ? C.ink : C.roseInk }); }
     buttons(st.ended ? .3 : 1);
     card(st.ended ? [`walk over after ${st.n} moves`, { s: 'tap here for a new walk', c: C.soft }] : [`moves: ${st.n}    ` + (st.b.filter(v => v > st.b[bm] - 1e-6).length > 1 ? `tied: cells ${st.b.map((v, x) => v > st.b[bm] - 1e-6 ? x + 1 : 0).filter(Boolean).join(', ')}` : `most likely: cell ${bm + 1}, chance ${f2(st.b[bm])}`), { s: `faint mouse = true cell; whiskers right ${Math.round(ACCS[st.acc] * 100)}%`, c: C.soft }, { s: 'tap here for a new walk', c: C.soft }], 5, { gap: 0 }); } },
 { h: 'Memory or belief?', p: ['Built with the map from every move and look, a belief holds all history says of Pip’s place, so choosing from it is Markov again. But it needs the map, and one number per state: hopeless for big worlds.', 'Memory needs no map: deep RL often uses a **recurrent network**, like an LSTM, reading one observation at a time; ordinary training shapes what it keeps.'],
   shot: [3, 9],
   draw(t) { mazeDraw({ t, nums: 1, fog: .3, bel: BX[3], belNums: .8 });
     card([{ s: 'belief: needs the map, exact, one number per state', c: '#2f6f6c' }, { s: 'memory: no map, learns what to keep', c: C.goldInk }, { s: 'recurrent network: memory learned from data', c: C.soft }], t - .8, { y: 176, gap: 1.4 }); } },
 { h: 'Where you meet fog', p: ['In poker you cannot see the other cards, so good players and programs reason about what an opponent might hold, given their bets: a belief.', 'A self-driving car that loses sight of a cyclist behind a bus must remember the cyclist is probably still there. In the DQN reel, the Atari player took in four recent screens: a short memory, so motion could be seen.'],
   shot: [3, 9],
   draw(t) { mazeDraw({ t, nums: 0, fog: .7 }); pip(cx(2), cy(0), 1, t, 1);
     card(['poker: hidden cards → a belief about hands', 'driving: a hidden cyclist → remember it', { s: 'Atari: last 4 screens → a short memory', c: C.soft }], t - .8, { y: 176, gap: 1.4 }); } },
 { h: 'Recap', p: ['In a **POMDP** the agent gets observations, not the state. Different states can look the same, so a rule on observations alone can fail, as Pip shuttled in the corridor.', 'Two fixes: **memory** of recent observations and moves, which Q-learning can learn without a map; or a **belief**, chances over states, updated after every move and look, which needs the map.'],
   shot: [3, 9],
   draw(t) { mazeDraw({ t, nums: 1, fog: .3, bel: BX[3], belNums: .6 });
     card(['observation ≠ state: twins look the same', 'memoryless rules can loop forever', 'memory: remember what you saw and did', { s: 'belief: move → shift chances, look → reweigh', c: '#2f6f6c' }], t - .3, { y: 176, gap: 1 }); } },
];

RLStory.register('pomdp', { title: 'Partial Observability (POMDPs)', track: 'Bigger worlds', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { R_SEE, R_R, R_L, R_MEM, AV, COIN, LEARN, BX, BS, BW, NOK } });
})();
