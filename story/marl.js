// Multi-agent RL as a story: Pip and a second mouse, Tam, learn side by side. From each one's view the other is part of
// the world, and that world keeps changing as the other learns. Two one-room games: the door game (team: big door needs
// both, small hole is safe) and the cat game (opponents: same door = caught). All numbers come from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const EPS = 0.2, ALPHA = 0.2;
// ───── the learners: two numbers each (guess of the payoff of each door), eps-greedy, ties by coin toss, nudge 1/5 of the way
// door 0 = left (big door / left door), door 1 = right (small hole / right door)
const TEAM = (a, b) => [a === 0 ? (b === 0 ? 1 : 0) : 0.3, b === 0 ? (a === 0 ? 1 : 0) : 0.3];
const ZERO = (a, b) => (a === b ? [-1, 1] : [1, -1]);              // [Pip, cat]: same door = caught
function pickWith(R, q) { if (R() < EPS) return [Math.floor(R() * 2), true]; if (Math.abs(q[0] - q[1]) < 1e-12) return [Math.floor(R() * 2), false]; return [q[0] > q[1] ? 0 : 1, false]; }
function play(seed, n, game) {
  const R = rng(seed), Q = [[0, 0], [0, 0]], out = [];
  for (let k = 0; k < n; k++) {
    const qb = Q.map(q => q.slice()), [a, ra] = pickWith(R, Q[0]), [b, rb] = pickWith(R, Q[1]), [r1, r2] = game(a, b);
    Q[0][a] += ALPHA * (r1 - Q[0][a]); Q[1][b] += ALPHA * (r2 - Q[1][b]);
    out.push({ a, b, ra, rb, r1, r2, qb, q: Q.map(q => q.slice()) });
  }
  return out;
}
const fav = q => (Math.abs(q[0] - q[1]) < 1e-12 ? -1 : q[0] > q[1] ? 0 : 1);
const pLeft = q => (fav(q) < 0 ? 0.5 : fav(q) === 0 ? 1 - EPS / 2 : EPS / 2);   // chance of picking the left door, from the numbers

// door game, story pair (seed 28), 200 rounds
const NR = 200, SEED = 28, T = play(SEED, NR, TEAM);
const R1 = T[0], FIRSTX = T.findIndex(x => x.a === 0);                          // Pip's first try at the big door (random)
const LL = T.map((x, i) => (x.a === 0 && x.b === 0 ? i : -1)).filter(i => i >= 0), LL1 = LL[0], LL2 = LL[1];
const LONE = T.slice(0, LL1).filter(x => x.a === 0).length;                      // Pip's lone tries before the first double
const SWITCH = T.findIndex(x => fav(x.q[0]) === 0 && fav(x.q[1]) === 0);          // round index after which both prefer the big door
const MID = T.slice(LL1 + 1, LL2);                                                // between the two doubles
const PWORLD = T.map(x => pLeft(x.qb[1]));                                        // chance the big door pays for Pip in each round
const FINQ = T[NR - 1].q;
// 100 pairs (seeds 1..100): which door both prefer after each round
const NP = 100, PAIRS = Array.from({ length: NP }, (_, i) => play(i + 1, NR, TEAM).map(x => { const f = fav(x.q[0]), g = fav(x.q[1]); return f === g ? f : -2; }));
const countAt = n => { let big = 0, hole = 0, other = 0; PAIRS.forEach(p => { const v = p[n - 1]; if (v === 0) big++; else if (v === 1) hole++; else other++; }); return { big, hole, other }; };
const C200 = countAt(NR), C50 = countAt(50);
// cat game: Pip vs the cat, both independent learners (seed 5), 200 rounds
const HSEED = 5, H = play(HSEED, NR, ZERO);
const HSW = H.reduce((s, x, i) => s + (i && fav(x.q[0]) !== fav(H[i - 1].q[0]) ? 1 : 0), 0);
const HLEFT = H.filter(x => x.a === 0).length, HCAUGHT = H.filter(x => x.r1 < 0).length;
// your turn: Pip learns against your door choices (cat = you). Pip's dice: rng(PSEED), one round at a time
const PSEED = 11;
function vsYou(cats) { const R = rng(PSEED), q = [0, 0], out = []; for (const c of cats) { const qb = q.slice(), [a] = pickWith(R, q), r = a === c ? -1 : 1; q[a] += ALPHA * (r - q[a]); out.push({ a, c, r, qb, q: q.slice() }); } return out; }
const ALWAYSL = vsYou(new Array(30).fill(0)), AL_ESC = ALWAYSL.filter(x => x.r > 0).length, AL_ESC20 = ALWAYSL.slice(10).filter(x => x.r > 0).length;
const AL_AVG = (() => { let c = 0; for (let s = 1; s <= 1000; s++) { const R = rng(5000 + s), q = [0, 0]; for (let i = 0; i < 30; i++) { const [a] = pickWith(R, q), r = a === 0 ? -1 : 1; if (r < 0) c++; q[a] += ALPHA * (r - q[a]); } } return c / 1000; })();   // always-left cat vs 1000 Pips with other dice
const COINCAT = (() => { const R = rng(99), cats = []; for (let i = 0; i < 1000; i++) cats.push(R() < .5 ? 0 : 1); return vsYou(cats); })(), COIN_ESC = COINCAT.filter(x => x.r > 0).length;
const f2 = v => (Math.abs(v) < .005 ? '0' : String(+v.toFixed(2)));
const fx = v => v.toFixed(2);

// ───── drawing: characters
function tam(x, y, face, s, t) { L.mouse(x, y, face, s, t); const ex = x + face * 4 * s, ey = y + Math.sin(t * 9) * 1.2 - 14 * s;
  dot(ex - 3.5, ey, 3, C.rose); dot(ex + 3.5, ey, 3, C.rose); dot(ex, ey, 1.8, C.roseInk); }
function catA(x, y, s = 1, t = 0) {
  ctx.save(); ctx.translate(x, y + Math.sin(t * 5) * .8); ctx.scale(s, s);
  ctx.beginPath(); ctx.moveTo(-13, -4); ctx.lineTo(-11, -17); ctx.lineTo(-4, -9); ctx.moveTo(13, -4); ctx.lineTo(11, -17); ctx.lineTo(4, -9);
  ctx.strokeStyle = C.ink; ctx.lineWidth = 1.3; ctx.fillStyle = '#c9b9a8'; ctx.fill(); ctx.stroke();
  L.rcircle(0, 0, 13, 41, { fill: '#c9b9a8', j: .8 });
  ctx.fillStyle = C.ink; ctx.beginPath(); ctx.arc(-5, -2, 2.1, 0, 7); ctx.fill(); ctx.beginPath(); ctx.arc(5, -2, 2.1, 0, 7); ctx.fill();
  [[-1, 5, -16, 3], [-1, 6, -16, 8], [1, 5, 16, 3], [1, 6, 16, 8]].forEach(([a, b, c, d]) => rline(a, b, c, d, a * 7 + d, { once: true, w: 1, j: .5 }));
  ctx.restore();
}
function name(s, x, y, a = 1) { txt(s, x, y, { s: 13, c: C.soft, a }); }
// ───── door game scene
const BD = { x: 52, y: 58, w: 112, h: 122 }, HOLE = { x: 300, y: 180, r: 24 }, FLOOR = 180;
const HOME = { pip: [150, 248], tam: [262, 248] }, FRONT = { pip: [[80, 226], [272, 226]], tam: [[140, 226], [334, 226]] };
function wall() {
  box(20, 30, 360, FLOOR - 30, '#efe4d3', .7); L.hachure(20, 30, 360, FLOOR - 30, '#d8c9b4', 14, 7, .35);
  rline(14, FLOOR, 386, FLOOR, 5, { w: 1.8 });
}
function bigDoor(lift, lab = 1) {
  const { x, y, w, h } = BD; box(x, y, w, h, '#5a514a', .55); if (lift > 0) L.cheese(x + w / 2, y + h - 30, 1.7);
  const hh = h * (1 - .82 * lift); box(x, y, w, hh, '#c9a27a');
  for (let k = 1; k < 4; k++) rline(x + k * w / 4, y + 2, x + k * w / 4, y + hh - 2, 40 + k, { c: '#9c7a55', w: 1, once: true });
  rrect(x, y, w, hh, 30, { w: 1.5 }); rrect(x, y, w, h, 34, { w: 1.2, c: C.line });
  if (lab) txt('big door: needs both', x + w / 2, y - 14, { s: 14, c: C.goldInk, a: lab });
}
function hole(lab = 1) {
  const { x, y, r } = HOLE; ctx.save(); ctx.fillStyle = '#4a423b'; ctx.beginPath(); ctx.arc(x, y, r, Math.PI, 0); ctx.closePath(); ctx.fill(); ctx.restore();
  rline(x - r, y, x - r * .7, y - r * .72, 61, { once: true }); rline(x + r * .7, y - r * .72, x + r, y, 62, { once: true });
  bcrumb(x + 42, y - 6); if (lab) txt('small hole: crumb 0.3', x + 8, y - r - 18, { s: 14, c: '#3f7f7c', a: lab });
}
function bcrumb(x, y) { ctx.save(); ctx.translate(x, y); ctx.beginPath(); ctx.moveTo(-7, 4); ctx.lineTo(7, 4); ctx.lineTo(7, -3); ctx.closePath(); ctx.fillStyle = '#f7d36b'; ctx.fill(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.1; ctx.stroke(); ctx.restore(); }
// one round of the door game. rec = { a, b, r1, r2 } or null; u = seconds into the round
function teamScene(rec, u, t, o = {}) {
  wall(); const both = rec && rec.a === 0 && rec.b === 0, out = u - 1.0;
  bigDoor(both ? ease(out / .6) : 0, o.lab ?? 1); hole(o.lab ?? 1);
  const mv = ease(clamp(u / .9)), P = rec ? FRONT.pip[rec.a] : HOME.pip, Q = rec ? FRONT.tam[rec.b] : HOME.tam;
  const px = lerp(HOME.pip[0], P[0], mv), py = lerp(HOME.pip[1], P[1], mv), qx = lerp(HOME.tam[0], Q[0], mv), qy = lerp(HOME.tam[1], Q[1], mv);
  L.mouse(px, py, rec && mv < 1 ? Math.sign(P[0] - HOME.pip[0]) || 1 : (rec ? (rec.a ? 1 : -1) : 1), 1.05, t); name('Pip', px, py + 24);
  tam(qx, qy, rec && mv < 1 ? Math.sign(Q[0] - HOME.tam[0]) || -1 : (rec ? (rec.b ? 1 : -1) : -1), 1.05, t); name('Tam', qx, qy + 24);
  if (rec && out > 0) {
    const lab = r => (r === 1 ? '+1' : r > 0 ? '+0.3' : '0'), col = r => (r === 1 ? '#b07d12' : r > 0 ? '#3f7f7c' : C.soft);
    txtUp(lab(rec.r1), P[0], P[1] - 32, out, col(rec.r1)); txtUp(lab(rec.r2), Q[0], Q[1] - 32, out - .15, col(rec.r2));
    if (rec.ra && o.dice !== 0) txt('random', P[0], P[1] + 38, { s: 12, c: C.roseInk, a: clamp(out * 3) });
    if (rec.rb && o.dice !== 0) txt('random', Q[0], Q[1] + 38, { s: 12, c: C.roseInk, a: clamp(out * 3) });
  }
}
function txtUp(s, x, y, u, col) { if (u <= 0) return; txt(s, x, y - 8 * ease(clamp(u * 2)), { s: 19, c: col, a: clamp(u * 3) }); }
// the four numbers as a little table in the card area
function qTable(qp, qt, a = 1, o = {}) {
  if (a <= 0) return; const x = o.x ?? 30, y = o.y ?? 292, w = o.w ?? 340, h = 80;
  alpha(a, () => { box(x, y, w, h, C.card); rrect(x, y, w, h, 210, { c: C.line, w: 1.1 });
    txt('big door', x + 175, y + 16, { s: 14, c: C.goldInk }); txt('hole', x + 270, y + 16, { s: 14, c: '#3f7f7c' });
    txt('Pip’s numbers', x + 14, y + 40, { s: 16, al: 'left' }); txt('Tam’s numbers', x + 14, y + 64, { s: 16, al: 'left' });
    [qp, qt].forEach((q, i) => q.forEach((v, k) => { const hl = o.hl && o.hl[i] === k; if (hl) box(x + 145 + k * 95, y + 30 + i * 24, 60, 20, C.yellow, .7);
      txt(f2(v), x + 175 + k * 95, y + 40 + i * 24, { s: 16 }); })); });
}
// payoff table: rows = first player's door, columns = second player's door; each box "first, second"
function payTable(game, names, a, o = {}) {
  if (a <= 0) return; const x = o.x ?? 92, y = o.y ?? 292, cw = 120, ch = 34, lab = o.doors || ['big door', 'hole'];
  alpha(a, () => {
    txt(`${names[1]} picks →`, x + cw + 10, y + 6, { s: 14, c: C.soft }); txt(names[0] + ' ↓', x - 40, y + 6, { s: 14, c: C.soft });
    lab.forEach((s, k) => { txt(s, x + k * cw + cw / 2, y + 24, { s: 14, c: C.soft }); txt(s, x - 8, y + 34 + k * ch + ch / 2, { s: 14, c: C.soft, al: 'right' }); });
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { const [p, q] = game(i, j), bx = x + j * cw, by = y + 34 + i * ch, u = clamp((a - .2 - (i * 2 + j) * .12) * 3);
      const hl = o.hl && o.hl(i, j); box(bx, by, cw, ch, hl ? C.yellow : C.card, hl ? .75 : 1); rrect(bx, by, cw, ch, 220 + i * 2 + j, { c: C.line, w: 1 });
      txt(`${sg(p)}, ${sg(q)}`, bx + cw / 2, by + ch / 2 + 1, { s: 17, a: u }); }
  });
}
const sg = v => (v > 0 ? (v === 1 ? '1' : String(v)) : v < 0 ? '−' + Math.abs(v) : '0');
// chart: chance that the big door pays for Pip, round by round (= Tam's chance of picking it)
function worldChart(n, o = {}) {
  const x0 = 52, x1 = 360, y0 = 392, y1 = 312, X = i => x0 + i * (x1 - x0) / (NR - 1), Y = v => lerp(y0, y1, v);
  rline(x0 - 4, y0, x1 + 4, y0, 960, { c: C.line, w: 1 }); rline(x0 - 4, Y(1), x1 + 4, Y(1), 961, { c: '#e4d8c7', w: 1, once: true });
  txt('1', x0 - 14, Y(1), { s: 12, c: C.soft }); txt('0', x0 - 14, y0, { s: 12, c: C.soft }); txt('0.5', x0 - 18, Y(.5), { s: 12, c: C.soft });
  txt(o.title || 'chance the big door pays for Pip, each round', x0 - 30, y1 - 14, { s: 14, al: 'left', c: C.soft });
  ctx.save(); ctx.strokeStyle = C.goldInk; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < Math.min(n, NR); i++) { const v = PWORLD[i]; if (!i) ctx.moveTo(X(0), Y(v)); else { ctx.lineTo(X(i), Y(PWORLD[i - 1])); ctx.lineTo(X(i), Y(v)); } }
  ctx.stroke(); ctx.restore();
  txt('round 1', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(`round ${NR}`, x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
  if (o.marks) [LL1, LL2].forEach((i, k) => { if (n > i) { rline(X(i), y0, X(i), y0 - 6, 970 + k, { c: C.ink, w: 1.2, once: true }); txt(String(i + 1), X(i) + (k ? 12 : -10), y0 - 12, { s: 12, c: C.ink }); } });
}
// ───── cat game scene
const DL = { x: 58, y: 66, w: 100, h: 114 }, DR = { x: 242, y: 66, w: 100, h: 114 }, CHOME = { pip: [140, 250], cat: [262, 250] };
const CFRONT = { pip: [[82, 228], [266, 228]], cat: [[138, 226], [322, 226]] };
function plainDoor(d, seed, open) { box(d.x, d.y, d.w, d.h, '#5a514a', .55); if (open > 0) L.cheese(d.x + d.w / 2, d.y + d.h - 26, 1.3);
  const hh = d.h * (1 - .8 * open); box(d.x, d.y, d.w, hh, '#c9a27a'); for (let k = 1; k < 3; k++) rline(d.x + k * d.w / 3, d.y + 2, d.x + k * d.w / 3, d.y + hh - 2, seed + k, { c: '#9c7a55', w: 1, once: true });
  rrect(d.x, d.y, d.w, hh, seed, { w: 1.5 }); rrect(d.x, d.y, d.w, d.h, seed + 5, { w: 1.2, c: C.line }); }
function catScene(rec, u, t, o = {}) {
  wall(); const out = u - 1.0, escaped = rec && rec.a !== rec.b;
  plainDoor(DL, 80, rec && escaped && rec.a === 0 ? ease(out / .6) : 0); plainDoor(DR, 90, rec && escaped && rec.a === 1 ? ease(out / .6) : 0);
  txt('left door', DL.x + DL.w / 2, DL.y - 14, { s: 14, c: C.soft }); txt('right door', DR.x + DR.w / 2, DR.y - 14, { s: 14, c: C.soft });
  const mv = ease(clamp(u / .9)), P = rec ? CFRONT.pip[rec.a] : CHOME.pip, Q = rec ? CFRONT.cat[rec.b] : CHOME.cat;
  const px = lerp(CHOME.pip[0], P[0], mv), py = lerp(CHOME.pip[1], P[1], mv), qx = lerp(CHOME.cat[0], Q[0], mv), qy = lerp(CHOME.cat[1], Q[1], mv);
  L.mouse(px, py, rec ? (P[0] >= CHOME.pip[0] ? 1 : -1) : 1, 1.05, t); name('Pip', px, py + 24); catA(qx, qy, 1.15, t); name(o.catName || 'cat', qx, qy + 26);
  if (rec && out > 0) {
    if (escaped) { txtUp('+1', P[0], P[1] - 30, out, '#b07d12'); txtUp('−1', Q[0], Q[1] - 34, out - .15, C.roseInk); }
    else { txtUp('caught!', (P[0] + Q[0]) / 2, P[1] - 40, out, C.roseInk); txtUp('−1', P[0] - 36, P[1] + 4, out - .2, C.roseInk); txtUp('+1', Q[0] + 36, Q[1] + 4, out - .2, '#b07d12'); }
  }
}
// two strips: each round's favourite door (from the numbers before the round) for Pip and the cat
function favStrips(n, o = {}) {
  const x0 = 70, x1 = 360, w = (x1 - x0) / NR, rows = [['Pip', 0, 318], ['cat', 1, 352]];
  txt(o.title || 'favourite door before each round', 30, 300, { s: 14, al: 'left', c: C.soft });
  rows.forEach(([s, who, y]) => { txt(s, x0 - 8, y + 8, { s: 14, al: 'right' }); for (let i = 0; i < Math.min(n, NR); i++) { const f = fav(H[i].qb[who]); box(x0 + i * w, y, w + .3, 16, f === 0 ? C.gold : f === 1 ? C.teal : '#e4d8c7'); } });
  txt('round 1', x0, 382, { s: 12, c: C.soft, al: 'left' }); txt(`round ${NR}`, x1, 382, { s: 12, c: C.soft, al: 'right' });
  box(150, 377, 10, 10, C.gold); txt('left', 164, 382, { s: 12, al: 'left', c: C.soft }); box(200, 377, 10, 10, C.teal); txt('right', 214, 382, { s: 12, al: 'left', c: C.soft });
}
function smallMaze(ox, focus, t, a) {
  const G = L.gridWorld({ cs: 30, ox, oy: 66 }); alpha(a, () => { G.maze({ t, hi: [[focus[0], focus[1], C.yellow, .8]] });
    if (focus === TAMC) tam(G.cx(4), G.cy(3) + 2, -1, .5, t); else L.mouse(G.cx(0), G.cy(3) + 2, 1, .5, t); });
  return G;
}
const TAMC = [4, 3], PIPC = [0, 3];
function round(rec, t, d, k0 = 0) { return { k: k0 + Math.floor(t / d), u: t % d }; }

const STEPS = [
 { h: 'Not alone any more', p: ['In every reel so far, Pip was the only one learning. The cheese, the cat, even the wet floor followed **fixed rules**: the same move in the same cell gave the same chances, every time.', 'Now meet **Tam**, a second mouse in the same maze, with a pink bow. Tam learns too. Several learners sharing one world: that is **multi-agent RL**.'],
   draw(t) { const G = L.gridWorld({}); G.maze({ t, draw: clamp(t / 1.5) }); if (t > 1) { G.pip(G.cx(0), G.cy(3), 1, t); name('Pip', G.cx(0), G.cy(3) + 22, clamp(t - 1)); }
     if (t > 2.5) { tam(G.cx(4), G.cy(3) + 3, -1, .95, t); name('Tam', G.cx(4), G.cy(3) + 22, clamp(t - 2.5)); }
     card(['before: one learner, fixed rules', 'now: two learners in one world', { s: 'each picks its own moves and gets its own rewards', c: C.soft }], t - 4, { gap: 1.2 }); } },
 { h: 'The others are part of the world', p: ['From Pip’s point of view, Tam is part of the world, like the floor: what happens after Pip’s move can depend on what Tam does. And for Tam, Pip is part of its world.', 'But the floor never changes its habits. Tam does, as it learns. So the chances in Pip’s world keep shifting. A world like that is called **non-stationary**.'],
   draw(t) { smallMaze(26, TAMC, t, clamp(t - .3)); smallMaze(226, PIPC, t, clamp(t - 1.6));
     txt('Pip’s world', 101, 48, { s: 16, a: clamp(t - .3) }); txt('Tam’s world', 301, 48, { s: 16, a: clamp(t - 1.6) });
     if (t > .8) { L.mouse(101, 214, 1, .9, t); name('Pip, outside: the learner', 101, 240, clamp(t - .8)); alpha(clamp(t - .8), () => arrow(101, 198, 101, 184, 77, { c: C.soft })); }
     if (t > 2.1) { tam(301, 214, -1, .9, t); name('Tam, outside: the learner', 301, 240, clamp(t - 2.1)); alpha(clamp(t - 2.1), () => arrow(301, 198, 301, 184, 78, { c: C.soft })); }
     card(['the floor: same chances, every time', 'Tam: changes its habits as it learns', { s: 'so Pip’s world keeps changing: non-stationary', c: C.roseInk }], t - 3.5, { gap: 1.3 }); } },
 { h: 'One room, two doors', p: ['To see it clearly, shrink the maze to one room. Each round, Pip and Tam each pick a door at the same moment, without seeing the other’s choice.', 'The **big door** is heavy: it opens only if both push it, and then each gets a big cheese, worth 1. Pushing it alone gets nothing. The **small hole** always has a crumb, worth 0.3, for each mouse that goes there. The round ends there, and the payoff is paid at once.'],
   shot: [3, 9],
   draw(t) { const recs = [{ a: 0, b: 0, r1: 1, r2: 1 }, { a: 0, b: 1, r1: 0, r2: .3 }, { a: 1, b: 1, r1: .3, r2: .3 }], d = 2.6, k = Math.floor((t - 1) / d);
     if (t < 1) teamScene(null, 0, t, { lab: clamp(t * 2) }); else teamScene(recs[k % 3], (t - 1) % d, t, { dice: 0 });
     card(['both at the big door: 1 each', 'alone at the big door: 0', 'at the hole: 0.3, whatever the other does'], t - 1.2, { gap: 2.6 }); } },
 { h: 'The payoff table', p: ['All four cases fit in a table. Rows are Pip’s door, columns are Tam’s. Each box shows Pip’s payoff, then Tam’s.', 'Both do best **together**, at the big door: 1 each. So Pip and Tam are a team. But the big door is a gamble: it pays only if the other comes too. The hole is safe. Game theorists call this a **stag hunt**: two hunters catch a stag only together, while a hare is safe alone.'],
   draw(t) { teamScene(null, 0, t); payTable(TEAM, ['Pip', 'Tam'], clamp(t - .5) * 2, { hl: (i, j) => t > 6 && i === 0 && j === 0 }); } },
 { h: 'How each mouse learns', p: ['Each mouse keeps two numbers: its guess of the payoff of each door. Both start at 0. Each round it takes the door with the bigger number, a coin toss if they are equal. One round in five it picks at random, to keep trying both.', 'After the round, it moves the number for its door a fifth of the way toward the payoff. Neither knows that the other is learning. These are **independent learners**.'],
   draw(t) { teamScene(null, 0, t); qTable([0, 0], [0, 0], clamp(t - .5), { y: 292 });
     if (t > 3) alpha(clamp(t - 3), () => { txt('new = old + 0.2 × (payoff − old)', 200, 388, { s: 18, c: C.ink }); });
     if (t > 5) alpha(clamp(t - 5), () => { txt('bigger number, but 1 round in 5: random', 200, 200, { s: 16, c: C.roseInk }); }); } },
 { h: 'The first rounds', p: [`Round 1: all numbers are 0, so both toss a coin, and both pick the hole. Each gets 0.3, and its number for the hole goes from 0 to ${f2(R1.q[0][1])}. Now the hole has the bigger number, so both keep going there.`, `Round ${FIRSTX + 1}: one of Pip’s random rounds sends it to the big door. Tam is at the hole, so the door stays shut: payoff 0, and Pip’s 0 for the big door stays 0.`],
   shot: [3.5, 14.5],
   draw(t) { const d = 2.4, ks = [0, 1, 2, 3, 4, FIRSTX], i = clamp(Math.floor(t / d), 0, ks.length - 1), u = t - i * d, rec = T[ks[i]];
     teamScene(rec, Math.min(u, 2.39), t); const show = u > 1.2 ? rec.q : rec.qb; qTable(show[0], show[1], 1, { hl: u > 1.2 ? [rec.a, rec.b] : null });
     caption(`round ${ks[i] + 1}`); } },
 { h: 'The big door, alone', p: [`Before round ${LL1 + 1}, Pip tried the big door ${LONE} times, all at random. Tam was at the hole every time, so each try paid 0.`, 'Is Pip wrong to keep 0 for the big door? Not really. Tam goes to the big door only in its random rounds, half of one in five: **one round in ten**. So the big door pays 0.1 on average, less than the crumb’s 0.3. For Tam’s current habit, the hole really is better.'],
   draw(t) { const tries = T.slice(0, LL1).map((x, i) => (x.a === 0 ? i : -1)).filter(i => i >= 0), d = 1.25, i = clamp(Math.floor(t / d), 0, tries.length - 1), u = t - i * d, rec = T[tries[i]];
     teamScene(rec, Math.min(u, 1.24) + .25, t, { dice: 0 }); qTable(rec.q[0], rec.q[1], 1, { hl: [0, null] });
     caption(t < tries.length * d ? `round ${tries[i] + 1}: try ${i + 1} of ${LONE}` : `${LONE} tries alone, all paid 0`); },
   shot: [4, 15] },
 { h: 'A lucky round', p: [`Round ${LL1 + 1}: both happen to make a random choice, and both pick the big door. It opens! Each number for the big door jumps from 0 to ${f2(T[LL1].q[0][0])}. Then each happens to try it once alone, and drops to ${f2(T[LL2].qb[0][0])}.`, `Round ${LL2 + 1}: another random double. ${f2(T[LL2].qb[0][0])} + 0.2 × ${f2(1 - T[LL2].qb[0][0])} = ${f2(T[LL2].q[0][0])}, just above the hole’s 0.3. Now both prefer the big door.`],
   shot: [3, 10],
   draw(t) { const d = 3.4, i = t < d ? 0 : 1, u = t - i * d, rec = T[i ? LL2 : LL1];
     teamScene(rec, Math.min(u, 3.39), t); const show = u > 1.2 ? rec.q : rec.qb; qTable(show[0], show[1], 1, { hl: u > 1.2 ? [0, 0] : null });
     caption(`round ${(i ? LL2 : LL1) + 1}`); } },
 { h: 'Same door, different world', p: ['This line shows the world from Pip’s side. Across: rounds 1 to 200. Up: the chance that the big door pays for Pip in that round, which is Tam’s chance of picking it. Round 1 shows 0.5, since Tam’s first pick is a coin toss.', `Nothing about the door changed. **Tam** changed: the chance jumped from 0.1 to 0.9 after round ${LL2 + 1}. On the wet floor, a move’s chances stayed fixed. Here Pip’s lessons can go stale at any time.`],
   draw(t) { const n = clamp(Math.floor(t / 7 * NR), 1, NR), rec = T[n - 1]; teamScene(null, 0, t, { lab: .5 });
     txt(`round ${n}: Tam ${pLeft(rec.qb[1]) > .5 ? 'prefers the big door' : pLeft(rec.qb[1]) < .5 ? 'prefers the hole' : 'has no favourite'}`, 200, 198, { s: 15, c: C.ink });
     worldChart(n, { marks: 1 }); } },
 { h: 'A hundred pairs', p: [`Now 100 pairs of mice, each with its own dice, play ${NR} rounds. Each dot is one pair: gold when both prefer the big door, teal when both prefer the hole, grey otherwise.`, `After 50 rounds, ${C50.hole} pairs share the hole. After ${NR}, still ${C200.hole}; ${C200.big} reached the big door. A pair at the hole is stuck in a sensible way: whoever switches alone gets 0. Only a lucky double gets them out.`],
   shot: [5, 14],
   draw(t) { const n = clamp(Math.floor((t - 1) / 9 * NR), 1, NR), cnt = countAt(n);
     for (let i = 0; i < NP; i++) { const v = PAIRS[i][n - 1], x = 110 + (i % 10) * 20, y = 40 + Math.floor(i / 10) * 21; dot(x, y, 7, v === 0 ? C.gold : v === 1 ? C.teal : '#d8cfc4', clamp(t * 2)); }
     txt(`after round ${n}`, 200, 262, { s: 16, a: clamp(t * 2) });
     card([{ s: `both prefer the big door: ${cnt.big}`, c: C.goldInk }, { s: `both prefer the hole: ${cnt.hole}`, c: '#3f7f7c' }, { s: `no shared favourite: ${cnt.other}`, c: C.soft }], 2, { gap: 0 }); } },
 { h: 'Now against the cat', p: ['Teammates share a goal. Opponents do not. New game: Pip runs to one of two doors, with cheese behind each, and the cat waits at one door. Same door: caught, Pip gets −1, the cat +1. Different doors: Pip eats, +1, and the cat gets −1.', 'What one wins, the other loses. This is called **zero-sum**: pure competition.'],
   shot: [3, 10],
   draw(t) { const recs = [{ a: 0, b: 1 }, { a: 1, b: 1 }], d = 2.8, k = Math.floor((t - .6) / d);
     if (t < .6) catScene(null, 0, t); else catScene(recs[k % 2], (t - .6) % d, t);
     payTable(ZERO, ['Pip', 'cat'], clamp(t - 4) * 2, { doors: ['left', 'right'] }); } },
 { h: 'Chasing each other', p: [`Pip and the cat learn exactly like the two mice: two numbers each, one round in five random. The strips show each one’s favourite door before every round: gold for left, teal for right.`, `The cat learns Pip’s favourite and waits there, so that door starts to cost Pip, and Pip switches. Then the cat follows. Pip’s favourite changed ${HSW} times in ${NR} rounds. Neither ever settles.`],
   shot: [4, 13],
   draw(t) { const n = clamp(Math.floor((t - 1) / 9 * NR), 0, NR); const rec = H[Math.max(0, n - 1)];
     catScene(n ? rec : null, n ? 1.6 : 0, t); favStrips(n); caption(n ? `round ${n}` : ''); } },
 { h: 'No resting point', p: ['The best way to play here is the rock, paper, scissors answer from the policy reel: a coin toss between the doors. If both play that way, neither gains by changing alone, so there is nothing left to learn. Such a resting point is called an **equilibrium**; the pair stuck at the hole was one too.', `Our learners never hold it. Each always has a favourite, taken nine rounds in ten, which the other can exploit. Pip went left ${HLEFT} of ${NR} times: close to half, but only on average.`],
   draw(t) { catScene(null, 0, t); favStrips(NR);
     alpha(clamp(t - 3), () => { txt('best: a coin toss, 50 : 50', 200, 202, { s: 17, c: C.goldInk }); }); } },
 { h: 'Your turn: be the cat', tap: tapDoor,
   p: [`You are the cat. Tap a door to wait there. Pip learns from every round, exactly as before. Its two numbers are shown below.`, `A cat that always waits at the left door caught this Pip ${30 - AL_ESC === 1 ? 'just once' : (30 - AL_ESC) + ' times'} in 30 rounds; against 1000 Pips with other dice, ${AL_AVG.toFixed(1)} times on average, mostly in Pip’s random rounds. A cat tossing a coin catches Pip about half the time: Pip escaped ${COIN_ESC} of 1000 rounds in our run. Can you catch Pip more than half the time?`],
   draw(t, st) { const hist = st.hist || [], res = vsYou(hist), n = res.length, last = res[n - 1], u = n ? t - st.tt : 0;
     catScene(last && u < 3.2 ? { a: last.a, b: last.c } : null, Math.min(u, 3), t, { catName: 'you' });
     const done = u > 1.2, shown = done ? res : res.slice(0, -1), m = shown.length, caught = shown.filter(x => x.r < 0).length, q = last ? (done ? last.q : last.qb) : [0, 0];
     card([`rounds: ${m}    caught: ${caught}    escaped: ${m - caught}`, `Pip’s numbers:  left ${f2(q[0])}   right ${f2(q[1])}`, { s: n && done ? (last.a === last.c ? 'you caught Pip' : 'Pip got away') : n ? '…' : 'tap the left or the right door', c: C.soft }], 10, { gap: 0 });
     if (!n) alpha(.6 + .4 * Math.sin(t * 3), () => { txt('tap', 108, 122, { s: 20, c: C.goldInk }); txt('tap', 292, 122, { s: 20, c: C.goldInk }); }); } },
 { h: 'Train together, act alone', p: ['One popular fix for the shifting world: **centralised training with decentralised acting**. While practising, a helper, often called a **critic**, learns a number for each pair of doors, like the payoff table. It sees what neither mouse can: the big door pays 1 when both come. Each mouse is nudged toward doors the helper rates highly.', 'When acting, the helper is gone and each mouse decides alone. MADDPG, from 2017, is one such method; this reel doesn’t run it.'],
   draw(t) { const a1 = clamp(t - .3), a2 = clamp(t - 3);
     alpha(a1, () => { rrect(24, 40, 170, 200, 400, { c: C.line, w: 1.1, fill: C.card }); txt('practice', 109, 58, { s: 16 });
       L.robot(109, 130, 1.4, t); txt('helper: sees all', 109, 80, { s: 13, c: C.soft }); L.mouse(64, 200, 1, .9, t); tam(156, 200, -1, .9, t);
       arrow(70, 186, 96, 160, 401, { c: C.soft }); arrow(150, 186, 124, 160, 402, { c: C.soft }); });
     alpha(a2, () => { rrect(206, 40, 170, 200, 410, { c: C.line, w: 1.1, fill: C.card }); txt('acting', 291, 58, { s: 16 });
       L.mouse(246, 200, 1, .9, t); tam(338, 200, -1, .9, t); rline(291, 150, 291, 230, 411, { c: C.line, w: 1.2 });
       txt('own view', 246, 160, { s: 13, c: C.soft }); txt('own view', 338, 160, { s: 13, c: C.soft }); });
     card(['practice: helper scores each pair of doors', 'acting: each uses only what it sees', { s: 'often shortened to CTDE', c: C.soft }], t - 4.5, { gap: 1.2 }); } },
 { h: 'Playing against yourself', p: ['In games where both sides play by the same rules, like chess, an agent can train against **copies of itself**. As it improves, so does its opponent, so practice is never too easy and never hopeless. This is **self-play**; AlphaZero learned chess and Go this way.', 'An uneven game like runner against cat needs both roles trained. And self-play can still go round in circles, as our learners did, so big systems also play against older copies.'],
   draw(t) { const bx = 140, by = 90, q = 30; alpha(clamp(t * 1.5), () => { for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) box(bx + i * q, by + j * q, q, q, (i + j) % 2 ? '#cdbfa8' : '#f6eedf', 1); rrect(bx, by, 4 * q, 4 * q, 430, { w: 1.6 }); });
     alpha(clamp(t - .8), () => { L.mouse(88, 180, 1, 1.05, t); name('Pip', 88, 210); });
     alpha(clamp(t - 1.6), () => { L.mouse(312, 180, -1, 1.05, t); name('Pip’s copy', 312, 210); txt('same rules for both sides', 200, 236, { s: 14, c: C.soft }); });
     card(['opponent = a copy of the learner', 'it gets better at the same pace', { s: 'uneven games: train both roles', c: C.soft }, { s: 'older copies too: fewer circles', c: C.soft }], t - 2.5, { gap: 1.2 }); } },
 { h: 'Out in the world', p: ['In April 2019, **OpenAI Five**, five copies of one network trained mostly by self-play, beat OG, the reigning world champions of the video game Dota 2. The same year, DeepMind’s **AlphaStar** reached Grandmaster level in StarCraft II, training a **league**: many agents, old and new, that play against each other.', 'Both took enormous amounts of practice. The shifting world of this reel is one reason multi-agent training is hard.'],
   draw(t) { const tm = [[60, C.teal], [340, C.rose]]; tm.forEach(([x, col], s) => { for (let i = 0; i < 5; i++) { const a = clamp((t - .4 - s * .8 - i * .15) * 3); if (a > 0) { alpha(a, () => L.robot(x + (s ? -1 : 1) * (i % 2) * 34, 50 + i * 44, 1.1, t + i, s ? '#e9b3b3' : '#9fc9c7')); } } });
     alpha(clamp(t - 2), () => { txt('5 against 5', 200, 120, { s: 18 }); txt('every agent learning at once', 200, 146, { s: 14, c: C.soft }); });
     card(['2019  OpenAI Five beats OG at Dota 2', '2019  AlphaStar: Grandmaster in StarCraft II', { s: 'both: self-play at a huge scale', c: C.soft }], t - 3, { gap: 1.2 }); } },
 { h: 'Recap', p: ['In **multi-agent RL**, several learners share a world. To each one, the others are part of that world, and because they keep learning, the world is **non-stationary**.', 'Teammates can get stuck on a safe but poor habit; opponents can chase each other forever. Helpers that see everything during training, and self-play, are common ways to cope.'],
   draw(t) { teamScene(null, 0, t, { lab: .6 });
     card(['several learners, one world', 'the others are part of each one’s world', 'they learn, so that world keeps changing', 'teams can settle low; rivals keep chasing'], t - .3, { gap: 1 }); } },
];
function tapDoor(x, y, st, t) {
  if (y > 286) return; const hist = st.hist || (st.hist = []);
  if (st.tt != null && t - st.tt < 1.3) return;                            // let the round finish
  hist.push(x < 200 ? 0 : 1); st.tt = t;
}
RLStory.register('marl', { title: 'Multi-Agent RL', track: 'Bigger worlds', onCtx: c => { ctx = c; }, steps: STEPS,
  _sim: { T, R1, FIRSTX, LL1, LL2, LONE, SWITCH, MID, PWORLD, FINQ, C200, C50, HSW, HLEFT, HCAUGHT, AL_ESC, AL_ESC20, COIN_ESC, ALWAYSL, countAt, vsYou, H } });
})();
