// Meta-RL as a story: a family of mazes where the cheese hides under a different one of five cups each time.
// Plain Q-learning starts from scratch on every maze; an RL²-style learner (a policy over cell + a memory that survives
// between walks) and a MAML-style learner (a learned starting point, trained with Reptile) learn to adapt fast.
// Every number comes from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
// ───── the room: 5 × 5, start bottom middle, five cups on the top row; the cheese is under one of them
const N = 5, MV = [[0, -1], [1, 0], [0, 1], [-1, 0]], GAM = 0.9, CAP = 30, NWK = 10, EPS = 0.2, ALPHA = 0.2, SX = 2, SY = 4;
const step = (x, y, m) => { const nx = x + MV[m][0], ny = y + MV[m][1]; return (nx < 0 || ny < 0 || nx >= N || ny >= N) ? [x, y] : [nx, ny]; };
function pickA(Q, base, R, eps) { if (R() < eps) return Math.floor(R() * 4); let b = -1e9, ms = []; for (let a = 0; a < 4; a++) { const v = Q[base + a]; if (v > b + 1e-12) { b = v; ms = [a]; } else if (v > b - 1e-12) ms.push(a); } return ms[Math.floor(R() * ms.length)]; }
// memory (RL²-style): which cups were found empty (5 bits), or where the cheese was (32 + cup). Kept between walks, wiped for a new maze.
const MEMN = 37, memUpd = (mem, x, y, g) => (mem >= 32 || y !== 0) ? mem : (x === g[0] && y === g[1]) ? 32 + x : mem | (1 << x);
const mi = (mem, x, y) => (mem * 25 + y * 5 + x) * 4;
// one walk of the memory learner. learn: update Q (training). Returns steps and the path.
function memWalk(Q, g, mem, R, eps, learn) { let x = SX, y = SY, k; const P = [[x, y]], M = [mem];
  for (k = 0; k < CAP; k++) { const s = mi(mem, x, y), a = pickA(Q, s, R, eps), [nx, ny] = step(x, y, a), done = nx === g[0] && ny === g[1], nm = memUpd(mem, nx, ny, g);
    if (learn) { const s2 = mi(nm, nx, ny), tg = done ? 1 : GAM * Math.max(Q[s2], Q[s2 + 1], Q[s2 + 2], Q[s2 + 3]); Q[s + a] += ALPHA * (tg - Q[s + a]); }
    x = nx; y = ny; mem = nm; P.push([x, y]); M.push(mem); if (done) { k++; break; } }
  return { steps: k, mem, P, M, found: x === g[0] && y === g[1] }; }
const TOPS = (R) => [Math.floor(R() * 5), 0];
function trainMem(n, seed = 5) { const R = rng(seed), Q = new Float64Array(25 * MEMN * 4); for (let t = 0; t < n; t++) { const g = TOPS(R); let mem = 0; for (let w = 0; w < NWK; w++) mem = memWalk(Q, g, mem, R, EPS, true).mem; } return Q; }
// plain Q-learning over cells, starting from numbers Q0, on one maze: nw walks
function adapt(Q0, g, R, nw = NWK) { const Q = Float64Array.from(Q0), out = [];
  for (let w = 0; w < nw; w++) { let x = SX, y = SY, k; const P = [[x, y]];
    for (k = 0; k < CAP; k++) { const s = (y * 5 + x) * 4, a = pickA(Q, s, R, EPS), [nx, ny] = step(x, y, a), done = nx === g[0] && ny === g[1], s2 = (ny * 5 + nx) * 4;
      const tg = done ? 1 : GAM * Math.max(Q[s2], Q[s2 + 1], Q[s2 + 2], Q[s2 + 3]); Q[s + a] += ALPHA * (tg - Q[s + a]); x = nx; y = ny; P.push([x, y]); if (done) { k++; break; } }
    out.push({ steps: k, P }); }
  return { Q, out }; }
// Reptile (a first-order cousin of MAML): copy the start, learn 3 walks on one maze, move the start a tenth of the way toward the result
const RK = 3, RB = 0.1;
function reptile(n, seed = 6) { const R = rng(seed), Q0 = new Float64Array(100); for (let t = 0; t < n; t++) { const g = TOPS(R), { Q } = adapt(Q0, g, R, RK); for (let i = 0; i < 100; i++) Q0[i] += RB * (Q[i] - Q0[i]); } return Q0; }
// ───── the experiments: train on 3000 mazes; test on the same 300 new mazes (own seed), every learner with ε = 0.2
const NTRAIN = 3000, NTEST = 300;
const TESTG = (fam) => { const R = rng(fam === 'top' ? 900 : 901); return Array.from({ length: NTEST }, () => fam === 'top' ? [Math.floor(R() * 5), 0] : [0, 1 + Math.floor(R() * 3)]); };
const QM = trainMem(NTRAIN), Q0 = reptile(NTRAIN), ZERO = new Float64Array(100);
function testMem(Q, gs, seed = 77) { const R = rng(seed), S = new Array(NWK).fill(0); gs.forEach(g => { let mem = 0; for (let w = 0; w < NWK; w++) { const r = memWalk(Q, g, mem, R, EPS, false); mem = r.mem; S[w] += r.steps / gs.length; } }); return S; }
function testQ(Q0, gs, seed = 77) { const R = rng(seed), S = new Array(NWK).fill(0); gs.forEach(g => adapt(Q0, g, R).out.forEach((w, i) => { S[i] += w.steps / gs.length; })); return S; }
const GT = TESTG('top'), GL = TESTG('left');
const RES = { scratch: testQ(ZERO, GT), maml: testQ(Q0, GT), mem: testMem(QM, GT) };
const OFF = { scratch: testQ(ZERO, GL), maml: testQ(Q0, GL), mem: testMem(QM, GL) };
const BEST = GT.reduce((s, g) => s + 4 + Math.abs(g[0] - 2), 0) / NTEST;                 // shortest walk, averaged over the test mazes
const EXPSTEPS = (() => { const R = rng(5); let s = 0; const Q = new Float64Array(25 * MEMN * 4); for (let t = 0; t < NTRAIN; t++) { const g = TOPS(R); let mem = 0; for (let w = 0; w < NWK; w++) { const r = memWalk(Q, g, mem, R, EPS, true); mem = r.mem; s += r.steps; } } return s; })();
// story maze: cheese under cup 5 (x = 4); one walk sequence of each learner (greedy-free: ε = 0.2 as in the tests)
const SG = [4, 0];
const DEMO_MEM = (() => { const R = rng(31); let mem = 0; const ws = []; for (let w = 0; w < 3; w++) { const r = memWalk(QM, SG, mem, R, 0, false); ws.push(r); mem = r.mem; } return ws; })();  // its own choices, no random moves
const DEMO_SCR = adapt(ZERO, SG, rng(32), 1).out[0];
// your turn: memory learner trained on n mazes (computed when chosen)
const NTR = [0, 10, 30, 100, 3000], YT = {};
const ytRes = i => YT[i] || (YT[i] = testMem(i === 4 ? QM : trainMem(NTR[i]), GT));
const startArrows = (() => { const A = []; for (let k = 0; k < 25; k++) { let b = 0; for (let a = 1; a < 4; a++) if (Q0[k * 4 + a] > Q0[k * 4 + b] + 1e-12) b = a; A.push(b); } return A; })();
const f1 = v => v.toFixed(1), f0 = v => Math.round(v).toLocaleString('en-US');
const sumA = a => a.reduce((x, y) => x + y, 0), BREAK = EXPSTEPS / (sumA(RES.scratch) - sumA(RES.mem));   // new cup mazes until training pays back, against from scratch
// best order to look under the cups, if no move were random: expected steps over the 5 cheese cups, all 120 orders
const ORDBEST = (() => { const perms = a => a.length <= 1 ? [a] : a.flatMap((x, i) => perms([...a.slice(0, i), ...a.slice(i + 1)]).map(p => [x, ...p])); let b = 1e9; for (const o of perms([0, 1, 2, 3, 4])) { let st = 4 + Math.abs(o[0] - SX), tot = st; for (let i = 1; i < 5; i++) { st += Math.abs(o[i] - o[i - 1]); tot += st; } b = Math.min(b, tot / 5); } return b; })();

// ───── drawing
let CS = 44, OX = 90, OY = 44;
const cx = x => OX + x * CS + CS / 2, cy = y => OY + y * CS + CS / 2;
function withGeom(cs, ox, oy, fn) { const o = [CS, OX, OY]; CS = cs; OX = ox; OY = oy; try { fn(); } finally { [CS, OX, OY] = o; } }
function cup(x, y, s, a = 1) { alpha(a, () => { ctx.beginPath(); ctx.moveTo(x - 11 * s, y + 9 * s); ctx.lineTo(x + 11 * s, y + 9 * s); ctx.lineTo(x + 7 * s, y - 8 * s); ctx.lineTo(x - 7 * s, y - 8 * s); ctx.closePath(); ctx.fillStyle = '#e9d9c3'; ctx.fill(); ctx.strokeStyle = C.ink; ctx.lineWidth = 1.2; ctx.stroke(); }); }
// room(o): o.g cheese cup [x, 0] (shown under a lifted cup if o.show), o.lift: cups already looked under, o.arrows, o.draw
function room(o = {}) {
  const p = o.draw ?? 1, t = o.t || 0;
  if (o.shade) for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const v = o.shade(x, y); if (v) box(OX + x * CS + 1, OY + y * CS + 1, CS - 2, CS - 2, v[0], v[1]); }
  ctx.save(); ctx.globalAlpha = .6 * clamp(p * 2); ctx.strokeStyle = C.line; ctx.lineWidth = .8;
  for (let k = 1; k < N; k++) { ctx.beginPath(); ctx.moveTo(OX + k * CS, OY); ctx.lineTo(OX + k * CS, OY + N * CS); ctx.moveTo(OX, OY + k * CS); ctx.lineTo(OX + N * CS, OY + k * CS); ctx.stroke(); }
  ctx.restore(); rrect(OX, OY, N * CS, N * CS, 90, { w: 2, p: clamp(p * 1.5) });
  const s = CS / 44;
  for (let x = 0; x < N; x++) { const lifted = o.lift && o.lift[x], here = o.g && o.g[0] === x && (o.g[1] ?? 0) === 0;
    if (here && (o.show || lifted)) L.cheese(cx(x), cy(0) + 4 * s, .55 * s);
    if (o.cups !== 0) { if (lifted) { if (!here) txt('empty', cx(x), cy(0) + 6 * s, { s: CS < 30 ? 9 : 12, c: C.soft }); } else cup(cx(x), cy(0) + 4 * s, s, clamp(p * 2 - .6) * (here && o.show ? .35 : 1)); } }
  if (o.offG) L.cheese(cx(o.offG[0]), cy(o.offG[1]) + 4 * s, .55 * s);
  if (o.nums) for (let x = 0; x < N; x++) txt('cup ' + (x + 1), cx(x), OY - 9, { s: 12, c: C.soft, a: o.nums });
  if (o.arrows) for (let k = 0; k < 25; k++) { const x = k % 5, y = Math.floor(k / 5), [dx, dy] = MV[o.arrows[k]]; alpha(o.arrowA ?? 1, () => arrow(cx(x) - dx * 8, cy(y) - dy * 8 + 2, cx(x) + dx * 11, cy(y) + dy * 11 + 2, 600 + k, { c: C.goldInk, w: 1.5, head: 5 })); }
}
// a tiny maze thumbnail: square of side w, cheese cup marked gold
function mini(x, y, w, g, seed) { rrect(x, y, w, w, 900 + seed, { w: 1, j: .6, fill: C.card }); for (let i = 0; i < 5; i++) box(x + 2 + i * (w - 4) / 5 + 1, y + 3, (w - 4) / 5 - 2, w / 7, i === g[0] ? '#f2c94c' : '#e2d4bf'); }
function pip(x, y, face, t, s) { L.mouse(x, y + 3, face || 1, s ?? .62 * CS / 44, t); }
// walk an array of cells P with dur per step; returns { x, y, k, done, face }
function along(P, t, dur) { const n = P.length - 1, k = Math.floor(t / dur), u = ease(clamp((t % dur) / dur * 1.4));
  if (k >= n) return { x: cx(P[n][0]), y: cy(P[n][1]), k: n, done: true, face: 1 };
  const [a, b] = P[k], [c, d] = P[k + 1]; return { x: lerp(cx(a), cx(c), u), y: lerp(cy(b), cy(d), u), k, face: c > a ? 1 : c < a ? -1 : 0 }; }
function trail(P, n, col = C.teal, a = .55) { for (let i = 0; i < Math.min(n, P.length); i++) dot(cx(P[i][0]) + ((i * 7) % 9) - 4, cy(P[i][1]) + ((i * 5) % 9) - 4, 2, col, a); }
// memory slots: five boxes, '?' unknown, '✗' empty, cheese found
function memBoxes(mem, x0, y0, a = 1) { alpha(a, () => { txt('memory:', x0 - 8, y0, { s: 14, c: '#2f6f6c', al: 'right' });
  for (let i = 0; i < 5; i++) { const X = x0 + i * 30; rrect(X, y0 - 12, 26, 24, 820 + i, { fill: C.card, c: '#6fb7b4', w: 1.2 });
    if (mem >= 32) { if (mem - 32 === i) L.cheese(X + 13, y0 + 2, .38); else txt('·', X + 13, y0, { s: 14, c: C.soft }); }
    else txt((mem >> i) & 1 ? '✗' : '?', X + 13, y0 + 1, { s: 15, c: (mem >> i) & 1 ? C.roseInk : C.soft }); } }); }
// chart: steps per walk, walks 1..10
const COL = { scratch: '#a59a8f', maml: '#c99a3a', mem: '#3f9a96' }, LAB = { scratch: 'from scratch', maml: 'good start (MAML-style)', mem: 'memory (RL²-style)' };
function chart(o) {
  const x0 = 70, x1 = 350, y0 = o.y0 ?? 384, y1 = o.y1 ?? 300, X = i => x0 + i / (NWK - 1) * (x1 - x0), Y = v => y0 - v / CAP * (y0 - y1);
  rline(x0 - 4, y0, x1 + 4, y0, 960, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1 - 4, 961, { c: C.line, w: 1, once: true });
  txt('0', x0 - 13, y0, { s: 12, c: C.soft }); txt(String(CAP), x0 - 15, y1, { s: 12, c: C.soft });
  for (let i = 0; i < NWK; i++) txt(String(i + 1), X(i), y0 + 11, { s: 12, c: C.soft });
  txt('walk on the new maze', (x0 + x1) / 2, y0 + 25, { s: 12, c: C.soft }); if (o.title) txt(o.title, x0 - 14, y1 - 14, { s: 13, c: C.soft, al: 'left' });
  if (o.best) { ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.grey; ctx.beginPath(); ctx.moveTo(x0, Y(o.best)); ctx.lineTo(x1, Y(o.best)); ctx.stroke(); ctx.restore(); txt(`shortest ${f1(o.best)}`, x1, Y(o.best) + 9, { s: 12, c: C.soft, al: 'right' }); }
  (o.lines || []).forEach((s, j) => { const n = Math.min(NWK, s.n ?? NWK); if (n < 1) return; ctx.save(); ctx.strokeStyle = s.col; ctx.lineWidth = s.w || 2.2; ctx.lineJoin = 'round'; ctx.beginPath();
    for (let i = 0; i < n; i++) { const x = X(i), y = Y(s.v[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); for (let i = 0; i < n; i++) dot(X(i), Y(s.v[i]), 2.6, s.col); });
  if (o.legend) o.legend.forEach(([s, col], i) => { box(x1 - 150, y1 - 2 + i * 17, 14, 4, col); txt(s, x1 - 130, y1 + i * 17, { s: 13, c: col, al: 'left' }); });
  return { X, Y };
}
const ln = (k, n) => ({ v: RES[k], col: COL[k], n });

const STEPS = [
 { h: 'A family of mazes', p: ['A new game for Pip: one room, five cups along the top. The cheese hides under one of them. Pip can’t see under a cup; it finds out only by stepping onto that cell. Stepping onto the cheese pays +1 and ends the walk. A walk is stopped after 30 steps. γ (gamma) 0.9.', 'Each new maze, the cheese moves to a random cup, and stays there for that maze’s 10 walks.'],
   shot: [3, 10],
   draw(t) { const k = clamp(Math.floor((t - 4.5) / 1.3), 0, 4), gs = [[1, 0], [4, 0], [0, 0], [2, 0], [3, 0]], g = gs[k];
     room({ t, draw: t / 1.4, nums: clamp(t - 1.5), g, show: t > 4.5 }); pip(cx(SX), cy(SY), 1, t);
     if (t > 4.5) caption(`maze ${k + 1}: cheese under cup ${g[0] + 1}`);
     if (t > 2) card(['the cheese is under one cup; Pip can’t see which', { s: 'onto the cheese: +1, walk over; max 30 steps', c: C.goldInk }, { s: 'a new maze: the cheese moves; 10 walks per maze', c: C.soft }], t - 2, { gap: 1.3 }); } },
 { h: 'Starting from scratch', p: [`Plain Q-learning, as in the Q-learning reel, treats every maze as brand new: numbers start at 0, one move in five random, α (alpha) 0.2. The chart: across, walks 1 to 10 on one new maze; up, the steps each took, averaged over ${NTEST} new mazes. 30 means the walk was stopped.`, `From scratch: ${f1(RES.scratch[0])} steps on walk 1, still ${f1(RES.scratch[9])} on walk 10. The shortest walk averages ${f1(BEST)}.`],
   shot: [4, 12],
   draw(t) { const n = Math.floor(clamp((t - .6) / 4) * DEMO_SCR.P.length); room({ t, g: SG, lift: Object.fromEntries(DEMO_SCR.P.slice(0, n).filter(p => p[1] === 0).map(p => [p[0], 1])) }); trail(DEMO_SCR.P, n); const m = n ? DEMO_SCR.P[n - 1] : [SX, SY]; pip(cx(m[0]), cy(m[1]), 1, t); if (n >= DEMO_SCR.P.length) txt(`this walk: stopped after ${DEMO_SCR.steps} steps`, 200, 22, { s: 14, c: C.soft });
     const nl = clamp(Math.floor((t - 5) * 3), 0, NWK); chart({ best: t > 8 ? BEST : 0, title: `steps per walk, mean of ${NTEST} new mazes`, lines: [ln('scratch', nl)], legend: t > 5 ? [[LAB.scratch, COL.scratch]] : [] });
 } },
 { h: 'Learning to learn', p: ['But these mazes are all alike: the cheese is always under a cup on the top row. A learner that has seen many of them could know that, and search smartly.', '**Meta-RL** trains on many related tasks so that a new one takes only a few tries. Two loops: an inner loop that adapts to one maze within its 10 walks, and an outer loop, across thousands of mazes, that improves how the inner loop adapts.'],
   shot: [3, 10],
   draw(t) { const gs = [[1, 0], [4, 0], [0, 0], [3, 0], [2, 0], [4, 0]];
     gs.forEach((g, i) => { const a = clamp(t - .6 - i * .4); if (a <= 0) return; alpha(a, () => mini(34 + i * 58, 50, 48, g, i)); });
     if (t > 3.5) { alpha(clamp(t - 3.5), () => { rrect(40, 140, 320, 120, 870, { c: C.line, w: 1.2 }); txt('outer loop: thousands of mazes', 200, 158, { s: 16, c: '#2f6f6c' }); rrect(90, 180, 220, 62, 871, { c: C.goldInk, w: 1.2 }); txt('inner loop: one maze, 10 walks', 200, 200, { s: 15, c: C.goldInk }); txt('adapt fast', 200, 224, { s: 15, c: C.goldInk }); }); }
     card(['inner loop: adapt to one maze in a few walks', { s: 'outer loop: get better at adapting, over many mazes', c: '#2f6f6c' }], t - 6, { gap: 1.2 }); } },
 { h: 'The task is hidden', p: ['Which cup hides the cheese is part of the state, and Pip can’t see it, just like the corridor cells in the fog reel. So this is a POMDP, and the same fixes apply.', 'With memory, Pip can carry what it found from one walk to the next. That is exactly what an adapting learner needs.'],
   shot: [3, 9],
   draw(t) { room({ t, g: SG }); pip(cx(SX), cy(SY), 1, t); [0, 1, 2, 3, 4].forEach(x => txt('?', cx(x), cy(0) - 14, { s: 18, c: C.roseInk, a: clamp(t - 1 - x * .2) }));
     card(['state = Pip’s cell + which cup hides the cheese', { s: 'the cup is hidden: a POMDP', c: C.roseInk }, { s: 'fix: memory, carried across walks', c: '#2f6f6c' }], t - 2.5, { gap: 1.3 }); } },
 { h: 'Memory across walks', p: ['So give Pip a memory that **survives between walks** of the same maze: which cups it found empty, and where the cheese was. It is wiped only when a new maze begins.', `Pip’s policy now reads its cell plus its memory: ${25 * MEMN} situations, ${f0(25 * MEMN * 4)} numbers. Fine for a table here; a real RL² agent’s memory is far bigger, so it uses a network. Pip is never told where the cheese is; it has to find out, and remember.`],
   shot: [3, 10],
   draw(t) { const lift = {}, st = t < 3 ? 0 : t < 5 ? 1 : t < 7 ? 2 : 3; let mem = 0; if (st >= 1) { mem |= 1; lift[0] = 1; } if (st >= 2) { mem |= 2; lift[1] = 1; } if (st >= 3) { mem = 32 + 4; lift[4] = 1; }
     room({ t, g: SG, lift, cups: 1 }); const pos = st === 0 ? [SX, SY] : st === 1 ? [0, 0] : st === 2 ? [1, 0] : [4, 0]; pip(cx(pos[0]), cy(pos[1]), 1, t);
     memBoxes(mem, 160, 290); if (st === 3) bubble('+1', cx(4), cy(0) - 18, t - 7);
     card([{ s: 'kept between walks of one maze, wiped for a new one', c: C.soft }], t - 8, { y: 320, gap: 1 }); } },
 { h: 'Training on many mazes', p: [`The outer loop is just ordinary Q-learning on cell plus memory, across mazes; the inner loop is the memory filling up within one maze. It runs over ${f0(NTRAIN)} mazes with 10 walks each, one move in five random, α 0.2. That is ${f0(EXPSTEPS)} steps of experience.`, 'Every maze rewards the same thing, the cheese. But to collect it quickly in a new maze, the habit that pays is: search the cups, then remember. That is what the numbers come to encode.'],
   shot: [3, 10],
   draw(t) { const n = Math.floor(clamp((t - .5) / 6) * 60); for (let i = 0; i < n; i++) { const R = rng(500 + i), g = [Math.floor(R() * 5), 0]; mini(30 + (i % 12) * 29, 36 + Math.floor(i / 12) * 44, 22, g, i); }
     caption(n >= 60 ? `…and so on: ${f0(NTRAIN)} mazes` : `maze ${n}`);
     if (t > 7) card([`${f0(NTRAIN)} mazes × 10 walks = ${f0(EXPSTEPS)} steps`, { s: 'never told where the cheese is', c: C.soft }], t - 7, { gap: 1.2 }); } },
 { h: 'A new maze', p: [`A maze Pip has never met, the cheese under cup 5. Walk 1: Pip heads for the cups and looks under them until it finds the cheese, in ${DEMO_MEM[0].steps} steps. Its memory now holds the spot. Its order, nearest cup first, then left, then right, is one of the best, ${f1(ORDBEST)} steps on average without random moves; cup 5 is its worst case.`, `Walk 2: straight there, ${DEMO_MEM[1].steps} steps, the shortest. These walks show Pip’s own choices, with no random moves.`],
   shot: [5, 13],
   draw(t) { const W1 = DEMO_MEM[0], W2 = DEMO_MEM[1], D = .55, T1 = (W1.P.length - 1) * D + 1, two = t > T1 + 1.2, w = two ? W2 : W1, tt = two ? t - T1 - 1.2 : t - .6, m = along(w.P, Math.max(0, tt), D), k = clamp(m.done ? w.P.length - 1 : Math.floor(Math.max(0, tt) / D + .29), 0, w.M.length - 1);   // k: cells reached (the step's slide ends at 1/1.4 of its time)
     const mem = w.M[two ? k : k], lift = {}; if (mem >= 32) lift[mem - 32] = 1; else for (let i = 0; i < 5; i++) if ((mem >> i) & 1) lift[i] = 1;
     room({ t, g: SG, lift }); trail(w.P, k + 1, two ? '#3f9a96' : C.teal, .5); pip(m.x, m.y, m.face, t); memBoxes(mem, 160, 296);
     if (m.done) bubble('+1', cx(4), cy(0) - 18, tt - (w.P.length - 1) * D);
     txt(two ? `walk 2: ${m.done ? W2.steps : k} steps` : `walk 1: ${m.done ? W1.steps : k} steps`, 200, 340, { s: 18, c: two ? '#2f6f6c' : C.ink }); } },
 { h: 'Where the learning lives', p: ['On new mazes we freeze Pip’s numbers: none of them changes, yet Pip adapts. The fast learning happens in its **memory**. The slow outer loop learned a fast inner learner.', 'This is the idea of **RL squared**, written RL², from Duan and colleagues and from Wang and colleagues, both 2016. There a recurrent network had to learn what to remember, too. Our toy hands Pip its memory slots ready-made.'],
   shot: [3, 9],
   draw(t) { room({ t, g: SG, lift: { 4: 1 } }); pip(cx(4), cy(0), 1, t); memBoxes(36, 160, 296);
     card(['outer loop (slow): changes the numbers', { s: 'inner loop (fast): only the memory changes', c: '#2f6f6c' }, { s: 'RL²: Duan et al.; Wang et al. (2016)', c: C.soft }], t - 1, { y: 320, gap: 1.3, lh: 22 }); } },
 { h: 'Another idea: a good start', p: ['A second family, **MAML** (Finn, Abbeel and Levine, 2017), keeps ordinary learning as the inner loop, and learns **starting numbers** from which a few learning steps fit any maze in the family.', `Our toy uses Reptile, a simpler cousin of MAML: copy the start, run Q-learning on one maze for ${RK} walks, then move the start a tenth of the way toward the result. Repeat for ${f0(NTRAIN)} mazes. Each move pulls the start toward one maze’s answer; averaged over thousands, it settles on what all mazes share.`],
   shot: [3, 10],
   draw(t) { room({ t, cups: 1 }); const steps = ['1. copy the starting numbers', `2. Q-learning on one maze, ${RK} walks`, '3. move the start 1/10 toward the result', `4. repeat, ${f0(NTRAIN)} mazes`];
     card(steps.map((s, i) => i === 2 ? { s, c: C.goldInk } : s), t - .8, { y: 290, gap: 1.2, lh: 22, fs: 16 }); } },
 { h: 'What the start knows', p: ['Here is the learned start: in each cell, the move with the biggest starting number. Follow the arrows from any cell and you reach the top row, where they sweep right along the cups, and back from cup 5. The start can’t know which cup, since that changes every maze; it only knows where cups are.', 'Without memory, it is one fixed set of numbers. On a new maze, ordinary Q-learning takes it from there.'],
   shot: [3, 9],
   draw(t) { room({ t, arrows: startArrows, arrowA: clamp(t - .8) }); card(['arrows: the best move under the learned start', { s: 'they all lead to the cups', c: C.goldInk }, { s: 'which cup: learned on each new maze', c: C.soft }], t - 3, { gap: 1.2 }); } },
 { h: 'Three learners, new mazes', p: [`Same ${NTEST} new mazes, all three still making one move in five at random. From scratch: ${f1(RES.scratch[0])} steps on walk 1. From the learned start: ${f1(RES.maml[0])}, and ${f1(RES.maml[2])} by walk 3, as Q-learning adapts.`, `With memory: ${f1(RES.mem[0])} on walk 1, then about ${f1(RES.mem.slice(1).reduce((x, y) => x + y, 0) / 9)} per walk from walk 2 on. Random moves keep both above the shortest, ${f1(BEST)}. The good start stalls higher: each visit moves a number only a fifth of the way, so overriding its pull toward every cup takes many walks.`],
   shot: [3, 11],
   draw(t) { const a = clamp(Math.floor((t - .5) * 4), 0, NWK), b = clamp(Math.floor((t - 3.5) * 4), 0, NWK), c = clamp(Math.floor((t - 6.5) * 4), 0, NWK);
     chart({ y0: 372, y1: 60, best: BEST, title: `steps per walk, mean of ${NTEST} new mazes`, lines: [ln('scratch', a), ln('maml', b), ln('mem', c)], legend: [[LAB.scratch, COL.scratch], [LAB.maml, COL.maml], [LAB.mem, COL.mem]].slice(0, c ? 3 : b ? 2 : 1) }); } },
 { h: 'The catch', p: [`Meta-learning only helps on tasks like its training ones. A new family: the cheese on the left wall, in one of three cells at random, 3 to 5 steps from the start. By walk 10: from scratch ${f1(OFF.scratch[9])} steps, learned start ${f1(OFF.maml[9])}, memory learner ${f1(OFF.mem[9])}.`, `Our memory has slots only for cups, so it can’t even record this cheese; a real RL² network might do better or worse. And the ${f0(EXPSTEPS)} training steps pay back only after about ${f0(Math.round(BREAK / 100) * 100)} new cup mazes.`],
   shot: [3, 10],
   draw(t) { const g = [0, 2]; room({ t, offG: t > 1 ? g : null, cups: 1 }); pip(cx(SX), cy(SY), 1, t);
     if (t > 2) card([{ s: 'cheese on the left wall instead (a new family):', c: C.roseInk }, 'steps on walk 10:', { s: `from scratch ${f1(OFF.scratch[9])},  learned start ${f1(OFF.maml[9])}`, c: '#9a6f14' }, { s: `memory learner ${f1(OFF.mem[9])}`, c: '#2f6f6c' }], t - 2, { gap: 1.3 }); } },
 { h: 'Your turn', ctl: { label: 'training mazes', values: NTR, def: 4, fmt: v => f0(v), onChange: () => {} },
   p: ['Slide to change how many mazes the memory learner trained on, from none to 3,000. The chart shows its steps per walk on the same new mazes, with the from-scratch line for comparison.', 'How much outer-loop experience does it take before Pip has learned how to learn?'],
   shot: [3, 9],
   draw(t, st) { const i = st.ci ?? 4, R = ytRes(i), n = t < (st.changed ?? 0) ? NWK : clamp(Math.floor((t - (st.changed ?? 0) - .2) * 8), 0, NWK);
     withGeom(24, 140, 22, () => { room({ t, cups: 1 }); pip(cx(SX), cy(SY), 1, t); });
     chart({ y0: 372, y1: 210, best: BEST, title: `memory learner after ${f0(NTR[i])} training mazes`, lines: [{ v: RES.scratch, col: COL.scratch }, { v: R, col: COL.mem, n }], legend: [[LAB.scratch, COL.scratch], [`trained on ${f0(NTR[i])}`, COL.mem]] });
     txt(`walk 1: ${f1(R[0])} steps,  walk 2: ${f1(R[1])}`, W / 2, 168, { s: 16, c: '#2f6f6c' }); } },
 { h: 'Where it matters', p: ['Meta-RL is studied for robots that must cope with a new load, a new surface or a damaged leg after a few tries, and for systems that must adapt quickly to a new user or opponent.', 'Most results so far are in simulation, and on task families that are close together, like Pip’s cups.'],
   shot: [3, 9],
   draw(t) { room({ t, g: SG, lift: { 4: 1 } }); pip(cx(4), cy(0), 1, t); box(0, 0, W, 285, C.bg, .4);
     card(['robots: new load, new floor, a damaged leg', 'new users, new opponents', { s: 'mostly simulation, close task families', c: C.soft }], t - .6, { y: 120, gap: 1.3 }); } },
 { h: 'Recap', p: ['**Meta-RL** trains on many related tasks so that a new one takes a few tries. The hidden task makes it a POMDP.', '**RL²** learns a fast learner that lives in memory: it searched the cups once, then remembered. **MAML** learns a good starting point for ordinary learning. Both help only on tasks like those they trained on.'],
   shot: [3, 9],
   draw(t) { room({ t, g: SG, lift: { 4: 1 } }); pip(cx(4), cy(0), 1, t); box(0, 0, W, 285, C.bg, .4);
     card(['outer loop over many tasks, inner loop adapts', { s: 'RL²: adaptation in memory, numbers fixed', c: '#2f6f6c' }, { s: 'MAML: a start that adapts in a few steps', c: C.goldInk }, { s: 'only for tasks like the training ones', c: C.roseInk }], t - .3, { y: 120, gap: 1 }); } },
];

RLStory.register('metarl', { title: 'Meta-RL: Learning to Learn', track: 'Bigger worlds', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { RES, OFF, BEST, EXPSTEPS, DEMO_MEM, DEMO_SCR, startArrows, ytRes } });
})();
