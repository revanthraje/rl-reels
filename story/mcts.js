// ===== game_kit.js (copy) =====
// Shared Model-based-track kit: Pip vs the cat at noughts and crosses (3 x 3, three in a row wins).
// Pip (cheese marks) always moves first, the cat (paw marks) second. Values are always Pip's points:
// win 1, draw 0.5, loss 0. See briefs/notes/game_world.md. Safe to load more than once.
(function () {
'use strict';
if (RLStory.game) return;
const L = RLStory.lib, { C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, dot, box, alpha } = L;
const cx = () => L.ctx;
const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
const PIP = 1, CAT = 2;

// ───── rules ─────
function turn(b) { let a = 0, c = 0; for (const v of b) { if (v === 1) a++; else if (v === 2) c++; } return a === c ? PIP : CAT; }
function winLine(b) { for (const l of LINES) if (b[l[0]] && b[l[0]] === b[l[1]] && b[l[0]] === b[l[2]]) return l; return null; }
function winner(b) { const l = winLine(b); return l ? b[l[0]] : 0; }
function full(b) { return b.every(v => v); }
// Pip's points if the game is over, else null
function points(b) { const w = winner(b); if (w === PIP) return 1; if (w === CAT) return 0; return full(b) ? .5 : null; }
function legal(b) { if (points(b) !== null) return []; const m = []; for (let i = 0; i < 9; i++) if (!b[i]) m.push(i); return m; }
function play(b, m) { const nb = b.slice(); nb[m] = turn(b); return nb; }
function code(b) { let k = 0; for (let i = 8; i >= 0; i--) k = k * 3 + b[i]; return k; }
function parse(s) { return s.replace(/\s/g, '').split('').map(ch => ch === 'P' ? 1 : ch === 'C' ? 2 : 0); }

// ───── perfect play (exact minimax over the whole game, memoised) ─────
const VMEM = new Map();
function perfect(b) { // Pip's points if both sides play perfectly from here
  const k = code(b); if (VMEM.has(k)) return VMEM.get(k);
  let v = points(b);
  if (v === null) { const p = turn(b), ms = legal(b); v = p === PIP ? -1 : 2; for (const m of ms) { const u = perfect(play(b, m)); v = p === PIP ? Math.max(v, u) : Math.min(v, u); } }
  VMEM.set(k, v); return v;
}
function bestMoves(b) { const p = turn(b), v = perfect(b); return legal(b).filter(m => Math.abs(perfect(play(b, m)) - v) < 1e-9); }
// every position reachable from the empty board (Pip first), keyed by code
let ALL = null;
function allPositions() {
  if (ALL) return ALL; ALL = new Map(); const st = [new Array(9).fill(0)];
  while (st.length) { const b = st.pop(), k = code(b); if (ALL.has(k)) continue; ALL.set(k, b); for (const m of legal(b)) st.push(play(b, m)); }
  return ALL;
}

// ───── random playout ─────
function playout(b, R, rec) { b = b.slice(); let v; while ((v = points(b)) === null) { const ms = legal(b), m = ms[Math.floor(R() * ms.length)]; b[m] = turn(b); if (rec) rec.push(m); } return v; }

// ───── MCTS with UCB at every node (UCT) ─────
// node = { b, move, parent, kids, untried, n, w (sum of Pip's points) }
function newNode(b, move, parent, R) { const u = legal(b); for (let i = u.length - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [u[i], u[j]] = [u[j], u[i]]; } return { b, move, parent, kids: [], untried: u, n: 0, w: 0, id: 0 }; }
// q of a child seen by the player choosing at the parent: Pip uses Pip's points, the cat uses 1 - Pip's points
function qFor(parent, ch) { const q = ch.w / ch.n; return turn(parent.b) === PIP ? q : 1 - q; }
function ucbScore(parent, ch, c) { return qFor(parent, ch) + c * Math.sqrt(2 * Math.log(parent.n) / ch.n); }
// mcts(b, N, { seed, c, trace }) -> { root, trace: [{ path: [nodes ids], newId, moves (playout), result }] }
function mcts(b, N, o = {}) {
  const R = rng(o.seed ?? 1), c = o.c ?? 1, root = newNode(b, -1, null, R), trace = o.trace ? [] : null; let nid = 0; root.id = nid++;
  for (let it = 0; it < N; it++) {
    let node = root; const path = [root];
    while (!node.untried.length && node.kids.length) { let best = -1e9, bk = null; for (const ch of node.kids) { const s = ucbScore(node, ch, c); if (s > best) { best = s; bk = ch; } } node = bk; path.push(node); }
    let newId = -1;
    if (node.untried.length) { const m = node.untried.pop(), ch = newNode(play(node.b, m), m, node, R); ch.id = nid++; node.kids.push(ch); node = ch; path.push(ch); newId = ch.id; }
    const rec = trace ? [] : null, v = playout(node.b, R, rec);
    for (const p of path) { p.n++; p.w += v; }
    if (trace) trace.push({ path: path.map(p => p.id), newId, moves: rec, result: v, leaf: node });
  }
  return { root, trace };
}
function mostVisited(root) { let best = -1, bm = -1; for (const ch of root.kids) if (ch.n > best) { best = ch.n; bm = ch.move; } return bm; }
// flat Monte Carlo: n random playouts after each move, pick the best average for the player to move
function flatMC(b, n, seed) { const R = rng(seed), p = turn(b), out = {}; for (const m of legal(b)) { let s = 0; const nb = play(b, m); for (let i = 0; i < n; i++) s += playout(nb, R); out[m] = s / n; } let bm = -1, bv = -1; for (const m in out) { const v = p === PIP ? out[m] : 1 - out[m]; if (v > bv) { bv = v; bm = +m; } } return { avg: out, move: bm }; }

// ───── drawing ─────
function cheeseMark(x, y, s, a = 1) { alpha(a * cx().globalAlpha, () => L.cheese(x, y + s * .1, s / 34)); }
function pawMark(x, y, s, a = 1, col = '#a57f70') {
  const k = s / 30, ga = cx().globalAlpha * a; dot(x, y + 4 * k, 7 * k, col, ga); [[-8, -4], [-3, -10], [3, -10], [8, -4]].forEach(([dx, dy]) => dot(x + dx * k, y + dy * k, 3.1 * k, col, ga));
}
// board(x, y, size, b, o): top-left x, y. o = { hi: { cell: colour }, ghost: { cell: who }, seed, mini (dots instead of icons), win, a, num }
function board(x, y, s, b, o = {}) {
  const c = s / 3, sd = o.seed ?? 500, a = (o.a ?? 1) * cx().globalAlpha, mini = o.mini ?? s < 60;
  alpha(a, () => {
    if (o.fill) box(x, y, s, s, o.fill, o.fa ?? 1);
    if (o.hi) for (const k in o.hi) box(x + (k % 3) * c + 2, y + Math.floor(k / 3) * c + 2, c - 4, c - 4, o.hi[k], o.hiA ?? .45);
    const w = mini ? 1 : 1.6;
    for (let i = 1; i < 3; i++) { rline(x + i * c, y + 2, x + i * c, y + s - 2, sd + i, { w, c: C.ink, j: mini ? .5 : 1.2 }); rline(x + 2, y + i * c, x + s - 2, y + i * c, sd + 5 + i, { w, c: C.ink, j: mini ? .5 : 1.2 }); }
    if (o.frame) rrect(x - 3, y - 3, s + 6, s + 6, sd + 20, { c: o.frame, w: o.frameW || 1.2 });
    for (let i = 0; i < 9; i++) { const v = b[i], gx = x + (i % 3) * c + c / 2, gy = y + Math.floor(i / 3) * c + c / 2, ga = o.cellA ? o.cellA(i) : 1;
      if (!v || ga <= 0) continue;
      if (mini) dot(gx, gy, c * .28, v === 1 ? '#e9b84a' : '#a57f70', ga * cx().globalAlpha);
      else v === 1 ? cheeseMark(gx, gy, c * .78, ga) : pawMark(gx, gy, c * .78, ga); }
    if (o.ghost) for (const k in o.ghost) { const gx = x + (k % 3) * c + c / 2, gy = y + Math.floor(k / 3) * c + c / 2, who = o.ghost[k], ga = o.ghostA ?? .45;
      if (mini) dot(gx, gy, c * .28, who === 1 ? '#e9b84a' : '#a57f70', ga * cx().globalAlpha); else who === 1 ? cheeseMark(gx, gy, c * .78, ga) : pawMark(gx, gy, c * .78, ga); }
    if (o.nums) for (const k in o.nums) txt(o.nums[k], x + (k % 3) * c + c / 2, y + Math.floor(k / 3) * c + c / 2, { s: o.numS || Math.max(12, c * .36), c: o.numC || C.goldInk });
    const wl = o.win === false ? null : winLine(b);
    if (wl) { const p = i => [x + (i % 3) * c + c / 2, y + Math.floor(i / 3) * c + c / 2], [x1, y1] = p(wl[0]), [x2, y2] = p(wl[2]); rline(x1, y1, x2, y2, sd + 30, { c: b[wl[0]] === 1 ? C.goldInk : C.roseInk, w: mini ? 1.5 : 3 }); }
  });
}
function cellAt(x, y, s, px, py) { const c = s / 3, i = Math.floor((px - x) / c), j = Math.floor((py - y) / c); return i >= 0 && i < 3 && j >= 0 && j < 3 ? j * 3 + i : -1; }

// ───── a tiny neural network (MLP, plain arrays) ─────
// net = { sizes, W: [Float64Array], b: [Float64Array] }; hidden layers tanh, last layer linear
function mlp(sizes, seed) { const R = rng(seed), W = [], B = []; for (let l = 0; l < sizes.length - 1; l++) { const n = sizes[l], m = sizes[l + 1], s = 1 / Math.sqrt(n), w = new Float64Array(n * m); for (let i = 0; i < w.length; i++) w[i] = (R() * 2 - 1) * s; W.push(w); B.push(new Float64Array(m)); } return { sizes, W, b: B }; }
function forward(net, x) { const acts = [Float64Array.from(x)]; let a = acts[0];
  for (let l = 0; l < net.W.length; l++) { const n = net.sizes[l], m = net.sizes[l + 1], w = net.W[l], o = new Float64Array(m);
    for (let j = 0; j < m; j++) { let s = net.b[l][j]; for (let i = 0; i < n; i++) s += a[i] * w[i * m + j]; o[j] = l < net.W.length - 1 ? Math.tanh(s) : s; }
    acts.push(o); a = o; }
  return acts; }
// backward: gOut = dLoss/dOutput; accumulates into grad {W, b}; returns dLoss/dInput
function backward(net, acts, gOut, grad) { let g = Float64Array.from(gOut);
  for (let l = net.W.length - 1; l >= 0; l--) { const n = net.sizes[l], m = net.sizes[l + 1], w = net.W[l], a = acts[l];
    if (l < net.W.length - 1) { const o = acts[l + 1]; for (let j = 0; j < m; j++) g[j] *= 1 - o[j] * o[j]; }
    const gi = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = 0; const ai = a[i]; for (let j = 0; j < m; j++) { grad.W[l][i * m + j] += ai * g[j]; s += w[i * m + j] * g[j]; } gi[i] = s; }
    for (let j = 0; j < m; j++) grad.b[l][j] += g[j];
    g = gi; }
  return g; }
function zeroGrad(net) { return { W: net.W.map(w => new Float64Array(w.length)), b: net.b.map(b => new Float64Array(b.length)) }; }
// Adam optimiser state
function adam(net, lr) { return { lr, t: 0, m: zeroGrad(net), v: zeroGrad(net) }; }
function adamStep(net, opt, grad, scale) { opt.t++; const b1 = .9, b2 = .999, c1 = 1 - b1 ** opt.t, c2 = 1 - b2 ** opt.t;
  const upd = (P, G, M, V) => { for (let i = 0; i < P.length; i++) { const g = G[i] * scale; M[i] = b1 * M[i] + (1 - b1) * g; V[i] = b2 * V[i] + (1 - b2) * g * g; P[i] -= opt.lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + 1e-8); } };
  for (let l = 0; l < net.W.length; l++) { upd(net.W[l], grad.W[l], opt.m.W[l], opt.v.W[l]); upd(net.b[l], grad.b[l], opt.m.b[l], opt.v.b[l]); } }
const sigmoid = z => 1 / (1 + Math.exp(-z));
function softmaxMasked(z, mask) { let mx = -1e9; for (let i = 0; i < z.length; i++) if (mask[i]) mx = Math.max(mx, z[i]); const p = new Float64Array(z.length); let s = 0; for (let i = 0; i < z.length; i++) if (mask[i]) { p[i] = Math.exp(z[i] - mx); s += p[i]; } for (let i = 0; i < z.length; i++) p[i] /= s; return p; }
// board features: 9 "Pip here" + 9 "cat here"
function feats(b) { const x = new Float64Array(18); for (let i = 0; i < 9; i++) { if (b[i] === 1) x[i] = 1; else if (b[i] === 2) x[9 + i] = 1; } return x; }

RLStory.game = { LINES, PIP, CAT, turn, winLine, winner, full, points, legal, play, code, parse, perfect, bestMoves, allPositions, playout,
  mcts, mostVisited, flatMC, qFor, ucbScore, cheeseMark, pawMark, board, cellAt, mlp, forward, backward, zeroGrad, adam, adamStep, sigmoid, softmaxMasked, feats };
})();
// ===== end game_kit =====
(function () {
'use strict';
const L = RLStory.lib, G = RLStory.game, { W, H, C, lerp, clamp, ease, rng, rline, rrect, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
const GOLD = '#e9b84a', BROWN = '#a57f70', TEAL = '#6fb7b4', TEALINK = '#3f8582';
const B0 = [2, 0, 0, 0, 0, 1, 0, 0, 0];         // cat top left, Pip middle right, Pip to move
const KIDS = [1, 2, 3, 4, 6, 7, 8], WIN = 2, CENTRE = 4;
const NAME = ['top left', 'top middle', 'top right', 'middle left', 'centre', 'middle right', 'bottom left', 'bottom middle', 'bottom right'];
const f2 = v => v.toFixed(2), memo = f => { let v; return () => v ?? (v = f()); };

// ───── simulations (seeded, memoised) ─────
const TREE_SIZE = memo(() => { let n = 0; (function f(b) { n++; for (const m of G.legal(b)) f(G.play(b, m)); })(B0); return n; });
const FLAT = memo(() => G.flatMC(B0, 2000, 5).avg);                 // 2000 random playouts after each move
const PERF = memo(() => { const o = {}; KIDS.forEach(m => o[m] = G.perfect(G.play(B0, m))); return o; });
// the run we follow: 1000 loops, seed 1
const RUN = memo(() => { const r = G.mcts(B0, 1000, { seed: 1, trace: true }), nodes = [];
  (function walk(n, d) { n.depth = d; nodes[n.id] = n; n.kids.forEach(k => walk(k, d + 1)); })(r.root, 0);
  r.trace.forEach((t, i) => { if (t.newId >= 0) nodes[t.newId].it = i; }); r.root.it = -1; r.nodes = nodes;
  // stable layout
  const YD = [39, 118, 190, 214, 236, 256, 274, 286];
  nodes.forEach(n => { if (n.depth === 0) { n.x = 200; } else if (n.depth === 1) n.x = 200 + (KIDS.indexOf(n.move) - 3) * 52;
    else { const sib = n.parent.kids.slice().sort((a, b) => a.move - b.move), i = sib.indexOf(n), sp = 7.4 / Math.pow(1.7, n.depth - 2); n.x = n.parent.x + (i - (sib.length - 1) / 2) * sp; }
    n.y = YD[n.depth]; });
  return r; });
let VA = { k: -1 };
function visitsAt(k) { if (VA.k === k) return VA; const R = RUN(), n = new Int32Array(R.nodes.length), w = new Float64Array(R.nodes.length);
  for (let i = 0; i < k; i++) { const t = R.trace[i]; for (const id of t.path) { n[id]++; w[id] += t.result; } } VA = { k, n, w }; return VA; }
function subSize(n) { let c = 1; n.kids.forEach(k => c += subSize(k)); return c; }
function maxDepth(n) { return n.kids.length ? 1 + Math.max(...n.kids.map(maxDepth)) : 0; }
const kid = (root, m) => root.kids.find(k => k.move === m);
// the first eight loops, for the walk-through
const T8 = memo(() => RUN().trace.slice(0, 10).map(t => ({ move: RUN().nodes[t.path[1]].move, result: t.result, leaf: t.leaf.b, moves: t.moves, path: t.path, newId: t.newId })));
// how often the search picks each move, over 100 seeds, for several loop budgets. Computed a little each frame.
const NS = [10, 30, 100, 300, 1000, 3000], RUNS = 100;
const ACC = NS.map(N => ({ N, done: 0, cnt: {}, q2: 0, q4: 0 }));
function pump(ms) { const t0 = Date.now(); for (const a of ACC) { while (a.done < RUNS) { const r = G.mcts(B0, a.N, { seed: a.done + 1 }).root, m = G.mostVisited(r); a.cnt[m] = (a.cnt[m] || 0) + 1; a.q2 += kid(r, 2).w / kid(r, 2).n; a.q4 += kid(r, 4).w / kid(r, 4).n; a.done++; if (Date.now() - t0 > ms) return false; } } return true; }
const accDone = () => ACC.every(a => a.done >= RUNS);
const ACCF = () => { if (!accDone()) pump(1e9); return ACC; };
const picks = (i, m) => ACCF()[i].cnt[m] || 0;
// averages of the top-right and centre children over the 100 runs (tree shallow vs deep)
const QAVG = () => [3, 5].map(i => ({ q2: ACCF()[i].q2 / RUNS, q4: ACCF()[i].q4 / RUNS }));
// one run (seed 1) at each budget, for "your turn"
const ONE = {}; const one = N => ONE[N] ?? (ONE[N] = G.mcts(B0, N, { seed: 1 }).root);
function deepFacts() { const r = G.mcts(B0, 3000, { seed: 1 }).root, c4 = kid(r, 4), c2 = kid(r, 2), c28 = kid(c2, 8);
  const top = n => n.kids.slice().sort((a, b) => b.n - a.n)[0];
  return { catVsCentre: top(c4).move, catVsCentreShare: top(c4).n / c4.n, catVsTR: top(c2).move, pipAfter: top(c28).move, nodes3000: subSize(r) }; }
const DEEP = memo(deepFacts);
// loops of the 1000-loop run that reached a finished game and added no node (all, and per root move)
const NONODE = memo(() => { const R = RUN(), o = { all: 0 }; R.trace.forEach(t => { if (t.newId >= 0) return; o.all++; const m = R.nodes[t.path[1]].move; o[m] = (o[m] || 0) + 1; }); return o; });

// ───── drawing helpers ─────
function bigBoard(b, o = {}) { G.board(o.x ?? 125, o.y ?? 40, o.s ?? 150, b, o); }
function legend(y, a = 1) { alpha(a, () => { G.cheeseMark(118, y, 18); txt('Pip', 132, y + 1, { s: 15, al: 'left' }); G.pawMark(232, y, 18); txt('cat', 246, y + 1, { s: 15, al: 'left' }); }); }
// tree after k loops (k ≤ 1000); o = { path, labels, grow (id -> 0..1), avg }
// depth 1: mini boards; depth 2: the cat's replies as dots (size = visits); deeper: one bar per level, width = number of nodes there
const DY = [0, 0, 192, 214, 230, 246, 262, 278];
function drawTree(k, o = {}) {
  const R = RUN(), V = visitsAt(k), N = R.nodes, show = n => n.it < k || (o.grow && o.grow[n.id] > 0), lab = o.labels;
  G.board(167, 6, 66, R.root.b, { frame: o.path ? TEALINK : null, seed: 2990 });
  if (lab && V.n[0]) txt('N ' + V.n[0], 242, 30, { s: 13, c: C.ink, al: 'left' });
  const cnt = {};
  for (const n of N) { if (!n || n.depth === 0 || !show(n)) continue; const p = n.parent, g = o.grow?.[n.id] ?? 1, onPath = o.path && o.path.includes(n.id);
    if (n.depth === 1) { rline(200, 74, lerp(200, n.x, g), lerp(74, 100, g), 3000 + n.id, { c: onPath ? TEALINK : C.line, w: onPath ? 2.4 : 1 });
      G.board(n.x - 18, 100, 36, n.b, { a: g, frame: onPath ? TEALINK : null, seed: 3100 + n.move, hi: { [n.move]: C.yellow }, hiA: .8 });
      if (lab && V.n[n.id]) { txt(o.top ? o.top(n) : 'n ' + V.n[n.id], n.x, 148, { s: 13, c: o.topC ? o.topC(n) : C.ink }); if (o.avg !== false) txt(f2(V.w[n.id] / V.n[n.id]), n.x, 162, { s: 13, c: C.goldInk }); } }
    else if (n.depth === 2) { const y0 = lab ? 170 : 142; rline(p.x, y0, lerp(p.x, n.x, g), lerp(y0, DY[2], g), 3000 + n.id, { c: onPath ? TEALINK : C.line, w: onPath ? 2.4 : .9, once: !onPath });
      const r = clamp(2.2 + Math.log2(V.n[n.id] + 1) * .5, 2.2, 5.2); dot(n.x, DY[2], r, BROWN, g); }
    else { let a = n; while (a.depth > 1) a = a.parent; const key = a.move + ':' + n.depth; cnt[key] = (cnt[key] || 0) + 1; if (onPath) cnt[key + 'p'] = 1; } }
  for (const key in cnt) { if (key.endsWith('p')) continue; const [m, d] = key.split(':').map(Number), x = 200 + (KIDS.indexOf(m) - 3) * 52, w = Math.min(46, 3 + 4.4 * Math.sqrt(cnt[key]));
    box(x - w / 2, DY[d] - 3.5, w, 7, d % 2 ? GOLD : BROWN, .85); if (cnt[key + 'p']) rrect(x - w / 2 - 2, DY[d] - 5.5, w + 4, 11, 3050 + d, { c: TEALINK, w: 1.6 }); }
}
// playout board: position at the new node, then the playout moves appear one by one (u = 0..1)
function playoutBoard(x, y, s, leafB, moves, u) { let b = leafB.slice(); const k = Math.floor(u * (moves.length + .999)); for (let i = 0; i < Math.min(k, moves.length); i++) b = G.play(b, moves[i]);
  const ghost = {}; G.board(x, y, s, b, { seed: 3400, cellA: i => leafB[i] ? 1 : .55, win: k >= moves.length }); }
const resWord = v => v === 1 ? 'Pip wins: 1 point' : v === .5 ? 'draw: ½ point' : 'cat wins: 0 points';
const resCol = v => v === 1 ? C.goldInk : v === .5 ? C.soft : C.roseInk;
// accuracy chart: x = loops (log), y = runs out of 100
function accChart(o = {}) { const x0 = o.x0 ?? 60, x1 = o.x1 ?? 360, y0 = o.y0 ?? 250, y1 = o.y1 ?? 80, X = i => lerp(x0 + 18, x1 - 8, i / (NS.length - 1)), Y = v => lerp(y0, y1, v / 100), a = o.a ?? 1;
  alpha(a, () => { rline(x0, y0, x1, y0, 3500, { c: C.ink }); rline(x0, y0, x0, y1 - 6, 3501, { c: C.ink });
    [0, 50, 100].forEach(v => { txt(String(v), x0 - 8, Y(v), { s: 12, c: C.soft, al: 'right' }); if (v) rline(x0, Y(v), x1, Y(v), 3510 + v, { c: '#e6dccd', w: .8, once: true }); });
    NS.forEach((N, i) => txt(N >= 1000 ? N / 1000 + 'k' : String(N), X(i), y0 + 13, { s: 12, c: C.soft }));
    txt(o.xl ?? 'loops per decision', (x0 + x1) / 2, y0 + 28, { s: 13, c: C.soft }); txt(o.yl ?? 'runs out of 100', x0 - 30, y1 - 18, { s: 13, c: C.soft, al: 'left' }); });
  (o.lines || []).forEach(({ m, col, ink, n, label }) => { const vs = NS.map((N, i) => picks(i, m)), shown = clamp(n ?? 99) * (NS.length - 1);
    for (let i = 0; i < NS.length - 1 && i < shown; i++) { const u = clamp(shown - i); rline(X(i), Y(vs[i]), lerp(X(i), X(i + 1), u), lerp(Y(vs[i]), Y(vs[i + 1]), u), 3600 + i + m * 10, { c: col, w: 2.2 }); }
    for (let i = 0; i <= Math.min(NS.length - 1, Math.floor(shown + .01)); i++) { dot(X(i), Y(vs[i]), 3.6, col); if (o.vals) txt(String(vs[i]), X(i), Y(vs[i]) + (o.below?.(m, i) ? 14 : -11), { s: 12, c: ink }); }
    if (label) { const j = o.legend.indexOf(m), la = clamp((n ?? 99) * 3); rline(x0 + 14, y1 + 2 + j * 18, x0 + 34, y1 + 2 + j * 18, 3690 + j, { c: col, w: 2.4, a: la }); txt(label, x0 + 40, y1 + 2 + j * 18, { s: 14, c: ink, al: 'left', a: la }); } });
  if (o.mark != null) { const i = o.mark; ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.soft; ctx.beginPath(); ctx.moveTo(X(i), y0); ctx.lineTo(X(i), y1); ctx.stroke(); ctx.restore(); }
}
// a small 9 by 9 Go board with a few stones
function goBoard(x, y, s, a) { alpha(a, () => { box(x - 6, y - 6, s + 12, s + 12, '#ecd3a0'); const g = s / 8;
  for (let i = 0; i < 9; i++) { rline(x, y + i * g, x + s, y + i * g, 4400 + i, { c: '#6b5a44', w: .9, j: .5, once: true }); rline(x + i * g, y, x + i * g, y + s, 4420 + i, { c: '#6b5a44', w: .9, j: .5, once: true }); }
  [[2, 2, 1], [6, 2, 0], [2, 6, 1], [6, 6, 0], [4, 4, 1], [3, 5, 0], [5, 3, 1], [4, 6, 0], [6, 4, 1], [2, 4, 0]].forEach(([i, j, b], k) => { if (a * 10 > k * .6) L.rcircle(x + i * g, y + j * g, g * .42, 4440 + k, { fill: b ? C.ink : '#fbf8f2', j: .4 }); }); }); }
// a hand-drawn tree diagram for the "four steps" picture
function miniTree(t) { const P = [[200, 40], [130, 100], [200, 100], [270, 100], [100, 160], [160, 160], [240, 160]], E = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [3, 6]];
  const ph = Math.floor(t / 2.6) % 4, u = (t % 2.6) / 2.6, path = [0, 1, 5];
  E.forEach(([a, b], i) => rline(P[a][0], P[a][1] + 9, P[b][0], P[b][1] - 9, 3700 + i, { c: (ph === 0 || ph === 3) && path.includes(a) && path.includes(b) ? TEALINK : C.line, w: (ph === 0 || ph === 3) && path.includes(a) && path.includes(b) ? 2.4 : 1.2 }));
  P.forEach(([x, y], i) => L.rcircle(x, y, 9, 3720 + i, { fill: i === 0 ? C.yellow : C.card }));
  const nx = 175, ny = 220; if (ph >= 1) { rline(160, 169, nx, ny - 9, 3740, { c: ph !== 2 ? TEALINK : C.line, w: 2, p: ph === 1 ? clamp(u * 2) : 1 }); if (ph > 1 || u > .4) L.rcircle(nx, ny, 9, 3741, { fill: '#cfe7e5' }); }
  if (ph === 2) { for (let k = 0; k < 4; k++) { const v = clamp(u * 5 - k); if (v > 0) dot(nx + Math.sin(k * 2.1) * 8, ny + 18 + k * 12, 2.6, k % 2 ? BROWN : GOLD, v); } if (u > .8) txt('1', nx + 16, ny + 62, { s: 16, c: C.goldInk }); }
  if (ph === 3) { [[nx, ny], ...path.slice().reverse().map(i => P[i])].forEach(([x, y], j) => { const v = clamp(u * 4 - j * .7); if (v > 0) txt('+1', x + 16, y - 10, { s: 14, c: C.goldInk, a: v }); }); }
  const names = ['1  select: walk down by UCB', '2  expand: add one new node', '3  simulate: random playout', '4  back up: add the result'];
  names.forEach((s, i) => txt(s, 220, 194 + i * 22, { s: 16, al: 'left', c: i === ph ? TEALINK : C.soft }));
}

// ───── steps ─────
const STEPS = [
 { h: 'Pip versus the cat', p: ['Pip and the cat play **noughts and crosses**: a 3 by 3 board, three in a row wins. Pip goes first and draws cheese; the cat draws paws. A full board with no line is a draw.', 'We keep score in **Pip’s points**: a win is 1, a draw ½, a loss 0. The cat wants Pip’s points low.'],
   shot: [3, 9],
   say: [["Welcome to a game night. Pip and the cat play noughts and crosses."], [["Pip draws cheese, the cat draws paws. Three in a row wins.", [110, 24, 180, 180]]], [["We keep score in Pip's points. A win is one, a draw is a half, a loss is zero.", [30, 300, 340, 64]], "The cat wants Pip's points low."]],
   draw(t) { const seq = [4, 0, 2, 6, 3, 5, 7, 1, 8], k = Math.min(seq.length, Math.floor(t / .9)); let b = Array(9).fill(0); for (let i = 0; i < k; i++) b = G.play(b, seq[i]);
     bigBoard(b, { y: 34 }); legend(214, clamp(t - .5));
     if (t > 7.5) card(['win 1   ·   draw ½   ·   loss 0', { s: 'all scores are Pip’s points', c: C.soft }], t - 7.5, { y: 300 });
     else caption(k >= seq.length ? 'a sample game: full board, no line: a draw' : 'a sample game', clamp(t - 1)); } },

 { h: 'Pip’s move', p: ['Another game, two moves in: Pip took the middle right, the cat answered in the top left corner. Now it is Pip’s turn, with **7 empty squares**.', 'Pip knows the rules exactly: where each move puts a mark and when a game ends. That rulebook is a **model** of the game, so Pip can play moves out in its head before choosing one.'],
   say: [[["Here's a game two moves in. Now it's Pip's turn, with seven empty squares.", [110, 24, 180, 180]]], ["Pip knows the rules exactly. Where each move puts a mark, and when the game ends."], [["That rulebook is a model. So Pip can play moves out in its head before choosing.", [30, 300, 340, 64]]]],
   draw(t) { bigBoard(B0, { y: 34, hi: Object.fromEntries(KIDS.map(m => [m, C.yellow])), hiA: .35 * clamp(t - 2) }); legend(214);
     if (t > 4) card(['Pip to move: which square?', { s: 'Pip can imagine moves, the cat’s replies, and so on', c: C.soft }], t - 4, { y: 300 }); } },

 { h: 'A tree of futures', p: () => [`Under the current board, a branch for each of Pip’s 7 moves; under each, the cat’s 6 replies, and so on until the games end. This is a **search tree**; each spot is a **node**.`, `Counting every partial game, even repeats from different move orders, the tree has ${TREE_SIZE().toLocaleString('en-US')} nodes, few enough to check. Go, with about 250 moves per turn, is not.`],
   shot: [4, 11],
   get say() { return [["Picture every possible future as a tree."], [["Under the board, a branch for each of Pip's seven moves. Under each, the cat's replies, and so on, until the games end.", [10, 20, 380, 270]]], [`Here that's ${TREE_SIZE().toLocaleString('en-US')} nodes, few enough to check. In Go, the tree is unimaginably bigger.`]]; },
   draw(t) { G.board(167, 6, 66, B0, { seed: 2990 });
     KIDS.forEach((m, i) => { const x = 200 + (i - 3) * 52, a = clamp((t - .5 - i * .35) * 2); if (a <= 0) return; rline(200, 74, x, 100, 3800 + i, { c: C.line, p: a }); G.board(x - 18, 100, 36, G.play(B0, m), { a, seed: 3100 + m, hi: { [m]: C.yellow }, hiA: .8 });
       const b1 = G.play(B0, m), rep = G.legal(b1); rep.forEach((r, j) => { const xx = x + (j - (rep.length - 1) / 2) * 7.4, aa = clamp((t - 3.5 - i * .3) * 2); if (aa > 0) { rline(x, 140, xx, 188, 3820 + i * 9 + j, { c: C.line, w: .8, once: true, p: aa }); dot(xx, 190, 2.6, BROWN, aa); }
         if (t > 6) for (let q = 0; q < 5; q++) { const x3 = xx + (q - 2) * 1.4, a3 = clamp((t - 6 - i * .2) * 2); rline(xx, 192, x3, 214, 3900 + i * 60 + j * 6 + q, { c: C.line, w: .5, once: true, p: a3 }); } }); });
     if (t > 1) txt('now', 248, 40, { s: 14, c: C.soft, al: 'left', a: clamp(t - 1) }); if (t > 2.5) txt('Pip’s 7 moves', 375, 92, { s: 13, c: C.goldInk, al: 'right', a: clamp(t - 2.5) });
     if (t > 4) txt('the cat’s 6 replies to each', 375, 200, { s: 13, c: C.roseInk, al: 'right', a: clamp(t - 4) });
     if (t > 6.5) txt('… and so on, down to the end of each game', 200, 236, { s: 14, c: C.soft, a: clamp(t - 6.5) });
     if (t > 8) card([`${TREE_SIZE().toLocaleString('en-US')} nodes from this position`, { s: 'Go: about 250 moves per turn: far too many', c: C.roseInk }], t - 8, { y: 300 }); } },

 { h: 'Play it out at random', p: ['A cheap way to judge a move: play it, then let both sides make **random moves** until the game ends. That is a **playout**. Here Pip tries bottom left, and the random game ends in a win: 1 point.', 'One playout is a coin flip. The average over many playouts estimates how good the move is. Averaging random trials is what **Monte Carlo** means.'],
   shot: [5, 11],
   say: [["Here's a cheap way to judge a move. Play it, then let both sides move at random until the game ends."], [["Watch this playout. Pip tries bottom left, and the random game ends in a win.", [10, 20, 380, 270]]], ["One playout is a coin flip. But average many, and you get an estimate. That's what Monte Carlo means."]],
   draw(t) { const e = T8()[0]; txt('Pip tries bottom left, then random moves', 200, 22, { s: 15, c: C.soft });
     playoutBoard(125, 40, 150, e.leaf, e.moves, clamp((t - 1) / 5)); legend(214);
     if (t > 6.5) bubble(resWord(e.result), 200, 250, t - 6.5, resCol(e.result));
     if (t > 8.5) card(['playout = random moves to the end', { s: 'faded marks: the random part', c: C.soft }], t - 8.5, { y: 300 }); } },

 { h: 'Averages from playouts', p: () => { const f = FLAT(); return [`So run **2000 playouts** after each of Pip’s 7 moves. Each number on the board is the average of Pip’s points over the 2000 random games that start with a move there.`, `The **centre** looks best, ${f2(f[4])}. Top right comes second at ${f2(f[2])}; bottom middle is worst at ${f2(f[7])}. Should Pip play the centre?`]; },
   get say() { const f = FLAT(); return [["So let's run two thousand random games after each of Pip's moves."], [["Each number is the average of Pip's points.", [110, 24, 180, 180]], `The centre looks best, at ${f2(f[4])}.`], ["Should Pip play the centre?"]]; },
   draw(t) { const f = FLAT(), nums = {}; KIDS.forEach((m, i) => { if (t > 1 + i * .5) nums[m] = f2(f[m]); });
     bigBoard(B0, { y: 34, nums, numS: 20, hi: t > 6 ? { 4: C.yellow } : null, hiA: .6 }); legend(214);
     if (t > 6.5) card(['number = average of Pip’s points', { s: 'over 2000 random playouts after that move', c: C.soft }], t - 6.5, { y: 300 }); } },

 { h: 'The careless cat', p: () => [`Checking every line backwards from its end, with Pip taking most points, the cat fewest, **only top right wins**. Centre and bottom left draw; the other four lose, even bottom right, which the playouts rated ${f2(FLAT()[8])}.`, 'Top right threatens the right column, the cat must block, then Pip’s centre makes two threats. Playouts missed it: their cat rarely defends. Moves must survive a **careful** opponent.'],
   shot: [4, 12],
   say: [["Not so fast. Check every line backwards from the end, with both sides playing their best."], [["Only top right wins.", [110, 24, 180, 180]], ["Watch why. Top right threatens the column, the cat must block, then the centre makes two threats at once.", [110, 24, 180, 180]]], [["The playouts missed it, because their random cat rarely defends.", [30, 300, 340, 64]]]],
   draw(t) { const P = PERF(), w = v => v === 1 ? 'win' : v === .5 ? 'draw' : 'loss', ph = t < 6 ? 0 : Math.min(3, Math.floor((t - 6) / 1.6) + 1);
     if (ph === 0) { const nums = {}; KIDS.forEach((m, i) => { if (t > .8 + i * .4) nums[m] = w(P[m]); }); bigBoard(B0, { y: 34, nums, numS: 19, numC: C.ink, hi: Object.fromEntries(KIDS.filter((m, i) => t > .8 + i * .4).map(m => [m, P[m] === 1 ? C.yellow : P[m] === .5 ? '#e8e2d8' : C.pink])), hiA: .7 }); caption('best play by both sides, checked exhaustively', clamp(t - 3)); }
     else { const seq = [2, 8, 4], b = seq.slice(0, ph).reduce((bb, m) => G.play(bb, m), B0); bigBoard(b, { y: 34, hi: ph === 3 ? { 3: C.yellow, 6: C.yellow } : ph === 1 ? { 8: C.pink } : null, hiA: .6 });
       txt(['', 'Pip: top right, threatens the right column', 'cat: must block bottom right', 'Pip: centre, two threats: cat can block only one'][ph], 200, 22, { s: 15, c: ph === 2 ? C.roseInk : C.goldInk }); }
     legend(214); if (t > 11) card(['random cat: rarely blocks', { s: 'random Pip: never follows its plan', c: C.soft }], t - 11, { y: 300 }); } },

 { h: 'Let the tree think for both', p: ['**Monte Carlo Tree Search** (MCTS) grows a tree over playouts. Both sides pick by **UCB**: each node a cheese arcade, its moves the machines. The cat uses **cat’s points**, 1 minus Pip’s.', 'Imagined pulls are free; UCB still aims loops where choices are decided. Growing trees make averages drift, breaking the arcade’s guarantee; a 2006 proof shows MCTS still finds best play, given enough loops.'],
   say: [["Monte Carlo tree search grows a tree on top of the playouts."], ["At every node, both sides choose with UCB, like the cheese arcade from the bandits reel. The cat uses its own points, one minus Pip's."], [["So the search aims its effort where the real choices are decided.", [30, 296, 340, 64]]]],
   draw(t) { G.board(167, 6, 66, B0, { seed: 2990 });
     KIDS.forEach((m, i) => { const x = 200 + (i - 3) * 52, a = clamp((t - .4 - i * .2) * 2); rline(200, 74, x, 100, 3800 + i, { c: C.line, p: a }); G.board(x - 18, 100, 36, G.play(B0, m), { a, seed: 3100 + m, hi: { [m]: C.yellow }, hiA: .8 }); });
     if (t > 2.5) { alpha(clamp(t - 2.5), () => { txt('score = average + c × √( 2 × ln N ÷ n )', 200, 170, { s: 18 }); txt('N: visits of this node,  n: visits of the move’s node,  c = 1', 200, 194, { s: 13, c: C.soft }); }); }
     if (t > 5) card(['Pip’s turn: average of Pip’s points', { s: 'cat’s turn: average of the cat’s points (1 − Pip’s)', c: C.roseInk }, { s: 'below the tree: random playouts, as before', c: C.soft }], t - 5, { y: 296, gap: 1.2 }); } },

 { h: 'One loop, four steps', p: ['Each node keeps **n**, its loop count, and the **total** of Pip’s points. Average = total ÷ n, a Monte Carlo estimate. **Select**: from the top, follow top UCB scores to an untried move. **Expand**: add it as a node. **Simulate**: one random playout. **Back up**: add its points along the path.', 'A walk ending at a finished game backs up its real result instead.'],
   shot: [3, 9, 12],
   say: [["Each loop has four steps."], [["Select. Follow the best UCB scores down the tree, to an untried move.", [50, 20, 300, 180]], ["Expand. Add it as a new node.", [50, 20, 300, 180]], ["Simulate. One random playout.", [50, 20, 300, 180]], ["Back up. Add its points along the path.", [50, 20, 300, 180]]]],
   draw(t) { miniTree(t); caption('the loop repeats: at most one new node each time', clamp(t - 10.4)); } },

 { h: 'The first seven loops', p: () => { const e = T8(); return ['At first the top node has untried moves, so each loop simply adds one of Pip’s moves, in random order, and plays it out once, like trying each arcade machine once.', `Results, left to right in the picture: ${e.slice(0, 7).sort((a, b) => KIDS.indexOf(a.move) - KIDS.indexOf(b.move)).map(x => NAME[x.move] + ' ' + (x.result === .5 ? '½' : x.result)).join(', ')}. Pip’s points, from one coin flip each.`]; },
   shot: [5, 13],
   say: [["At first, the top node has untried moves. So each loop just adds one of Pip's moves, and plays it out once."], [["Watch the seven moves appear, one per loop.", [10, 20, 380, 270]], ["Each gets one coin flip of a result.", [10, 20, 380, 270]]], [["The card explains the numbers under each move.", [30, 300, 340, 64]]]],
   draw(t) { const e = T8(), D = 1.7, k = Math.min(7, Math.floor(t / D)), u = k < 7 ? (t - k * D) / D : 1, grow = {};
     if (k < 7) grow[e[k].newId] = clamp((u - .15) * 4);
     drawTree(k + (k < 7 && u > .85 ? 1 : 0), { labels: true, grow, path: k < 7 ? e[k].path : null });
     if (k < 7) { const x = e[k]; playoutBoard(40, 300, 84, x.leaf, x.moves, clamp((u - .3) / .45)); txt(`loop ${k + 1}: Pip tries ${NAME[x.move]}`, 140, 318, { s: 16, al: 'left' }); txt('then a random playout', 140, 342, { s: 14, al: 'left', c: C.soft });
       if (u > .76) txt(resWord(x.result), 140, 368, { s: 16, al: 'left', c: resCol(x.result) }); }
     else card(['under each move:  n = visits', { s: 'gold number = average of Pip’s points', c: C.goldInk }], t - 7 * D, { y: 300 }); } },

 { h: 'Loop 8: UCB chooses', p: () => { const b = Math.sqrt(2 * Math.log(7)); return [`Every move has n = 1; the top, N = 7. Each bonus is √(2 × ln 7 ÷ 1) = ${f2(b)}. The three winners tie; the tie goes to the earliest added, so loop 8 goes to **bottom left**.`, `It adds a random untried reply, the cat in the centre, whose playout the cat wins. Bottom left now shows n = 2, average 0.5.`]; },
   shot: [4, 11],
   say: [["Loop eight is the first real choice."], [["Every move has been tried once, so the exploration bonuses are equal.", [10, 20, 380, 270]], ["The winners tie, and the earliest one, bottom left, gets the visit.", [10, 20, 380, 270]]], ["The search adds a cat reply, and plays it out. This time the cat wins."]],
   draw(t) { const e = T8()[7], b = Math.sqrt(2 * Math.log(7)), R = RUN(), V = visitsAt(7), grow = {}; grow[e.newId] = clamp((t - 5) * 2);
     drawTree(t > 9.5 ? 8 : 7, { labels: true, path: t > 3.5 ? e.path : null, grow });
     if (t < 4.5) { R.root.kids.forEach(n => { const s = V.w[n.id] / V.n[n.id] + b, a = clamp((t - .6) * 2); txt(f2(s), n.x, 182, { s: 14, c: s > 2.9 ? C.goldInk : C.soft, a }); }); if (t > 1.2) txt('UCB score of each move', 200, 202, { s: 13, c: C.soft, a: clamp(t - 1.2) }); }
     if (t > 5.5) { playoutBoard(40, 300, 84, e.leaf, e.moves, clamp((t - 6) / 2.5)); txt('loop 8: bottom left, then', 140, 318, { s: 16, al: 'left' }); txt('the cat answers in the centre', 140, 342, { s: 14, al: 'left', c: C.roseInk }); if (t > 8.7) txt(resWord(e.result), 140, 368, { s: 16, al: 'left', c: resCol(e.result) }); } } },

 { h: 'The tree grows', p: ['Loop after loop, the same four steps. Under each move, brown dots are the cat’s replies; bigger means visited more. Deeper levels are bars, gold for Pip’s moves, brown for the cat’s: wider bar, more nodes at that depth.', 'Moves that keep scoring well get walked often, so their branches grow **deeper**. Weak moves still get an occasional look: their bonus creeps up while ignored.'],
   shot: [6, 13],
   say: [["Now loop after loop, the same four steps."], [["Watch the tree grow. Bigger dots mean more visits.", [10, 20, 380, 270]], ["Moves that keep scoring well get walked often, so their branches grow deeper.", [10, 20, 380, 270]]], ["Weak moves still get an occasional look. Their bonus creeps up while they're ignored."]],
   draw(t) { const k = Math.min(1000, Math.round(8 * Math.pow(1000 / 8, clamp(t / 12)))); drawTree(k, { labels: true }); caption(`after ${k} loops  ·  bars: gold Pip, brown cat`); } },

 { h: 'After 1000 loops', p: () => { const r = RUN().root, c2 = kid(r, 2), c4 = kid(r, 4); return [`Top right was visited ${c2.n} times, the centre ${c4.n}, no other over ${Math.max(...r.kids.filter(k => k.move !== 2 && k.move !== 4).map(k => k.n))}. It also averages best, ${f2(c2.w / c2.n)} against ${f2(c4.w / c4.n)}.`, 'MCTS plays the move it **visited most**. A rare move’s average can be lucky; visits pile up only on moves that keep scoring. Averages include exploring loops and playout blunders, so even a sure win averages below 1. It can stop anytime with an answer.']; },
   shot: [4, 11],
   get say() { const r = RUN().root, c2 = kid(r, 2); return [["After a thousand loops, here's the tree."], [["Top right got by far the most visits.", [10, 20, 380, 270]], `${c2.n} of the thousand.`], [["MCTS plays the move it visited most. A lucky average can fool you, but visits only pile up on moves that keep scoring.", [30, 300, 340, 64]]]]; },
   draw(t) { const r = RUN().root; drawTree(1000, { labels: true });
     if (t > 3) { const n = kid(r, 2); rrect(n.x - 24, 94, 48, 76, 3990, { c: C.goldInk, w: 2 }); }
     if (t > 5) card(['play the most visited move: top right', { s: 'Pip’s points: averages in gold, visits n', c: C.soft }], t - 5, { y: 300 }); } },

 { h: 'Where the effort went', p: () => { const r = RUN().root, c2 = kid(r, 2), c7 = kid(r, 7); return [`1000 loops built ${subSize(r)} nodes, top one included, out of ${TREE_SIZE().toLocaleString('en-US')}. ${NONODE().all} loops reached finished games and added none, ${NONODE()[2]} under top right. Top right’s branch has ${subSize(c2)} nodes, ${maxDepth(c2)} moves deep; bottom middle’s only ${subSize(c7)}, ${maxDepth(c7)} deep.`, 'Value iteration also plans with a model, but sweeps everything. MCTS goes **deep where it matters**, shallow elsewhere, so it handles games too big to search fully.']; },
   get say() { return [["Where did the effort go?"], [["Top right's branch runs deep. Weak moves stay shallow.", [10, 20, 380, 270]], [`A thousand loops built only ${subSize(RUN().root)} nodes.`, [30, 300, 340, 64]]], ["Value iteration sweeps everything. MCTS goes deep where it matters, so it handles games too big to search fully."]]; },
   draw(t) { const r = RUN().root; drawTree(1000, { labels: t > 1, avg: false, top: n => String(subSize(n)), topC: n => n.move === 2 ? C.goldInk : C.ink });
     if (t > 4) card([`built ${subSize(r)} of ${TREE_SIZE().toLocaleString('en-US')} nodes`, { s: 'number under each move: nodes in its branch', c: C.soft }, { s: 'deep under the strong moves, shallow elsewhere', c: C.soft }], t - 4, { y: 300, gap: 1.1 }); } },

 { h: 'How many loops are enough?', p: () => [`Is 1000 loops luck? Run the search 100 times, with different random playouts, for several budgets. Left to right: loops per decision, 10, 30, 100, 300, 1000 and 3000. Upward: how many of the 100 runs chose that move.`, `Gold: top right, the winning move. With 10 loops, only ${picks(0, 2)} runs: about a random guess among 7. With 1000 loops, ${picks(4, 2)}; with 3000, ${picks(5, 2)}.`],
   shot: [4, 12],
   say: [["Was a thousand loops just lucky? Let's run the search a hundred times, at several budgets."], [["Across, the number of loops. Up, how many runs chose each move.", [40, 66, 330, 200]], ["Gold is top right, the winning move.", [40, 66, 330, 200]], ["With few loops, it's a guess. With many, gold rises to the top.", [40, 66, 330, 200]]]],
   draw(t) { accChart({ a: clamp(t * 2), lines: t > 4 ? [{ m: WIN, col: GOLD, ink: C.goldInk, n: (t - 4) / 5, label: 'top right, the winning move' }] : [], vals: true, legend: [2, 4] });
     if (t > 1.5 && t < 4) caption('first the axes: loops across, runs upward', clamp(t - 1.5)); } },

 { h: 'The tree first agrees with the playouts', p: () => { const q = QAVG(); return [`In teal, the centre. MCTS picks it more up to 300 loops, ${picks(3, 4)} of 100 runs: a shallow tree’s averages, in the table, are playout averages, near the playouts’ ${f2(FLAT()[2])} and ${f2(FLAT()[4])}.`, `Deeper trees reach the cat’s replies: under the centre, its one good answer, middle left; under top right, Pip’s plan. At 3000 loops top right leads, below 1 as exploring and playouts blunder.`]; },
   shot: [3, 12],
   say: [[["Teal is the centre. With a few hundred loops, MCTS often picks it.", [40, 66, 330, 200]]], ["A shallow tree's averages are mostly playout averages. So it believes them too."], [["Only deeper trees discover the cat's one good answer to the centre. Then top right takes the lead.", [40, 66, 330, 200]]]],
   draw(t) { accChart({ lines: [{ m: WIN, col: GOLD, ink: C.goldInk, label: 'top right, the winning move', n: 99 }, { m: CENTRE, col: TEAL, ink: TEALINK, n: (t - 1) / 4, label: 'centre' }], vals: true, below: (m, i) => (m === 4) === (picks(i, 4) < picks(i, 2)), legend: [2, 4] });
     if (t > 7) { const q = QAVG(), a = clamp((t - 7) * 2); card([' ', ' ', ' '], t - 7, { y: 290, lh: 22 });
       alpha(a, () => { txt('average, Pip’s points', 44, 310, { s: 15, al: 'left', c: C.soft }); txt('300 loops', 250, 310, { s: 15, c: C.soft }); txt('3000 loops', 330, 310, { s: 15, c: C.soft });
         [[2, 'top right', C.goldInk], [4, 'centre', TEALINK]].forEach(([m, nm, col], j) => { const y = 334 + j * 22; txt(nm, 44, y, { s: 16, al: 'left', c: col }); txt(f2(m === 2 ? q[0].q2 : q[0].q4), 250, y, { s: 16, c: col }); txt(f2(m === 2 ? q[1].q2 : q[1].q4), 330, y, { s: 16, c: col }); }); }); } } },

 { h: 'Your turn', ctl: { label: 'loops', values: NS, def: 4, fmt: v => v }, p: () => ['Drag the number of **loops** per decision. The board shows one search: the share of loops that went through each move, with the move it would play highlighted. Below, the result over 100 searches.', `With few loops the visits spread out. The centre peaks at 300 loops, chosen in ${picks(3, 4)} of 100 searches. By 3000, every one of the 100 searches plays top right.`],
   shot: [3, 8],
   say: [["Your turn. Drag the number of loops per decision."], [["The board shows one search, with the share of loops through each move.", [100, 20, 200, 160]]], [["Below, the result over a hundred searches.", [60, 180, 300, 90]], "Notice the centre peak at a few hundred loops, then fade away."]],
   draw(t, st) { const ci = st.ci ?? 4, N = NS[ci], r = one(N), nums = {}, hi = {}; r.kids.forEach(k => { nums[k.move] = Math.round(100 * k.n / N) + '%'; });
     const ch = G.mostVisited(r); hi[ch] = ch === WIN ? C.yellow : C.pink; G.board(140, 8, 120, B0, { nums, numS: 17, hi, hiA: .7 });
     txt(`one search, ${N} loops: plays ${NAME[ch]}`, 200, 144, { s: 15, c: ch === WIN ? C.goldInk : C.roseInk });
     accChart({ y0: 262, y1: 186, x0: 70, x1: 350, a: 1, lines: [{ m: WIN, col: GOLD, ink: C.goldInk }, { m: CENTRE, col: TEAL, ink: TEALINK }], mark: ci, xl: ' ', yl: ' ' });
     txt('runs out of 100 that chose:', 70, 296, { s: 13, c: C.soft, al: 'left' });
     txt(`top right ${picks(ci, 2)}`, 70, 318, { s: 16, c: C.goldInk, al: 'left' }); txt(`centre ${picks(ci, 4)}`, 200, 318, { s: 16, c: TEALINK, al: 'left' }); txt(`other ${RUNS - picks(ci, 2) - picks(ci, 4)}`, 300, 318, { s: 16, c: C.soft, al: 'left' });
     caption(N <= 30 ? 'few loops: close to a guess' : N <= 300 ? 'shallow tree: it trusts the random playouts' : 'deep enough to see the cat’s defence'); } },

 { h: 'The weak spot: random playouts', p: () => [`Random playouts judge crudely: they rated the centre above the only winning move, costing MCTS about 1000 loops. Even in this tiny game, 3000 loops built ${DEEP().nodes3000.toLocaleString('en-US')} nodes, a sixth of the tree.`, 'In Go, random games say little about who leads, and the tree covers only a sliver. Programs added handmade rules for less random playouts. The next reel learns a replacement.'],
   say: [["Here's the weak spot. Random playouts judge crudely."], [["They rated the centre above the only winning move, and that cost MCTS about a thousand loops.", [30, 200, 340, 87]]], ["In Go, random games say very little about who's ahead. The next reel learns a better replacement."]],
   draw(t) { const f = FLAT(), P = PERF();
     G.board(40, 40, 120, B0, { nums: { 2: f2(f[2]), 4: f2(f[4]) }, numS: 17, hi: { 4: C.yellow }, hiA: .5, seed: 4100 }); txt('random playouts', 100, 24, { s: 15, c: C.soft });
     alpha(clamp(t - 1.5), () => { G.board(240, 40, 120, B0, { nums: { 2: 'win', 4: 'draw' }, numS: 17, numC: C.ink, hi: { 2: C.yellow }, hiA: .5, seed: 4200 }); txt('careful play', 300, 24, { s: 15, c: C.soft }); });
     if (t > 3.5) card([`3000 loops: ${DEEP().nodes3000.toLocaleString('en-US')} of ${TREE_SIZE().toLocaleString('en-US')} nodes`, { s: 'Go: the tree is astronomically bigger', c: C.roseInk }, { s: 'better judges of a position are needed', c: C.soft }], t - 3.5, { y: 200, gap: 1.2 }); } },

 { h: 'Where it is used', p: ['In 2006 Rémi Coulom named Monte Carlo tree search in his program Crazy Stone, and Kocsis and Szepesvári published **UCT**, the UCB rule at every node used here. MoGo soon used UCT for Go.', 'For a decade the strongest Go programs used MCTS. They reached strong amateur level, losing to top professionals on the full board. AlphaGo, in 2016, kept tree search, adding learned networks.'],
   say: [["In two thousand six, Rémi Coulom named Monte Carlo tree search, in his program Crazy Stone. Kocsis and Szepesvári published UCT, the rule used here."], [["For a decade, the strongest Go programs used MCTS.", [130, 4, 140, 140]], ["They reached strong amateur level. Then AlphaGo added learned networks.", [30, 150, 340, 110]]]],
   draw(t) { goBoard(140, 12, 120, clamp(t));
     card(['2006  Coulom: MCTS in Crazy Stone', '2006  Kocsis and Szepesvári: UCT', '2006 to 2015  MCTS Go: strong amateur level', '2016  AlphaGo: tree search plus networks'], t - 1, { y: 150, gap: 1.1, fs: 16 }); } },

 { h: 'Recap', p: ['**MCTS** grows a search tree from the current position, at most one node per loop: select by UCB, expand, simulate a random playout, back up the result.', 'Both sides choose by UCB inside the tree, so the search learns to expect good replies. It goes deep where moves look strong, and plays the most visited move. Its weak spot is the random playout.'],
   say: [["So, MCTS grows a search tree, one node per loop. Select, expand, simulate, and back up."], [["Both sides choose with UCB, so the search expects good replies. It goes deep where moves look strong.", [10, 20, 380, 270]], "Its weak spot is the random playout."]],
   draw(t) { drawTree(1000, { labels: false });
     card(['select by UCB, expand one node', 'simulate at random, back up the points', { s: 'each side uses its own points', c: C.soft }, { s: 'play the most visited move', c: C.soft }], t - .3, { y: 296, gap: 1, lh: 22 }); } },
];
STEPS.forEach(s => { const d = s.draw; s.draw = (t, st) => { if (!accDone()) pump(6); d(t, st); }; if (typeof s.p === 'function') { const f = s.p; Object.defineProperty(s, 'p', { get: f, enumerable: true }); } });
RLStory.register('mcts', { title: 'Monte Carlo Tree Search', track: 'Model-based', onCtx: c => { ctx = c; }, steps: STEPS,
  _facts: () => ({ tree: TREE_SIZE(), flat: FLAT(), perf: PERF(), run: RUN().root.kids.map(k => [k.move, k.n, +(k.w / k.n).toFixed(3), subSize(k), maxDepth(k)]), nodes1000: subSize(RUN().root), t8: T8().map(x => [x.move, x.result, x.path.length, x.newId]), acc: ACCF().map(a => [a.N, a.cnt]), qavg: QAVG(), deep: DEEP() }) });
})();
