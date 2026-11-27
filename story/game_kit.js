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
