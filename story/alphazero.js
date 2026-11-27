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
const GOLD = '#e9b84a', BROWN = '#a57f70', TEAL = '#6fb7b4', TEALINK = '#3f8582', BLUE = '#8fb3d9', BLUEINK = '#4f74a0';
const B0 = [2, 0, 0, 0, 0, 1, 0, 0, 0];
const NAME = ['top left', 'top middle', 'top right', 'middle left', 'centre', 'middle right', 'bottom left', 'bottom middle', 'bottom right'];
const f2 = v => v.toFixed(2), pc = v => Math.round(100 * v) + '%', memo = f => { let v; return () => v ?? (v = f()); };

// ───── the network: 19 inputs (9 "Pip here", 9 "cat here", 1 whose turn) → 32 tanh → 9 move scores + 1 value ─────
const SIZES = [19, 32, 10], NW = 19 * 32 + 32 + 32 * 10 + 10;
function inp(b) { const x = new Float64Array(19); for (let i = 0; i < 9; i++) { if (b[i] === 1) x[i] = 1; else if (b[i] === 2) x[9 + i] = 1; } x[18] = G.turn(b) === 1 ? 1 : -1; return x; }
// prior p over the empty squares (softmax), value v = Pip's expected points (sigmoid)
function netEval(net, b) { const a = G.forward(net, inp(b)), o = a[a.length - 1]; return { p: G.softmaxMasked(o.slice(0, 9), b.map(v => !v)), v: G.sigmoid(o[9]), acts: a }; }
const clone = net => ({ sizes: net.sizes, W: net.W.map(w => Float64Array.from(w)), b: net.b.map(w => Float64Array.from(w)) });

// ───── AlphaZero-style search: PUCT selection, network value at new nodes, no playouts ─────
// score = average (seen by the chooser; ½ if unvisited) + c × prior × √N ÷ (1 + n), c = 1.5
const CP = 1.5;
function search(net, b, S, o = {}) {
  const root = { b, n: 0, w: 0, kids: null, P: 1 }; let rootDone = false;
  function expand(nd) { const e = netEval(net, nd.b); nd.v = e.v; nd.kids = G.legal(nd.b).map(m => ({ b: G.play(nd.b, m), m, n: 0, w: 0, kids: null, P: e.p[m] }));
    if (!rootDone && o.noise) { const d = nd.kids.map(() => -Math.log(1 - o.R())), sd = d.reduce((a, x) => a + x, 0); nd.kids.forEach((k, i) => k.P = (1 - o.noise) * k.P + o.noise * d[i] / sd); }
    rootDone = true; return e.v; }
  for (let s = 0; s < S; s++) { let nd = root; const path = [root];
    while (nd.kids && nd.kids.length) { const pip = G.turn(nd.b) === 1; let best = -1e9, bk = null;
      for (const k of nd.kids) { const q = k.n ? (pip ? k.w / k.n : 1 - k.w / k.n) : .5, sc = q + CP * k.P * Math.sqrt(nd.n) / (1 + k.n); if (sc > best) { best = sc; bk = k; } }
      nd = bk; path.push(nd); }
    const term = G.points(nd.b), v = term !== null ? term : expand(nd);
    for (const p of path) { p.n++; p.w += v; } }
  return root;
}
const mostVisited = r => r.kids.reduce((a, k) => k.n > a.n ? k : a).m;
function topPrior(net, b) { const e = netEval(net, b); let bm = -1, bp = -1; for (const m of G.legal(b)) if (e.p[m] > bp) { bp = e.p[m]; bm = m; } return bm; }
const move = (net, b, S) => S ? mostVisited(search(net, b, S)) : topPrior(net, b);
// board symmetries (rotations, mirrors), used to show each training position in a random orientation
const SYM = (() => { const r = i => [6, 3, 0, 7, 4, 1, 8, 5, 2][i], f = i => [2, 1, 0, 5, 4, 3, 8, 7, 6][i], out = []; let id = [0, 1, 2, 3, 4, 5, 6, 7, 8]; for (let k = 0; k < 4; k++) { out.push(id.slice()); out.push(id.map(f)); id = id.map(r); } return out; })();
// answer key: positions where the choice matters (not every legal move is a best move)
const POS = memo(() => { const out = []; for (const [, b] of G.allPositions()) if (G.points(b) === null) { const bm = G.bestMoves(b); if (bm.length < G.legal(b).length) out.push({ b, bm }); } return out; });
function policyAcc(net) { let ok = 0; for (const x of POS()) if (x.bm.includes(topPrior(net, x.b))) ok++; return ok / POS().length; }

// ───── self-play training (seed 11), run as a generator so it can be spread over frames ─────
const CFG = { seed: 11, gens: 12, games: 50, sims: 40, sampleMoves: 4, noise: .25, keep: 4, epochs: 8, batch: 32, lr: .01 };
const RES = { gen: [], done: false, games: [] };
function* job() {
  const o = CFG, R = rng(o.seed), net = G.mlp(SIZES, o.seed), opt = G.adam(net, o.lr); let buf = [];
  const snap = g => { const e = netEval(net, B0); RES.gen.push({ g, net: clone(net), acc: policyAcc(net), p: Array.from(e.p), v: e.v, top: topPrior(net, B0), srch: mostVisited(search(net, B0, o.sims)) }); };
  snap(0); yield;
  for (let g = 0; g < o.gens; g++) {
    for (let gm = 0; gm < o.games; gm++) { let b = Array(9).fill(0); const rec = []; let k = 0;
      while (G.points(b) === null) { const r = search(net, b, o.sims, { noise: o.noise, R }), tot = r.kids.reduce((a, q) => a + q.n, 0), pi = new Float64Array(9); r.kids.forEach(q => pi[q.m] = q.n / tot);
        let m; if (k < o.sampleMoves) { let u = R() * tot; for (const q of r.kids) { u -= q.n; if (u < 0) { m = q.m; break; } } if (m == null) m = r.kids[r.kids.length - 1].m; } else m = mostVisited(r);
        rec.push({ b: b.slice(), pi, m, p: r.kids.map(q => [q.m, q.P]) }); b = G.play(b, m); k++; }
      const z = G.points(b); rec.forEach(x => buf.push({ b: x.b, pi: x.pi, z, g })); if (gm === 0) RES.games.push({ g: g + 1, rec, z, end: b }); if (gm % 5 === 4) yield; }
    buf = buf.filter(x => x.g > g - o.keep);
    for (let e = 0; e < o.epochs; e++) { const idx = buf.map((_, i) => i); for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
      for (let s = 0; s < idx.length; s += o.batch) { const gr = G.zeroGrad(net); let n = 0;
        for (const i of idx.slice(s, s + o.batch)) { const x = buf[i], sy = SYM[Math.floor(R() * 8)], b = Array(9), pi = new Float64Array(9); for (let c = 0; c < 9; c++) { b[sy[c]] = x.b[c]; pi[sy[c]] = x.pi[c]; }
          const ev = netEval(net, b), go = new Float64Array(10); for (let c = 0; c < 9; c++) go[c] = ev.p[c] - pi[c]; go[9] = ev.v - x.z; G.backward(net, ev.acts, go, gr); n++; }
        G.adamStep(net, opt, gr, 1 / n); }
      yield; }
    snap(g + 1); yield; }
  RES.samples = buf.length; RES.trained = true;
  // after training: search vs instinct on 300 positions, then games against a perfect player
  const fin = RES.gen[o.gens].net, P = POS();
  { const R2 = rng(9), T = { n: 300, S: [0, 10, 40, 200], ok: [0, 0, 0, 0] }, T0 = [0, 0], n0 = RES.gen[0].net;
    for (let i = 0; i < T.n; i++) { const x = P[Math.floor(R2() * P.length)]; T.S.forEach((S, j) => { if (x.bm.includes(move(fin, x.b, S))) T.ok[j]++; }); if (x.bm.includes(move(n0, x.b, 0))) T0[0]++; if (x.bm.includes(move(n0, x.b, 40))) T0[1]++; if (i % 10 === 9) yield; }
    RES.teach = T; RES.teach0 = T0; }
  RES.vs0 = yield* vsPerfect(RES.gen[0].net, 40, 30, 5);
  RES.vs = []; for (const S of [0, 40, 200]) RES.vs.push({ S, ...(yield* vsPerfect(fin, S, 30, 5)) });
  RES.done = true;
}
function* vsPerfect(net, S, n, seed) { const R = rng(seed), res = { pip: [0, 0, 0], cat: [0, 0, 0] };
  for (const side of [1, 2]) for (let g = 0; g < n; g++) { let b = Array(9).fill(0);
    while (G.points(b) === null) { let m; if (G.turn(b) === side) m = move(net, b, S); else { const bm = G.bestMoves(b); m = bm[Math.floor(R() * bm.length)]; } b = G.play(b, m); }
    const z = G.points(b), sc = side === 1 ? z : 1 - z; (side === 1 ? res.pip : res.cat)[sc === 1 ? 0 : sc === .5 ? 1 : 2]++; yield; }
  return res; }
let JOB = null; const accDone = () => RES.done;
function pump(ms) { if (RES.done) return; JOB = JOB || job(); const t0 = Date.now(); while (Date.now() - t0 < ms) { if (JOB.next().done) return; } }
// run the job only as far as needed
function need(cond) { if (!cond()) { JOB = JOB || job(); while (!cond()) if (JOB.next().done) break; } return RES; }
const GEN = g => need(() => RES.gen[g]).gen[g], FINAL = () => GEN(CFG.gens), GAME1 = () => need(() => RES.games[0]).games[0];
const TEACH = () => need(() => RES.teach).teach, TEACH0 = () => need(() => RES.teach0).teach0, VS = () => need(() => RES.done).vs, VS0 = () => need(() => RES.vs0).vs0;
// positions that matter met by the trained network alone in its 60 games against the perfect player, and its prior's errors there
const VSEEN = memo(() => { const fin = FINAL().net, R = rng(5); let seen = 0, wrong = 0; for (const side of [1, 2]) for (let g = 0; g < 30; g++) { let b = Array(9).fill(0); while (G.points(b) === null) { let m; const bm = G.bestMoves(b); if (G.turn(b) === side) { m = move(fin, b, 0); if (bm.length < G.legal(b).length) { seen++; if (!bm.includes(m)) wrong++; } } else m = bm[Math.floor(R() * bm.length)]; b = G.play(b, m); } } return { seen, wrong }; });
const ALLGEN = () => need(() => RES.trained).gen;
const STORY = memo(() => { const net = FINAL().net, out = {}; [10, 25, 50].forEach(S => { const r = search(net, B0, S); out[S] = { kids: r.kids.map(k => ({ m: k.m, n: k.n, q: k.n ? k.w / k.n : null, P: k.P })), pick: mostVisited(r) }; const k8 = r.kids.find(k => k.m === 8); if (S === 50 && k8.kids) out.br = k8.kids.map(k => ({ m: k.m, P: k.P, n: k.n })); }); return out; });

// ───── drawing ─────
function legend(y, a = 1) { alpha(a, () => { G.cheeseMark(118, y, 18); txt('Pip', 132, y + 1, { s: 15, al: 'left' }); G.pawMark(232, y, 18); txt('cat', 246, y + 1, { s: 15, al: 'left' }); }); }
// the network picture: board in, hidden units, two heads out. o = { b, p, v, a, glow }
function netPic(o = {}) { const a = o.a ?? 1, b = o.b || B0, bx = 18, by = 70, bs = 84, hx = 190, outx = 262;
  alpha(a, () => {
    G.board(bx, by, bs, b, { seed: 5000, mini: false });
    txt('board in', bx + bs / 2, by - 14, { s: 14, c: C.soft });
    const hy = i => 34 + i * 6.4; for (let i = 0; i < 32; i++) { rline(bx + bs + 6, by + bs / 2, hx - 4, hy(i), 5100 + i, { c: '#e3d9ca', w: .5, once: true }); }
    for (let i = 0; i < 32; i++) { rline(hx + 4, hy(i), outx - 4, 60, 5200 + i, { c: '#efe6d8', w: .5, once: true }); rline(hx + 4, hy(i), outx - 4, 196, 5300 + i, { c: '#efe6d8', w: .5, once: true }); dot(hx, hy(i), 2.4, C.soft); }
    txt('32 hidden units', hx, 254, { s: 13, c: C.soft });
    // value head
    const va = o.heads ?? 1; alpha(va, () => { rrect(outx, 36, 120, 50, 5400, { c: BLUEINK, fill: '#eef3f9' }); txt('value', outx + 60, 50, { s: 15, c: BLUEINK }); txt(o.v != null ? f2(o.v) : '?', outx + 60, 72, { s: 19, c: BLUEINK });
      txt('Pip’s expected points', outx + 60, 98, { s: 12, c: C.soft });
      // policy head
      txt('prior', outx + 60, 126, { s: 15, c: C.goldInk }); const nums = {}; if (o.p) for (let i = 0; i < 9; i++) if (!b[i]) nums[i] = Math.round(o.p[i] * 100);
      G.board(outx + 15, 138, 90, b, { seed: 5500, mini: false, nums, numS: 15, hi: o.hiMove != null ? { [o.hiMove]: C.yellow } : null, hiA: .7 });
      txt('% per empty square', outx + 60, 240, { s: 12, c: C.soft }); });
  }); }
function priorBoard(x, y, s, b, p, o = {}) { const nums = {}; for (let i = 0; i < 9; i++) if (!b[i]) nums[i] = o.fmt ? o.fmt(p[i], i) : Math.round(p[i] * 100) + '%'; G.board(x, y, s, b, Object.assign({ nums, numS: o.numS || 15 }, o)); }
// learning curve: x = generation 0..12, y = % of positions where the top prior is a best move
function curve(o = {}) { const x0 = o.x0 ?? 60, x1 = o.x1 ?? 360, y0 = o.y0 ?? 250, y1 = o.y1 ?? 80, X = g => lerp(x0, x1, g / CFG.gens), Y = v => lerp(y0, y1, v), a = o.a ?? 1, gs = ALLGEN();
  alpha(a, () => { rline(x0, y0, x1, y0, 5600, { c: C.ink }); rline(x0, y0, x0, y1 - 6, 5601, { c: C.ink });
    [0, .5, 1].forEach(v => { txt(pc(v), x0 - 8, Y(v), { s: 12, c: C.soft, al: 'right' }); if (v) rline(x0, Y(v), x1, Y(v), 5610 + v * 10, { c: '#e6dccd', w: .8, once: true }); });
    [0, 4, 8, 12].forEach(g => txt(String(g), X(g), y0 + 13, { s: 12, c: C.soft })); txt('generation (50 self-play games each)', (x0 + x1) / 2, y0 + 28, { s: 13, c: C.soft });
    if (o.yl !== false) txt('top move of the prior is a best move', x0 - 30, y1 - 20, { s: 13, c: C.soft, al: 'left' }); });
  const n = clamp(o.n ?? 1) * CFG.gens;
  for (let g = 0; g < CFG.gens && g < n; g++) { const u = clamp(n - g); rline(X(g), Y(gs[g].acc), lerp(X(g), X(g + 1), u), lerp(Y(gs[g].acc), Y(gs[g + 1].acc), u), 5700 + g, { c: GOLD, w: 2.2 }); }
  for (let g = 0; g <= Math.floor(n + .01); g++) dot(X(g), Y(gs[g].acc), 3.2, GOLD);
  if (o.lab) o.lab.forEach(g => { if (g <= n + .01) txt(pc(gs[g].acc), X(g) + (g ? 0 : 16), Y(gs[g].acc) - 12, { s: 13, c: C.goldInk }); });
  if (o.mark != null) { const g = o.mark; ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.soft; ctx.beginPath(); ctx.moveTo(X(g), y0); ctx.lineTo(X(g), y1); ctx.stroke(); ctx.restore(); dot(X(g), Y(gs[g].acc), 5, GOLD); }
}
// self-play game record: boards in a row with the search's visit share for the chosen move
function gameStrip(gm, t, o = {}) { const n = gm.rec.length, gap = 8, s = Math.min(58, (350 - gap * (n - 1)) / n), x0 = 200 - (n * (s + gap) - gap) / 2, y = o.y ?? 120;
  gm.rec.forEach((r, i) => { const a = clamp((t - i * .8) * 2); if (a <= 0) return; const b = G.play(r.b, r.m), x = x0 + i * (s + gap);
    G.board(x, y, s, b, { a, seed: 5800 + i, hi: { [r.m]: C.yellow }, hiA: .8 }); txt(pc(r.pi[r.m]), x + s / 2, y + s + 12, { s: 12, c: TEALINK, a }); });
  const te = n * .8 + .5; if (t > te) { const z = gm.z; txt(z === 1 ? 'Pip won: z = 1' : z === .5 ? 'draw: z = ½' : 'cat won: z = 0', 200, y + s + 36, { s: 16, c: z === 1 ? C.goldInk : z === .5 ? C.soft : C.roseInk, a: clamp(t - te) }); }
}

// ───── steps ─────
const STEPS = [
 { h: 'A better judge', p: ['Back to Pip’s puzzle from the tree search reel: cat in the top left corner, Pip on the middle right, Pip to move. Only **top right** wins.', 'Random playouts rated the centre higher, 0.69 against 0.65, and the search needed about 1000 loops to see past them. What if Pip could **learn** to judge a position, instead of playing it out at random?'],
   draw(t) { G.board(125, 34, 150, B0, { nums: t > 2 ? { 2: '0.65', 4: '0.69' } : null, numS: 20, hi: t > 4.5 ? { 2: C.yellow } : null, hiA: .6 }); legend(214);
     if (t > 2) caption('random-playout averages from the tree search reel', clamp(t - 2));
     if (t > 6) card(['random playouts: crude judges', { s: 'idea: a learned judge of positions', c: C.goldInk }], t - 6, { y: 300 }); } },

 { h: 'One network, two answers', p: () => [`**AlphaZero** uses one small neural network. It reads the board, which squares hold cheese, which hold paws and whose turn it is, and gives two answers.`, `The **value**: Pip’s expected points from this position. The **prior**: for each empty square, how promising that move looks, as shares adding up to 100 percent. Ours has 32 hidden units and ${NW} knobs; before training they are random.`],
   shot: [4, 11],
   draw(t) { const e = GEN(0); netPic({ a: clamp(t * 2), heads: clamp(t - 2.5), v: t > 6 ? e.v : null, p: t > 6 ? e.p : null });
     if (t > 6) caption('an untrained network: its answers are noise', clamp(t - 6)); } },

 { h: 'A judge inside the search', p: ['The search loop stays: select, expand, back up. But there is no random playout. When a loop reaches a new node, the network looks at it once: its **value** is backed up the path, and its **priors** are stored for the node’s moves.', 'If the loop reaches a finished game, the real result is backed up instead: the rules say who won.'],
   shot: [4, 11],
   draw(t) { const P = [[200, 34], [120, 100], [200, 100], [280, 100], [160, 170]], E = [[0, 1], [0, 2], [0, 3], [2, 4]], ph = t % 6;
     E.forEach(([a, b], i) => rline(P[a][0], P[a][1] + 10, P[b][0], P[b][1] - 10, 5900 + i, { c: (a === 0 && b === 2) || (a === 2 && b === 4) ? TEALINK : C.line, w: 2 }));
     P.forEach(([x, y], i) => L.rcircle(x, y, 10, 5910 + i, { fill: i === 4 ? '#cfe7e5' : i ? C.card : C.yellow }));
     // the network box next to the new node
     rrect(232, 150, 120, 44, 5920, { c: BLUEINK, fill: '#eef3f9' }); txt('network', 292, 172, { s: 16, c: BLUEINK }); arrow(172, 170, 228, 172, 5921, { c: BLUEINK, w: 1.4 });
     if (ph > 1.5) { txt('value 0.72', 292, 214, { s: 15, c: BLUEINK, a: clamp(ph - 1.5) }); txt('priors for its moves', 292, 234, { s: 13, c: C.goldInk, a: clamp(ph - 2) }); }
     if (ph > 3) [[160, 170], [200, 100], [200, 34]].forEach(([x, y], j) => { const u = clamp((ph - 3 - j * .5) * 2); if (u > 0) txt('+0.72', x - 30, y - 6, { s: 14, c: BLUEINK, a: u }); });
     caption('made-up numbers, to show the flow', .9);
     card(['new node: ask the network once', { s: 'back up its value, keep its priors', c: C.soft }, { s: 'finished game: back up the real result', c: C.soft }], t - 5, { y: 296, gap: 1.1 }); } },

 { h: 'The prior steers the bonus', p: ['AlphaZero swaps in a bonus using the prior, shown on screen. The average still uses the chooser’s points, and c = 1.5 weighs the bonus.', 'A move the network likes gets tried early; one it dislikes waits, but its bonus grows while ignored. Untried moves count as a draw, ½. Unlike UCB, this rule has no proof; it was chosen because it works.'],
   draw(t) { txt('score = average + c × prior × √N ÷ (1 + n)', 200, 46, { s: 20, a: clamp(t * 2) });
     [['average', 'Pip’s or the cat’s points so far (½ if untried)', C.ink, 1.5], ['prior', 'the network’s share for this move', C.goldInk, 3], ['N, n', 'visits of the node, of this move', C.soft, 4.5], ['c', '1.5 here: weight of the bonus', C.soft, 6]].forEach(([s, d, col, a0], i) => { const a = clamp(t - a0); txt(s + ':', 90, 92 + i * 28, { s: 17, c: col, a, al: 'right' }); txt(d, 100, 92 + i * 28, { s: 15, a, al: 'left' }); });
     if (t > 8) card(['liked moves first, others wait their turn', { s: 'the bonus still shrinks as a move is visited', c: C.soft }], t - 8, { y: 300 }); } },

 { h: 'Where the lessons come from', p: ['No human games, no hints. The network plays **against itself**, network plus search choosing every move for both sides: **self-play**.', `Games start from the empty board. To make games differ, the first moves are drawn at random, weighted by visits, and noise is mixed into the top priors, where the move is chosen, so disliked moves still get tried. Later moves take the most visited.`],
   draw(t) { L.mouse(120, 120, 1, 1.6, t); L.cat(280, 120, t, 1.4); arrow(160, 112, 240, 112, 6000, { c: C.soft }); arrow(240, 132, 160, 132, 6001, { c: C.soft });
     txt('same network + search on both sides', 200, 190, { s: 16, c: C.ink, a: clamp(t - 1) });
     if (t > 4) card([`${CFG.sims} loops of search per move`, `first ${CFG.sampleMoves} moves: random, weighted by visits`, { s: 'plus a little noise in the top priors', c: C.soft }], t - 4, { y: 290, gap: 1.1 }); } },

 { h: 'What one game teaches', p: () => { const gm = GAME1(); return [`Game 1, untrained network. Under each board: the visit share of the move played. The cat’s ${pc(gm.rec[3].pi[gm.rec[3].m])} move ignores Pip’s diagonal: a random opening draw. It ends in ${gm.z === .5 ? 'a draw' : gm.z === 1 ? 'a Pip win' : 'a cat win'}.`, 'Each position becomes a lesson: the visit shares, **π** (pi), and the result, **z**. Training shrinks the value’s miss on z and the **cross-entropy**, a standard mismatch score for shares, between prior and π.']; },
   shot: [5, 12],
   draw(t) { const gm = GAME1(); txt('generation 1, game 1', 200, 22, { s: 15, c: C.soft }); gameStrip(gm, t, { y: 38 });
     if (t > 5) { const r = gm.rec[0], nums = {}; for (let i = 0; i < 9; i++) if (!r.b[i]) nums[i] = pc(r.pi[i]); alpha(clamp(t - 5), () => { G.board(60, 170, 96, r.b, { nums, numS: 14, numC: TEALINK, seed: 5890 }); txt('first position: π, the share of', 176, 196, { s: 15, al: 'left', c: TEALINK }); txt('the search’s visits for each move', 176, 218, { s: 15, al: 'left', c: TEALINK }); }); }
     if (t > 8.5) card(['each position:  π = search’s visit shares', 'z = how the game ended, Pip’s points', { s: 'value learns z, prior learns π', c: C.goldInk }], t - 8.5, { y: 290, gap: 1.1 }); } },

 { h: 'Search as a teacher', p: ['Why copy the search? Because search plus network chooses better than the network alone: it looks ahead, and checks the network’s guesses against positions further down. Training the prior toward π makes the network’s **instinct** more like its own careful thinking.', 'The value head learns what a position is worth from how whole games turned out, z: a Monte Carlo target, not a TD one.'],
   draw(t) { L.mouse(110, 120, 1, 1.4, t); txt('instinct', 110, 160, { s: 15, c: C.goldInk }); txt('prior', 110, 178, { s: 13, c: C.soft });
     rrect(230, 80, 130, 90, 6100, { c: TEALINK, fill: '#eaf4f3' }); txt('search', 295, 108, { s: 16, c: TEALINK }); txt('looks ahead', 295, 132, { s: 13, c: C.soft }); txt('visit shares π', 295, 152, { s: 13, c: TEALINK });
     arrow(225, 140, 150, 140, 6101, { c: TEALINK, w: 1.6 }); txt('learn from', 188, 128, { s: 13, c: TEALINK, a: clamp(t - 1) });
     if (t > 4) card(['prior  ←  π  (the search’s choice)', 'value  ←  z  (how the game ended)', { s: 'better network → better search → better lessons', c: C.soft }], t - 4, { y: 290, gap: 1.1 }); } },

 { h: 'The loop', p: () => [`One **generation**: play ${CFG.games} self-play games, then train on the lessons of the last ${CFG.keep} generations, shown ${CFG.epochs} times each, each randomly rotated or mirrored. Older games, played by a clumsier network, are dropped.`, `The new network plays the next ${CFG.games} games. We run ${CFG.gens} generations, ${CFG.gens * CFG.games} games in all. Each nudge changes the knobs a little toward the lessons, like earlier reels’ gradient steps.`],
   draw(t) { const P = [[200, 60, 'self-play games'], [310, 170, 'lessons: π and z'], [200, 260, 'train the network'], [90, 170, 'better network']];
     P.forEach(([x, y, s], i) => { const a = clamp((t - i * .9) * 2); alpha(a, () => { rrect(x - 66, y - 18, 132, 36, 6200 + i, { fill: C.card, c: i === 2 ? BLUEINK : C.line }); txt(s, x, y, { s: 15 }); }); });
     [[0, 1], [1, 2], [2, 3], [3, 0]].forEach(([a, b], i) => { if (t > (i + 1) * .9) { const [x1, y1] = P[a], [x2, y2] = P[b]; arrow(lerp(x1, x2, .3), lerp(y1, y2, .3), lerp(x1, x2, .7), lerp(y1, y2, .7), 6210 + i, { c: C.soft }); } });
     const g = Math.min(CFG.gens, 1 + Math.floor(Math.max(0, t - 4) / .7)); if (t > 4) txt(`generation ${g} of ${CFG.gens}`, 200, 310, { s: 18, c: C.goldInk }); } },

 { h: 'Watching it learn', p: () => [`Our answer key lists the best moves in all ${POS().length.toLocaleString('en-US')} positions where the choice matters. The chart: how often the prior’s favourite is a best move.`, `Untrained: ${pc(GEN(0).acc)}. After generation 4: ${pc(GEN(4).acc)}. It peaks at ${pc(GEN(8).acc)} after generation 8, ending at ${pc(FINAL().acc)}. Nobody showed it a good move. Truth enters through the rules: finished games report real results, in search and as z.`],
   shot: [4, 12],
   draw(t) { curve({ a: clamp(t * 2), n: t > 4 ? (t - 4) / 5 : 0, lab: [0, 4, CFG.gens] }); if (t < 4) caption('first the axes: generations across, accuracy up', clamp(t - 1)); } },

 { h: 'Its instinct on our puzzle', p: () => { const F = FINAL(); return [`After training, the network looks at Pip’s puzzle. Its prior gives top right ${pc(F.p[2])}, the centre ${pc(F.p[4])} and bottom right ${pc(F.p[8])}. Its value: ${f2(F.v)} of a point for Pip; the truth is 1, a sure win.`, 'So the instinct already leans the right way, but the value is unsure. Next, the search checks it.']; },
   draw(t) { const F = FINAL(); netPic({ v: F.v, p: F.p, hiMove: t > 5 ? 2 : null, a: 1 }); if (t > 6) caption('trained network, no search yet', clamp(t - 6)); } },

 { h: 'The search checks the instinct', p: () => { const S = STORY(), k = m => S[50].kids.find(x => x.m === m), br = S.br.slice().sort((a, b) => b.P - a.P)[0]; return [`With 50 loops, the first only asks the network about the current board; the board shows where the rest went. Top right gets the most, and its average climbs to ${f2(k(2).q)}. Random playouts needed about 1000 loops.`, `Why so few? The value head judges each new position directly. After bottom right, the prior gives ${pc(br.P)} to the cat’s only non-losing reply, the top right corner.`]; },
   shot: [4, 12],
   draw(t) { const S = STORY(), k = S[50].kids, nums = {}; k.forEach(x => { nums[x.m] = String(x.n); }); const show = t > 1.5;
     G.board(125, 30, 150, B0, { nums: show ? nums : null, numS: 20, numC: TEALINK, hi: t > 4 ? { 2: C.yellow } : null, hiA: .6 }); txt('loops through each move (50 loops, the first at the top)', 200, 196, { s: 14, c: C.soft, a: clamp(t - 1.5) });
     if (t > 5) card([`top right: average ${f2(k.find(x => x.m === 2).q)}`, { s: 'random playouts needed about 1000 loops', c: C.soft }], t - 5, { y: 300 }); } },

 { h: 'The teacher stays ahead', p: () => { const T = TEACH(); return [`Take ${T.n} random positions where the choice matters; the bars count best moves found by the trained network, alone and with search.`, `Untrained, the gap was wider: ${TEACH0()[0]} alone, ${TEACH0()[1]} with 40 loops. Visit shares are the better target, and that gap keeps the loop improving. Caveat: only 20 of the 300 have six or more empty squares, so 40 loops often reach the end.`]; },
   draw(t) { const T = TEACH(), x0 = 80, w = 60; txt(`best move found, out of ${T.n} positions`, 200, 40, { s: 15, c: C.soft });
     T.S.forEach((S, i) => { const x = x0 + i * 80, h = T.ok[i] / T.n * 180, u = clamp((t - .8 - i * .8) * 2); box(x - w / 2, 250 - h * u, w, h * u, i ? TEAL : GOLD, .85); rrect(x - w / 2, 70, w, 180, 6300 + i, { c: C.line, w: 1 });
       txt(i ? `${S} loops` : 'network', x, 266, { s: 14, c: i ? TEALINK : C.goldInk }); if (!i) txt('alone', x, 282, { s: 13, c: C.soft }); if (u >= 1) txt(String(T.ok[i]), x, 264 - h, { s: 16 }); });
     if (t > 5) card(['search + network beats the network alone', { s: 'so the prior learns from the search', c: C.soft }], t - 5, { y: 310 }); } },

 { h: 'Against perfect play', p: () => { const V = VS(), v0 = VS0(), pl = (k, w) => `${k} ${w}${k === 1 ? '' : (w === 'loss' ? 'es' : 's')}`, s = r => `${pl(r[0], 'win')}, ${pl(r[1], 'draw')}, ${pl(r[2], 'loss')}`; return [`It plays a perfect player, 30 games on each side. Perfect play draws. Untrained, even with 40 loops, it lost ${v0.cat[2]} of 30 as the cat and ${v0.pip[2]} as Pip.`, `Its search did well on random positions, but here one wrong early move loses. Trained, losses almost vanish. The card’s error rate is overall; in these games it erred in ${VSEEN().wrong} of ${VSEEN().seen} such positions.`]; },
   draw(t) { const V = VS(), v0 = VS0(); const rows = [['untrained + 40 loops', v0], ['trained, network alone', V[0]], ['trained + 40 loops', V[1]], [`trained + ${V[2].S} loops`, V[2]]];
     txt('60 games against a perfect player', 200, 40, { s: 15, c: C.soft }); txt('losses', 300, 70, { s: 14, c: C.roseInk }); txt('draws', 360, 70, { s: 14, c: C.soft });
     rows.forEach(([lab, r], i) => { const a = clamp((t - .8 - i * 1.2) * 2), y = 100 + i * 34; alpha(a, () => { txt(lab, 24, y, { s: 16, al: 'left' }); txt(String(r.pip[2] + r.cat[2]), 300, y, { s: 17, c: C.roseInk }); txt(String(r.pip[1] + r.cat[1]), 360, y, { s: 17, c: C.ink }); }); });
     if (t > 6) card(['a draw is the best possible result', { s: `even so, its prior is wrong in ${pc(1 - FINAL().acc)} of positions that matter`, c: C.soft }], t - 6, { y: 300 }); } },

 { h: 'Your turn', ctl: { label: 'generation', values: Array.from({ length: CFG.gens + 1 }, (_, i) => i), def: CFG.gens, fmt: v => v }, p: () => ['Drag through the **generations**. The board shows the network’s prior on Pip’s puzzle; the chart marks how often its favourite is a best move overall.', `At generation 0 the prior is close to even. Watch when top right takes the lead, and when the search with ${CFG.sims} loops starts to pick it.`],
   shot: [3, 8],
   draw(t, st) { const g = st.ci ?? CFG.gens, e = GEN(g); priorBoard(30, 14, 120, B0, e.p, { hi: { [e.top]: e.top === 2 ? C.yellow : C.pink }, hiA: .7 });
     txt(`value ${f2(e.v)}`, 236, 34, { s: 17, c: BLUEINK, al: 'left' }); txt('prior’s favourite:', 172, 68, { s: 14, c: C.soft, al: 'left' }); txt(NAME[e.top], 172, 88, { s: 16, c: e.top === 2 ? C.goldInk : C.roseInk, al: 'left' });
     txt(`search, ${CFG.sims} loops:`, 172, 114, { s: 14, c: C.soft, al: 'left' }); txt(NAME[e.srch], 172, 134, { s: 16, c: e.srch === 2 ? C.goldInk : C.roseInk, al: 'left' });
     curve({ y0: 330, y1: 190, x0: 60, x1: 360, mark: g, yl: false });
     caption(`generation ${g}: favourite is a best move in ${pc(e.acc)} of positions`); } },

 { h: 'The catch', p: () => [`Even this toy took ${CFG.gens * CFG.games} games with ${CFG.sims} loops per move. Go needs vastly more: AlphaZero played tens of millions of self-play games each for chess, shogi and Go, on thousands of special chips.`, 'Every loop, the search **plays moves by the rules** to reach new positions. AlphaZero needs the rules. Atari games and the real world have no rulebook. That is the next reel.'],
   draw(t) { G.board(130, 30, 140, G.play(B0, 2), { seed: 6400, hi: { 2: C.yellow }, hiA: .6 }); txt('the rulebook says what a move does', 200, 194, { s: 15, c: C.ink, a: clamp(t - 1) });
     if (t > 3) card([`toy: ${CFG.gens * CFG.games} games,  ${CFG.sims} loops per move`, { s: 'Go: tens of millions of games', c: C.roseInk }, { s: 'and the search needs the rules', c: C.roseInk }], t - 3, { y: 290, gap: 1.2 }); } },

 { h: 'AlphaGo to AlphaZero', p: ['In 2016 DeepMind’s **AlphaGo** beat Lee Sedol, a top Go player, 4 games to 1. It learned from human games first and used playouts. In 2017 **AlphaGo Zero** dropped both, learned by self-play alone, and beat that AlphaGo 100 games to 0.', 'Then **AlphaZero**, preprint 2017, journal paper 2018: given only the rules, the same method learned chess, shogi and Go, and beat leading programs.'],
   draw(t) { card(['2016  AlphaGo beats Lee Sedol, 4 to 1', { s: '         human games + playouts + networks', c: C.soft }, '2017  AlphaGo Zero: self-play only', { s: '         one network, no playouts', c: C.soft }, '2018  AlphaZero: chess, shogi, Go', { s: '         from the rules alone', c: C.soft }], t - .5, { y: 60, gap: 1, lh: 26 }); } },

 { h: 'Recap', p: ['**AlphaZero** replaces random playouts with a network that gives a **value** and a **prior** for each position. The search uses the value at new nodes, and the prior to decide which moves to try.', 'It learns by self-play: the value learns how games ended, the prior learns the search’s visit shares. Better network, better search, better lessons. It still needs the rules.'],
   draw(t) { const F = FINAL(); priorBoard(130, 20, 140, B0, F.p, { hi: { 2: C.yellow }, hiA: .6 });
     card(['value: Pip’s expected points, no playouts', 'prior: which moves to search first', { s: 'self-play: prior ← π,  value ← z', c: C.soft }, { s: 'still needs the rules to search', c: C.soft }], t - .3, { y: 196, gap: 1, lh: 24 }); } },
];
STEPS.forEach(s => { const d = s.draw; s.draw = (t, st) => { if (!accDone()) pump(8); d(t, st); }; if (typeof s.p === 'function') { const f = s.p; Object.defineProperty(s, 'p', { get: f, enumerable: true }); } });
RLStory.register('alphazero', { title: 'AlphaZero', track: 'Model-based', onCtx: c => { ctx = c; }, steps: STEPS,
  _facts: () => ({ nw: NW, pos: POS().length, gens: need(() => RES.done).gen.map(g => [g.g, +g.acc.toFixed(3), +g.v.toFixed(3), g.top, g.srch, g.p.map(x => +x.toFixed(2))]), game1: GAME1().rec.map(r => [r.m, +r.pi[r.m].toFixed(2)]), z1: GAME1().z, story: STORY(), teach: TEACH(), teach0: TEACH0(), vs: VS(), vs0: VS0(), samples: RES.samples }) });
})();
