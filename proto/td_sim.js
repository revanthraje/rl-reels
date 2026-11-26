// Deterministic TD(0) story simulation: Pip the mouse in a 5x4 maze.
var TDSim = (function () {
  const COLS = 5, ROWS = 4, START = [0, 3], CHEESE = [4, 0], CAT = [3, 1], ROCKS = [[2, 2], [1, 1]];
  const MOVES = [[0, -1], [1, 0], [0, 1], [-1, 0]];
  const key = (c, r) => r * COLS + c;
  const isRock = (c, r) => ROCKS.some(([x, y]) => x === c && y === r);
  const term = (c, r) => (c === CHEESE[0] && r === CHEESE[1]) ? 1 : (c === CAT[0] && r === CAT[1]) ? -1 : 0;
  function rng(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function step(c, r, m) { const nc = c + MOVES[m][0], nr = r + MOVES[m][1]; if (nc < 0 || nr < 0 || nc >= COLS || nr >= ROWS || isRock(nc, nr)) return [c, r]; return [nc, nr]; }
  // Run episodes; record every transition with the update applied.
  function run(opt) {
    const { gamma = .9, alpha = .2, episodes = 200, seed = 7, eps: epsx = .2 } = opt || {};
    const R = rng(seed), V = new Float64Array(COLS * ROWS), eps = [];
    for (let e = 0; e < episodes; e++) {
      let [c, r] = START; const tr = [];
      for (let k = 0; k < 400; k++) {
        const m = R() < epsx ? Math.floor(R() * 4) : pick(V, c, r, gamma, R), [nc, nr] = step(c, r, m), rew = term(nc, nr), done = rew !== 0;
        const old = V[key(c, r)], next = done ? 0 : V[key(nc, nr)], target = rew + gamma * next, nw = old + alpha * (target - old);
        V[key(c, r)] = nw;
        tr.push({ c, r, nc, nr, rew, old, next, target, nw });
        c = nc; r = nr; if (done) break;
      }
      eps.push({ tr, end: tr[tr.length - 1].rew, V: Array.from(V) });
    }
    return { eps, gamma, alpha };
  }
  // Pip's habit: peek at each neighbour, go to the best-looking one (ties broken at random).
  function pick(V, c, r, gamma, R) {
    let best = -1e9, ms = [];
    for (let m = 0; m < 4; m++) { const [nc, nr] = step(c, r, m); if (nc === c && nr === r) continue; const rew = term(nc, nr), q = rew + (rew ? 0 : gamma * V[key(nc, nr)]);
      if (q > best + 1e-9) { best = q; ms = [m]; } else if (q > best - 1e-9) ms.push(m); }
    return ms[Math.floor(R() * ms.length)];
  }
  // Greedy move using the known maze: maximise reward + gamma * V(next).
  function greedy(V, c, r, gamma) {
    let best = -1e9, bm = 0;
    for (let m = 0; m < 4; m++) { const [nc, nr] = step(c, r, m); if (nc === c && nr === r) continue; const rew = term(nc, nr), q = rew + (rew ? 0 : gamma * V[key(nc, nr)]); if (q > best + 1e-12) { best = q; bm = m; } }
    return bm;
  }
  function greedyPath(V, gamma, from) {
    let [c, r] = from || START; const path = [[c, r]];
    for (let k = 0; k < 30 && !term(c, r); k++) { [c, r] = step(c, r, greedy(V, c, r, gamma)); path.push([c, r]); }
    return path;
  }
  return { COLS, ROWS, START, CHEESE, CAT, ROCKS, MOVES, key, isRock, term, step, run, greedy, greedyPath };
})();
if (typeof window !== 'undefined') window.TDSim = TDSim;
