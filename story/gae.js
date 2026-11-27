// n-step returns & GAE as a story: how should Pip grade a move? One real step + the critic's guess (TD error: steady but
// biased when the critic is off) or the whole real return (unbiased but noisy)? n-step grades sit in between; GAE mixes all
// of them with a fading weight set by λ. Shown on one move of the actorcritic reel's run (frozen after walk 50 and 80), then
// in training. Every number comes from the seeded runs below.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
// ===== pg kit (copied from pg.js) =====
const softmax = th => { const m = Math.max(...th), e = th.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
function fans(M, TH, o = {}) {
  for (let r = 0; r < M.rows; r++) for (let c = 0; c < M.cols; c++) {
    if (M.term(c, r) || M.isRock(c, r)) continue; const k = M.key(c, r), p = softmax(Array.from(TH.slice(k * 4, k * 4 + 4))), best = p.indexOf(Math.max(...p)), x = M.cx(c), y = M.cy(r) + 4;
    p.forEach((q, m) => { const [dx, dy] = M.MOVES[m], len = 3 + q * (M.cs / 2 - 7), big = m === best && q > .3;
      alpha(o.a ?? 1, () => arrow(x + dx * 4, y + dy * 4, x + dx * (4 + len), y + dy * (4 + len), 600 + k * 4 + m, { c: big ? C.goldInk : C.soft, w: big ? 2 : 1.2, head: 4 + q * 4 })); });
  }
}
// ===== end pg kit =====
const G = L.gridWorld({ cols: 5, rows: 4, cheese: [4, 0], cat: [3, 1], rocks: [[2, 2], [1, 1]] });
const { cx, cy, key, maze, trail, pip } = G, OX = G.ox, OY = G.oy, CS = G.cs;
const GAM = 0.9, CAP = 100, SK = key(0, 3), KUP = key(0, 2), BEST = Math.pow(GAM, 6), NW = 300, NRUN = 100, AT = 1, AV = 0.2, NS = 2000, ND = 40;
const LAMS = [0, .25, .5, .75, .9, .95, 1], NSTEPS = [1, 2, 4, 8, Infinity], TL = [0, .5, .9, 1];
const CELLS = []; for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (!G.term(c, r) && !G.isRock(c, r)) CELLS.push([c, r]);
const f3 = v => v.toFixed(3).replace('-', '−');
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−');
const NX = new Int8Array(80), RW = new Int8Array(80), TERMC = new Uint8Array(20);
for (let s = 0; s < 20; s++) { const c = s % 5, r = (s / 5) | 0; TERMC[s] = (G.term(c, r) || G.isRock(c, r)) ? 1 : 0; for (let a = 0; a < 4; a++) { const [nc, nr] = G.step(c, r, a); NX[s * 4 + a] = nr * 5 + nc; RW[s * 4 + a] = G.term(nc, nr); } }
const P = new Float64Array(4);
function pol(th, s) { const o = s * 4; let m = th[o]; for (let a = 1; a < 4; a++) if (th[o + a] > m) m = th[o + a]; let z = 0; for (let a = 0; a < 4; a++) { P[a] = Math.exp(th[o + a] - m); z += P[a]; } for (let a = 0; a < 4; a++) P[a] /= z; }
function pick(u) { let c = 0; for (let a = 0; a < 3; a++) { c += P[a]; if (u < c) return a; } return 3; }

// ───── 1. the actorcritic reel's story run (actor step 1, critic step 0.2, dice rng(5000)), frozen after walk 50 and 80
function acSnaps(stops) { const R = rng(5000), th = new Float64Array(80), V = new Float64Array(20), out = {};
  for (let e = 1; e <= Math.max(...stops); e++) { let s = SK;
    for (let k = 0; k < CAP; k++) { pol(th, s); const a = pick(R()), ns = NX[s * 4 + a], rew = RW[s * 4 + a], dl = rew + (rew ? 0 : GAM * V[ns]) - V[s];
      if (dl !== 0) { V[s] += AV * dl; for (let b = 0; b < 4; b++) th[s * 4 + b] += AT * dl * ((b === a) - P[b]); } s = ns; if (rew) break; }
    if (stops.includes(e)) out[e] = { th: Float64Array.from(th), V: Array.from(V) }; }
  return out; }
const SNAPS = acSnaps([50, 80]);
// the true grade of "up from the start": average real return after moving up (rulebook, exact) − the critic's guess at the start
function exactV(th) { let V = new Float64Array(20); for (let it = 0; it < 400; it++) { const N = new Float64Array(20); for (let s = 0; s < 20; s++) { if (TERMC[s]) continue; pol(th, s); let v = 0;
  for (let a = 0; a < 4; a++) { const r = RW[s * 4 + a]; v += P[a] * (r + (r ? 0 : GAM * V[NX[s * 4 + a]])); } N[s] = v; } V = N; } return V; }
// NS walks that start with "up" from the start, then follow the frozen chances (dice rng(777)); per walk: rewards and TD errors
function study(snap) { const { th, V } = snap, Vx = exactV(th), R = rng(777), walks = [];
  for (let i = 0; i < NS; i++) { let s = SK; const st = [];
    for (let k = 0; k < CAP; k++) { pol(th, s); const a = k === 0 ? 0 : pick(R()), ns = NX[s * 4 + a], rew = RW[s * 4 + a]; st.push({ s, a, ns, rew, dl: rew + (rew ? 0 : GAM * V[ns]) - V[s] }); s = ns; if (rew) break; }
    walks.push(st); }
  const truth = GAM * Vx[KUP] - V[SK];
  // n-step grade = n real rewards + 0.9^n × guess there − guess here = Σ_{k<n} 0.9^k δ_k ;  GAE(λ) = Σ_k (0.9 λ)^k δ_k
  const nstep = (w, n) => { let g = 0, f = 1; for (let k = 0; k < Math.min(n, w.length); k++) { g += f * w[k].dl; f *= GAM; } return g; };
  const gae = (w, l) => { let g = 0; for (let k = w.length - 1; k >= 0; k--) g = w[k].dl + GAM * l * g; return g; };
  const stats = xs => { const m = xs.reduce((a, b) => a + b, 0) / xs.length, sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length); return { xs, m, sd, bias: m - truth, miss: Math.sqrt((m - truth) ** 2 + sd * sd), lo: Math.min(...xs.slice(0, ND)), hi: Math.max(...xs.slice(0, ND)) }; };
  const N = {}, Lm = {}; NSTEPS.forEach(n => { N[n] = stats(walks.map(w => nstep(w, n))); }); LAMS.forEach(l => { Lm[l] = stats(walks.map(w => gae(w, l))); });
  const best = LAMS.reduce((b, l) => Lm[l].miss < Lm[b].miss ? l : b, 0);
  return { th, V, Vx, truth, walks, N, Lm, best, pUp: softmax(Array.from(th.slice(SK * 4, SK * 4 + 4)))[0] };
}
const S50 = study(SNAPS[50]), S80 = study(SNAPS[80]);
// an example walk for the n-step picture and the δ sum: the first sampled walk at walk 50 that is 6 to 12 moves long and ends on the cheese
const EXI = S50.walks.findIndex(w => w.length >= 6 && w.length <= 12 && w[w.length - 1].rew > 0), EXW = S50.walks[EXI];

// ───── 2. training with GAE grades (Pip computes them when a walk ends): run k uses dice rng(5000 + k), as in the other reels.
// After each walk: δ_t from the pre-walk critic, A_t = Σ (0.9 λ)^k δ_{t+k}; actor θ(s_t,·) += 1 × A_t × (onehot − pre-walk chance);
// critic: V(s_t) moves 0.2 of the way toward its λ-target A_t + old V(s_t), one step at a time.
const SS = new Int8Array(CAP), AA = new Int8Array(CAP), RR = new Int8Array(CAP), NN = new Int8Array(CAP), PP = new Float64Array(CAP * 4), AD = new Float64Array(CAP);
function runGAE(seed, lam) { const R = rng(seed), th = new Float64Array(80), V = new Float64Array(20), ret = new Float32Array(NW), d = new Float64Array(80), V0 = new Float64Array(20);
  for (let e = 0; e < NW; e++) { let s = SK, n = 0;
    for (let k = 0; k < CAP; k++) { pol(th, s); const a = pick(R()); SS[n] = s; AA[n] = a; for (let b = 0; b < 4; b++) PP[n * 4 + b] = P[b]; RR[n] = RW[s * 4 + a]; NN[n] = NX[s * 4 + a]; n++; const rew = RW[s * 4 + a]; s = NX[s * 4 + a]; if (rew) break; }
    let g = 0; for (let i = n - 1; i >= 0; i--) { const dl = RR[i] + (RR[i] ? 0 : GAM * V[NN[i]]) - V[SS[i]]; g = dl + GAM * lam * g; AD[i] = g; }
    ret[e] = RR[n - 1] ? RR[n - 1] * Math.pow(GAM, n - 1) : 0;
    d.fill(0); for (let i = 0; i < n; i++) { const o = SS[i] * 4; for (let b = 0; b < 4; b++) d[o + b] += AT * AD[i] * ((b === AA[i]) - PP[i * 4 + b]); }
    V0.set(V); for (let i = 0; i < n; i++) V[SS[i]] += AV * (AD[i] + V0[SS[i]] - V[SS[i]]);
    for (let i = 0; i < 80; i++) th[i] += d[i]; }
  return ret; }
function summarise(rets) { const cur = new Float64Array(NW); rets.forEach(r => { for (let i = 0; i < NW; i++) cur[i] += r[i] / rets.length; });
  const sm = Array.from(cur, (v, i) => { let s = 0, n = 0; for (let j = Math.max(0, i - 9); j <= i; j++) { s += cur[j]; n++; } return s / n; });
  const late = rets.map(r => { let s = 0; for (let i = NW - 100; i < NW; i++) s += r[i]; return s / 100; });
  return { cur: sm, below: late.filter(v => v < .4).length }; }
const RR_ = {}, rets = (l, upto = NRUN) => { const a = RR_[l] ||= []; while (a.length < upto) a.push(runGAE(5000 + a.length, l)); return a; };
const RACE = {}, race = l => RACE[l] ||= summarise(rets(l));
if (typeof setTimeout === 'function') { const tick = () => { const l = TL.find(l => (RR_[l] || []).length < NRUN); if (l == null) return; rets(l, (RR_[l] || []).length + 5); setTimeout(tick, 15); }; setTimeout(tick, 400); }

// ───── drawing helpers
function nums(V, a = 1, only) { for (const [c, r] of CELLS) { if (only && !only(c, r)) continue; const v = V[key(c, r)]; box(OX + c * CS + CS - 34, OY + r * CS + 4, 31, 16, C.card, .85 * a); txt(f2(v), OX + c * CS + CS - 5, OY + r * CS + 12, { s: 14, al: 'right', a, c: v < -.004 ? C.roseInk : v > .004 ? C.goldInk : C.soft }); } }
function dim(a = .62) { box(0, 0, W, 285, C.bg, a); }
// cloud of grades: x = grade, dots = the first 40 walks, gold = true grade, teal = average of all 2000
function cloud(st, S, t, o = {}) {
  const x0 = 50, x1 = 360, lo = -.25, hi = .65, X = v => x0 + (clamp(v, lo, hi) - lo) / (hi - lo) * (x1 - x0), yb = o.yb ?? 384, y0 = o.y ?? 288;
  box(20, y0, 360, yb - y0 + 22, C.card, .92); rrect(20, y0, 360, yb - y0 + 22, 1950, { c: C.line, w: 1 });
  rline(x0, yb, x1, yb, 1951, { c: C.line, w: 1, once: true }); [-.2, 0, .2, .4, .6].forEach(v => txt(String(v).replace('-', '−'), X(v), yb + 11, { s: 12, c: C.soft }));
  txt(o.title || 'grade of “up”: 40 of the 2,000 walks, one dot each', 30, y0 + 12, { s: 13, c: C.soft, al: 'left' });
  st.xs.slice(0, ND).forEach((v, i) => { const u = clamp((t - i * (o.dt ?? .06)) * 3); if (u <= 0) return; dot(X(v), yb - 9 - (i % 5) * 8, 3.1, C.rose, .8 * u); });
  const tr = clamp(t - (o.tTrue ?? 0)); if (tr > 0) alpha(tr, () => { rline(X(S.truth), yb - 52, X(S.truth), yb, 1952, { c: C.goldInk, w: 2.4, once: true }); txt(`true ${f3(S.truth)}`, X(S.truth) + 4, yb - 58, { s: 13, c: C.goldInk, al: 'left' }); });
  const mu = clamp(t - (o.tMean ?? 99)); if (mu > 0) alpha(mu, () => { rline(X(st.m), yb - 40, X(st.m), yb, 1953, { c: '#3f8f8b', w: 2, once: true }); txt(`average ${f3(st.m)}`, X(st.m) - 4, yb - 46, { s: 13, c: '#3f7f7c', al: 'right' }); });
  return X;
}
// a walk drawn on the maze with arcs n steps ahead
function arc(w, n, col, a, off) { const pts = [[0, 3]]; for (let k = 0; k < Math.min(n, w.length); k++) { const ns = w[k].ns; pts.push([ns % 5, (ns / 5) | 0]); }
  ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = 2.4; ctx.setLineDash([6, 4]); ctx.beginPath(); pts.forEach(([c, r], k) => { const x = cx(c) + off, y = cy(r) + off; k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.restore();
  const [c2, r2] = pts[pts.length - 1]; dot(cx(c2) + off, cy(r2) + off, 4.5, col, a); }
// error chart across λ: |bias| (rose), spread (blue), typical miss (ink)
function lamChart(S, t, o = {}) {
  const x0 = 70, x1 = 350, y0 = o.y0 ?? 262, y1 = o.y1 ?? 70, top = .25, X = l => x0 + l * (x1 - x0), Y = v => y0 - clamp(v / top) * (y0 - y1);
  rline(x0, y0, x1, y0, 1960, { c: C.line, w: 1, once: true }); rline(x0, y0, x0, y1, 1961, { c: C.line, w: 1, once: true });
  [0, .5, 1].forEach(l => txt(`λ ${l}`, X(l), y0 + 11, { s: 12, c: C.soft })); txt('0', x0 - 8, y0, { s: 12, c: C.soft, al: 'right' }); txt(String(top), x0 - 8, y1, { s: 12, c: C.soft, al: 'right' });
  if (o.title) txt(o.title, x0 - 40, y1 - 14, { s: 13, c: C.soft, al: 'left' });
  [[l => Math.abs(S.Lm[l].bias), C.rose, 1.8, 0], [l => S.Lm[l].sd, C.blue, 1.8, .8], [l => S.Lm[l].miss, C.ink, 2.6, 1.6]].forEach(([f, col, w, t0]) => { const u = clamp((t - t0) / 1.2); if (u <= 0) return;
    ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = w; ctx.globalAlpha = u; ctx.beginPath(); LAMS.forEach((l, i) => { const x = X(l), y = Y(f(l)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.restore();
    LAMS.forEach(l => dot(X(l), Y(f(l)), 2.6, col, u)); });
  if (t > 2.8) { const b = S.best; alpha(clamp(t - 2.8), () => { rrect(X(b) - 7, Y(S.Lm[b].miss) - 7, 14, 14, 1965, { c: C.goldInk, w: 1.6 }); txt(`best λ ${b}`, X(b) + (b > .7 ? -10 : 10), Y(S.Lm[b].miss) - 16, { s: 13, c: C.goldInk, al: b > .7 ? 'right' : 'left' }); }); }
  return { X, Y };
}
function lamLegend(x, y, a = 1) { alpha(a, () => [[C.rose, 'bias (how far off on average)'], [C.blue, 'spread (walk-to-walk noise)'], [C.ink, 'typical miss (both together)']].forEach(([col, lab], i) => { rline(x, y + i * 18, x + 20, y + i * 18, 1970 + i, { c: col, w: 2.5, once: true }); txt(lab, x + 26, y + i * 18, { s: 13, al: 'left' }); })); }
function raceChart(n, ls, o = {}) {
  const x0 = 60, x1 = 340, y0 = o.y0 ?? 262, y1 = o.y1 ?? 60, lo = -.1, hi = .6, X = i => x0 + i / (NW - 1) * (x1 - x0), Y = v => y0 - (v - lo) / (hi - lo) * (y0 - y1);
  rline(x0, Y(0), x1, Y(0), 960, { c: C.line, w: 1, once: true }); rline(x0, y0, x0, y1, 961, { c: C.line, w: 1, once: true });
  txt('0', x0 - 8, Y(0), { s: 12, c: C.soft, al: 'right' }); txt('0.5', x0 - 8, Y(.5), { s: 12, c: C.soft, al: 'right' }); txt('walk 1', x0, y0 + 10, { s: 12, c: C.soft, al: 'left' }); txt(`walk ${NW}`, x1, y0 + 10, { s: 12, c: C.soft, al: 'right' });
  if (o.title) txt(o.title, x0 - 30, y1 - 12, { s: 13, c: C.soft, al: 'left' });
  ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.grey; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(x0, Y(BEST)); ctx.lineTo(x1, Y(BEST)); ctx.stroke(); ctx.restore(); txt(`best ${f2(BEST)}`, x1 + 3, Y(BEST), { s: 12, c: C.soft, al: 'left' });
  ls.forEach(l => { const S = race(l), m = Math.min(n, NW); if (m < 2) return; ctx.save(); ctx.strokeStyle = LCOL[l]; ctx.lineWidth = 2; ctx.beginPath(); for (let i = 0; i < m; i++) { const x = X(i), y = Y(S.cur[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); });
  return { X, Y };
}
const LCOL = { 0: '#3f8f8b', .5: C.gold, .9: C.orange, 1: C.rose };
const NCOL = { 1: '#3f8f8b', 2: '#7fae9a', 4: C.gold, 8: C.orange, Infinity: C.rose };
const nName = n => n === Infinity ? 'whole walk' : `n = ${n}`;

const STEPS = [
 { h: 'How far should Pip look?', p: ['Same maze: +1 for the cheese, −1 for the cat, either ends the walk; γ (gamma) is 0.9. Pip’s actor needs a **grade** for each move: how much better it went than the critic expected.', 'Actor-critic graded with one real step plus the critic’s guess. REINFORCE with a baseline used the whole real return. Is there something in between, and is it better?'],
   shot: [3, 9],
   draw(t) { maze({ V: S50.V, t }); nums(S50.V, clamp(t - .8)); fans(G, S50.th, { a: clamp(t - .4) }); pip(cx(0), cy(3), 1, t);
     if (t > 4) card([{ s: 'one real step + critic’s guess   (actor-critic)', c: '#3f7f7c' }, { s: 'the whole real return   (REINFORCE + baseline)', c: C.roseInk }, { s: '… or something in between?', c: C.goldInk }], t - 4, { gap: 1.2 }); } },
 { h: 'Two grades, one move', p: [`Actor-critic reel again: Pip frozen after walk 50, going up from the start. Now ${NS.toLocaleString('en-US')} walks, not 40. The one-step grade, the TD error, is ${f2(S50.N[1].m)} every time.`, `Whole-walk grade, return minus start guess: average ${f3(S50.N[Infinity].m)}, but it swings; 40 walks shown as dots. Gold line: the **true grade**, ${f3(S50.truth)}: up’s exact value under these chances, from the rulebook, minus the start guess.`],
   shot: [4, 11],
   draw(t) { maze({ V: S50.V, t, hi: [[0, 3, C.teal, .2], [0, 2, C.gold, .2]] }); nums(S50.V, .8); fans(G, S50.th, { a: .6 }); pip(cx(0), cy(3), 1, t);
     const X = cloud(S50.N[Infinity], S50, t - .8, { tTrue: 5, tMean: 6.5, title: 'grades of “up”: whole walk (dots), one step (line)' });
     if (t > 3.5) alpha(clamp(t - 3.5), () => { rline(X(S50.N[1].m), 330, X(S50.N[1].m), 384, 1954, { c: '#3f8f8b', w: 3, once: true }); txt(`one step ${f2(S50.N[1].m)}`, X(S50.N[1].m) - 4, 324, { s: 13, c: '#3f7f7c', al: 'right' }); }); } },
 { h: 'Bias and noise', p: [`The one-step grade has no noise: on the dry floor up never slips. But it is always ${f2(-S50.N[1].bias)} too low: the guess for that cell lags. Leaning one way every time is **bias**.`, `The whole-walk grade is right on average: ${f3(S50.N[Infinity].m)} against ${f3(S50.truth)}; the gap is just luck over 2,000 walks. But each walk’s luck moves it: spread ${f2(S50.N[Infinity].sd)}. That jumping is **variance**, or noise.`],
   shot: [4, 10],
   draw(t) { maze({ V: S50.V, t }); dim(.5); fans(G, S50.th, { a: .3 });
     card([{ s: `one step:   ${f2(-S50.N[1].bias)} too low,  spread 0`, c: '#3f7f7c' }, { s: `whole walk:   off by ${f2(Math.abs(S50.N[Infinity].bias))},  spread ${f2(S50.N[Infinity].sd)}`, c: C.roseInk }, { s: 'bias: leans one way · variance: jumps around', c: C.soft }], t - .5, { gap: 1.4, y: 70 });
     const X = cloud(S50.N[Infinity], S50, 9, { tTrue: 0, tMean: 0 }); alpha(1, () => { rline(X(S50.N[1].m), 330, X(S50.N[1].m), 384, 1954, { c: '#3f8f8b', w: 3, once: true }); txt(`one step`, X(S50.N[1].m) - 4, 324, { s: 13, c: '#3f7f7c', al: 'right' }); }); } },
 { h: 'n steps, then the guess', p: ['From the traces reel, the **n-step target**: the next n real rewards, discounted 0.9 per step as in a return, plus the critic’s guess where Pip is then, discounted n times. Grade: target minus the start guess.', 'n = 1 gives the TD error. If the walk ends sooner, no guess: just the whole return minus the start guess. Between, n is a dial.'],
   shot: [4, 11],
   draw(t) { maze({ V: S50.V, t }); nums(S50.V, .5); pip(cx(0), cy(3), 1, t);
     [[1, '#3f8f8b', 1, -12], [3, C.orange, 3, 0], [EXW.length, C.rose, 5, 12]].forEach(([n, col, t0, off]) => { const a = clamp(t - t0); if (a > 0) arc(EXW, n, col, a, off); });
     card([{ s: 'n = 1:  reward + 0.9 × guess one cell on', c: '#3f7f7c' }, { s: 'n = 3:  3 rewards + 0.9³ × guess three cells on', c: '#c4581f' }, { s: 'whole walk:  all rewards, no guess', c: C.roseInk }, { s: 'grade  =  target  −  guess where Pip started', c: C.soft }], t - 1, { gap: 1.6, fs: 16, lh: 22 }); } },
 { h: 'Turning the n dial', p: [`Grades for n = 1, 2, 4, 8 and the whole walk. As n grows, the average nears the true grade, ${f2(S50.N[1].m)} to ${f2(S50.N[8].m)} at n = 8; spread grows from 0 to ${f2(S50.N[8].sd)}.`, `More real steps: less bias, more noise. n = 8 is noisier even than the whole walk, spread ${f2(S50.N[Infinity].sd)}: it adds the guess after 8 moves, and guesses differ by cell.`],
   shot: [3, 7.5, 12],
   draw(t) { const i = clamp(Math.floor(t / 2.6), 0, 4), n = NSTEPS[i], st = S50.N[n]; maze({ V: S50.V, t }); nums(S50.V, .5); pip(cx(0), cy(3), 1, t);
     arc(EXW, n === Infinity ? EXW.length : n, NCOL[n], .9, 0);
     box(W / 2 - 60, 268, 120, 20, C.bg, .9); txt(nName(n), W / 2, 278, { s: 20, c: NCOL[n] === C.gold ? C.goldInk : NCOL[n] === C.rose ? C.roseInk : NCOL[n] });
     cloud(st, S50, 9, { tTrue: 0, tMean: 0, title: `grade of “up”, ${nName(n)}:  average ${f2(st.m)},  spread ${f2(st.sd)}` }); } },
 { h: 'GAE: mix every n', p: ['Why one n? **Generalised advantage estimation**, GAE, averages them all with fading weights; its grade is the baseline reel’s advantage. n = 1 gets 1 − λ; each next n, λ times the last. They add up to 1, so it stays an average; weight past the walk’s end goes to the whole-walk grade.', '**λ** (lambda), between 0 and 1, sets how fast they fade.'],
   shot: [4, 10],
   draw(t) { [[.5, 60], [.9, 200]].forEach(([l, y], g) => { const a = clamp(t - .5 - g * 2.5); if (a <= 0) return; txt(`λ = ${l}`, 30, y, { s: 17, al: 'left', c: g ? C.orange : C.goldInk, a });
       for (let n = 1; n <= 12; n++) { const w = (1 - l) * Math.pow(l, n - 1), h = w / .5 * 70 * clamp((t - .7 - g * 2.5 - n * .08) * 3); box(70 + n * 24, y + 80 - h, 16, h, g ? C.orange : C.gold, .85); txt(String(n), 78 + n * 24, y + 90, { s: 11, c: C.soft, a }); }
       txt(`n = 1: ${f2(1 - l)}`, 380, y + 10, { s: 13, c: C.soft, al: 'right', a }); txt('n →', 72, y + 90, { s: 11, c: C.soft, a, al: 'right' }); });
     if (t > 6) card(['weight of n:  (1 − λ) × λ, n − 1 times', { s: 'the weights add up to 1', c: C.soft }], t - 6, { gap: 1.2, y: 320 }); } },
 { h: 'A neat shortcut', p: [`The mix has a simple form. Number the walk’s TD errors δ₀, δ₁, and so on. A two-step grade is δ₀ + 0.9 δ₁, because the guess in between cancels. Mixing all n weights each later δ by another 0.9 λ.`, `Later surprises count less, like the traces reel’s fading marks. This walk, λ = 0.9: grade ${f2(EXG(.9))}. One move, δ₃, bumped into the wall.`],
   shot: [4, 11],
   draw(t) { maze({ V: S50.V, t }); nums(S50.V, .6); const D = .7, k = clamp(Math.floor((t - .5) / D) + 1, 0, EXW.length);
     const pts = [[0, 3]].concat(EXW.slice(0, k).map(q => [q.ns % 5, (q.ns / 5) | 0])); trail(pts, .7);
     const cnt = {}; EXW.slice(0, k).forEach((q, i) => { const c = q.s % 5, r = (q.s / 5) | 0, j = cnt[q.s] = (cnt[q.s] ?? -1) + 1, y = cy(r) + 4 + j * 15; box(cx(c) - 30, y - 7, 60, 14, C.card, .9); txt(`δ${'₀₁₂₃₄₅₆₇₈₉'[i] ?? i} ${f2(q.dl)}${q.ns === q.s ? ' bump' : ''}`, cx(c), y, { s: 12, c: q.dl > 0 ? C.goldInk : C.roseInk }); });
     const p = k ? EXW[k - 1] : null; pip(p ? cx(p.ns % 5) : cx(0), p ? cy((p.ns / 5) | 0) : cy(3), 1, t);
     if (t > .5 + EXW.length * D) card(['grade  =  δ₀  +  0.9λ × δ₁  +  (0.9λ)² × δ₂  + …', `this walk, λ = 0.9:  ${f2(EXG(.9))}`, { s: `λ = 0: just δ₀ = ${f2(EXW[0].dl)};  λ = 1: the whole walk, ${f2(EXG(1))}`, c: C.soft }], t - .5 - EXW.length * D, { gap: 1.2, fs: 16, lh: 22 }); } },
 { h: 'Your turn', ctl: { label: 'λ', values: LAMS, def: 4, onChange: st => {} },
   p: ['Drag **λ** and watch the grades of the same move. Gold is the true grade, teal the average of all the walks.', 'At 0 the dots stack on one wrong spot; at 1 they centre on the truth but spread wide. The **typical miss** combines both: the square root of bias squared plus spread squared, how far one grade usually lands from the truth.'],
   shot: [3],
   draw(t, st) { const l = LAMS[st.ci ?? 4], s = S50.Lm[l]; maze({ V: S50.V, t }); dim(.55); fans(G, S50.th, { a: .25 });
     card([`λ = ${l}`, { s: `average ${f3(s.m)}   (true ${f3(S50.truth)})`, c: C.ink }, { s: `${Math.abs(s.bias) < .005 ? 'no bias' : f2(Math.abs(s.bias)) + (s.bias < 0 ? ' too low' : ' too high')}    spread ${f2(s.sd)}`, c: C.soft }, { s: `typical miss ${f2(s.miss)}`, c: C.goldInk }], 3, { gap: 0, y: 70 });
     cloud(s, S50, (t - (st.changed ?? 0)) * 2 + .2, { tTrue: 0, tMean: 0, dt: .02 }); } },
 { h: 'Which λ is best?', p: ['Across: λ from 0 to 1. Rose: the bias, blue: the spread, black: the typical miss, which combines both. Lower is better.', `With this critic, frozen after walk 50, the typical miss keeps falling all the way, so it is smallest at λ = ${S50.best}: ${f2(S50.Lm[S50.best].miss)}, against ${f2(S50.Lm[0].miss)} at λ = 0. The critic still lags, so trusting it less pays.`],
   shot: [4.5, 10],
   draw(t) { txt('frozen after walk 50: grades of “up” from the start', W / 2, 26, { s: 14, c: C.soft }); lamChart(S50, t - .5); lamLegend(110, 300, clamp(t - .3)); } },
 { h: 'A better critic', p: [`Freeze Pip after walk 80. The critic caught up: one-step grades miss by only ${f2(Math.abs(S80.Lm[0].bias))}. Pip’s chances improved, up ${f2(S80.pUp)}, not ${f2(S50.pUp)}, so walks wander less and long grades calm: spread ${f2(S80.Lm[1].sd)} at λ = 1.`, `Best λ: ${S80.best}, typical miss ${f2(S80.Lm[S80.best].miss)}, against ${f2(S80.Lm[1].miss)} at λ = 1. The critic wins despite calmer walks: the best λ depends on how much you **trust the critic**.`],
   shot: [4.5, 10],
   draw(t) { txt('frozen after walk 80: grades of “up” from the start', W / 2, 26, { s: 14, c: C.soft }); lamChart(S80, t - .5); lamLegend(110, 300, clamp(t - .3)); } },
 { h: 'Learning with GAE', get p() { const a = race(0), b = race(.5), c = race(1);
     return [`Now Pip learns with GAE: after each walk, every move gets its grade. The actor is nudged by it, step 1; the critic moves toward grade plus its guess, step 0.2. ${NRUN} runs per λ, same dice. Chart: return from the start, averaged over runs and 10 neighbouring walks.`, `After 50 walks: ${f2(a.cur[49])} at λ = 0, ${f2(b.cur[49])} at 0.5, ${f2(c.cur[49])} at 1.`]; },
   shot: [5, 12],
   draw(t) { const n = Math.floor(clamp((t - .8) / 6) * NW), g = raceChart(n, TL, { title: `return from the start, average of ${NRUN} runs` });
     alpha(1, () => TL.forEach((l, i) => { rline(200, 170 + i * 18, 222, 170 + i * 18, 1990 + i, { c: LCOL[l], w: 2.5, once: true }); txt(`λ = ${l}`, 228, 170 + i * 18, { s: 13, al: 'left' }); }));
     if (n >= 50) { const X = g.X(49); rline(X, 60, X, 262, 1999, { c: C.line, w: 1, once: true }); txt('walk 50', X + 3, 252, { s: 12, c: C.soft, al: 'left' }); }
     if (n >= NW) card(TL.map(l => ({ s: `λ ${l}:  walk 50 ${f2(race(l).cur[49])},  walk ${NW} ${f2(race(l).cur[NW - 1])},  runs below 0.4: ${race(l).below}`, c: LCOL[l] === C.gold ? C.goldInk : LCOL[l] === C.rose ? C.roseInk : LCOL[l] === C.orange ? '#c4581f' : '#3f7f7c' })), t - 7, { gap: .5, y: 290, fs: 15, lh: 21 }); } },
 { h: 'Middle ground', get p() { const a = race(0), b = race(.5), c = race(1);
     return [`λ = 0 starts slowest: its grades lean on a blank critic. λ = 1 starts fast, stays noisy: ${c.below} runs below 0.4, against ${a.below} for λ = 0.`, `At walk 50, λ = 0.5 is fastest. λ = 0.9 is no better than 1: ${race(.9).below} runs against ${c.below}. λ = 0 ends best, ${f2(a.cur[NW - 1])}: the critic is good by then, as at walk 80.`]; },
   shot: [4, 10],
   draw(t) { raceChart(NW, TL, { title: `return from the start, average of ${NRUN} runs` });
     TL.forEach((l, i) => { rline(200, 170 + i * 18, 222, 170 + i * 18, 1990 + i, { c: LCOL[l], w: 2.5, once: true }); txt(`λ = ${l}`, 228, 170 + i * 18, { s: 13, al: 'left' }); });
     card([{ s: 'λ 0: steady but biased while the critic is new', c: '#3f7f7c' }, { s: 'λ 1: unbiased but noisy, more runs left behind', c: C.roseInk }, { s: 'in between: a compromise', c: C.goldInk }], t - .8, { gap: 1.3, y: 296 }); } },
 { h: 'The price: waiting', p: ['A GAE grade needs the TD errors of the moves that come after, so Pip cannot grade a move the moment it makes it. Here Pip waits for the end of each walk.', 'Real systems cut experience into chunks, a few hundred or a few thousand steps, and let the critic’s guess stand in for whatever comes after the cut.'],
   shot: [4, 10],
   draw(t) { maze({ V: S50.V, t }); nums(S50.V, .5); const D = .5, k = clamp(Math.floor((t - .5) / D) + 1, 0, EXW.length); const pts = [[0, 3]].concat(EXW.slice(0, k).map(q => [q.ns % 5, (q.ns / 5) | 0])); trail(pts, .7);
     const p = k ? EXW[k - 1] : null; pip(p ? cx(p.ns % 5) : cx(0), p ? cy((p.ns / 5) | 0) : cy(3), 1, t);
     if (k < EXW.length) caption(`move ${k}: its grade still waits for later TD errors`); else bubble('+1', cx(4), cy(0) - 18, t - .5 - EXW.length * D);
     if (t > 1 + EXW.length * D) card(['GAE needs the moves that come after', { s: 'here: grade every move when the walk ends', c: C.soft }, { s: 'in practice: chunks, then the critic’s guess', c: C.goldInk }], t - 1 - EXW.length * D, { gap: 1.2 }); } },
 { h: 'In practice', p: ['GAE was introduced by John Schulman and colleagues in 2015. With neural networks as actor and critic, **λ = 0.95** together with γ = 0.99 became a common default; the PPO paper used exactly that.', 'So GAE is the same dial as TD(λ) in the traces reel, now used to grade moves for the actor instead of to update values.'],
   shot: [4, 10],
   draw(t) { txt('λ', 40, 140, { s: 26, c: C.soft, a: clamp(t - .3) }); rline(70, 140, 350, 140, 2001, { w: 2, c: C.line }); [[0, 'one step (TD)'], [.95, '0.95'], [1, 'whole walk (Monte Carlo)']].forEach(([l, s], i) => { const x = 70 + l * 280, a = clamp(t - .6 - i * .5);
       dot(x, 140, l === .95 ? 6 : 4, l === .95 ? C.gold : C.soft, a); txt(s, x + (i === 2 ? -4 : i === 1 ? -10 : 0), 140 + (i === 1 ? -22 : 22), { s: 14, c: l === .95 ? C.goldInk : C.soft, a, al: i === 2 ? 'right' : i === 1 ? 'right' : 'left' }); });
     if (t > 2.5) card(['2015: GAE, John Schulman and colleagues', 'common default: λ = 0.95, γ = 0.99 (PPO)', { s: 'same dial as TD(λ), now for grading moves', c: C.soft }], t - 2.5, { gap: 1.2, y: 220 }); } },
 { h: 'Recap', p: ['An **n-step grade** adds n real rewards, then trusts the critic: small n is steady but biased when the critic is off, large n is unbiased but noisy.', '**GAE** mixes every n with weights that fade by λ, which is the same as summing TD errors weighted by (0.9 λ) per step. The best λ depends on how far the critic can be trusted.'],
   shot: [3, 9],
   draw(t) { maze({ V: S80.V, t }); nums(S80.V, .7); fans(G, S80.th, { a: .6 }); pip(cx(0), cy(3), 1, t);
     card(['n-step grade = n rewards + 0.9ⁿ × guess − guess here', 'GAE = δ₀ + 0.9λ δ₁ + (0.9λ)² δ₂ + …', 'λ = 0: TD error  ·  λ = 1: whole return', { s: 'small λ: biased · large λ: noisy', c: C.soft }], t - .3, { gap: 1, fs: 16, lh: 22 }); } },
];
function EXG(l) { let g = 0; for (let k = EXW.length - 1; k >= 0; k--) g = EXW[k].dl + GAM * l * g; return g; }

RLStory.register('gae', { title: 'n-step Returns & GAE', track: 'Policy Gradients', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { S50, S80, EXW, EXI, EXG, race, runGAE, SNAPS } });
})();
