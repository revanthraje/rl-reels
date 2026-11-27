// DDPG and TD3 as a story: Pip's jump (pg_world World 3) becomes a row of identical gaps, so a jump also decides whether
// Pip keeps going. The jump strength is a number, so there is no menu to take a max over: a deterministic actor (one
// number) climbs the slope of a learned critic Q(strength). Then the critic turns out too optimistic, and TD3's three
// fixes are tried one by one. Every number comes from the seeded runs below (runs k = 0..7 use rng(100 + k)).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;

// ───── engine-independent maths: + − × ÷ and sqrt are exact in every browser, Math.exp / log / cos are not, and a long
// training run amplifies last-bit differences. So the simulation uses its own exp, log and bell-curve noise.
const LN2 = 0.6931471805599453;
function xexp(x) { if (x < -700) return 0; const k = Math.round(x / LN2), r = x - k * LN2; let term = 1, sum = 1; for (let n = 1; n < 18; n++) { term = term * r / n; sum += term; }
  if (k > 0) for (let i = 0; i < k; i++) sum *= 2; else for (let i = 0; i < -k; i++) sum /= 2; return sum; }
function xlog(x) { let m = x, e = 0; while (m < 0.7071067811865476) { m *= 2; e--; } while (m > 1.4142135623730951) { m /= 2; e++; }
  const y = (m - 1) / (m + 1), y2 = y * y; let p = y, sum = 0; for (let n = 1; n < 40; n += 2) { sum += p / n; p *= y2; } return 2 * sum + e * LN2; }
function xgauss(R) { for (;;) { const u = 2 * R() - 1, v = 2 * R() - 1, q = u * u + v * v; if (q > 0 && q < 1) return u * Math.sqrt(-2 * xlog(q) / q); } }   // polar method
// ───── the world: gaps of 0.7 m, narrow cheese ledge from 0.4 to 1.0 m, wind wobble, falls end the run
const GAM = 0.9, GAP = 0.7, FALL = 0.3, AMAX = 1.2, WOB = 0.05, K = 15, WD = 0.07, NT = 1201;
const rew = l => { const z = (l - GAP) / 0.15; return Math.abs(l - GAP) > FALL ? 0 : xexp(-(z * z)); };
const clip = (v, a, b) => Math.max(a, Math.min(b, v));
// the critic: 15 smooth bumps along the strength axis, Q(a) = sum of weight × bump(a) (a tiny network with one layer)
const CEN = Array.from({ length: K }, (_, i) => AMAX * i / (K - 1));
const PHI = new Float64Array(NT * K); for (let j = 0; j < NT; j++) for (let i = 0; i < K; i++) { const z = (AMAX * j / (NT - 1) - CEN[i]) / WD; PHI[j * K + i] = xexp(-(z * z) / 2); }
const idx = a => Math.round(clip(a, 0, AMAX) / AMAX * (NT - 1)) * K;
const Qw = (w, a) => { const o = idx(a); let s = 0; for (let i = 0; i < K; i++) s += w[i] * PHI[o + i]; return s; };
const slopeW = (w, a) => (Qw(w, a + .002) - Qw(w, a - .002)) / .004;
// the truth, from the rulebook (Pip never sees it): average over the wind
const ZK = Array.from({ length: 61 }, (_, k) => -3 + 6 * k / 60), PW = ZK.map(z => xexp(-z * z / 2)), ZS = PW.reduce((a, b) => a + b);
function rp(a) { let r = 0, p = 0; ZK.forEach((z, k) => { const l = a + WOB * z; r += PW[k] * rew(l); p += PW[k] * (Math.abs(l - GAP) > FALL ? 0 : 1); }); return [r / ZS, p / ZS]; }
const Vtrue = m => { const [r, p] = rp(m); return r / (1 - GAM * p); };          // worth of always jumping m, as if the row never ends
const Qtrue = (m, a) => { const [r, p] = rp(a); return r + GAM * p * Vtrue(m); };  // jump a now, then always m
// TD3's critic values a jumper whose later jumps are jiggled (target noise, clipped at 2 spreads): its own "truth"
// quadrature on [−2, 2] plus the clipped tails (2.275 % each side land exactly on ±2 spreads)
const ZJ = Array.from({ length: 81 }, (_, k) => -2 + 4 * k / 80), PJ0 = ZJ.map(z => xexp(-z * z / 2)), PJS = PJ0.reduce((a, b) => a + b), TAIL = 0.02275;
const PJ = PJ0.map((w, k) => w * (k === 0 || k === 80 ? .5 : 1) / (PJS - .5 * PJ0[0] - .5 * PJ0[80]) * (1 - 2 * TAIL) + (k === 0 || k === 80 ? TAIL : 0));
const VtrueJ = (m, sm) => { let r = 0, p = 0; ZJ.forEach((z, k) => { const [rr, pp] = rp(m + sm * z); r += PJ[k] * rr; p += PJ[k] * pp; }); return r / (1 - GAM * p); };
const QtrueJ = (m, a, sm) => { const [r, p] = rp(a); return r + GAM * p * VtrueJ(m, sm); };
// ───── training: DDPG, plus TD3's fixes as switches. o = { twin, delay, smooth, k }
const P = { T: 12000, B: 8, lr: .1, alr: .01, tau: .02, expl: .1, every: 50, SM: .03, CR: 1 };
// a run that can be advanced a slice at a time (so the background work never blocks a frame for long)
function makeRun(o) {
  const R = rng(100 + o.k), g = () => xgauss(R), nc = o.twin ? 2 : 1, w = [new Float64Array(K), new Float64Array(K)], wt = [new Float64Array(K), new Float64Array(K)];
  let m = .3, mt = .3, t = 0; const buf = [], log = [], jumps = []; const run = { log, jumps, cap: null, m, done: false };
  run.adv = n => {
    for (const end = Math.min(P.T, t + n); t < end; t++) {
      const a = clip(m + P.expl * g(), 0, AMAX), l = a + WOB * g(), crumb = P.CR ? 1 + P.CR * (2 * R() - 1) : 1, r = rew(l) * crumb, d = Math.abs(l - GAP) > FALL;
      buf.push([a, r, d]); if (t < 400) jumps.push({ m, a, l, r, d });
      if (t >= 99 && t < 400 && !run.cap && !d) { const qn = Qw(wt[0], mt), q = Qw(w[0], a), y = r + GAM * qn; const rd = v => Math.round(v * 100) / 100, ok = rd(rd(rew(l)) * rd(crumb)) === rd(r) && rd(rd(r) + GAM * rd(qn)) === rd(y);
        if (ok && y > q + .3 && crumb > .8 && crumb < 1.2) run.cap = { t: t + 1, a, l, r, crumb, curve: rew(l), d, mt, qn, q, y }; }
      for (let b = 0; b < P.B; b++) for (let k = 0; k < nc; k++) {
        const [aa, rr, dd] = buf[Math.floor(R() * buf.length)];
        let a2 = mt; if (o.smooth) a2 = clip(a2 + clip(o.smooth * g(), -2 * o.smooth, 2 * o.smooth), 0, AMAX);
        let qn = Qw(wt[0], a2); if (o.twin) qn = Math.min(qn, Qw(wt[1], a2));
        const y = rr + (dd ? 0 : GAM * qn), oi = idx(aa); let q = 0; for (let i = 0; i < K; i++) q += w[k][i] * PHI[oi + i];
        const e = y - q; for (let i = 0; i < K; i++) w[k][i] += P.lr * e * PHI[oi + i];
      }
      if (t % (o.delay || 1) === 0) {
        const sl = slopeW(w[0], m); if (!run.snap && t >= 19 && sl >= 1.5) run.snap = { t: t + 1, m, w: Array.from(w[0]) };
        m = clip(m + P.alr * sl, 0, AMAX);
        for (let k = 0; k < nc; k++) for (let i = 0; i < K; i++) wt[k][i] += P.tau * (w[k][i] - wt[k][i]); mt += P.tau * (m - mt);
      }
      if ((t + 1) % P.every === 0) { let q = Qw(w[0], m); if (o.twin) q = Math.min(q, Qw(w[1], m)); log.push({ t: t + 1, m, q, tv: Qtrue(m, m), tvj: o.smooth ? QtrueJ(m, m, o.smooth) : null, w: Array.from(w[0]), w1: o.twin && t + 1 === P.T ? Array.from(w[1]) : null }); }
    }
    run.m = m; run.done = t >= P.T; return run;
  };
  return run;
}
const train = o => makeRun(o).adv(Infinity);
// four versions: DDPG, + two critics, + delayed actor, + target noise (= TD3). 8 runs each, computed in the background.
const CFG = [{ name: 'DDPG', o: {} }, { name: '+ two critics', o: { twin: 1 } }, { name: '+ delay', o: { twin: 1, delay: 2 } }, { name: 'TD3: + target noise', o: { twin: 1, delay: 2, smooth: P.SM } }];
const NR = 8, RES = CFG.map((c, ci) => Array.from({ length: NR }, (_, k) => makeRun(Object.assign({ k }, c.o)))), JOBS = [];
[0, 3, 1, 2].forEach(c => { for (let k = 0; k < NR; k++) JOBS.push(RES[c][k]); });
function job(c, k) { const r = RES[c][k]; return r.done ? r : r.adv(Infinity); }
const GW0 = 4000;                                     // the gap is judged on jumps 4000 to 12000, once the critic has caught up
const memo = {};
function agg(c) { if (memo[c]) return memo[c]; const runs = RES[c].map((r, k) => job(c, k)), n = runs[0].log.length, i0 = GW0 / P.every;
  const gap = Array.from({ length: n }, (_, i) => runs.reduce((s, r) => s + r.log[i].q - r.log[i].tv, 0) / NR);
  const gapJ = runs[0].log[0].tvj == null ? null : Array.from({ length: n }, (_, i) => runs.reduce((s, r) => s + r.log[i].q - r.log[i].tvj, 0) / NR);
  const fin = runs.map(r => { const w = r.log.slice(i0); return w.reduce((s, x) => s + x.q - x.tv, 0) / w.length; });
  const finJ = runs[0].log[0].tvj == null ? null : runs.map(r => { const w = r.log.slice(i0); return w.reduce((s, x) => s + x.q - x.tvj, 0) / w.length; });
  const tv = runs.reduce((s, r) => s + r.log.slice(i0).reduce((a, x) => a + x.tv, 0) / (n - i0), 0) / NR;
  return (memo[c] = { runs, gap, gapJ, fin, finJ, meanJ: finJ && finJ.reduce((a, b) => a + b, 0) / NR, posJ: finJ && finJ.filter(v => v > 0).length, sdiff: Vtrue(.7) - VtrueJ(.7, P.SM), mean: fin.reduce((a, b) => a + b, 0) / NR, pos: fin.filter(v => v > 0).length, tv, m: runs.reduce((s, r) => s + r.m, 0) / NR }); }
(function pump() { if (typeof setTimeout === 'undefined') return; while (JOBS.length && JOBS[0].done) JOBS.shift(); if (!JOBS.length) return; JOBS[0].adv(400); setTimeout(pump, 8); })();
const ready = c => RES[c].every(r => r.done);

// ───── numbers the story quotes (story run = DDPG run k = 6: clear early falls, a typical-to-large final gap)
const SK = 6, S = job(0, SK), SL = S.log, SJ = S.jumps, SEND = SL[SL.length - 1];
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−'), sg2 = v => (v >= .005 ? '+' : '') + f2(v);
const PERFECT = Vtrue(GAP), R07 = rp(GAP)[0];
const F50 = SJ.slice(0, 50).filter(j => j.d).length, MAT = t => SL.find(x => x.t === t).m;
const SNAP = S.snap, SNAPS = slopeW(SNAP.w, SNAP.m), SNAPN = SNAP.m + P.alr * SNAPS;
const SNAPMAX = Math.max(...Array.from({ length: 121 }, (_, i) => Qw(SNAP.w, i / 100))), SNAPTOP = [.5, 1, 2, 5, 10].find(v => v >= SNAPMAX * 1.05);
const CAP = S.cap, CAPY = CAP.r + (CAP.d ? 0 : GAM * CAP.qn);
const MT = SL.findIndex(x => Math.abs(x.m - GAP) < .02);                 // first log point with the actor within 0.02 of 0.7
const FALLS_AFTER = SJ.slice(100, 400).filter(j => j.d).length;
const STAVG = SL.slice(GW0 / P.every).reduce((a, x) => a + x.q - x.tv, 0) / (SL.length - GW0 / P.every);
// close-up moment: the latest log point after jump 4000 whose gap is nearest this run's average gap
const SN2 = SL.slice(GW0 / P.every).reduce((b, x) => Math.abs(x.q - x.tv - STAVG) <= Math.abs(b.q - b.tv - STAVG) ? x : b);
const CATCH = SL.find(x => x.q > x.tv - .5).t, DIP = SL.filter(x => x.t > 1000).reduce((b, x) => x.tv < b.tv ? x : b);
// where the critic is above the truth at the end of the story run
const fx = (x, d = 2) => (+x).toFixed(d);

// ───── drawing: the ledges (shared ruler with the chart below)
const X = d => 70 + d / 1.2 * 290, GY = 182;
function water(t) { box(X(0), GY + 50, 380 - X(0), 18, '#cfe3e2', .9); for (let i = 0; i < 6; i++) { const x = X(0) + 10 + i * 52 + Math.sin(t * 1.5 + i) * 4; rline(x, GY + 58, x + 18, GY + 58, 1200 + i, { c: '#7fb0ae', w: 1, once: true }); } }
function ledges(t, o = {}) {
  water(t);
  box(14, GY, X(0) - 14, 62, '#d9d1c6'); rrect(14, GY, X(0) - 14, 62, 890);
  box(X(.4), GY, X(1) - X(.4), 62, '#d9d1c6'); rrect(X(.4), GY, X(1) - X(.4), 62, 891);
  if (o.cheese !== 0) L.cheese(X(GAP), GY - 9, .75);
  if (o.ruler !== 0) { const y = GY + 80; rline(X(0), y, X(1.2), y, 894, { c: C.line, w: 1 });
    [0, .4, .7, 1, 1.2].forEach((d, i) => { rline(X(d), y - 4, X(d), y + 4, 895 + i, { c: C.line, w: 1, once: true }); txt(d + ' m', X(d), y + 13, { s: 12, c: C.soft }); }); }
}
// Pip's jump with strength a, landing at l (wind), u = 0..1 progress; fell → drops into the water
function jumpPos(a, l, d, u) {
  const x0 = X(0) - 14, x1 = X(l), h = 34 + 60 * a, uu = clamp(u);
  if (!d || uu < .72) { const v = clamp(uu / .72); return { x: lerp(x0, x1, v), y: GY - 10 - h * 4 * v * (1 - v), land: v >= 1 }; }
  const v = clamp((uu - .72) / .28); return { x: x1, y: GY - 10 + v * 58, sink: v };
}
function pipAt(p, t, face = 1) { if (p.sink >= 1) return; L.mouse(p.x, p.y, face, .8, t); }
function splash(x, u) { if (u <= 0 || u >= 1) return; for (let i = 0; i < 5; i++) { const an = -Math.PI * (.15 + .7 * i / 4); dot(x + Math.cos(an) * 14 * u, GY + 50 + Math.sin(an) * 16 * u, 2.2, '#7fb0ae', 1 - u); } }
// one animated jump, from t0, lasting dur seconds; returns the pose
function oneJump(j, t, t0, dur) { const u = (t - t0) / dur, p = jumpPos(j.a, j.l, j.d, u); if (j.d) splash(X(j.l), (u - .95) * 2.5); return p; }

// ───── the chart: same left-right ruler as the scene, up = score
const CY0 = 388, CY1 = 304, QMAX = 10, CY = (q, top = QMAX) => CY0 - clamp(q, -.1 * top, 1.1 * top) / top * (CY0 - CY1);
function chartFrame(o = {}) {
  const y0 = o.y0 ?? CY0, y1 = o.y1 ?? CY1, top = o.top ?? QMAX;
  rline(X(0) - 4, y0, X(1.2) + 4, y0, 960, { c: C.line, w: 1, once: true }); rline(X(0) - 4, y0, X(0) - 4, y1 - 4, 961, { c: C.line, w: 1, once: true });
  txt('0', X(0) - 13, y0, { s: 12, c: C.soft }); txt(String(top), X(0) - 15, y1, { s: 12, c: C.soft });
  if (o.title) txt(o.title, X(0) - 8, y1 - 13, { s: 13, c: C.soft, al: 'left' });
  [.4, .7, 1].forEach((d, i) => { ctx.save(); ctx.globalAlpha = .35; ctx.setLineDash([2, 4]); ctx.strokeStyle = C.line; ctx.beginPath(); ctx.moveTo(X(d), y0); ctx.lineTo(X(d), y1); ctx.stroke(); ctx.restore(); });
  txt('strength (m), same ruler as above', X(.6), y0 + 11, { s: 12, c: C.soft, a: o.xl ?? 1 });
}
function curve(f, col, o = {}) { const y0 = o.y0 ?? CY0, y1 = o.y1 ?? CY1, top = o.top ?? QMAX, Y = q => y0 - clamp(q, -1, top * 1.1) / top * (y0 - y1), pmax = o.p ?? 1;
  ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = o.w || 1.8; ctx.lineJoin = 'round'; if (o.dash) ctx.setLineDash(o.dash); ctx.globalAlpha = o.a ?? 1; ctx.beginPath();
  for (let i = 0; i <= 240 * pmax; i++) { const a = i / 240 * AMAX, x = X(a), y = Y(f(a)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); }
const criticCurve = (w, o = {}) => curve(a => Qw(w, a), '#c99a3a', o);
const trueCurve = (m, o = {}) => curve(a => Qtrue(m, a), '#8a8178', Object.assign({ dash: [5, 4], w: 1.3 }, o));
function actorMark(m, q, o = {}) { const y = CY(q, o.qtop); ctx.save(); ctx.globalAlpha = (o.a ?? 1) * .5; ctx.setLineDash([2, 3]); ctx.strokeStyle = '#3f8582'; ctx.beginPath(); ctx.moveTo(X(m), y); ctx.lineTo(X(m), GY + 92); ctx.stroke(); ctx.restore();
  dot(X(m), y, 4.5, '#3f8582', o.a ?? 1); if (o.label) txt(o.label, X(m) + (o.dx ?? 0), y - 12, { s: 13, c: '#3f8582', a: o.a ?? 1 }); }
function slopeMark(w, m, a = 1, top = QMAX) { if (a <= 0) return; const s = slopeW(w, m), q = Qw(w, m), dx = .08;
  rline(X(m - dx), CY(q - s * dx, top), X(m + dx), CY(q + s * dx, top), 1300, { c: C.rose, w: 2, once: true });
  arrow(X(m) + 6, CY(q, top) - 26, X(m + (s > 0 ? .1 : -.1)), CY(q, top) - 26, 1301, { c: '#3f8582', w: 1.6, head: 6 }); txt('uphill', X(m + (s > 0 ? .13 : -.13)), CY(q, top) - 26, { s: 12, c: '#3f8582', al: s > 0 ? 'left' : 'right' }); }
function legend(items, x, y, a = 1) { items.forEach(([s, col, dash], i) => { const yy = y + i * 15; alpha(a, () => { ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 1.8; if (dash) ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(x, yy); ctx.lineTo(x + 16, yy); ctx.stroke(); ctx.restore(); });
  txt(s, x + 20, yy, { s: 12, c: C.soft, al: 'left', a }); }); }

// gap chart: critic's score of its own jump minus true worth, average of 8 runs, jumps 4000 to 12000
const GX0 = 70, GX1 = 360, GT0 = GW0, GLO = -1, GHI = 1, GYB = 392, GYT = 306;
const GXt = t => GX0 + (t - GT0) / (P.T - GT0) * (GX1 - GX0), GYv = v => GYB - (clamp(v, GLO, GHI) - GLO) / (GHI - GLO) * (GYB - GYT);
function gapFrame(a = 1) { if (a <= 0) return; alpha(a, () => {
  rline(GX0 - 4, GYB, GX0 - 4, GYT - 2, 970, { c: C.line, w: 1, once: true });
  ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.ink; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(GX0, GYv(0)); ctx.lineTo(GX1, GYv(0)); ctx.stroke(); ctx.restore();
  txt('+1', GX0 - 16, GYv(1), { s: 12, c: C.soft }); txt('0', GX0 - 14, GYv(0), { s: 12, c: C.soft }); txt('−1', GX0 - 16, GYv(-1), { s: 12, c: C.soft });
  txt('too optimistic', GX1, GYT - 12, { s: 12, c: C.roseInk, al: 'right' }); txt('too pessimistic', GX1 - 70, GYB + 10, { s: 12, c: '#4f74a0', al: 'right' });
  txt('jump 4000', GX0, GYB + 10, { s: 12, c: C.soft, al: 'left' }); txt('jump 12000', GX1, GYB + 10, { s: 12, c: C.soft, al: 'right' });
  txt('critic’s score minus true worth', GX0 - 8, GYT - 12, { s: 13, c: C.soft, al: 'left' }); }); }
function gapLine(arr, col, o = {}) { const i0 = GT0 / P.every - 1, n = arr.length, last = Math.round(lerp(i0, n - 1, o.p ?? 1)); ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = o.w || 1.8; ctx.globalAlpha = o.a ?? 1; ctx.beginPath();
  for (let i = i0; i <= last; i++) { const x = GXt((i + 1) * P.every), y = GYv(arr[i]); i > i0 ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); }
const CCOL = ['#c4581f', '#4f74a0', '#8a6fb0', '#3f8582'];
// a label on a small paper patch, so chart lines never run through it
function tag(s, x, y, col, al = 'center', sz = 13) { ctx.save(); ctx.font = `${sz}px 'Patrick Hand', cursive`; const w = ctx.measureText(s).width; ctx.restore(); const x0 = al === 'center' ? x - w / 2 : al === 'right' ? x - w : x; box(x0 - 3, y - 8, w + 6, 16, C.bg, .92); txt(s, x, y, { s: sz, c: col, al }); }
function waitNote(c) { if (ready(c)) return false; txt('simulating 8 runs…', W / 2, 350, { s: 15, c: C.soft }); return true; }

const STEPS = [
 { h: 'How hard to jump?', p: ['Back to the jump from the SAC reel. Pip stands on a ledge and picks a **jump strength**: how many metres it flies. Any number will do: 0.5, 0.71, 0.7034.', 'The cheese lies 0.7 metres out. Landing right on it pays 1; a bit short or long pays less, following the curve below. The reward is paid on landing.'],
   shot: [4, 11],
   draw(t) { ledges(t); const demo = [{ a: .5, l: .5 }, { a: .82, l: .82 }, { a: .7, l: .7 }], k = clamp(Math.floor((t - 1) / 2.2), 0, 2), u = (t - 1 - k * 2.2) / 1.3;
     const p = t < 1 ? jumpPos(0, 0, 0, 0) : jumpPos(demo[k].a, demo[k].l, 0, u); pipAt(p, t);
     if (t > 1 && u > 1) txt(`${demo[k].a} m: reward ${f2(rew(demo[k].l))}`, X(demo[k].l), GY - 44, { s: 15, c: C.goldInk, a: clamp((u - 1) * 3) });
     const a = clamp(t - 7.6); if (a > 0) { chartFrame({ top: 1, title: 'reward for each strength', xl: a }); curve(rew, C.gold, { top: 1, p: ease((t - 7.6) / 1.5), w: 2 });
       demo.forEach((d, i) => { if (i <= k) dot(X(d.a), CY0 - rew(d.a) * (CY0 - CY1), 3.5, C.goldInk, a); }); } } },
 { h: 'A row of ledges', p: ['In this reel the cheese ledge is **narrow**, from 0.4 to 1.0 metres. Land off it and Pip falls into the stream: reward 0, and the run is over. Land on it and Pip walks to its front edge and faces the same picture again: the next crumb, again 0.7 metres out.', 'Two more changes. Wind shifts each landing by a few centimetres. And crumbs vary in size, from nothing to double, 1 on average, so the reward is the curve’s value times the crumb. Real rewards are noisy too. γ (gamma) is 0.9.'],
   shot: [4, 11],
   draw(t) { ledges(t); const js = [{ a: .68, l: .7, d: 0 }, { a: 1.12, l: 1.1, d: 1 }], k = t < 4.5 ? 0 : 1, u = (t - (k ? 4.5 : 1)) / 1.6;
     pipAt(t < 1 ? jumpPos(0, 0, 0, 0) : oneJump(js[k], t, k ? 4.5 : 1, 1.6), t);
     if (k === 0 && u > 1) txt('landed: next gap, next crumb', X(.7), GY - 46, { s: 14, c: C.goldInk, a: clamp((u - 1) * 2) * (1 - clamp(t - 4)) });
     if (k === 1 && u > 1) txt('too far: splash, run over', X(.95), GY - 46, { s: 14, c: C.roseInk, a: clamp((u - 1) * 2) });
     if (t > 2) txt('next ledge →', 384, GY - 12, { s: 12, c: C.soft, al: 'right', a: clamp(t - 2) });
     if (t > 7) card(['land within 0.3 m of the crumb: reward, jump again', { s: 'land off the ledge: reward 0, run over', c: C.roseInk }, { s: 'wind: landing shifts by about 0.05 m', c: C.soft }, { s: 'crumb size: 0 to 2, average 1 · γ = 0.9', c: C.soft }], t - 7, { gap: 1, lh: 22, fs: 16 }); } },
 { h: 'No menu to pick from', p: ['DQN kept one Q-value per move and took the **biggest**: up, right, down or left, four numbers to compare. Here the move is a number, and there are endlessly many of them. Taking the max means searching them all, at every jump.', 'Chop strengths into steps? A robot arm with 7 joints and 10 settings each has 10 million combinations to compare, every single step.'],
   shot: [3, 10],
   draw(t) { const names = ['up', 'right', 'down', 'left'], qs = [.31, .52, .18, .27];
     if (t < 4.8) alpha(1, () => { const fa = clamp(t - .3) * (1 - clamp((t - 4) * 1.5)); if (fa <= 0) return; ctx.globalAlpha = fa; txt('DQN: one score per move', W / 2, 40, { s: 17, a: fa });
       qs.forEach((q, i) => { const x = 90 + i * 74, h = q * 260; box(x - 18, 210 - h, 36, h, i === 1 ? C.gold : C.yellow, fa); rrect(x - 18, 210 - h, 36, h, 1400 + i, { c: C.line, w: 1 }); txt(names[i], x, 224, { s: 14, c: C.soft, a: fa }); txt(f2(q), x, 200 - h, { s: 14, c: i === 1 ? C.goldInk : C.soft, a: fa }); });
       if (t > 2) txt('take the biggest: right', W / 2, 252, { s: 16, c: C.goldInk, a: Math.min(fa, clamp(t - 2)) }); });
     if (t > 5) { ledges(t, { cheese: 1 }); pipAt(jumpPos(0, 0, 0, 0), t);
       for (let i = 0; i <= 60; i++) { const d = i / 50; if (d > 1.2) break; rline(X(d), GY - 26, X(d), GY - 26 - (i % 5 ? 8 : 16), 1500 + i, { c: C.soft, w: 1, once: true, j: .2, p: clamp((t - 5) * 2 - i / 60) }); }
       txt('0.7?  0.71?  0.705?  0.7049?  …', X(.6), GY - 70, { s: 16, c: C.roseInk, a: clamp(t - 6.5) }); }
     if (t > 8) card(['4 moves: compare 4 numbers', { s: 'a strength: endlessly many to compare', c: C.roseInk }, { s: '7 joints × 10 settings: 10 million', c: C.roseInk }], t - 8, { gap: 1.1 }); } },
 { h: 'A critic for any strength', p: [`The **critic** scores any strength: **Q(s, a)**, the reward to expect from jumping a now and the actor’s clean answer ever after, each later jump multiplied by another 0.9. Here the situation s is always a gap’s edge.`, `Dashed: the true Q once the actor answers 0.7, from the rules. Jumping 0.7 is then worth ${f2(PERFECT)}; past the cliffs at 0.4 and 1.0 Pip falls. Pip does not have this curve: the critic must learn it.`],
   shot: [4, 11],
   draw(t) { ledges(t); pipAt(jumpPos(0, 0, 0, 0), t); chartFrame({ title: 'score Q for each strength' }); trueCurve(GAP, { p: ease((t - 1) / 2.5) });
     if (t > 4) { const y = CY(PERFECT); dot(X(GAP), y, 3.5, C.ink); txt(`${f2(PERFECT)} at 0.7 m`, X(GAP) + 8, y - 10, { s: 13, al: 'left', a: clamp(t - 4) }); }
     if (t > 6) { txt('falls', X(.3), CY(1.4), { s: 13, c: C.roseInk, a: clamp(t - 6) }); txt('falls', X(1.1), CY(1.4), { s: 13, c: C.roseInk, a: clamp(t - 6) }); }
     if (t > 2) card([`jumping 0.7: ${R07.toFixed(3)} per jump on average (wind)`, `${R07.toFixed(3)} + 0.9 × ${R07.toFixed(3)} + 0.81 × … = ${R07.toFixed(3)} ÷ 0.1 ≈ ${f2(PERFECT)}`], t - 8.5, { y: 14, w: 340, lh: 22, fs: 15 }); } },
 { h: 'An actor that just answers', p: ['The **actor** looks at the situation and answers with one strength. No chances, no dice: same situation, same answer. That is a **deterministic** actor.', 'Earlier reels needed chances so that a small nudge made a small change; a strength is a smooth number, so nudging it already does that. In real tasks the actor is a network, situation in, action out; here it is one number, starting at 0.3 metres. Pip falls.'],
   shot: [3, 9],
   draw(t) { ledges(t); const j = { a: .3, l: .3, d: 1 };
     const p = t < 3 ? jumpPos(0, 0, 0, 0) : oneJump(j, t, 3, 1.8); pipAt(p, t);
     alpha(clamp(t - .5), () => { rrect(120, 30, 160, 64, 1600, { fill: C.card }); txt('actor', 200, 46, { s: 15, c: '#3f8582' }); txt('gap’s edge  →  0.3 m', 200, 72, { s: 17 }); });
     if (t > 5) { bubble('reward 0', X(.3), GY + 20, t - 5, C.roseInk); txt('0.3 m: lands 0.4 m short', X(.35), GY - 60, { s: 14, c: C.roseInk, a: clamp(t - 5.3) }); }
     if (t > 6.5) card(['deterministic: same situation, same answer', { s: 'real tasks: a network, situation in, action out', c: C.soft }, { s: 'here: one number, starting at 0.3 m', c: '#3f8582' }], t - 6.5, { gap: 1.2 }); } },
 { h: 'Teaching the critic', p: ['The critic learns like DQN: **target** = reward + 0.9 × its score for the next jump. DQN used the best next move; here the critic asks the actor what it would jump next. A fall ends the run: the target is just the reward, 0.',
   `Jump ${CAP.t}: Pip jumps ${f2(CAP.a)} and lands at ${f2(CAP.l)}: reward ${f2(CAP.curve)} × crumb ${f2(CAP.crumb)} = ${f2(CAP.r)}. Next the actor would jump ${f2(CAP.mt)}, scored ${f2(CAP.qn)}. Target ${f2(CAPY)}, above the critic’s ${f2(CAP.q)}, so its curve is nudged up there.`],
   shot: [4, 12],
   draw(t) { ledges(t); const j = { a: CAP.a, l: CAP.l, d: CAP.d }, p = t < .8 ? jumpPos(0, 0, 0, 0) : oneJump(j, t, .8, 1.5); pipAt(p, t);
     if (t > 2.4) bubble(`+${f2(CAP.r)}`, X(CAP.l), GY - 30, t - 2.4);
     card([`reward ${f2(CAP.r)}`, `+ 0.9 × score of the actor’s next jump (${f2(CAP.mt)}): ${f2(CAP.qn)}`, { s: `target  =  ${f2(CAP.r)} + 0.9 × ${f2(CAP.qn)}  =  ${f2(CAPY)}`, c: C.goldInk }, { s: `critic now says ${f2(CAP.q)}: nudge up`, c: C.soft }, { s: 'fell?  target = reward, 0', c: C.roseInk }], t - 3, { gap: 1.4, lh: 22, fs: 16 }); } },
 { h: 'Try a little around it', p: ['A deterministic actor always gives the same answer, so the critic would only ever see one strength and could not tell whether a harder or softer jump is better. So while training, every jump gets random **noise** added: a bell curve with a spread of 0.1 metres around the actor’s answer.', 'DDPG explored with noise like this too. The noise is only for the jumps Pip makes: the target’s next jump uses the actor’s clean answer.'],
   shot: [3, 9],
   draw(t) { ledges(t); const m = .45, bell = (d, a = 1) => { ctx.save(); ctx.globalAlpha = a; ctx.fillStyle = 'rgba(111,183,180,.3)'; ctx.strokeStyle = '#3f8582'; ctx.lineWidth = 1.6; ctx.beginPath();
       for (let i = 0; i <= 120; i++) { const x = i / 100, y = GY - 24 - 60 * Math.exp(-(((x - m) / .1) ** 2) / 2); i ? ctx.lineTo(X(x), y) : ctx.moveTo(X(x), y); } ctx.stroke(); ctx.restore(); };
     bell(0, clamp(t - .5)); txt('actor 0.45', X(m), GY - 94, { s: 14, c: '#3f8582', a: clamp(t - .5) });
     const R = rng(42), tries = Array.from({ length: 9 }, () => clamp(m + .1 * xgauss(R), 0, 1.2));
     tries.forEach((a, i) => { const u = clamp(t - 2 - i * .6); if (u > 0) { dot(X(a), GY - 6, 3, C.orange, u); } });
     pipAt(jumpPos(0, 0, 0, 0), t);
     if (t > 7.5) card(['training jump = actor’s answer + noise', { s: 'noise: bell curve, spread 0.1 m', c: C.soft }], t - 7.5, { gap: 1.2 }); } },
 { h: 'Climb the critic’s slope', p: ['How should the actor improve without trying every strength? It asks the critic: if I jumped a little harder, would your score go up? That is the **slope** of the critic’s curve at the actor’s strength.', `Then it steps uphill: new strength = old + 0.01 × slope. At jump ${SNAP.t} of Pip’s first run, the actor says ${f2(SNAP.m)}, the critic’s slope there is ${f2(SNAPS)}, so the actor moves to ${f2(SNAPN)}.`],
   shot: [4, 11],
   draw(t) { ledges(t); pipAt(jumpPos(0, 0, 0, 0), t); chartFrame({ title: `the critic’s curve at jump ${SNAP.t} (note the scale)`, top: SNAPTOP }); criticCurve(SNAP.w, { p: ease((t - .5) / 2), top: SNAPTOP });
     const m = t > 7.5 ? lerp(SNAP.m, SNAPN, ease((t - 7.5) / 1)) : SNAP.m;
     if (t > 2.5) actorMark(m, Qw(SNAP.w, m), { a: clamp(t - 2.5), label: `actor ${f2(m)}`, dx: -34, qtop: SNAPTOP });
     if (t > 4) slopeMark(SNAP.w, SNAP.m, clamp(t - 4), SNAPTOP);
     legend([['critic', '#c99a3a']], 300, 300, clamp(t - 1));
     if (t > 5) card([`slope at ${f2(SNAP.m)}:  ${f2(SNAPS)}  (rose line)`, `${f2(SNAP.m)} + 0.01 × ${f2(SNAPS)}  =  ${f2(SNAPN)}`], t - 5, { y: 14, lh: 22, fs: 15, gap: 1.3 }); } },
 { h: 'Many knobs', p: ['In a real task the actor is a network with many knobs. The critic’s slope says how much the score rises per extra metre of strength. For each knob, ask too how much the strength moves when that knob turns a little. Multiply the two.', 'That product, the chain rule, says how much turning each knob raises the score; backpropagation computes it, and every knob moves a small step times it. So it is still a policy gradient. Here the actor is one number, so the second factor is 1.'],
   shot: [4, 11],
   draw(t) { box(0, 0, W, 285, C.bg); const bx = [[18, 'situation', C.card], [108, 'actor', '#dbecea'], [208, 'strength', C.card], [298, 'critic', '#f6e7c8']], y = 70;
     bx.forEach(([x, l, f], i) => { if (t < .3 + i * .5) return; rrect(x, y, 80, 40, 2200 + i, { fill: f }); txt(l, x + 40, y + 20, { s: 15 }); if (i) arrow(x - 12, y + 20, x - 2, y + 20, 2210 + i, { c: C.soft }); });
     if (t > 2.3) { arrow(378, y + 20, 392, y + 20, 2215, { c: C.soft }); txt('score', 384, y - 8, { s: 13, c: C.soft }); }
     if (t > 1) for (let k = 0; k < 4; k++) { const kx = 120 + k * 18; rcircle(kx, y + 56, 6, 2220 + k, { fill: C.grey }); rline(kx, y + 56, kx + 4 * Math.cos(t + k), y + 56 - 4 * Math.sin(t + k), 2230 + k, { w: 1.4, once: true }); }
     if (t > 1) txt('knobs', 148, y + 76, { s: 13, c: C.soft });
     if (t > 4) { arrow(338, y + 48, 258, y + 48, 2240, { c: C.rose, w: 1.6 }); txt('score per metre: the critic’s slope', 330, y + 106, { s: 13, c: C.roseInk, al: 'right' }); }
     if (t > 6) { arrow(240, y + 60, 172, y + 60, 2241, { c: '#3f8582', w: 1.6 }); txt('metres per turn of a knob', 228, y + 124, { s: 13, c: '#3f8582', al: 'right' }); }
     if (t > 8) card(['score per turn of a knob  =', '     (score per metre) × (metres per turn)', { s: 'nudge each knob by step × that', c: C.goldInk }, { s: 'here: one knob, the strength itself, × 1', c: C.soft }], t - 8, { y: 220, gap: 1.1, fs: 16, lh: 22 }); } },
 { h: 'Borrowed from DQN', p: ['Two tricks from the Deep RL reels keep this steady. A **replay buffer**: every jump is kept, and after each jump the critic studies 8 random old ones, not just the latest.', '**Target networks**: the score of the next jump comes from slow copies, moved 2 percent toward the real ones after each jump, so targets do not shift under the critic’s feet. The actor gets a slow copy too, because the target asks it for the next jump.'],
   shot: [3, 10],
   draw(t) { const R = rng(9); box(0, 0, W, 285, C.bg);
     alpha(clamp(t - .3), () => { txt('replay buffer: every jump so far', 110, 34, { s: 15, c: C.goldInk });
       for (let i = 0; i < 40; i++) { const x = 30 + (i % 10) * 17, y = 54 + Math.floor(i / 10) * 20, pick = [3, 11, 17, 22, 26, 31, 35, 38].includes(i) && t > 2.5; rrect(x, y, 13, 15, 1700 + i, { fill: pick ? C.gold : C.card, c: C.line, w: 1 }); }
       if (t > 2.5) txt('8 random ones per lesson', 110, 148, { s: 14, c: C.soft, a: clamp(t - 2.5) }); });
     alpha(clamp(t - 4.5), () => { txt('target copies', 300, 34, { s: 15, c: '#3f8582' });
       const u = (t % 3) / 3; rrect(240, 60, 52, 40, 1800, { fill: '#f6e7c8' }); txt('critic', 266, 80, { s: 14 }); rrect(312, 60, 52, 40, 1801, { fill: '#e8efe9' }); txt('copy', 338, 80, { s: 14 });
       arrow(294, 80, 310, 80, 1802, { c: C.soft }); txt('2 percent per jump', 302, 116, { s: 13, c: C.soft });
       rrect(240, 136, 52, 40, 1803, { fill: '#dbecea' }); txt('actor', 266, 156, { s: 14 }); rrect(312, 136, 52, 40, 1804, { fill: '#e8efe9' }); txt('copy', 338, 156, { s: 14 }); arrow(294, 156, 310, 156, 1805, { c: C.soft }); });
     if (t > 7) card(['replay: learn from 8 random old jumps', { s: 'target copies: slow, steady targets', c: '#3f8582' }, { s: 'noise on the jump: keeps exploring', c: C.soft }], t - 7, { y: 200, gap: 1.1 }); } },
 { h: 'DDPG', p: [`Actor climbing a critic, plus replay, target copies and noise: that is **DDPG**, Deep Deterministic Policy Gradient (Lillicrap and colleagues, 2015). Here is Pip’s first run, sped up.`, `Of the first 50 jumps, ${F50} end in the stream. By jump 100 the actor says ${f2(MAT(100))}, and it stays near 0.7. In jumps 100 to 400 Pip falls only ${FALLS_AFTER === 1 ? 'once' : FALLS_AFTER + ' times'}: the exploring noise can still send it off the ledge.`],
   shot: [3, 8, 13],
   draw(t) { ledges(t); const D = .6, nf = t < 6 ? t / D : Math.min(399.99, 10 + (t - 6) / 6 * 390), n = Math.floor(nf), j = SJ[n], u = nf - n; pipAt(t > 12.2 ? jumpPos(0, 0, 0, 0) : jumpPos(j.a, j.l, j.d, u * 1.1), t);
     if (t > 6 && t < 12.2) txt('sped up', X(.15), GY - 70, { s: 14, c: C.soft });
     const li = clamp(Math.floor((n + 1) / P.every) - 1, 0, SL.length - 1), lg = SL[li];
     chartFrame({ title: 'critic (gold) and actor (dot)' }); if (n >= 49) { criticCurve(lg.w); actorMark(lg.m, Qw(lg.w, lg.m), { label: f2(lg.m) }); }
     const falls = SJ.slice(0, n + 1).filter(q => q.d).length; txt(`jump ${n + 1}   falls so far ${falls}`, 384, 22, { s: 15, al: 'right' });
     txt('each rose dot is a fall; a row is 100 jumps', 16, 44, { s: 12, c: C.soft, al: 'left' });
     for (let i = 0; i <= n; i++) if (SJ[i].d) dot(18 + (i % 100) * 3.6, 60 + Math.floor(i / 100) * 9, 1.8, C.rose); } },
 { h: 'The critic catches up', p: [`Gold: the critic’s score for the actor’s own jump, every 50 jumps. Dashed: that jump’s true worth. The actor found 0.7 by about jump ${SL[MT].t}; the critic needs about ${CATCH} jumps: a worth near 9 is built one reward plus 0.9 × an older guess at a time, as in the TD reel, and the slow copy slows it more.`, `Dips: the actor drifted, to ${f2(DIP.m)} at jump ${DIP.t}, so every jump paid less. At jump ${P.T}: critic ${f2(SEND.q)}, truth ${f2(SEND.tv)}.`],
   shot: [4, 11],
   draw(t) { const n = clamp(Math.floor((t - 1) * 50), 0, SL.length), x0 = 70, x1 = 360, y0 = 260, y1 = 60, Xi = i => x0 + i / (SL.length - 1) * (x1 - x0), Yq = q => y0 - q / 10 * (y0 - y1);
     rline(x0 - 4, y0, x1 + 4, y0, 980, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1 - 4, 981, { c: C.line, w: 1, once: true });
     txt('0', x0 - 14, y0, { s: 12, c: C.soft }); txt('10', x0 - 16, y1, { s: 12, c: C.soft }); txt('jump 50', x0, y0 + 12, { s: 12, c: C.soft, al: 'left' }); txt('jump ' + P.T, x1, y0 + 12, { s: 12, c: C.soft, al: 'right' });
     txt('score of the actor’s own jump', x0 - 8, y1 - 18, { s: 13, c: C.soft, al: 'left' });
     ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = C.soft; ctx.lineWidth = 1.3; ctx.beginPath(); for (let i = 0; i < n; i++) { const x = Xi(i), y = Yq(SL[i].tv); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore();
     ctx.save(); ctx.strokeStyle = '#c99a3a'; ctx.lineWidth = 1.8; ctx.beginPath(); for (let i = 0; i < n; i++) { const x = Xi(i), y = Yq(SL[i].q); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore();
     legend([['critic’s score', '#c99a3a'], ['true worth', C.soft, 1]], 250, 200, clamp(t - 1));
     if (n >= SL.length) card([`after ${P.T} jumps:  critic ${f2(SEND.q)},  truth ${f2(SEND.tv)}`, { s: `actor ${f2(SEND.m)}`, c: '#3f8582' }], t - 5.5, { y: 306, gap: 1.2 }); } },
 { h: 'Too optimistic', p: [`From jump ${GW0} on, this critic is too high by ${f2(STAVG)} on average: it believes in reward that will never come. Close-up at jump ${SN2.t}: critic ${f2(SN2.q)}, truth ${f2(SN2.tv)}. Why? The actor goes where the critic’s curve is highest, including spots that are too high by the luck of the samples. Every target leans on that inflated score, so the error is copied back, jump after jump.`, 'The Double DQN reel saw the same thing: taking the max picks lucky guesses.'],
   shot: [4, 11],
   draw(t) { const m = SN2.m, sx = d => 40 + (d - .45) / .5 * 320, top = 10.2, bot = 7.6, sy = q => 250 - (q - bot) / (top - bot) * 200;
     rline(36, 250, 364, 250, 990, { c: C.line, w: 1, once: true }); [.5, .6, .7, .8, .9].forEach((d, i) => txt(d + ' m', sx(d), 262, { s: 12, c: C.soft }));
     txt(`close-up near the top, jump ${SN2.t} of the first run`, 40, 30, { s: 13, c: C.soft, al: 'left' }); txt(String(bot), 26, 250, { s: 12, c: C.soft }); txt('10', 26, sy(10), { s: 12, c: C.soft });
     const draw = (f, col, dash, p) => { ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 1.8; if (dash) ctx.setLineDash([5, 4]); ctx.beginPath(); for (let i = 0; i <= 200 * p; i++) { const d = .45 + i / 200 * .5, y = sy(clamp(f(d), bot - .2, top)); i ? ctx.lineTo(sx(d), y) : ctx.moveTo(sx(d), y); } ctx.stroke(); ctx.restore(); };
     draw(a => Qtrue(m, a), C.soft, 1, ease((t - .5) / 1.5)); draw(a => Qw(SN2.w, a), '#c99a3a', 0, ease((t - 2) / 1.5));
     if (t > 4) { const q = Qw(SN2.w, m), tv = Qtrue(m, m); dot(sx(m), sy(q), 4.5, '#3f8582'); dot(sx(m), sy(tv), 3.5, C.soft); rline(sx(m), sy(q), sx(m), sy(tv), 991, { c: C.rose, w: 2, once: true });
       txt(`critic ${f2(q)}`, sx(m) + 10, sy(q) - 10, { s: 14, c: C.goldInk, al: 'left', a: clamp(t - 4) }); txt(`true ${f2(tv)}`, sx(m) + 10, sy(tv) + 12, { s: 14, c: C.soft, al: 'left', a: clamp(t - 4) }); }
     legend([['critic', '#c99a3a'], ['true Q', C.soft, 1]], 270, 200, clamp(t - 1));
     if (t > 6) card(['actor sits where the critic is highest', { s: 'targets lean on that score: the error is copied back', c: C.roseInk }], t - 6, { gap: 1.4 }); } },
 { h: 'Not just bad luck', get p() { return [`Is that one unlucky run? Run DDPG 8 times with different dice. The chart shows the critic’s score of its own jump minus the true worth, averaged over the 8 runs, from jump ${GW0} to ${P.T}, once the critics have caught up. Above the dashed zero line the critic is too optimistic, below too pessimistic.`, `DDPG stays above zero: over those jumps it is too high by ${f2(agg(0).mean)} on average, and too high in ${agg(0).pos} of the 8 runs.`]; },
   shot: [4, 11],
   draw(t) { ledges(t, { next: 0 }); pipAt(jumpPos(0, 0, 0, 0), t); box(0, 290, W, 130, C.bg); if (waitNote(0)) return; const A = agg(0);
     gapFrame(clamp(t - .5)); if (t > 2.5) gapLine(A.gap, CCOL[0], { p: ease((t - 2.5) / 2.5) });
     if (t > 5.5) { A.fin.forEach((v, k) => dot(GX1 + 10, GYv(v), 2.6, CCOL[0], .7)); tag(`DDPG ${sg2(A.mean)}`, 200, GYv(A.mean) - 14, CCOL[0]); }
     card([`DDPG, 8 runs: too high by ${f2(A.mean)} on average`, { s: `too high in ${A.pos} of 8 runs (dots on the right)`, c: C.soft }], t - 7, { y: 14, gap: 1.2, fs: 16 }); } },
 { h: 'TD3, fix one: two critics', get p() { return ['In 2018, Scott Fujimoto, Herke van Hoof and David Meger proposed **TD3**, Twin Delayed DDPG: three fixes. The first: train **two critics**, each studying its own random handful of old jumps, and build one shared target for both from the **smaller** of their two scores. The actor climbs the first critic.', `A lucky high guess is rarely shared by both. Over the same 8 runs, the average gap shrinks from ${f2(agg(0).mean)} to ${f2(agg(1).mean)}.`]; },
   shot: [4, 11],
   draw(t) { box(0, 0, W, 285, C.bg); if (waitNote(1)) return; const A = agg(0), B = agg(1);
     alpha(clamp(t - .3), () => { const z = B.runs[0].log[B.runs[0].log.length - 1], q1 = Qw(z.w, z.m), q2 = Qw(z.w1, z.m), lo = Math.min(q1, q2);
       rrect(60, 40, 110, 60, 1900, { fill: '#f6e7c8' }); txt('critic 1', 115, 60, { s: 15 }); txt(f2(q1), 115, 84, { s: 16, c: C.goldInk });
       rrect(230, 40, 110, 60, 1901, { fill: '#f6e7c8' }); txt('critic 2', 285, 60, { s: 15 }); txt(f2(q2), 285, 84, { s: 16, c: C.goldInk });
       if (t > 2) { arrow(115, 104, 190, 150, 1902, { c: C.soft }); arrow(285, 104, 210, 150, 1903, { c: C.soft }); txt(`target uses the smaller: ${f2(lo)}`, 200, 168, { s: 16, c: '#4f74a0', a: clamp(t - 2) }); txt(`their scores for a ${f2(z.m)} jump, end of a run`, 200, 190, { s: 12, c: C.soft, a: clamp(t - 2) }); } });
     gapFrame(clamp(t - 3)); if (t > 3) gapLine(A.gap, CCOL[0], { a: .45 }); if (t > 4) gapLine(B.gap, CCOL[1], { p: ease((t - 4) / 2) });
     if (t > 6.5) { tag(`DDPG ${sg2(A.mean)}`, 150, GYv(A.mean) - 12, CCOL[0]); tag(`two critics ${sg2(B.mean)}`, 290, GYv(B.mean) + 14, CCOL[1]); } } },
 { h: 'Fix two: let the critic settle', get p() { return ['Second: update the actor and the slow copies only every **second** jump, while the critics learn after every jump. Why? The actor chases the critic’s peak; if the critic is still wobbling, the actor chases the wobble.', `In this small task it changes nothing you can see: the gap is ${f2(agg(2).mean)}${f2(agg(2).mean) === f2(agg(1).mean) ? ', the same as with two critics alone' : `, against ${f2(agg(1).mean)} without the delay, a difference within run-to-run luck`}. The TD3 authors aimed it at large networks, whose critics take longer to settle.`]; },
   shot: [4, 11],
   draw(t) { box(0, 0, W, 285, C.bg); if (waitNote(2)) return; const B = agg(1), D = agg(2);
     const row = (y, lab, every, col) => { txt(lab, 40, y, { s: 15, al: 'left', c: col }); for (let i = 0; i < 10; i++) { const on = i % every === 0, u = clamp(t - .5 - i * .25); if (u > 0) dot(150 + i * 22, y, on ? 6 : 3, on ? col : C.grey, u); } };
     row(70, 'critics learn', 1, C.goldInk); row(110, 'actor + copies', 2, '#3f8582'); txt('jump  1  2  3  …', 150, 140, { s: 13, c: C.soft, al: 'left', a: clamp(t - 3) });
     if (t > 3.5) txt('two rounds of critic lessons per actor step', W / 2, 180, { s: 15, c: C.soft, a: clamp(t - 3.5) });
     gapFrame(clamp(t - 4)); if (t > 4) gapLine(B.gap, CCOL[1], { a: .45 }); if (t > 5) gapLine(D.gap, CCOL[2], { p: ease((t - 5) / 2) });
     if (t > 7.5) { tag(`two critics ${sg2(B.mean)}`, 150, GYv(B.mean) - 14, CCOL[1]); tag(`+ delay ${sg2(D.mean)}`, 300, GYv(D.mean) + 16, CCOL[2]); } } },
 { h: 'Fix three: blur the target', get p() { return [`Third: when scoring the next jump for a target, jiggle the actor’s strength a little, a spread of ${P.SM} metres, at most ${2 * P.SM}, so a narrow false peak in the critic cannot pull the target up.`, `All three fixes read ${f2(agg(3).mean)}, but that is mostly built in: with jiggled next jumps the critic values a slightly clumsier jumper, truly worth ${f2(agg(3).sdiff)} less. Against that worth, TD3 is off by ${Math.abs(agg(3).meanJ) < .005 ? 'less than 0.01' : sg2(agg(3).meanJ)} (dashed line). The TD3 authors argued a small underestimate is safer anyway.`]; },
   shot: [4, 11],
   draw(t) { ledges(t, { next: 0 }); if (waitNote(3)) return; const D = agg(2), T3 = agg(3), m = SEND.m;
     box(0, 0, W, 150, C.bg);
     alpha(clamp(t - .3), () => { txt('next jump scored at the actor’s strength…', W / 2, 30, { s: 15 }); });
     const R = rng(5); for (let i = 0; i < 12; i++) { const u = clamp(t - 1.5 - i * .25); if (u <= 0) continue; const a = clip(m + clip(P.SM * xgauss(R), -2 * P.SM, 2 * P.SM), 0, 1.2); dot(X(a), GY - 8, 3, '#8a6fb0', u); }
     if (t > 1.5) txt('…plus a small jiggle (purple dots)', W / 2, 52, { s: 15, c: '#8a6fb0', a: clamp(t - 1.5) });
     pipAt(jumpPos(0, 0, 0, 0), t); box(0, 290, W, 130, C.bg);
     gapFrame(clamp(t - 4)); if (t > 4) gapLine(D.gap, CCOL[2], { a: .45 }); if (t > 5) gapLine(T3.gap, CCOL[3], { p: ease((t - 5) / 2) });
     if (t > 8.5) { ctx.save(); ctx.setLineDash([4, 3]); gapLine(T3.gapJ, CCOL[3], { w: 1.4, p: ease((t - 8.5) / 1.5) }); ctx.restore(); }
     if (t > 7.5) { tag(`+ delay ${sg2(D.mean)}`, 150, GYv(D.mean) - 14, CCOL[2]); tag(`TD3 vs clean jumps ${sg2(T3.mean)}`, 270, GYv(T3.mean) + 14, CCOL[3]); }
     if (t > 10) tag(`vs jiggled jumps ${sg2(T3.meanJ)}`, 300, GYv(T3.meanJ) - 14, CCOL[3]); } },
 { h: 'Your turn', ctl: { label: 'fixes', values: [0, 1, 2, 3], def: 3, fmt: v => ['DDPG', '+ 2 critics', '+ delay', 'TD3'][v], onChange: st => {} },
   get p() { return ['Drag to switch TD3’s fixes on one at a time: thin lines are the 8 runs, the thick line their average. For TD3 the gap is measured against the jiggled jumper it really values.', `Each run’s average gap: DDPG ${f2(Math.min(...agg(0).fin))} to ${f2(Math.max(...agg(0).fin))}, two critics ${f2(Math.min(...agg(1).fin))} to ${f2(Math.max(...agg(1).fin))}. Pip’s jump is about equally good in all four: with one hump, false optimism has nowhere to lure the actor. In big tasks it can.`]; },
   shot: [3, 9],
   draw(t, st) { const c = st.ci ?? 3; ledges(t, { next: 0 }); box(0, 290, W, 130, C.bg); if (waitNote(c)) return; const A = agg(c);
     const j = SJ[300 + Math.floor(t / 1.6) % 60], u = (t % 1.6) / 1.4; pipAt(jumpPos(A.m, A.m, 0, u), t);
     txt(`${CFG[c].name}`, W / 2, 24, { s: 18, c: CCOL[c] });
     txt(`gap, jumps ${GW0} to ${P.T}: ${sg2(c === 3 ? A.meanJ : A.mean)} on average${c === 3 ? ' (vs jiggled jumper)' : ''}`, W / 2, 50, { s: 15 }); txt(`actor: ${f2(A.m)} m on average, true worth ${f2(A.tv)}`, W / 2, 72, { s: 14, c: C.soft });
     const J = c === 3; gapFrame(); A.runs.forEach((r, k) => gapLine(r.log.map(x => x.q - (J ? x.tvj : x.tv)), CCOL[c], { a: .25, w: 1 })); gapLine(J ? A.gapJ : A.gap, CCOL[c], { w: 2.4 }); } },
 { h: 'Where it’s used', p: ['DDPG (2015) learned more than 20 simulated physics tasks, like swinging up a pole, grasping, walking and driving, many straight from camera pixels. TD3 (2018) clearly beat it on standard simulated robot benchmarks such as Hopper and HalfCheetah.', 'Today TD3 is a common starting point for continuous control, robot arms included. Soft Actor-Critic, in its own reel, also uses two critics, but with a policy that has spread.'],
   shot: [3, 9],
   draw(t) { ledges(t); const D = 1.4, k = Math.floor(t / D) % 40, j = SJ[340 + k], u = (t % D) / 1.2; pipAt(jumpPos(j.a, j.l, j.d, u), t);
     const R = (x, y, ph) => { rline(x, y, x + 30 * Math.cos(ph), y - 30 * Math.sin(ph), 2000, { w: 3 }); rcircle(x, y, 5, 2001, { fill: C.grey }); const x2 = x + 30 * Math.cos(ph), y2 = y - 30 * Math.sin(ph); rline(x2, y2, x2 + 24 * Math.cos(ph - 1), y2 - 24 * Math.sin(ph - 1), 2002, { w: 3 }); };
     if (t > 1) { const ph = 1 + .3 * Math.sin(t * 1.3), ph2 = ph - 1 + .4 * Math.sin(t * .9); rrect(30, 140, 50, 14, 2003, { fill: C.grey });
       const x1 = 55 + 60 * Math.cos(ph), y1 = 140 - 60 * Math.sin(ph), x2 = x1 + 50 * Math.cos(ph2), y2 = y1 - 50 * Math.sin(ph2);
       rline(55, 140, x1, y1, 2004, { w: 4 }); rline(x1, y1, x2, y2, 2005, { w: 4 }); rcircle(55, 140, 6, 2006, { fill: C.card }); rcircle(x1, y1, 5, 2007, { fill: C.card });
       txt('each joint’s motor: a number,', 150, 70, { s: 15, c: C.soft, al: 'left' }); txt('like Pip’s jump strength', 150, 90, { s: 15, c: C.soft, al: 'left' }); }
     if (t > 3) card(['DDPG, 2015: 20+ simulated physics tasks', 'TD3, 2018: beat it on Hopper, HalfCheetah', { s: 'see also: Soft Actor-Critic', c: '#3f8582' }], t - 3, { gap: 1.2 }); } },
 { h: 'Recap', p: ['When the action is a number, nobody can take a max over all of them. **DDPG** lets a deterministic actor climb the slope of a critic Q(s, a), with noise for exploring, a replay buffer and slow target copies.', 'The actor seeks the critic’s peaks, so the critic drifts too optimistic. **TD3** takes the smaller of two critics, which removed the optimism here, delays the actor and blurs the target action.'],
   shot: [3, 9],
   draw(t) { ledges(t); pipAt(jumpPos(0, 0, 0, 0), t);
     card(['actor: situation → one action, no dice', 'actor step = slope of the critic at its action', 'critic target = reward + γ × Q(next, actor’s action)', { s: 'TD3: smaller of two critics, delay, blur', c: '#3f8582' }], t - .3, { y: 296, gap: 1, fs: 16, lh: 22 }); } },
];

RLStory.register('ddpg', { title: 'Continuous Control: DDPG & TD3', track: 'Policy Gradients', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { P, S, PERFECT, R07, F50, SNAP: { t: SNAP.t, m: SNAP.m, slope: SNAPS, next: SNAPN }, CAP, CAPY, MT: SL[MT]?.t, FALLS_AFTER, CATCH, DIP: { t: DIP.t, m: DIP.m, tv: DIP.tv }, SNAPTOP, M100: MAT(100), SEND: { q: SEND.q, tv: SEND.tv, m: SEND.m }, STAVG, SN2: { t: SN2.t, q: SN2.q, tv: SN2.tv, m: SN2.m }, agg, train, Qtrue, Vtrue } });
})();
