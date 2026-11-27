// Entropy and Soft Actor-Critic as a story: in the door room (pg_world World 1) an actor-critic Pip can lock onto the
// wrong door after an unlucky start and never notice when the world changes. Adding α × entropy ("how spread out the
// chances are") to the score keeps it curious. Then the same idea for a bell curve over jump strengths (World 3).
// Every number comes from the seeded runs below (run k uses rng(900 + k)) or from exact formulas.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
// ===== pg kit (copied from pg.js) =====
const softmax = th => { const m = Math.max(...th), e = th.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
const DCOL = ['#8fb3d9', '#ef8f5a', '#6fb7b4'], DINK = ['#4f7fb0', '#c4581f', '#3f7f7c'], DNAME = ['A', 'B', 'C'], DX = [90, 200, 310];
function pie(x, y, r, pr, o = {}) {
  let a0 = -Math.PI / 2; pr.forEach((p, i) => { const a1 = a0 + p * Math.PI * 2; ctx.save(); ctx.globalAlpha = o.a ?? 1; ctx.fillStyle = (o.cols || DCOL)[i];
    ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, r, a0, a1); ctx.closePath(); ctx.fill(); ctx.restore();
    if (o.labels && p > .06) { const am = (a0 + a1) / 2; txt((o.names || DNAME)[i], x + Math.cos(am) * r * .62, y + Math.sin(am) * r * .62, { s: 13, a: o.a ?? 1 }); } a0 = a1; });
  rcircle(x, y, r, 880, { j: .6 });
}
// ===== end pg kit =====

// engine-independent maths for the simulation (Math.exp / log may differ in the last bit between browsers)
const LN2 = 0.6931471805599453;
function xexp(x) { if (x < -700) return 0; const k = Math.round(x / LN2), r = x - k * LN2; let term = 1, sum = 1; for (let n = 1; n < 18; n++) { term = term * r / n; sum += term; }
  if (k > 0) for (let i = 0; i < k; i++) sum *= 2; else for (let i = 0; i < -k; i++) sum /= 2; return sum; }
function xlog(x) { let m = x, e = 0; while (m < 0.7071067811865476) { m *= 2; e--; } while (m > 1.4142135623730951) { m /= 2; e++; }
  const y = (m - 1) / (m + 1), y2 = y * y; let p = y, sum = 0; for (let n = 1; n < 40; n += 2) { sum += p / n; p *= y2; } return 2 * sum + e * LN2; }
const xsoftmax = th => { const m = Math.max(...th), e = th.map(x => xexp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
// ───── the door room: crumb chances A 0.2, B 0.5, C 0.8; from pick 401 on, A becomes 0.95 (step 11 says so)
const P0 = [.2, .5, .8], P1 = [.95, .5, .8], TC = 400, T = 1000, MEM = 200, NB = 8, CLR = .1, ETA = 3;
const ALS = [0, .02, .05, .1, .2, .5], AL = .2, NR = 200;
const lg = p => xlog(Math.max(p, 1e-300));
// one run of soft actor-critic Pip. Critic: a score per door, nudged a tenth toward the crumb (1 or 0) of 8 random
// memories after every pick (memory = last 200 picks). Actor: worth = score + α × surprise (−log chance);
// pref += 3 × chance × (worth − the pie's average worth), for all three doors at once.
function makeRun(al, k, keep) {
  const R = rng(900 + k); let th = [0, 0, 0]; const Q = [0, 0, 0], mem = [], hist = []; let t = 0, r1 = 0, r2 = 0, r1b = 0;
  const run = { done: false, hist };
  run.adv = n => { for (const end = Math.min(T, t + n); t < end; t++) {
    const P = t < TC ? P0 : P1, p = xsoftmax(th); let u = R(), d = 0; while (d < 2 && u > p[d]) { u -= p[d]; d++; }
    const r = R() < P[d] ? 1 : 0; if (t < TC) { r1 += r; if (t >= 200) r1b += r; } else if (t >= T - 300) r2 += r;
    mem.push([d, r]); if (mem.length > MEM) mem.shift();
    for (let b = 0; b < NB; b++) { const [dd, rr] = mem[Math.floor(R() * mem.length)]; Q[dd] += CLR * (rr - Q[dd]); }
    const A = Q.map((q, i) => q - al * lg(p[i])), Ab = A.reduce((s, a, i) => s + p[i] * a, 0); th = th.map((v, i) => v + ETA * p[i] * (A[i] - Ab));
    if (keep) hist.push({ d, r, p, Q: Q.slice(), pn: xsoftmax(th) });
    if (t === TC - 1) { run.p400 = xsoftmax(th); run.Q400 = Q.slice(); }
  } run.done = t >= T; if (run.done) { run.p = xsoftmax(th); run.Q = Q.slice(); run.r1 = r1 / TC; run.r1b = r1b / 200; run.r2 = r2 / 300; } return run; };
  return run;
}
// 200 runs for each temperature, computed in the background in small slices
const RUNS = ALS.map(al => Array.from({ length: NR }, (_, k) => makeRun(al, k, false))), QUEUE = [];
[0, 4, 1, 2, 3, 5].forEach(i => RUNS[i].forEach(r => QUEUE.push(r)));
(function pump() { if (typeof setTimeout === 'undefined') return; let n = 0; while (QUEUE.length && n < 12) { const r = QUEUE.shift(); if (!r.done) r.adv(Infinity); n++; } if (QUEUE.length) setTimeout(pump, 10); })();
const ready = i => RUNS[i].every(r => r.done), memo = {};
const best = p => p.indexOf(Math.max(...p));
function agg(i) { if (memo[i]) return memo[i]; const rs = RUNS[i].map(r => r.done ? r : r.adv(Infinity));
  return (memo[i] = { stuck: rs.filter(r => best(r.p400) !== 2).length, foundA: rs.filter(r => best(r.p) === 0).length,
    r1: rs.reduce((s, r) => s + r.r1, 0) / NR, r1b: rs.reduce((s, r) => s + r.r1b, 0) / NR, r2: rs.reduce((s, r) => s + r.r2, 0) / NR, pc: rs.reduce((s, r) => s + r.p400[2], 0) / NR, pa: rs.reduce((s, r) => s + r.p[0], 0) / NR }); }

// ───── the story run (same dice for both Pips; the dice stream drifts apart once their pies differ)
const SK = 32, G0 = makeRun(0, SK, true).adv(Infinity), G2 = makeRun(AL, SK, true).adv(Infinity), H0 = G0.hist, H2 = G2.hist;
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−'), pct = v => Math.round(v * 100);
const cnt = (H, d, a, b) => H.slice(a, b).filter(h => h.d === d).length, crumbs = (H, d, a, b) => H.slice(a, b).filter(h => h.d === d && h.r).length;
const C30 = [0, 1, 2].map(d => cnt(H0, d, 0, 30)), K30 = [0, 1, 2].map(d => crumbs(H0, d, 0, 30));
const CLATE = cnt(H0, 2, 100, TC), CEARLY = cnt(H0, 2, 0, 100);
const NA0 = cnt(H0, 0, TC, T), NA2 = cnt(H2, 0, TC, T);
const SWITCH2 = H2.findIndex((h, i) => i >= TC && best(h.pn) === 0) + 1;            // pick after which A leads the pie
const ent = p => -p.reduce((s, q) => s + (q > 0 ? q * Math.log(q) : 0), 0);
const soft = (Q, a) => a <= 0 ? Q.map((q, i) => i === best(Q) ? 1 : 0) : softmax(Q.map(q => q / a));   // best pie for temperature a
const PIES = [[1 / 3, 1 / 3, 1 / 3], [.1, .1, .8], [0, 0, 1]];
// ───── continuous: a bell curve of jump strengths centred on the cheese (World 3), spread σ
const WR = .15, ER = s => 1 / Math.sqrt(1 + 2 * s * s / (WR * WR)), HB = s => .5 * Math.log(2 * Math.PI * Math.E * s * s);
const sigStar = a => { if (a <= 0) return 0; let lo = 1e-4, hi = 2; for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2, d = -(2 * m / (WR * WR)) * Math.pow(1 + 2 * m * m / (WR * WR), -1.5) + a / m; d > 0 ? lo = m : hi = m; } return (lo + hi) / 2; };
const BAL = [.01, .03, .1], BSIG = BAL.map(sigStar);

// ───── drawing
function door(i, o = {}) { const x = DX[i], y0 = 50, w = 62, h = 96;
  box(x - w / 2, y0, w, h, DCOL[i], o.glow ? .6 : .3); rrect(x - w / 2, y0, w, h, 820 + i * 5, { w: 1.6 }); dot(x + w / 2 - 10, y0 + h / 2 + 4, 2.6, C.ink);
  txt(DNAME[i], x, y0 - 14, { s: 20, c: DINK[i] }); box(x - 20, y0 + 14, 40, 20, C.card); rrect(x - 20, y0 + 14, 40, 20, 840 + i, { c: C.line, w: 1 });
  const P = o.P; txt(P ? String(P[i]) : '?', x, y0 + 24, { s: 15, c: P ? C.goldInk : C.soft }); }
// room: o = { p (chances), Q (scores), P (shown crumb chances), pick, t, pipX }
function room(t, o = {}) {
  rline(20, 146, 380, 146, 801, { w: 1.6 }); for (let i = 0; i < 3; i++) door(i, { P: o.P, glow: o.pick === i });
  if (o.Q) for (let i = 0; i < 3; i++) txt(`score ${f2(o.Q[i])}`, DX[i], 162, { s: 14, c: C.soft });
  if (o.p) { for (let i = 0; i < 3; i++) txt(`chance ${f2(o.p[i])}`, DX[i], 180, { s: 15, c: DINK[i] }); pie(300, 240, 34, o.p, { labels: true }); }
  L.mouse(o.mx ?? 110, o.my ?? 244, 1, 1, t);
}
// Pip walking to door d and back during u ∈ [0, 1]
function walkTo(d, u) { const go = ease(u / .45) * (1 - ease((u - .65) / .35)); return { mx: lerp(110, DX[d], go), my: lerp(244, 206, go) }; }
// chart of the three chances over picks [0, n): x0..x1, y0 (bottom) .. y1 (top)
function chanceChart(H, n, o = {}) {
  const x0 = o.x0 ?? 62, x1 = o.x1 ?? 360, y0 = o.y0 ?? 388, y1 = o.y1 ?? 312, N = o.N ?? T, X = i => x0 + i / N * (x1 - x0), Y = v => y0 - v * (y0 - y1);
  rline(x0 - 4, y0, x1 + 4, y0, 960, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1 - 4, 961, { c: C.line, w: 1, once: true });
  txt('0', x0 - 14, y0, { s: 12, c: C.soft }); txt('1', x0 - 14, y1, { s: 12, c: C.soft }); txt('pick 1', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt('pick ' + N, x1, y0 + 11, { s: 12, c: C.soft, al: 'right' });
  if (o.title) txt(o.title, x0 - 8, y1 - 12, { s: 13, c: C.soft, al: 'left' });
  if (o.change && N > TC) { ctx.save(); ctx.setLineDash([3, 4]); ctx.strokeStyle = C.rose; ctx.beginPath(); ctx.moveTo(X(TC), y0); ctx.lineTo(X(TC), y1); ctx.stroke(); ctx.restore(); txt('A → 0.95', X(TC) + 4, y1 + 2, { s: 12, c: C.roseInk, al: 'left' }); }
  for (let d = 0; d < 3; d++) { ctx.save(); ctx.strokeStyle = DINK[d]; ctx.lineWidth = 1.8; ctx.beginPath(); for (let i = 0; i < Math.min(n, N); i++) { const x = X(i + 1), y = Y(H[i].pn[d]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.restore(); }
}
function legendDoors(x, y) { DNAME.forEach((s, i) => { rline(x + i * 52, y, x + i * 52 + 14, y, 970 + i, { c: DINK[i], w: 2, once: true }); txt(s, x + i * 52 + 20, y, { s: 13, c: DINK[i], al: 'left' }); }); }
function waitNote(i) { if (ready(i)) return false; txt('simulating 200 runs…', W / 2, 350, { s: 15, c: C.soft }); return true; }
// the jump scene with a bell curve (World 3)
const JX = d => 70 + d / 1.2 * 290, JGY = 200;
function jumpScene(t, sig, o = {}) { const MU = o.mu ?? .7;
  box(20, JGY, JX(0) - 20, 60, '#d9d1c6'); rrect(20, JGY, JX(0) - 20, 60, 890); box(JX(.55), JGY, JX(1.2) - JX(.55) + 10, 60, '#d9d1c6'); rrect(JX(.55), JGY, JX(1.2) - JX(.55) + 10, 60, 891);
  L.cheese(JX(.7), JGY - 10, .8); L.mouse(JX(0) - 18, JGY - 10, 1, .9, t);
  rline(JX(0), JGY + 76, JX(1.2), JGY + 76, 892, { c: C.line, w: 1 }); [0, .7, 1.2].forEach((d, i) => { rline(JX(d), JGY + 72, JX(d), JGY + 80, 893 + i, { c: C.line, w: 1, once: true }); txt(d + ' m', JX(d), JGY + 90, { s: 12, c: C.soft }); });
  if (sig > 0) { const hmax = 90, peak = 1 / (sig * Math.sqrt(2 * Math.PI)), sc = Math.min(hmax / peak, 18); ctx.save(); ctx.fillStyle = 'rgba(111,183,180,.3)'; ctx.strokeStyle = DINK[2]; ctx.lineWidth = 1.8; ctx.beginPath();
    for (let i = 0; i <= 240; i++) { const d = i / 200, y = JGY - 22 - sc * Math.exp(-(((d - MU) / sig) ** 2) / 2) / (sig * Math.sqrt(2 * Math.PI)); i ? ctx.lineTo(JX(d), y) : ctx.moveTo(JX(d), y); } ctx.stroke(); ctx.restore(); }
}

const STEPS = [
 { h: 'Back in the door room', p: ['The door room from the policy gradient reel. Behind doors A, B and C a crumb, reward 1, appears with a hidden chance: 0.2, 0.5 and 0.8. Pip picks one door per visit, and the visit ends.', 'Pip’s policy is a **pie** of chances made from three preferences by softmax. It starts at a third each.'],
   shot: [3, 9],
   draw(t) { room(t, { p: [1 / 3, 1 / 3, 1 / 3], P: t > 3 ? P0 : null });
     if (t > 3) txt('hidden from Pip', 200, 26, { s: 13, c: C.soft, a: clamp(t - 3) });
     if (t > 5) card(['one pick per visit: crumb 1 or nothing 0', { s: 'crumb chances 0.2, 0.5, 0.8 (hidden)', c: C.soft }, { s: 'policy: a pie of chances, a third each', c: DINK[2] }], t - 5, { gap: 1.1 }); } },
 { h: 'A critic for each door', p: ['Pip learns as an **actor-critic**. The critic keeps a **score** per door, starting at 0. After each pick it studies 8 random picks from its memory of the last 200, the **replay** idea from DQN, moving that door’s score a tenth of the way toward the crumb it got.', 'The actor uses all three scores at once: the policy gradient reel’s nudge, chance × (score − the pie’s average score), times a step of 3. No sampled returns needed.'],
   shot: [3, 10],
   draw(t) { room(t, { p: [1 / 3, 1 / 3, 1 / 3], Q: [0, 0, 0] });
     card(['critic: a score per door, from its memory', '   score += 0.1 × (crumb − score)', { s: 'actor: pref += 3 × chance × (score − average)', c: DINK[2] }, { s: 'average = the pie’s average score', c: C.soft }], t - 1, { gap: 1.3, fs: 16 }); } },
 { h: 'Why old picks still count', p: ['REINFORCE and PPO judged moves by the returns of Pip’s own recent walks, so walks made under an older policy were stale. This critic learns something else: what each door pays, like DQN’s Q-values. A crumb behind C is evidence about C, whoever picked it and whenever.', 'The actor is trained against the critic’s scores, not against its own returns. So the critic may replay old picks: the method is **off-policy**. SAC works the same way.'],
   shot: [3, 9],
   draw(t) { box(0, 0, W, 285, C.bg); const R = rng(77), seq = Array.from({ length: 30 }, () => Math.floor(R() * 3));
     txt('memory: old picks, any policy', 110, 40, { s: 15, c: C.goldInk }); seq.forEach((d, i) => { if (t < .3 + i * .06) return; const x = 24 + (i % 10) * 18, y = 60 + Math.floor(i / 10) * 22; rrect(x, y, 15, 18, 2300 + i, { fill: DCOL[d], c: C.line, w: 1 }); txt(DNAME[d], x + 7.5, y + 9, { s: 11 }); });
     if (t > 2.5) { arrow(210, 90, 250, 90, 2340, { c: C.soft }); rrect(260, 64, 110, 52, 2341, { fill: '#f6e7c8' }); txt('critic: what', 315, 82, { s: 14 }); txt('each door pays', 315, 100, { s: 14 }); }
     if (t > 4) { arrow(315, 120, 315, 160, 2342, { c: C.soft }); rrect(260, 164, 110, 40, 2343, { fill: '#dbecea' }); txt('actor', 315, 184, { s: 15 }); txt('learns from the scores', 315, 220, { s: 13, c: C.soft }); }
     if (t > 6) card(['REINFORCE, PPO: returns of fresh walks', { s: 'here: scores of doors, from any old pick', c: DINK[2] }, { s: 'learning from old data = off-policy', c: C.soft }], t - 6, { gap: 1.2 }); } },
 { h: 'An unlucky start', p: [`Here is Pip’s first run. In its first 30 picks it tries A ${C30[0]} times, B ${C30[1]} times${C30[2] ? ' and C ' + C30[2] + ' times' : ', and C never'}. B pays ${K30[1]} crumbs, so its score climbs, while C’s score stays at its starting 0.`, 'After B’s first crumbs, the actor pushed B’s chance up within a few picks, so Pip tried the other doors even less. Luck decided the first picks; the rule did the rest. We chose this run to show getting stuck; the next steps show how often it happens.'],
   shot: [4, 12],
   draw(t) { const D = .42, n = clamp(Math.floor(t / D), 0, 30), h = H0[Math.max(0, n - 1)], u = (t % D) / D, cur = H0[Math.min(n, 29)];
     const w = n < 30 ? walkTo(cur.d, u) : {}; room(t, Object.assign({ p: n ? h.pn : [1 / 3, 1 / 3, 1 / 3], Q: n ? h.Q : [0, 0, 0], pick: n < 30 ? cur.d : null }, w));
     if (n < 30 && u > .4 && u < .7) txt(cur.r ? '+1' : '0', DX[cur.d], 34, { s: 16, c: cur.r ? C.goldInk : C.soft });
     txt(`pick ${n}`, 20, 296, { s: 15, al: 'left' }); [0, 1, 2].forEach(d => txt(`${DNAME[d]}: ${cnt(H0, d, 0, n)} picks, ${crumbs(H0, d, 0, n)} crumbs`, 20, 320 + d * 22, { s: 15, al: 'left', c: DINK[d] })); } },
 { h: 'Stuck on B', p: [`The chart shows Pip’s three chances, pick by pick: A blue, B orange, C teal. By pick 400 the chance of B is ${f2(G0.p400[1])}. ${CEARLY + CLATE === 0 ? 'In all 400 picks Pip never tries C' : `In picks 100 to 400 Pip tries C ${CLATE === 0 ? 'not once' : CLATE + ' times'}`}, so the critic’s score for C stays ${f2(G0.Q400[2])}, far below its true 0.8.`, 'The critic only learns about doors Pip picks. A’s score is left over from its 2 early tries, and B’s jumps around because each study moves it a tenth. Pip collects about half a crumb per visit instead of 0.8, with almost no way out.'],
   shot: [4, 11],
   draw(t) { const n = clamp(Math.floor((t - 1) * 120), 0, TC), h = H0[Math.max(0, n - 1)]; room(t, { p: n ? h.pn : [1 / 3, 1 / 3, 1 / 3], Q: n ? h.Q : [0, 0, 0] });
     chanceChart(H0, n, { N: TC, title: 'chances, pick by pick (first run)' }); legendDoors(250, 296); } },
 { h: 'Not one unlucky run', get p() { return [`Run it 200 times with different dice. In ${agg(0).stuck} of the 200 runs, by pick 400 Pip’s favourite door is not C.`, 'It is the trap from the Exploration versus Exploitation reel: a learner that stops trying a door never finds out it was wrong. Pip needs a reason to keep trying doors it currently rates low.']; },
   shot: [3, 9],
   draw(t) { room(t, { p: G0.p400, Q: G0.Q400 }); box(0, 290, W, 130, C.bg); if (waitNote(0)) return; const A = agg(0);
     const n = clamp(Math.floor((t - 1) * 60), 0, NR); for (let k = 0; k < n; k++) { const r = RUNS[0][k], bd = best(r.p400); dot(40 + (k % 40) * 8, 304 + Math.floor(k / 40) * 12, 3, bd === 2 ? DCOL[2] : C.rose); }
     if (n >= NR) { txt(`teal: C favourite · rose: stuck on a worse door (${A.stuck})`, W / 2, 378, { s: 14, c: C.soft }); } } },
 { h: 'How spread out?', p: ['A door’s **surprise** is −log of its chance: a door Pip almost never picks would be very surprising, a sure door not at all. **Entropy** is the average surprise. It measures how spread out the chances are.', `A third each: entropy ${f2(ent(PIES[0]))}, the most possible for three doors. Chances 0.1, 0.1, 0.8: ${f2(ent(PIES[1]))}. Always the same door: 0.`],
   shot: [3, 10],
   draw(t) { box(0, 0, W, 285, C.bg); PIES.forEach((p, i) => { const a = clamp(t - .5 - i * 1.5); if (a <= 0) return; pie(80 + i * 120, 120, 40, p, { labels: true }); txt(`entropy ${f2(ent(p))}`, 80 + i * 120, 182, { s: 16, c: C.goldInk }); });
     if (t > 5) txt('spread out  ←→  certain', W / 2, 230, { s: 16, c: C.soft });
     card(['surprise of a door = −log(chance)', '   chance 1/3: 1.10    chance 0.1: 2.30    chance 1: 0', { s: 'entropy = the average surprise', c: C.goldInk }], t - 6, { gap: 1.2, fs: 16 }); } },
 { h: 'A bonus for staying spread', p: ['Now change what Pip aims for: average reward **plus α × entropy**. α (alpha) is the **temperature**: how many crumbs one unit of spread is worth. Careful: in earlier reels α was a step size; here it is not.', 'For the actor, a door’s worth becomes its score + α × its surprise, and the nudge uses worth instead of score. That is exactly the uphill direction of the new aim. Rare doors earn a bonus, so their preferences do not sink forever.'],
   shot: [3, 10],
   draw(t) { room(t, { p: G0.p400, Q: G0.Q400 });
     card(['aim: average reward + α × entropy', { s: 'worth of a door = score + α × surprise', c: DINK[2] }, { s: 'pref += 3 × chance × (worth − average worth)', c: C.soft }], t - 1, { gap: 1.4 }); } },
 { h: 'Why not random picks?', p: ['Why not simply pick a random door 1 time in 5, as ε-greedy does? Because the actor’s own aim would still push its pie toward certainty, and the random picks would go to good and bad doors alike, forever.', 'With the bonus, staying spread is part of the aim itself. The actor keeps spare chances where they cost little: on doors whose scores are close to the best.'],
   shot: [3, 9],
   draw(t) { room(t, { p: G2.p400, Q: G2.Q400 });
     card([{ s: 'ε-greedy: random picks bolted on, any door', c: C.roseInk }, { s: 'bonus: spread is part of the aim', c: DINK[2] }, { s: 'spare chances go to near-best doors', c: C.soft }], t - 1, { gap: 1.3 }); } },
 { h: 'Turning the temperature', p: [`If Pip knew the true crumb chances, the best pie for a temperature would give each door a chance in proportion to e to the power of its crumb chance divided by α. At α = 0.2: A ${f2(soft(P0, .2)[0])}, B ${f2(soft(P0, .2)[1])}, C ${f2(soft(P0, .2)[2])}.`, 'That is softmax with each preference set to crumb chance ÷ α, and it is where the actor’s rule settles once the critic is right. Cold, near 0: always C. Hot: closer to a third each.'],
   shot: [3, 11],
   draw(t) { box(0, 0, W, 285, C.bg); const as = [.02, .2, .5, 2]; as.forEach((a, i) => { const u = clamp(t - .5 - i * 1.3); if (u <= 0) return; const p = soft(P0, a); pie(60 + i * 93, 110, 34, p, { labels: true });
       txt(`α = ${a}`, 60 + i * 93, 162, { s: 15 }); txt(p.map(f2).join('  '), 60 + i * 93, 184, { s: 12, c: C.soft }); txt(`entropy ${f2(ent(p))}`, 60 + i * 93, 202, { s: 12, c: C.goldInk }); });
     if (t > 6) txt('cold  →  hot', W / 2, 240, { s: 16, c: C.soft }); } },
 { h: 'Same dice, with a bonus', get p() { return [`Rerun Pip’s first run with α = 0.2. The dice are the same, so it starts the same way, but once its chances differ the picks drift apart. The bonus keeps C in play: by pick 400 the chance of C is ${f2(G2.p400[2])} and its score ${f2(G2.Q400[2])}.`, `Over 200 runs, the favourite at pick 400 is not C in ${agg(4).stuck} runs, against ${agg(0).stuck} without the bonus.`]; },
   shot: [4, 11],
   draw(t) { const n = clamp(Math.floor((t - 1) * 120), 0, TC), h = H2[Math.max(0, n - 1)]; room(t, { p: n ? h.pn : [1 / 3, 1 / 3, 1 / 3], Q: n ? h.Q : [0, 0, 0] });
     chanceChart(H2, n, { N: TC, title: 'chances, pick by pick (α = 0.2)' }); legendDoors(250, 296); } },
 { h: 'Curiosity has a price', get p() { return [`Even when it has found C, the curious Pip keeps a few percent on A and B. Averaged over all 200 runs, stuck ones included, its chance of C at pick 400 is ${f2(agg(4).pc)}, against ${f2(agg(0).pc)} without the bonus. Crumbs per visit over the first 400 picks: a tie, ${f2(agg(4).r1)} and ${f2(agg(0).r1)}.`, `So at α = 0.2 curiosity costs about what it saves before anything changes. A cooler α = 0.1 is already ahead: ${f2(agg(3).r1)} per visit. The bigger payoff comes when the world changes.`]; },
   shot: [3, 9],
   draw(t) { room(t, { p: G2.p400, Q: G2.Q400 }); box(0, 290, W, 130, C.bg); if (waitNote(4) || waitNote(0)) return; const A = agg(0), B = agg(4);
     card([`average over 200 runs, picks 1 to 400`, { s: `no bonus: chance of C ${f2(A.pc)}, ${f2(A.r1)} crumbs per visit`, c: C.roseInk }, { s: `α = 0.2: chance of C ${f2(B.pc)}, ${f2(B.r1)} crumbs per visit`, c: DINK[2] }], t - 1, { gap: 1.3, fs: 16 }); } },
 { h: 'The world changes', get p() { return [`After pick 400, door A suddenly pays a crumb 95 times in 100. The certain Pip picks A ${NA0 === 0 ? 'not once' : NA0 === 1 ? 'once' : NA0 + ' times'} in the next 600 picks and never finds out. The curious Pip tries A ${NA2} times; A leads its pie from pick ${SWITCH2}.`, `Over 200 runs, A is the favourite at pick 1000 in ${agg(4).foundA} runs with the bonus, ${agg(0).foundA} without.`]; },
   shot: [4, 12],
   draw(t) { const n = clamp(Math.floor((t - .5) * 110), 0, T); box(0, 0, W, 285, C.bg);
     chanceChart(H0, n, { y0: 130, y1: 50, change: 1, title: 'no bonus: chances, pick by pick' }); chanceChart(H2, n, { y0: 264, y1: 184, change: 1, title: 'α = 0.2' }); legendDoors(250, 30);
     if (t > 9.6 && ready(0) && ready(4)) card([`A favourite at pick 1000, of 200 runs:`, { s: `no bonus ${agg(0).foundA}   ·   α = 0.2: ${agg(4).foundA}`, c: DINK[2] }], t - 9.6, { gap: 1, y: 300 }); } },
 { h: 'Your turn', ctl: { label: 'α', values: ALS, def: 4, onChange: st => {} },
   p: ['Drag the **temperature** α and rerun the 200 runs. The pie is the best pie for that temperature if Pip knew the chances; the numbers are what learning Pips actually did.', 'Too cold and Pip gets stuck and misses the change; too hot and it wastes visits on bad doors. Which α collects the most after the change?'],
   shot: [3, 9],
   draw(t, st) { const i = st.ci ?? 4, a = ALS[i]; box(0, 0, W, 285, C.bg); pie(90, 110, 50, soft(P0, a), { labels: true }); txt(`best pie, α = ${a}`, 90, 182, { s: 15 }); txt('(chances before the change)', 90, 200, { s: 12, c: C.soft });
     if (waitNote(i)) return; const A = agg(i);
     const lines = [[`stuck at pick 400`, `${A.stuck} of 200`], [`crumbs per visit, picks 1–400`, f2(A.r1)], [`found A after the change`, `${A.foundA} of 200`], [`crumbs per visit, last 300 picks`, f2(A.r2)]];
     lines.forEach(([l, v], k) => { txt(l, 170, 56 + k * 44, { s: 15, al: 'left', c: C.soft }); txt(v, 170, 76 + k * 44, { s: 17, al: 'left', c: k % 2 ? C.goldInk : DINK[2] }); });
     ALS.forEach((aa, k) => { if (!ready(k)) return; const B = agg(k), x = 70 + k * 52, h = B.r2 * 80; box(x - 14, 390 - h, 28, h, k === i ? C.gold : C.yellow); txt(String(aa), x, 402, { s: 12, c: C.soft }); txt(f2(B.r2), x, 382 - h, { s: 12, c: C.soft }); });
     txt('crumbs per visit after the change, by α', 40, 300, { s: 13, c: C.soft, al: 'left' }); } },
 { h: 'A jump instead of doors', p: ['Now a different task, sketched in the policy gradient reel. Pip stands on a ledge and picks a **jump strength**: how many metres it flies, any number. The cheese lies 0.7 metres out. Landing right on it pays 1; short or long pays less, along the curve below: a miss of 0.15 metres pays about 0.37.', 'Strengths are not a menu, so the policy is a **bell curve**: a centre, the strength Pip aims for, and a spread, how far its jumps scatter around it.'],
   shot: [3, 10],
   draw(t) { box(0, 0, W, 420, C.bg); const mu = .5, sg = .1; jumpScene(t, t > 5 ? sg : 0, { mu });
     if (t > 5) txt('bell: centre 0.5, spread 0.1 (an example)', W / 2, 40, { s: 14, c: DINK[2] });
     txt('slider answer: α = 0.2 collected most after the change', W / 2, 18, { s: 12, c: C.soft });
     if (t > 1) { const x0 = JX(0), x1 = JX(1.2), y0 = 396, y1 = 320; rline(x0 - 4, y0, x1 + 4, y0, 2400, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1 - 4, 2401, { c: C.line, w: 1, once: true });
       txt('reward for each strength', x0 - 8, y1 - 12, { s: 13, c: C.soft, al: 'left' }); txt('1', x0 - 14, y1, { s: 12, c: C.soft }); txt('0', x0 - 14, y0, { s: 12, c: C.soft });
       ctx.save(); ctx.strokeStyle = C.gold; ctx.lineWidth = 2; ctx.beginPath(); const n = Math.round(clamp((t - 1) / 1.5) * 240); for (let i = 0; i <= n; i++) { const d = i / 200, y = y0 - Math.exp(-(((d - .7) / WR) ** 2)) * (y0 - y1); i ? ctx.lineTo(JX(d), y) : ctx.moveTo(JX(d), y); } ctx.stroke(); ctx.restore(); } } },
 { h: 'Bell curves too', p: [`A bell’s entropy grows with its spread: ${f2(Math.log(2))} more each time the spread doubles. Without a bonus the bell narrows toward one strength. With it, the bell keeps the width where the reward lost by spreading balances the bonus. Strengths are kept between 0 and 1.2 metres.`, `Exact numbers for a bell centred on the cheese: α = 0.01 keeps a spread of ${f2(BSIG[0])} metres, α = 0.03 keeps ${f2(BSIG[1])}, α = 0.1 keeps ${f2(BSIG[2])}.`],
   shot: [3, 10],
   draw(t) { const k = clamp(Math.floor((t - 1) / 2.5), 0, 2), s = BSIG[k]; box(0, 0, W, 285, C.bg); jumpScene(t, s);
     txt(`α = ${BAL[k]}: spread ${f2(s)} m, average reward ${f2(ER(s))}`, W / 2, 22, { s: 16, c: DINK[2] });
     // close-up of the three bells, 0.6 to 0.8 m
     const x0 = 70, x1 = 360, y0 = 392, X = d => x0 + (d - .6) / .2 * (x1 - x0); rline(x0, y0, x1, y0, 990, { c: C.line, w: 1, once: true });
     [.6, .7, .8].forEach((d, i) => txt(d + ' m', X(d), y0 + 10, { s: 12, c: C.soft })); txt('close-up near the cheese (each bell has the same area)', x0 - 8, 308, { s: 13, c: C.soft, al: 'left' });
     BSIG.forEach((sg, j) => { if (j > k) return; const sc = 1.85; ctx.save(); ctx.strokeStyle = j === k ? DINK[2] : '#9fc9c7'; ctx.lineWidth = j === k ? 2 : 1.3; ctx.beginPath();
       for (let i = 0; i <= 200; i++) { const d = .6 + i / 1000, y = y0 - sc * Math.exp(-(((d - .7) / sg) ** 2) / 2) / (sg * Math.sqrt(2 * Math.PI)); i ? ctx.lineTo(X(d), y) : ctx.moveTo(X(d), y); } ctx.stroke(); ctx.restore();
       txt(`α = ${BAL[j]}: tallest`.replace(': tallest', ''), 300, 330 + j * 15, { s: 12, c: j === k ? DINK[2] : '#7fb0ae', al: 'left' }); rline(282, 330 + j * 15, 296, 330 + j * 15, 995 + j, { c: j === k ? DINK[2] : '#9fc9c7', w: j === k ? 2 : 1.3, once: true }); }); } },
 { h: 'Soft values', p: ['In longer tasks the bonus is added to the reward at every step. Then the critic’s value of a situation counts future bonuses too: a spot with several good ways on is worth more than a spot with only one, so the agent learns to prefer places that keep options open.', 'Why “soft”? The best pie is a softmax of the values: a soft version of always taking the maximum. As α shrinks to 0 it becomes the hard maximum again.'],
   shot: [3, 10],
   draw(t) { box(0, 0, W, 285, C.bg); const spot = (x, y, n, lab, a) => { if (a <= 0) return; rcircle(x, y, 14, 2500 + n, { fill: C.card }); for (let k = 0; k < n; k++) { const an = -Math.PI / 2 + (k - (n - 1) / 2) * .5; arrow(x + 16 * Math.cos(an), y + 16 * Math.sin(an), x + 70 * Math.cos(an), y + 70 * Math.sin(an), 2510 + n * 5 + k, { c: DINK[2], w: 1.6 }); } txt(lab, x, y + 34, { s: 14, c: C.soft }); };
     spot(110, 190, 1, 'one good way on', clamp(t - .5)); spot(290, 190, 3, 'three good ways on', clamp(t - 1.5)); L.cheese(110, 100, .7); [-1, 0, 1].forEach(k => L.cheese(290 + 70 * Math.sin(k * .5), 190 - 70 * Math.cos(k * .5) - 14, .6));
     if (t > 3) txt('with the bonus: worth more', 290, 250, { s: 15, c: C.goldInk });
     if (t > 5) card(['value = rewards + α × future entropy', { s: 'soft = softmax instead of a hard max', c: C.soft }], t - 5, { gap: 1.2 }); } },
 { h: 'Soft Actor-Critic', p: ['**SAC** (Haarnoja, Zhou, Abbeel and Levine, 2018) puts this into an actor-critic for continuous actions. Its actor cannot list every strength, so it draws a few from its bell, asks the critic which way each should move, and shifts and reshapes the bell.', 'It replays old experience, since its critic learns values. It keeps two critics and uses the smaller score: the actor is drawn to spots where a critic is too high by luck, and two critics rarely share a lucky error. Later, α was tuned automatically.'],
   shot: [3, 9],
   draw(t) { box(0, 0, W, 285, C.bg); jumpScene(t, BSIG[2]);
     card(['actor: try strengths from the bell, ask the critic', 'replay old experience (off-policy)', 'two critics, take the smaller score', { s: 'later: α tuned automatically', c: C.soft }], t - 1, { y: 304, gap: 1.1, fs: 16, lh: 22 }); } },
 { h: 'Where it’s used', p: ['SAC is a standard choice for continuous control, from simulated robots to real ones. The same group trained a real four-legged robot, Minitaur, to walk with about two hours of real experience: short for a real robot, where every trial costs time and wear. They also taught a robot hand to turn a valve.', 'Entropy bonuses are common elsewhere too: many policy gradient methods add a small one to keep exploring.'],
   shot: [3, 9],
   draw(t) { box(0, 0, W, 285, C.bg); const x = 200, y = 150, ph = t * 3;
     rrect(x - 50, y - 24, 100, 30, 2100, { fill: C.grey }); [-38, -14, 14, 38].forEach((dx, i) => { const s = Math.sin(ph + i * 1.6) * 8; rline(x + dx, y + 6, x + dx + s, y + 40, 2101 + i, { w: 3 }); rline(x + dx + s, y + 40, x + dx + s * .4, y + 70, 2110 + i, { w: 3 }); });
     txt('a four-legged robot: each motor is a number', W / 2, 60, { s: 15, c: C.soft }); rline(60, y + 72, 340, y + 72, 2120, { c: C.line }); } },
 { h: 'Recap', p: ['**Entropy** measures how spread out the chances are: the average surprise. Adding α × entropy to the reward keeps Pip curious, so an unlucky start or a changed world does not trap it.', 'The temperature α sets the price of curiosity. **SAC** puts this bonus into an actor-critic for continuous actions.'],
   shot: [3, 9],
   draw(t) { room(t, { p: G2.p, Q: G2.Q });
     card(['entropy = average surprise = spread of chances', 'aim: reward + α × entropy', { s: 'α: cold = greedy, hot = random', c: C.soft }, { s: 'SAC: actor-critic with the bonus', c: DINK[2] }], t - .3, { y: 296, gap: 1, fs: 16, lh: 22 }); } },
];

RLStory.register('sac', { title: 'Entropy & Soft Actor-Critic', track: 'Policy Gradients', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { SK, C30, K30, CLATE, CEARLY, p400_0: G0.p400, Q400_0: G0.Q400, p400_2: G2.p400, Q400_2: G2.Q400, NA0, NA2, SWITCH2, endP0: G0.p, endP2: G2.p,
    ents: PIES.map(ent), soft02: soft(P0, .2), BSIG, BER: BSIG.map(ER), agg, ALS } });
})();
