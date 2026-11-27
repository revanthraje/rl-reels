// Policy gradient methods as a story: Pip stops learning values and instead keeps a policy made of chances,
// produced from adjustable preference numbers (softmax). In the door room (one decision, three doors with hidden
// crumb chances) the expected return and its exact gradient can be worked out, so the viewer sees "which way to nudge
// each preference". Every number comes from the exact formulas below (no randomness needed in this reel).
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;
// ===== pg kit (shared helpers for the Policy Gradients track; copy, do not import) =====
const softmax = th => { const m = Math.max(...th), e = th.map(x => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
const DCOL = ['#8fb3d9', '#ef8f5a', '#6fb7b4'], DINK = ['#4f7fb0', '#c4581f', '#3f7f7c'], DNAME = ['A', 'B', 'C'], DX = [90, 200, 310];
// pie of chances at (x, y), radius r, slices coloured per action; o.labels shows chances
function pie(x, y, r, pr, o = {}) {
  let a0 = -Math.PI / 2; pr.forEach((p, i) => { const a1 = a0 + p * Math.PI * 2; ctx.save(); ctx.globalAlpha = o.a ?? 1; ctx.fillStyle = (o.cols || DCOL)[i];
    ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, r, a0, a1); ctx.closePath(); ctx.fill(); ctx.restore();
    if (o.labels && p > .06) { const am = (a0 + a1) / 2; txt((o.names || DNAME)[i], x + Math.cos(am) * r * .62, y + Math.sin(am) * r * .62, { s: 13, a: o.a ?? 1 }); } a0 = a1; });
  rcircle(x, y, r, 880, { j: .6 });
}
// arrow fans in a maze cell: one arrow per move, length ∝ chance (chance 1 reaches the cell edge)
function fans(M, TH, o = {}) {
  for (let r = 0; r < M.rows; r++) for (let c = 0; c < M.cols; c++) {
    if (M.term(c, r) || M.isRock(c, r)) continue; const k = M.key(c, r), p = softmax(TH.slice(k * 4, k * 4 + 4)), best = p.indexOf(Math.max(...p)), x = M.cx(c), y = M.cy(r) + 4;
    p.forEach((q, m) => { const [dx, dy] = M.MOVES[m], len = 3 + q * (M.cs / 2 - 7), big = m === best && q > .3;
      alpha(o.a ?? 1, () => arrow(x + dx * 4, y + dy * 4, x + dx * (4 + len), y + dy * (4 + len), 600 + k * 4 + m, { c: big ? C.goldInk : C.soft, w: big ? 2 : 1.2, head: 4 + q * 4 })); });
  }
}
// ===== end pg kit =====
const P = [.2, .5, .8];                                       // hidden crumb chance behind doors A, B, C
const J = th => softmax(th).reduce((a, p, i) => a + p * P[i], 0);
const grad = th => { const p = softmax(th), j = J(th); return p.map((q, i) => q * (P[i] - j)); };
const fz = v => Math.abs(v) < .005 ? '0' : f2(v);
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−');
const AL = 2, NCL = 20;                                       // gradient ascent: step size and number of steps shown
const CLIMB = (() => { let th = [0, 0, 0]; const out = [{ th, j: J(th), p: softmax(th) }]; for (let i = 0; i < NCL; i++) { const g = grad(th); th = th.map((t, k) => t + AL * g[k]); out.push({ th, j: J(th), p: softmax(th), g }); } return out; })();
const U = [0, 0, 0], G0 = grad(U), T1 = [0, 0, 1], P1 = softmax(T1), E1 = Math.exp(1);
const SM0 = softmax([0, .01, -.01]), SM1 = softmax([0, -.01, .01]);
const TS = [1, 1, 2], PS = softmax(TS);                       // "only differences matter": add 1 to all of (0, 0, 1)
const SLOPE = G0[2];                                           // slope of J along θC at 0
const DEMO = (() => { const R = L.rng(21); return Array.from({ length: 9 }, (_, i) => R() < P[i % 3]); })();   // seeded tries for the door-room demo
const MZ = L.gridWorld({}), MZTH = new Array(80).fill(0);

// ───── drawing ─────
function door(i, o = {}) {
  const x = DX[i], y0 = 50, w = 62, h = 96, a = o.a ?? 1;
  alpha(a, () => { box(x - w / 2, y0, w, h, DCOL[i], o.glow ? .55 : .3); rrect(x - w / 2, y0, w, h, 820 + i * 5, { w: 1.6 }); dot(x + w / 2 - 10, y0 + h / 2 + 4, 2.6, C.ink);
    txt(DNAME[i], x, y0 - 14, { s: 20, c: DINK[i] });
    const sec = o.secret ?? 0; box(x - 18, y0 + 14, 36, 20, C.card); rrect(x - 18, y0 + 14, 36, 20, 840 + i, { c: C.line, w: 1 });
    txt(sec > .5 ? String(P[i]) : '?', x, y0 + 24, { s: 15, c: sec > .5 ? C.goldInk : C.soft }); });
}
function room(t, o = {}) {
  rline(20, 146, 380, 146, 801, { w: 1.6 }); for (let i = 0; i < 3; i++) door(i, { secret: o.secret, glow: o.glow === i, a: o.doorA });
  if (o.rows !== false) { const th = o.th || U, p = softmax(th);
    for (let i = 0; i < 3; i++) { txt(`pref ${f2(th[i])}`, DX[i], 162, { s: 14, c: C.soft, a: o.prefA ?? 1 }); txt(`chance ${f2(p[i])}`, DX[i], 180, { s: 15, c: DINK[i], a: o.chA ?? 1 }); } }
  if (o.pie !== false) pie(o.px ?? 300, o.py ?? 238, 34, softmax(o.th || U), { labels: true, a: o.pieA ?? 1 });
  if (o.pip !== false) L.mouse(o.mx ?? 110, o.my ?? 244, 1, 1, t);
}
// J along θC (θA = θB = 0): axes explained in the text
function hill(t, o = {}) {
  const x0 = 60, x1 = 350, y0 = 386, y1 = 306, lo = .3, hi = .8, X = v => x0 + (v + 3) / 6 * (x1 - x0), Y = j => y0 - (j - lo) / (hi - lo) * (y0 - y1);
  rline(x0, y0, x1, y0, 860, { c: C.line, w: 1 }); rline(x0, y0, x0, y1 - 4, 861, { c: C.line, w: 1 });
  txt('expected return', x0 - 6, y1 - 12, { s: 13, c: C.soft, al: 'left' }); txt('0.8', x0 - 16, Y(.8), { s: 12, c: C.soft }); txt('0.3', x0 - 16, Y(.3), { s: 12, c: C.soft }); txt('0.5', x0 - 16, Y(.5), { s: 12, c: C.soft });
  ctx.save(); ctx.setLineDash([3, 4]); ctx.strokeStyle = C.grey; ctx.beginPath(); ctx.moveTo(X(0), y0); ctx.lineTo(X(0), Y(.5)); ctx.lineTo(x0, Y(.5)); ctx.stroke(); ctx.restore();
  txt('preference of door C →', x1, y0 + 12, { s: 13, c: C.soft, al: 'right' }); [-3, 0, 3].forEach(v => txt(String(v).replace('-', '−'), X(v), y0 + 12, { s: 12, c: C.soft, a: v === 3 ? 0 : 1 }));
  const n = Math.round(clamp(o.p ?? 1) * 120); ctx.save(); ctx.strokeStyle = DINK[2]; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i <= n; i++) { const v = -3 + i / 20, y = Y(J([0, 0, v])); i ? ctx.lineTo(X(v), y) : ctx.moveTo(X(v), y); } ctx.stroke(); ctx.restore();
  if (o.at != null) { const v = o.at, j = J([0, 0, v]); dot(X(v), Y(j), 5, C.ink); if (o.tan) { const s = grad([0, 0, v])[2], d = 1.1; alpha(o.tan, () => arrow(X(v), Y(j), X(v + d), Y(j + s * d), 870, { c: C.roseInk, w: 2 })); } }
}
function climbChart(n) {
  const x0 = 60, x1 = 350, y0 = 388, y1 = 312, Y = j => y0 - (j - .4) / .4 * (y0 - y1), X = i => x0 + i / NCL * (x1 - x0);
  rline(x0, y0, x1, y0, 862, { c: C.line, w: 1 }); rline(x0, y0, x0, y1 - 4, 863, { c: C.line, w: 1 });
  txt('expected return', x0 - 6, y1 - 12, { s: 13, c: C.soft, al: 'left' }); txt('0.8', x0 - 16, Y(.8), { s: 12, c: C.soft }); txt('0.4', x0 - 16, Y(.4), { s: 12, c: C.soft });
  ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = C.grey; ctx.beginPath(); ctx.moveTo(x0, Y(.8)); ctx.lineTo(x1, Y(.8)); ctx.stroke(); ctx.restore(); txt('best: always C', x1, Y(.8) - 9, { s: 12, c: C.soft, al: 'right' });
  txt('nudge 0', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt(`nudge ${NCL}`, x1, y0 + 11, { s: 12, c: C.soft, al: 'right' });
  for (let i = 0; i <= n; i++) { dot(X(i), Y(CLIMB[i].j), 3, DINK[2]); if (i) rline(X(i - 1), Y(CLIMB[i - 1].j), X(i), Y(CLIMB[i].j), 864 + i, { c: DCOL[2], w: 1.2, once: true, j: .2 }); }
}
// the jump: Pip on a ledge, cheese ledge at 0.7 m, bell curve of jump strengths (μ, σ)
function jump(t, mu, sg, o = {}) {
  const X = d => 70 + d / 1.2 * 290, gy = 200;
  box(20, gy, X(0) - 20, 60, '#d9d1c6'); rrect(20, gy, X(0) - 20, 60, 890); box(X(.55), gy, X(1.2) - X(.55) + 10, 60, '#d9d1c6'); rrect(X(.55), gy, X(1.2) - X(.55) + 10, 60, 891);
  L.cheese(X(.7), gy - 10, .8); L.mouse(X(0) - 18, gy - 10, 1, .9, t);
  rline(X(0), gy + 76, X(1.2), gy + 76, 892, { c: C.line, w: 1 }); [0, .3, .7, 1.2].forEach(d => { rline(X(d), gy + 72, X(d), gy + 80, 893 + d * 10, { c: C.line, w: 1, once: true }); txt(d + ' m', X(d), gy + 90, { s: 12, c: C.soft }); });
  ctx.save(); ctx.globalAlpha = o.a ?? 1; ctx.fillStyle = 'rgba(111,183,180,.35)'; ctx.strokeStyle = DINK[2]; ctx.lineWidth = 1.8; ctx.beginPath();
  for (let i = 0; i <= 120; i++) { const d = i / 100, y = gy - 30 - 70 * Math.exp(-(((d - mu) / sg) ** 2) / 2); i ? ctx.lineTo(X(d), y) : ctx.moveTo(X(d), y); } ctx.stroke(); ctx.restore();
  txt(`μ = ${mu}`, X(mu), gy - 112, { s: 14, c: DINK[2], a: o.a ?? 1 });
}

const STEPS = [
 { h: 'Two ways to get better', p: ['In the Monte Carlo and Q-learning reels, Pip learned a number for every move, Q, and then mostly picked the move with the biggest number, with a few random moves. Values first, choices second.', 'This reel tries the other way round. Pip keeps its **choices themselves**, as chances, and adjusts them directly: moves that led to good outcomes become more likely. These are **policy gradient methods**.'],
   shot: [3, 9],
   draw(t) { const A = L.gridWorld({ cs: 36, ox: 16, oy: 60 }), B = L.gridWorld({ cs: 36, ox: 206, oy: 60 });
     alpha(clamp(t - .3), () => { A.maze({ t }); 'RRRR./U.U.U/UL.RU/URRRU'.split('/').forEach((row, r) => row.split('').forEach((ch, c) => { const m = 'URDL'.indexOf(ch); if (m >= 0) A.arrowIn(c, r, m, 700 + r * 5 + c, { w: 1.3 }); }));
       txt('Q per move → mostly the biggest', A.ox + 90, 40, { s: 15, c: C.goldInk }); txt('one fixed arrow per cell', A.ox + 90, 222, { s: 14, c: C.soft }); });
     alpha(clamp(t - 2.5), () => { B.maze({ t }); fans(B, MZTH); txt('adjust the chances directly', B.ox + 90, 40, { s: 15, c: DINK[2] }); txt('a chance for every move', B.ox + 90, 222, { s: 14, c: C.soft }); });
     if (t > 4.5) card(['old way: learn how good each move is, pick the best', { s: 'new way: keep chances, make good moves likelier', c: DINK[2] }], t - 4.5, { gap: 1.4 }); } },
 { h: 'The door room', p: ['To see the idea clearly, start with one decision. Pip stands in a room with **three doors**, A, B and C. It picks one and walks through, and that ends the episode.', 'Behind each door a crumb appears with a hidden chance, worth reward 1; otherwise the reward is 0. Pip does not know the chances. The question marks on the doors stand for them.'],
   shot: [3, 9],
   draw(t) { room(t, { rows: false, pie: false, pip: false, doorA: clamp(t - .3) });
     const k = Math.floor(Math.max(0, t - 1.5) / 3) % 3, u = Math.max(0, t - 1.5) % 3, go = ease(u / 1.1) * (1 - ease((u - 2.2) / .8)), x = lerp(200, DX[k], go), y = lerp(236, 172, go);
     L.mouse(x, y, DX[k] < x - 1 || (u > 2.2 && DX[k] > 200) ? -1 : 1, 1, t);
     if (t > 1.5 && u > 1.1 && u < 2.4) { const hit = DEMO[Math.floor((t - 1.5) / 3) % DEMO.length]; txt(hit ? 'crumb! +1' : 'nothing: 0', DX[k], 120, { s: 17, c: hit ? C.goldInk : C.soft, a: clamp((u - 1.1) * 3) }); }
     if (t > 4) card(['one choice: door A, B or C, then the episode ends', 'reward 1 if a crumb appears, else 0', { s: 'crumb chances are hidden from Pip', c: C.soft }], t - 4, { gap: 1.2 }); } },
 { h: 'A policy made of chances', p: ['Pip’s policy is a chance for each door, drawn as a **pie**. At first it is a third each: Pip has no idea which door is best, so it tries all of them.', 'This is a stochastic policy from the policies reel, π of a given s. Here there is only one situation, the room, so the pie is the whole policy.'],
   shot: [3, 9],
   draw(t) { room(t, { prefA: 0, chA: clamp(t - 2), pieA: clamp(t - .6) });
     if (t > 4) card(['policy = a chance per door', { s: 'now: A 0.33, B 0.33, C 0.33', c: C.goldInk }, { s: 'the chances always add up to 1', c: C.soft }], t - 4, { gap: 1.2 }); } },
 { h: 'Turn scores into chances', p: ['Pip keeps one **preference** number per door, any number, positive or negative. To turn them into chances it raises e, about 2.72, to the power of each preference, then divides each by their total. This recipe is called **softmax**.', `All preferences 0: e to the 0 is 1 for each door, the total is 3, so a third each. Raise C’s preference to 1: C gets ${E1.toFixed(2)}, the total is ${(2 + E1).toFixed(2)}, and C’s chance becomes ${f2(P1[2])}.`],
   shot: [3, 11],
   draw(t) { const u = ease((t - 6) / 1.2), th = [0, 0, u]; room(t, { th });
     const l = [{ s: 'chance = e^pref ÷ (sum of e^pref over doors)', c: C.ink }, 'prefs 0, 0, 0  →  1, 1, 1  ÷ 3  →  0.33 each'];
     if (t > 5.5) l.push({ s: `prefs 0, 0, 1  →  1, 1, ${E1.toFixed(2)}  ÷ ${(2 + E1).toFixed(2)}  →  ${f2(P1[0])}, ${f2(P1[1])}, ${f2(P1[2])}`, c: DINK[2] });
     card(l, t - .8, { gap: 1.8, fs: 16 }); } },
 { h: 'Why preferences?', p: [`Pip can push any preference up or down freely, and softmax always returns proper chances: all above 0 and adding up to 1. So no move ever gets chance exactly 0: Pip keeps trying every door now and then, which helps it explore, but it can only approach “always C”, never reach it.`, `Only the **differences** matter. Add 1 to every preference, from 0, 0, 1 to 1, 1, 2, and the chances stay ${f2(PS[0])}, ${f2(PS[1])}, ${f2(PS[2])}.`],
   shot: [3, 9],
   draw(t) { const u = ease((t - 4.5) / 1.2), th = [u, u, 1 + u]; room(t, { th });
     card(['softmax always gives proper chances', '   every chance above 0, total 1', { s: t > 4.5 ? 'add 1 to every preference: same chances' : ' ', c: C.goldInk }], t - .6, { gap: 1.3 }); } },
 { h: 'How good is a policy?', p: ['Let us peek at the hidden chances: A 0.2, B 0.5, C 0.8. Pip still does not know them; we use them to score Pip’s policy.', `The **expected return** of a policy is its average reward per episode: for each door, Pip’s chance of picking it times the crumb chance behind it, added up. With a third each: 0.2 plus 0.5 plus 0.8, divided by 3, gives ${f2(J(U))}. Always picking C would give 0.8.`],
   shot: [3, 10],
   draw(t) { room(t, { secret: t > 1 ? 1 : 0, glow: t > 1 && t < 2.2 ? 2 : null });
     card(['expected return = sum of  chance × crumb chance', `= 0.33 × 0.2 + 0.33 × 0.5 + 0.33 × 0.8 = ${f2(J(U))}`, { s: 'best possible: always C, 0.8', c: C.soft }], t - 3, { gap: 1.5 }); } },
 { h: 'Which way is uphill?', p: ['Now the key question: which way should Pip nudge each preference to raise the expected return? The answer is the **gradient**, as in machine learning, but we climb instead of descend.', `The chart keeps A and B at 0. Along the bottom: C’s preference. Up the side: expected return. At 0 the curve rises with slope ${f2(SLOPE)}: raise C’s preference a little, and the return goes up about ${f2(SLOPE)} times as much. Then the dot steps uphill.`],
   shot: [4, 11],
   draw(t) { const v = t > 7 ? lerp(0, 1.2, ease((t - 7) / 2)) : 0; room(t, { secret: 1, th: [0, 0, v] }); hill(t, { p: (t - .8) / 2, at: t > 3 ? v : null, tan: clamp(t - 4.5) * (1 - clamp(t - 7)) }); } },
 { h: 'Every preference at once', p: [`The gradient has one entry per preference. For softmax it comes out simple: Pip’s chance of a door times how much that door beats the current expected return. A: 0.33 × (0.2 − 0.5) = ${f2(G0[0])}. B: ${fz(G0[1])}, it matches the average, for now. C: +${f2(G0[2])}.`, 'Why? Raising C’s preference grows C’s chance and takes a little from A and B. Count gains minus losses, each times its crumb chance, and what remains is C’s chance × (0.8 − 0.5).'],
   shot: [4, 11],
   draw(t) { room(t, { secret: 1 });
     G0.forEach((g, i) => { const a = clamp(t - 1.5 - i * 1.2); if (a <= 0) return; const len = g / .1 * 26; if (Math.abs(g) > 1e-9) alpha(a, () => arrow(DX[i] + 44, 178 + len / 2, DX[i] + 44, 178 - len / 2, 900 + i, { c: g > 0 ? DINK[2] : C.roseInk, w: 2.4 }));
       txt(`${g > 0 ? '+' : ''}${fz(g)}`, DX[i], 198, { s: 16, c: g > 0 ? DINK[2] : g < 0 ? C.roseInk : C.soft, a }); });
     card(['raise C’s preference a little:', '   C gains chance(C) × (1 − chance(C)), each worth 0.8', '   A, B each lose chance(·) × chance(C), worth 0.2, 0.5', { s: '   net: chance(C) × (0.8 − expected return)', c: C.goldInk }, { s: `gradient:  A ${f2(G0[0])}   B ${fz(G0[1])}   C +${f2(G0[2])}`, c: C.goldInk }], t - 5, { gap: 1.2, fs: 15, lh: 20 }); } },
 { h: 'Climbing, nudge by nudge', p: [`Pip adds the gradient times a **step size** α (alpha) = ${AL} to the preferences, then works out the new gradient, and repeats. The gradient only describes the slope right where Pip stands, so it takes many small steps.`, `After 5 nudges the expected return is ${f2(CLIMB[5].j)}, after ${NCL} it is ${f2(CLIMB[NCL].j)}, and C’s chance has grown to ${f2(CLIMB[NCL].p[2])}. The gains shrink as Pip nears the best possible 0.8.`],
   shot: [3, 12],
   draw(t) { const n = clamp(Math.floor((t - 1) / .45), 0, NCL); room(t, { secret: 1, th: CLIMB[n].th }); climbChart(n); caption(`after ${n} nudge${n === 1 ? '' : 's'}: expected return ${f2(CLIMB[n].j)}`); } },
 { h: 'Small nudges, small changes', p: ['Why chances? First, they change **smoothly**. Suppose Pip had guesses A 0.2, B 0.51, C 0.49 and always picked the biggest. Move B and C by 0.02 each, to 0.49 and 0.51, and its choice flips completely from B to C.', `With preferences A 0, B 0.01, C −0.01, moving B and C by 0.02 each only shifts the chances from B ${f2(SM0[1])}, C ${f2(SM0[2])} to B ${f2(SM1[1])}, C ${f2(SM1[2])}. Learning becomes a gentle slide, not sudden jumps.`],
   shot: [3, 10],
   draw(t) { const flip = t > 4.5, u = ease((t - 4.5) / 1);
     txt('pick the biggest guess', 100, 40, { s: 15, c: C.roseInk }); txt('softmax of preferences', 300, 40, { s: 15, c: DINK[2] });
     const g = flip ? [.49, .51] : [.51, .49]; txt(`A 0.2  B ${g[0]}  C ${g[1]}`, 100, 70, { s: 15 }); pie(100, 150, 46, flip ? [0, 0, 1] : [0, 1, 0], { labels: true });
     const th = [0, lerp(.01, -.01, u), lerp(-.01, .01, u)], p = softmax(th); txt(`A 0  B ${f2(th[1])}  C ${f2(th[2])}`, 300, 70, { s: 15 }); pie(300, 150, 46, p, { labels: true });
     txt(flip ? 'all C' : 'all B', 100, 215, { s: 15, c: C.roseInk }); txt(`A ${f2(p[0])}  B ${f2(p[1])}  C ${f2(p[2])}`, 300, 215, { s: 15, c: DINK[2] });
     if (t > 6) card(['tiny change in guesses: choice flips completely', { s: 'tiny change in preferences: chances move a little', c: DINK[2] }], t - 6, { gap: 1.3 }); } },
 { h: 'Any strength you like', p: ['Second, actions need not be a short list. Suppose Pip must jump a gap with any strength it likes. A table of guesses would need an entry for every possible strength.', 'A policy can instead be a bell curve over strengths, set by two numbers: its centre **μ** (mu) and its spread **σ** (sigma). The same idea applies: nudge μ toward strengths that landed better, and σ can be learned too. Later reels show exactly how.'],
   shot: [3, 10],
   draw(t) { const mu = +lerp(.3, .7, ease((t - 5) / 2.5)).toFixed(2); jump(t, mu, .12, { a: clamp(t - 1.5) });
     if (t > 7.5) card(['continuous action: jump strength in metres', { s: 'policy: bell curve, centre μ and spread σ', c: DINK[2] }], t - 7.5, { gap: 1.2, y: 300 }); } },
 { h: 'Random on purpose', p: ['Third, some problems need a random policy. In rock, paper, scissors, any fixed choice gets beaten once the opponent spots it; the best play is a third each.', 'This matters against opponents that adapt, or when Pip cannot see everything; in the fixed maze one move per cell is enough. Picking the biggest guess, even with ε-greedy’s random moves, cannot learn a mix like a third each. A policy of chances can.'],
   shot: [3, 9],
   draw(t) { const xs = [110, 200, 290]; ['rock', 'paper', 'scissors'].forEach((s, i) => { const a = clamp(t - .5 - i * .4); alpha(a, () => { rcircle(xs[i], 110, 26, 970 + i, { fill: C.card }); txt(['✊', '✋', '✌'][i], xs[i], 111, { s: 24 }); txt(s, xs[i], 152, { s: 15, c: C.soft }); txt('1/3', xs[i], 70, { s: 16, c: C.goldInk }); }); });
     if (t > 3) pie(200, 228, 34, [1 / 3, 1 / 3, 1 / 3], { cols: [C.yellow, C.blue, C.pink], names: ['R', 'P', 'S'], labels: true });
     if (t > 4.5) card(['biggest guess: always the same move, easy to beat', { s: 'chances: can learn a third each', c: DINK[2] }], t - 4.5, { gap: 1.3 }); } },
 { h: 'Back in the maze', p: ['The same idea works in Pip’s maze. Each cell gets its own four preferences, one per move, and its own softmax. With 16 cells that is 64 numbers.', 'In each cell the four arrows show the chances: the longer the arrow, the more likely the move. All preferences start at 0, so every arrow is a quarter, the random Pip from the first reel. Raise one preference, and its arrow grows.'],
   shot: [3, 10],
   draw(t) { const th = MZTH.slice(), u = 2 * ease((t - 7) / 1.5), k = MZ.key(0, 2); th[k * 4] = u; MZ.maze({ t, labels: clamp(t - 1), hi: t > 6.5 ? [[0, 2, C.gold, .25]] : null }); fans(MZ, th, { a: clamp(t - .5) }); MZ.pip(MZ.cx(0), MZ.cy(3), 1, t);
     const l = ['each cell: 4 preferences → softmax → 4 chances', { s: 'all 0 at first: a quarter each', c: C.soft }]; if (t > 7) l.push({ s: `example: “up” preference ${f2(u)} → up ${f2(softmax(th.slice(k * 4, k * 4 + 4))[0])}`, c: C.goldInk });
     if (t > 4) card(l, t - 4, { gap: 1.2 }); } },
 { h: 'The catch', p: ['Everything so far used the hidden crumb chances. The gradient needed them, and Pip does not have them. In the maze it would also need the rulebook, because where each move leads decides which rewards follow.', 'So Pip must **estimate** the gradient from its own tries: pick a door, see what happens, nudge. That is the next reel, REINFORCE. Its nudges are noisy, which is the theme of the reels after it.'],
   shot: [3, 9],
   draw(t) { room(t, { secret: t < 2 ? 1 : 1 - clamp((t - 2) * 2) });
     if (t > 3) card([{ s: 'exact gradient: needs the hidden chances  ✗', c: C.roseInk }, { s: 'estimate it from tries: next reel, REINFORCE', c: DINK[2] }], t - 3, { gap: 1.4 }); } },
 { h: 'Your turn', tap: tapDoor, p: ['Tap a door to raise its preference by 0.5. Tap the floor to reset. Watch the chances, the expected return, and the gradient arrows telling you which way is uphill.', 'Try raising A a few times, then C. Or tap all three doors once each: the chances do not change.'],
   shot: [3, 8],
   draw(t, st) { const th = st.th || U, g = grad(th); room(t, { secret: 1, th, glow: st.last });
     g.forEach((v, i) => { const len = clamp(v / .1, -1.4, 1.4) * 22; if (Math.abs(v) > .015) arrow(DX[i] + 44, 178 + len / 2, DX[i] + 44, 178 - len / 2, 910 + i, { c: v > 0 ? DINK[2] : C.roseInk, w: 2.2 }); });
     card([`expected return ${f2(J(th))}   (best 0.8)`, { s: `gradient  A ${f2(g[0])}  B ${f2(g[1])}  C ${f2(g[2])}`, c: C.goldInk }], 9, { gap: 0 });
     caption('tap a door: +0.5 · tap the floor: reset'); } },
 { h: 'Where it is used', p: ['Policy gradient methods train controllers for simulated and real robots, where actions are motor strengths. **PPO**, a policy gradient method from 2017, trained OpenAI Five, which beat the Dota 2 world champions in 2019.', 'AlphaGo’s policy network was improved by policy gradient in self-play, and PPO-style training is used to fine-tune chatbots from human feedback.'],
   shot: [3, 9],
   draw(t) { alpha(clamp(t - .5), () => { L.robot(90, 130, 2, t); txt('robots: motor strengths', 90, 190, { s: 15 }); });
     alpha(clamp(t - 2), () => { rrect(170, 95, 60, 60, 950, { fill: C.card }); txt('Dota 2', 200, 125, { s: 16 }); txt('games', 200, 190, { s: 15 }); });
     alpha(clamp(t - 3.5), () => { rrect(270, 100, 80, 46, 951, { fill: C.card }); txt('hi! …', 310, 123, { s: 16 }); txt('chatbots', 310, 190, { s: 15 }); });
     if (t > 5) card(['all keep a policy of chances (or a bell curve)', { s: 'and nudge it uphill along a gradient estimate', c: C.soft }], t - 5, { gap: 1.2 }); } },
 { h: 'Recap', p: ['A **policy gradient method** keeps the policy itself as adjustable numbers and nudges them to raise the expected return. Softmax turns preferences into chances.', 'The gradient says which way to nudge each preference: here, toward doors that beat the current average. Chances change smoothly, handle any action, and can stay random on purpose.'],
   shot: [3, 9],
   draw(t) { room(t, { secret: 1, th: CLIMB[NCL].th });
     card(['preferences → softmax → chances', 'expected return: average reward of the policy', 'gradient: which way to nudge each preference', { s: 'climb in small steps of size α', c: C.soft }], t - .3, { gap: 1 }); } },
];

function tapDoor(x, y, st, t) {
  const th = (st.th || U).slice(); const i = DX.findIndex(d => Math.abs(x - d) < 36 && y > 40 && y < 150);
  if (i < 0) { if (y > 150 && y < 285) { st.th = U; st.last = null; } return; }
  th[i] += .5; st.th = th; st.last = i;
}

RLStory.register('pg', { title: 'Policy Gradient Methods', track: 'Policy Gradients', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { J0: J(U), G0, P1, PS, SLOPE, climb: CLIMB.map(c => [c.j, c.p[2]]) } });
})();
