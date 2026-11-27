// GRPO as a story: Bot (from the RLHF reel) does Pip's sums, like 3 + 4 × 2. A program checks each final answer: 1 if right, 0 if wrong.
// Bot's "policy" is a stand-in: chances over four ways of answering (guess, left to right, times first, times first and check).
// For each question Bot answers 8 times; each answer's advantage is its score minus the group's average, divided by the group's spread.
// No critic. KL leash to the original Bot. Every number comes from the seeded simulation below (reel.sim); the sums are really computed.
(function () {
'use strict';
const L = RLStory.lib, { W, C, lerp, clamp, ease, rng, rline, rrect, rcircle, txt, arrow, card, bubble, caption, dot, box, alpha } = L;
let ctx;

// ───── the world: questions, four ways of answering, the checker
const SN = ['guess', 'left to right', 'times first', 'times first, then check'];
const SCOL = ['#c9b9a8', '#ef8f5a', '#8fb3d9', '#5d9a5b'], SINK = ['#8a8178', '#c4581f', '#4f7fb0', '#3f7a3d'];
const REF = [0.3, 0.4, 0.2, 0.1], SLIP = [0, 0, 0.15, 0.03];
const ri = (R, a, b) => a + Math.floor(R() * (b - a + 1));
function question(R, hard) { const a = ri(R, 2, 9), b = ri(R, 2, 9), c = ri(R, 2, 9); return hard ? { a, b, c, hard: 1, ans: a + b * c, s: `${a} + ${b} × ${c}` } : { a, b, hard: 0, ans: a + b, s: `${a} + ${b}` }; }
function answer(q, s, R) { const exact = q.ans;
  if (s === 0) return exact + ri(R, -3, 3);                                   // a guess near the right size
  if (s === 1) return q.hard ? (q.a + q.b) * q.c : exact;                      // works left to right: wrong whenever × follows +
  return R() < SLIP[s] ? exact + (R() < .5 ? -1 : 1) : exact; }               // right method, with a small slip (the check catches most)
const check = (q, x) => x === q.ans ? 1 : 0;
// accuracy of each way on 400 test questions (half easy a + b, half a + b × c); only the simulation knows these
const TEST = (() => { const R = rng(99), T = []; for (let i = 0; i < 400; i++) T.push(question(R, i % 2)); return T; })();
const ACC = [0, 1, 2, 3].map(s => { const R = rng(500 + s); return TEST.filter(q => check(q, answer(q, s, R))).length / TEST.length; });
const softmax = th => { const m = Math.max(...th), e = th.map(x => x === -Infinity ? 0 : Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
const accOf = P => P.reduce((s, p, i) => s + p * ACC[i], 0);
function pick(P, R) { let x = R(), s = 0; while (s < 3 && x >= P[s]) { x -= P[s]; s++; } return s; }
// one group: G answers to the same question, scored, advantage = (score − group average) ÷ group spread (0 if all scores equal)
function group(q, P, G, R) { const S = []; for (let i = 0; i < G; i++) { const s = pick(P, R), x = answer(q, s, R); S.push({ s, x, r: check(q, x) }); }
  const mu = S.reduce((a, y) => a + y.r, 0) / G, sd = Math.sqrt(S.reduce((a, y) => a + (y.r - mu) ** 2, 0) / G); S.forEach(y => { y.A = sd ? (y.r - mu) / sd : 0; }); return { q, S, mu, sd }; }
// GRPO on the four chances: per update NQ questions (each easy or hard at random) × G answers; one gradient step (ratio 1, so the clip never acts);
// KL leash β to the original Bot; step size LR.
const BETA = 0.04, LR = 1, NU = 40;
function gradStep(th, P, groups, ref) { const g = [0, 0, 0, 0], n = groups.reduce((a, gr) => a + gr.S.length, 0);
  for (const gr of groups) for (const y of gr.S) for (let b = 0; b < 4; b++) g[b] += y.A * ((b === y.s) - P[b]) / n;
  const kl = P.reduce((a, p, i) => a + (p > 0 ? p * Math.log(p / ref[i]) : 0), 0); for (let b = 0; b < 4; b++) if (P[b] > 0) g[b] -= BETA * P[b] * (Math.log(P[b] / ref[b]) - kl);
  return th.map((v, i) => v + LR * g[i]); }
function train(G, NQ, seed, ref = REF) { const R = rng(seed); let th = ref.map(Math.log); const H = [{ P: ref.slice(), acc: accOf(ref), silent: null, silentE: null, silentH: null }];
  for (let u = 0; u < NU; u++) { const P = softmax(th), groups = []; for (let k = 0; k < NQ; k++) groups.push(group(question(R, R() < .5 ? 1 : 0), P, G, R));
    th = gradStep(th, P, groups, ref); const Pn = softmax(th), sil = gr => gr.sd === 0, E = groups.filter(gr => !gr.q.hard), Hh = groups.filter(gr => gr.q.hard);
    H.push({ P: Pn, acc: accOf(Pn), silent: groups.filter(sil).length / NQ, nE: E.length, sE: E.filter(sil).length, nH: Hh.length, sH: Hh.filter(sil).length }); }
  return H; }
// ───── the story run: groups of 8, 8 questions per update (64 answers), seed 7
const RUN = train(8, 8, 7), FIN = RUN[NU];
const silentAt = (a, b) => { let s = 0, n = 0, sE = 0, nE = 0, sH = 0, nH = 0; for (let i = a; i <= b; i++) { s += RUN[i].silent * 8; n += 8; sE += RUN[i].sE; nE += RUN[i].nE; sH += RUN[i].sH; nH += RUN[i].nH; } return { all: s / n, e: sE / nE, h: sH / nH }; };
const SIL0 = silentAt(1, 10), SIL1 = silentAt(31, 40);
// the example group: the first hard question from rng(31) whose group of 8 (at the starting chances) mixes right and wrong
const EX = (() => { const R = rng(31); for (;;) { const q = question(R, 1), gr = group(q, REF, 8, R); const k = gr.S.reduce((a, y) => a + y.r, 0); if (k >= 2 && k <= 4 && gr.S.some(y => y.s === 0) && gr.S.some(y => y.s === 1)) return gr; } })();
const EXK = EX.S.reduce((a, y) => a + y.r, 0), EXP1 = softmax(gradStep(REF.map(Math.log), REF, [EX], REF));
const EXQ = EX.q, EXW = [0, 1, 2, 3].map(s => answer(EXQ, s, () => .7));   // sample answers to show for each way (no slip, guess offset +2)
// the "your turn" runs: group size G with 64 answers per update; average of 20 runs
const GS = [2, 4, 8, 16], gm = {};
function gStats(G) { if (gm[G]) return gm[G]; let acc = 0, sil = 0; const one = train(G, 64 / G, 7);
  for (let k = 0; k < 20; k++) { const H = train(G, 64 / G, 100 + k); acc += H[NU].acc / 20; sil += H.slice(1).reduce((a, h) => a + h.silent, 0) / NU / 20; }
  return (gm[G] = { acc, sil, one }); }
// the missing way: if Bot never checks (chance 0), no group ever contains a check
const REFN = [0.33, 0.44, 0.23, 0]; const RUNN = train(8, 8, 7, REFN), FINN = RUNN[NU];

// ───── formatting and drawing helpers
const f2 = v => (Math.abs(v) < .005 ? 0 : v).toFixed(2).replace('-', '−'), sg = v => (v > .005 ? '+' : '') + f2(v);
const GREEN = '#5d9a5b', BLUE = '#4f7fb0';
function say(s, x, y, w, o = {}) { const h = o.h ?? 26, a = o.a ?? 1; if (a <= 0) return;
  alpha(a, () => { box(x, y, w, h, o.fill || C.card); rrect(x, y, w, h, o.seed || 500, { c: o.c || C.line, w: o.lw || 1.2 });
    if (o.tx != null) { const bx = clamp(o.tx, x + 10, x + w - 10), by = o.ty < y ? y : y + h; rline(bx - 5, by, o.tx, o.ty, (o.seed || 500) + 9, { c: o.c || C.line, w: 1.2, once: true }); rline(bx + 5, by, o.tx, o.ty, (o.seed || 500) + 10, { c: o.c || C.line, w: 1.2, once: true }); }
    txt(s, x + w / 2, y + h / 2 + 1, { s: o.s || 15, c: o.tc }); }); }
function scene(t, q, o = {}) { L.mouse(56, 96, 1, 1.1, t); L.robot(330, 92, 2, t); say(`Bot, what is ${q.s}?`, 18, 26, 190, { tx: 66, ty: 74, seed: 510 }); }
// chances as a row of bars with names
function chances(P, y0, o = {}) { P.forEach((p, s) => { const y = y0 + s * 22, a = o.a ? o.a(s) : 1; if (a <= 0) return;
  txt(SN[s], 22, y, { s: 14, al: 'left', c: SINK[s], a }); box(200, y - 6, p * 140, 12, SCOL[s], .85 * a); if (o.ref) rline(200 + o.ref[s] * 140, y - 8, 200 + o.ref[s] * 140, y + 8, 3000 + s, { c: C.ink, w: 1.2, once: true, j: .2 });
  txt(f2(p), 378, y, { s: 14, al: 'right', a }); }); }
function chartAcc(H, n, o = {}) { const x0 = 60, x1 = 340, y0 = o.y0 ?? 150, y1 = o.y1 ?? 50, X = i => x0 + i / NU * (x1 - x0), Y = v => y0 - v * (y0 - y1);
  rline(x0 - 4, y0, x1 + 4, y0, 3100, { c: C.line, w: 1, once: true }); rline(x0 - 4, y0, x0 - 4, y1 - 2, 3101, { c: C.line, w: 1, once: true });
  [0, .5, 1].forEach(v => txt(String(v), x0 - 9, Y(v), { s: 12, c: C.soft, al: 'right' })); txt('update 0', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt('update ' + NU, x1, y0 + 11, { s: 12, c: C.soft, al: 'right' });
  txt(o.title || 'test accuracy: share of test sums Bot gets right', x0 - 8, y1 - 16, { s: 13, c: C.soft, al: 'left' });
  ctx.save(); ctx.strokeStyle = o.col || BLUE; ctx.lineWidth = 2.2; ctx.beginPath(); for (let i = 0; i <= n; i++) i ? ctx.lineTo(X(i), Y(H[i].acc)) : ctx.moveTo(X(i), Y(H[i].acc)); ctx.stroke(); ctx.restore();
  dot(X(n), Y(H[n].acc), 3.5, o.col || BLUE); txt(f2(H[n].acc), X(n) + 8, Y(H[n].acc) + (o.dy ?? -10), { s: 13, c: o.col || BLUE, al: 'left' }); }
function answerCard(gr, t, o = {}) { const x0 = o.x ?? 22, y0 = o.y ?? 150;
  gr.S.forEach((y, i) => { const u = clamp((t - (o.t0 ?? 0) - i * (o.gap ?? .35)) * 3); if (u <= 0) return; const x = x0 + (i % 4) * 92, yy = y0 + Math.floor(i / 4) * 64;
    alpha(u, () => { box(x, yy, 84, 54, C.card); rrect(x, yy, 84, 54, 3200 + i, { c: y.r ? GREEN : C.rose, w: 1.6 }); txt(String(y.x), x + 42, yy + 16, { s: 18 }); txt(SN[y.s].split(',')[0], x + 42, yy + 34, { s: 11, c: SINK[y.s] });
      txt(y.r ? '✓ 1' : '✗ 0', o.adv ? x + 8 : x + 42, yy + 46, { s: 12, c: y.r ? GREEN : C.roseInk, al: o.adv ? 'left' : 'center' });
      if (o.adv) txt(sg(o.norm ? y.A : y.r - gr.mu), x + 79, yy + 46, { s: 12, c: y.r ? GREEN : C.roseInk, al: 'right', a: clamp(t - (o.advT ?? 0)) }); }); }); }

const STEPS = [
 { h: 'Bot’s homework', p: ['Bot, the answer machine from the RLHF reel, now helps Pip with sums, like 4 + 7 × 3. These answers can be **checked by a program**: compute the true value, compare the final number. Right scores 1, wrong scores 0.',
   'No people comparing answers and no reward model to fool: just a cheap, exact check. This is a **verifiable reward**.'],
   shot: [4, 11],
   draw(t) { const q = { s: '4 + 7 × 3', ans: 25 }; scene(t, q); if (t > 2) say('25', 250, 142, 70, { tx: 316, ty: 116, a: clamp(t - 2), seed: 520 });
     card(['true value: 4 + 21 = 25', { s: 'Bot said 25  →  right, score 1', c: GREEN }, { s: 'any other number  →  wrong, score 0', c: C.roseInk }], t - 4, { y: 200, gap: 1.2 }); } },
 { h: 'Four ways to answer', p: ['A real language model writes its working word by word. Bot is a stand-in: it picks one of four **ways of answering**, each with a chance, learned by copying examples.',
   `Guess a number. Work left to right, which goes wrong whenever times follows plus. Do times first, with an occasional slip. Or do times first and then check, which catches most slips.`],
   shot: [4, 11],
   draw(t) { L.robot(200, 60, 1.8, t); txt(`Bot’s starting chances`, 22, 116, { s: 13, c: C.soft, al: 'left' }); chances(REF, 140, { a: s => clamp((t - 1 - s * .8) * 3) });
     const q = EXQ; if (t > 5) { txt(`on ${q.s} (true value ${q.ans}):`, 22, 238, { s: 14, al: 'left', a: clamp(t - 5) });
       card(SN.map((n, s) => ({ s: `${n}: ${s === 0 ? 'some number near it' : s === 1 ? `(${q.a} + ${q.b}) × ${q.c} = ${(q.a + q.b) * q.c}` : s === 2 ? `${q.a} + ${q.b * q.c} = ${q.ans}, but may slip` : `${q.ans}, slips mostly caught`}`, c: SINK[s] })), t - 5.5, { y: 252, gap: .8, fs: 14, lh: 21 }); } } },
 { h: 'How good is each way?', p: [`Only the simulation knows this. On 400 test sums, half like 3 + 5 and half like 3 + 5 × 2: guessing gets ${f2(ACC[0])} right, left to right ${f2(ACC[1])}, times first ${f2(ACC[2])}, times first and check ${f2(ACC[3])}.`,
   `With its starting chances, Bot gets ${f2(accOf(REF))} of the test sums right. Bot never sees this table; it only sees the checker’s 1s and 0s.`],
   shot: [4, 11],
   draw(t) { L.robot(200, 60, 1.8, t); txt('share of test sums right', 22, 116, { s: 13, c: C.soft, al: 'left' });
     ACC.forEach((a, s) => { const y = 140 + s * 22, u = clamp((t - .8 - s * .7) * 3); if (u <= 0) return; txt(SN[s], 22, y, { s: 14, al: 'left', c: SINK[s], a: u }); box(200, y - 6, a * 140 * ease(u), 12, SCOL[s], .85); txt(f2(a), 378, y, { s: 14, al: 'right', a: u }); });
     card([`Bot at the start: ${f2(accOf(REF))} right`, { s: '= chances × accuracies, added up', c: C.soft }], t - 4.5, { y: 250, gap: 1 }); } },
 { h: 'A group of eight', p: [`Take one question, ${EXQ.s}. Bot answers it **8 times**, a **group**, each time picking its way of answering by chance. The checker marks each one.`,
   `${EXK} of the 8 are right. The small names show which way Bot used: both right answers did times first; a third times-first answer slipped to ${EX.S.find(y => y.s === 2 && !y.r).x}.`],
   shot: [5, 11],
   draw(t) { scene(t, EXQ); answerCard(EX, t, { t0: 1, gap: .45, y: 150 });
     card([`${EXK} right, ${8 - EXK} wrong`, { s: `true value ${EXQ.ans}`, c: C.soft }], t - 5.5, { y: 300, gap: 1 }); } },
 { h: 'Grade on the curve', p: [`The group’s average score is ${f2(EX.mu)}. That is the **yardstick**, as in the baselines reel. Each answer’s advantage is its score minus the average: right answers get ${sg(1 - EX.mu)}, wrong ones ${sg(-EX.mu)}.`,
   `GRPO, **Group Relative Policy Optimization**, also divides by the group’s spread, here ${f2(EX.sd)}: so ${sg((1 - EX.mu) / EX.sd)} and ${sg(-EX.mu / EX.sd)}. Above-average answers get pushed up, the rest down.`],
   shot: [4, 11],
   draw(t) { scene(t, EXQ); answerCard(EX, 9, { y: 150, adv: true, norm: t > 6, advT: 1.5 });
     card([`average ${f2(EX.mu)},  spread ${f2(EX.sd)}`, t > 6 ? { s: `advantage = (score − ${f2(EX.mu)}) ÷ ${f2(EX.sd)}`, c: C.goldInk } : { s: `advantage = score − ${f2(EX.mu)}`, c: C.goldInk }], t - 2.5, { y: 300, gap: 1 }); } },
 { h: 'No critic needed', p: ['PPO usually trains a **critic**, a second network that predicts the expected score, to serve as the yardstick. For a language model the critic is about as big as the model itself.',
   'GRPO uses the group’s average instead. The price is several answers per question; the gain is one less big network to train and store.'],
   shot: [4, 11],
   draw(t) { const u = clamp(t - 3);
     alpha(1 - u * .6, () => { L.robot(100, 90, 2, t); L.robot(170, 90, 2, t, '#e3c7a0'); txt('model', 100, 140, { s: 14, c: C.soft }); txt('critic', 170, 140, { s: 14, c: C.soft }); txt('PPO', 135, 30, { s: 18 }); });
     if (t > 3) alpha(u, () => { L.robot(290, 90, 2, t); txt('model', 290, 140, { s: 14, c: C.soft }); txt('GRPO', 290, 30, { s: 18 }); for (let i = 0; i < 8; i++) dot(250 + i * 11, 165, 4, i < EXK ? GREEN : C.rose); txt('group average = yardstick', 290, 186, { s: 13, c: C.goldInk }); });
     card(['PPO: yardstick from a learned critic', { s: 'GRPO: yardstick from the group average', c: C.goldInk }, { s: 'cost: several answers per question', c: C.soft }], t - 5, { y: 240, gap: 1.1 }); } },
 { h: 'One update', p: [`Now nudge the chances: the more an answer beat its group, the more its way of answering gains. From this one group, times first rises from ${f2(REF[2])} to ${f2(EXP1[2])}, left to right falls from ${f2(REF[1])} to ${f2(EXP1[1])}.`,
   `As in the RLHF reel, a **KL leash** with β (beta) = ${BETA} pulls toward the original Bot. Real GRPO also clips the ratio like PPO; we take one pass per batch, so the clip never acts.`],
   shot: [4, 11],
   draw(t) { const u = ease((t - 2) / 2), P = REF.map((p, i) => lerp(p, EXP1[i], u)); txt('Bot’s chances (black tick: before)', 22, 40, { s: 13, c: C.soft, al: 'left' }); chances(P, 66, { ref: REF });
     card(['one step: chances × advantages, step size 1', `KL leash β = ${BETA} toward the original`, { s: 'one pass per batch: the clip never acts', c: C.soft }], t - 4, { y: 180, gap: 1 }); } },
 { h: 'A training run', p: [`Each update: 8 questions, half easy on average, 8 answers each, so 64 answers. The chart shows test accuracy after each update; the bars, Bot’s chances.`,
   `After ${NU} updates, accuracy rises from ${f2(RUN[0].acc)} to ${f2(FIN.acc)}. Guessing and left to right fade to ${f2(FIN.P[0])} and ${f2(FIN.P[1])}; times first and check grows to ${f2(FIN.P[3])}. Even times first drops to ${f2(FIN.P[2])}: its slips lose out to the checked answers in the same groups.`],
   shot: [4, 12],
   draw(t) { const n = Math.round(clamp((t - 1) / 7) * NU); chartAcc(RUN, n); txt(`Bot’s chances after update ${n}`, 22, 196, { s: 13, c: C.soft, al: 'left' }); chances(RUN[n].P, 220, { ref: REF });
     caption('black tick: the starting chance', clamp(t - 2)); } },
 { h: 'Silent groups', p: [`If all 8 answers are right, or all wrong, every advantage is 0: the group teaches nothing. Over updates 1 to 10, ${Math.round(SIL0.e * 100)} percent of groups on easy sums were silent, and ${Math.round(SIL0.h * 100)} percent on hard ones.`,
   `Over updates 31 to 40, as Bot improves, that rises to ${Math.round(SIL1.e * 100)} and ${Math.round(SIL1.h * 100)} percent. Useful questions are ones Bot sometimes gets right and sometimes wrong.`],
   shot: [4, 11],
   draw(t) { const x0 = 60, x1 = 340, y0 = 200, y1 = 60, X = i => x0 + (i - 1) / (NU - 1) * (x1 - x0), Y = v => y0 - v * (y0 - y1);
     rline(x0 - 4, y0, x1 + 4, y0, 3300, { c: C.line, w: 1, once: true }); [0, .5, 1].forEach(v => txt(String(v), x0 - 9, Y(v), { s: 12, c: C.soft, al: 'right' }));
     txt('share of the 8 groups that were silent, per update', x0 - 8, y1 - 16, { s: 13, c: C.soft, al: 'left' }); txt('update 1', x0, y0 + 11, { s: 12, c: C.soft, al: 'left' }); txt('update ' + NU, x1, y0 + 11, { s: 12, c: C.soft, al: 'right' });
     const n = Math.round(clamp((t - 1) / 5) * NU); for (let i = 1; i <= n; i++) box(X(i) - 2.5, Y(RUN[i].silent), 5, y0 - Y(RUN[i].silent), C.grey, .9);
     card([`updates 1–10: easy ${Math.round(SIL0.e * 100)}%, hard ${Math.round(SIL0.h * 100)}% silent`, `updates 31–40: easy ${Math.round(SIL1.e * 100)}%, hard ${Math.round(SIL1.h * 100)}% silent`, { s: 'all right or all wrong: advantage 0', c: C.soft }], t - 6, { y: 250, gap: 1.1 }); } },
 { h: 'Your turn', ctl: { label: 'group', values: GS, def: 2, fmt: v => String(v) },
   p: ['Drag the **group size**. Every update still uses 64 answers, so bigger groups mean fewer questions. The curve shows the chance that a group is not silent, for a question Bot gets right with chance p.',
   'The card gives the average of 20 training runs after 40 updates. Tiny groups are often silent; bigger ones give a steadier yardstick.'],
   shot: [4, 10],
   draw(t, st) { const G = GS[st.ci ?? 2], S = gStats(G), x0 = 60, x1 = 340, y0 = 200, y1 = 60, X = p => x0 + p * (x1 - x0), Y = v => y0 - v * (y0 - y1);
     rline(x0, y0, x1, y0, 3400, { c: C.line, w: 1, once: true }); rline(x0, y0, x0, y1, 3401, { c: C.line, w: 1, once: true });
     [0, .5, 1].forEach(v => { txt(String(v), X(v), y0 + 12, { s: 12, c: C.soft }); txt(String(v), x0 - 8, Y(v), { s: 12, c: C.soft, al: 'right' }); });
     txt('p: chance Bot gets this question right', x1, y0 + 28, { s: 13, c: C.soft, al: 'right' }); txt('chance the group is not silent', x0 - 8, y1 - 16, { s: 13, c: C.soft, al: 'left' });
     ctx.save(); ctx.strokeStyle = BLUE; ctx.lineWidth = 2.2; ctx.beginPath(); for (let k = 0; k <= 100; k++) { const p = k / 100, v = 1 - p ** G - (1 - p) ** G; k ? ctx.lineTo(X(p), Y(v)) : ctx.moveTo(X(p), Y(v)); } ctx.stroke(); ctx.restore();
     card([`group of ${G}, ${64 / G} questions per update`, { s: `test accuracy after ${NU} updates: ${f2(S.acc)}`, c: BLUE }, { s: `silent groups: ${Math.round(S.sil * 100)}% on average`, c: C.soft }], 1, { y: 252, gap: 0 }); } },
 { h: 'It sharpens, it does not invent', p: [`Suppose Bot never checks its work: that chance starts at 0. Then no group ever contains a check, and nothing can reward it. The same training reaches ${f2(FINN.acc)}, against ${f2(FIN.acc)} with checking available.`,
   'GRPO raises the ways Bot already sometimes uses. And a checker is only as honest as its rules: one that accepted any answer containing the right number could be gamed.'],
   shot: [4, 11],
   draw(t) { const n = Math.round(clamp((t - 1) / 5) * NU); chartAcc(RUN, n, { col: '#b9cde3' }); chartAcc(RUNN, n, { col: C.orange, title: ' ', dy: 8 });
     txt(`Bot’s chances, never checking, after update ${n}`, 22, 196, { s: 13, c: C.soft, al: 'left' }); chances(RUNN[n].P, 220);
     caption('pale blue: the run with checking, from before', clamp(t - 2)); } },
 { h: 'Where it is used', p: ['GRPO was introduced in **DeepSeekMath** (Shao and colleagues, 2024) to train a model on maths problems. **DeepSeek-R1** (2025) used it with rule-based rewards: is the final answer right, is the format followed.',
   'During R1’s training its answers grew longer and more often contained checking and rethinking. Many labs now train reasoning models in a similar way.'],
   shot: [3, 10],
   draw(t) { alpha(clamp(t - .5), () => { rrect(40, 40, 140, 90, 3501, { fill: C.card }); txt('∫ x² dx = ?', 110, 70, { s: 16 }); txt('answer: x³/3 ✓', 110, 100, { s: 14, c: GREEN }); txt('2024: DeepSeekMath', 110, 150, { s: 14 }); });
     alpha(clamp(t - 3), () => { rrect(220, 40, 140, 90, 3502, { fill: C.card }); txt('think … wait,', 290, 64, { s: 14, c: C.soft }); txt('let me check …', 290, 84, { s: 14, c: C.soft }); txt('answer ✓', 290, 106, { s: 14, c: GREEN }); txt('2025: DeepSeek-R1', 290, 150, { s: 14 }); });
     card(['2024: GRPO, in DeepSeekMath', '2025: DeepSeek-R1, rule-based rewards', { s: 'groups of answers, no critic, KL leash', c: C.soft }], t - 5, { y: 200, gap: 1.1 }); } },
 { h: 'Recap', p: ['When answers can be checked by a program, the checker gives the reward. For each question, sample a **group** of answers and score them.',
   'Each answer’s advantage is its score against the group’s average, scaled by the spread: **no critic** needed. A KL leash keeps the model near where it started, and groups that are all right or all wrong teach nothing.'],
   shot: [3, 9],
   draw(t) { scene(t, EXQ); say(String(EXQ.ans), 250, 142, 70, { tx: 316, ty: 116, seed: 520, tc: GREEN });
     card(['1. checker: right 1, wrong 0', '2. a group of answers per question', '3. advantage: score vs group average', { s: '4. no critic; KL leash; silent groups teach nothing', c: C.goldInk }], t - .3, { y: 200, gap: 1, fs: 16 }); } },
];

RLStory.register('grpo', { title: 'RL for LLM Reasoning (GRPO)', track: 'Learning from people', onCtx: c => { ctx = c; }, steps: STEPS,
  sim: { ACC, acc0: accOf(REF), ex: { q: EXQ.s, ans: EXQ.ans, k: EXK, mu: EX.mu, sd: EX.sd, ways: EX.S.map(y => [y.s, y.x, y.r]), P1: EXP1 }, fin: { acc: FIN.acc, P: FIN.P }, SIL0, SIL1, finN: FINN.acc,
    groups: () => GS.map(G => { const S = gStats(G); return [G, S.acc, S.sil]; }) } });
})();
