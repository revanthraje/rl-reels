// Turns slide content into the sentence that gets narrated. Shared by app.js and tts/export.html
// (the pre-recorded clips are keyed by a hash of exactly this text).
'use strict';
(() => {
const SAY = [[/\*\*|`/g, ''], [/γᵗ/g, 'gamma to the t'], [/γ/g, 'gamma'], [/ε/g, 'epsilon'], [/λ/g, 'lambda'], [/α/g, 'alpha'], [/δ/g, 'delta'], [/θ/g, 'theta'],
  [/π\*/g, 'pi star'], [/π/g, 'pi'], [/Q\(s,\s*a\)/g, 'Q of s, a'], [/V\(s′\)/g, 'V of s prime'], [/V\(s\)/g, 'V of s'], [/s′/g, 's prime'], [/a′/g, 'a prime'],
  [/←/g, ' becomes '], [/→/g, ' to '], [/↔/g, ' versus '], [/≈/g, ' about '], [/×/g, ' times '], [/−/g, ' minus '], [/≥/g, ' at least '], [/∂/g, 'd '],
  [/\be\.g\./g, 'for example'], [/\bvs\b/g, 'versus'], [/\bRL\b/g, 'R L'], [/\bDQN\b/g, 'D Q N'], [/\bMDP\b/g, 'M D P'], [/\bMCTS\b/g, 'M C T S'], [/\bPPO\b/g, 'P P O'],
  [/\bRLHF\b/g, 'R L H F'], [/\bSAC\b/g, 'sack'], [/\bTD\b/g, 'T D'], [/\bUCB\b/g, 'U C B'], [/\bGRPO\b/g, 'G R P O'], [/\bTRPO\b/g, 'T R P O'], [/\bA2C\b/g, 'A 2 C'], [/\bA3C\b/g, 'A 3 C'], [/\bHER\b/g, 'H E R'], [/\bRND\b/g, 'R N D'], [/\bC51\b/g, 'C 51'], [/\bLinUCB\b/g, 'Lin U C B'], [/\bMAML\b/g, 'mammal'], [/²/g, ' squared'], [/÷/g, ' divided by '], [/√/g, 'the square root of '], [/\bln\b/g, 'log'], [/DeepSeek-R1/g, 'DeepSeek R 1'], [/\bPOMDPs?\b/g, 'P O M D P'], [/\bTD3\b/g, 'T D 3'], [/\bDDPG\b/g, 'D D P G'], [/\bGAE\b/g, 'G A E'], [/\bKL\b/g, 'K L'], [/\bCQL\b/g, 'C Q L'], [/\bIQL\b/g, 'I Q L'], [/\bDPO\b/g, 'D P O'], [/\bGAIL\b/g, 'gale'], [/\bDAgger\b/g, 'dagger'], [/\bMARL\b/g, 'mar L'], [/\bHRL\b/g, 'H R L'], [/\bIRL\b/g, 'I R L'], [/\bLLMs?\b/g, 'L L M'], [/\p{Extended_Pictographic}|️/gu, ''], [/[^\S\n]+/g, ' '], [/ ?\n ?/g, '\n']];
SAY.push([/ = /g, ' equals '], [/=/g, ' equals '], [/\+(?=\d)/g, 'plus '], [/ \+ /g, ' plus '], [/±/g, ' plus or minus '], [/β/g, 'beta'], [/μ/g, 'mu'], [/σ/g, 'sigma'], [/½/g, ' one half'],
  [/₀/g, ' zero'], [/₁/g, ' one'], [/₂/g, ' two'], [/₃/g, ' three'], [/ᵗ/g, ' to the t'], [/⁶/g, ' to the sixth'], [/ \| /g, ' given '], [/\|/g, ' given '], [/…/g, '...'], [/·/g, ' times '], [/&/g, ' and '], [/\*/g, ' star']);
// speak like a person: "gamma (gamma)" -> "gamma"; brackets become pauses; years read as "in 2018"
const DIG = ['oh', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
const spellDec = d => d.split('').map(x => DIG[+x]).join(' ');
const SAY2 = [[/\b(\d+)\.(\d+)\b/g, (m, a, b) => (a === '0' ? '' : a + ' ') + 'point ' + spellDec(b)], [/\bZero point\b/g, 'Point'], [/\bzero point\b/g, 'point'],
  [/\b(point(?: (?:zero|oh|one|two|three|four|five|six|seven|eight|nine))+)/gi, m => m.replace(/\bzero\b/gi, 'oh')],[/\b(\w+)\s*\(\s*\1\s*\)/gi, '$1'], [/\s*\((\d{4})\)/g, ', in $1,'], [/\s*\(/g, ', '], [/\)/g, ','], [/;/g, '.'], [/\s*,\s*([.,!?:])/g, '$1'], [/,\s*,/g, ','], [/ {2,}/g, ' ']];
const clean = s => SAY.reduce((a, [r, w]) => a.replace(r, w), s || '').trim();
function capFor(s) { // caption under an animation: slide override, else the animation's default (per mode first)
  if (!s || !s.a) return ''; if (s.cap) return s.cap;
  const C = (window.RLAnim && window.RLAnim.CAP) || {}; return C[s.a + ':' + ((s.p && s.p.mode) || '')] || C[s.a] || '';
}
const isUnit = x => Array.isArray(x) && x.length === 2 && typeof x[0] === 'string' && Array.isArray(x[1]);
function sayUnits(step) { // spoken script -> [{text, focus, para}]; focus = [x, y, w, h] box or [x, y, r] circle on the 400x420 canvas
  const v = typeof step.say === 'function' ? step.say() : step.say, out = [];
  [].concat(isUnit(v) ? [v] : v).forEach((para, pi) => (Array.isArray(para) && !isUnit(para) ? para : [para]).forEach(u => {
    const [text, focus] = typeof u === 'string' ? [u, null] : u;
    out.push({ text: SAY2.reduce((a, [r, w]) => a.replace(r, w), text).trim(), focus, para: pi });
  }));
  return out;
}
function narrationText(t, i) {
  if (i === 0) return `${t.title}. ${t.hook}`;
  if (i === t.slides.length + 1) return t._prep ? 'Reel complete.' : 'In a nutshell. ' + t.recap.join('. ') + '.';
  const s = t.slides[i - 1], cap = capFor(s);
  if (s.step && s.step.say) return sayUnits(s.step).map((u, k, A) => (k && A[k - 1].para !== u.para ? '\n' : k ? '\uE000' : '') + u.text).join(''); return [s.h && s.h + '.', s.t, cap && 'In the animation: ' + cap, s.try && 'Try it: ' + s.try, s.g && s.g.map(([w, d]) => `${w}: ${d.replace(/\.$/, '')}.`).join(' '), s.ex && 'For example: ' + s.ex].filter(Boolean).join(' ');
}
window.RLSpeech = { clean, narrationText, capFor, sayUnits, clip: (t, i) => clean(narrationText(t, i)) };
})();
