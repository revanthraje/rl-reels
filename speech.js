// Turns slide content into the sentence that gets narrated. Shared by app.js and tts/export.html
// (the pre-recorded clips are keyed by a hash of exactly this text).
'use strict';
(() => {
const SAY = [[/\*\*|`/g, ''], [/γᵗ/g, 'gamma to the t'], [/γ/g, 'gamma'], [/ε/g, 'epsilon'], [/λ/g, 'lambda'], [/α/g, 'alpha'], [/δ/g, 'delta'], [/θ/g, 'theta'],
  [/π\*/g, 'pi star'], [/π/g, 'pi'], [/Q\(s,\s*a\)/g, 'Q of s, a'], [/V\(s′\)/g, 'V of s prime'], [/V\(s\)/g, 'V of s'], [/s′/g, 's prime'], [/a′/g, 'a prime'],
  [/←/g, ' becomes '], [/→/g, ' to '], [/↔/g, ' versus '], [/≈/g, ' about '], [/×/g, ' times '], [/−/g, ' minus '], [/≥/g, ' at least '], [/∂/g, 'd '],
  [/\be\.g\./g, 'for example'], [/\bvs\b/g, 'versus'], [/\bRL\b/g, 'R L'], [/\bDQN\b/g, 'D Q N'], [/\bMDP\b/g, 'M D P'], [/\bMCTS\b/g, 'M C T S'], [/\bPPO\b/g, 'P P O'],
  [/\bRLHF\b/g, 'R L H F'], [/\bSAC\b/g, 'sack'], [/\bTD\b/g, 'T D'], [/\bUCB\b/g, 'U C B'], [/\bLLMs?\b/g, 'L L M'], [/\p{Extended_Pictographic}|️/gu, ''], [/\s+/g, ' ']];
const clean = s => SAY.reduce((a, [r, w]) => a.replace(r, w), s || '').trim();
function narrationText(t, i) {
  if (i === 0) return `${t.title}. ${t.hook}`;
  if (i === t.slides.length + 1) return 'In a nutshell. ' + t.recap.join('. ') + '.';
  const s = t.slides[i - 1]; return [s.h && s.h + '.', s.t, s.ex && 'For example: ' + s.ex].filter(Boolean).join(' ');
}
window.RLSpeech = { clean, narrationText, clip: (t, i) => clean(narrationText(t, i)) };
})();
