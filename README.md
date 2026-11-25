# RL Reels

Instagram-reels-style micro lessons on RL topics. Static PWA, no build step, no dependencies.

- `app/topics.js` — 48 topics (cover + ≤8 slides + recap, ≤10 slides each), with prerequisites (`pre`).
- `app/anim.js` — code-generated looping canvas animations (the "gifs"), one function per animation.
- `app/app.js` — feed, gestures, memory (localStorage key `rlreels.v1`) and the scheduler.
- `app/test.html` — renders every animation and lists errors; `app/flow.html?m=done|skim|partial|lib|slide` — scripted UI checks.

Run locally: `cd app && python3 -m http.server 8765` → http://localhost:8765

Controls: ▶ button toggles autoplay (off by default; hold or `p` pauses it); 🔊 button cycles narration+effects / effects only / mute; tap right / swipe left = next slide, tap left = back; swipe up/down (or wheel, ↑/↓) = next/previous reel; `l` = topic map.
Game layer: XP (+20 per new reel, +5 per unlock, +8 per refresher), daily streak, confetti, likes.

Memory heuristic: reaching the recap slide with ≥1.5 s per content slide = watched (unlocks dependants, re-shown for review after 3/8/21/60 days).
Reaching the end faster = "skimmed"; leaving midway = "partial". Any reel you leave early (even on its cover) cools down for ≥8 reels AND ≥30 min;
when nothing else is unlocked, a not-yet-unlocked topic (EARLY) is served before any skipped one repeats. "↻ Again" forces a repeat.
After editing `app/`, bump `CACHE` in `app/sw.js` (the service worker is network-first, so updates show on the next load).
