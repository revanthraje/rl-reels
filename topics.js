// Reel content. Each topic becomes one reel: auto cover slide + up to 8 content slides + auto recap slide (<= 10).
// Slide fields: h = heading, t = text (supports **bold** and `code`), a = animation name, p = animation params,
// ex = a concrete example box.  pre = prerequisite topic ids (must be "watched" before the reel is served).
'use strict';

window.RLTopics = [
// ───────────── Foundations ─────────────
{ id: 'loop', title: 'The RL Loop', emoji: '🔁', track: 'Foundations', pre: [],
  hook: 'Learning by doing: act, see what happens, adjust.',
  slides: [
    { h: 'Two players', t: 'An **agent** (the learner) and an **environment** (everything else). They take turns, forever or until the episode ends.', a: 'loop' },
    { h: 'One step', t: 'The agent sees a **state**, picks an **action**; the environment answers with a **new state** and a **reward** (a number).' },
    { h: 'No answer key', t: 'Unlike supervised learning, nobody says what the right action was. The agent only gets a score, often **late**.', ex: 'A chess bot learns it lost only at the end of 60 moves. Which move was the mistake?' },
    { h: 'Everyday examples', t: '🐕 Dog: sit → treat.  🎮 Game AI: press buttons → points.  🦾 Robot: move motors → closer to the cup.' },
    { h: 'The goal', t: 'Collect as much reward as possible **over time**, not just right now.' },
  ],
  recap: ['Agent acts, environment responds with state + reward', 'Learning from rewards, not labelled answers', 'Goal: maximise total reward over time'] },

{ id: 'returns', title: 'Rewards & Returns', emoji: '💰', track: 'Foundations', pre: ['loop'],
  hook: 'Why the agent cares about the sum, not the next snack.',
  slides: [
    { h: 'Reward = instant feedback', t: 'One number after each step. Positive is good, negative is bad, zero is "nothing happened".' },
    { h: 'Return = the total', t: 'The **return G** adds up all rewards from now until the end of the episode.', a: 'returns' },
    { h: 'Short-term vs long-term', t: 'An action with a small reward now can lead to a huge reward later. The agent should judge actions by the return.', ex: 'Studying tonight (−1 fun) → passing the exam next week (+10).' },
    { h: 'Episodes', t: 'Some tasks end (a game, a maze) — **episodic**. Others go on forever (a thermostat) — **continuing**.' },
    { h: 'The reward hypothesis', t: 'Big idea of RL: any goal can be expressed as "maximise the expected sum of a reward signal". Debatable, but useful!' },
  ],
  recap: ['Reward: per-step number', 'Return: sum of future rewards', 'Judge actions by their long-run return'] },

{ id: 'discount', title: 'The Discount Factor γ', emoji: '⏳', track: 'Foundations', pre: ['returns'],
  hook: 'A reward tomorrow is worth a bit less than a reward today.',
  slides: [
    { h: 'Shrink the future', t: 'Multiply a reward t steps away by **γᵗ**, with γ (gamma) between 0 and 1.', a: 'discount' },
    { h: 'Why bother?', t: 'Keeps infinite sums finite, encodes impatience, and acts like uncertainty about whether the future happens.' },
    { h: 'γ near 0', t: 'Myopic agent: only the next reward matters.', ex: 'γ = 0.5 → a reward 10 steps away is worth 0.001 of a reward now.' },
    { h: 'γ near 1', t: 'Far-sighted agent: plans long ahead, but learning gets harder and noisier.' },
    { h: 'Rule of thumb', t: 'The agent effectively looks about **1 / (1 − γ)** steps ahead. γ = 0.99 → about 100 steps.' },
  ],
  recap: ['Future reward weighted by γᵗ', 'Small γ = short-sighted, γ→1 = far-sighted', 'Horizon ≈ 1/(1−γ)'] },

{ id: 'mdp', title: 'Markov Decision Process', emoji: '🗺️', track: 'Foundations', pre: ['loop'],
  hook: 'The board game that almost every RL problem is written as.',
  slides: [
    { h: 'The ingredients', t: '**States**, **actions**, **transition probabilities** (where you might land), **rewards**, and a discount γ.' },
    { h: 'A tiny MDP', t: 'A race car that can be Cool, Warm or Overheated. Driving fast pays +2 but risks overheating (−10).', a: 'mdp' },
    { h: 'Randomness is allowed', t: 'The same action in the same state can lead to different next states, with fixed probabilities.' },
    { h: 'The Markov property', t: 'The **current state tells you everything** you need to predict the future. History adds nothing.', ex: 'Chess: the board position is Markov. A single frame of Pong is not — you can’t tell which way the ball moves.' },
    { h: 'Why it matters', t: 'Because the state is a full summary, the agent can decide using only "where am I now?".' },
  ],
  recap: ['MDP = states, actions, transitions, rewards, γ', 'Transitions can be random', 'Markov: the present state is a sufficient summary'] },

{ id: 'policy', title: 'Policies', emoji: '🧭', track: 'Foundations', pre: ['mdp'],
  hook: 'The agent’s rulebook: in this situation, do that.',
  slides: [
    { h: 'What is a policy?', t: 'A **policy π** maps each state to an action (or to probabilities over actions). It *is* the agent’s behaviour.', a: 'gridPolicy' },
    { h: 'Deterministic', t: 'Always the same action in a given state.', ex: '"At a red light: stop."' },
    { h: 'Stochastic', t: 'A probability for each action. Useful for exploring, and for games where being predictable is bad.', ex: 'Rock-paper-scissors: the best policy is ⅓ each.' },
    { h: 'Learning = improving π', t: 'Almost every RL algorithm is a recipe for turning the current policy into a better one.' },
    { h: 'The optimal policy π*', t: 'A policy whose expected return is at least as good as any other policy, from every state.' },
  ],
  recap: ['Policy π: state → action (or action probabilities)', 'Deterministic vs stochastic', 'RL searches for the optimal π*'] },

{ id: 'values', title: 'Value Functions', emoji: '📈', track: 'Foundations', pre: ['policy', 'discount'],
  hook: 'How good is it to be here? How good is this move?',
  slides: [
    { h: 'State value V(s)', t: 'Expected return if you start in state s and follow your policy afterwards.', a: 'gridPolicy', p: { heat: true } },
    { h: 'Action value Q(s, a)', t: 'Expected return if you take action a in state s first, then follow the policy.' },
    { h: 'Why Q is handy', t: 'If you know Q, acting well is easy: **pick the action with the biggest Q**. No model of the world needed.' },
    { h: 'Example', t: 'Chess intuition is a value function.', ex: '"This position is winning for white" ≈ high V. "Taking the queen here is great" ≈ high Q.' },
    { h: 'Values depend on the policy', t: 'The same square is worth more to a good player than to a random one. V^π and Q^π are always "under policy π".' },
  ],
  recap: ['V(s): expected return from a state', 'Q(s, a): expected return of an action in a state', 'Greedy on Q → good behaviour'] },

{ id: 'bellman', title: 'The Bellman Equation', emoji: '🪜', track: 'Foundations', pre: ['values'],
  hook: 'Value of here = reward now + discounted value of next.',
  slides: [
    { h: 'A recursive idea', t: 'You don’t need to imagine the whole future: one step of reward plus the value of where you land is enough.', a: 'backup' },
    { h: 'In words', t: '**V(s) = expected [ r + γ · V(next state) ]**. For the best policy, take the max over actions.' },
    { h: 'Backup diagrams', t: 'Values flow **upwards** from future states to the current one. This "backing up" is the heart of RL.' },
    { h: 'Example', t: 'A square next to the goal is worth almost +1. The square next to *that* is worth γ × that, and so on.', ex: 'Like a rumour of treasure spreading outward from the chest.' },
    { h: 'Everywhere in RL', t: 'Value iteration, TD learning, Q-learning and DQN are all ways of making the Bellman equation true.' },
  ],
  recap: ['Value = reward + γ × next value', 'Values are "backed up" from the future', 'Foundation of most RL algorithms'] },

// ───────────── Bandits & exploration ─────────────
{ id: 'explore', title: 'Exploration vs Exploitation', emoji: '🧪', track: 'Bandits', pre: [],
  hook: 'Order your favourite dish, or try something new?',
  slides: [
    { h: 'The dilemma', t: '**Exploit**: use what you know works. **Explore**: try other things that might be even better.' },
    { h: 'Pure greedy gets stuck', t: 'One lucky early win and a greedy agent may stick with a mediocre option forever.', a: 'bandit', p: { mode: 'greedy' } },
    { h: 'Everyday examples', t: '🍜 Restaurants.  🎵 Spotify suggesting new songs.  💊 Clinical trials: give the drug that looks best, or keep testing?' },
    { h: 'Too much exploring', t: 'Endless experimenting wastes reward. The trick is exploring **less as you become more sure**.' },
    { h: 'Strategies ahead', t: 'ε-greedy, optimism (UCB), Thompson sampling, and curiosity bonuses — each its own reel.' },
  ],
  recap: ['Exploit known good vs explore unknown', 'Greedy can lock in a bad choice', 'Explore less as certainty grows'] },

{ id: 'bandits', title: 'Multi-Armed Bandits', emoji: '🎰', track: 'Bandits', pre: ['explore'],
  hook: 'RL with one state and lots of slot machines.',
  slides: [
    { h: 'The setup', t: 'Several slot machines ("arms"), each with an unknown average payout. Every turn you pull one. Maximise your winnings.' },
    { h: 'Just one state', t: 'Actions don’t change the situation — no states to plan over. Pure exploration vs exploitation.', a: 'bandit', p: { mode: 'eps' } },
    { h: 'Regret', t: 'How much less you won than if you had always pulled the best arm. Good algorithms keep regret growing slowly (like log t).' },
    { h: 'Real uses', t: 'A/B testing websites, choosing ads, news recommendations, picking which hyper-parameters to try next.', ex: 'Show headline A or B? Each click is a reward.' },
    { h: 'Contextual bandits', t: 'Add a **context** (who the user is) to choose better. A stepping stone between bandits and full RL.' },
  ],
  recap: ['k arms, unknown payouts, pick one each turn', 'Measure performance by regret', 'Used in A/B tests and recommendations'] },

{ id: 'epsgreedy', title: 'ε-Greedy', emoji: '🎲', track: 'Bandits', pre: ['bandits'],
  hook: 'Mostly do the best thing, occasionally roll the dice.',
  slides: [
    { h: 'The rule', t: 'With probability **1 − ε** pick the best-looking action; with probability **ε** pick a random one.', a: 'bandit', p: { mode: 'eps' } },
    { h: 'Why it works', t: 'Every action keeps getting tried now and then, so wrong estimates eventually get corrected.' },
    { h: 'Decaying ε', t: 'Start with lots of exploration (ε = 1) and shrink it over time as estimates firm up.', ex: 'DQN on Atari: ε goes from 1.0 to 0.1 over the first million frames.' },
    { h: 'Weakness', t: 'Explores **blindly**: it wastes pulls on arms already known to be terrible, just as often as on promising ones.' },
  ],
  recap: ['Random action with prob ε, else greedy', 'Simple and surprisingly strong', 'Explores blindly; often decay ε'] },

{ id: 'ucb', title: 'Upper Confidence Bound', emoji: '🌟', track: 'Bandits', pre: ['bandits'],
  hook: 'Be optimistic about what you haven’t tried much.',
  slides: [
    { h: 'Optimism', t: 'Score each arm by **estimate + uncertainty bonus**. Pick the highest score.', a: 'bandit', p: { mode: 'ucb' } },
    { h: 'The bonus shrinks', t: 'The more often an arm is pulled, the smaller its bonus. Rarely-tried arms look tempting until checked.' },
    { h: 'Smart exploring', t: 'Unlike ε-greedy, it stops trying arms it is confident are bad.', ex: 'A new café gets a "benefit of the doubt" visit; after 3 bad coffees, you stop going.' },
    { h: 'Famous descendant', t: '**UCT** — UCB applied to tree search — is how MCTS decides which moves to explore (AlphaGo!).' },
  ],
  recap: ['Score = estimate + uncertainty bonus', 'Bonus shrinks with more pulls', 'Basis of tree search (UCT)'] },

{ id: 'thompson', title: 'Thompson Sampling', emoji: '🎯', track: 'Bandits', pre: ['bandits'],
  hook: 'Act on a random guess drawn from your beliefs.',
  slides: [
    { h: 'Beliefs, not numbers', t: 'Keep a probability distribution for each arm’s payout instead of a single estimate.', a: 'bandit', p: { mode: 'ts' } },
    { h: 'The rule', t: 'Each turn, draw one sample from every arm’s belief and pull the arm whose sample is highest.' },
    { h: 'Built-in balance', t: 'Uncertain arms produce wild samples, so they win sometimes (exploration). Clearly-good arms win most of the time.' },
    { h: 'Example', t: 'Online shops use it to pick which product photo to show.', ex: 'Photo B had 3 clicks out of 4 views: wide belief → still gets tried.' },
    { h: 'Also from 1933!', t: 'One of the oldest RL ideas, and still state of the art in many bandit problems.' },
  ],
  recap: ['Keep a belief distribution per arm', 'Sample from beliefs, act greedily on the sample', 'Natural exploration from uncertainty'] },

// ───────────── Planning ─────────────
{ id: 'valueiter', title: 'Value Iteration', emoji: '🌊', track: 'Planning', pre: ['bellman'],
  hook: 'If you know the rules of the world, spread value outward from the goal.',
  slides: [
    { h: 'Dynamic programming', t: 'When the MDP is fully known, you can **compute** the best policy instead of learning by trial.' },
    { h: 'The sweep', t: 'Repeatedly update every state: V ← max over actions of [reward + γ V(next)].', a: 'valueIter' },
    { h: 'Watch it spread', t: 'After sweep 1 only squares next to the goal know about it. Each sweep pushes the news one step further.' },
    { h: 'Read off the policy', t: 'Once values stop changing, act greedily: move toward the neighbour with the highest value.' },
    { h: 'Limits', t: 'Needs the full model and a sweep over **every** state — fine for a gridworld, impossible for Go (10¹⁷⁰ states).', ex: 'Used in robot path planning on small maps and inventory problems.' },
  ],
  recap: ['Needs a known model', 'Repeat Bellman max-backups over all states', 'Converged values → greedy optimal policy'] },

{ id: 'policyiter', title: 'Policy Iteration', emoji: '🔄', track: 'Planning', pre: ['bellman'],
  hook: 'Evaluate, improve, repeat — until nothing changes.',
  slides: [
    { h: 'Two alternating steps', t: '**Evaluate**: work out V for the current policy. **Improve**: switch every state to the greedy action under that V.', a: 'policyIter' },
    { h: 'Always gets better', t: 'Each improvement is guaranteed to be at least as good. With finitely many policies it must stop — at the optimum.' },
    { h: 'Few rounds', t: 'Usually converges in a handful of iterations, though each evaluation step is costly.' },
    { h: 'Generalised policy iteration', t: 'Nearly all RL is a loose version: estimate values a bit, improve the policy a bit, repeat.', ex: 'Actor-critic methods = policy iteration with learning instead of exact computation.' },
  ],
  recap: ['Alternate evaluation and greedy improvement', 'Monotonic improvement → optimal policy', 'Template ("GPI") for most RL algorithms'] },

// ───────────── Model-free ─────────────
{ id: 'mc', title: 'Monte Carlo Learning', emoji: '🎬', track: 'Model-free', pre: ['values'],
  hook: 'Play to the end, then learn from what really happened.',
  slides: [
    { h: 'No model needed', t: 'Learn values from **experience** only: play full episodes and record the return that followed each state.' },
    { h: 'Average the returns', t: 'V(s) ≈ the average return seen after visiting s. More episodes → better estimates.', a: 'mc' },
    { h: 'Unbiased but noisy', t: 'Uses real outcomes (no guessing), but one lucky or unlucky episode swings the estimate a lot.' },
    { h: 'Must wait', t: 'Can only learn when an episode **finishes** — useless for never-ending tasks.', ex: 'Estimating a blackjack hand’s value by playing thousands of hands and averaging wins.' },
  ],
  recap: ['Learn from complete episodes', 'Value = average observed return', 'Unbiased, high variance, needs episodes to end'] },

{ id: 'td', title: 'Temporal-Difference Learning', emoji: '⚡', track: 'Model-free', pre: ['values', 'bellman'],
  hook: 'Learn a guess from a slightly better guess — every step.',
  slides: [
    { h: 'Bootstrapping', t: 'After each step, nudge V(s) toward **r + γ V(s′)** — the reward plus your current guess for the next state.', a: 'td' },
    { h: 'The TD error δ', t: 'δ = r + γV(s′) − V(s): "was that better or worse than I expected?" Learning is driven by this surprise.' },
    { h: 'TD vs Monte Carlo', t: 'TD learns online and with less noise, but its targets are biased by its own guesses.', ex: 'Predicting your commute: MC waits until you arrive; TD updates when you hit traffic halfway.' },
    { h: 'Brains do it too', t: 'Dopamine neurons fire like a TD error: they react to **unexpected** reward, not to reward itself.' },
    { h: 'Next up', t: 'TD for actions gives SARSA and Q-learning.' },
  ],
  recap: ['Update from one step + next-state guess', 'δ = surprise signal', 'Online, lower variance, a bit biased'] },

{ id: 'qlearning', title: 'Q-Learning', emoji: '🧩', track: 'Model-free', pre: ['td', 'epsgreedy'],
  hook: 'Learn the value of every move, even while acting randomly.',
  slides: [
    { h: 'The update', t: 'Q(s,a) ← Q(s,a) + α [ r + γ **max** Q(s′, ·) − Q(s,a) ].', a: 'qgrid' },
    { h: 'Assume the best afterwards', t: 'The target uses the best next action, regardless of what you actually do next. So it learns the **optimal** Q.' },
    { h: 'Off-policy', t: 'You can explore with ε-greedy (or even random moves) and still learn the greedy, optimal policy.' },
    { h: 'A table of numbers', t: 'Classic Q-learning stores one number per (state, action). Great for small problems.', ex: 'Taxi-v3 (500 states × 6 actions) is solved in seconds.' },
    { h: 'The leap', t: 'Replace the table with a neural network → **DQN**, which learned to play Atari from pixels.' },
  ],
  recap: ['TD update toward r + γ max Q', 'Learns optimal Q while exploring', 'Table version → DQN with neural nets'] },

{ id: 'sarsa', title: 'SARSA', emoji: '🚶', track: 'Model-free', pre: ['td', 'epsgreedy'],
  hook: 'Learn the value of what you’ll actually do next.',
  slides: [
    { h: 'The name', t: 'Uses the tuple **S, A, R, S′, A′**: state, action, reward, next state, next action actually chosen.' },
    { h: 'The update', t: 'Q(s,a) ← Q(s,a) + α [ r + γ Q(s′, **a′**) − Q(s,a) ] — no max, the real next action.' },
    { h: 'On-policy', t: 'It evaluates the policy it is following, exploration included. So it learns to be careful where mistakes hurt.', a: 'cliff' },
    { h: 'Cliff walking', t: 'With random exploring steps, walking along the cliff edge is dangerous. SARSA learns a safer detour.', ex: 'Like a learner driver keeping extra distance because they know they sometimes swerve.' },
  ],
  recap: ['Target uses the actual next action a′', 'On-policy: values include exploration', 'Safer behaviour while still learning'] },

{ id: 'onoff', title: 'On-Policy vs Off-Policy', emoji: '🔀', track: 'Model-free', pre: ['qlearning', 'sarsa'],
  hook: 'Learn from your own behaviour, or from anyone’s?',
  slides: [
    { h: 'Two policies', t: '**Behaviour policy**: the one generating data. **Target policy**: the one being learned.' },
    { h: 'On-policy', t: 'They are the same. Learn about what you do. (SARSA, PPO)', a: 'cliff' },
    { h: 'Off-policy', t: 'They differ. Learn the best policy while acting differently — or from old data. (Q-learning, DQN, SAC)' },
    { h: 'Why off-policy is powerful', t: 'Reuse old experience, learn from humans or other agents, explore freely.', ex: 'Learning to drive by watching dashcam videos of other people.' },
    { h: 'Why it is harder', t: 'Data doesn’t match the policy being learned; corrections (like importance sampling) or tricks are needed for stability.' },
  ],
  recap: ['Behaviour vs target policy', 'On: same policy (SARSA, PPO)', 'Off: different — reuse data (Q-learning, DQN)'] },

{ id: 'traces', title: 'Eligibility Traces & TD(λ)', emoji: '✨', track: 'Model-free', pre: ['td', 'mc'],
  hook: 'Give credit to recent steps, fading into the past.',
  slides: [
    { h: 'The credit problem', t: 'A reward arrives. Which earlier steps deserve credit? TD(0) credits only the last one; MC credits all equally.' },
    { h: 'A fading memory', t: 'Each visited state gets a **trace** that decays by γλ every step. A reward updates states in proportion to their trace.', a: 'traces' },
    { h: 'λ dials between TD and MC', t: 'λ = 0 → one-step TD. λ = 1 → Monte Carlo. In between often learns fastest.' },
    { h: 'Example', t: 'TD-Gammon (1992) used TD(λ) to reach world-class backgammon by self-play.', ex: 'Like remembering the last few moves that led to a win more vividly than the opening.' },
  ],
  recap: ['Traces mark recently visited states', 'Reward spreads back along the trace', 'λ interpolates TD(0) ↔ Monte Carlo'] },

// ───────────── Deep RL ─────────────
{ id: 'funcapprox', title: 'Function Approximation', emoji: '🕸️', track: 'Deep RL', pre: ['values'],
  hook: 'Too many states for a table? Learn a function that generalises.',
  slides: [
    { h: 'Tables don’t scale', t: 'A table needs one entry per state. Images, robot joints, Go boards: far too many states, most never seen.' },
    { h: 'Generalise', t: 'Represent V or Q with a function (linear model, neural network) so learning about one state also informs similar ones.', a: 'funcApprox' },
    { h: 'Features', t: 'The function works on **features** of the state — hand-made (distances, angles) or learned (deep nets from pixels).' },
    { h: 'The catch', t: 'Combining function approximation + bootstrapping + off-policy data can diverge — the "deadly triad".', ex: 'A small update for one state changes the value of many others, which changes the targets…' },
  ],
  recap: ['Replace tables with parameterised functions', 'Generalises to unseen states', 'Beware the deadly triad'] },

{ id: 'dqn', title: 'Deep Q-Network (DQN)', emoji: '👾', track: 'Deep RL', pre: ['qlearning', 'funcapprox'],
  hook: 'Q-learning + a neural network = superhuman Atari from pixels.',
  slides: [
    { h: 'The idea', t: 'A convolutional network reads the game screen and outputs a Q-value for every action.', a: 'dqn' },
    { h: '2015 milestone', t: 'One algorithm, same settings, learned 49 Atari games from raw pixels and score — many beyond human level.' },
    { h: 'Why naive fails', t: 'Consecutive frames are highly correlated and the targets move as the net learns → unstable training.' },
    { h: 'Two key tricks', t: '**Experience replay** and a **target network** — each gets its own reel.' },
    { h: 'Frame stacking', t: 'It feeds the last 4 frames so the state shows motion — restoring the Markov property.', ex: 'One frame of Breakout can’t tell you whether the ball is rising or falling.' },
  ],
  recap: ['Neural net outputs Q for each action', 'Learned Atari from pixels (2015)', 'Needs replay + target network to be stable'] },

{ id: 'replay', title: 'Experience Replay', emoji: '📼', track: 'Deep RL', pre: ['dqn'],
  hook: 'Store experiences, then learn from random old memories.',
  slides: [
    { h: 'The buffer', t: 'Save every transition (s, a, r, s′) in a big memory, e.g. the last 1 million steps.', a: 'replay' },
    { h: 'Random minibatches', t: 'Train on random samples from the buffer, not on the latest steps in order.' },
    { h: 'Why it helps', t: 'Breaks correlation between consecutive samples, and reuses each experience many times → more data-efficient.' },
    { h: 'Prioritised replay', t: 'Replay surprising transitions (big TD error) more often.', ex: 'Like re-reading the exam questions you got wrong rather than the easy ones.' },
  ],
  recap: ['Store transitions in a buffer', 'Sample random minibatches', 'Decorrelates data and reuses it'] },

{ id: 'targetnet', title: 'Target Networks', emoji: '🎯', track: 'Deep RL', pre: ['dqn'],
  hook: 'Stop chasing your own tail: freeze the target for a while.',
  slides: [
    { h: 'Moving targets', t: 'The TD target r + γ max Q(s′) uses the same network you’re training. Every update shifts the target too.' },
    { h: 'The fix', t: 'Keep a second, **frozen copy** of the network to compute targets. Copy the weights over every N steps.', a: 'target' },
    { h: 'Soft updates', t: 'Alternatively blend slowly: θ_target ← 0.995 θ_target + 0.005 θ (used in DDPG, SAC).' },
    { h: 'Intuition', t: 'It turns a chase into a series of short, fixed regression problems.', ex: 'Hitting a target that stays put for a minute is easier than one that jumps every time you aim.' },
  ],
  recap: ['Targets computed with a lagged copy', 'Hard copies every N steps or soft blending', 'Much more stable training'] },

{ id: 'doubledqn', title: 'Double DQN', emoji: '👯', track: 'Deep RL', pre: ['dqn'],
  hook: 'Taking a max over noisy guesses is too optimistic.',
  slides: [
    { h: 'Overestimation', t: 'Even if all Q-estimates are unbiased, the **max** of noisy estimates is biased upward.', a: 'overest' },
    { h: 'The fix', t: 'Use one network to **choose** the best next action and another to **evaluate** it. Noise no longer lines up.' },
    { h: 'Cheap to add', t: 'DQN already has two networks (online and target), so Double DQN is a one-line change.' },
    { h: 'Same idea elsewhere', t: 'TD3 and SAC use **two critics and take the min** to fight the same bias in continuous control.', ex: 'Asking two friends to rate a restaurant and trusting the lower score.' },
  ],
  recap: ['max over noise → overestimation', 'Select with one net, evaluate with the other', 'Twin critics in TD3/SAC follow the same idea'] },

// ───────────── Policy gradients ─────────────
{ id: 'pg', title: 'Policy Gradient Methods', emoji: '📐', track: 'Policy Gradients', pre: ['policy'],
  hook: 'Skip the values: directly tweak the policy to win more.',
  slides: [
    { h: 'Learn π directly', t: 'Make the policy a parameterised function (a neural net that outputs action probabilities) and do gradient ascent on expected return.' },
    { h: 'The intuition', t: 'Actions followed by good outcomes become **more likely**; bad ones less likely.', a: 'pg' },
    { h: 'Why not just use Q?', t: 'Policy methods handle **continuous** actions naturally and can learn stochastic policies.', ex: 'A robot arm choosing a torque of 0.37 N·m — there’s no "max over all torques" table.' },
    { h: 'The price', t: 'Gradient estimates are noisy and need many samples. Much of the field is about reducing that noise.' },
  ],
  recap: ['Parameterise π and do gradient ascent', 'Good outcomes → raise those action probabilities', 'Great for continuous actions; noisy'] },

{ id: 'reinforce', title: 'REINFORCE', emoji: '🏋️', track: 'Policy Gradients', pre: ['pg', 'mc'],
  hook: 'The simplest policy gradient: weight actions by the episode’s return.',
  slides: [
    { h: 'The recipe', t: 'Play an episode. For every step, push up log π(a|s) in proportion to the **return** that followed.', a: 'reinforce' },
    { h: 'Monte Carlo flavour', t: 'Uses real returns: unbiased, but very noisy because one episode’s luck colours every action in it.' },
    { h: 'Example', t: 'A winning game makes every move in it more likely — even the blunders that happened not to matter.', ex: 'Williams, 1992. Still used for training text generators with a reward.' },
    { h: 'Fixing the noise', t: 'Subtract a **baseline** → advantage. Replace returns with a learned critic → actor-critic.' },
  ],
  recap: ['∇ ≈ Σ return × ∇ log π(a|s)', 'Simple and unbiased', 'High variance → baselines & critics'] },

{ id: 'baseline', title: 'Baselines & Advantage', emoji: '⚖️', track: 'Policy Gradients', pre: ['reinforce', 'values'],
  hook: 'Not "was it good?" but "was it better than usual?"',
  slides: [
    { h: 'The problem', t: 'If every return is around +100, REINFORCE pushes **every** action up. Signal hides in noise.' },
    { h: 'Subtract a baseline', t: 'Use (G − b) instead of G. A common choice: b = V(s), the expected return from that state.', a: 'baseline' },
    { h: 'Advantage', t: '**A(s,a) = Q(s,a) − V(s)**: how much better this action was than the policy’s average from here.' },
    { h: 'Still unbiased', t: 'A baseline that doesn’t depend on the action doesn’t change the expected gradient — only cuts its variance.', ex: 'Grading on a curve: 75% is great if the class average is 60%, poor if it’s 90%.' },
  ],
  recap: ['Subtract a baseline from the return', 'Advantage = better/worse than average', 'Same expectation, much lower variance'] },

{ id: 'actorcritic', title: 'Actor-Critic', emoji: '🎭', track: 'Policy Gradients', pre: ['baseline', 'td'],
  hook: 'One network acts, another grades every move.',
  slides: [
    { h: 'Two roles', t: '**Actor** = the policy. **Critic** = a value function that estimates how good things are.', a: 'actorCritic' },
    { h: 'Learning every step', t: 'The critic’s TD error δ tells the actor whether the last action was better or worse than expected.' },
    { h: 'Best of both', t: 'Policy-gradient flexibility + TD’s lower variance and online learning.' },
    { h: 'Big family', t: 'A2C/A3C, PPO, DDPG, TD3, SAC are all actor-critics.', ex: 'An athlete (actor) and a coach (critic) shouting "better!" or "worse!" after each move.' },
  ],
  recap: ['Actor = policy, critic = value estimate', 'TD error drives the actor', 'Most modern deep RL is actor-critic'] },

{ id: 'gae', title: 'n-step Returns & GAE', emoji: '🎚️', track: 'Policy Gradients', pre: ['actorcritic', 'traces'],
  hook: 'How far should you look before trusting the critic’s guess?',
  slides: [
    { h: 'n-step returns', t: 'Add up n real rewards, then plug in the critic’s value estimate for the rest.' },
    { h: 'Bias vs variance', t: 'Small n: low noise, but relies on a possibly-wrong critic. Large n: accurate on average, noisy.' },
    { h: 'GAE: blend them all', t: '**Generalised Advantage Estimation** mixes every n with weights (1−λ)λⁿ⁻¹. One knob λ picks the trade-off.', a: 'nstep' },
    { h: 'In practice', t: 'λ ≈ 0.95 is the default in PPO almost everywhere.', ex: 'Like asking for advice: trust a forecast for the far future, but your own eyes for the next few steps.' },
  ],
  recap: ['n-step = n rewards + bootstrap', 'GAE = exponentially weighted mix (λ)', 'Default λ ≈ 0.95 in PPO'] },

{ id: 'ppo', title: 'PPO', emoji: '🛡️', track: 'Policy Gradients', pre: ['actorcritic'],
  hook: 'Improve the policy, but not too much at once.',
  slides: [
    { h: 'The danger', t: 'One overly large policy update can wreck performance, and then the bad policy collects bad data.' },
    { h: 'Clip the ratio', t: 'Compare new vs old probability of each action (ratio r). Once r leaves [1−ε, 1+ε], stop pushing.', a: 'ppoClip' },
    { h: 'Reuse data safely', t: 'Clipping allows several epochs of gradient steps on the same batch without going off the rails.' },
    { h: 'Everywhere', t: 'Default choice for games, robotics in simulation, and the RL step of ChatGPT-style RLHF.', ex: 'OpenAI Five (Dota 2) was trained with PPO.' },
    { h: 'Ancestor: TRPO', t: 'Trust Region Policy Optimization enforced the same "small step" with heavier maths; PPO gets most of the benefit simply.' },
  ],
  recap: ['Limit how far each update moves the policy', 'Clipped probability ratio', 'Simple, robust, hugely popular'] },

{ id: 'sac', title: 'Entropy & Soft Actor-Critic', emoji: '🌈', track: 'Policy Gradients', pre: ['actorcritic', 'explore'],
  hook: 'Reward the agent for keeping its options open.',
  slides: [
    { h: 'Entropy', t: 'Measures how spread out a policy’s action choices are. High entropy = varied, low = always the same.', a: 'entropy' },
    { h: 'Maximum-entropy RL', t: 'Maximise **reward + α × entropy**: succeed, but in as many different ways as possible.' },
    { h: 'Why that helps', t: 'Built-in exploration, robustness to disturbances, and no premature collapse onto one mediocre strategy.' },
    { h: 'SAC', t: 'Soft Actor-Critic: off-policy, twin critics, automatic tuning of α. A go-to for continuous control.', ex: 'Real robots learning to walk in a couple of hours of experience.' },
  ],
  recap: ['Entropy = randomness of the policy', 'Objective: reward + α·entropy', 'SAC: stable off-policy continuous control'] },

{ id: 'ddpg', title: 'Continuous Control: DDPG & TD3', emoji: '🦿', track: 'Policy Gradients', pre: ['actorcritic', 'dqn'],
  hook: 'DQN for steering wheels and robot joints.',
  slides: [
    { h: 'The problem', t: 'With infinitely many actions you can’t compute max over Q(s, a) by checking each one.' },
    { h: 'Learn the argmax', t: 'An **actor** outputs one action; it is trained to climb the critic’s Q(s, a) surface.', a: 'continuous' },
    { h: 'DQN tricks reused', t: 'Replay buffer, target networks, and noise added to actions for exploration.' },
    { h: 'TD3 fixes', t: 'Twin critics (take the min), delayed actor updates and target smoothing — much more reliable than DDPG.', ex: 'Benchmarks: MuJoCo HalfCheetah, Hopper, robotic arms.' },
  ],
  recap: ['Deterministic actor climbs ∂Q/∂a', 'Off-policy with replay & targets', 'TD3 = DDPG + overestimation fixes'] },

// ───────────── Model-based ─────────────
{ id: 'modelbased', title: 'Model-Based RL', emoji: '🔮', track: 'Model-based', pre: ['mdp', 'values'],
  hook: 'Learn how the world works, then plan in your head.',
  slides: [
    { h: 'A model', t: 'Something that predicts **next state and reward** given state and action. Given (like chess rules) or learned from data.' },
    { h: 'Plan before acting', t: 'Use the model to imagine outcomes of different actions and pick the best.', a: 'modelBased' },
    { h: 'Sample efficiency', t: 'Imagined experience is free; real experience (robots, patients) is costly. Models can cut real data needs a lot.' },
    { h: 'Model errors compound', t: 'Small prediction mistakes snowball over long imagined rollouts, and planners love to exploit them.', ex: 'Dyna (1990): learn a model and mix real + imagined updates.' },
  ],
  recap: ['Model predicts next state & reward', 'Plan / imagine before acting', 'Sample-efficient but model errors compound'] },

{ id: 'mcts', title: 'Monte Carlo Tree Search', emoji: '🌳', track: 'Model-based', pre: ['modelbased', 'ucb'],
  hook: 'Grow a search tree toward the most promising moves.',
  slides: [
    { h: 'Four steps', t: '**Select** a path with UCB, **expand** a new node, **simulate** a playout, **back up** the result.', a: 'mcts' },
    { h: 'Asymmetric', t: 'The tree grows deep where moves look good and stays shallow elsewhere — no need to search everything.' },
    { h: 'Anytime', t: 'Stop whenever time runs out; play the most-visited move at the root.' },
    { h: 'Where it shines', t: 'Games with huge branching like Go, where classic minimax search drowns.', ex: 'Board-game AIs, puzzle solvers, and planning in robotics.' },
  ],
  recap: ['Select → expand → simulate → backup', 'Focuses search on promising branches', 'Core of AlphaGo-style systems'] },

{ id: 'alphazero', title: 'AlphaZero', emoji: '♟️', track: 'Model-based', pre: ['mcts', 'actorcritic'],
  hook: 'Self-play + tree search + one neural network.',
  slides: [
    { h: 'The loop', t: 'Play games against itself using MCTS, train a network on those games, use the better network for the next games.', a: 'selfplay' },
    { h: 'One net, two heads', t: '**Policy head** suggests moves to search; **value head** judges positions — no random playouts needed.' },
    { h: 'MCTS as a teacher', t: 'Search finds better moves than the raw network, and the network learns to imitate the search.' },
    { h: 'Results', t: 'From zero knowledge to superhuman at chess, shogi and Go in days.', ex: 'MuZero (2019) went further: it learned the game rules (the model) too.' },
  ],
  recap: ['Self-play generates its own data', 'Network guides search; search trains network', 'Superhuman Go/chess from scratch'] },

{ id: 'worldmodels', title: 'World Models & Dreamer', emoji: '💭', track: 'Model-based', pre: ['modelbased', 'funcapprox'],
  hook: 'Learn to dream the game, then practise in the dream.',
  slides: [
    { h: 'Latent world model', t: 'Compress observations into a small **latent state** and learn how that latent evolves with actions.', a: 'worldModel' },
    { h: 'Learning in imagination', t: 'Train the policy entirely on imagined rollouts in latent space — thousands of them, very cheaply.' },
    { h: 'Dreamer', t: 'DreamerV3 used one set of settings across many domains and was first to collect diamonds in Minecraft from scratch.' },
    { h: 'Why latent?', t: 'Predicting every pixel is wasteful; predicting what matters for reward is enough.', ex: 'You can imagine catching a ball without picturing every blade of grass.' },
  ],
  recap: ['Learn a compact latent dynamics model', 'Train the policy in imagination', 'Dreamer, World Models, MuZero'] },

// ───────────── Rewards & exploration (advanced) ─────────────
{ id: 'shaping', title: 'Reward Shaping', emoji: '🧲', track: 'Rewards', pre: ['returns', 'explore'],
  hook: 'Add hints so the agent isn’t searching in the dark.',
  slides: [
    { h: 'Sparse rewards', t: 'If reward only comes at the very end, random exploration may never find it.' },
    { h: 'Shaping', t: 'Add intermediate rewards that point in the right direction.', a: 'shaping' },
    { h: 'Danger', t: 'Badly designed hints change what the agent optimises. It may farm the hints and ignore the real goal.' },
    { h: 'The safe recipe', t: '**Potential-based shaping**: reward = γΦ(s′) − Φ(s). Proven not to change which policy is optimal.', ex: 'Φ = −distance to goal: moving closer earns a little, moving away costs a little.' },
  ],
  recap: ['Sparse rewards make learning slow', 'Shaping adds guiding rewards', 'Potential-based shaping keeps the optimum'] },

{ id: 'hacking', title: 'Reward Hacking', emoji: '🐛', track: 'Rewards', pre: ['shaping'],
  hook: 'You get what you reward — not what you meant.',
  slides: [
    { h: 'Specification gaming', t: 'The agent finds a loophole that scores high reward while ignoring the designer’s intent.', a: 'hacking' },
    { h: 'The boat race', t: 'A boat-racing agent learned to spin in circles hitting respawning targets instead of finishing the race.' },
    { h: 'More examples', t: 'Simulated robots exploiting physics bugs; a Tetris agent pausing forever to avoid losing.', ex: 'Goodhart’s law: "When a measure becomes a target, it ceases to be a good measure."' },
    { h: 'Why it matters', t: 'A big topic in AI safety, especially for RLHF language models that may learn to please the reward model rather than be right.' },
  ],
  recap: ['Agents exploit loopholes in the reward', 'Classic: boat circling for points', 'Central concern for AI safety'] },

{ id: 'curiosity', title: 'Curiosity & Intrinsic Motivation', emoji: '🔭', track: 'Rewards', pre: ['explore'],
  hook: 'Reward the agent for being surprised.',
  slides: [
    { h: 'Intrinsic reward', t: 'A bonus the agent gives itself for visiting **new or surprising** states, on top of the task reward.', a: 'curiosity' },
    { h: 'How to measure novelty', t: 'Visit counts, or the error of a model trying to predict the next state: high error = unfamiliar.' },
    { h: 'Wins', t: 'Montezuma’s Revenge, a famously sparse Atari game, was cracked by curiosity methods like RND.' },
    { h: 'The noisy-TV problem', t: 'A screen of random static is endlessly "surprising" — a naive curious agent stares at it forever.', ex: 'Fix: be curious only about things you can learn to predict.' },
  ],
  recap: ['Bonus reward for novelty / surprise', 'Counts or prediction error', 'Helps sparse tasks; beware the noisy TV'] },

// ───────────── Learning from data & people ─────────────
{ id: 'imitation', title: 'Imitation Learning', emoji: '🪞', track: 'Learning from people', pre: ['policy'],
  hook: 'Skip the reward: copy an expert.',
  slides: [
    { h: 'Behaviour cloning', t: 'Treat expert demonstrations as labelled data: state → expert’s action. Plain supervised learning.' },
    { h: 'Compounding errors', t: 'Small mistakes lead to states the expert never visited, where the learner has no idea what to do.', a: 'imitation' },
    { h: 'DAgger', t: 'Let the learner drive, ask the expert what they *would* have done in those states, and add that to the data.' },
    { h: 'Used everywhere', t: 'Self-driving prototypes, robot manipulation from teleoperation, pre-training of LLMs (imitating text).', ex: 'NVIDIA’s 2016 car learned steering from 72 hours of human driving.' },
  ],
  recap: ['Learn from expert demonstrations', 'Errors compound in unseen states', 'DAgger collects expert labels on learner states'] },

{ id: 'irl', title: 'Inverse RL', emoji: '🕵️', track: 'Learning from people', pre: ['imitation', 'values'],
  hook: 'Watch the expert, infer what they’re trying to achieve.',
  slides: [
    { h: 'Flip the problem', t: 'Normal RL: reward → behaviour. **Inverse RL**: behaviour → the reward that explains it.', a: 'irl' },
    { h: 'Why recover a reward?', t: 'A reward is a compact, transferable description of the goal; it can be re-optimised in new situations.' },
    { h: 'Ambiguity', t: 'Many rewards explain the same behaviour (even "reward = 0" does). Methods like maximum-entropy IRL pick a sensible one.' },
    { h: 'Example', t: 'Learning drivers’ preferences (speed vs comfort vs safety) from their trajectories.', ex: 'GAIL: a GAN-style method that imitates without writing the reward down.' },
  ],
  recap: ['Infer reward from demonstrations', 'Rewards transfer to new situations', 'Ill-posed: many rewards fit'] },

{ id: 'offline', title: 'Offline RL', emoji: '💾', track: 'Learning from people', pre: ['onoff', 'funcapprox'],
  hook: 'Learn a policy from a fixed log of data — no new trials.',
  slides: [
    { h: 'The setting', t: 'You only have a dataset collected by someone else. No exploring allowed.', ex: 'Hospital records, past robot runs, logged recommendations.' },
    { h: 'Why it is hard', t: 'The learner can believe an action it has **never seen** is amazing — and nothing in the data corrects it.', a: 'offline' },
    { h: 'Stay close to the data', t: 'Methods like CQL, IQL, and BCQ penalise or avoid actions outside the dataset’s support.' },
    { h: 'Why it matters', t: 'Lets RL use the huge logs that already exist where online trial-and-error is unsafe or expensive.' },
  ],
  recap: ['Learn from a fixed dataset only', 'Danger: overvaluing unseen actions', 'Conservative methods stay near the data'] },

{ id: 'rlhf', title: 'RLHF', emoji: '💬', track: 'Learning from people', pre: ['ppo', 'imitation'],
  hook: 'How chatbots are tuned with human preferences.',
  slides: [
    { h: 'Hard-to-write rewards', t: 'What is a "helpful, honest answer"? Impossible to code, easy for people to compare.' },
    { h: 'The pipeline', t: '1) Humans compare pairs of answers. 2) Train a **reward model** to predict their choice. 3) RL (often PPO) to maximise it.', a: 'rlhf' },
    { h: 'Stay close', t: 'A penalty (KL) keeps the model near its starting version so it doesn’t exploit the reward model.' },
    { h: 'Variants', t: 'DPO skips the explicit RL step; RLAIF uses AI feedback; constitutional methods use written principles.', ex: 'InstructGPT (2022) showed a small RLHF model was preferred over a 100× bigger plain one.' },
  ],
  recap: ['Preferences → reward model → RL', 'KL penalty limits reward hacking', 'Core of modern chatbot alignment'] },

// ───────────── Bigger worlds ─────────────
{ id: 'pomdp', title: 'Partial Observability (POMDPs)', emoji: '🌫️', track: 'Bigger worlds', pre: ['mdp'],
  hook: 'When you can’t see the whole state.',
  slides: [
    { h: 'Observations ≠ states', t: 'The agent gets **clues** (sensor readings, camera images) that don’t reveal everything.' },
    { h: 'Belief states', t: 'Keep a probability distribution over where you might be, updated by each action and observation.', a: 'pomdp' },
    { h: 'Memory helps', t: 'In deep RL, recurrent networks (LSTMs) or transformers give the agent memory of past observations.', ex: 'Poker: you can’t see opponents’ cards, so you reason about what they might hold.' },
    { h: 'Act to learn', t: 'Sometimes the best action is one that gathers information.', ex: 'Opening a door just to see what’s behind it.' },
  ],
  recap: ['Agent sees observations, not the full state', 'Belief = distribution over states', 'Memory / recurrent policies'] },

{ id: 'hrl', title: 'Hierarchical RL', emoji: '🏢', track: 'Bigger worlds', pre: ['policy', 'values'],
  hook: 'Think in sub-goals, not in individual muscle twitches.',
  slides: [
    { h: 'Temporal abstraction', t: 'Long tasks are easier as a few big decisions ("go to the kitchen") than thousands of tiny ones.' },
    { h: 'Manager & worker', t: 'A high-level policy picks sub-goals; a low-level policy figures out the steps.', a: 'hrl' },
    { h: 'Options', t: 'An **option** is a mini-policy with a start and stop condition — a reusable skill.', ex: 'Make coffee = [walk to kitchen] → [grind beans] → [brew].' },
    { h: 'Open problem', t: 'Discovering good sub-goals automatically, rather than hand-designing them, is still hard.' },
  ],
  recap: ['Decompose long tasks into sub-goals', 'Manager/worker or options', 'Skill discovery remains open'] },

{ id: 'marl', title: 'Multi-Agent RL', emoji: '👥', track: 'Bigger worlds', pre: ['mdp', 'policy'],
  hook: 'When several learners share one world.',
  slides: [
    { h: 'Many agents', t: 'Cooperative (teammates), competitive (opponents), or a mix (traffic, markets).', a: 'marl' },
    { h: 'Non-stationarity', t: 'Other agents keep learning, so the environment you face keeps changing — old lessons go stale.' },
    { h: 'Centralised training, decentralised execution', t: 'Train with access to everyone’s info, but each agent acts on its own observations at test time.' },
    { h: 'Highlights', t: 'OpenAI Five (Dota 2), AlphaStar (StarCraft II), emergent tool use in hide-and-seek.', ex: 'Self-play is a special case: you vs a copy of yourself.' },
  ],
  recap: ['Several agents learning together', 'Others’ learning → non-stationary world', 'CTDE and self-play are key tools'] },

{ id: 'sim2real', title: 'Sim-to-Real & Domain Randomisation', emoji: '🦾', track: 'Bigger worlds', pre: ['loop'],
  hook: 'Train in simulation, deploy on a real robot.',
  slides: [
    { h: 'Why simulate?', t: 'Real robots are slow, break, and can’t run 1000 copies in parallel. Simulators can.' },
    { h: 'The reality gap', t: 'Simulated physics, friction and visuals never match reality exactly; policies can fail on the real robot.' },
    { h: 'Domain randomisation', t: 'Randomise friction, masses, colours, lighting, delays during training, so reality looks like one more variation.', a: 'domainRand' },
    { h: 'Example', t: 'OpenAI’s robot hand solved a Rubik’s cube after training only in heavily randomised simulation.', ex: 'Also used for legged robots (ANYmal, Spot-like quadrupeds) walking on rough terrain.' },
  ],
  recap: ['Simulation is cheap and parallel', 'Reality gap breaks naive transfer', 'Randomise the sim to cover reality'] },
];
