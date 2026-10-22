# Safe Over Short: Learning to Trace Unseen Characters with Reinforcement Learning

A pen starts at one end of a handwritten character and has to follow the stroke to the other end. Each step it moves up, down, left or right, but **5% of the time the hand slips** and a random move happens instead. If the pen leaves the stroke, the episode ends.

The shortest path cuts corners, where a single slip throws the pen off the stroke. A good policy stays in the middle of the "ink road", even if that takes a few more steps. **Safe beats short.**

Course project for **Reinforcement Learning (RWTHx, RWTH Aachen University, edX)**. Every method from the course is implemented from scratch and compared on this one problem, so the differences come from the algorithms, not from different tasks.

## Key results

### Unseen alphabets (500 test tasks)

| Policy | Success ↑ | SPL ↑ |
|---|:-:|:-:|
| Random | 0.0% | 0.0 |
| Compass (step towards the goal) | 4.4% | 4.3 |
| Shortest path (full map) | 75.4% | 72.6 |
| **DP optimal, plans for slips (full map)** | **95.0%** | **88.1** |
| Behaviour cloning of the DP planner (11×11 window) | 65.8% | 60.3 |

*SPL = success weighted by path length (Anderson et al., 2018), so wandering is penalized.*

### One glyph, tabular methods (mean over 3 seeds)

| Method | Success with 5% slips |
|---|:-:|
| Shortest-path policy | 66.2% |
| **Optimal policy (value iteration)** | **94.1%** |
| Sarsa | 93.3% ± 2.0 |
| Q-learning | 92.9% ± 2.5 |
| Dyna-Q (5 planning steps) | 93.0% ± 0.7 |
| On-policy Monte Carlo (exploring starts) | 78.1% ± 4.4 |
| Off-policy Monte Carlo (weighted importance sampling) | 0.0% |

## Findings

1. **Planning for uncertainty matters more than path length.** On the same maps, the slip-aware DP policy succeeds 95.0% of the time on unseen alphabets, against 75.4% for the shortest path. DP finds this only because the slip is part of its model.
2. **TD methods learn near-optimal behaviour.** Sarsa, Q-learning and Dyna-Q reach about 93% success, within about one point of the true optimum (94.1%).
3. **Off-policy Monte Carlo fails here.** It never learned a working policy in the step budget. The most likely reason is that with long episodes and a greedy target policy, the importance weights vanish after the first non-greedy action, so very little of each episode is used for learning.
4. **Reward hacking in deep RL.** In the first full run, the deep agents (DQN, REINFORCE, A2C, PPO) learned to end episodes early instead of tracing, because each step on the stroke cost a small penalty and leaving early was cheaper than risking the trace. The reward was redesigned to remove this shortcut: a much larger penalty for leaving the stroke, a smaller per-step cost, and **potential-based reward shaping** (Ng, Harada and Russell, 1999), which guides the agent towards the goal without changing which policy is optimal.

## The task

- **Data:** real Omniglot characters (Lake et al., 2015). Each glyph is skeletonized and widened to a 3-pixel "ink road"; the start and goal are the two ends of its longest on-stroke path.
- **Tabular part:** one glyph, 28 × 28 = 784 states, solved exactly so that every learner can be measured against the true optimum.
- **Generalization part:** 1,852 training tasks and 500 test tasks from alphabets never seen in training. Agents only see an **11×11 window** around the pen (stroke and visited cells) plus the direction to the goal, never the full map.
- **Episode limit:** 3 × the shortest path length.

## Methods implemented from scratch

| Course part | Methods |
|---|---|
| MDPs and dynamic programming | policy evaluation, policy iteration, value iteration |
| Monte Carlo | on-policy control with exploring starts, off-policy control with weighted importance sampling |
| Temporal-difference learning | Sarsa, Q-learning |
| Planning and learning | Dyna-Q with a planning-steps sweep (0, 5, 20) |
| Supervised learning | behaviour cloning of the DP planner |
| Function approximation | DQN with replay buffer and target network, 16 parallel environments |
| Policy gradients | REINFORCE with baseline, A2C, PPO |

**A detail that matters:** the textbook Dyna-Q model stores only the last outcome of each state-action pair, which is right for deterministic worlds. With slips it would sometimes remember a slip as the action's only result, and planning would keep repeating that mistake. Here the model keeps every observed outcome and samples from them, which matches the real slip distribution.

## How to run

1. Open the notebook in Google Colab with a GPU runtime and select **Run all** (roughly 30 to 40 minutes).
2. Set `FAST_DEV_RUN = True` first for a few-minute end-to-end check.
3. Sections 1 to 4 (DP, Monte Carlo, TD, Dyna-Q) use only NumPy; the deep RL sections use PyTorch.

## Main libraries

NumPy · PyTorch · scikit-image · pandas · Matplotlib

## Why this task

It connects to my research on few-shot glyph generation. An agent that can follow strokes is a step towards generating characters stroke by stroke, as in SPIRAL (Ganin et al., 2018) and Learning to Paint (Huang et al., 2019).

## Limitations

The deep RL numbers in the first run were affected by the reward issue described above; they will be updated after the full run with the redesigned reward. One seed per deep method.

## Author

**Sadaf Zehra**, MS Data Science, NED University of Engineering and Technology
[GitHub](https://github.com/SadafZehra) · [LinkedIn](https://www.linkedin.com/in/sadaf-zehra-796858230) · [ORCID](https://orcid.org/0009-0009-7530-9006)
