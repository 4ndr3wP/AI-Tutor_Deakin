## RL so far:

Agent takes actions in its environment

Maximise sum of rewards

Can all problems be expressed in the form of rewards?

## Are rewards enough?

Sutton's Reward Hypothesis: "All of what we mean by goals and purposes can be well thought of as maximization of the expected value of the cumulative sum of a received scalar signal (reward)."

Artificial Intelligence

Volume 299, October 2021, 103535

## Reward is enough

David Silver 2 &amp;amp;%, Satinder Singh, Doina Precup, Richard S. Sutton

Show more V

    - AddtoMendeley Share 35 Cite

https://doi.org/10.1016/j.artint.2021.103535 A

Under a Creative Commons license A

Get rights and content "

@

open access

But how to compress all objectives into such a scalar reward? - Not easy!

A more pragmatic approach: we often care about multiple objectives, which need to be optimised together Multi-Objective Reinforcement Learning

## Multiobjective RL

How to deal with multiple objectives?

One reward function for each objective

TN)

1

But how deal with these?

Combine them into a scalar value:

T = WN, + Wo

w specifies the preferences

## Multiobjective RL

Objective 2

Single objective: only 1 solution

MultiObjective: multiple solutions (even with 2 objectives)

Optimal Solutions (Pareto front): non-dominated policies

-Same performance in terms of the scalarised rewards

In general, n \_ objectives&amp;gt;2

Objective 1

## Specifying Preferences

At the end, we care about agent behaviours

Pick a w, check if any of the solutions correspond to the desired behaviour

If not, pick a different w

Semi-blind process

## Problems with Scalarisation

Undue burden on engineers/designers

Linear model cannot encompass complex preferences we may have

Preferences change over time

Power production application -

T= Wyl + Wel for double the power, cannot just double w,,

The solutions are not explainable

## MORL Examples

Wind farms: maximise power, minimise wear

Other non-linear factors

Transport: minimise travel time, also minimise cost

Can prepare yourself with a set of policies - useful when trains are cancelled etc.,

SIT796 Reinforcement Learning

MORL: Problem setting and Formulation

Presented by: Thommen George Karimpanal School of Information Technology

MultiObjective MDP

S is the state space

A is the action space

T:SxXAxXS - [0, lis a probabilistic transition function

y € [0, 1) is a discount factor

Hw : S &amp;gt; [0, 1]is a probability distribution over initial states

R:SxAxS-- R¢ is a vector-valued reward function, specifying the immediate reward for each of the considered d &amp;gt; 2 objectives

## MOMDP: Utility Function

## In MOMDPS, the value function is a vector

Utility functions scalarise the multiobjective value vector to a scalar

The optimal policy is not clearly defined unless we know how the objectives are prioritised

However, scalarisation has its own problems, as discussed earlier

SIT796 Reinforcement Learning

MORL: Taxonomy

Presented by: Thommen George Karimpanal School of Information Technology

| MORL: Taxonomy                                                                                   | MORL: Taxonomy   | MORL: Taxonomy                                                                 |
|--------------------------------------------------------------------------------------------------|------------------|--------------------------------------------------------------------------------|
| Single                                                                                           | vs               | Multiple policies                                                              |
| Single policy- If utility is knownatthe time of planning                                         |                  | Multiple policies- If utility is unknown                                       |
| Linear utility preferences                                                                       | vs               | Non-linear Utility policies                                                    |
| User maynotbe adequately expressed                                                               |                  | Maybetterexpressuser preferences                                               |
| Deterministic policies Whenutility is linear , the optimal policy is deterministic andstationary | vs               | Stochastic policies In somecases, stochastic policies should never bepermitted |

## MORL: Optimisation Criteria

## Scalarised Expected Returns (SER)

## Expected Scalarised Returns (ESR)

These lead to different solutions when the utility is non-linear

## MORL: Metrics

## Hypervolume

x

Part of

non-dominated

x

Dominated

Larger the hypervolume the better set

## MORL: Metrics

When the hypervolume is similar, choose the solution that is most spread out in the space

Sparsity metric for m objectives. S is the pareto front approximation

SIT796 Reinforcement Learning

MORL: Algorithms

Presented by: Thommen George Karimpanal School of Information Technology

## MORL: Single Policy Algorithms

## Adaptation of Q learning

- Q vectors instead of Q values

Scalarisation function is needed to for action selection

*Miay fail to converge if transitions are stochastic

## MORL: Multi-Policy Algorithms

Pareto Qlearning: based on dynamic programming variant that returned pareto dominating policies

Episodic problems with terminal state

Model-free

Produces deterministic non-stationary policies

Set Evaluation Mechanisms. (based on hypervolume, cardinality etc.,)

## MORL: Multi-Policy Algorithms

## Pareto Q learning

- 1: Initialize Qser(s, a)'s as empty sets
- 2: for each episode t do
- 3: Initialize state s

## Set Evaluation Mechanisms

- 4 repeat
- 5 Choose action a from s using a policy derived from the Qse¢'s
- 6: Take action a and observe state s' € S and reward vector r € R™

7:

- 8 ND;(s,a) - ND(UaQset(s', a'))
- 9 R(s,a) - R(s,a) +
- 10: ses!
- 11: until s is terminal
- 12: end for
- ae)

&amp;gt; Update ND policies of s' in s &amp;gt; Update average immediate rewards &amp;gt; Proceed to next state

## MORL: Multi-Policy Algorithms

## Hypervolume Set Evaluation

- : Retrieve current state s
- : evaluations = {}
- : for each action a do
- ha - HV(Qset(s, a))
- Append hv, to evaluations
- : end for
- : return evaluations
- : Retrieve current state s
- : for each action a in s do
- none : allQs = {}

CaorNtawh

- for each Q in Qset(8, a) do
- Append [a, Q] to allQs
- end for
- : end for
- : NDQs - ND(allQs)
- : return NDQs

Cardinality Set Evaluation: based on number of Pareto dominating O-vectors of the Qset of each action

- &amp;gt; Store hypervolume of the Qset(s, a)

&amp;gt; Store for each Q-vector a reference to a

&amp;gt; Keep only the non-dominating solutions

## MORL Benchmarks

MultiObjective Gymnasium https://mo-gymnasium.farama.org/index.html

SIT796 Reinforcement Learning

MORL: Related Topics and Open Questions

Presented by: Thommen George Karimpanal School of Information Technology

## MORL Related Topics

Human-alignmenthow to take humans' preferences into account

RL SafetyTraining RL algorithms efficiently, but at the same time, avoiding unsafe actions

Explainable RL, Moral decision making https://www.moralmachine.net/

## MORL Open Questions

Many-Objective Problems (n \_ objectives&amp;gt;4)

MultiAgent RL Problems (MOMADM). -several challenging problems

How to dynamically identify and add objectives?

## Readings

This lecture focused on introducing Multi-objective RL.

For more detailed information see:

- ¢ Hayes, Conor F., et al. "A practical guide to multi-objective reinforcement learning and planning." Autonomous Agents and Multi-Agent Systems 36.1 (2022): 26.
- ¢ Van Moffaert, Kristof, and Ann Nowé. "Multi-objective reinforcement learning using sets of pareto dominating policies." The Journal of Machine Learning Research 15.1 (2014): 3483-3512.
- ¢ Miguel Terra-Neves, Ines Lynce, Vasco Manquinho - Stratification for Constraint-Based Multi-Objective Combinatorial Optimization
- Diederik M. Roijers, Luisa M. Zintgraf, Pieter Libin, Ann Now'e - Interactive Multi-Objective Reinforcement Learning in Multi-Armed Bandits for Any Utility Function
- ¢ Felten, Florian, et al. "A toolkit for reliable benchmarking and research in multi-objective reinforcement learning." Advances in Neural Information Processing Systems 36 (2024).