## Markov Models and Agents

|                           | No Agents                 | Single Agent                  | Multiple Agents                       |
|---------------------------|---------------------------|-------------------------------|---------------------------------------|
|                           | Markov Chain              | Markov Decision Process (MDP) | Markov Game (a.k.a. Stochastic G ame) |
| State Observed Indirectly | Hidden Markov Model (HMM) | Partially-Observable          | Markov Decision Process (POMDP)       |

(a)

(b)

Figure: (a) Markov decision process (MDP) (b) Partially observable Markov decision process (POMDP) (c) Decentralized partially observable Markov decision process with two agents (Dec-POMDP)

## Multi-agent Applications

- e Antenna tilt Control
- ¢ The joint configuration of cellular base stations can be optimized according to the distribution of usage and topology of the local environment. (Each base station can be modelled as one of multiple agents covering a city.)
- ¢ Traffic congestion reduction
- ¢ By intelligently controlling the speed of a few autonomous vehicles wecan drastically increase the traffic flow
- ¢ Other interesting phenomena (Braess Paradox)

## Braess Paradox

- ¢ Traffic Control Strategies:
- -Build more roads where there is more traffic?

T=Traffic

If 200 vehicles: Total time= 20+(100/10)=30min

=

(The 200 drivers split up as 100+100)

Sometimes, closing down roads can help traffic flow!

## Multi-agent Applications

## ° OpenAl Five

- ¢ Dota 2 Al agents are trained to coordinate with each other to compete against humans.
- Each of the five Al players is implemented as a separate neural network policy and trained together with large-scale PPO.
- They defeated ateam of human pros.

SIT796 Reinforcement Learning

Multi-Agent Reinforcement Learning

Presented by: Thommen George Karimpanal School of Information Technology

## Multi-agent Reinforcement Learning ( AY yay R L)

Source: Nowe, Vrancx &amp;amp; De Hauwere 2012

reward ry

"

oF

a

os

o

a

ae

5

a

Solution

Methods

Tabular

Solution

Methods

Single Agent

- ° MARL
- e Multiple agents joint actions

Approximate

Approximate

Solution

Methods

Tabular

Solution

Methods

Multiple (e.g. 2) Agents

## Types of MARL Settings

## ¢ Decentralized:

- e All agents learn individually
    - Communication limitations defined by environment

## ¢ Descriptive:

    - Forecast how agent will behave

## * Neither:

- e Agents maximize their utility which may require cooperating and/or competing
    - General-sum game

## ¢ Centralized:

    - One brain / algorithm deployed across many agents

## . * Prescriptive:

- ° Suggests how agents should behave

## * Competitive:

- e Agents compete against each other
    - Zero-sum games
    - Individual opposing rewards

## * Cooperative:

- e Agents cooperate to achieve a goal
- ¢ Shared team reward

## Foundations of MARL

## Benefits:

- ¢ Sharing experience via communication ' teaching, imitation
- ¢ Parallel computation due to decentralized task structure
- ¢ Robustness redundancy, having multiple agents to accomplish a task

## Challenges in Multi-agent Learning Systems

## ¢ Curse of dimensionality

- ¢ Exponential growth in computational complexity from increase in state and action dimensions.
- ¢ Also a challenge for single-agent problems.
- ¢ Specifying a good (learning) objective
- e Agent returns are correlated and cannot be maximized independently.

## ¢ The system in which to learn is a moving target

- e As some agents learn, the system which contains these agents changes, and so may the best policy.
- ¢ Also called a system with non-stationary or time-dependent dynamics.

## ¢ Need for coordination

- e Agent actions affect other agents and could confuse other agents (or herself) if not careful. Also called destabilizing training.

## Challenges: Non-stationarity of . Environment

Figure 2: Non-stationarity of environment: Initially (a), the red agent learns to regulate the speed of the traffic by slowing down. However, over time the blue agents learn to bypass the red agent (b), rendering the previous experiences of the red agent invalid.

## Challenges: High Variance of Estimates

Figure 4: High variance of advantage estimates: In this traffic gridlock situation, it is unclear which agents' actions contributed most to the problem --and when the gridlock is resolved, from any global reward it will be unclear which agents get credit.

## In Summary...

- e In single agent RL, agents need only to adapt their behaviour in accordance with their own actions and how they change the environment.
- ¢ In MARL agents also need to adapt to other agents' learning and actions. The effect is that agents can execute the same action on the same state and receive different rewards.

SIT796 Reinforcement Learning

Game Theory

Presented by: Thommen George Karimpanal School of Information Technology

## Game Theory: Concepts

What is Game Theory?

- -The mathematics of conflict
- -Proposed by John Nash in his 27 page PhD thesis
- -Assumes players are rational
- -Increasing number of applications in Al
- -Applications: economics, politics, robotics, etc.,

John Nash

## A simple game

2 player zero sum finite deterministic game with perfect information

## A simple game

b:

@.v:

R

R

7

7

2

3

3

2

-1

4

2

2

2

2

Matrix form of the game

2 player zero sum finite deterministic game with perfect information

L

R

R

R

L

R

## A simple game: minimax

Matrix form of the game

a tries to pick the best row for it b tries to pick the best column for it

Or the other way around - One tries to 'max', the other tries to 'min'

Value of the game

## Nash Equilibrium

Given n players with strategies: S = {So, ... Sj, ... 5}

So € So, St € $4, Sz € So..... Sp € Sy Are in Nash Equilibrium iff:

Basically, in a Nash Equilibrium, if you pick a player at random, they would prefer to not deviate from their optimal strategy, given the optimal strategies of other players

SIT796 Reinforcement Learning

Multi-Agent Reinforcement Learning Formulation

Presented by: Thommen George Karimpanal School of Information Technology

## Stochastic Games

S:State space

A;: Action space for each agent

a€A,,b€E Az

R;: Rewards for each player i

R1(s, (a, b)), R2(s, (a, D))

T: Transitions function

T (s, (a, b), s')

y: Discount factor

Generalisation of the MDP formulation (Shapley) - published before Bellman

## Zero sum Stochastic Games: Bellman 7 Equation

Single agent:

Two agents (zero sum):

But we are no longer the only agent trying to maximise reward! Use minimax!

## First MARL Algorithm: Minimax-Q 7 (Littman '94)

Q-values are over joint actions:

Q(s, a,

    - s= state
    - a= your action
    - o= action of the opponent

Instead of updating Q values with maxQ(s'', a'), use MaxMin

\

Only change from Q learning

## Multi-agent Deep Q-Network (MADQN)

## ¢ MADQN is a Deep Q-Network for Multi-agent RL

- ° Npursuit-evasion - a set of agents (the pursuers) are attempting to chase another set of agents (the evaders)
- ¢ The agents in the problem are
- ¢ self-interested (or heterogeneous), i.e. they have different objectives
- ¢ The two pursuers are attempting to catch the two evaders

## Other Deep RL approaches

- ¢ MADDPG (multi agent deep deterministic policy gradients): multiagent extension of DDPG
- ¢ Multi-Agent Common Knowledge Reinforcement Learning: more focused on cooperative tasks
- ¢ Qmix: For training decentralised policies

SIT796 Reinforcement Learning

Other Related Topics: Action Advising

Presented by: Thommen George Karimpanal School of Information Technology

## Teacher-Student Framework

Does not explicitly fall under multiagent learning, but involves one agent teaching the other

Teacher already knows a good policy

Student learns from scratch, but can ask for advice

Advice is limited, can have an associated cost

Teachers cannot access student knowledge

How can the student quickly best leverage the provided advice while staying within the advice budget?

## Action Advising

## n: Advice Budget

procedure EARLYADVISING(z, n)

for each student state s do

ifn &amp;gt; 0 then

nen-1

Advise 7(s)

procedure IMPORTANCEADVISING(z, 1, t)

for each student state s do ifn &amp;gt; Oand I(s) &amp;gt;

¢t nen-l

Advise 7r(s)

then procedure MISTAKECORRECTING(7, n, t)

for each student state s do Observe student's announced action a if n &amp;gt; Oand I(s) &amp;gt; t anda ¥ 7(s) then nen-1

Advise 7(s)

procedure PREDICTIVEADVISING(z, n, t)

for each student state s do Predict student's intended action a ifn &amp;gt; Oand J(s) &amp;gt; t and a ¥ 7(s) then

nen-l

Advise 7(s)

## Action Advising

bak

0

*Teaching on a Budget: Agents Advising Agents in Reinforcement Learning, Torrey&amp;amp; Taylor (AAMAS, 2013)

2

©

=

=

©

8

Q

3

®

©

=

50

100

150

Training advising

Mistake correcting

Predictive

Importance

Early advising

No advising

advice

200

episodes

--

=

~~

250

300

## Readings

This lecture focused on introducing Multi-agent RL.

For more detailed information see:

- ¢ https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
- ¢ Littman, Michael L. "Markov games as a framework for multi-agent reinforcement learning." Machine learning proceedings 1994. Morgan Kaufmann, 1994. 157-163.
- ¢ Teaching on a Budget: Agents Advising Agents in Reinforcement Learning, Torrey&amp;amp; Taylor (AAMAS, 2013)
- ¢ Multiagent Reinforcement Learning presentation by Marc Lanctot RLSS @Lille, July 11th 2019 http://mlanctot.info/files/papers/Lanctot \_ MARL \_ RLSS2019 \_ Lille.pdf

- https://www.cs.utexas.edu/*larg/ijcai17 tutorial