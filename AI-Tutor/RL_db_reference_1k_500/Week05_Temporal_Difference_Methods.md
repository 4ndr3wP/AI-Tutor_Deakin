## Introduction &amp;amp; History,

## RL formulation: States, Actions, Rewards, Transition function, value function, action-value function

Formally, the RL problem is formulated as a Markov Decision Process (MDP)

- ¢ AnMDP is atuple M={S, A, JT, y, R}
- - § -The set of possible states
- - AThe set of actions the agent can take
- -- J -transition probabilities
- - y -The discount rate or the discount factor.
- - RAreward distribution function conditioned.

Psychological aspects

## Multiarmed Bandits

## Types of MDPs

## Dynamic Programming

## Iterative Policy Evaluation, for estimating V ¥ vu,

. Input 7, the policy to be evaluated

Algorithm parameter: a small threshold 9 &amp;gt; 0 determinir Initialize V(s) arbitrarily, for s € 8, and V(terminal) to

Loop:

A&amp;lt;O0

## Policy Iteration (using iterative policy evaluation) for estimating 7

    1. Initialization

V(s) € R and z(s) € A(s) arbitrarily for all s € 8; V(terminal) = 0

    1. Policy Evaluation

Loop:

A+0

Loop for each s € 8:

v¢

V(s)

F

. . until A &amp;lt; @ (a small positive number determining the ¢

## Value Iteration, for estimating 7 ~ 7,

Loop for each s € 8:

v&amp;lt;

V(s)

V(s)

-

Yo,

(als)

Ate until A &amp;lt; 6

max(A, lu

\_

Algorithm parameter: a small threshold @ &amp;gt; 0 determining accuracy of estimation Initialize V(s), for all s € 8*, arbitrarily except that V(terminal) = 0

Loop:

Aco

Loop for each s € 8:

ve

Vs)

V(s)

A

&amp;lt;

maxa max(A,

&amp;gt;&amp;gt;,

-

until A &amp;lt; 6

V(s

r

[r

o

Vv(

@)

[r+

7V(s')]

## Monte-Carlo Methods

## - \_ Monte-Carlo Prediction

## First-visit MC prediction, for estimating V ~ 1

3 episodes, y = 1

Input: a policy 7 to be evaluated

Initialize:

V(s) ER, arbitrarily, for all s € $

Returns(s) + an empty list, for all s € §

Loop

## foreve

## Monte Carlo ES (Exploring Starts), for estimating 7 ~ 7,

Generat

Geo

Loop for

Ge nie

4

}

## On-policy first-visit MC control (for ¢-soft policies), estimates 7 ~ 7,

## Initialize:

m(s) € A(s) ), for alls € 8

Q(s,a) ER alls €8,a€ A(s)

Returns(s,a) + empty list, for all s € 8, a € A(s)

(arbitrarily (arbitrarily), for

Loop forever (for each episode):

Choose So € 8, Ao € A(So) randomly such that all pairs have probability &amp;gt; 0 : : Generate an episode from So, Ao, following 7: So, Ao, Ri,...,Sr-1, Ar-1, Rr Geo

Loop for each step of episode, t = T-1,T-2,...,0:

Ge yG+ Riti

Unless the pair S;, Ay appears in So, Ao, $1, A1..., S¢-1, At \_ 1:

Append G to Returns(S;, At)

Q(S:, At) &amp;lt; average(Returns(S;, At aca

Algorithm parameter: small ¢ &amp;gt; 0

Initialize:

m &amp;lt; an arbitrary ¢-soft policy

Q(s, a) € R (arbitrarily), for all s € 8, a € A(s)

Returns(s,a) - empty list, for all s € 8, a € A(s)

Repeat forever (for each episode):

Generate an episode following 7: So, Ao, Ri,...,Sr-1, Ar-1, Rr

Gro

Loop for each step of episode, t = T-1,T-2,...,0:

Ge yG+ Riz

Unless the pair S;, A; appears in So, Ap, $1, Ai... ,S¢-1, At-1: Append G to Returns(St, At)

At)

Q(St,

- \_ prerage( Return (Ee, Ae)

For all

A 5 ae pe

zisa&amp;lt;

'Monte-Carlo Cantrol

## Pros &amp;amp; Cons

| DP                                                  | Monte-Carlo                                                               |
|-----------------------------------------------------|---------------------------------------------------------------------------|
| Transition and reward model required                | No model needed                                                           |
| Requires full sweep of the state/state-action space | Requires full trajectories of experience                                  |
| Convergence guaranteed                              | Converges only if states/state-actions are visited enough number of times |

In reality, we don't have perfect models, and we don't need full . . trajectories to learn

- - we learn on-the-go

SIT796 Reinforcement Learning

Temporal Difference (TD) Learning

Presented by: Thommen George Karimpanal School of Information Technology

## Temporal Difference (TD) Learning

A combination of Dynamic Programming and Monte Carlo methods:

- ¢ Like Monte Carlo it learns from experience and doesn't need a model
- ¢ Like Dynamic Programming it updates estimates using other learned estimates

Temporal difference

Related to time

Learning happens as time progresses -can update values without having to wait till the end of the episode

## Temporal Difference (TD) Learning

In Monte-Carlo, we wait till the end of the episode and update values accordingly:

This is the target

In TD, we take one step (and experience one reward) and still try to update

the value:

This is the target for TD

## Temporal Difference (TD) Learning

## Tabular TD(0) for estimating v,

Input: the policy 7 to be evaluated Algorithm parameter: step size a € (0, 1] Initialize V(s), for all s € S*, arbitrarily except that V(terminal) = 0 Loop for each episode: Initialize S Loop for each step of episode: A+ action given by z for S Take action A, observe R, 9' V(S) - V(S) +a[R+ W(S') - V(S)] Ses! until 3 is terminal

Combination of MC sampling and ; b ootstra p pl n g

The general name is TD(A). This just corresponds to A = 0

The TD target R+ yV(S'). isa combination of both samples R and an estimate yV(S') :

## TD -- error

Depends on the reward and next state

This is the error TD methods aim to minimise

## TD vs MC vs DP

## Clearly TD (Like MC) is better than DP if you do not have a model

- ¢ Most real world problems we do not have a complete or even a partial model - making DP impossible
- ¢ DPis also highly computationally intensive.

## TD is also obviously better over MC in online environments

- ¢ = MCmethods must wait until the end of the episode to be updated - problematic in long or continuing tasks.
- ¢ - If exploration has occurred, then many updates can not be made.
- ¢ Whereas, TD you can update more often (each step) - doesn't matter how long the episode is.
- ¢ TD is less susceptible to issues around exploratory actions - only affects the step when the exploration was taken.

## TD methods have been proven to converge when the policy is fixed.

- ¢ - Shown for table-based methods in MDPs

## But which is faster (MC or TD)?

A simple MDP where you start at C and move left or right until you hit a terminal state.

- ¢ - Provided policy is just a 50/50 random split between left or right actions.
- ¢ TD Prediction aims to learn the true value for each state
- ¢ - In this example TD is always better than MC.

TD makes more efficient use of data

## TD Control

## TD for Control

- ¢ On-Policy - where our behaviour policy is the same policy we are learning (SARSA)
- ° Off-Policy - where we have a separate behaviour policy from the target policy we are attempting to learn (Q learning)

## SARSA: On-Policy TD Control (2)

Uses quintuple (S;, Az, Res4, Sta1, Ats1). Hence the name SARSA

## Sarsa (on-policy TD control) for estimating Q ~ q,

```
Algorithm parameters: step size a € (0,1), small « > 0 Initialize Q(s, a), for all s € 8*,a € A(s), arbitrarily except that Q(terminal,-) = 0 Loop for each episode: Initialize S Choose A from S using policy derived from Q (e.g., e-greedy) Loop for each step of episode: Take action A, observe R, S' Choose A' from S$' using policy derived from Q (e.g., ¢-greedy) Q(S,A) & Q(S, A) + o[R + 7Q(S', A') - Q(S, A)] S¢S',ACA'; until S is terminal
```

Convergence of SARSA is guaranteed if all state action pairs are guaranteed to be sampled an infinite number of times

## Q-Learning: Off-Policy TD Control

The off-policy equivalent to SARSA is known as Q-Learning (Watkins, PhD thesis 1989).

Notice the subtle difference to SARSA - the learning target and the way the actions are selected is different

## SARSA vs Q-learning

. . . . Both SARSA and Q learning implemented with ¢-greedy exploration (€ = 0.1)

Q learning target only "cares" about the best action

Even if there is some non-zero probability of falling into the cliff, Q learning still prefers the risky (but optimal) path

However, SARSA is on-policy - so it accounts for the action-selection and prefers the safer path

Note: if € was reduced to 0 overtime then both algorithms will converge to . "as » . an optimal "cliff edge" policy

## Expected SARSA

Expected Sarsa updates its value based on the expected reward - incorporating how likely an action is to be taken.

- e It constructs the learning target based on the probability of each action - doesn't rely on maxQ(s',a') like Q learning, nor does it rely on the taken action.
    - So it can act as either On-policy or Off-policy.

SIT796 Reinforcement Learning

Maximisation bias and double Q Learning

Presented by: Thommen George Karimpanal School of Information Technology

## Maximization bias

MAX

aq:

Qtrue(S, a')

=0

MAXq:Qest(s, a') &amp;gt; 0

This positive bias is called maximization bias

## Double Q-Learning

Proposed as a solution to biased learning

Maintain two values for each action (Q;(a) and Q&amp;gt;(a))

We then randomly choose one of these values to decide which action we are going to use to select the maximum action

But then update the value of the other Q value for that action.

Can be proved that this addresses the maximization bias issue

## Double Q-Learning

## Double Q-learning, for estimating Q) ~ Q2 ~ q

Such "double" versions also exist for SARSA and expected SARSA

Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement S P learning with double qlearning. Proceedin gs of the / conference on artificial intelligence. Vol. 30. No. 1. 2016.

## Double Q-Learning

## n-step TD learning

: The TD target relies partially on samples (experience) and partially on bootstrapping estimates

So far, we only looked at 1 step of experience. But in general, we can roll out multiple steps (nsteps)

## : aReciateetetat stan alee eh ee telah a"

| Ift&lt;T, then: Take an action according to 7(-   | S;) Observe and store the next reward as R;,, and the next state as S14, If S:41 is terminal, then T &lt;-t+1 7T&lt;-t—n+1 (ris the time whose state’s estimate is being updated)   |    |    |    | Ifr&gt;0:   |    |    | Ge yrmin(r tn, .T) yi T-1R, t=T+1 Ifr+n&lt;T, then: Ge G4+y7"V(S,4n) (Gr:rtn) V(S,) + V(S,) + a[G — V(S,)   |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----|----|----|-------------|----|----|-------------------------------------------------------------------------------------------------------------|

## n-step Sarsa (TD On-policy Control)

## This bootstrapping idea can easily be extended to On-Policy Control (and Expected Sarsa)

- ¢ Whereas, in n-step TD we add all the rewards along with the final state estimate. In Sarsa we add all the rewards with the final state-action value estimate.

- ¢ - In Expected Sarsa we do the same except in the last step we use a weighted sum of the estimates of possible actions.

    - Where V;(s) is the expected approximate value of state s.
- ¢ - This is calculate using the estimated action values of all states from s weighted by the probability of their being selected using policy 7

| 2-step Sarsa   | n-step Sarsa   | co-step Sarsa aka Monte Carlo   | Expected Sarsa n-step   |
|----------------|----------------|---------------------------------|-------------------------|
| 1              | I e-—O-—*-—O   | e-—C-—e-—(C.)+—0                | @+—O+—0-+—(O)+—         |
|                | wee            |                                 | +++                     |
|                |                | eee                             |                         |
|                | e-—C-—e        |                                 |                         |
|                |                | Gi-e                            | on                      |

## Example: Gridworld with n-step Sarsa

In Searching for a goal using on-policy control

- ¢ The path taken which are also all the states that will be backed up using MC
- ¢ The state-actions learnt using a one-step Sarsa and 10-step Sarsa
    - It is evident that 10 step Sarsa will learn more state-action values.

## Path taken

Action values increased by one-step Sarsa

G

4

Action values increased by 10-step Sarsa

## Putting it all together

## Readings

This lecture focused on methods for solving MDPs using Temporal difference.

- ¢ Future topics will look into advanced topics in using Temporal difference learning.
- ¢ Ensure you understand what was discussed here before doing the following topics

For more detailed information see Sutton and Barto (2018) Reinforcement Learning: An Introduction

    - Chapter 6: Temporal-Difference Learning
- ° http://incompleteideas.net/book/RLbook2020.pdf
- ¢ Lecture content has been borrowed from the above mentioned book