## Temporal Difference (TD) Learning

## Tabular TD(0) for estimating v,

the 7 to be evaluated

Input: policy

Algorithm parameter: step size a € (0, 1] Initialize V(s), for all s € 8, arbitrarily except that V(terminal) = 0 Loop for each episode: Initialize S Loop for each step of episode: A + action given by m for 3 Take action A, observe R, S' V(S) - V(S) +o[R + »V(S') - V(S)] Ses' until S is terminal

Combination of MC . sampling and . bootstra Ppl ng

The general name is TD(A). This just corresponds to A = 0

## Example: Gridworld with n-step Sarsa

In Searching for a goal using on-policy control

- ¢ The path taken which are also all the states that will be backed up using MC
    - The state-action's learnt using a one-step Sarsa and 10-step Sarsa
- ¢ tis evident that 10 step Sarsa will learn more state-action values.

Path taken

Action values increased by one-step Sarsa

G

Action values increased by 10-step Sarsa

## af aley iam acks

It would be useful to extend what learned at t+1 also to previous states, so to accelerate learning.

Key idea - when performing the update, instead of updating just the current state, why not also update states that led to that state?

Eg: Key helped unlock the treasure, but door led to the key. So when updating the the value of the 'Key' state, make sure some of the credit goes to the 'Door' state as well

## Eligibility Traces: the backward view

## af aley iam acks

Eligibility trace values of non-visited traces decay by a factor of A

E,(s) = yA E:-1(s),

If a state is visited, increase its trace value:

Accummulating traces:

Dutch traces:

E,(S;) = yAEy \_ 1(S;) + 1

Replacing traces:

£;(5;) = 1

Sn

Eligibility traces essentially serve as a sort of memory of which states are relevant to the current update. The end result is convergence speeds up.

-

Ys €8,5 4 Si,

## Learning with Eligibility Traces

Initialize V(s) arbitrarily (but set to 0 if s is terminal) Vola + 0 Repeat (for each episode): Initialize E(s) = 0, for all s € § Initialize S Repeat (for each step of episode): A + action given by a for S Take action A, observe reward, R, and next state, S' Ate V(S) \_ - Vola Vola = V(S") 6¢+ R+7V(S') -V(S) E(S) + (1-a)E(S)4+1 For all s € 8: V(s) «+ V(s) +a(6+ A) E(s) E(s) + yA E(s) V(S)&amp;lt;«+V(S)-aA Ses!

until S is terminal

## Set A = 0, we get TD(0)

## A = 1 corresponds to Monte-Carlo

## Re ea UA

```
Initialize Q(s,a) arbitrarily, for all s € 8,a € A(s) Repeat (for each episode): E(s,a) = 0, for all s € 8,a € A(s) Initialize S, A Repeat (for each step of episode): Take action A, observe R, S' Choose A' from S' using policy derived from Q (e.g., e-greedy) 5 R+7Q(S', 4!) - Q(S, A) E(S,A) «+ E(S,A)+1 (accumulating traces) or E(S,A) + (l-a)E(S,A)+1 (dutch traces) or E(S,A) «+ 1 (replacing traces) For all s € 8,a € A(s): Q(s,a) & Q(s,a) + 05 E(s,a) E(s,a) + yAE(s,a) SS ACA' until S is terminal
```

## Taya ed ae He] 0)

Initialize Q(s,a) arbitrarily, for all s € 8,a € A(s) Repeat (for each episode): E(s,a) = 0, for all s € 8,a € A(s) Initialize S, A Repeat (for each step of episode): Take action A, observe R, S' Choose A' from S' using policy derived from Q (e.g., e-greedy) A* + argmax, Q(S',a) (if A' ties for the max, then A* + A') 6&amp;lt; R+ 7Q(S', A*) ~ Q(S, A) E(S, A) + E(S,A)+1 (accumulating traces) or E(S,A) «+ (l-a)E(S,A)+1 (dutch traces) or E(S,A) + 1 (replacing traces) For all s € 8,a € A(s): Q(s,a) + Q(s,a) +06 E(s,a) If A' = A*, then E(s,a) + yAE(s, a)

else E(s,a) + 0

S¢S;A¢ A!

until S is terminal

SIT796 Reinforcement Learning

Planning, Learning and Dyna

Presented by: Thommen George Karimpanal School of Information Technology

## Learning with a model

So far we discussed about learning from interactions

We don't actually need to experience things to learn - we can imagine!

We imagine by building and playing out models of the world, build from interaction data

## Learning with a model

Updating values with interactions with the real world: Learning

- -Real world interactions are expensive
- -Robots/environment can get damaged
- . . -Experiments can take time

Updating values with interactions with the real world: Planning

- -But generally, our model is not perfect
- -If it was we would never really need to interact with the real world
- -Models are built using interactions in the first place

. n VO i a eo r Oo x Ss oO zv

## Integrating Planning and Learning

Both planning and direct RL aims to update the value/policy

So both processes can be interrupted if needed

## DYNA architecture

Initialize Q(s,a) and Model(s,a) for all s € 8 and a € A(s)

Do forever:

- (a) S + current (nonterminal) state
- (b) A &amp;lt; e-greedy(S, Q)
- (c) Execute action A; observe resultant reward, R, and state, S'

| (f) Repeat n times: S &lt; random previously observed state A «+ random action previously taken in S R, S' + Model(S, A) Q(S, A) — Q(S, A) +a [R + ymax Q(S',a) — Q(S, A)   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

- --

## Learning

Model update

Planning

## DYNA architecture

More planning steps means faster learning

## When the model is wrong: Blocking Maze

During the latter half of learning, the model makes optimistic predictions, which never come true

After sometime, the correct path is found

Providing a dedicated exploration bonus (as in Dyna-Q+) is useful

## When the model is wrong: Shortcut Maze

Dyna-Q tends to get stuck and fails to improve - because the model's predictions are correct and the agent still reaches the goal

Dyna-Q+ eventually discovers the shortcut

## Prioritized Sweeping

Similar to Dyna, but experiences are prioritized

Q: learning rate Y: discount factor

Temporal difference (TD) error

Higher error experiences are prioritized

## Prioritized Sweeping

Initialize Q(s,a), Model(s,a), for all s,a, and PQueue to empty Do forever:

- (a) S + current (nonterminal) state
- (b) A &amp;lt; policy(S, Q)
- (c) Execute action A; observe resultant reward, R, and state, S'
- (d) Model(S, A) + R, S'
- (f) if P &amp;gt; 0, then insert S, A into PQueue with priority P
    - (f) if P &amp;gt; 0, then insert S, A into PQueue with priority P
- (g) Repeat n times, while PQueue is not empty:
- S,A&amp;lt; first(PQueue)

R, S' + Model(S, A)

Q(S, A) € Q(S,A) + a[R + ymax, Q(5',a) - Q(S,A)]

Repeat, for all S,A predicted to lead to S:

- R + predicted reward for S$, A, 9

| R+~ymax, Q(S, a) — Q(S, A)   |
|------------------------------|

- if P &amp;gt; @ then insert S,A into PQueue with priority P

Model update

Prioritization

## Planning

Update priority queue

## Readings

## This lecture focused on eligibility traces and DYNA.

- ¢ Future topics will look at function approximation techniques and alternatives to value based approaches

For more detailed information see Sutton and Barto (2018) Reinforcement Learning: An Introduction (Version 2)

- ¢ Chapter 12 and Chapter 8
- ° - http://incompleteideas.net/book/RLbook2020.pdf

Other Readings -Sutton and Barto (1998) Reinforcement Learning: An Introduction (Version 1)

- ¢ Chapter 7: Eligibility Traces
- ° http://incompleteideas.net/book/first/ebook
- ¢ - Note: this book is now primarily obsolete but some parts are worth knowing as they are commonly used