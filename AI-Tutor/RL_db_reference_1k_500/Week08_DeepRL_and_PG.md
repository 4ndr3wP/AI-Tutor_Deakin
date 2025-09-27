## Recap: Function approximation

## Neural Networks as a function approximator

- Rumelhart 1986 - great early success
- Interest subsides in the late 90's as other models are introduced - SVMs, Graphical models, etc.
- Convolutional Neural Nets - LeCun ,1989 ,for Image recognition, speech, etc.
- Deep Belief nets (Hinton) and Stacked auto-encoders (Bengio) in 2006
- Unsupervised pre-training followed by supervised training
- Good feature extractors.
- 2012 Initial successes with supervised approaches which overcome vanishing gradient

## What is a Deep Network?

Krizhevsky, Sutskever, Hinton - NIPS 2012

- First layer learns 1st order features (e.g. edges...)
- Higher order layers learn combinations of features (combinations of edges, etc.)
- Some models learn in an unsupervised mode and discover general features of the input space - serving multiple tasks related to the unsupervised instances (image recognition, etc.)
- Final layer of transformed features are fed into supervised layer(s)
- And entire network is often subsequently tuned using supervised training of the entire net, using the initial weightings learned in the unsupervised phase

## Properties

- Neural networks have been shown to be a universal function approximator (it can approximate any function given enough data and model complexity)
- Convolutional nets - for image inputs, MLP: for other type of inputs, Time series: Recurrent NNs etc.,
- It is after all, a supervised learning technique. So data must be i.i.d. (Independent and identically distributed)

SIT796 Reinforcement Learning

Deep Nets and Reinforcement Learning

Presented by: Thommen George Karimpanal School of Information Technology

## What is Deep Reinforcement Learning?

- ¢ Deep reinforcement learning is standard reinforcement learning where a deep neural network is used to approximate either a policy or a value function
- ¢ Deep neural networks require lots of real/simulated interaction with the environment to learn
- ¢ Lots of trials/interactions is possible in simulated environments
- ¢ Wecan easily parallelise the trials/interaction in simulated environments
- ¢ We cannot do this with robotics (no simulations) because action execution takes time, accidents/failures are expensive and there are safety concerns

## Deep Q-Networks (DQN)

- ¢ It is common to use a function approximator Q(s, a; 8) to approximate the action-value function in Q-learning
- ¢ Deep Q-Networks is Q-learning with a deep neural network function approximator called the Q-network
    - Discrete and finite set of actions A
- ¢ Example: Breakout has 3 actions - move left, move right, no movement
- ¢ Uses epsilon-greedy policy to select actions
- Core idea: We want the neural network to learn a non-linear hierarchy of features or feature representation that gives accurate Q-value estimates
- The neural network has a separate output unit for each possible action, which gives the Q-value estimate for that action given the input state (Can also be coded such that each state-action pair produces one Q value as output)
- The neural network is trained using mini-batch stochastic gradient updates and experience replay
- -Go though batches in the dataset
- -These batches make up for an epoch
- -Go through several epochs until convergence

## Experience Replay

- ¢ Experience is a sequence of states, actions, rewards and next states e, = (S,, a, lt Stat)
- ¢ Store the agent's experiences at each time step e, = (S;, ay, I, Si41) in a dataset D =e,,..., €, pooled over many episodes into a replay memory
- ¢ In practice, only store the last N experience tuples in the replay memory and sample when performing updates
- e These experiences occur in sequence. So need to randomise them to make them i.i.d.
- e It may require a lot of experience to obtain enough samples.

SIT796 Reinforcement Learning

Deep Q-Learning

Presented by: Thommen George Karimpanal School of Information Technology

## Q-Network Training

- Sample random mini-batch of experience tuples uniformly at random from D (replay buffer)
- Similar to Q-learning update rule but:
- Use mini-batch stochastic gradient updates
- The gradient of the loss function for a given iteration with respect to the parameter @; is the difference between the target value and the actual value is multiplied by the gradient of the Q function approximator Q(s, a; 8) with respect to that specific parameter
- Use the gradient of the loss function to update the Q function approximator

## Loss Function Gradient Derivation

network. We refer to a neural network function approximator with weights @ as a Q-network. A Q-network can be trained by minimising a sequence of loss functions L;(6;) that changes at each iteration i,

| r + ymax,' Q(s',a';0;_,)   |
|----------------------------|

Rather than computing the full expectations in the above gradient, it is often computationally expedient to optimise the loss function by stochastic gradient descent. If the weights are updated after every time-step, and the expectations are replaced by single samples from the behaviour distribution p and the emulator € respectively, then we arrive at the familiar Q-/earning algorithm [26].

## DQN Algorithm

## Algorithm 1 Deep Q-learning with Experience Replay

```
Initialize replay memory D to capacity N Initialize action-value function Q with random weights for episode = 1, M do Initialise sequence s; = {x} and preprocessed sequenced ¢; = ¢(s1) for t = 1.7 do With probability € select a random action a, otherwise select a, = max, Q*(@(s,), a; 0) Execute action a; in emulator and observe reward r; and image 2:4. Set 8441 = 82, @¢, 2:41 and preprocess d:41 = (8:41) Store transition (o; at,Tr, $141) in D Sample random minibatch of transitions (¢;,a;,7j;,;41) from D rj for terminal Oj41 rj +ymaxe Q(dj+1, 0'; 9) for non-terminal ¢; 41 Perform a gradient descent step on (y; - Q(¢;, a;; 0)) according to equation|3} end for end for Set y; = {
```

## DQN in Practice: Trick 1 - Experience Repla

- ¢ It was previously thought that the combination of simple online reinforcement learning algorithms with deep neural networks was fundamentally unstable
    - The sequence of observed data (states) encountered by an online reinforcement learning agent is non-stationary and online updates are strongly correlated
- ¢ The technique of DQN is stable because it stores the agent's data in experience replay memory so that it can be randomly sampled from different time-steps
- ° Aggregating over memory reduces non-stationarity and decorrelates updates but limits methods to off-policy reinforcement learning algorithms
    - Experience replay updates use more memory and computation per real interaction than online updates, and require off-policy learning algorithms that can update from data generated by an older policy

## DQN in Practice: Trick 2

Initialize replay memory D to capacity N

Initialize action-value function Q with random weights 0

Initialize target action-value function Q with weights 0 = 0

For episode = 1, M do

Initialize sequence s; = {x, } and preprocessed sequence ¢, = $(s,)

ith ith probability é select a random action a, otherwise select a; =argmax, O(6(s;),a; 0)

ecute action a, in emulator and observe reward r, and image x; + ;

et St41

= Ste Xt+1 and $141 =O(St41)

a.

erform a gradient descent step on network aiapmeatig 0

if episode terminates at step j+1

otherwise

(45,4)

0))

2

with respect

(

-Q

to the

Sampling

Training

Use two networks: a policy network and target network

Freeze the target network and

update it only after C steps

Tends to stabilize learning

## DQN in Practice: Trick 3

Clip rewards to some fixed range [-1,1]

Not so important, but it helps

Using these 3 "tricks", DQN training became stable

No convergence guarantees!

Converges to a local optimum that is not far from the global optimum

## DQN Example: Playing Atari Games

- ¢ The input is the 8x8 image region about the current position of the snake.
- ¢ Q-network with 3 convolutional layers of size
    - 32x8x8;stride 4
    - 64x4x4;stride 4
- ¢ 64x3x3;stride 2
- ¢ The final two layers are fully connected layers with 512

Mnih et al. (2015). Human-level control through deep reinforcement learning

## DQN Example: Playing Atari Games

&amp;gt;= Humans

Poor sample efficiency

Each game learned from scratch

## DQN Example: Playing Atari Games

Better than human performance

May need excessive data ~ 10? samples (if 1s per sample, then &amp;gt;31 years!)

Still needs actions to be discrete - not feasible in many cases

SIT796 Reinforcement Learning

Dealing with continuous actions

Presented by: Thommen George Karimpanal School of Information Technology

## RL algorithm types

Three approaches to find RL policy:

    1. Value-based methods (everything covered so far, incl. DQN)

s'

(next state)

    1. Directly obtain policy (policy gradient methods)
    1. Actor-critic methods (combination of 1. and 2.)

## Policy Gradient Methods

- ¢ Several kinds:
- ° Finite Difference Policy Gradient
- ° Monte Carlo Policy Gradient
- ° Actor-Critic Policy Gradient
- ¢ Directly parameterize and learn the policy (als) = fol b(s, a)

Feature vector of state-action pair

- ° Can have several forms such as:

mals) x exp(O' (s, a))

No need to be related to the value function!

exp(lalv(s,

Temperature of soft-max

## What are Policy Gradient Methods?

- ° Before: Learn the values of actions and then select actions based on their estimated action-values. The policy was generated directly from the value function
- ° We want to learn a parameterised policy that can select actions without consulting a value function. The parameters of the policy are called policy weights
- ° A value function may be used to learn the policy weights but this is not required for action selection
- ° Policy gradient methods are methods for learning the policy weights using the gradient of some performance measure with respect to the policy weights
- ° Policy gradient methods seek to maximise performance and so the policy weights are updated using gradient ascent

## Policy-based Reinforcement Learning

- ¢ Search directly for the optimal policy m*
- ¢ Recall that the optimal policy is the policy that achieves maximum future return
    - ¢ Recall that the optimal policy is the policy that achieves maximum future return

SIT796 Reinforcement Learning

Policy Approximation

Presented by: Thommen George Karimpanal School of Information Technology

## Gradient Descent

- Optimizer for functions.
- Guaranteed to find optimum for convex functions.
- Non-convex = find /ocal optimum.
- Works for multi-variate functions.
- Need to compute matrix of partial derivatives ("Jacobian")
    1. Start with a random value of w (e.g. w = 12)
    1. Compute the gradient (derivative) of L(w) at point w = 12. (e.g. dL/dw = 6)
    1. Recompute w as:

w = w-A(dL(w) / dw)

## Policy Gradient: General Idea

Directly learn policy from objective function:

tT: trajectory of (s,a,r) pairs

Directly maximize J(@) : obtain the update equation:

Policy gradient theorem: It can be shown that:

Refer to textbook for derivation. . . It forms the basis for the policy gradient family of methods

## Policy Approximation

- ¢ Basic assumption: policy is differentiable w.r.t. 8 . Eg:

soft-max in action preferences

- ° Action preferences could also be linear:

- ¢ Forsome problems, it is simpler to learn the policy directly rather than learning the value functions, and extracting the policy from it later.
- ¢ If using learned value functions, the value function's weight vector is w
    - ¢ If using learned value functions, the value function's weight vector is w

SIT796 Reinforcement Learning

## REINFORCE

Presented by: Thommen George Karimpanal School of Information Technology

## REINFORCE

Using the policy gradient theorem, the update rule can be modified to be:

Iteratively updating 6 with the above update rule will lead to the optimal policy 7*

## REINFORCE Properties and Algorithm

- ° On-policy method based on SGD
- ° Uses the complete return from time t, which includes all future rewards until the end of the episode
- ° REINFORCE is thus a Monte Carlo algorithm and is only well-defined for the episodic case with all updates made in retrospect after the episode is completed

## REINFORCE: Monte-Carlo Policy-Gradient Control (episodic) for 7,

Input:

Algorithm parameter:

step size a &amp;gt; 0

Initialize policy parameter 6 € R® (e.g., to 0)

Loop forever (for each episode):

Generate an episode So, Ao, Ri,...,S7 \_ -1, Ar \_

Loop for each step of the episode t = 0,1,...,7'- 1:

ae RR,

## Actor-Critic Methods

Actor-critic methods are a fusion of policy gradient and value-based methods

Actor: deals with the policy (policy gradient-based) a,~7g

Critic: evaluates the actor's action (value-based) 6, = m4, +yV(Sst41) -V(s;)

6+ 1s updated to minimise 6; with the update rule:

6, 6,+ BO;

where f is a positive step size hyperparameter

Both actor and critic are updated using 6, to determine the optimal policy 7*

Modern approaches include several variants of the actor-critic algorithm.

## One-step Actor-Critic Update Rules

- ¢ On-policy method
- e The state-value function update rule is the TD(O) update rule
- ¢ The policy function update rule is shown below.

## One-step Actor-Critic Algorithm

## One-step Actor-Critic (episodic), for estimating 7

| s, ™) Input: a differentiable state-value function parameterization 6(s,w) Parameters: step sizes a? &gt; 0, a¥ &gt; 0 Initialize policy parameter 0 € R” and state-value weights w € R? (e.g., to 0) Loop forever (for each episode): Initialize S (first state of episode) Il Loop while S is not terminal (for each time step): Aw (-   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Ses!

## Modern Algorithms

- e¢ Trust Region Policy Optimisation
- e¢ - Proximal Policy Optimisation (Simplification of TRPO)
- ¢ Soft Actor-Critic (SAC)
- e DDPG (Deep deterministic policy gradient)
- ¢ and many more!

## Readings

## This lecture focused on introducing Deep Learning for RL.

- ¢ Future topics will expand on this topic by looking at particular methods in Deep RL.
- e Ensure you understand what was discussed here before doing the following topics

## For more detailed information see:

- ° https://www.ics.uci.edu/~dechter/courses/ics-295/fall2019/texts/An \_ Introduction \_ to Deep Reinforcement \_ Learning.pdf
- ° https://rail.eecs.berkeley.edu/deepricourse
- ¢ Other Readings:
- ° Playing Atari with Deep Reinforcement Learning (https://arxiv.org/abs/1312.5602)