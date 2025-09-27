## Approach so far

## Tabular approach:

- ¢ Single entry in a table of values
- ¢ States and actions have discrete representations
    - Working set of algorithms
    - Convergence proofs

| State   | S,   | ay       | Q(s,a,)   | a2        | Q(s;az)   | a3   | Q(s; a3)   |
|---------|------|----------|-----------|-----------|-----------|------|------------|
|         | Sy   |          | Q(Sp.a,)  |           | Ql(S2a2)  |      | Q(s, a3)   |
| Action  | S3   |          | Q(s3.a;)  |           | Q(S3a2)   |      | Q(s3 a3)   |
|         | Sa   | Q(S4,a,) | Q(S4,a2)  | Q(S,4,a3) |           |      |            |

## Limitations of Tabular Approach

## Function Approximation

## Let's first consider state value prediction v,, with function approximation.

- ¢ Instead of representing each state in a table of values, we represent it in a parameterised functional form using a weight vector w € R@
- ¢ Hence, we write the value for the approximate value of state s as given the weight vector w: B(s,w) = v,(s)
- - Hence changing the value of one weight changes the value of many weights
    - - Hence changing the value of one weight changes the value of many weights
- - Problem now is to find a w* that best approximates v,,(s) (or Q,-(s, a))

## Examples of what @ might be

Could be a linear function of features in the state where w is the weight of each feature

Could be non-linear function of features in the state computed over an Artificial Neural Network

Could be function computed by a decision tree where wis all the numbers defining the split points of the tree

## Linear Methods

A special case of function approximation commonly used Linear function approximation.

- ¢ That is our approximation function 0(-, w) is linear function over the weight vector w.
    - Now we can represent our state-value function using the inner product of w and x(s)
        - Now we can represent our state-value function using the inner product of w and x(s)

| x(s)   | =   | w   |
|--------|-----|-----|

    - Here x(s) is referred to as the feature vector for the state s.

It worth noting that the new notation can still represent the tabular approaches seen so far.

For instance, represent the state as a vector of the table entries. The feature representing the location of the agent is set to 1 and all others to 0.

x(s) = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]

w = [0.0,0.1,0.2,0.4,0.1,0.3,0.4,0.6,0.2,0.4,0.6,0.8,0.4,0.6,0.8,1.0]

## State Aggregation

Probably the simplest approach for function approximation is State Aggregation.

- ¢ The aim is simply to group similar states together and treat them as a single state
- ¢ The grouped states are represented with a single value within the weight vector

## For example

Image we have a grid world with 10,000 grids in two dimensions = 100 million discrete states.

Recall the more states then the more values to learn and less often we get to visit them relative to all the states

Now if we know an action may move us between one step up to 100 grids during each time step then we may group 50 states of each dimension as a single state.

This will reduce our states to 200X200 = 40,000 states

## This is 1 state now

Faster learning but each group of 50 states will each be assigned a single common value.

SIT796 Reinforcement Learning

Stochastic Gradient MC

Presented by: Thommen George Karimpanal School of Information Technology

## Stochastic Gradient MC

In Gradient Descent our aim is to reduce the error of all our examples.

- However, this is not useful in RL because we do not have all the examples when learning online through interaction.

Stochastic Gradient Descent (SGD) allows us to use a single example by adjusting our weight vector w in the direction of the estimated error, governed by a small factor a.

- To do this we must be able to identify the direction of the error, hence our function 0(s, w) must be differentiable.
- - We find the slope of a function by finding its partial derivative in this case with respect to w

## Input:

    - Now when calculating an update we move the weight vector towards a : . . . smaller error by moving a small amount in the direction of the error.

- ¢ Inthe linear case this reduces to:

E.g. the gradient for the example on the previous slide is

VO(St, Ww) = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]

the policy 1 to be evaluate 3 3 P a. d A differentiable function 0: 5xR° &amp;gt;

Algorithm Parameter:

. Step size a € (0,1]

Initialise:

Loop forever (for each episode):

a weR arbitrarily eg.w = 0

Generate an episode based on Tt: So, Ao, Ry, $4, A1, Ro, .--, Sp-1, AT-1, Rr Loop for each step of the episode t = 0,1,...,T - 1:

w= w+ alG; - 0(S;, w)]VO(S;, w)

R

## Semi-Gradient TD(0)

When using applying these ideas to bootstrapping methods (e.g. TD) then we are basing updates on an estimate rather than the true value.

Semi-gradient methods need to replace the true target with an estimate of the target

Why semi-gradient?

When we update w, we take into account the change in estimate, but don't take into account the change in the target (which also depends on w)

So the gradient computed only represents part of the true gradient

## Semi-Gradient TD(A)

Recall we were able to use eligibility traces to define an algorithm that sat between MC and TD(0).

- We obviously also wish to have this ability when using function approximation.
- application to continuing problems. This will allow us to update the weight vector each iteration, balance computation throughout instead of just at the end, and allow its

| z   | =   | w   |
|-----|-----|-----|

Z \_ 4 = 0

0 &amp;lt;t&amp;lt; T

Now we can update the weight vector proportionally to the trace vector

Where the TD-error is calculated the same as TD (0) Ot = Rea + YO(Sta1, We) - (Sp, We)

Instead of having to iterate through each e(s) as we did in the tabular case, we simply update the trace vector in a single operation

Input:

The policy mt to be evaluate A differentiable function 0:SxR¢ &amp;gt; R

Algorithm Parameter:

Step size a € (0,1]

Trace decay rate A € [0,1]

Initialise:

w € R@ arbitrarily e.g. w = 0

Loop forever (for each episode):

Initialize S

Reset z = 0

Loop for each step of the episode until S € S(Terminal)

; A, R, w)

Take action observe S'

Z&amp;lt; yAz + Vi(S,

6&amp;lt;-R+y0(S',w) - 0(S,w)

w=w+e a6z

SoS'

SIT796 Reinforcement Learning

Function approximation with Control

Presented by: Thommen George Karimpanal School of Information Technology

## Semi-Gradient Control

Now that we have a method of prediction, we can investigate a method of control.

- ¢ Now we want to approximate the action-value function G = q,.
- ¢ That is, we will represent the parameterized functional form using a weight vector w.

Hence, One-step Sarsa with function approximation is defined with the update rule

Control, however, is not just mapping the state to our vector w but also our actions.

For discrete actions this is perfectly fine - use a matrix where each column represents one of the possible actions.

Can also use a function approximation, such as action aggregation approach on the actions

However, every action increases the dimensionality of our value function

Neither of these are always suitable - Sometimes we want to have precise actions

E.g. The angle we turn the steering wheel on a car must be very precise to avoid an accident

This type of continuous action control is very much still an open question.

For now, we will assume there is a manageable discrete set of actions to select from

## Semi-Gradient Sarsa(0)

w=wtalR+yG(S, A, w)]Vq(S, A, w), special case for terminal state can't include future state

Input: A differentiable state-action value function 9: Sx AxR?¢ &amp;gt; R A policy zz if predicting or q,, if estimating (e.g. using e - greedy) Algorithm Parameter: Step size a € (0,1] Initialise: w € R@ arbitrarily e.g. w = 0 Loop forever (for each episode): S,A &amp;lt; Initial state and action of episode (e.g. using € - greedy) Loop for each step of the episode until S € S(Terminal) . Take action A, observe R, S" If S' © Sterminal then: else: Choose A' as a function of G(S',-, w) (e.g. using € - greedy) w=wtal[R + yG(S', A', w) - G(S,4, w)]VG(S, A, w)

S&amp;lt;S'

A&amp;lt;A'

## Sarsa(A) with Linear Function Approximation

Sarsa can also be extended to use eligibility traces similar to TD (A) - Note: This assumes binary features

```
Input: A policy z: if predicting or q,, if estimating (e.g. using e - greedy) A feature function F(s, a) returning the set of active feature indices for s,a Algorithm Parameter: Step size a € (0,1] Trace decay rate A € [0,1] Initialise: w € R¢ arbitrarily e.g. w = 0 Loop for each episode: Initialise S Choose A~z(|S) initial action of episode (e.g. using € - greedy) Reset z = 0 Loop for each step of the episode until S' is a terminal state: Take action A, observe R, S' 6-RDier(s,a) Wi for alli € F(S, A): zZ<-2Z+10rz« 1 depending on using accumulating or replacing traces If S' € Sterminal then: wewe aoz else: Choose A'~z:(|S') action (e.g. using € - greedy) 66+ Y Lier (s',A") Wi wewe a6z Z<-yAz SS'; ayy
```

## Semi-Gradient Off-Policy Control

There is a function approximation version of Watkins's Q-Learning that has been very popular.

- ¢ Below is a binary features version of the approach which aligns with the Sarsa implementation.

## The Deadly Triad

Watkins' Q-Learning with function approximation however has been found to be unstable and does not always converge.

"the danger of instability and divergence arises whenever we combine all of the following three elements, making up what we call the deadly triad."

- ¢ Function approximation
- ¢ Bootstrapping
- ¢ Off-Policy training

Can we give up one of them? Presence of any 2 appears to be manageable (leads to stable learning)

Function approximation - Can not be replaced in large state spaces without introducing the Curse-of-Dimensionality

Bootstrapping - Is possible but at the cost of significantly more computation and loss of efficiency with increased memory costs

Off-Policy training - we can just use on-policy methods instead and often that is good. However, there are many use cases where we want to learn multiple policies simultaneously in parallel.

## Stable Off-policy Methods

There are a number of approaches recently that provide stable off-poly methods with function approximation.

- For instance, one approach might be to select off-policy behaviours that are close to the target policy can be effective

## Gradient-TD methods.

- ¢ - Aim to minimize the Projected Bellman Error instead of reducing the TD-error
- ¢ They achieve this but effectively double the computational complexity

## Emphatic-TD methods.

- These methods rewrite the state transitions using importance sampling s.t. they are appropriate for learning the target policy
- - It does this while using the behaviour policy distributions

## These methods however are not explored in these classes.

- ¢ They can be explored for your major research task

SIT796 Reinforcement Learning

Feature Construction

Presented by: Thommen George Karimpanal School of Information Technology

## Function Construction

For function approximation, we need to convert the state-value function to feature vectors.

This can be done using any of the following:

- Polynomial features
- Fourier basis
- Coarse coding
- Tile coding
- Sparse coding
- Dictionary Learning

| =   | w   |
|-----|-----|

## Polynomial Features

In a polynomial feature, x(s) is a polynomial basis, that is, for a set of states s = {51, Sz, 53, ..., Sx}, the polynomial features can be written as

where cj ; is a non-negative integer such that c;; = {0,1, 2,...,n}

Thus, we have

and x,(s) is a n-order polynomial basis in a k-dimensional space spanned by (n + 1)* different features

## Polynomial Feature Example

There are several kinds of polynomial bases, such as:

- e Lagrange
- ¢ Newton
    - Orthogonal polynomials
    - Chebyshev

The Newton basis functions are;

Which gives, in the third order case, the following

SIT796 Reinforcement Learning

Fourier Basis

Presented by: Thommen George Karimpanal School of Information Technology

## Fourier Transform

- ¢ Polynomials are not the best -unstable and not very physically meaningful.
    - Easier to talk about "signals" in terms of its "frequencies" (how fast/often signals change, etc).
- ¢ Any periodic function can be rewritten as a weighted sum of Sines and Cosines of different frequencies (Jean Baptiste Joseph Fourier, 1807).

The aim is then to understand the frequency @ of our signal.

So, let's reparametrize the signal by @ instead of x:

For every w from 0 to inf, F(@) holds the amplitude A and phase ¢ of the corresponding sine

## Fourier Transform

Represent the signal as an infinite weighted sum of an infinite number of sinusoids

Note:

Arbitrary function

©

--}&amp;gt;

Single Analytic Expression

Spatial Domain (x)

--&amp;gt;

Frequency Domain (u)

(Frequency Spectrum F(u))

Inverse Fourier Transform (IFT)

## Fourier Transform

This means that F can encode both using the imaginary numbers.

Consider

The amplitude is given by

and the phase is

Square Wave with 50% duty cycle Fourier transform

## Fourier Features

In a Fourier basis, the set of states s = [S1, Sz, S3,..., sl? can be expressed using a cosine in the following way x;(s) = cos(ms'c*)

Thus, we have:

SIT796 Reinforcement Learning

Coding Methods

Presented by: Thommen George Karimpanal School of Information Technology

## Coarse Coding

    - Coarse coding is used for continuous spaces, but can also be used for state-spaces that are binary or to further encode other features.
- ¢ ach ball or sphere is usually called a receptive field.
- ¢ Features with large receptive fields give broad generalization, but might yield very coarse approximations
- ¢ The trade-off is often a question to be solved for their implementation:
- -Too coarse may not be discriminative enough
- -Too large and the complexity may increase too much

Narrow generalization

Broad generalization

Asymmetric generalization

The type of receptive field affects the nature of generalisation

## Coarse Coding

approx-

## Tile Coding

- Makes use of tiles and tilings
- Tiles are elements of tilings
- If only 1 tiling is used - state aggregation
- Many tilings are used to obtain the ability to represent the state space as finely or coarsely as required (Generalisation)
- Uses binary features
- Tilings are offset from each other uniformly in each dimension

## Tile Coding

- ¢ Tilings are offset from each other : : : : uniformly in each dimension
- . ¢ Diagonal elements become too prominent
    - To fix this, the offset can be done unsymmetrically

## Readings

## This lecture focused on exploring how we can use function approximation in RL.

- ¢ Future topics will expand on this topic by looking at Deep RL.
- e Ensure you understand what was discussed here before doing the following topics

For more detailed information see Sutton and Barto (2018) Reinforcement Learning: An Introduction. Several illustrations here were borrowed from the book.

- ¢ Chapter 9.5: Feature Construction for Linear Methods
- . . ° http://incompleteideas.net/book/RLbook2020.pdf

## Other readings:

- ¢ Isabelle Guyon, Steve Gunn, Masoud Nikravesh, and Lofti Zadeh (Eds.), "Feature Extraction: Foundations and Applications", Springer, 2006.
- ° http://www.causality.inf.ethz.ch/ciml/FeatureExtractionManuscript.pdf
- ° https://sites.fas.harvard.edu/~cs278/papers/ksvd.pdf