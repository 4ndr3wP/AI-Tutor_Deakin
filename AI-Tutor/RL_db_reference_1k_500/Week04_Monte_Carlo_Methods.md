## Monte Carlo Methods

Monte-carlo sampling:

The term Monte Carlo is used for any algorithm that is significantly random in its operation Calculate 7 by randomly placing a dot and counting frequency of dots inside or outside the circle.

Area of circular portion: Ac = mr*/4

Area of square: A, = r?

## Monte Carlo Methods

Disadvantage of Dynamic Programming: requires a full and complete model of the MDP Monte Carlo (MC) methods only require experience.

In RL, MC methods simply need to sample: states, actions and rewards from the environment

Episode: one attempt at solving the task (episodes end based on terminal condition)

MC methods are incremental in an episode-by-episode sense.

In this sense they are similar to bandits - except stretched out over a number of steps (each acting like a separate bandit problem).

## Monte Carlo (Prediction)

MC can be used to learn the state value function for a given policy - as we have seen this is referred to as prediction

- Recall - The value of a state is the expected cumulative future discounted reward starting at that state (expected return).
- The simplest approach to learn this from experience is to average the observed returns after visiting the state.
- This is the notion that underlies MC.
- The following example considers the case where we only calculate for the first-visit.

## First-visit MC prediction, for estimating V ¥ v,

Input: a policy 7 to be evaluated

Initialize:

V(s) ER, arbitrarily, for all s € &amp;amp; Returns(s) + an empty list, for all s € &amp;amp;

Loop forever (for each episode):

Generate an

episode following

Geo

Loop for

each

Ge step

of

7G

episode,

Reza

Unless

S;

appears

7:

So,

Ao,

Ri,

$1,

Ai,

t

=

T-1,T-2,...,0:

in

So,

5i,...,S¢-1:

Append

G

to

Returns(S;)

V(St) average(Returns(S:))

4

Ro,...,S7-1,

Ar-1,

Rr

G=y*Otrp

Gay*r ty}

## First Visit Monte Carlo (example)

3 episodes, y = 1

V(s1)=(3+3+1)/3=2.33

V(s4)=(2+2+1)/3=1.66

## Every Visit Monte Carlo (example)

## Monte Carlo (Prediction)

## If we compare a MC backup diagram against that of DP, below:

    - DP updates a single V(s) for all actions based on the policy being followed.
- e Whereas MC updates the V(s) based only on the path followed by the policy.
- - Does not build upon other states estimates - no bootstrapping.

Importantly the computational cost is independent of how many states there are.

- ¢ Particularly useful if we only want to calculate a subset of states.
- ° Learn only from states that occur

## Example - Blackjack

## A card game where you play against the dealer

- ¢ Aimis to get a higher score than the dealer without going bust (&amp;gt; 21).
- ¢ You are both given two cards (one of the dealer's cards is initially hidden).
- ° - if you are dealt an ace and ten (21) and the dealer doesn't have an ace or a ten then you win immediately
- ¢ You can keep drawing more cards (called a hit) until you have as high as you want to risk.
- -If you draw too many and go over 21 you have lost
    - Once you have drawn all the cards you desire you finish (called stick)
- ¢ The dealer then reveals the hidden card
- ¢ The dealer must keep drawing a card until they are &amp;gt; 17.
- -If the dealer goes over 21 then you automatically win
- ¢ Once the dealer finishes drawing cards, if they arent bust
- -you win if your score is higher
- -You draw (keep your money) if your sums are the same
- -Lose if your sum is less
- ¢ - Note an Ace (can be 1 or 11) allows us to take a risk (when 11) even when we have a good score

The difficulty is knowing when to draw based on what you have and the one card you can see the dealer has.

## Example -Blackjack with MC Prediction

## Let's consider a policy that we want to test

## After 10,000 episodes

- ¢ - The policy sticks if the player has a sum of 20 or 21
    - Otherwise it always hits

To find the state value of this policy using MC prediction we can simulate many games using the policy

- -average the returns observed.

Why does the estimated value function jump up for the last two rows in the rear?

Why does it drop off for the whole last row on the left?

Why are the frontmost values higher in the upper diagrams than in the lower?

## After 500,000 episodes

Usable ace

No usable ace

SIT796 Reinforcement Learning

Monte Carlo Control

Presented by: Dr. Thommen George Karimpanal School of Information Technology

## Monte Carlo with Action Values

In DP we calculated state values because we can simply select the action that leads to the best next state.

- This is possible because we have a model of the environment - hence we can track what states come next and their value.
- However if we don't have a model we can not do this.
- If we can estimate the value of actions taken from a particular state (rather than just the value of the state), we can just choose the best actions based on the Qvalue.
- Often referred to as state-action values and denoted Q,,(s, a)
- MC, with enough experience can estimate Q*

In reality we do this by visiting state-action pairs multiple times to estimate the value.

- Therefore, MC finds the average of the returns from that state-action pair.
- Some state-action pairs may never be visited. Eg if 7 never uses an action we will never learn a value for it. This fits with the general problem we discussed with multi-armed bandits - maintaining exploration
- Exploring starts - starting at a random state-action pair, where there is a probability of starting in any state-action pair
- Use a stochastic policy where all state-actions have a probability of being selected.

## Monte Carlo (Control)

## Recall, Control is a search for a policy

- ¢ whereas prediction was aiming to find the value - given a policy.

## Recall the GPI approach in DP

- ¢ Recall, GPI maintains both a policy and an approximate value function.
- -The value function is updated to approach the value function of the current policy
- - The policy is improved based on the current value function
    - Hence these work against each other to some degree

## In MC we can follow a similar approach

- ¢ - Again we can alternate between policy evaluation and policy iteration. E IE I E I E To &amp;gt; Any 7 11 In, &amp;gt; hz 0 Fe Fx
- ¢ Except now we apply the evaluation based on the the complete trajectory.
- ¢ Policy iteration is performed by selecting the greedy policy with respect to the current value function
- ° This is also applied on the state-action values instead of state values and hence do not require a model m(s) = arg max q(s, a)

## Monte Carlo (Control): exploring starts (ES)

## If we assume exploring starts and an infinite number of iterations

- ¢ Then MC is guaranteed to converge using the following algorithm

## Monte Carlo ES (Exploring Starts), for estimating 7 ~ 7,.

Initialize: m™(s) € A(s) (arbitrarily), for all s € § Q(s,a) € R (arbitrarily), for all s € 8, a € A(s) Returns(s,a) &amp;lt; empty list, for all s € 8, a € A(s) Loop forever (for each episode): Choose So € 8, Ap € A(So) randomly such that all pairs have probability &amp;gt; 0 Generate an episode from So, Ao, following 7: So, Ao, Ri,...,S7-1, Ar \_ -1, Rr Goo Loop for each step of episode, t = T-1,T-2,...,0: Ge 7G + Riza Unless the pair S;, A; appears in So, Ao, $1, A1..., S¢-1, At \_ 1: Append G to Returns(S:, At) Q(S;, Ar) &amp;lt; average(Returns(S;, A;)) m(S;) &amp;lt; argmax, Q(S;, a)

## Example -Solve Blackjack with MC Control

## Monte Carlo (Control): on-Poticy without Es

## Exploring starts is a form of On-Policy Control

- ¢ - On-Policy - evaluates and improves the policy being followed when making decisions.

## However, ES is fundamentally an unlikely assumption for most tasks.

- ¢ To avoid ES we need a way of visiting every state-action pair
    - Can be done using a stochastic policy. That is (als) &amp;gt; 0, for alls € S,a € A(s)
- ¢ - Then overtime shift this stochastic policy more towards a deterministic policy
- ¢ In our discussion of multi-armed bandits we discussed some ways of doing this:
- - €--greedy, Upper-Confidence-Bound, soft-max.
- - There are many others eg Thomson Sampling, we have not discussed.

## If we assume € - greedy we say this is a stochastic policy because:

- . The greedy action will have 1 - € + */4(,)) probability of being selected.
    - . The greedy action will have 1 - € + */4(,)) probability of being selected.
- . Therefore, (als) = */ ,
- . Any of the above € - soft policies also provide convergence guarantees as stochastic policies.

Thus, total probability of best action:

## Monte Carlo (Control): on-Poticy without Es

## MC On-Policy Control is still GPI

    - Our eé-soft policy should move towards a greedy policy but as we can not actually achieve infinite trials we can not actually reach a fully greedy policy.
    - Hence our € - soft policy should never actually reach greedy

## On-policy first-visit MC control (for ¢-soft policies), estimates 7 ~ 7

Algorithm parameter: small ¢ &amp;gt; 0

Initialize:

am + an arbitrary e-soft policy Q(s, a) € R (arbitrarily), for all s € 8, a € A(s)

Returns(s,a) &amp;lt; empty list, for all s € 8, a € A(s)

Repeat forever (for each episode):

Generate an episode following 7: So, Ao, Ri,...,S7-1, Ar-1, Rr Geo

Loop for each step of episode, t = T-1,T-2,...,0:

Ge 1G + Riza

Unless the pair S;, A; appears in Sp, Ag, $1, Ay... , S¢-1,

At \_

Append G to Returns(S;,

Q(St, At) + average(Returns(S;, At))

A* + argmax, Q(S;, a)

For all a € A(S;):

| A(S,)   |
|---------|

## On/Off-Policy Learning

On-Policy evaluates and improves the policy being followed.

- ¢ The issue is that the agent must perform non-optimal behaviour to learn about all state-actions
- ¢ - This means the state-action values being learnt is based on the current policy (which may not be optimal)
- -Hence, may learn significantly inferior values.
- May think a state value is not-optimal (when it is) because the later actions were exploratory.

Behavior Policy

Off-Policy instead evaluates and improves the optimal (target) policy instead of the one followed. Allows the agent to explore without damaging the state-action values being learnt.

## To do this you effectively maintain two policies

The current optimal policy being learnt - called the target policy

A behaviour policy that generates the decisions and can be more exploratory

In this way, the state-action values learn from data "off" the target policy, hence the approach is called off-policy learning

In this unit we will consider both on-policy and off-policy approaches to different algorithms On-policy is generally simpler and is considered first.

You can think of On-policy as simply having the same behaviour and target policies - hence is a special case of off-policy Generally, on-policy learns faster, but off-policy is often more powerful

Target Policy

## Off-Policy Monte Carlo Prediction

Let's suppose we have collected months of data about how your household uses its solar power cell.

- ¢ This data represents a policy - the behaviour followed by your house (Behaviour Policy)

Let's say we want to learn the optimal policy so we can advise you where to save power

    - How do we do this when we only have data for your actual behaviour and not the optimal behaviour?

Hence, we want to learn a policy, v,, or q,,, but only have a policy b, where b # 7

## This can be done using off-policy learning.

- ° However, it does require the assumption of coverage to be met:
- ° E.g. we require every action to be taken under z must also be taken at least sometimes in b.
- - Hence, z(als) &amp;gt; 0 &amp;gt; b(als) &amp;gt; 0
- ¢ Therefore, our behaviour policy, b should be at least to some degree stochastic.

If our problem meets the assumption of coverage then we can learn the target policy z from our observations of b using importance sampling.

## Importance Sampling

Importance Sampling allows us to estimate expected values under one distribution given samples from a separate distribution.

- In off-policy learning we weight the returns observed based on the relative probability of their future trajectories occurring under both the target and behaviour policies.
- ¢ - This is called the importance-sampling ratio
- Given the start state S;, we want to calculate the probability of the future path, A;, S¢44, A¢4i, ---, Sp,occurring under any policy 7 using

Where p is the state transition probability

Therefore, the relative probability or importance-sampling ratio is calculated as:

; . Note: while the transition probabilities of the MDP are unknown, because they are the same for the numerator and denominator they cancel

## Ordinary Importance Sampling

Recall, we aim to find the state or state-action values (expected returns) of the target policy.

- ¢ Instead, we use the importance-sampling ratio to transform the returns to the correct expected value.
    - ¢ Instead, we use the importance-sampling ratio to transform the returns to the correct expected value.

- ¢ E.g. simply multiply our observed return by the importance-sampling ratio.
    - - To calculate v,,(s) we use:
- ¢ - This equation has effectively joined each episode end to end (time steps continue across episode boundaries).
- ¢ - This simple average of samplings is called an ordinary importance sampling

## Weighted Importance Sampling

A powerful alternative is a weighted importance sampling using a weighted average.

- ¢ - If the importance of the sample (denominator) is zero then v,(s) = 0.
- ¢ In practice, the weighted importance sampling has lower variance and is preferred.
- - The ordinary importance sampling is easier to apply when we look at function approximation

SIT796 Reinforcement Learning

Monte Carlo: Off-policy Estimation

Presented by: Dr. Thommen George Karimpanal School of Information Technology

## Incremental Implementation

Recall, when tracking our Q-value over time in our multi-armed bandit (Topic 4), we could maintain an incremental average.

    - Therefore, we only needed to store the current Q,, and the value of n

We can perform a similar approach when using ordinary importance sampling except use the scaled return in place of the reward. E.g.

- ° Where W; = p,-r(t;)-1, €-8the weighting of a particular sample.

When using the weighted importance sampling however we need to perform a weighted average of the returns.

- ¢ Now we need to also track the cumulative sum C,,, where C,4, = Cy, + Wy44, of the weights given and use this instead of n. E.g.

- 23 ° Where W, is the weighting of the sample in the n episode and C, is the cumulative weights up to and including the n™ episode.

## Off-Policy Monte Carlo Prediction

## First-visit MC prediction, for estimating V © vu,

Input: a policy 7 to be evaluated

Initialize:

V(s)

ER, arbitrarily, for all s € &amp;amp;

Returns(s) + an empty list, for all s € 8

Loop forever (for each episode):

Generate an episode following 7: So, Ao, Ri, $1, Ai, Ro,...,Sr-1, Ar-1, Rr

Geo

Loop for each step of episode, t = T-1,T-2,...,0:

Ge 7G + Resa

Unless S; appears in So, S1,...,S¢-1:

Append G to Returns(S;)

V (St) + average(Returns(S'))

## Off-policy MC prediction (policy evaluation) for estimating Q ~ q,

Input: an arbitrary target policy 7

Initialize, for all s € 8, a € A(s):

Q(s,a) € R (arbitrarily)

C(s,a) +0

Loop forever (for each episode):

b + any policy with coverage of

Generate an episode following b: So, Ao, Ri,...,Sr-1, Ar-1, Rr

Gro

Wel

Loop for each step of episode, t = T-1,T-2,...,0, while W 40:

7 Ge 7G + Ri41 O(St, Ar) - C(St, Ar) + W Q(St, At) - Q(St, At) + ageay IG - Q(St, At)]

## Off-Policy Monte Carlo Control

Recall, On-policy Control evaluates and improves the policy being followed when making decisions.

In Off-Policy the policy used to generate behaviour may be different to that is being evaluated as the target.

- ¢ Advantage of this approach is target policy can be deterministic (greedy) while the behaviour policy can sample other actions.

## Off-Policy MC Control can use one of the two methods discussed for Off-policy MC Prediction

- ¢ As we require the behaviour policy to cover all possible state-actions that might be required by the target policy, we must have a nonzero probability of selecting all actions ,

## Off-policy MC control, for estimating

- - This is required to insure the assumption of coverage.
- . . . - Todo this we must use an € - soft behaviour policy

\_ nies, for all 6 E Gi, @ G AUG)

Q(s, a) € R (arbitrarily) C(s,a) +0 m(s) + argmax S,a with ties broken consistent] (s) gmax, Q(s,a) ( ¥) Loop forever (for each episode): b + any soft policy Generate an episode using b: So, Ao, Ri,...,Sr-1, Ar-1, Rr G+to W+l1 Loop for each step of episode, t = T-1,T-2,...,0: Ge 7G ae Revi C(St, At) = C(S¢, At) +W Q(St, At) - Q(St, At) + agray [G - Q(St, At)]

a(S) + argmax, Q(S;,a) (with ties broken consistently)

If A; A 7(S;) then exit inner Loop (proceed to next episode)

## Readings

This lecture focused on Monte Carlo methods for prediction and control.

- ¢ Same capabilities as DP without a transition model!
- °¢ However, it needs full trajectories/episodes - TD learning will solve this issue.

For more detailed information see Sutton and Barto (2018) Reinforcement Learning: An Introduction

- °* Chapter 5: Monte Carlo Methods
- ° http://incompleteideas.net/book/RLbook2020. pdf