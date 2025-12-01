# GRPO

GRPO is a Reinforecement Learning algorithm for training LLMs. It stands for Group Relative Policy Optimization.

GRPO generates multiple answers and instead of estimating value & using critic (like PPO), it computes relative performance.

For each prompt:

- Sample G different outputs from the current model.
- Compute rewards for each output.
- Compute the average reward in the group.
- For each output:
  
      Advantage ≈ reward – group_average
  
So:
- Outputs better than group average => positive advantage => increase probability.
- Outputs worse than group average => negative advantage => decrease probability.

We learn only by contrasting answers inside group. This is why it is called Group Relative optimization. Group itself acts as 'judge'.

## Setup

- Agent / Policy: GPT-Neo-125M model.

- Environment: your analogy prompts.

- State: the prompt string.

- Action: the next one word the model outputs.

- Reward:

  - 1 if the word is in the correct target list (e.g., "cunning" for fox)

  - 0 otherwise.

- Objective: change the policy so that correct words are more likely.

## Training

For each batch of prompts:

- For each prompt, generate G = 4 possible answers (actions).

- For each answer, compute reward (1 or 0).

- Compute group mean reward for each prompt.

- Convert reward-to-advantage:

  - If answer’s reward is above group mean → positive advantage.

  - If below → negative.

- Compute a loss that:

  - increases probability of answers with positive advantage,

  - decreases probability of answers with negative advantage.

- Add a KL penalty to keep policy close to the original model.

- Update model parameters (policy).

Over many steps:

- The model learns to answer:

  - fox → “cunning”

  - fish → “swim”

  - son → “child”

- Result: baseline reward 0.15 → GRPO reward 0.8.
