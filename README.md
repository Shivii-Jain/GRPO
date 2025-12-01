# GRPO

GRPO is a Reinforecement Learning algorithm for training LLMs. It stands for Group Relative Policy Optimization.

GRPO generates multiple answers and instead of estimating value, it computes relative performance.

- Above average answers → pushed stronger

- Below average answers → suppressed

We learn only by contrasting answers inside group. This is why it is called Group Relative optimization.
