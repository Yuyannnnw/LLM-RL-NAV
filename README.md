# LLM-RL-NAV

## What Has Been Done So Far
1. **Implemented DQN from Scratch:**  
   Developed a custom Deep Q-Network (DQN) implementation. Initial results are promising, though the stability of the algorithm still needs to be double-checked.
2. **Tested on Highway-Fast Environment:**  
   Utilized the [Highway Environment](https://highway-env.farama.org/) (fast version) with its default configuration to compare the performance of our custom DQN against the DQN implementation from Stable Baselines3.

## What Needs to Be Done
1. **Integrate and Tune LLM for Reward Adjustment:**  
   Load a Language Model (LLM) and tune it so that it takes the DQN’s action, reward, current state, and observation space as input and outputs an adjustment to the reward.
2. **Integrate Reward Shaping:**  
   Incorporate reward shaping into the custom DQN to improve training performance.
3. **Develop Observation Mapping Function:**  
   Create a function that maps raw observations to relative position vectors, and add noise to these vectors to simulate a decentralized, partially observable setting.

**Note:**  
Focus on single-agent scenarios for now. Once the system is tuned and provides the desired results, we will move on to implementing a multi-agent environment.
