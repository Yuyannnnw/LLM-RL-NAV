# LLM-RL-NAV

## What Has Been Done So Far
1. **Implemented DQN from Scratch:**  
   - Developed a custom Deep Q-Network (DQN) implementation.
   - Initial results are promising, although further stability verification is required.

2. **Tested on Highway-Fast Environment:**  
   - Utilized the [Highway Environment](https://highway-env.farama.org/) (fast version) with its default configuration.
   - Compared the performance of our custom DQN against the DQN implementation from Stable Baselines3.

3. **Basic LLM Prompt using Ollama:**  
   - Provided a simple example of integrating Ollama through Python (compatible with Mac, Linux, Windows, M1/M2/M3, AMD, and NVIDIA architectures).  
   - **Setup Instructions:**  
     a) Install Ollama from the official website.  
     b) Run one instance to start the service.

## What Needs to Be Done
1. **Integrate and Tune LLM for Reward Adjustment:**  
   - Load a Language Model (LLM) and configure it to accept the DQN’s action, reward, current state, and observation space as input.
   - The LLM should output adjustments to the reward to guide the learning process.

2. **Integrate Reward Shaping:**  
   - Incorporate reward shaping into the custom DQN to enhance training performance.

3. **Develop Observation Mapping Function:**  
   - Create a function that transforms raw observations into relative position vectors.
   - Introduce noise to these vectors to simulate a decentralized, partially observable setting.

**Note:**  
For now, focus on single-agent scenarios. Once the system is tuned and meets the desired performance, we will extend the implementation to a multi-agent environment.
