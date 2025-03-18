# LLM-RL-NAV

## What Has Been Done So Far
1. **Implemented DQN from Scratch:**  
   - Developed a custom Deep Q-Network (DQN) implementation.
   - Initial results are promising, although further stability verification is required.

2. **Tested on Highway-Fast Environment:**  
   - Utilized the [Highway Environment](https://highway-env.farama.org/) (fast version) with its default configuration.
   - Compared the performance of our custom DQN against the DQN implementation from Stable Baselines3.
   - Configure decentralized environment by modifying the observation space, which result in a drastic decrease in the collision-free success rate.

3. **Basic LLM Prompt using Ollama:**  
   - Provided a simple example of integrating Ollama through Python (compatible with Mac, Linux, Windows, M1/M2/M3, AMD, and NVIDIA architectures).  
   - **Setup Instructions:**  
     a) Install Ollama from the official website.  
     b) Run one instance to start the service.

4. **Integrate LLM for Reward Adjustment:**  
   - Load a Language Model (LLM) and configure it to accept the DQN’s action, reward, current state, and observation space as input.
   - The LLM output adjustments to the reward to guide the learning process.
   
5. **Integrate Reward Shaping:**  
   - Incorporate reward shaping (adding a float term) into the custom DQN.

## What Needs to Be Done
1. **Develop Observation Mapping Function:**  
   - Create a function that transforms raw observations into relative position vectors.
   - Introduce noise to these vectors to simulate a decentralized, partially observable setting.

2. **Improve the Prompt to LLM:**  
   - The prompt to LLM should provide sufficient information to generate result.
   - Should interpret the observation space correctly.
   
3. **Tune LLM for Reward Adjustment:**  
   - Tune LLM for better adjustment.
   - Have to find a way to tune it.

4. **Improvement Collision-Free:**  
   - Considering methods such as approximate the intentions of other agents or introducing safety constrains.


**Note:**  
For now, focus on single-agent scenarios. Once the system is tuned and meets the desired performance, we will extend the implementation to a multi-agent environment.

## Future Timeline


