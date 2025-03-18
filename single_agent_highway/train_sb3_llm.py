import gymnasium as gym
import highway_env
import ollama
import json
from stable_baselines3 import DQN

def call_llm_for_shaping(prev_obs, next_obs, action, base_reward):
    """
    Queries an LLM to adjust the reward based on previous state, next state, action, and base reward.
    """
    print(prev_obs)
    print(next_obs)
    print(action)
    print(base_reward)

    prompt = f"""
    You are a reinforcement learning assistant helping to fine-tune rewards of an autonomous vehicle.
    Only response a numerical float. Do not give me any other information.
    The value should be in range -5 to 5, where higher value refers to encourage human-like action.
    Action space is discrete where 0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'.   
    Given the following information:
    - Previous Observation: {prev_obs}
    - Action Taken: {action}
    - New Observation: {next_obs}
    - Base Reward: {base_reward}
    
    Adjust the reward to improve learning.  
    """
    print("Prompt to LLM:")
    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}]
    )
    print(f"LLM Response: {response}")
    
    try:
        # Extract numerical adjustment from LLM response
        shaping_value = float(response['message']['content'].strip())
    except ValueError:
        shaping_value = 0.0  # Default to 0 if parsing fails

    print(shaping_value)
    
    return shaping_value

class LLMShapingWrapper(gym.Wrapper):
    """
    A wrapper that applies LLM-based reward shaping.
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        print("Environment Reset:", obs)
        self.prev_obs = obs
        return obs, info

    def step(self, action):
        next_obs, base_reward, done, truncated, info = self.env.step(action)
        print(f"Action taken: {action}, Reward before shaping: {base_reward}")  # Debugging print
        
        # Query LLM for reward adjustment
        shaping_term = call_llm_for_shaping(self.prev_obs, next_obs, action, base_reward)
        print(f"Shaping term: {shaping_term}")  # Debugging print
        
        # Adjusted reward
        total_reward = base_reward + shaping_term
        self.prev_obs = next_obs
        
        return next_obs, total_reward, done, truncated, info

def main():
    # Load the base environment
    env = gym.make("highway-fast-v0")
    
    # Wrap with LLM-based reward shaping
    env = LLMShapingWrapper(env)

    print("Successfully wrap with LLM-based reward shaping.")

    # Configure and train DQN
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=16,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log="models/highway_dqn/")

    # Train with 20,000 timesteps
    model.learn(total_timesteps=1e4)

    # Save the trained model
    model.save("models/highway_dqn_sb3/model")
    print("Training complete and model saved.")

if __name__ == "__main__":
    # Ensure LLM model is available
    #ollama.pull(model='llama3.2')
    ollama.pull(model='deepseek-r1')
    print("Successfully pull the LLM.")
    main()
