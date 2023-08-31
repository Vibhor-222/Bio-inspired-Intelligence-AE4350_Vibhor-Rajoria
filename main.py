import gymnasium as gym
import os
import matplotlib.pyplot as plt
import json
import time
import numpy as np
from ddqn import DDQN, DQN  # Import your custom agent classes
from utils import plot_learning_curve

# Set environment variable to prevent graphical output
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Import clear_output from IPython.display
from IPython.display import clear_output

# Define the frequency of learning
freq_learn = 4

def train_agent(n_episodes=2000):
    print("Training a DDQN agent on {} episodes.".format(n_episodes))
    env = gym.make("LunarLander-v2")  # Create LunarLander environment
    agent = DDQN(gamma=0.99, epsilon=1.0, epsilon_dec=0.995, lr=0.0001,
                         mem_size=200000, batch_size=128, epsilon_end=0.01)  # Create an agent
    
    scores = []  # To store episode scores
    eps_history = []  # To track epsilon values over episodes
    start = time.time()  # Record start time
    for i in range(n_episodes):
        done = False
        score = 0
        state = env.reset()[0]  # Reset environment and get initial state
        steps = 0
        while not done:
            action = agent.choose_action(state)  # Choose an action using the agent's policy
            new_state, reward, terminated, truncated, info = env.step(action)  # Take an action in the environment
            done = terminated or truncated  # Update done flag
            agent.save(state, action, reward, new_state, terminated)  # Save experience for learning
            state = new_state  # Update the state
            if steps > 0 and steps % freq_learn == 0:
                agent.learn()  # Perform learning every LEARN_EVERY steps
            steps += 1
            score += reward  # Accumulate rewards
        
        eps_history.append(agent.epsilon)  # Track epsilon value for this episode
        scores.append(score)  # Store the final episode score
        avg_score = np.mean(scores[max(0, i-100):(i+1)])  # Calculate the rolling average score
        
        if (i+1) % 10 == 0 and i > 0:
            # Report progress and expected time
            print('Episode number {} in {:.2f} min. Current and Average Score {:.2f}/{:.2f}]'.format((i+1),(((time.time() - start)/i)*n_episodes)/60,
                                                                                                                      score, avg_score))
           


    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(x, scores, eps_history)  # Plot the learning curve

    return agent  # Return the trained agent

agent = train_agent(n_episodes=4000)  # Train the agent using 4000 episodes
