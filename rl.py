import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy

# ------------------------
# Policy Network Definition
# ------------------------


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

# ------------------------
# Environment Setup
# ------------------------


def make_task(gravity):
    """
    Create a CartPole environment with modified gravity.
    """
    env = gym.make('CartPole-v1')
    env.env.gravity = gravity
    return env

# ------------------------
# Inner Loop (Task Adaptation)
# ------------------------


def inner_loop(env, policy, optimizer, steps=100):
    """
    Perform adaptation for a single task (environment).
    """
    policy.train()
    for _ in range(steps):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action_probs = policy(obs_tensor)
            action = torch.multinomial(action_probs, 1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            loss = -torch.log(action_probs[action]) * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# ------------------------
# Meta-Training Loop
# ------------------------


def meta_training(tasks, policy, meta_optimizer, inner_steps=100, meta_epochs=50):
    """
    Meta-training across multiple tasks.
    """
    for epoch in range(meta_epochs):
        meta_optimizer.zero_grad()
        meta_loss = 0

        # Perform meta-learning across tasks
        for env in tasks:
            # Clone the policy for inner loop adaptation
            adapted_policy = deepcopy(policy)
            inner_optimizer = optim.Adam(adapted_policy.parameters(), lr=0.01)

            # Inner loop adaptation
            inner_loop(env, adapted_policy, inner_optimizer, steps=inner_steps)

            # Evaluate loss post-adaptation
            obs, _ = env.reset()
            done = False
            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                action_probs = adapted_policy(obs_tensor)
                action = torch.multinomial(action_probs, 1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                meta_loss += -torch.log(action_probs[action]) * reward

        # Meta-update
        meta_loss /= len(tasks)
        meta_loss.backward()
        meta_optimizer.step()
        print(
            f"Epoch [{epoch+1}/{meta_epochs}], Meta Loss: {meta_loss.item():.4f}")

# ------------------------
# Meta-Testing Loop
# ------------------------


def meta_testing(task, policy, inner_steps=100):
    """
    Test the meta-learned policy on a new task.
    """
    test_policy = deepcopy(policy)
    optimizer = optim.Adam(test_policy.parameters(), lr=0.01)

    print("\n--- Meta Testing ---")
    inner_loop(task, test_policy, optimizer, steps=inner_steps)
    obs = task.reset()
    done = False
    total_reward = 0
    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action_probs = test_policy(obs_tensor)
            action = torch.multinomial(action_probs, 1).item()
        obs, reward, done, _ = task.step(action)
        total_reward += reward
    print(f"Total Reward on Test Task: {total_reward}")


# ------------------------
# Main Execution
# ------------------------
if __name__ == "__main__":
    # Hyperparameters
    INPUT_SIZE = 4  # CartPole state space
    OUTPUT_SIZE = 2  # CartPole action space
    META_LR = 0.01
    META_EPOCHS = 50
    INNER_STEPS = 100

    # Initialize policy network
    meta_policy = PolicyNetwork(INPUT_SIZE, OUTPUT_SIZE)
    meta_optimizer = optim.Adam(meta_policy.parameters(), lr=META_LR)

    # Define tasks with varying gravity
    task_gravities = [9.8, 15.0, 20.0]
    tasks = [make_task(g) for g in task_gravities]

    # Meta-training
    print("=== Starting Meta-Training ===")
    meta_training(tasks, meta_policy, meta_optimizer,
                  inner_steps=INNER_STEPS, meta_epochs=META_EPOCHS)

    # Meta-testing on a new task
    test_task = make_task(gravity=12.0)  # Unseen gravity
    meta_testing(test_task, meta_policy, inner_steps=INNER_STEPS)
