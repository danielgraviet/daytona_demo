"""
cartpole_task.py
----------------
CartPole-v1 training script using Stable-Baselines3 PPO.
Runs inside a Daytona sandbox; results are printed as JSON to stdout
and collected by the orchestrator.

Usage:
    python cartpole_task.py <sandbox_id> <eval_episodes> <learning_rate>
"""

import json
import sys
import time
import numpy as np

SANDBOX_ID      = sys.argv[1] if len(sys.argv) > 1 else "unknown"
EPISODES        = int(sys.argv[2])   if len(sys.argv) > 2 else 300
LEARNING_RATE   = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
SOLVE_THRESHOLD = 195.0
TRAIN_TIMESTEPS = 50_000
SEED            = hash(SANDBOX_ID) % (2**31)

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
except ImportError:
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "stable-baselines3", "gymnasium", "--quiet"],
        check=True,
    )
    import gymnasium as gym
    from stable_baselines3 import PPO


def train():
    env = gym.make("CartPole-v1")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        verbose=0,
        seed=SEED,
    )
    model.learn(total_timesteps=TRAIN_TIMESTEPS)

    episode_rewards = []
    solved_at = None
    start = time.time()

    for ep in range(1, EPISODES + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)
        avg_100 = float(np.mean(episode_rewards[-100:]))

        if avg_100 >= SOLVE_THRESHOLD and solved_at is None:
            solved_at = ep

        if ep % 50 == 0 or ep == EPISODES:
            print(json.dumps({
                "sandbox_id": SANDBOX_ID,
                "episode":    ep,
                "reward":     round(total_reward, 1),
                "avg_100":    round(avg_100, 1),
                "solved":     solved_at is not None,
                "solved_at":  solved_at,
            }), flush=True)

    elapsed   = round(time.time() - start, 1)
    final_avg = round(float(np.mean(episode_rewards[-100:])), 1)
    best      = round(max(episode_rewards), 1)

    print(json.dumps({
        "sandbox_id": SANDBOX_ID,
        "status":     "complete",
        "final_avg":  final_avg,
        "best":       best,
        "solved":     solved_at is not None,
        "solved_at":  solved_at,
        "elapsed_s":  elapsed,
        "lr":         LEARNING_RATE,
    }), flush=True)

    env.close()


if __name__ == "__main__":
    train()