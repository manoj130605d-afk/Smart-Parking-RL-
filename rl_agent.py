"""Tabular Q-learning agent and baseline policies for the parking simulator."""
from collections import defaultdict
import random
from parking_env import ParkingEnv

class QLearningAgent:
    def __init__(self, n_actions=3, alpha=0.15, gamma=0.93,
                 epsilon=1.0, epsilon_min=0.04, epsilon_decay=0.992, seed=7):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = defaultdict(lambda: [0.0] * n_actions)
        self.rng = random.Random(seed)

    def act(self, state, valid_actions, explore=True):
        if not valid_actions:
            return 0
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)
        values = self.q[state]
        best = max(valid_actions, key=lambda a: values[a])
        return best

    def learn(self, state, action, reward, next_state, next_valid, done):
        old = self.q[state][action]
        future = max((self.q[next_state][a] for a in next_valid), default=0.0)
        target = reward if done else reward + self.gamma * future
        self.q[state][action] += self.alpha * (target - old)

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_agent(episodes=1800, seed=7):
    env = ParkingEnv(seed=seed)
    agent = QLearningAgent(seed=seed)
    episode_rewards = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            valid = env.valid_actions()
            action = agent.act(state, valid, explore=True)
            next_state, reward, done, _ = env.step(action)
            next_valid = env.valid_actions()
            agent.learn(state, action, reward, next_state, next_valid, done)
            state = next_state
            total += reward
        agent.end_episode()
        episode_rewards.append(total)
    return agent, episode_rewards

def baseline_action(env, policy):
    valid = env.valid_actions()
    if not valid:
        return 0
    if policy == "first_available":
        return valid[0]
    if policy == "nearest_available":
        return min(valid, key=lambda a: env.distances[a])
    if policy == "least_occupied":
        return min(valid, key=lambda a: len(env.parked[a]) / env.capacities[a])
    raise ValueError("Unknown policy: " + policy)

def evaluate(agent=None, policy="rl", episodes=100, seed=1000):
    rows = []
    for ep in range(episodes):
        env = ParkingEnv(seed=seed + ep)
        state = env.reset()
        done = False
        while not done:
            valid = env.valid_actions()
            if policy == "rl":
                action = agent.act(state, valid, explore=False)
            else:
                action = baseline_action(env, policy)
            state, _, done, _ = env.step(action)
        m = env.metrics()
        m["episode"] = ep
        rows.append(m)
    return rows
