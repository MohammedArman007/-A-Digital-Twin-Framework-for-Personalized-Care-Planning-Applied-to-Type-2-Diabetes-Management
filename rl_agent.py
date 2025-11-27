"""Reinforcement Learning Agent Stub (simple tabular Q-learning for prototyping)
This is a minimal illustrative agent to show how an RL agent might be structured.
It is NOT production-ready for clinical use.
"""
import numpy as np
import random

class SimpleCGMEnv:
    """Very small toy environment:
       state = discretized glucose level bucket (0..N-1)
       actions = {0: do_nothing, 1: small_walk, 2: carbs, 3: insulin}
       transition is stochastic and simple
    """
    def __init__(self, n_states=20):
        self.n_states = n_states
        self.state = n_states//2
    def reset(self):
        self.state = self.n_states//2
        return self.state
    def step(self, action):
        # simple dynamics: actions shift state distribution
        if action==0:
            drift = np.random.choice([-1,0,1], p=[0.3,0.4,0.3])
        elif action==1:
            drift = -2 + np.random.choice([-1,0,1], p=[0.2,0.6,0.2])
        elif action==2:
            drift = 3 + np.random.choice([0,1], p=[0.7,0.3])
        else:
            drift = -3 + np.random.choice([0,-1], p=[0.7,0.3])
        self.state = int(np.clip(self.state + drift, 0, self.n_states-1))
        # reward: prefer middle-range states (representing TIR)
        mid = self.n_states//2
        reward = 1.0 - abs(self.state - mid)/(self.n_states/2)
        done = False
        return self.state, reward, done, {}

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, eps=0.1):
        self.q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    def select_action(self, s):
        if random.random() < self.eps:
            return random.randrange(self.q.shape[1])
        return int(np.argmax(self.q[s]))
    def update(self, s,a,r,s2):
        self.q[s,a] += self.alpha * (r + self.gamma * self.q[s2].max() - self.q[s,a])

def train_agent(episodes=200):
    env = SimpleCGMEnv(n_states=30)
    agent = QLearningAgent(n_states=30, n_actions=4)
    for ep in range(episodes):
        s = env.reset()
        total = 0
        for t in range(100):
            a = agent.select_action(s)
            s2, r, done, _ = env.step(a)
            agent.update(s,a,r,s2)
            s = s2
            total += r
        if (ep+1) % 50 == 0:
            print(f'Episode {ep+1}, avg reward {total/100:.3f}')
    return agent

if __name__ == '__main__':
    agent = train_agent(episodes=500)
    print('Trained Q-table shape:', agent.q.shape)
