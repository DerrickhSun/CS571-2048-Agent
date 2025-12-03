import random
import pickle
from collections import defaultdict
from env_2048 import Game2048Env, ACTIONS

def epsilon_greedy(Q, state, epsilon, env):
    if random.random() < epsilon:
        valid_actions = env.get_valid_actions()
        if len(valid_actions) == 0:
            return random.choice(ACTIONS)
        return random.choice(valid_actions)

    q_vals = Q[state]
    max_q = max(q_vals)
    best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
    return random.choice(best_actions)

def reward_basic(state, next_state, reward):
    return reward

def reward_shaped(state, next_state, reward, beta=0.1, gamma_empty=0.3):
    before_max = max(state)
    after_max = max(next_state)
    bonus_max = max(0, after_max - before_max)

    empty_before = state.count(0)
    empty_after = next_state.count(0)
    bonus_empty = empty_after - empty_before

    return reward + beta * bonus_max + gamma_empty * bonus_empty


def train_q_learning(
    num_episodes=5000,
    alpha=0.1,
    discount=0.99,
    epsilon_start=1.0,
    epsilon_min=0.1,
    epsilon_decay=0.9999,
    report_every=500,
    reward_fn=reward_basic  
):
    env = Game2048Env()
    Q = defaultdict(lambda: [0.0 for _ in ACTIONS])

    epsilon = epsilon_start
    scores = []

    for episode in range(num_episodes):
        state = env.reset()
        unused = Q[state] 
        done = False
        episode_reward = 0

        while not done:
            action = epsilon_greedy(Q, state, epsilon, env)

            next_state, reward, done = env.step(action)

            # use whichever reward function was passed in
            r = reward_fn(state, next_state, reward)
            episode_reward += r

            best_next_q = 0.0 if done else max(Q[next_state])
            td_target = r + discount * best_next_q
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        scores.append(episode_reward)

        if (episode + 1) % report_every == 0:
            avg_reward = sum(scores[-report_every:]) / report_every
            print(f"Episode {episode+1}: avg reward {avg_reward:.2f}, epsilon {epsilon:.3f}")

    return Q

if __name__ == "__main__":
    Q = train_q_learning(reward_fn=reward_shaped)

    # Save Q table
    with open("q_table_shaped.pkl", "wb") as f:
        pickle.dump(dict(Q), f)
    print("Saved Q table with", len(Q), "states")