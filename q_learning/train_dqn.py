import argparse
from collections import deque, namedtuple
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from env_2048 import Game2048Env, ACTIONS, encode_state
import math
import numpy as np


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, device):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 512, device=device)
        self.layer2 = nn.Linear(512, 256, device=device)
        self.layer3 = nn.Linear(256, n_actions, device=device)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
steps_done = 0

def train(batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.01, eps_decay=2500, tau=0.005, lr=3e-4, num_episodes=200, C=10, device="cuda"):
    n_actions = len(ACTIONS)
    env2048 = Game2048Env()
    env2048.reset()

    n_observations = len(encode_state(env2048.matrix))
    policy_net = DQN(n_observations, n_actions, device)
    target_net = DQN(n_observations, n_actions, device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad= True)
    memory = ReplayMemory(10000)

    def select_action(state: torch.Tensor) -> torch.Tensor:
        global steps_done
        sample = random.random()
        eps_threshold = eps_end  + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                action_dist : torch.Tensor = policy_net(state)
                non_valid_actions : set[int] = set(ACTIONS) - set(env2048.get_valid_actions())

                filtered_action_dist = action_dist[0].put(torch.tensor(list(non_valid_actions), device=device, dtype=torch.long), torch.tensor(-torch.inf, device = device).repeat(len(non_valid_actions)), False)
                return filtered_action_dist.argmax().unsqueeze(0).unsqueeze(0)
        else:
            return torch.tensor([[random.choice(env2048.get_valid_actions())]], device=device, dtype=torch.long)

    def optimize_model() -> torch.Tensor:
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size=batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = device, dtype= torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        action_dist : torch.Tensor = policy_net(state_batch)

        state_action_values = action_dist.gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)

        with torch.no_grad():
            q_value_action_dist : torch.Tensor = target_net(non_final_next_states)
            next_state_values[non_final_mask] = q_value_action_dist.max(1).values

        expected_state_action_values = (next_state_values * gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 10)
        optimizer.step()

        return loss

    episode_durations = []
    episode_losses = []

    for i_episode in range(num_episodes):
        state = env2048.reset()
        state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
        t = 0
        episode_loss = []
        while True:
            t += 1
            action = select_action(state)
            observation, reward, terminated  = env2048.step(action)
            reward = torch.tensor([reward], device=device)
            
            if terminated:
                print("Max Value:", torch.max(state.cpu().detach()))
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            memory.push(state, action, next_state, reward)

            state = next_state

            loss = optimize_model()

            if (loss is not None):
                episode_loss.append(loss.cpu().detach().numpy())
                # if (steps_done % 500 == 0):
                #     print(f"Episode {i_episode} Loss: {loss} | Step count: {steps_done}")

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            if (steps_done % C == 0):
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
                target_net.load_state_dict(target_net_state_dict)

            if terminated:
                episode_losses.append(episode_loss)
                episode_durations.append(t + 1)
                print(f"Episode {i_episode}: {np.mean(np.array(episode_loss))} Duration: {t + 1}")
                break

    torch.save({
        'policy_net_state': policy_net.state_dict(), 
        'target_net_state': target_net.state_dict(),
        },
        f"policy_target_net_state_{num_episodes}"
    )

    print('Complete')

if __name__ == "__main__":
    # Take the arguments and run
    # batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.01, eps_decay=2500, tau=0.005, lr=3e-4, device="cuda"
    parser = argparse.ArgumentParser(description="DQN training")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="batch size for sampling from the replay buffer")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--gamma", "-g", type=float, default=0.99, help="gamma value")
    parser.add_argument("--eps_start", type=float, default=0.9, help="epsilon start value")
    parser.add_argument("--eps_end", type=float, default=0.01, help="epsilon end value")
    parser.add_argument("--eps_decay", type=int, default= 2500, help="epsilon decay")
    parser.add_argument("--tau", "-t", type=float, default=0.005, help="tau value")
    parser.add_argument("--num_episodes", type=int, default = 200, help="Maximum Number of Episodes")
    parser.add_argument("-C", type = int, default=10, help = "Target State Copy")
    args = parser.parse_args()

    # Expose parsed args to the rest of the module (run can be updated to use them)
    train.cli_args = args

    print(f"Starting with args: batch_size={args.batch_size}, lr={args.lr}, device={args.device}, gamma={args.gamma} ,eps_start={args.eps_start},eps_end={args.eps_end},eps_decay={args.eps_decay}, tau={args.tau}")
    train(
        batch_size=args.batch_size, 
        gamma = args.gamma, 
        eps_start = args.eps_start, 
        eps_end = args.eps_end, 
        eps_decay = args.eps_decay, 
        tau = args.tau, 
        lr = args.lr, 
        num_episodes = args.num_episodes,
        C = args.C,
        device = args.device
    )





