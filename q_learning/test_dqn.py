import argparse
import torch
from env_2048 import ACTIONS, Game2048Env, encode_state
from train_dqn import DQN
import random
import math

steps_done = 0

def test(model_path, device, eps_end, eps_start, eps_decay):
    n_actions = len(ACTIONS)
    env2048 = Game2048Env()
    state = env2048.reset()
    state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)

    n_observations = len(encode_state(env2048.matrix))

    loaded_states = torch.load(model_path)
    policy_net = DQN(n_observations, n_actions, device)

    policy_net.load_state_dict(loaded_states['policy_net_state'])
    policy_net.eval()

    def select_action(state: torch.Tensor):
        global steps_done
        sample = random.random()
        eps_threshold = eps_end  + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                action_dist : torch.Tensor = policy_net(state)
                non_valid_actions : set[int] = set(ACTIONS) - set(env2048.get_valid_actions())

                filtered_action_dist = action_dist[0].put(torch.tensor(list(non_valid_actions), device=device, dtype=torch.long), torch.tensor(-torch.inf, device = device).repeat(len(non_valid_actions)), False)
                return filtered_action_dist.argmax().cpu().detach().numpy().astype(int)
        else:
            return random.choice(env2048.get_valid_actions())

    while True:
        action = select_action(state)
        observation, reward, terminated  = env2048.step(action)
        reward = torch.tensor([reward], device=device)
        
        if terminated:
            print(observation, max(observation))
            break
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN training")
    parser.add_argument("--model_path", type=str, default = None, help = "model path")
    parser.add_argument("--device", "-d", type=str, choices=["cuda", "cpu"], default="cuda", help="inference device")
    parser.add_argument("--eps_start", type=float, default=0.9, help="epsilon start value")
    parser.add_argument("--eps_end", type=float, default=0.01, help="epsilon end value")
    parser.add_argument("--eps_decay", type=int, default= 2500, help="epsilon decay")
    args = parser.parse_args()
    test(
        model_path=args.model_path,
        device=args.device, 
        eps_decay=args.eps_decay, 
        eps_end=args.eps_end, 
        eps_start=args.eps_start
    )
