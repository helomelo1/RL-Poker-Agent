"""
    Training:
        python3 train_ppo.py --updates 200 --hands-per-update 128
"""

import argparse
import random

from matplotlib.cm import tab10
import numpy as np
import torch
from torch._inductor.config import distributed_max_autotune_gemm
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from opponents import CallingStation, RandomOpponent, RuleBasedOpponent
from poker_env import NUM_ACTIONS, PokerEnv, STARTING_STACK
from state import STATE_SIZE, encode_state


class ActorCritic(nn.Module):
    "Shared card/betting representation with policy and value heads."

    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(STATE_SIZE, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.policy = nn.Linear(128, NUM_ACTIONS)
        self.value = nn.Linear(128, 1)

    def forward(self, observations):
        features = self.body(observations)
        logits = self.policy(features)
        values = self.value(features).squeeze(-1)

        return logits, values


class SavedAgentOpponent:
    """Uses a saved PPO checkpoint as an opponent."""

    name = "saved_agent"

    def __init__(self, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.model = ActorCritic()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.device = device

    def __call__(self, observation):
        raw    = observation["raw_obs"]
        stakes = raw["stakes"]

        state = encode_state(
            hole_cards=raw["hand"],
            community_cards=raw["public_cards"],
            is_button=False,
            pot=float(raw["pot"]),
            my_chips=float(stakes[1]),
            opponent_chips=float(stakes[0]),
            to_call=float(max(raw["all_chips"]) - raw["my_chips"]),
            starting_stack=STARTING_STACK,
        )

        legal  = list(observation["legal_actions"])
        mask   = np.zeros(NUM_ACTIONS, dtype=bool)
        for a in legal:
            mask[int(a)] = True

        obs_t  = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.as_tensor(mask,  dtype=torch.bool).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(obs_t)
            return masked_distribution(logits, mask_t).sample().item()



def masked_distribution(logits, masks):
    "Giving illegal actions zero probablity"
    if not torch.all(masks.any(dim=-1)):
        raise ValueError("A decision state has no legal actions")

    masked_logits = logits.masked_fill(~masks, -1e9)
    return Categorical(logits=masked_logits)


def compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=0.95):
    "gae: generalized advantage estimation"
    advantages = np.zeros(len(rewards), dtype=np.float32)
    next_advantage = 0.0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0.0
            next_advantage = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        next_advantage = delta + gamma * gae_lambda * next_advantage
        advantages[t] = next_advantage

    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def collect_hands(env, model, opponents, hands, device, rng):
    "Collect complete hands with the current (old) policy"
    rollout = {
        "observations": [],
        "masks": [],
        "actions": [],
        "log_probs": [],
        "values": [],
        "rewards": [],
        "dones": []
    }

    chip_payoffs = []
    skipped_hands = 0

    model.eval()

    for _ in range(hands):
        env.opponent_policy = rng.choice(opponents)
        observation, info = env.reset()
        skipped_hands += info.get("skipped_hands", 0)

        done = False

        while not done:
            mask = info["action_mask"].astype(bool)
            obs_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            mask_tensor = torch.as_tensor(
                mask, dtype=torch.bool, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                logits, value = model(obs_tensor)
                distribution = masked_distribution(logits, mask_tensor)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)

            next_observation, reward, terminated, truncated, next_info = (
                env.step(action.item())
            )
            done = terminated or truncated

            if truncated:
                raise RuntimeError("Trainer expects complete hands, not incomplete ones")

            rollout["observations"].append(observation.copy())
            rollout["masks"].append(mask.copy())
            rollout["actions"].append(action.item())
            rollout["log_probs"].append(log_prob.item())
            rollout["values"].append(value.item())
            rollout["rewards"].append(float(reward))
            rollout["dones"].append(done)

            observation, info = next_observation, next_info

        chip_payoffs.append(info["payoff_chips"])

    advantages, returns = compute_gae(
        rollout["rewards"],
        rollout["values"],
        rollout["dones"]
    )
    rollout["advantages"] = advantages
    rollout["returns"] = returns

    return rollout, float(np.mean(chip_payoffs)), skipped_hands


def ppo_update(model, optimizer, rollout, device, epochs=4, batch_size=128):
    observations = torch.as_tensor(
        np.asarray(rollout["observations"]),
        dtype=torch.float32,
        device=device
    )
    masks = torch.as_tensor(
        np.asarray(rollout["masks"]),
        dtype=torch.bool,
        device=device
    )
    actions = torch.as_tensor(
        rollout["actions"], dtype=torch.long, device=device
    )
    old_log_probs = torch.as_tensor(
        rollout["log_probs"], dtype=torch.float32, device=device
    )
    returns = torch.as_tensor(
        rollout["returns"], dtype=torch.float32, device=device
    )
    advantages = torch.as_tensor(
        rollout["advantages"], dtype=torch.float32, device=device
    )
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )

    n = len(actions)
    model.train()
    losses = []

    for _ in range(epochs):
        indices = torch.randperm(n, device=device)

        for start in range(0, n, batch_size):
            batch = indices[start : start + batch_size]

            logits, values = model(observations[batch])
            distribution = masked_distribution(logits, masks[batch])
            new_log_probs = distribution.log_prob(actions[batch])

            ratio = torch.exp(new_log_probs - old_log_probs[batch])
            unclipped = ratio * advantages[batch]
            clipped = torch.clamp(ratio, 0.8, 1.2) * advantages[batch]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = (values - returns[batch]).square().mean()
            entropy = distribution.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            losses.append(loss.item())

    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--hands-per-update", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vs-saved",
        type=str,
        default=None,
        metavar="PATH",
        help="Also train against a saved checkpoint, e.g. --vs-saved poker_ppo.pt",
    )
    args = parser.parse_args()

    if args.updates < 1 or args.hands_per_update < 1:
        parser.error("--updates and --hands-per-update must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device("cpu")

    opponents = [
        CallingStation(),
        RandomOpponent(seed=args.seed + 1),
        RuleBasedOpponent(seed=args.seed + 2),
        RuleBasedOpponent(seed=args.seed + 3),
        RuleBasedOpponent(seed=args.seed + 4),
        RuleBasedOpponent(seed=args.seed + 5),
    ]
    if args.vs_saved:
        opponents.append(SavedAgentOpponent(args.vs_saved))
        print(f"Added saved-agent opponent from: {args.vs_saved}")
    env = PokerEnv(seed=args.seed)
    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    try:
        for update in range(1, args.updates + 1):
            rollout, avg_chips, skipped = collect_hands(
                env=env,
                model=model,
                opponents=opponents,
                hands=args.hands_per_update,
                device=device,
                rng=rng
            )
            loss = ppo_update(model, optimizer, rollout, device)

            print(
                f"Update {update:4}/{args.updates} | "
                f"decisions={len(rollout['actions']):4} | "
                f"avg chips/hand={avg_chips:+7.2f} | "
                f"skipped={skipped:3} | "
                f"loss={loss:+.4f}",
                flush=True,
            )
    finally:
        env.close()

    save_path = "poker_ppo_selfplay.pt" if args.vs_saved else "poker_ppo.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "state_size": STATE_SIZE,
        "num_actions": NUM_ACTIONS,
        "updates": args.updates,
        "seed": args.seed,
    }, save_path)
    print(f"Saved trained model to {save_path}")


if __name__ == "__main__":
    main()
