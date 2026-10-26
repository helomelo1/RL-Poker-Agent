import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

from opponents import CallingStation, RandomOpponent, RuleBasedOpponent
from poker_env import PokerEnv
from state import STATE_SIZE, encode_state
from train_ppo import ActorCritic, masked_distribution, SavedAgentOpponent


def evaluate(model, opponent, hands, seed):
    env = PokerEnv(opponent_policy=opponent, seed=seed)
    payoffs = []
    skipped = 0

    try:
        for _ in range(hands):
            observation, info = env.reset()
            skipped += info.get("skipped_hands", 0)
            done = False

            while not done:
                obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                mask = torch.as_tensor(info["action_mask"].astype(bool)).unsqueeze(0)

                with torch.no_grad():
                    logits, _ = model(obs)
                    action = masked_distribution(logits, mask).sample().item()

                observation, _, terminated, truncated, info = env.step(action)
                if truncated:
                    raise RuntimeError("Unexpected truncated hand")
                done = terminated

            payoffs.append(info["payoff_chips"])
    finally:
        env.close()

    payoffs = np.asarray(payoffs, dtype=float)
    return{
        "average": payoffs.mean(),
        "win_rate": np.mean(payoffs > 0),
        "tie_rate": np.mean(payoffs == 0),
        "skipped": skipped
    }


def training_history(path):
    updates, average = [], []
    pattern = re.compile(
        r"Update\s+(\d+)/\d+.*avg chips/hand=\s*([+-]?\d+(?:\.\d+)?)"
    )

    with open(path) as log:
        for line in log:
            match = pattern.search(line)
            if match:
                updates.append(int(match.group(1)))
                average.append(float(match.group(2)))

    return np.asarray(updates), np.asarray(average)


def hand_strength_grid(model, n_samples=50):
    """
    13x13 preflop hand chart showing raise frequency.
    Upper triangle = suited, lower triangle = offsuit, diagonal = pairs
    """
    from state import encode_state
    
    RANKS = "AKQJT98765432"  # high to low (standard poker chart order)
    grid = np.zeros((13, 13), dtype=float)

    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i == j:
                # pairs: e.g. AA, KK
                hole_cards = [f"S{r1}", f"H{r1}"]
            elif i < j:
                # suited (upper triangle): e.g. AKs
                hole_cards = [f"S{r1}", f"S{r2}"]
            else:
                # offsuit (lower triangle): e.g. AKo
                hole_cards = [f"S{r1}", f"H{r2}"]

            raise_count = 0
            for _ in range(n_samples):
                state = encode_state(
                    hole_cards=hole_cards,
                    community_cards=[],
                    is_button=True,
                    pot=3.0,
                    my_chips=99.0,
                    opponent_chips=98.0,
                    to_call=1.0,
                    starting_stack=100.0,
                )
                obs = torch.as_tensor(
                    state, dtype=torch.float32
                ).unsqueeze(0)
                mask = torch.ones(5, dtype=torch.bool).unsqueeze(0)

                with torch.no_grad():
                    logits, _ = model(obs)
                    dist = masked_distribution(logits, mask)
                    action = dist.sample().item()

                if action in [2, 3, 4]:  # any raise
                    raise_count += 1

            grid[i, j] = raise_count / n_samples

    return grid


def plot_hand_grid(grid, path="hand_grid.png"):
    RANKS = "AKQJT98765432"
    fig, ax = plt.subplots(figsize=(10, 9))

    im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Raise frequency")

    # labels
    ax.set_xticks(range(13))
    ax.set_yticks(range(13))
    ax.set_xticklabels(list(RANKS), fontsize=11)
    ax.set_yticklabels(list(RANKS), fontsize=11)

    # cell text
    for i in range(13):
        for j in range(13):
            # label type: pair, suited, offsuit
            if i == j:
                label = f"{RANKS[i]}{RANKS[j]}"
            elif i < j:
                label = f"{RANKS[i]}{RANKS[j]}s"
            else:
                label = f"{RANKS[j]}{RANKS[i]}o"

            ax.text(
                j, i,
                f"{label}\n{grid[i,j]:.0%}",
                ha="center", va="center",
                fontsize=6.5,
                color="black"
            )

    ax.set_title(
        "Agent Preflop Raise Frequency\n"
        "Upper triangle = suited | Lower = offsuit | Diagonal = pairs",
        fontsize=12
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"Saved hand grid to {path}")

def position_awareness_test(model, n_samples=200):
    test_hands = [
        ("AA",  ["SA", "HA"]),
        ("KK",  ["SK", "HK"]),
        ("AKs", ["SA", "SK"]),
        ("QQ",  ["SQ", "HQ"]),
        ("AKo", ["SA", "HK"]),
        ("JTs", ["SJ", "ST"]),
        ("72o", ["D7", "H2"]),
        ("83o", ["C8", "S3"]),
    ]

    results = {}

    for name, hole_cards in test_hands:
        raise_btn, raise_nobtn = 0, 0

        for _ in range(n_samples):
            for is_button in [True, False]:
                state = encode_state(
                    hole_cards=hole_cards,
                    community_cards=[],
                    is_button=is_button,
                    pot=3.0,
                    my_chips=99.0,
                    opponent_chips=98.0,
                    to_call=1.0,
                    starting_stack=100.0,
                )
                obs = torch.as_tensor(
                    state, dtype=torch.float32
                ).unsqueeze(0)
                mask = torch.ones(5, dtype=torch.bool).unsqueeze(0)

                with torch.no_grad():
                    logits, _ = model(obs)
                    dist = masked_distribution(logits, mask)
                    action = dist.sample().item()

                if action in [2, 3, 4]:
                    if is_button:
                        raise_btn += 1
                    else:
                        raise_nobtn += 1

        results[name] = {
            "button": raise_btn / n_samples,
            "no_button": raise_nobtn / n_samples,
        }
        print(
            f"{name:6} | "
            f"button={raise_btn/n_samples:.0%} | "
            f"no_button={raise_nobtn/n_samples:.0%} | "
            f"diff={(raise_btn - raise_nobtn)/n_samples:+.0%}"
        )

    return results

def plot_position_awareness(results, path="position_awareness.png"):
    names = list(results.keys())
    btn_vals = [results[n]["button"] * 100 for n in names]
    nobtn_vals = [results[n]["no_button"] * 100 for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, btn_vals, width, label="Button (in position)", color="steelblue")
    ax.bar(x + width/2, nobtn_vals, width, label="No button (out of position)", color="salmon")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Raise Frequency (%)")
    ax.set_title("Position Awareness: Does Agent Raise More On The Button?")
    ax.legend()
    ax.set_ylim(0, 110)

    for rect, val in zip(ax.patches, btn_vals + nobtn_vals):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 1,
            f"{val:.0f}%",
            ha="center", fontsize=9
        )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"Saved to {path}")


def bet_sizing_distribution(model, n_samples=200):
    hand_buckets = {
        "Trash\n(72o,83o)":      [["D7", "H2"], ["C8", "S3"]],
        "Weak\n(J5o,T6o)":       [["HJ", "D5"], ["CT", "S6"]],
        "Playable\n(KTo,QJo)":   [["SK", "HT"], ["DQ", "CJ"]],
        "Strong\n(AKo,AQo)":     [["SA", "HK"], ["DA", "CQ"]],
        "Premium\n(AA,KK,QQ)":   [["SA", "HA"], ["SK", "HK"]],
    }

    results = {}

    for bucket_name, hands in hand_buckets.items():
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # all 5 actions
        total = 0

        for hole_cards in hands:
            for _ in range(n_samples):
                state = encode_state(
                    hole_cards=hole_cards,
                    community_cards=[],
                    is_button=True,
                    pot=3.0,
                    my_chips=99.0,
                    opponent_chips=98.0,
                    to_call=1.0,
                    starting_stack=100.0,
                )
                obs = torch.as_tensor(
                    state, dtype=torch.float32
                ).unsqueeze(0)
                mask = torch.ones(5, dtype=torch.bool).unsqueeze(0)

                with torch.no_grad():
                    logits, _ = model(obs)
                    dist = masked_distribution(logits, mask)
                    action = dist.sample().item()

                counts[action] += 1
                total += 1

        results[bucket_name] = {
            "fold":      counts[0] / total * 100,
            "call":      counts[1] / total * 100,
            "half_pot":  counts[2] / total * 100,
            "pot":       counts[3] / total * 100,
            "all_in":    counts[4] / total * 100,
        }

        print(
            f"{bucket_name.replace(chr(10), ' '):25} | "
            f"fold={results[bucket_name]['fold']:.0f}% | "
            f"call={results[bucket_name]['call']:.0f}% | "
            f"half_pot={results[bucket_name]['half_pot']:.0f}% | "
            f"pot={results[bucket_name]['pot']:.0f}% | "
            f"all_in={results[bucket_name]['all_in']:.0f}%"
        )

    return results


def plot_bet_sizing(results, path="bet_sizing.png"):
    names = list(results.keys())
    fold     = [results[n]["fold"]     for n in names]
    call     = [results[n]["call"]     for n in names]
    half_pot = [results[n]["half_pot"] for n in names]
    pot      = [results[n]["pot"]      for n in names]
    all_in   = [results[n]["all_in"]   for n in names]

    x = np.arange(len(names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - 2*width, fold,     width, label="Fold",           color="gray")
    b2 = ax.bar(x - width,   call,     width, label="Call",           color="steelblue")
    b3 = ax.bar(x,           half_pot, width, label="Raise half-pot", color="mediumseagreen")
    b4 = ax.bar(x + width,   pot,      width, label="Raise pot",      color="orange")
    b5 = ax.bar(x + 2*width, all_in,   width, label="All-in",         color="crimson")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Action Frequency (%)")
    ax.set_title(
        "Action Distribution by Hand Strength\n"
        "Fold / Call / Raise Half-Pot / Raise Pot / All-in"
    )
    ax.legend()
    ax.set_ylim(0, 110)

    for bars in [b1, b2, b3, b4, b5]:
        for rect in bars:
            h = rect.get_height()
            if h > 3:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    h + 1, f"{h:.0f}%",
                    ha="center", fontsize=7.5
                )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"Saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.hands < 1:
        parser.error("--hands must be positive")

    checkpoint = torch.load("poker_ppo_selfplay.pt", map_location="cpu", weights_only=True)
    if checkpoint["state_size"] != STATE_SIZE:
        raise ValueError("Checkpoint and state encoder have different sizes")

    model = ActorCritic()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    opponents = [
        CallingStation(),
        RandomOpponent(seed=args.seed + 1),
        RuleBasedOpponent(seed=args.seed + 2),
        SavedAgentOpponent("poker_ppo.pt"),
    ]

    names, chip_averages = [], []

    for i, opponent in enumerate(opponents):
        result = evaluate(model, opponent, args.hands, args.seed + i)
        names.append(opponent.name)
        chip_averages.append(result["average"])

        print(
            f"{opponent.name:16} "
            f"avg={result['average']:+7.2f} chips/hand | "
            f"win={result['win_rate']:.1%} | "
            f"tie={result['tie_rate']:.1%} | "
            f"evaluated={args.hands} | "
            f"skipped={result['skipped']}"
        )

    updates, averages = training_history("train.log")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if len(updates):
        axes[0].plot(updates, averages, alpha=0.3, label="Per update")
        window = min(20, len(averages))
        if len(averages) >= window:
            rolling = np.convolve(
                averages, np.ones(window) / window, mode="valid"
            )
            axes[0].plot(
                updates[window - 1 :], rolling,
                label=f"{window}-update moving average",
            )
        axes[0].legend()

    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(
        title="Training reward (mixed opponents)",
        xlabel="PPO update",
        ylabel="Average chips / hand",
    )

    axes[1].bar(names, chip_averages)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(
        title=f"Saved policy ({args.hands} fresh hands/opponent)",
        ylabel="Average chips / hand",
    )
    axes[1].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig("poker_results.png", dpi=160)
    print("Saved plot to poker_results.png")

    print("\nGenerating preflop hand strength grid...")
    grid = hand_strength_grid(model, n_samples=50)
    plot_hand_grid(grid)

    print("\nRunning bet sizing distribution test...")
    sizing_results = bet_sizing_distribution(model)
    plot_bet_sizing(sizing_results)

if __name__ == "__main__":
    main()
