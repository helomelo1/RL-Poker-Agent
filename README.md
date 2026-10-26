# RL Poker Agent

A heads-up no-limit Texas Hold'em agent trained with hand-written PPO in PyTorch.

## Setup

```bash
git clone https://github.com/yourname/RL-Poker-Agent.git
cd RL-Poker-Agent
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Train

```bash
# Train from scratch against three opponents
python3 train_ppo.py --updates 200 --hands-per-update 128

# Continue training with a saved agent as a fourth opponent
python3 train_ppo.py --updates 200 --hands-per-update 128 --vs-saved poker_ppo.pt
```

## Evaluate

```bash
# Evaluate the default checkpoint against three opponents
python3 evaluate.py --hands 10000
```

## Stack

- [RLCard](https://github.com/datamllab/rlcard) — poker environment
- PyTorch — PPO implementation
- Gymnasium — environment interface
- Matplotlib — plots
