import rlcard
import numpy as np

env = rlcard.make('no-limit-holdem')
obs, info = env.reset()

all_actions_seen = set()
done = False

while not done:
    legal = list(obs['raw_obs']['legal_actions'])
    all_actions_seen.update(legal)
    print("Stage:", obs['raw_obs']['stage'])
    print("Legal actions:", legal)
    print("---")
    
    action = np.random.choice(legal)
    obs, info = env.step(action)  # changed to 2 values
    done = info.get('done', False)  # get done from info

print("ALL ACTIONS SEEN:", all_actions_seen)