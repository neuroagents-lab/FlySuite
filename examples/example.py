import numpy as np

from flysuite.environments import walk_on_ball

env = walk_on_ball()

print(f"Observation spec: \n {env.observation_spec()}\n")
print(f"Action spec: \n {env.action_spec()}\n")


# Environment loop.
def random_action_policy(observation: np.ndarray) -> np.ndarray:
    n_actions = env.action_spec().shape[0]
    random_action = np.random.uniform(-0.5, 0.5, n_actions)
    return random_action


n_steps = 200
rewards = np.zeros((n_steps,))
frames = np.zeros((n_steps, 480, 640, 3))
timestep = env.reset()
print(f"Running policy for {n_steps} steps.")
for idx in range(n_steps):
    # Running the random policy
    action = random_action_policy(timestep.observation)
    timestep = env.step(action)
    rewards[idx] = timestep.reward
    frames[idx] = env.physics.render(camera_id=0, width=640, height=480)

print(f"Rewards: \n {np.around(rewards, 2)[::10]}")
