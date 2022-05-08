import numpy as np

from multiprocessing import Pool


class World:
    def __init__(self, num_actions):
        self.__means = np.random.rand(num_actions)
        self.__deviations = np.ones(num_actions)

    def take_action(self, action_id) -> float:
        return np.random.normal(self.__means[action_id], self.__deviations[action_id])


class Agent:
    def __init__(self, num_actions: int, steps_per_epoch: int):
        self.__num_actions = num_actions
        self.__rewards = np.zeros((num_actions, steps_per_epoch))
        self.__steps_elapsed = 0

    def receive_reward(self, action_id: int, reward: float) -> None:
        self.__rewards[action_id][self.__steps_elapsed] = reward
        self.__steps_elapsed += 1

    def next_action(self) -> int:
        return np.random.randint(0, self.__num_actions)

    def reset(self):
        self.__steps_elapsed = 0


num_actions = 10
steps_per_epoch = 2
num_worlds = 1
num_agents = 1

worlds = [World(num_actions) for _ in range(num_worlds)]
agents = [Agent(num_actions, steps_per_epoch) for _ in range(num_agents)]

for agent in agents:
    for world in worlds:
        for _ in range(steps_per_epoch):
            action_id = agent.next_action()
            reward = world.take_action(action_id)
            agent.receive_reward(action_id, reward)

        agent.reset()





