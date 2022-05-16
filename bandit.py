import random
import numpy as np
import abc

from multiprocessing import Pool


class World:
    def __init__(self, idx: int, num_actions: int):
        self._id = idx
        self._means = np.random.normal(loc=0, scale=1, size=num_actions)
        self._deviations = np.ones(num_actions)
        self._optimal_action = np.argmax(self._means)

    def take_action(self, action_id: int) -> float:
        return np.random.normal(self._means[action_id], self._deviations[action_id])

    @property
    def id(self):
        return self._id
    
    def __repr__(self):
        return f"world_{self.id}"


class Agent(abc.ABC):
    def __init__(self, num_actions: int, steps_per_epoch: int):
        self._num_actions = num_actions
        self._rewards = np.zeros((num_actions, steps_per_epoch))
        self._steps_elapsed = 0

    def receive_reward(self, action_id: int, reward: float) -> None:
        self._rewards[action_id][self._steps_elapsed] = reward
        self._steps_elapsed += 1

    @abc.abstractmethod
    def next_action(self) -> int:
        pass

    def total_reward(self) -> int:
        return int(np.sum(self._rewards))

    @property
    def name(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return self.name


class RandomAgent(Agent):
    def next_action(self) -> int:
        return np.random.randint(0, self._num_actions)


class GreedyAgent(Agent):
    def next_action(self) -> int:
        return int(np.argmax(np.max(self._rewards, axis=1)))


class EGreedyAgent(Agent):
    def __init__(self, num_actions: int, steps_per_epoch: int, eps: float = 0.01):
        super().__init__(num_actions, steps_per_epoch)
        self._eps = eps

    def next_action(self) -> int:
        if random.random() <= self._eps:
            return np.random.randint(0, self._num_actions)
        else:
            return int(np.argmax(np.max(self._rewards, axis=1)))

    @property
    def name(self):
        return f"{super().name}_{self._eps}"


def run_world(world: World, agent: Agent, num_steps: int) -> tuple[World, Agent]:
    for _ in range(num_steps):
        action_id = agent.next_action()
        reward = world.take_action(action_id)
        agent.receive_reward(action_id, reward)
    return world, agent


def main():
    num_actions = 10
    num_worlds = 100
    steps_per_epoch = 2000

    worlds = [World(i, num_actions) for i in range(num_worlds)]

    agent_creators = [
        lambda: RandomAgent(num_actions, steps_per_epoch),
        lambda: GreedyAgent(num_actions, steps_per_epoch),
        lambda: EGreedyAgent(num_actions, steps_per_epoch, eps=0.01),
        lambda: EGreedyAgent(num_actions, steps_per_epoch, eps=0.1),
    ]

    with Pool(6) as pool:
        world_agent_pairs = [
            (world, create_agent(), steps_per_epoch) for world in worlds for create_agent in agent_creators
        ]
        world_agent_pairs = pool.starmap(run_world, world_agent_pairs)

    for (world, agent) in enumerate(world_agent_pairs):
        print(agent.name, agent.total_reward())


if __name__ == '__main__':
    main()



