import random
import numpy as np
import abc
from typing import Callable

from multiprocessing import Pool


class World:
    def __init__(self, idx: int, num_actions: int):
        self._id = idx
        self._means = np.random.normal(loc=0, scale=1, size=num_actions)
        self._deviations = np.ones(num_actions)
        self._optimal_action = np.argmax(self._means)

    def take_action(self, action_id: int) -> tuple[float, int]:
        is_optimal = int(action_id == self._optimal_action)
        return np.random.normal(self._means[action_id], self._deviations[action_id]), is_optimal

    @property
    def id(self):
        return self._id
    
    def __repr__(self):
        return f"world_{self.id}"


class Agent(abc.ABC):
    def __init__(self, num_actions: int):
        self._num_actions = num_actions
        self._reward_estimates = np.zeros(num_actions)
        self._total_reward = 0

    def receive_reward(self, action_id: int, reward: float) -> None:
        self._reward_estimates[action_id] += 0.01*(reward - self._reward_estimates[action_id])
        self._total_reward += reward

    @abc.abstractmethod
    def next_action(self) -> int:
        pass

    @property
    def total_reward(self) -> int:
        return self._total_reward

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
        return int(np.argmax(self._reward_estimates))


class EGreedyAgent(Agent):
    def __init__(self, num_actions: int, eps: float = 0.01):
        super().__init__(num_actions)
        self._eps = eps

    def next_action(self) -> int:
        if random.random() <= self._eps:
            return np.random.randint(0, self._num_actions)
        else:
            return int(np.argmax(self._reward_estimates))

    @property
    def name(self):
        return f"{super().name}_{self._eps}"


def run_experiment(agent: Agent, agent_id: int, world: World, world_id: int, steps_per_world: int):
    result = np.zeros((steps_per_world, 4))
    for s in range(steps_per_world):
        action_id = agent.next_action()
        reward, is_optimal = world.take_action(action_id)
        agent.receive_reward(action_id, reward)

        result[s][0] = agent_id
        result[s][1] = world_id
        result[s][2] = reward
        result[s][3] = is_optimal
    return result


def main():
    num_actions = 10
    num_worlds = 10
    steps_per_world = 20

    agent_creators = [
        lambda: RandomAgent(num_actions),
        lambda: GreedyAgent(num_actions),
        lambda: EGreedyAgent(num_actions, eps=0.01),
        lambda: EGreedyAgent(num_actions, eps=0.1),
    ]

    create_world = lambda i: World(i, num_actions)

    args = [
        (create_agent(), agent_id, create_world(world_id), world_id, steps_per_world)
        for world_id in range(num_worlds)
        for agent_id, create_agent in enumerate(agent_creators)
    ]
    with Pool(len(agent_creators)) as pool:
        results = pool.starmap(run_experiment, args)

    aggregated_result = np.stack(results, axis=2)
    print(aggregated_result.shape)


if __name__ == '__main__':
    main()



