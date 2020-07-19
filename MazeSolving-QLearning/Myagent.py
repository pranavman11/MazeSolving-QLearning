import random
from datetime import datetime
import numpy as np


class Myagent:

    def __init__(self, maze):
        self.environment = maze
        self.Q = dict()  # table with value for (state, action) combination

    def train(self, stop_at_convergence=False, **kwargs):
        discount = kwargs.get("discount", 0.90)
        exploration_rate = kwargs.get("exploration_rate", 0.10)
        exploration_decay = kwargs.get("exploration_decay", 0.995)  # % reduction per step = 100 - exploration decay
        learning_rate = kwargs.get("learning_rate", 0.10)
        episodes = kwargs.get("episodes", 1000)

        cumulative_reward = 0
        win_history = []
        cumulative_reward_history = []

        start_time = datetime.now()

        for episode in range(1, episodes + 1):
            # optimization: make sure to start from all possible cells
            # if not start_list:
            #     start_list = self.environment.empty.copy()
            start_cell = (0, 0)
            # start_list.remove(start_cell)

            state = self.environment.reset(start_cell)
            state = tuple(state.flatten())  # change np.ndarray to tuple so it can be used as dictionary key

            while True:
                # choose action epsilon greedy (off-policy, instead of only using the learned policy)
                if np.random.random() < exploration_rate:
                    action = random.choice(self.environment.actions)
                else:
                    action = self.predict(state)

                next_state, reward, status = self.environment.step(action)
                next_state = tuple(next_state.flatten())

                cumulative_reward += reward

                if (state, action) not in self.Q.keys():  # ensure value exists for (state, action) to avoid a KeyError
                    self.Q[(state, action)] = 0.0

                max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])

                self.Q[(state, action)] += learning_rate * (reward + discount * max_next_Q - self.Q[(state, action)])

                if status in ("win", "lose"):  # terminal state reached, stop training episode
                    break

                state = next_state
                # self.environment.render_q(self)
            cumulative_reward_history.append(cumulative_reward)
            # if episode % 5 == 0:
            #
            #     # check if the current model wins from all starting cells
            #     # can only do this if there is a finite number of starting states
            #
            #     w_all, win_rate = self.environment.win_all(self)
            #     win_history.append((episode, win_rate))
            #     if w_all is True and stop_at_convergence is True:
            #
            #         break

            exploration_rate *= exploration_decay
            print(exploration_rate)
        return cumulative_reward_history, win_history, episode, datetime.now() - start_time

    def predict(self, state):
        """ Policy: choose the action with the highest value from the Q-table.
            Random choice if multiple actions have the same (max) value.

            :param np.ndarray state: Game state.
            :return int: Chosen action.
        """
        q = self.q(state)
        actions = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value
        return random.choice(actions)

    def q(self, state):
        """ Get q values for all actions for a certain state. """
        if type(state) == np.ndarray:
            state = tuple(state.flatten())

        return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])
