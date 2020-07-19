import datetime
import random

import matplotlib.pyplot as plt
import numpy as np

from Myagent import Myagent
from env import *
# import Myagent



maze = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 1, 0, 0, 0],

])  # 0 = free, 1 = occupied
game = Maze(maze)
game.render("training")
model = Myagent(game)
h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=100)

# try:
#     h  # force a NameError exception if h does not exist (and thus don't try to show win rate and cumulative reward)
#     fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
#     fig.canvas.set_window_title("Qlearning")
#     plt.show()
# except NameError:
#     pass
game.render("moves")
game.play(model, start_cell=(0, 0))
plt.show()


