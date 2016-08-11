from game import Game
import random
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links     as L
import numpy             as np
from collections import deque

class QNetwork(Chain):
    def __init__(self):
        super(QNetwork, self).__init__(
            conv1=F.Convolution2D(1, 32, ksize=8),
            l1=F.Linear(1024, 512),
            l2=F.Linear(512, 3))

    def __call__(self, state):
        h1 = F.relu(self.conv1(state))
        h2 = F.relu(self.l1(h1))
        h3 = F.relu(self.l2(h2))
        return F.softmax(h3)

class AiController():
    OBSERVE_FRAME = 3200
    REPLAY_MEMORY = 50000

    def __init__(self, game):
        self.__game = game
        self.__network = QNetwork()
        self.__optimizer = optimizers.Adam()
        self.__optimizer.setup(self.__network)
        self.__history = deque()

    def display_as_state(self):
        state = []
        for line in self.__game.current_display():
            state_line = []
            state.append(state_line)
            for point in line:
                if point == None:
                    state_line.append(0)
                else:
                    state_line.append(point.state_value())
        return state

    def get_command(self):
        if len(self.__history) > 0:
            prev_frame = self.__history[-1]

        state = np.asarray(self.display_as_state())
        state = state.astype(np.float32)
        state = state.reshape(1, 1, Game.DISPLAY_HEIGHT, Game.DISPLAY_WIDTH)
        q_value = self.__network(state)
        action = np.argmax(q_value.data)
        self.__history.append({
            "state": state,
            "action": action,
        })
        print(q_value.data)
        return action
