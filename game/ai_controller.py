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
    OBSERVE_FRAME = 100
    REPLAY_MEMORY = 50000
    BATCH = 32
    GAMMA = 0.99

    def __init__(self, game, player):
        self.__game = game
        self.__player = player
        self.__network = QNetwork()
        self.__optimizer = optimizers.Adam()
        self.__optimizer.setup(self.__network)
        self.__history = deque()
        self.__timestamp = 0

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

    def current_state(self):
        state = np.asarray(self.display_as_state())
        state = state.astype(np.float32)
        state = state.reshape(1, 1, Game.DISPLAY_HEIGHT, Game.DISPLAY_WIDTH)
        return state

    def next(self):
        if len(self.__history) > 0:
            prev_frame = self.__history[-1]
        if len(self.__history) > AiController.REPLAY_MEMORY:
            self.__history.popleft()

        state = self.current_state()
        q_value = self.__network(state)
        action = np.argmax(q_value.data)
        print(q_value.data)

        if action == 0:
            self.__player.move_left()
        elif action == 1:
            self.__player.move_right()
        elif action == 2:
            self.__player.shoot_bullet()

        prev_point = self.__game.total_point()

        self.__game.render()
        self.__timestamp += 1

        print("TIME: {}, GAME SCORE: {}".format(self.__timestamp, self.__game.total_point()))
        reward = self.__game.total_point() - prev_point
        state_prime = self.current_state()

        self.__history.append({
            "state": state,
            "action": action,
            "reward": reward,
            "state_prime": state_prime,
            "q_value": q_value
        })

        if self.__timestamp > AiController.OBSERVE_FRAME:
            minibatch = random.sample(self.__history, AiController.BATCH)

            inputs = np.zeros((AiController.BATCH, 1, Game.DISPLAY_HEIGHT, Game.DISPLAY_WIDTH))
            targets = np.zeros((inputs.shape[0], 3))

            for i in range(0, len(minibatch)):
                data = minibatch[i]
                state = data['state']
                action = data['action']
                reward = data['reward']
                state_prime = data['state_prime']

                inputs[i : i + 1] = state
                targets[i] = self.__network(state).data
                Q_sa = self.__network(state_prime)

                targets[i, action] = reward + AiController.GAMMA * np.max(Q_sa.data)

            x = self.__network(inputs.astype(np.float32))
            self.__optimizer.update(F.MeanSquaredError(), x, targets.astype(np.float32))

