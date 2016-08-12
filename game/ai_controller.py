from game import Game
import random
from chainer import Chain, Variable, optimizers
import chainer.functions as F
import chainer.links     as L
import numpy             as np
from collections import deque
import chainer.serializers as S
import os.path

class QNetwork(Chain):
    def __init__(self):
        super(QNetwork, self).__init__(
            conv1=F.Convolution2D(3, 32, ksize=3),
            l1=F.Linear(3744, 512),
            l2=F.Linear(512, 3))

    def __call__(self, state):
        h1 = F.leaky_relu(self.conv1(state))
        h2 = F.leaky_relu(self.l1(h1))
        h3 = F.leaky_relu(self.l2(h2))
        return h3

class AiController():
    OBSERVE_FRAME = 3200
    REPLAY_MEMORY = 50000
    BATCH = 32
    GAMMA = 0.9

    def __init__(self, game, player):
        self.__game = game
        self.__player = player
        self.__network = QNetwork()
        self.__optimizer = optimizers.Adam()
        self.__optimizer.setup(self.__network)
        self.__history = deque()
        self.__timestamp = 0
        self.load()

    def save(self):
        S.save_hdf5('network.model', self.__network)

    def load(self):
        if os.path.isfile('network.model'):
            S.load_hdf5('network.model', self.__network)

    def display_as_state(self):
        state = []
        for line in self.__game.current_display():
            state_line = []
            state.append(state_line)
            for point in line:
                value = [0.0, 0.0, 0.0]
                if point != None:
                    value[point.state_index()] = 1.0
                state_line.append(value)
        return state

    def current_state(self):
        state = np.asarray(self.display_as_state())
        state = state.astype(np.float32)
        state = state.reshape(1, 3, Game.DISPLAY_HEIGHT, Game.DISPLAY_WIDTH)
        return state

    def next(self):
        if len(self.__history) > 0:
            prev_frame = self.__history[-1]
        if len(self.__history) > AiController.REPLAY_MEMORY:
            self.__history.popleft()

        state = self.current_state()
        if random.random() < 0.1:
            action = random.randint(0, 2)
            print("RANDOM: {}", action)
        else:
            q_value = self.__network(state)
            q_value_soft = F.softmax(q_value)
            action = np.argmax(q_value_soft.data)
            print("Q: {}, SOFTMAX: {}".format(q_value.data, q_value_soft.data))

        if action == 0:
            self.__player.move_left()
        elif action == 1:
            self.__player.move_right()
        elif action == 2:
            self.__player.shoot_bullet()

        prev_point = self.__game.total_point()

        ###################################################################################
        self.__game.render()
        self.__timestamp += 1
        ###################################################################################

        print("TIME: {}, GAME SCORE: {}".format(self.__timestamp, self.__game.total_point()))
        reward = (self.__game.total_point() - prev_point) / 100.0
        state_prime = self.current_state()

        self.__history.append({
            "state": state,
            "action": action,
            "reward": reward,
            "state_prime": state_prime
        })

        if self.__timestamp > AiController.OBSERVE_FRAME:
            minibatch = random.sample(self.__history, AiController.BATCH)

            inputs = np.zeros((AiController.BATCH, 3, Game.DISPLAY_HEIGHT, Game.DISPLAY_WIDTH))
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
            loss = self.__optimizer.update(F.MeanSquaredError(), x, targets.astype(np.float32))

            if self.__timestamp % 100 == 0:
                print('save model!')
                self.save()

