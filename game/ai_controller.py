# -*- coding: utf-8 -*-

from game import Game
import random
from chainer import Chain, Variable, optimizers, cuda
import chainer.functions as F
import chainer.links     as L
import numpy             as np
from collections import deque
import chainer.serializers as S
import os.path

class QNetwork(Chain):
    def __init__(self):
        super(QNetwork, self).__init__(
            conv1=F.Convolution2D(3, 32, ksize=(1, 10), pad=0),
            l1=F.Linear(960, 256),
            l2=F.Linear(256, 3))

    def __call__(self, state):
        h1 = F.leaky_relu(self.conv1(state))
        h4 = F.leaky_relu(self.l1(h1))
        h5 = self.l2(h4)
        return h5

class Average(object):
    def __init__(self, size):
        self.__size = size
        self.__values = deque()

    def add(self, value):
        self.__values.append(value)
        if len(self.__values) > self.__size:
            self.__values.popleft()

    def average(self):
        return sum(self.__values) / float(len(self.__values))

class AiController(object):
    OBSERVE_FRAME = 3200
    REPLAY_MEMORY = 50000
    BATCH = 32
    GAMMA = 0.97

    def __init__(self, game, player, policy, with_train, verbose, gpu):
        self.__with_train = with_train
        self.__verbose = verbose
        self.__policy = policy
        self.__game = game
        self.__player = player
        self.__network = QNetwork()

        if gpu >= 0:
            device = cuda.get_device(int(gpu))
            device.use()
            self.__network.to_gpu()
            print('GPU MODE: {0}'.format(device))

        self.xp = self.__network.xp

        self.__optimizer = optimizers.Adam()
        self.__optimizer.setup(self.__network)

        self.__timestamp = 0
        self.__point = 0.0

        self.__history = deque()
        self.__loss_average = Average(1000)

        self.__train_inputs = self.__network.xp.zeros((AiController.BATCH, 3, Game.DISPLAY_HEIGHT, Game.DISPLAY_WIDTH))
        self.__train_targets = self.__network.xp.zeros((AiController.BATCH, 3))

        self.load()

    def log(self, str):
        if self.__verbose:
            print(str)

    def save(self):
        S.save_hdf5('network.model', self.__network)

    def load(self):
        if os.path.isfile('network.model'):
            S.load_hdf5('network.model', self.__network)

    def get_display_as_state(self):
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
        state = self.xp.asarray(self.get_display_as_state())
        state = state.astype(self.xp.float32)
        state = state.reshape(1, 3, Game.DISPLAY_HEIGHT, Game.DISPLAY_WIDTH)
        return state

    def asarray(self, x):
        return self.__network.xp.asarray(x)

    def random_history(self, size):
        return random.sample(self.__history, AiController.BATCH)

    def push_history(self, step_data):
        self.__history.append(step_data)
        if len(self.__history) > AiController.REPLAY_MEMORY:
            self.__history.popleft()

    def next(self):
        state = self.asarray(self.current_state())
        q_value = self.__network(state)

        if self.__policy == 'greedy':
            action = self.xp.argmax(q_value.data.reshape(-1))
            self.log("GREEDY: {}".format(action))
        elif self.__policy == 'egreedy':
            if random.random() < 0.1:
                action = random.randint(0, 2)
                self.log("ε-greedy RANDOM: {}".format(action))
            else:
                action = self.xp.argmax(q_value.data.reshape(-1))
                self.log("ε-greedy GREEDY: {}".format(action))
        elif self.__policy == 'softmax':
            q_value_soft = F.softmax(q_value / 0.1)
            prob = q_value_soft.data.reshape(-1)
            action = np.random.choice(len(prob), p=prob)
            self.log("Q: {}, SOFTMAX: {}".format(q_value.data, q_value_soft.data))

        if action == 0:
            self.__player.move_left()
        elif action == 1:
            self.__player.move_right()
        elif action == 2:
            self.__player.shoot_bullet()

        prev_point = self.__game.total_point()

        ###################################################################################
        self.__game.update()
        self.__timestamp += 1
        ###################################################################################

        curr_point = self.__game.total_point() - prev_point
        self.__point = self.__point * AiController.GAMMA + (curr_point / 100.0)

        self.log("TIME: {}, GAME SCORE: {}".format(self.__timestamp, self.__point))

        reward = (curr_point) / 100.0
        state_prime = self.current_state()

        self.push_history({
            "state": state,
            "action": action,
            "reward": reward,
            "state_prime": state_prime,
        })

        if self.__timestamp > AiController.OBSERVE_FRAME:
            minibatch = self.random_history(AiController.BATCH)

            for i in range(0, len(minibatch)):
                data = minibatch[i]
                state = data['state']
                action = int(data['action'])
                reward = data['reward']
                state_prime = data['state_prime']
                
                Q_value = self.__network(state)
                Q_sa = self.__network(state_prime)
                self.__train_inputs[i : i + 1] = state
                self.__train_targets[i] = Q_value.data
                self.__train_targets[i, action] = reward + AiController.GAMMA * self.xp.max(Q_sa.data)

            x = self.__network(self.__train_inputs.astype(self.xp.float32))
            t = self.__train_targets.astype(self.xp.float32)
            loss = F.mean_squared_error(x, t)
            self.__optimizer.update(lambda: loss)
            self.__loss_average.add(loss.data)
            print("Average LOSS: {}".format(self.__loss_average.average()))

            if self.__timestamp % 10000 == 0:
                self.log('save model!')
                self.save()

