# -*- coding: utf-8 -*-
from position import Position
from bullet import Bullet

class Player():

    def __init__(self, game):
        self.position = Position(0, 0)
        self.is_dead = False
        self.__mesh = "僕"
        self.__update_prior = 3
        self.__game = game

    def update(self):
        """
        Player must be controlled by controller
        Player move owing to it
        """
        pass

    def move_left(self):
        self.position.x -= 1

    def move_right(self):
        self.position.x += 1

    def shoot_bullet(self):
        bullet = Bullet(Position(self.position.x, self.position.y + 1), self.__game)
        self.__game.add(bullet)

    def mesh(self):
        return self.__mesh

    def update_prior(self):
        return self.__update_prior

    def state_index(self):
        return 0
