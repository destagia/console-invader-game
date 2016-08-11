# -*- coding: utf-8 -*-

class Enemy():
    def __init__(self, first_position, game):
        self.position = first_position
        self.is_dead = False
        # Enemy moves owing to move_couner
        self.__move_counter = 0
        self.__update_prior = 0
        self.__mesh = "敵"
        self.__game = game

    def update(self):
        if self.__move_counter >= 5:
            self.position.y -= 1
            if self.position.y == -1:
                self.is_dead = True
                self.__game.add_point(-100)
                return
            self.__move_counter = 0
        self.__move_counter += 1

    def update_prior(self):
        return self.__update_prior

    def mesh(self):
        return self.__mesh

    def state_value(self):
        return 1.0

