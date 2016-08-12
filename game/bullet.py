# -*- coding: utf-8 -*-
from enemy import Enemy
from game import Game

class Bullet():
    SYMBOL = "弾"
    UPDATE_PRIOR = 10

    def __init__(self, first_position, game):
        self.position = first_position
        self.is_dead = False
        self.__game = game

    def update(self):
        self.position.y += 1
        if self.position.y >= Game.DISPLAY_HEIGHT - 1:
            self.is_dead = True
            return

        for go in self.__game.game_objects():
            if go != None and isinstance(go, Enemy) and go.position == self.position:
                # go is an Enemy instance, so it has is_dead property
                go.is_dead = True
                self.is_dead = True
                self.__game.add_point(100)

    def mesh(self):
        return Bullet.SYMBOL

    def update_prior(self):
        return Bullet.UPDATE_PRIOR

    def state_index(self):
        return 1
