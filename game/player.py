# -*- coding: utf-8 -*-
from position import Position
from bullet import Bullet

class Player():

    def __init__(self, controller, game):
        self.position = Position(0, 0)
        self.__mesh = "僕"
        self.__update_prior = 3
        self.__controller = controller
        self.__game = game

    def update(self):
        """
        Player must be controlled by controller
        Player move owing to it
        """
        command = self.__controller.get_command()
        if command == 0:
            self.position.x -= 1
        elif command == 1:
            self.position.x += 1
        elif command == 2:
            self.shoot_bullet()

    def shoot_bullet(self):
        bullet = Bullet(Position(self.position.x, self.position.y + 1))
        self.__game.add(bullet)

    def mesh(self):
        return self.__mesh

    def update_prior(self):
        return self.__update_prior

