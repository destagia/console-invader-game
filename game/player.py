# -*- coding: utf-8 -*-
from position import Position

class Player():

    def __init__(self):
        self.position = Position(0, 0)
        self.__mesh = "僕"
        self.__update_prior = 3

    def update(self):
        """
        Player must be controlled by controller
        Player move owing to it
        """
        pass

    def mesh(self):
        return self.__mesh

    def update_prior(self):
        return self.__update_prior

