# -*- coding: utf-8 -*-
class Bullet():
    SYMBOL = "弾"
    UPDATE_PRIOR = 10

    def __init__(self, first_position):
        self.position = first_position
        pass

    def update(self):
        self.position.y += 1

    def mesh(self):
        return Bullet.SYMBOL

    def update_prior(self):
        return Bullet.UPDATE_PRIOR
