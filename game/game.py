# -*- coding: utf-8 -*-
class Game():

    DISPLAY_WIDTH = 11
    DISPLAY_HEIGHT = 15

    def __init__(self):
        self.__game_objects = []
        self.__hr = " " + "ー" * Game.DISPLAY_WIDTH

    def add(self, game_object):
        self.__game_objects.append(game_object)
        self.__game_objects = sorted(self.__game_objects, key=lambda go: go.update_prior(), reverse=True)

    def render(self):
        print(self.__hr)
        for line in reversed(self.__create_filled_display()):
            line_str = "|"
            for game_object in line:
                if game_object == None:
                    line_str += "　"
                else:
                    line_str += game_object.mesh()
            line_str += "|"
            print(line_str)
        print(self.__hr)

    def game_objects(self):
        return self.__game_objects

    def __create_filled_display(self):
        display = self.__create_white_display()
        self.__game_objects = [go for go in self.__game_objects if not go.is_dead]
        for go in self.__game_objects:
            go.update()
            self.__adjust_position(go.position)
            display[go.position.y][go.position.x] = go
        return display

    def __create_white_display(self):
        display = []
        for _ in range(0, Game.DISPLAY_HEIGHT):
            line = []
            for _ in range(0, Game.DISPLAY_WIDTH):
                line.append(None)
            display.append(line)
        return display


    def __adjust_position(self, position):
        if position.x < 0:
            position.x = 0
        elif position.x >= Game.DISPLAY_WIDTH:
            position.x = Game.DISPLAY_WIDTH - 1
        if position.y < 0:
            position.y = 0
        elif position.y >= Game.DISPLAY_HEIGHT:
            position.y = Game.DISPLAY_HEIGHT - 1