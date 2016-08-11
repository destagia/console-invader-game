
class Enemy():
    def __init__(self, first_position):
        self.position = first_position
        # Enemy moves owing to move_couner
        self.__move_counter = 0

    def update(self):
        if self.__move_counter >= 5:
            self.position.x -= 1
            self.__move_counter = 0
        self.__move_counter += 1

