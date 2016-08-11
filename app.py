from game import Game, Player
import time


player = Player()
game = Game()

game.add(player)

while True:
    game.render()
    time.sleep(0.2)
