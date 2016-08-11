from game import Game, Player, Enemy, Position
import time


player = Player()
game = Game()

player.position.x = int(Game.DISPLAY_WIDTH / 2)

for y in range(Game.DISPLAY_HEIGHT - 4, Game.DISPLAY_HEIGHT - 2):
    for x in range(2, Game.DISPLAY_WIDTH - 2):
        game.add(Enemy(Position(x, y)))

game.add(player)

while True:
    game.render()
    time.sleep(0.2)