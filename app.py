from game import *
import time
import random

game = Game()
enemy_pool = EnemyPool()

player = Player(game)
controller = AiController(game, player)

player.position.x = int(Game.DISPLAY_WIDTH / 2)

# for y in range(Game.DISPLAY_HEIGHT - 4, Game.DISPLAY_HEIGHT - 2):
#     for x in range(2, Game.DISPLAY_WIDTH - 2):
#         game.add(Enemy(Position(x, y)))

game.add(player)

while True:
    if random.uniform(0, 1) < 0.1:
        e_x = random.randint(0, Game.DISPLAY_WIDTH - 1)
        e_y = random.randint(Game.DISPLAY_HEIGHT / 2, Game.DISPLAY_HEIGHT - 1)
        obj = game.get_by_position(e_x, e_y)
        if obj == None:
            game.add(Enemy(Position(e_x, e_y)))
    controller.next()
    time.sleep(0.2)
