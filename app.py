from game import *
import time
import random
import matplotlib.pyplot as plt
import numpy as np

game = Game()
enemy_pool = EnemyPool()

player = Player(game)
controller = AiController(game, player)

player.position.x = int(Game.DISPLAY_WIDTH / 2)

# for y in range(Game.DISPLAY_HEIGHT - 4, Game.DISPLAY_HEIGHT - 2):
#     for x in range(2, Game.DISPLAY_WIDTH - 2):
#         game.add(Enemy(Position(x, y)))

game.add(player)

# Matplot Lib
fig, ax = plt.subplots(1, 1)
lines, = ax.plot([0], [0])
plt.xlabel("Time stamp")
plt.ylabel("Reward")

frame_count = 0
point_history = []
prev_frame_total_point = 0
reward_per_100frames = 0

while True:
    # Spawn enemy randomly!
    if random.uniform(0, 1) < 0.1:
        e_x = random.randint(0, Game.DISPLAY_WIDTH - 1)
        e_y = random.randint(Game.DISPLAY_HEIGHT / 2, Game.DISPLAY_HEIGHT - 1)
        obj = game.get_by_position(e_x, e_y)
        if obj == None:
            game.add(Enemy(Position(e_x, e_y), game))

    # Move player and go next frame
    controller.next()

    curr_frame_total_point = game.total_point()
    reward_per_100frames += (curr_frame_total_point - prev_frame_total_point)
    prev_frame_total_point = curr_frame_total_point

    # Plot the reward per 100 frames
    if frame_count % 100 == 0:
        point_history.append(reward_per_100frames)
        lines.set_data(range(len(point_history)), point_history)
        ax.set_xlim((0, len(point_history)))
        ax.set_ylim((np.min(point_history), np.max(point_history)))
        plt.waitforbuttonpress(timeout=.01)
        reward_per_100frames = 0

    frame_count += 1

