from game import *
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Invader Game with Deep Reinforcement Learning!')
parser.add_argument('-m','--mode', help='train / run', required=True)
parser.add_argument('-o','--output', help='game / data', default="game")
parser.add_argument('-p','--plot', help='on / off', default="off")
parser.add_argument('-s','--strategy', help='greedy / egreedy / softmax', default="softmax")
parser.add_argument('-g','--gpu', help='device number', default=-1)

args = parser.parse_args()

game = Game()
enemy_pool = EnemyPool()

player = Player(game)
controller = AiController(game,
                          player,
                          args.strategy,
                          args.mode == "train",
                          args.output == "game",
                          args.gpu)

player.position.x = int(Game.DISPLAY_WIDTH / 2)

# for y in range(Game.DISPLAY_HEIGHT - 4, Game.DISPLAY_HEIGHT - 2):
#     for x in range(2, Game.DISPLAY_WIDTH - 2):
#         game.add(Enemy(Position(x, y)))

game.add(player)

# Matplot Lib
if args.plot == 'on':
    fig, ax = plt.subplots(1, 1)
    lines, = ax.plot([0], [0])
    plt.xlabel("Time stamp")
    plt.ylabel("Reward")

frame_count = 0
point_history = []
prev_frame_total_point = 0
reward_per_100frames = 0
enemy_count_per100frames = 0

def console_game():
    game.render()

def console_data():
    print("{},{}".format(frame_count, reward_per_100frames - (enemy_count_per100frames * 100)))

def empty_func():
    pass

if args.output == 'game':
    console = console_game
elif args.output == 'data':
    console = console_data
else:
    console = empty_func

def plot_on():
    lines.set_data(range(len(point_history)), point_history)
    ax.set_xlim((0, len(point_history)))
    ax.set_ylim((np.min(point_history) - 100, np.max(point_history) + 100))
    plt.waitforbuttonpress(timeout=.01)

if args.plot == 'on':
    plot = plot_on
else:
    plot = empty_func

while True:
    # Spawn enemy randomly!
    if random.uniform(0, 1) < 0.1:
        e_x = random.randint(0, Game.DISPLAY_WIDTH - 1)
        e_y = random.randint(Game.DISPLAY_HEIGHT / 2, Game.DISPLAY_HEIGHT - 1)
        obj = game.get_by_position(e_x, e_y)
        if obj == None:
            game.add(Enemy(Position(e_x, e_y), game))
            enemy_count_per100frames += 1

    # Move player and go next frame
    controller.next()

    curr_frame_total_point = game.total_point()
    reward_per_100frames += (curr_frame_total_point - prev_frame_total_point)
    prev_frame_total_point = curr_frame_total_point

    frame_count += 1

    # Plot the reward per 100 frames
    if frame_count % 100 == 0:
        # This value converges to 0 if the AI is perfect
        point_history.append(reward_per_100frames - (enemy_count_per100frames * 100))
        plot()
        reward_per_100frames = 0
        enemy_count_per100frames = 0

    console()

