# infering the position of a ball in a grid

# We have an NxM grid and place a ball in one cell at random.
# Each cell has an equal chance of being picked.
#
# Our goal is to figure out where the ball is most likely to be, given information about whether a
# randomly chosen cell is north, south, east or west of the ball


import numpy as np
from numpy.random import uniform
from time import sleep
from os import system


width = 20
height = 20
number_of_observations = 200

np.set_printoptions(threshold=8000, linewidth=200)

# ball position

position = uniform((width, height)).astype(int)


# generate observations

def information_about_position_relative_to_ball(pos):
    return [pos[0] < position[0], pos[1] < position[1]]


positions = np.array([

    information_about_position_relative_to_ball(uniform((width, height)).astype(int))
    for i in range(number_of_observations)

])


# assign priors
# objectively, each cell should have an equal prior probability because we know they were picked
# uniformly at random

probs = np.ones((width, height)) / (width * height)


# fuzzy likelihood

fuzz_mean = 0
fuzz_stddev = 0

def fuzz(p):

    p = p + fuzz_mean + fuzz_stddev * np.random.randn()
    p = max(min(p, 1 - 0.0001), 0.0001)

    return p


# snap a probability to one of n buckets.
# This represents reducing certaintiy about the *exact* value.
# For example, with two buckets, this equivalent to stating whether the probability is
# above or below 0.5 (i.e. likely or unlikely).
# With four buckets, it's more akin to (very unlikely, unlikely, likely, very likely).
#

# outer_boundry - essentially the maximum confidence
# inner_boundry - essentially the minimum confidence


def quantize(p, buckets=4, max_buckets=100, outer_boundry=0.0001, inner_boundry=0.5):

    bucket_value = ((p * max_buckets) // (max_buckets // buckets)) * (max_buckets // buckets)

    new_p = bucket_value / max_buckets
    new_p = min(max(new_p, outer_boundry), 1 - outer_boundry)
    new_p = max(inner_boundry, new_p) if new_p > 0.5 else min(1 - inner_boundry, new_p)

    return new_p


# beleif updating method

def update(observation, buckets=500, max_buckets=500, inner_boundry=0.5, outer_boundry=0.000001):
    global probs

    left = observation[0]
    above = observation[1]

    for row in range(height):
        for col in range(width):

            # joint distribution p(above, left | <x, y>) = p(above | <x, y>) * p(left | <x, y>)
            # = p(above | y) * p(left | x)

            prow = row / height if above else 1 - (row / height)
            pcol = col / width if left else 1 - (col / width)
            pob = quantize(fuzz(prow * pcol), buckets=buckets, max_buckets=max_buckets, inner_boundry=inner_boundry, outer_boundry=outer_boundry)

            probs[row][col] *= pob

    probs = probs / probs.sum()


convergence_threshold = 0.2

def main( buckets=500, max_buckets=500, inner_boundry=0.5, outer_boundry=0.000001):
    for i in range(number_of_observations):
        #sleep(0.1)
        #print('\033[H\033[2J') # clear
        #print('Infering Position of Ball')
        #print(f'observation #{i}\n')
        #print(probs.round(2))
        update(positions[i],  buckets=buckets, max_buckets=max_buckets, inner_boundry=inner_boundry, outer_boundry=outer_boundry)

        mean_row = 0
        mean_col = 0

        for y, row in enumerate(probs):
            for x, col in enumerate(row):

                p = probs[y, x]
                mean_row += p * y
                mean_col += p * x

        mse = np.mean(np.square(position - [mean_col, mean_row]))
        #print('MSE: ', mse.round(3))

        if mse < convergence_threshold:
            #print(f'[+] Converged after {i + 1} observations.')
            #print(f'[-] Ground truth: {position}')
            #print(f'[-] Posterior mean: {np.array([mean_col, mean_row]).round(3)}, mse = {mse}')
            break

    return i + 1, mse


def exp(buckets=500, max_buckets=500, inner_boundry=0.5, outer_boundry=0.000001):
    global position
    global positions
    global probs

    position = uniform((width, height)).astype(int)
    probs = np.ones((width, height)) / (width * height)

    positions = np.array([

        information_about_position_relative_to_ball(uniform((width, height)).astype(int))
        for i in range(number_of_observations)

    ])

    return main(buckets=buckets, max_buckets=max_buckets, inner_boundry=inner_boundry, outer_boundry=outer_boundry)



exp_1 = [exp(inner_boundry=0.8)[0] for _ in range(100)]
exp_2 = [exp(inner_boundry=0.5)[0] for _ in range(100)]
