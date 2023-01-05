import numpy as np
from numpy.random import uniform
from time import sleep
from os import system

width = 10
height = 10
number_of_observations = 200


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

def quantize(p, buckets=4, max_buckets=100, boundry=0.0001):

        bucket_value = ((p * max_buckets) // (max_buckets // buckets)) * (max_buckets // buckets)
        new_p = bucket_value / max_buckets

        return min(max(new_p, boundry), 1 - boundry)

# beleif updating method

def update(observation):
    global probs

    left = observation[0]
    above = observation[1]

    for row in range(height):
        for col in range(width):

            # joint distribution p(above, left | <x, y>) = p(above | <x, y>) * p(left | <x, y>)
            # = p(above | y) * p(left | x)

            prow = row / height if above else 1 - (row / height)
            pcol = col / width if left else 1 - (col / width)
            pob = quantize(fuzz(prow * pcol), buckets=50)

            probs[row][col] *= pob

    probs = probs / probs.sum()



def main():
    for i in range(number_of_observations):
        sleep(0.1)
        print('\033[H\033[2J') # clear
        print('Infering Position of Ball')
        print(f'observation #{i}\n')
        print(probs.round(2))
        update(positions[i])

    print('true position: ', position)

