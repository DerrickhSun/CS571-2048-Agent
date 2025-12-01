from env_2048 import Game2048Env, encode_state
import numpy as np

env = Game2048Env()

state = env.reset()
print("Initial state:", state)
print("Length:", len(state))

print("Matrix form:")
for row in env.matrix:
    print(row)

next_state, reward, done = env.step(0)  # action 0 = up

print("Next state:", next_state)
print("Reward:", reward)
print("Done:", done)

print("Matrix after step:")
for row in env.matrix:
    print(row)

env.matrix = [
    [2, 4, 8, 16],
    [32, 64, 128, 256],
    [512, 1024, 2, 4],
    [8, 16, 32, 64]
]

s, r, d = env.step(0)
print("reward for invalid move:", r)
print("done:", d)

env.matrix = [
    [2, 2, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]

s, r, d = env.step(2)  # action 2 = left
print("Reward should be 4:", r)

env.matrix = [
    [1024, 1024, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]

s, r, d = env.step(2)  # left
print("reward:", r)
print("done:", d)

env.reset()
before = sum(env.matrix[i][j] != 0 for i in range(4) for j in range(4))

env.step(0)
after = sum(env.matrix[i][j] != 0 for i in range(4) for j in range(4))

print("tiles before:", before)
print("tiles after:", after)
