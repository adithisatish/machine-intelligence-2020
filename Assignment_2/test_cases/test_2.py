from PESU_MI_0027_0150_0204 import *

cost = [
    [0, 0, 0, 0],
	[0, 0, 5, 10],
	[0, -1, 0, 5],
	[0, -1, -1, 0]]

heuristic = [0, 0, 0, 0]

print("----------------------------------TESTS----------------------------------------")
print(tri_traversal(cost, heuristic, 1, [3]))

