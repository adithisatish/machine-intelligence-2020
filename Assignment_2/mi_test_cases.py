# I take no responsibility for the correctness of the test cases.
# Test cases may included goals unreachable from start point, in which case the result for each search is an empty list
# ~ST dm for more test cases :3
from PESU_MI_0027_0150_0204 import *

file = open("D:\\PESU\Sem 5\Machine Intelligence\MI_Assignment\Assignment_2\mi_test_cases.txt", "r")
test_num = 1
failed = []
file.readline()
file.readline()
file.readline()
for i in range(10):
	size = int(file.readline())
	cost = [list(map(int, file.readline().split())) for x in range(size)]
	for j in range(10):
		file.readline()
		file.readline()
		heuristic = list(map(int, file.readline().split()))
		start_point = int(file.readline())
		goals = list(map(int, file.readline().split()))
		print('Goal States:', goals)
		correct_answer = [list(map(int, s.split())) for s in file.readline().split(",")]
		correct_answer.pop()
		your_answer = tri_traversal(cost, heuristic, start_point, goals)
		if(your_answer == correct_answer):
			print("Test",test_num,": PASSED!")
		else:
			print("Test",test_num,": FAILED!")
			print("Your answer :", your_answer)
			print("Correct answer :", correct_answer)
			failed.append(test_num)
		test_num += 1
	file.readline()
	print("================================================================")

if(not failed):
	print("ALL TEST CASES PASSED!")
else:
	print("ono FOLLOWING TEST CASES FAILED!")
	print(*failed)
