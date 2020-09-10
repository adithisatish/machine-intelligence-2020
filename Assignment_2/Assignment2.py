'''
Function tri_traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''


def getMinimumPath(priorityQueue):
    minPath = priorityQueue[0]

    for i in priorityQueue[1:]:
        if minPath[1] > i[1]:
            minPath = i
        elif minPath[1] == i[1]:
            if minPath[0] > i[0]:
                minPath = i
    
    return minPath

def uniformCostSearch(cost, start_point, goals):
    visited = [0]*len(cost[0])
    priorityQueue = []

    visited[start_point] = 1
    priorityQueue.append((start_point,0))
    
    i = start_point
    cur_path = (start_point,0)
    
    while(len(priorityQueue) != 0):
        if cur_path in priorityQueue:
            priorityQueue.remove(cur_path)

        for j in range(1, len(cost[i])):
            if cost[i][j] <= 0:
                continue
            if visited[j] == 1:
                continue
            priorityQueue.append((j,cur_path[1]+cost[i][j]))
       
        while True:
            cur_path = getMinimumPath(priorityQueue)

            if visited[cur_path[0]] == 1:
                priorityQueue.remove(cur_path)
            else:
                break
            
        visited[cur_path[0]] = 1
        i = cur_path[0]
        priorityQueue.remove(cur_path)

        if cur_path[0] in goals:
            return cur_path
    
    return cur_path #Is this what should be returned by default?


def tri_Traversal(cost, heuristic, start_point, goals):
    l = []


    # t1 <= DFS_Traversal
    # t2 <= UCS_Traversal
    # t3 <= A_star_Traversal
    #print("En")
    ufs = uniformCostSearch(cost,start_point,goals)
    print("Goal and Minimum Cost:",ufs)
    #l.append(t1)
    #l.append(t2)
    #l.append(t3)
    return l