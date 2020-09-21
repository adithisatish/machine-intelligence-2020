#PESU-MI_0027_0150_0204

#Helper Functions:

def stateExceptions(n,start_point,goals): #Here n represents the number of vertices
    if start_point < 1 or start_point > n:
        return -1
    if start_point in goals:
        return 1

    return 0

def checkValidity(heuristic,cost):
    nodes = len(cost[0])
    
    if len(heuristic)!=nodes:
        return 0

    for i in heuristic:
        if i <0:
            return 0
    
    return 1

def getMinimumPath(priorityQueue,heuristic):
    minPath = priorityQueue[0]
 
    for i in priorityQueue[1:]:
        if minPath[1] + heuristic[minPath[0]] > i[1] + heuristic[i[0]]: #The first comparison done with respect to costs
            minPath = i
        elif minPath[1] + heuristic[minPath[0]] == i[1] + heuristic[i[0]]:
            if minPath[0] >= i[0]: #The second comparison done to maintain lexicographical order
                minPath = i
 
    return minPath
    
#Main functions:
 
def A_star_Traversal(cost, heuristic, start_point, goals):
    validHeuristic = checkValidity(heuristic,cost)
 
    # if validHeuristic == 0:
    #     print("The heuristic is not valid")
    #     return []
 
    return UCS_Traversal(cost = cost,start_point=start_point,goals=goals, heuristic=heuristic, ucs_astar=1)
 
    
 
def UCS_Traversal(cost, heuristic, start_point, goals, ucs_astar):
    
    #The ucs_astar parameter here determines whether the algorithm is UCS or A*
    #i.e whether to consider the heuristic or not.
    #If ucs_astar = 0 => UCS algorithm, and all elements of the heuristic array are set to 0
    #If ucs_astar = 1 => A* algorithm 
 
    if ucs_astar == 0:
        heuristic = [0]*len(cost[0])
    
 
    visited = [0]*len(cost[0]) #The Explored Set: keeps track of all the nodes that have already been visited
    priorityQueue = [] #In UCS, the Frontier is a Priority Queue that is dependent on the minimum path cost
 
    pathTrack = {start_point:0} #To keep track of the child and corresponding parent nodes
    res = [] # The resulting list of nodes which represents the minimum cost path
 
    exception = stateExceptions(len(visited),start_point,goals)
 
    if exception == -1:
        return []
    
    if exception == 1:
        return [start_point]
 
    visited[start_point] = 1
    priorityQueue.append((start_point,0,0))
 
    i = start_point
    
    cur_path = (start_point,0,i) #Updated everytime with the current node chosen, the cost upto that node, and said node's parent
    
    while(len(priorityQueue) != 0):
        if cur_path in priorityQueue: #Remove the initial node (i.e start state) from the frontier
            priorityQueue.remove(cur_path)
 
        for j in range(1, len(cost[i])):
            if cost[i][j] < 0: 
                continue
            if visited[j] == 1: 
                continue
            priorityQueue.append((j,(cur_path[1]+cost[i][j]),i)) #Calculating new cost of path for each of the unvisited children and inserting to frontier
 
        while len(priorityQueue)!=0:
            
            cur_path = getMinimumPath(priorityQueue,heuristic) #Returns the minimum path node of all nodes in frontier
 
            if visited[cur_path[0]] == 1:
                priorityQueue.remove(cur_path) #Repeat until a minimum path node is found that is unvisited by removing all visited minimum path nodes
            else:
                break
       
        pathTrack[cur_path[0]] = cur_path[2] #Update the child:parent pair in order to keep track of path taken
 
        visited[cur_path[0]] = 1 #Add current path node to explored set
        i = cur_path[0]
        
        if cur_path[0] in goals: #A minimum path goal state has been achieved
            child = cur_path[0]
            
            while child !=0: #Traverse backwards from goal to start state in order to find the path taken
                res.append(child) 
                child = pathTrack[child]
            break
    
    if len(priorityQueue)==0 and cur_path[0] not in goals: #To handle the case where all the goal states are unreachable
        return []
 
    res.reverse() #To get the correct traversal order i.e start state to goal
    return res

def DFS_Traversal(cost,start_point, goals): #add your parameters 
    visited=[0]*len(cost[0]) # To keep track of visited nodes.
    stack=[]

    exception = stateExceptions(len(visited)-1,start_point,goals)

    if exception == -1:
        return []
    
    if exception == 1:
        return [start_point]

    visited[start_point]=1
    stack.append(start_point)
    while stack:
        cur=stack[-1] #cur points to the last element in the stack
      #  print(cur)
        if cur in goals:
       #     print(stack)
            return stack
        if visited[cur]!=1:
            visited[cur]=1
        flag=1
        for i in range(1,len(cost[cur])):
            if cost[cur][i]>=0 and visited[i]!=1:
                stack.append(i)
                flag=0
                break
        if flag:
                stack.pop()
    return stack


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
1<=m<=n
cost[n][n] , heuristic[n][n], start_point, goals[m]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''

def tri_traversal(cost, heuristic, start_point, goals):
    l = []

    t1 = DFS_Traversal(cost, start_point, goals)
    #print(t1)
    t2 = UCS_Traversal(cost, heuristic,start_point,goals, 0)
    #print(t2)
    t3 = A_star_Traversal(cost,heuristic,start_point,goals)
    #print(t3)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l
