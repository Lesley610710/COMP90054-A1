# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from cmath import inf
from itertools import accumulate
from json.encoder import INFINITY
from queue import PriorityQueue
import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# Please DO NOT change the following code, we will use it later
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    myPQ = util.PriorityQueue()
    startState = problem.getStartState()
    startNode = (startState, '',0, [])
    myPQ.push(startNode,heuristic(startState,problem))
    visited = set()
    best_g = dict()
    while not myPQ.isEmpty():
        node = myPQ.pop()
        state, action, cost, path = node
        if (not state in visited) or cost < best_g.get(state):
            visited.add(state)
            best_g[state]=cost
            if problem.isGoalState(state):
                path = path + [(state, action)]
                actions = [action[1] for action in path]
                del actions[0]
                return actions
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                myPQ.push(newNode,heuristic(succState,problem)+cost+succCost)
    util.raiseNotDefined()


def enforcedHillClimbing(problem, heuristic=nullHeuristic):
    """
    Local search with heuristic function.
    You DO NOT need to implement any heuristic, but you DO have to call it.
    The heuristic function is "manhattanHeuristic" from searchAgent.py.
    It will be pass to this function as second argument (heuristic).
    """
    "*** YOUR CODE HERE FOR TASK 1 ***"
    def improve(ininode):    
        state0, action0, cost0, path0 = ininode
        queue = util.Queue()    
        queue.push(ininode)
        visited = set()
        while not queue.isEmpty():
            state, action, cost, path = queue.pop()
            if not state in visited:
                visited.add(state)
                if heuristic(state, problem) < heuristic(state0, problem):
                    return (state, action, cost, path)
                for succ in problem.getSuccessors(state):
                    succState, succAction, succCost = succ
                    newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                    queue.push(newNode)
    
    startState = problem.getStartState()
    endNode = (startState, '', 0, [])
    endState = startState
    while not problem.isGoalState(endState):
        endNode =  improve(endNode)
        state, action, cost, path = endNode
        endState = state
    path = path + [(state, action)]
    actions = [action[1] for action in path]
    del actions[0]
    return actions

from math import inf as INF   
def bidirectionalAStarEnhanced(problem, heuristic=nullHeuristic, backwardsHeuristic=nullHeuristic):
    
    """
    Bidirectional global search with heuristic function.
    You DO NOT need to implement any heuristic, but you DO have to call them.
    The heuristic functions are "manhattanHeuristic" and "backwardsManhattanHeuristic" from searchAgent.py.
    It will be pass to this function as second and third arguments.
    You can call it by using: heuristic(state,problem) or backwardsHeuristic(state,problem)
    """
    "*** YOUR CODE HERE FOR TASK 2 ***"
    # The problem passed in going to be BidirectionalPositionSearchProblem    
    of = util.PriorityQueue()
    ofstartState = problem.getStartState()
    ofstartNode = (ofstartState, '', 0, [])
    of.push(ofstartNode,0 + heuristic(ofstartState,problem) + 0 - 0 - backwardsHeuristic(ofstartState, problem))
    ofvisited = set()
    ofbest_g = dict()

    ob = util.PriorityQueue()
    for obgoalState in problem.getGoalStates():
        obstartNode = (obgoalState  , '', 0, [])
        ob.push(obstartNode, 0 + backwardsHeuristic(obgoalState,problem) + 0 - 0 - heuristic(obgoalState, problem))
    obvisited = set()
    obbest_g = dict()

    L = 0
    U = INFINITY
    pi = []
    x = 0 #0 = forward, 1 = backward

    while not of.isEmpty() and not ob.isEmpty() :
        ofMin = of.getMinimumPriority()
        obMin = ob.getMinimumPriority()
        L = (ofMin + obMin) / 2
        if x == 0: #forward
            n = of.pop()
            state, action, cost, path = n                                                                                                 
            if (not str(state) in ofvisited) or (cost < ofbest_g[str(state)][0]): #if its the 1st time visit the state or it's been visited and smaller than previous 
                ofvisited.add(str(state)) #update the visited
                ofbest_g[str(state)] = (cost, path)
            if str(state) in obbest_g and (ofbest_g[str(state)][0] + obbest_g[str(state)][0]) < U:
                U = ofbest_g[str(state)][0] + obbest_g[str(state)][0]
                pi = ofbest_g[str(state)][1] + list(reversed(obbest_g[str(state)][1]))
            if L >= U:
                return pi
            for succ in problem.getSuccessors(state):
                succState, succAction, succCost = succ
                if not str(succState) in ofvisited: #not update visted state
                    newNode = (succState, succAction, cost + succCost, path + [succAction])
                    bxn_succ = (heuristic(succState,problem) + cost + succCost) + (ofbest_g[str(state)][0] - backwardsHeuristic(state, problem)) #fxn + dxn
                    of.push(newNode, bxn_succ)
        else: #backward
            n = ob.pop()
            state, action, cost, path = n
            if (not str(state) in obvisited) or (cost < obbest_g[str(state)][0]): #if its the 1st time visit the state or it's been visited and smaller than previous 
                obvisited.add(str(state)) #update the visited
                obbest_g[str(state)] = (cost, path)
            if str(state) in ofbest_g and (obbest_g[str(state)][0] + ofbest_g[str(state)][0]) < U:
                U = obbest_g[str(state)][0] + ofbest_g[str(state)][0]
                # print(obbest_g[str(state)][0])
                pi = ofbest_g[str(state)][1] + list(reversed(obbest_g[str(state)][1]))
            if L >= U:
                return pi
            for succ in problem.getBackwardsSuccessors(state):
                succState, succAction, succCost = succ
                if not str(succState) in obvisited: #not update visted state
                    newNode = (succState, succAction, cost + succCost, path + [succAction])
                    bxn_succ = (backwardsHeuristic(succState,problem) + cost + succCost) + (obbest_g[str(state)][0] - heuristic(state, problem)) 
                    ob.push(newNode, bxn_succ)
        #print(of.isEmpty(),ob.isEmpty())
        ofMin = of.getMinimumPriority()
        obMin = ob.getMinimumPriority()
        if ofMin > obMin:
            x = 1
        else:
            x = 0
    #return pi

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


ehc = enforcedHillClimbing
bae = bidirectionalAStarEnhanced


