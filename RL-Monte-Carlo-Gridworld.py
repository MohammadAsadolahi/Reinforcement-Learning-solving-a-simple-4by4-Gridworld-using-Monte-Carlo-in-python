import numpy as np
import copy

class GridWorld:
    def __init__(self):
        # S O O O
        # O O O *
        # O * O O
        # O * 0 T
        self.currentState = None
        self.qTable = None
        self.actionSpace = ('U', 'D', 'L', 'R')
        self.actions = {
            (0, 0): ('D', 'R'),
            (0, 1): ('L', 'D', 'R'),
            (0, 2): ('L', 'D', 'R'),
            (0, 3): ('L', 'D'),
            (1, 0): ('U', 'D', 'R'),
            (1, 1): ('U', 'L', 'D', 'R'),
            (1, 2): ('U', 'L', 'D', 'R'),
            (1, 3): ('U', 'L', 'D'),
            (2, 0): ('U', 'D', 'R'),
            (2, 1): ('U', 'L', 'D', 'R'),
            (2, 2): ('U', 'L', 'D', 'R'),
            (2, 3): ('U', 'L', 'D'),
            (3, 0): ('U', 'R'),
            (3, 1): ('U', 'L', 'R'),
            (3, 2): ('U', 'L', 'R')
        }

        self.rewards = {(3, 3): 5, (1, 3): -2, (2, 1): -2, (3, 1): -2}
        self.initialQtable()
        self.explored = 0
        self.exploited = 0

    def getRandomPolicy(self):
        policy = {}
        for state in self.actions:
            policy[state] = np.random.choice(self.actions[state])
        return policy

    def initialQtable(self):
        self.qTable = {}
        for state in self.actions:
            self.qTable[state]={}
            for move in self.actions[state]:
                self.qTable[state][move]=0
        print(self.qTable)

    def printQtable(self):
        print(self.qTable)

    def getCurrentState(self):
        if not self.currentState:
            self.currentState = (0, 0)
        return self.currentState

    def updateQtable(self, newQ):
        for state in self.qTable:
            for action in self.qTable[state]:
                self.qTable[state][action] = self.qTable[state][action]+(0.05*(newQ[state][action]-self.qTable[state][action]))
    def printPolicy(self, policy):
        line = ""
        counter = 0
        for item in policy:
            line += f" | {policy[item]} | "
            counter += 1
            if counter > 3:
                print(line)
                print("----------------------------")
                counter = 0
                line = ""
        print(line)
        print("----------------------------")

    def is_terminal(self, s):
        return s not in self.actions

    def chooseAction(self, state, policy, exploreRate):
        if exploreRate > np.random.rand():
            self.explored += 1
            return np.random.choice(self.actions[state])
        self.exploited += 1
        return policy[state]

    def greedyChoose(self, state, values):
        actions = self.actions[state]
        stateValues = []
        for item in actions:
            i, j = zip(state)
            row = int(i[0])
            column = int(j[0])
            if item == 'U':
                row -= 1
            elif item == 'D':
                row += 1
            elif item == 'L':
                column -= 1
            elif item == 'R':
                column += 1
            if (row, column) in values:
                stateValues.append(values[(row, column)])
        return actions[np.argmax(stateValues)]

    def getActionReward(self, state, action):
        i, j = zip(state)
        row = int(i[0])
        column = int(j[0])
        if action == 'U':
            row -= 1
        elif action == 'D':
            row += 1
        elif action == 'L':
            column -= 1
        elif action == 'R':
            column += 1
        if (row, column) in self.rewards:
            return self.rewards[(row, column)]
        else:
            return 0

    def move(self, state, policy, exploreRate):
        action = self.chooseAction(state, policy, exploreRate)
        i, j = zip(state)
        row = int(i[0])
        column = int(j[0])
        if action == 'U':
            row -= 1
        elif action == 'D':
            row += 1
        elif action == 'L':
            column -= 1
        elif action == 'R':
            column += 1
        if (row, column) in self.rewards:
            return action, (row, column), self.rewards[(row, column)]
        return action, (row, column), 0
    
enviroment = GridWorld()
policy = enviroment.getRandomPolicy()
enviroment.printPolicy(policy)

explorationRate=0.1

for i in range(250):
    estimatedQ = copy.deepcopy(enviroment.qTable)
    for state in estimatedQ:
        for action in estimatedQ[state]:
            estimatedQ[state][action] = 0
    collectedSampls = 0
    for j in range(5000):
        counter = 0
        trajectory = []
        state = enviroment.getCurrentState()
        while True:
            action, nextState, reward = enviroment.move(state, policy, explorationRate)
            trajectory.append(((state, action), reward))
            # print(f"j:{i}  i:{j} state{state} move:{policy[state]} r:{reward} nextS:{nextState} ")
            counter += 1
            if counter == 20 or enviroment.is_terminal(nextState):
                break
            state = nextState
        # print(trajectory)
        collectedSampls += 1
        rewards = 0
        for item in reversed(trajectory):
            q,reward=zip(item)
            rewards +=0.9*(reward[0])
            estimatedQ[q[0][0]][q[0][1]] = estimatedQ[q[0][0]][q[0][1]] + ((1 / collectedSampls) * (rewards - estimatedQ[q[0][0]][q[0][1]]))
    enviroment.updateQtable(estimatedQ)
    
    for state in policy:
        policy[state] = max(enviroment.qTable[state], key=enviroment.qTable[state].get)
    
    if (i+1)%25==0:
        print(f"\n\n\n step:{i}")
        enviroment.printPolicy(policy)
        
print("Q-table:")
enviroment.printQtable()

print(f"exploited:{enviroment.exploited}  explored:{enviroment.explored}")
