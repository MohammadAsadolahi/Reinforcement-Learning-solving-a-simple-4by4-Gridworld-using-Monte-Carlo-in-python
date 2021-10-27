# Reinforcement-Learning-solving-a-simple-4by4-Gridworld-using-Monte-Carlo
solving a simple 4*4 Gridworld almost similar to openAI gym frozenlake using Monte-Carlo method Reinforcement Learning
WRITTEN BY MOHAMMAD ASADOLAHI  
Mohammad.E.Asadolahi@gmail.com  
https://github.com/mohammadAsadolahi  
this program is using Reinfrocement learning to solve a 4*4 gridworld like frozen lake enviroment in open ai gym  
the method used is policy iteration whitch is one of fundamental manners of Dynamic Programing  

     | S | O | O | O |  
     | O | O | O | * |  
     | O | * | O | O |  
     | O | * | O | T |  

  
  S= start cell  
  O= normal cells  
  *= penalized cells  
  T= terminate cell  
  
our agent goal is to find policy to go from S(start) cell to T(goal) cell with maximum reward(or minimum negative reward)  
valid actions are storend in GridWorld actions array.  
positive and negative rewards in each cell is stored in Gridworld  "Rewards" dictionary and can be modified by user .the current rewards for *(hole) cells ant T(goal) cell has been set to:  
self.rewards = {(3, 3): 5, (1, 3): -2, (2, 1): -2, (3, 1): -2}  
for example reward to go in (3,3) in enviroment witch is the goal will be +5 so agent gets +5 reward whenever go to cell (3,3)  
the size of Gridworld can be changed in GridWorld calss by adding space actions  
***************************
Algorithm Flow
***************************
  first we initialize a random policy that indicate prefered moves in every cell:  
  
    | D |  | L |  | R |  | D | 
    ----------------------------
    | U |  | U |  | R |  | D | 
    ----------------------------
    | D |  | R |  | R |  | U | 
    ----------------------------
    | U |  | L |  | R | 
    ----------------------------
 
U = going up  
D = going down  
L = going left  
R = going right  
  
and we initialize Q table like:  

    (0, 0): {'D': 0, 'R': 0},
    (0, 1): {'L': 0, 'D': 0, 'R': 0},
    (0, 2): {'L': 0, 'D': 0, 'R': 0},
    (0, 3): {'L': 0, 'D': 0},
    (1, 0): {'U': 0, 'D': 0, 'R': 0},
    (1, 1): {'U': 0, 'L': 0, 'D': 0, 'R': 0},
    (1, 2): {'U': 0, 'L': 0, 'D': 0, 'R': 0},
    (1, 3): {'U': 0, 'L': 0, 'D': 0},
    (2, 0):{'U': 0, 'D': 0, 'R': 0},
    (2, 1): {'U': 0, 'L': 0, 'D': 0, 'R': 0},
    (2, 2): {'U': 0, 'L': 0, 'D': 0, 'R': 0},
    (2, 3): {'U': 0, 'L': 0, 'D': 0},
    (3, 0): {'U': 0, 'R': 0},
    (3, 1): {'U': 0, 'L': 0, 'R': 0},
    (3, 2): {'U': 0, 'L': 0, 'R': 0}}
  
then we start using monte carlo sampling accordint to our generated policy to update this Q-table and as well as policy till convergence of policy witch will be optimal policy  
  sample outputs:
  
 ***************************
 random generated policy is:
***************************
 | D |  | L |  | R |  | D |   
 | U |  | U |  | R |  | D |   
 | D |  | R |  | R |  | U |   
 | U |  | L |  | R |   

  --------------------------------  
 step:0   
   --------------------------------  
   
       Policy:  
 | D |  | D |  | L |  | L |   
 | U |  | L |  | L |  | U |   
 | D |  | D |  | D |  | D |   
 | U |  | R |  | R |   

  --------------------------------  
 step:25   
   --------------------------------  
   
       Policy:  
 | R |  | D |  | L |  | L |   
 | R |  | R |  | D |  | U |   
 | D |  | D |  | D |  | D |   
 | R |  | R |  | R |   

  --------------------------------  
 step:50  
   --------------------------------  
        Policy:  
 | R |  | D |  | L |  | L |   
 | R |  | R |  | D |  | U |   
 | D |  | D |  | D |  | D |   
 | R |  | R |  | R |   

  --------------------------------  
 step:75   
   --------------------------------  
          Policy:  
 | R |  | D |  | L |  | L |   
 | R |  | R |  | D |  | U |   
 | D |  | D |  | D |  | D |   
 | R |  | R |  | R |   

  --------------------------------  
 step:100   
   --------------------------------  
        Policy:  
 | R |  | D |  | L |  | L |   
 | R |  | R |  | D |  | U |   
 | D |  | D |  | D |  | D |   
 | R |  | R |  | R |   

  --------------------------------  
 step:125
   --------------------------------  
          Policy:  
 | R |  | D |  | L |  | L |   
 | R |  | R |  | D |  | U |   
 | D |  | D |  | D |  | D |   
 | R |  | R |  | R |   
  
--------------------------------  
 step:150
 --------------------------------  
        Policy:  
 | R |  | D |  | L |  | L |   
 | R |  | R |  | D |  | U |   
 | D |  | D |  | D |  | D |   
 | R |  | R |  | R |   

  --------------------------------  
 step:175   
   --------------------------------  
          Policy:  
 | R |  | D |  | L |  | L |   
 | R |  | R |  | D |  | U |    
 | D |  | D |  | D |  | D |   
 | R |  | R |  | R |   

  --------------------------------  
 step:200   
   --------------------------------  
             Policy:  
 | R |  | D |  | L |  | L |   
 | R |  | R |  | D |  | U |    
 | D |  | D |  | D |  | D |   
 | R |  | R |  | R |   

  --------------------------------  
 step:225 
   --------------------------------  
             Policy:  
 | R |  | D |  | L |  | L |   
 | R |  | R |  | D |  | U |   
 | D |  | D |  | D |  | D |   
 | R |  | R |  | R |   

  ----------------------------
    exploited:7984250  explored:1407712
  --------------------------------  
