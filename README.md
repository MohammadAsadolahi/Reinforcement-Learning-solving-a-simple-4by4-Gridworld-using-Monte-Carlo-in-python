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
    (2, 0): {'U': 0, 'D': 0, 'R': 0},
    (2, 1): {'U': 0, 'L': 0, 'D': 0, 'R': 0},
    (2, 2): {'U': 0, 'L': 0, 'D': 0, 'R': 0},
    (2, 3): {'U': 0, 'L': 0, 'D': 0},
    (3, 0): {'U': 0, 'R': 0},
    (3, 1): {'U': 0, 'L': 0, 'R': 0},
    (3, 2): {'U': 0, 'L': 0, 'R': 0}}
  
we start using monte carlo sampling accordint to our generated policy. we play untill end of the episode and after episode termination we count over every transiton in reverse and use the transiton reward to update Q-table and as well as policy till convergence of policy witch will be optimal policy  
for choosing next action we use epsilon greedy method. for more information about epsilon greedy method go to :https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/
  sample outputs of program :
  
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
 
   --------------------------------  
 Q-table:
   --------------------------------  
          (0, 0): {'D': 1.4231111446110687, 'R': 2.7825312901637713},
          (0, 1): {'L': 1.1954361778316396, 'D': 2.7742013116068813, 'R': 1.2509234550696697},
          (0, 2): {'L': 1.6175230984831273, 'D': 0.17183770184907848, 'R': 0.10523112338768514}, 
          (0, 3): {'L': 0.25315532831734844, 'D': -0.11210604663503392},
          (1, 0): {'U': 1.69660913428058, 'D': -0.07762194441963757, 'R': 0.1342960474941175}, 
          (1, 1): {'U': 0.6168928204772729, 'L': 0.588214146463321, 'D': -1.9868817828128778, 'R': 3.1484181985052597},
          (1, 2): {'U': 0.70178234887395, 'L': 0.8610527852605808, 'D': 3.534471307576427, 'R': -1.496012894908975}, 
          (1, 3): {'U': 0.04208996058903998, 'L': 0.06030382085868598, 'D': 1.293025247558745}, 
          (2, 0): {'U': 0.05279680010674521, 'D': -0.10294746211473406, 'R': -0.13746104149339952}, 
          (2, 1): {'U': 0.07503155634758919, 'L': -0.0624724089893041, 'D': -0.11197071232894086, 'R': 1.9193926007952289},
          (2, 2): {'U': 1.0095321553385863, 'L': -1.6561346336559677, 'D': 3.9384240972370463, 'R': 1.136110436649536},
          (2, 3): {'U': -0.17710315668212362, 'L': 0.07079802717090475, 'D': 2.2025787785151425},
          (3, 0): {'U': -0.061445018480211534, 'R': -0.07837951530860222}, 
          (3, 1): {'U': -0.14749083099901897, 'L': -0.029828637303674892, 'R': 1.4299351028815779}, 
          (3, 2): {'U': 1.0640013704042732, 'L': -2.2750343869388443, 'R': 4.499853592119016}      
            
    exploited:7984250  explored:1407712
 
