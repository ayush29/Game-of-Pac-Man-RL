import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random

class PacmanEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, grid_size, num_food):
        self.n = grid_size
        #state = tuple(?L,?R,?U,?D) ?:status of cell safe,food,danger
        #Tuple((spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3), spaces.Discrete(3))) total 81 states
        self.observation_space = spaces.Discrete(81)
        self.action_space = spaces.Discrete(4) # go towards L, R, U, D
#         self.grid = np.zeros(grid_size,grid_size)
        self.num_cells = self.n**2
        self.num_food = num_food
        temp = random.sample(range(0, self.num_cells), self.num_food+1) #+1 for random pos of pacman
        self.food = []
        for pos in temp[:-1]:
            x = int(pos/self.n)
            y = int(pos%self.n)
            self.food.append((x,y))
        self.pacman = (int(temp[-1]/self.n), int(temp[-1]%self.n))
        
        #(-1,_) , (_,-1) , (n,_) , (_,n) are boundary positions
        
        #After ghost reinit default ghost policy is if placed on
        # row left end then go right
        # row right end then go left
        # col top end then go down
        # col bottom end then go up
        #*****possible improvement in ghost policy is to add stochasticity in taking above actions******
        self.ghost = None
        self.ghost_action = None
        self.init_ghost()
        
        self.cur_state = self.get_cur_state()
        self.score =0 #num of food eaten by pacman
        
    def init_food(self):
        temp = random.sample(range(0, self.num_cells), self.num_food)
        for pos in temp[:-1]:
            x = int(pos/self.n)
            y = int(pos%self.n)
            self.food.append((x,y))
            
    def init_ghost(self):
        temp = random.random()
        if temp<0.25: #row ,left
            self.ghost = (self.pacman[0],0)
            self.ghost_action = 1 #R
            if self.ghost == self.pacman:
                self.ghost = (self.pacman[0],self.n-1)
                self.ghost_action = 0 #L
        elif temp<0.5: #row, right
            self.ghost = (self.pacman[0],self.n-1)
            self.ghost_action = 0 #L
            if self.ghost == self.pacman:
                self.ghost = (self.pacman[0],0)
                self.ghost_action = 1 #R
        elif temp<0.75: #col, U
            self.ghost = (0,self.pacman[1])
            self.ghost_action = 3 #D
            if self.ghost == self.pacman:
                self.ghost = (self.n-1,self.pacman[1])
                self.ghost_action = 2 #U
        else: #col, D
            self.ghost = (self.n-1,self.pacman[1])
            self.ghost_action = 2 #U
            if self.ghost == self.pacman:
                self.ghost = (0,self.pacman[1])
                self.ghost_action = 3 #D
    
    def get_cur_state(self):
        # use updated grid variables and pacman pos to get cur state
        posL = (self.pacman[0],self.pacman[1]-1)
        posR = (self.pacman[0],self.pacman[1]+1)
        posU = (self.pacman[0]-1,self.pacman[1])
        posD = (self.pacman[0]+1,self.pacman[1])
        state = [-1,-1,-1,-1]
        state[0] = self.get_cell_status(posL)
        state[1] = self.get_cell_status(posR)
        state[2] = self.get_cell_status(posU)
        state[3] = self.get_cell_status(posD)
        
        return tuple(state)
    
    def state2index(self,state):
        for i in range(0,4):
            if state[i]<0 or state[i]>2:
                return -1 #invalid state
        return state[0]*27 + state[1]*9 + state[2]*3 + state[3]

    def index2state(self,index):
        state = [-1,-1,-1,-1] #invalid state

        if index >=0 and index <=80:#check valid index
            i=3
            while index:
                state[i] = int(index%3)
                index = int(index/3)
                i = i-1

        return tuple(state)
    
    def get_cell_status(self, pos): #return 0 danger(ghost or boundary), 1 safe(empty or pacman), 2 food given a cell position, -1 invalid
        if pos == self.ghost:
            return 0
        if pos[0] == -1 or pos[1] ==-1 or pos[0] == self.n or pos [1] == self.n: #boundary cell
            return 0
        if pos[0]<-1 or pos[1] <-1 or pos[0] > self.n or pos [1] > self.n: # out of boundary cell, invalid cell
            return -1 #invalid cell
        if pos in self.food: # food cell
            return 2
        
        return 1 #empty cell
    
    def step(self, action):
        reward = self.get_reward(self.cur_state,action)
        done = self.take_action(action)
        self.cur_state = self.get_cur_state()
        return self.state2index(self.cur_state), reward, done, {'score':self.score, 'food_available':len(self.food)}
        
    def take_action(self,action): # one action of ghost according to ghost policy, one specified action for pacman
        #update pacman position and its consequences
        #         posL = (self.pacman[0],self.pacman[1]-1)
        #         posR = (self.pacman[0],self.pacman[1]+1)
        #         posU = (self.pacman[0]-1,self.pacman[1])
        #         posD = (self.pacman[0]+1,self.pacman[1])
        done = False
        if action ==0:
            self.pacman = (self.pacman[0],self.pacman[1]-1)
        elif action == 1:
            self.pacman = (self.pacman[0],self.pacman[1]+1)
        elif action == 2:
            self.pacman = (self.pacman[0]-1,self.pacman[1])
        else:
            self.pacman = (self.pacman[0]+1,self.pacman[1])
        pacman_status = self.get_cell_status(self.pacman)
        if pacman_status ==2: #food
            self.food.remove(self.pacman)
            self.score += 1
            #randomly putting food as food finished
            if len(self.food)==0:
                self.init_food()
        if pacman_status ==0: #end of episode , hit boundary or ghost
            done = True            
        
        #update ghost position 
        if self.ghost_action ==0:
            self.ghost = (self.ghost[0],self.ghost[1]-1)
        elif self.ghost_action == 1:
            self.ghost = (self.ghost[0],self.ghost[1]+1)
        elif self.ghost_action == 2:
            self.ghost = (self.ghost[0]-1,self.ghost[1])
        else:
            self.ghost = (self.ghost[0]+1,self.ghost[1])
        if self.ghost == self.pacman: #end of episode
            done = True
        if self.ghost[0]==-1 or self.ghost[1] ==-1 or self.ghost[0] == self.n or self.ghost [1] == self.n: #boundary cell
            self.init_ghost()
         
        return done
    
    def reset(self):
        #initialise the environment with pacman and food pallets at random locations
        #place ghost at end of pacman col or row randomly
        
        temp = random.sample(range(0, self.num_cells), self.num_food+1) #+1 for random pos of pacman
        self.food = []
        for pos in temp[:-1]:
            x = int(pos/self.n)
            y = int(pos%self.n)
            self.food.append((x,y))
        self.pacman = (int(temp[-1]/self.n), int(temp[-1]%self.n))
        
        #(-1,_) , (_,-1) , (n,_) , (_,n) are boundary positions
 
        self.init_ghost()
        
        self.cur_state = self.get_cur_state()
        self.score = 0
        return self.state2index(self.cur_state)
    
    #display function
    def render(self, mode='console'):
        if mode!='console':
            raise NotImplementedError()
        for i in range(0,self.n):
            print('\n')
            for j in range(0,self.n):
                pos = (i,j)
                if pos == self.ghost:
                    print('g',end='')
                elif pos == self.pacman:
                    print('p',end='')
                elif pos in self.food:
                    print('*',end='')
                else:
                    print('-',end='')
    
    
    def close(self):
        pass
        
    def get_reward(self,state, action): #calculate reward before making changes in grid
        if state[action] ==0: # danger: ghost or wall
            return -10
        elif state[action] == 1: #safe :empty cell
            return 0
        else: # food
            return 2
    
  