import time
import math
import random
import numpy as np
import helper
from typing import Tuple


class Node:
    def __init__(self,delta,board_state,player_id,parent=None):
        self.delta = delta
        self.parent = parent
        self.children = []
        self.untried_actions = np.argwhere(board_state == 0).tolist()
        self.board_state=board_state
        self.action_player_id=player_id
        self.visits = 0
        self.wins = 0 
        self.dim=len(board_state[0])/2
        self.rave_wins = 0
        self.rave_visits = 0
        self.rave_weight=1
    def is_terminal(self):
        if(self.parent==None):
            return False
        return helper.check_win(self.board_state,self.delta,self.parent.action_player_id)
    def is_leaf(self):
        if(len(self.untried_actions)>0):
            return True
        return (len(self.children)==0)
    def find_threat(self,id):
        temp_board=np.copy(self.board_state)
        x,y=self.delta
        temp_board[x][y]=id
        left_actions=np.argwhere(self.board_state==0).tolist()
      
        for x2,y2 in left_actions:
            temp_board[x2][y2]=id
       
            win,str_val=helper.check_win(temp_board,(x2,y2),id)
            if win:
             
                return True
            temp_board[x2][y2]=0
        return False


    def calculate_distance_to_neighbors(self, action, neighbors):
        """ Calculate the Manhattan or Euclidean distance from the action to all neighbors. """
        x1, y1 = action
        distances = [math.sqrt((x1 - x2)**2 + (y1 - y2)**2) for x2, y2 in neighbors]
        distances.sort()
        sum_distances=0
        for i in range(min(3,len(distances))):
            sum_distances+=distances[i]
        return sum_distances 
     
    def get_neighbor_heuristic(self):
        neighbors=np.argwhere(self.board_state == self.action_player_id).tolist()
        if not neighbors or self.delta is None:
            return 0 
        distance = self.calculate_distance_to_neighbors(self.delta, neighbors)
        
        x,y=self.delta
        win_sel_blk=1
    
        temp_board=np.copy(self.board_state)

        figure,str_val=helper.check_win(temp_board,self.delta,3-self.action_player_id)
        
        if figure:

            win_sel_blk+=40
        else:
            win=self.find_threat(3-self.action_player_id)
            if win:
                win_sel_blk+=5

        temp_board=np.copy(self.board_state)
        temp_board[x][y]=self.action_player_id

        figure,str_val=helper.check_win(temp_board,self.delta,self.action_player_id)        
        if figure:
            win_sel_blk+=30
        else:
            win=self.find_threat(self.action_player_id)
            if win:
                win_sel_blk+=3
        return (win_sel_blk)/ ((1 + distance) )
  
    def get_ucb(self):
        
        exploitation=(self.wins)/(self.visits)
        exploration=math.sqrt(2*(math.log(self.parent.visits))/self.visits)
        heuristic = self.get_neighbor_heuristic() 
        
        return (exploitation+exploration+heuristic)
    def get_rave_ucb(self, R=100):
        
        if self.visits > 0:
            exploitation = self.wins / self.visits
        else:
            exploitation = 0
        exploration = math.sqrt( 2*math.log(self.parent.visits) / self.visits) if self.visits > 0 else float('inf')
        
        # RAVE value
        if self.rave_visits > 0:
            rave_value = self.rave_wins / self.rave_visits
        else:
            rave_value = 0
        
        # Weighted RAVE UCB
        alpha = R / (R + self.visits)
        heuristic = self.get_neighbor_heuristic() 
        value=(1 - alpha) * (exploitation + exploration) + (alpha * rave_value) + heuristic
        return value
    def best_child(self):
        return max(self.children, key=lambda child: child.get_rave_ucb())
    def backpropagate(self,win_player_id,simulation_path):
        node = self 
        
        while node is not None:
            node.visits += 1
            if node.action_player_id == win_player_id:
                node.wins += 1
            
            # Update RAVE stats for all moves in this path
            for move_node in simulation_path:
                if move_node.action_player_id == win_player_id:
                    move_node.rave_wins += 1
                move_node.rave_visits += 1
                
            node = node.parent
class MonteCarloTree:
    def __init__(self,root,player_id):
        self.root=root
        self.root_player_id=player_id
        self.other_player_id=3-player_id
      
  
    def select(self):
        node=self.root
        while(not(node.is_leaf())):
            node=node.best_child()
        
        return node
    

    def simulate(self,child):
        
        board_state=np.copy(child.board_state)
        
        player_id=child.action_player_id
        action=child.delta

        path = [child]

        prev_node=child
        if(player_id==self.root_player_id):
            prev_player_id=self.other_player_id
        else:
            prev_player_id=self.root_player_id
        while(True):
        
            win_check,str_val=helper.check_win(board_state,action,prev_player_id)
            if(win_check):
            
                return prev_player_id,path
            possible_actions=np.argwhere(board_state == 0).tolist()
            if(len(possible_actions) == 0):
                return -1,path
            min_dist=float('inf')
            action=tuple(random.choice(possible_actions))
            for any_action in possible_actions:
                neighbors=np.argwhere(board_state == player_id).tolist()
                heuristic=prev_node.calculate_distance_to_neighbors(any_action,neighbors)
                if heuristic<min_dist:
                    action=tuple(any_action)



            
            new_node=Node(action,board_state,player_id,prev_node)
            path.append(new_node)

            prev_node=new_node
            

            x,y=action
            board_state[x][y]=player_id
            

            temp=player_id
            player_id=prev_player_id
            prev_player_id=temp
       
        
    def expand(self,leaf):
        if(len(leaf.untried_actions)>0):
            
                
            action= random.choice(leaf.untried_actions) 
   
            leaf.untried_actions.remove(action)
            board_state=np.copy(leaf.board_state)
            x,y=action
            board_state[x][y]=leaf.action_player_id
            if(leaf.action_player_id==self.root_player_id):
                player_id=self.other_player_id
            else:
                player_id=self.root_player_id            
            n=Node(tuple(action),board_state,player_id,leaf)
            leaf.children.append(n)
            return n
        else:
            return leaf
    def get_next_move(self):
        root=self.root
        move=root.best_child()
        x,y=move.delta
       
        return tuple([int(x),int(y)])
def MCTS_sample(node:Node,tree:MonteCarloTree):

    if len(node.untried_actions)==0:
        child =tree.select()
        return child
        
    else:
        child=tree.expand(node)
        winner,path=tree.simulate(child)
        child.backpropagate(winner,path)
        return node


   
class AIPlayer:

    def __init__(self, player_number: int, timer):
        """
        Intitialize the AIPlayer Agent

        # Parameters
        `player_number (int)`: Current player number, num==1 starts the game
        
        `timer: Timer`
            - a Timer object that can be used to fetch the remaining time for any player
            - Run `fetch_remaining_time(timer, player_number)` to fetch remaining time of a player
        """
   
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
    

    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Given the current state of the board, return the next move

        # Parameters
        `state: Tuple[np.array]`
            - a numpy array containing the state of the board using the following encoding:
            - the board maintains its same two dimensions
            - spaces that are unoccupied are marked as 0
            - spaces that are blocked are marked as 3
            - spaces that are occupied by player 1 have a 1 in them
            - spaces that are occupied by player 2 have a 2 in them

        # Returns
        Tuple[int, int]: action (coordinates of a board cell)
        """
        moves_remaining = (np.count_nonzero(state == 0))/2
        total_moves=(np.count_nonzero(state != 3))/2
        if moves_remaining>0.3*(total_moves):
            time_percent=0.5
        else: 
            time_percent=1
        
        root=Node(None,state,self.player_number,None)
        tree=MonteCarloTree(root,self.player_number)
        # Do the rest of your implementation here
        time_is_remaining=True

        time_per_move=(helper.fetch_remaining_time(self.timer,self.player_number))*(time_percent/moves_remaining)
        
        start_time = time.time()
        leaf=root
        while(time_is_remaining):

            leaf=MCTS_sample(leaf,tree)
            

            elapsed_time = time.time() - start_time
            time_is_remaining= (elapsed_time <= time_per_move)
        output=tree.get_next_move()
        print(type(output))
        for i in output:
            print(type(i))
        if isinstance(output, tuple) and all(isinstance(i, int) for i in output):
            print("It is a tuple of integers")  # Output: It is a tuple of integers
        else:
            print("The tuple does not contain only integers")
        print(output)
        return output
