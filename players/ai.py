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
    def is_terminal(self):
        if(self.parent==None):
            return False
        return helper.check_win(self.board_state,self.delta,self.parent.action_player_id)
    def is_leaf(self):
        if(len(self.untried_actions)>0):
            return True
        return (len(self.children)==0)
    def get_ucb(self):
        exploitation=(self.wins)/(self.visits)
        exploration=math.sqrt(2*(math.log(self.parent.visits))/self.visits)
        return (exploitation+exploration)
    def best_child(self):
        return max(self.children, key=lambda child: child.get_ucb())
    def backpropagate(self,win_player_id):
        node=self
        while(node is not None):
            if(node.action_player_id==win_player_id):
                node.wins+=1
                node.visits+=1
            else:
                node.visits+=1
            node=node.parent
class MonteCarloTree:
    def __init__(self,root,player_id):
        self.root=root
        self.root_player_id=player_id
        self.other_player_id=player_id+0.5
    def select(self):
        node=self.root
        while(not(node.is_leaf())):
            node=node.best_child()
        return node
    def simulate(self,child):
        board_state=child.board_state
        player_id=child.action_player_id
        action=child.delta
        if(player_id==self.root_player_id):
            prev_player_id=self.other_player_id
        else:
            prev_player_id=self.root_player_id
        while(True):
            if(helper.check_win(board_state,action,prev_player_id)):
                return prev_player_id
            possible_actions=np.argwhere(board_state == 0).tolist()
            if(len(possible_actions) == 0):
                return -1
            action=tuple(random.choice(possible_actions))
            board_state[action]=player_id
            temp=player_id
            player_id=prev_player_id
            prev_player_id=temp
    def expand(self,leaf):
        if(len(leaf.untried_actions)>0):
            action= random.choice(leaf.untried_actions) 
            leaf.untried_actions.remove(action)
            board_state=leaf.board_state
            board_state[action]=leaf.action_player_id
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
        return move.delta
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
        root=Node(None,state,self.player_number,None)
        tree=MonteCarloTree(root,self.player_number)
        # Do the rest of your implementation here
        time_is_remaining=True
        time_per_move=(helper.fetch_remaining_time(self.timer,self.player_number))/moves_remaining
        start_time = time.time()
        while(time_is_remaining):

            leaf=tree.select()
            child=tree.expand(leaf)
            winner=tree.simulate(child)
            child.backpropagate(winner)

            elapsed_time = time.time() - start_time
            time_is_remaining= (elapsed_time >= time_per_move)
        
        return tree.get_next_move()