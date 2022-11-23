# Student agent: Add your own agent here
from logging import root
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
# import math #MUST ADD THIS TO REQUIREMENTS.TXT!!!
EXP_PARAM = 2

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        # define Monte Carlo Tree Searc
    class MCTS:
        pass
        # define a node of the Monte Carlo Tree
        class Node:
            def __init__(self, board, move, parent, evaluator):
                self.board = board
                self.move = move
                self.parent = parent
                self.visits = 0
                self.wins = 0
                self.children = []
                self.evaluator = evaluator

            def addVisit(self):
               self. visit+=1
            
            def addWin(self, w):
                self.wins+=w

            def getRandomChild(self):
                if not self.children:
                    return None
                return self.children[np.random.randint(0, len(self.children))]
            
            def getMaxChild(self):
                return max(self.children, key=self.evaluator)
        
        def __init__(self, board):
            self.root = self.Node(board, move = None, parent = None, evaluator = self.uct_evaluator)
    

        def uct_evaluator(node):
            if( node.visits == 0): return 1000000000
            return (node.wins / node.visits) + EXP_PARAM * np.sqrt(np.log(node.parent.visits)/node.visits)

        def choose_move(board):
            pass

        

        



            

            



    


    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # dummy return
        return my_pos, self.dir_map["u"]
