# Student agent: Add your own agent here
from logging import root
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time
from copy import deepcopy
# import math #MUST ADD THIS TO REQUIREMENTS.TXT!!!
EXP_PARAM = 2
TIME_LIMIT = 1980

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
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        self.tree = self.MCTS()

        # define Monte Carlo Tree Searc
    class MCTS:
        # define a node of the Monte Carlo Tree
        class Node:
            def __init__(self, board, move, parent, mypos, advpos):
                self.board = board
                self.move = move
                self.mypos = mypos
                self.advpos = advpos
                self.parent = parent
                self.visits = 0
                self.wins = 0
                self.children = []
                

            def uct_evaluator(node):
                if( node.visits == 0): return 1000000000
                return (node.wins / node.visits) + EXP_PARAM * np.sqrt(np.log(node.parent.visits)/node.visits)

            def addVisit(self):
               self. visit+=1
            
            def addWin(self, w):
                self.wins+=w

            def getRandomChild(self):
                if not self.children:
                    return None
                return self.children[np.random.randint(0, len(self.children))]
            
            def getMaxChild(self):
                if(not self.children):
                    return None
                return max(self.children, key=self.evaluator)
        
        def __init__(self, board):
            self.root = self.Node(board, move = None, parent = None)
    

        

    def check_endgame(self, my_pos, adv_pos, board_size, chess_board):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        # player_win = None
        # win_blocks = -1
        # if p0_score > p1_score:
        #     player_win = 0
        #     win_blocks = p0_score
        # elif p0_score < p1_score:
        #     player_win = 1
        #     win_blocks = p1_score
        # else:
        #     player_win = -1  # Tie
        # if player_win >= 0:
        #     logging.info(
        #         f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
        #     )
        # else:
        #     logging.info("Game ends! It is a Tie!")
        return True, p0_score, p1_score

    def set_barrier(self, r, c, dir):
        # Set the barrier to True
        self.chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        self.chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def expand(self, node:MCTS.Node):
        cur_board = node.board
        mypos = deepcopy(node.mypos)
        advpos = deepcopy(node.advpos)
        

        valid_moves = self.get_valid_moves(cur_board, mypos, advpos)

        for move in valid_moves:
            new_board = deepcopy(cur_board)
            new_mypos, new_advpos = self.apply_move(new_board, mypos , advpos, move) #TODO WRITE THIS FUNCTION
            child = self.MCTS.Node(board=new_board, move = move, parent = node, mypos = new_mypos, advpos = new_advpos)
            node.children.append(child)



    def get_valid_moves(self, chess_board, mypos, advpos, max_step ):
        #TODO finish this
        moves = []
        return moves


    def select(self, root:MCTS.Node):
        cur = root
        while(cur.children):
            cur = cur.getMaxChild()
        return cur

    def simulate(self, node):
        pass

    def propagate_to_parent(self, node:MCTS.Node, win):
        cur = node
        while(cur is not None):
            cur.visits+=1
            cur.wins+=1
            cur = cur.parent

    def simulate(node):
        my_pos = deepcopy(node.mypos)
        adv_pos = deepcopy(node.advpos)
        cur_board = deepcopy(node.board) 

        while()

        

        

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
        tree = self.MCTS(chess_board, my_pos, adv_pos, max_step)
        limit_time = time() + TIME_LIMIT

        while time.time() < limit_time:
            best_node = self.select(tree.root)

            if best_node is None:
                best_node = tree.root
            
            game_over, _ , _ = self.check_endgame(best_node)

            if( not game_over):
                self.expand(best_node)

            explorationNode = best_node.getRandomChild()

            if not explorationNode:
                explorationNode = best_node
            
            win = simulate(explorationNode)

            self.propagate_to_parent(explorationNode, win)


        optimal_node = tree.root.getMaxChild()
        # convert to proper form

        return my_pos, self.dir_map["u"]
