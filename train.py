import os
import json

import numpy as np
from itertools import product
from tqdm import tqdm

class TicTacToeTrainer:
    """
    Uses Value Iteration method in Reinforcement Learning.
    Treat it self as X (1) and opponent as O (-1).
    """

    def __init__(self, game, gamma=0.9):
        """
        Parameters
        ----------
        game : TicTacToe
             The tic-tac-toe game to train on.
        gamma : float
             Discount factor for future rewards.
        """
        self.game = game
        self.gamma = gamma
        self.states = self._generate_states()
        self.policy = {}

    def _remove_impossible_states(self, states):
        """
        Remove impossible states from state space.
        """
        valid_states = []
        for board in states:
            x_count = np.sum(board == 1)
            o_count = np.sum(board == -1)

            # As both players alternate turns, the count of X's and O's can differ by at most 1.
            if abs(o_count - x_count) > 1:
                continue

            # check if Anyone already won
            winner = self.game.check_winner(board)
            if winner == 1 and winner == -1:
                continue
            
            # check if match already draw
            if self.game.is_draw(board):
                continue

            valid_states.append(board)

        return valid_states

    def _generate_states(self):
        """
        Generate all possible board states.

        Returns
        -------
        list[np.ndarray]
        """
        states = []

        # Generate all combinations of -1, 0, 1 for 9 positions (3x3 board)
        for state in product([-1, 0, 1], repeat=9):
            board = np.array(state).reshape(3, 3)
            states.append(board)

        # Remove impossible states
        states = self._remove_impossible_states(states)

        return states

    def board_to_index(self, board, computer=1):
        """
        Convert board -> unique index using base-3 encoding.

        Parameters
        ----------
        board : np.ndarray (3x3)
        computer : int
            The player for whom to generate the index (1 for computer, -1 for opponent)

        Returns
        -------
        int
        """
        # Flip board as encoding assumes computer is 1 and opponent is -1
        if computer == -1:
            board = -board
        
        flat = board.flatten()
        mapped = flat + 1

        idx = 0
        for i, v in enumerate(mapped):
            idx += v * (3 ** i)

        return int(idx)

    def save_policy(self, path="tictactoe_policy.json"):
        """
        Save policy to disk.
        """
        with open(path, 'w') as f:
            json.dump(self.policy, f)
        print(f"Policy saved → {path}")
        
    def load_policy(self, path="tictactoe_policy.json"):
        """
        Load policy from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Policy file not found: {path}")
        with open(path, 'r') as f:
            self.policy = json.load(f)
        print(f"Policy loaded ← {path}")
        
    def train(self, episodes=1000):
        # Make v_max using state index as key
        v_max = {self.board_to_index(state): 0 for state in self.states}
        v_max_old = v_max.copy()
        
        # Value Iteration
        for _ in tqdm(range(episodes), desc="Training"):
            # For each state, compute best Value on all possible actions
            for idx, state in enumerate(self.states):
                # Get valid actions for computer for current state
                actions = self.game.get_valid_actions(state)
                
                # If no actions available, it's draw and a terminal state. Set value to 0 and continue.
                if not actions:
                    v_max[idx] = 0
                    continue
                
                # For each action get best Value
                v_max_curr = -float('inf')
                # Iterate over all possible actions for computer
                for action in actions:
                    # Simulate computer move
                    next_state = self.game.make_move(state.copy(), action, 1)
                    
                    # Check whether match draw with this move
                    is_comp_draw = self.game.is_draw(next_state)
                    if is_comp_draw:
                        v_max_curr = max(v_max_curr, 0)  # Draw gives reward 0
                        continue

                    # Check winner with this move
                    winner = self.game.check_winner(next_state)
                    if winner == 1:  # Computer wins
                        v_max_curr = max(v_max_curr, 1)  # Win gives reward 1
                    elif winner == -1:  # Opponent wins
                        v_max_curr = max(v_max_curr, -1)  # Loss gives reward -1
                    else:  # No winner and not a draw yet
                        # Simulate opponent move
                        opponent_actions = self.game.get_valid_actions(next_state)
                        if len(opponent_actions) == 0:
                            v_max_curr = max(v_max_curr, 0)  # Draw gives reward 0
                        else:
                            # Iterate over opponent's possible moves
                            for opponent_action in opponent_actions:
                                opponent_next_state = self.game.make_move(next_state.copy(), opponent_action, -1)
                                is_opp_draw = self.game.is_draw(opponent_next_state)
                                if is_opp_draw:
                                    v_max_curr = max(v_max_curr, 0)  # Draw gives reward 0
                                    break
                                opponent_winner = self.game.check_winner(opponent_next_state)
                                if opponent_winner == -1:  # Opponent wins
                                    v_max_curr = max(v_max_curr, -1)  # Loss gives reward -1
                                    break
                                elif opponent_winner == 1:  # Computer wins (unlikely in opponent's turn)
                                    v_max_curr = max(v_max_curr, 1)  # Win gives reward 1
                                    break
                                else:
                                    # If no winner and no draw, use value of next state
                                    next_state_index = self.board_to_index(opponent_next_state)
                                    v_max_curr = max(v_max_curr, self.gamma * v_max_old[next_state_index])
                state_idx = self.board_to_index(state)
                v_max[state_idx] = v_max_curr
            # Update old values for next iteration
            v_max_old[:] = v_max
    
        # Derive policy from value function
        for idx, state in enumerate(self.states):
            # Get valid actions for computer for current state
            actions = self.game.get_valid_actions(state)
            best_action = -1
            best_value = -float('inf')
            # Iterate over all possible actions for computer
            for action in actions:
                # Simulate computer move
                next_state = self.game.make_move(state.copy(), action, 1)
                
                # Check whether match draw with this move
                is_comp_draw = self.game.is_draw(next_state)
                if is_comp_draw:
                    action_value = 0  # Draw gives reward 0
                else:
                    # Check winner with this move
                    winner = self.game.check_winner(next_state)
                    if winner == 1:  # Computer wins
                        action_value = 1  # Win gives reward 1
                    elif winner == -1:  # Opponent wins
                        action_value = -1  # Loss gives reward -1
                    else:  # No winner and not a draw yet
                        opponent_actions = self.game.get_valid_actions(next_state)
                        if len(opponent_actions) == 0:
                            action_value = 0  # Draw gives reward 0
                        else:
                            opponent_values = []
                            for opponent_action in opponent_actions:
                                opponent_next_state = self.game.make_move(next_state.copy(), opponent_action, -1)
                                is_opp_draw = self.game.is_draw(opponent_next_state)
                                if is_opp_draw:
                                    opponent_values.append(0)  # Draw gives reward 0
                                else:
                                    opponent_winner = self.game.check_winner(opponent_next_state)
                                    if opponent_winner == -1:  # Opponent wins
                                        opponent_values.append(-1)  # Loss gives reward -1
                                    elif opponent_winner == 1:  # Computer wins (unlikely in opponent's turn)
                                        opponent_values.append(1)  # Win gives reward 1
                                    else:
                                        next_state_index = self.board_to_index(opponent_next_state)
                                        opponent_values.append(self.gamma * v_max_old[next_state_index])
                            action_value = min(opponent_values)  # Assume opponent plays optimally (minimize our value)
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            self.policy[self.board_to_index(state)] = best_action

if __name__ == "__main__":
    from game import TicTacToe
    game = TicTacToe()
    trainer = TicTacToeTrainer(game)
    trainer.train()
    trainer.save_policy()
