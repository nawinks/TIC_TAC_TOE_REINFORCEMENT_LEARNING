import numpy as np


class TicTacToe:
    """
    TicTacToe game environment containing board logic and rules.

    Board Encoding
    --------------
    0 : empty
    1 : player X
    -1: player O

    Functions
    ---------
    reset() -> reset board
    make_move(action, player) -> place mark
    get_valid_actions() -> list of free cells
    check_winner() -> return winner
    is_draw() -> check draw
    """

    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        
    def map_XO_to_number(self, player):
        if player == 'X':
            return 1
        elif player == 'O':
            return -1
        else:
            raise ValueError("Invalid player: must be 'X' or 'O'")

    def reset(self):
        """Reset board to empty state"""
        self.board[:] = 0

    def make_move(self, board, action: int, player: int) -> bool:
        """
        Place a move on board.

        Parameters
        ----------
        action : int
            cell index (0-8)

        player : int
            1 for X, -1 for O

        Returns
        -------
        bool
            True if move valid, False otherwise
        """
        r, c = divmod(action, 3)

        if board[r, c] != 0:
            raise ValueError("Invalid move: cell already occupied")

        board[r, c] = player
        return board

    def get_valid_actions(self, board):
        """
        Returns list of empty cell indices
        """
        actions = []
        for i in range(9):
            r, c = divmod(i, 3)
            if board[r, c] == 0:
                actions.append(i)
        return actions

    def check_winner(self, board):
        """
        Check winner of game.

        Returns
        -------
        int
            1 if X wins
            -1 if O wins
            0 if no winner
        """
        lines = []

        # rows
        lines.extend(board)

        # columns
        lines.extend(board.T)

        # diagonals
        lines.append(np.diag(board))
        lines.append(np.diag(np.fliplr(board)))

        for line in lines:
            s = np.sum(line)
            if s == 3:
                return 1
            if s == -3:
                return -1

        return 0

    def is_draw(self, board):
        """
        Check if game ended in draw.
        """
        return np.all(board != 0) and self.check_winner(board) == 0