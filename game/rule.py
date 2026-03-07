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

    def reset(self):
        """Reset board to empty state"""
        self.board[:] = 0

    def make_move(self, action: int, player: int) -> bool:
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

        if self.board[r, c] != 0:
            return False

        self.board[r, c] = player
        return True

    def get_valid_actions(self):
        """
        Returns list of empty cell indices
        """
        actions = []
        for i in range(9):
            r, c = divmod(i, 3)
            if self.board[r, c] == 0:
                actions.append(i)
        return actions

    def check_winner(self):
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
        lines.extend(self.board)

        # columns
        lines.extend(self.board.T)

        # diagonals
        lines.append(np.diag(self.board))
        lines.append(np.diag(np.fliplr(self.board)))

        for line in lines:
            s = np.sum(line)
            if s == 3:
                return 1
            if s == -3:
                return -1

        return 0

    def is_draw(self):
        """
        Check if game ended in draw.
        """
        return np.all(self.board != 0) and self.check_winner() == 0