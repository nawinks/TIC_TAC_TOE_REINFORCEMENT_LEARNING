import pygame
import sys
import random
from game import TicTacToe
import numpy as np
from train import TicTacToeTrainer


# =============================
# WINDOW SETTINGS
# =============================

WIDTH = 640
HEIGHT = 760

TOP_BAR = 100
BOARD_SIZE = 600

GRID = 3
CELL = BOARD_SIZE // GRID

MACHINE_DELAY = 600


# =============================
# COLORS
# =============================

BG = (18, 18, 18)
GRID_COLOR = (200, 200, 200)

X_COLOR = (255, 120, 120)
O_COLOR = (120, 180, 255)

TEXT = (240, 240, 240)

BUTTON = (55, 55, 55)
BUTTON_HOVER = (95, 95, 95)

BAR = (28, 28, 28)
WIN_LINE = (255, 215, 0)


# =========================================================
# UI
# =========================================================

class TicTacToeUI:

    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tic Tac Toe")

        self.clock = pygame.time.Clock()

        self.font_big = pygame.font.SysFont("Arial", 48)
        self.font = pygame.font.SysFont("Arial", 30)

        self.game = TicTacToe()

        self.state = "MENU"
        self.mode = None

        self.current_player = -1
        self.game_over = False
        self.winner = 0
        self.win_line = None

        self.machine_thinking = False
        self.machine_start = None

        self.init_buttons()
        
        self.train = TicTacToeTrainer(self.game)
        self.train.load_policy("tictactoe_policy.json")

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def init_buttons(self):

        self.home_btn = pygame.Rect(20, 20, 120, 40)
        self.restart_btn = pygame.Rect(WIDTH - 140, 20, 120, 40)

        self.hvh_btn = pygame.Rect(WIDTH // 2 - 160, 320, 320, 70)
        self.hvm_btn = pygame.Rect(WIDTH // 2 - 160, 420, 320, 70)

    # =========================================================
    # DRAW HELPERS
    # =========================================================

    def draw_button(self, rect, text):

        mouse = pygame.mouse.get_pos()
        color = BUTTON_HOVER if rect.collidepoint(mouse) else BUTTON

        pygame.draw.rect(self.screen, color, rect, border_radius=12)

        label = self.font.render(text, True, TEXT)

        self.screen.blit(
            label,
            (
                rect.centerx - label.get_width() // 2,
                rect.centery - label.get_height() // 2
            )
        )

    # =========================================================
    # MENU
    # =========================================================

    def draw_menu(self):

        title = self.font_big.render("Tic Tac Toe", True, TEXT)

        self.screen.blit(
            title,
            (WIDTH // 2 - title.get_width() // 2, 180)
        )

        self.draw_button(self.hvh_btn, "Human vs Human")
        self.draw_button(self.hvm_btn, "Human vs Machine")

    # =========================================================
    # HEADER
    # =========================================================

    def draw_header(self):

        pygame.draw.rect(self.screen, BAR, (0, 0, WIDTH, TOP_BAR))

        text = self.get_status_text()

        label = self.font.render(text, True, TEXT)

        self.screen.blit(
            label,
            (WIDTH // 2 - label.get_width() // 2, 35)
        )

        self.draw_button(self.home_btn, "Home")
        self.draw_button(self.restart_btn, "Restart")

    def get_status_text(self):

        if self.winner == 1:
            return "X Wins"

        if self.winner == -1:
            return "O Wins"

        if self.game_over:
            return "Draw"

        return f"Turn : {'X' if self.current_player == 1 else 'O'}"

    # =========================================================
    # BOARD RENDERING
    # =========================================================

    def draw_grid(self):

        offset = TOP_BAR

        for i in range(1, GRID):

            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (0, offset + i * CELL),
                (WIDTH, offset + i * CELL),
                6
            )

            pygame.draw.line(
                self.screen,
                GRID_COLOR,
                (i * CELL, offset),
                (i * CELL, offset + BOARD_SIZE),
                6
            )

    def draw_marks(self):

        board = self.game.board
        offset = TOP_BAR

        for r in range(GRID):
            for c in range(GRID):

                cx = c * CELL + CELL // 2
                cy = offset + r * CELL + CELL // 2

                if board[r, c] == 1:
                    self.draw_x(cx, cy)

                elif board[r, c] == -1:
                    self.draw_o(cx, cy)

    def draw_x(self, x, y):

        d = CELL // 3

        pygame.draw.line(self.screen, X_COLOR, (x - d, y - d), (x + d, y + d), 10)
        pygame.draw.line(self.screen, X_COLOR, (x - d, y + d), (x + d, y - d), 10)

    def draw_o(self, x, y):

        pygame.draw.circle(self.screen, O_COLOR, (x, y), CELL // 3, 10)

    # =========================================================
    # WIN LINE
    # =========================================================

    def draw_win_line(self):

        if not self.win_line:
            return

        (r1, c1), (r2, c2) = self.win_line

        offset = TOP_BAR

        x1 = c1 * CELL + CELL // 2
        y1 = offset + r1 * CELL + CELL // 2

        x2 = c2 * CELL + CELL // 2
        y2 = offset + r2 * CELL + CELL // 2

        pygame.draw.line(self.screen, WIN_LINE, (x1, y1), (x2, y2), 12)

    # =========================================================
    # GAME LOGIC
    # =========================================================

    def after_move(self):

        winner = self.game.check_winner(self.game.board)

        if winner != 0:

            self.winner = winner
            self.win_line = self.detect_win_line()
            self.game_over = True
            return

        if self.game.is_draw(self.game.board):

            self.game_over = True
            return

        self.current_player *= -1

        if self.mode == "HVM" and self.current_player == 1:
            self.start_machine_turn()

    def detect_win_line(self):

        b = self.game.board

        for r in range(3):
            if abs(sum(b[r, :])) == 3:
                return ((r, 0), (r, 2))

        for c in range(3):
            if abs(sum(b[:, c])) == 3:
                return ((0, c), (2, c))

        if abs(b[0, 0] + b[1, 1] + b[2, 2]) == 3:
            return ((0, 0), (2, 2))

        if abs(b[0, 2] + b[1, 1] + b[2, 0]) == 3:
            return ((0, 2), (2, 0))

        return None

    # =========================================================
    # MACHINE
    # =========================================================

    def start_machine_turn(self):

        self.machine_thinking = True
        self.machine_start = pygame.time.get_ticks()

    def update_machine(self):

        if not self.machine_thinking:
            return

        if pygame.time.get_ticks() - self.machine_start < MACHINE_DELAY:
            return

        actions = self.game.get_valid_actions(self.game.board)

        if actions:
            # move = random.choice(actions)
            idx = self.train.board_to_index(self.game.board, computer=self.current_player)
            move = self.train.policy[str(idx)]
            self.game.board = self.game.make_move(self.game.board, move, self.current_player)

        self.machine_thinking = False
        self.after_move()

    # =========================================================
    # INPUT
    # =========================================================

    def handle_click(self, pos):

        if self.home_btn.collidepoint(pos):
            self.reset()
            self.state = "MENU"
            return

        if self.restart_btn.collidepoint(pos):
            self.reset()
            return

        if self.game_over:
            return

        x, y = pos

        if not (TOP_BAR <= y <= TOP_BAR + BOARD_SIZE):
            return

        col = x // CELL
        row = (y - TOP_BAR) // CELL

        action = row * 3 + col

        self.game.board = self.game.make_move(self.game.board, action, self.current_player)
        self.after_move()

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.game.reset()

        self.current_player = -1
        self.winner = 0
        self.win_line = None
        self.game_over = False

        self.machine_thinking = False

    # =========================================================
    # MAIN LOOP
    # =========================================================

    def run(self):

        while True:

            self.screen.fill(BG)

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:

                    if self.state == "MENU":
                        self.handle_menu_click(event.pos)

                    else:
                        self.handle_click(event.pos)

            if self.state == "MENU":
                self.draw_menu()

            else:
                self.update_machine()
                self.draw_header()
                self.draw_grid()
                self.draw_marks()
                self.draw_win_line()

            pygame.display.update()
            self.clock.tick(60)

    # =========================================================
    # MENU INPUT
    # =========================================================

    def handle_menu_click(self, pos):

        if self.hvh_btn.collidepoint(pos):
            self.mode = "HVH"
            self.state = "GAME"

        elif self.hvm_btn.collidepoint(pos):
            self.mode = "HVM"
            self.state = "GAME"


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    ui = TicTacToeUI()
    ui.run()