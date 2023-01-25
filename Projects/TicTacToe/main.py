# Tic-Tac-Toe

import os
import random

INFINITE = 10000
loop = 0


def print_ttt(list):
    for i in range(len(list)):
        for j in range(len(list)):
            print("", list[i][j], "|", end="")
        print("\b \b")

        if i < len(list) - 1:
            for k in range(4 * len(list) - 1):
                print("-", end="")

        print("\n", end="")


class board:

    p1 = "X"
    p2 = "O"
    players = {"p1": "X", "p2": "O"}

    def __init__(self, brd_dim):
        self.state = [[" " for i in range(brd_dim)] for i in range(brd_dim)]
        self.dim = brd_dim

    def nextfree(self):
        free_list = []
        for i in range(self.dim):
            for j in range(self.dim):
                if self.state[i][j] == " ":
                    return True

        return False

    def hvh(self, turn):

        os.system("cls")
        print_ttt(self.state)
        # pos_y, pos_x = input("Enter coordinates to fill: ").split()
        pos_x, pos_y = map(int, input("Enter coordinates to fill: ").split(","))

        # Check that coordinates are in range-
        if pos_x > self.dim or pos_y > self.dim:
            return self.hvh(turn)

        # Check for overlapping boxes-
        if self.state[pos_y][pos_x] == " ":
            self.state[pos_y][pos_x] = turn
        else:
            return self.hvh(turn)

        # Check if current player won-
        if self.checker(turn):
            os.system("cls")
            print_ttt(self.state)
            print(f"{turn} has won!")
            return turn
        if turn == self.p1:
            return self.hvh(self.p2)
        else:
            return self.hvh(self.p1)

    def hvai(self, turn):

        has_won = None

        if turn == self.p1:
            # Human Turn-

            os.system("cls")
            print_ttt(self.state)

            # pos_y, pos_x = input("Enter coordinates to fill: ").split()
            pos_x, pos_y = map(int, input("Enter coordinates to fill: ").split(","))

            # Check that coordinates are in range-

            if pos_x >= self.dim or pos_y >= self.dim:
                return self.hvai()

            if self.state[pos_y][pos_x] == " ":
                self.state[pos_y][pos_x] = turn
            else:
                return self.hvai()

            has_won = board.checker(self, turn)

            if has_won:
                print_ttt(self.state)
                print(f"{turn} has won!")
                return turn

            if self.nextfree() is False:
                os.system("cls")
                print_ttt(self.state)
                print("It is a tie")
                return

            return board.hvai(self, self.p2)

        # AI turn

        if turn == self.p2:

            os.system("cls")
            print_ttt(self.state)

            turn = self.p2

            best_move = board.findBestMove(self)

            self.state[best_move[0]][best_move[1]] = turn

            has_won = board.checker(self, turn)

            if has_won:
                print_ttt(self.state)
                print(f"{turn} has won!")
                return turn

            if self.nextfree() is False:
                os.system("cls")
                print_ttt(self.state)
                print("It is a tie")
                return

            return board.hvai(self, self.p1)

    def checker(self, player):

        rows_lost = 0
        cols_lost = 0
        p_won_d1 = 1
        p_won_d2 = 1

        # for player in ["X", "O"]:
        for i in range(self.dim):
            for j in range(self.dim):
                if self.state[i][j] != player:
                    rows_lost += 1
                    break
        if rows_lost < self.dim:
            return True

        for i in range(self.dim):
            for j in range(self.dim):
                if self.state[j][i] is not player:
                    cols_lost += 1
                    break
        if cols_lost < self.dim:
            return True

        for i in range(self.dim):
            if self.state[i][i] is not player:
                p_won_d1 = 0
                break

        for i in range(self.dim):
            if self.state[i][self.dim - 1 - i] is not player:
                p_won_d2 = 0
                break

        return p_won_d1 or p_won_d2

    def minmax(self, is_max):

        # global loop
        # print(loop)
        # loop += 1

        if self.checker(self.p1):
            return 10

        if self.checker(self.p2):
            return -10

        if self.nextfree() is False:
            return 0

        # Maximizing Player:

        if is_max:

            best_eva = -INFINITE

            for i in range(self.dim):
                for j in range(self.dim):
                    if self.state[i][j] == " ":
                        self.state[i][j] = self.p1
                        eva = self.minmax(False)
                        best_eva = max(eva, best_eva)
                        self.state[i][j] = " "

            return best_eva

        # Minimizing Player

        else:

            best_eva = INFINITE

            for i in range(self.dim):
                for j in range(self.dim):
                    if self.state[i][j] == " ":
                        self.state[i][j] = self.p2
                        eva = self.minmax(True)
                        best_eva = min(eva, best_eva)
                        self.state[i][j] = " "

            return best_eva

    def findBestMove(self):
        bestVal = INFINITE
        bestMove = (-1, -1)

        # Traverse all cells, evaluate minimax function for
        # all empty cells. And return the cell with optimal
        # value.
        for i in range(self.dim):
            for j in range(self.dim):

                # Check if cell is empty
                if self.state[i][j] == " ":

                    # Make the move
                    self.state[i][j] = self.p2

                    # compute evaluation function for this
                    # move.

                    # Start next calculation as the maximising player
                    moveVal = self.minmax(True)

                    # Undo the move
                    self.state[i][j] = " "

                    # If the value of the current move is
                    # more than the best value, then update
                    # best/
                    if moveVal < bestVal:
                        bestMove = (i, j)
                        bestVal = moveVal

        # print("The value of the best Move is :", bestVal)
        # print()
        return bestMove


if __name__ == "__main__":

    os.system("cls")
    playBoard = board(3)

    choice1 = input("1. Human vs Human \n2. Human vs AI\n")

    if choice1 is "1":
        os.system("cls")
        choice_human = input("1. Start Random \n2. Start X \n3. Start O\n")

        if choice_human is "1":
            cointoss = random.randint(0, 1)
            if cointoss:
                playBoard.hvh(playBoard.p2)
            else:
                playBoard.hvh(playBoard.p1)

        elif choice_human is "2":
            playBoard.hvh(playBoard.p1)

        else:
            playBoard.hvh(playBoard.p2)

    else:
        os.system("cls")
        choice_ai = input("1. Start Random \n2. Start Human \n3. Start AI\n")

        if choice_ai is "1":
            cointoss = random.randint(0, 1)
            if cointoss:
                playBoard.hvai(playBoard.p2)
            else:
                playBoard.hvai(playBoard.p1)

        elif choice_ai is "2":
            playBoard.hvai(playBoard.p1)

        else:
            playBoard.hvai(playBoard.p2)
