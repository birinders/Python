# Sudoku Solver

import sys
import os

sys.setrecursionlimit(5000)

loop = 0

# This is considered the world's hardest sudoku board-
board = [
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 6, 0, 0, 0, 0, 0],
    [0, 7, 0, 0, 9, 0, 2, 0, 0],
    [0, 5, 0, 0, 0, 7, 0, 0, 0],
    [0, 0, 0, 0, 4, 5, 7, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 3, 0],
    [0, 0, 1, 0, 0, 0, 0, 6, 8],
    [0, 0, 8, 5, 0, 0, 0, 1, 0],
    [0, 9, 0, 0, 0, 0, 4, 0, 0],
]


tester_board = [[1 for i in range(9)] for j in range(9)]


def print_rows(list):
    for i in list:
        print(i)


def unit_finder(y, x):
    return 3 * (y // 3), 3 * (x // 3)


def next_zero(board):
    for y in range(9):
        for x in range(9):
            if board[y][x] == 0:
                return [y, x]

    return False


def is_possible(board, y, x, to_fill):
    range_y, range_x = unit_finder(y, x)

    for i in range(9):
        if board[y][i] == to_fill:
            return 0

    for i in range(9):
        if board[i][x] == to_fill:
            return 0

    for i in range(range_y, range_y + 3):
        for j in range(range_x, range_x + 3):
            if board[i][j] == to_fill:
                return 0

    return 1


def recursive_solver(board):
    global loop
    # print(loop)
    loop += 1

    if not next_zero(board):
        return board

    next_y, next_x = next_zero(board)

    for i in range(1, 10):
        if is_possible(board, next_y, next_x, i):

            # new_board = copier(board)
            board[next_y][next_x] = i

            if recursive_solver(board):
                return board

            board[next_y][next_x] = 0

    return False


solution = recursive_solver(board)
# solution = solver(board)

os.system("cls")

if solution:
    print_rows(solution)
else:
    print("No solution/Board Invalid")
