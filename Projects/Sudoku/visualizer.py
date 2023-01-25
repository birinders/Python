# Sudoku Solver

import sys
import os
import time
import copy

# import winsound

sys.setrecursionlimit(1000)

if len(sys.argv) > 1:
    visualizer_mode = int(sys.argv[1])
# print(sys.argv[0])

loop = 0
backtrack = 0
maxdepth = 0

placer = []

dim = 9
x_divs = 3
y_divs = 3
openers = ["┏━━━", "┯━━━", "┳━━━", "┓"]
closers = ["┗━━━", "┷━━━", "┻━━━", "┛"]


# This is considered the world's hardest sudoku board-
# board = [
#     [8, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 3, 6, 0, 0, 0, 0, 0],
#     [0, 7, 0, 0, 9, 0, 2, 0, 0],
#     [0, 5, 0, 0, 0, 7, 0, 0, 0],
#     [0, 0, 0, 0, 4, 5, 7, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 3, 0],
#     [0, 0, 1, 0, 0, 0, 0, 6, 8],
#     [0, 0, 8, 5, 0, 0, 0, 1, 0],
#     [0, 9, 0, 0, 0, 0, 4, 0, 0],
# ]

# board = [
#     [5, 3, 0, 0, 7, 0, 0, 0, 0],
#     [6, 0, 0, 1, 9, 5, 0, 0, 0],
#     [0, 9, 8, 0, 0, 0, 0, 6, 0],
#     [8, 0, 0, 0, 6, 0, 0, 0, 3],
#     [4, 0, 0, 8, 0, 3, 0, 0, 1],
#     [7, 0, 0, 0, 2, 0, 0, 0, 6],
#     [0, 6, 0, 0, 0, 0, 2, 8, 0],
#     [0, 0, 0, 0, 1, 9, 0, 0, 5],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],
# ]

board = [
    [0, 3, 0, 0, 1, 0, 0, 6, 0],
    [7, 5, 0, 0, 3, 0, 0, 4, 8],
    [0, 0, 6, 9, 8, 4, 3, 0, 0],
    [0, 0, 3, 0, 0, 0, 8, 0, 0],
    [9, 1, 2, 0, 0, 0, 6, 7, 4],
    [0, 0, 4, 0, 0, 0, 5, 0, 0],
    [0, 0, 1, 6, 7, 5, 2, 0, 0],
    [6, 8, 0, 0, 9, 0, 0, 1, 5],
    [0, 9, 0, 0, 4, 0, 0, 3, 0],
]


visualizer_board = copy.deepcopy(board)
for i in range(dim):
    for j in range(dim):
        if visualizer_board[i][j] == 0:
            visualizer_board[i][j] = " "
original_board = copy.deepcopy(visualizer_board)


tester_board = [[1 for i in range(dim)] for j in range(dim)]


def clear():

    if os.name == "nt":
        _ = os.system("cls")

    else:
        _ = os.system("clear")


def color_text(code):
    return f"\33[{code}m".format(code=code)


def print_rows(list):
    for i in list:
        print(i)


def sud_printer(list):
    # \u0305
    DIVPRINT = 9
    flag = 0

    # Main-
    # for line in list:
    #     for unit in line:
    #         if not flag:
    #             print(f"│\u0305\u0332{unit}", end="")
    #         else:
    #             print(f"│\u0332{unit}", end="")
    #     print("│")
    #     flag = 1

    for i in range(dim):
        print("┼───", end="")
    print("┼")

    for i in range(dim):
        for j in range(dim):
            print(f"│ {list[i][j]} ", end="")
        print("│")

        for i in range(dim):
            print("┼───", end="")
        print("┼")


def line_printer(list):

    for i in range(dim):
        if i == 0:
            print(f"{list[0]}", end="")
        elif (i) / 3 != (i) // 3:
            print(f"{list[1]}", end="")
        else:
            print(f"{list[2]}", end="")

    print(
        f"{list[3]}",
    )


def new_printer(list):

    line_printer(["┏━━━", "┯━━━", "┳━━━", "┓"])

    for i in range(dim):
        for j in range(dim):

            outer = ""

            # New-

            if j == 0 and i / 3 != i // 3:
                print(f"┃ {list[i][j]} ", end="")

            elif j / 3 != j // 3:
                print(f"│ {list[i][j]} ", end="")

            elif j / 3 == j // 3:
                print(f"┃ {list[i][j]} ", end="")
        print("┃")

        if i < dim - 1:
            for k in range(dim):
                if k == 0 and ((i + 1) / 3 != (i + 1) // 3 or i == 0):
                    print("┠───", end="")
                    outer = "┨"

                elif k == 0 and (i + 1) / 3 == (i + 1) // 3:
                    print("┣━━━", end="")
                    outer = "┫"

                elif k / 3 != k // 3 and ((i + 1) / 3 != (i + 1) // 3 or i == 0):
                    print("┼───", end="")

                elif k / 3 == k // 3 and ((i + 1) / 3 != (i + 1) // 3 or i == 0):
                    print("╂───", end="")

                elif k / 3 != k // 3 and (i + 1) / 3 == (i + 1) // 3:
                    print("┿━━━", end="")

                elif k / 3 == k // 3 and (i + 1) / 3 == (i + 1) // 3:
                    print("╋━━━", end="")

            print(outer)

    line_printer(["┗━━━", "┷━━━", "┻━━━", "┛"])


def unit_finder(y, x):
    return y_divs * (y // y_divs), x_divs * (x // x_divs)


def next_zero(board):
    for y in range(dim):
        for x in range(dim):
            if board[y][x] == 0:
                return [y, x]

    return False


def is_possible(board, y, x, to_fill):
    range_y, range_x = unit_finder(y, x)

    for i in range(dim):
        if board[y][i] == to_fill:
            return 0

    for i in range(dim):
        if board[i][x] == to_fill:
            return 0

    for i in range(range_y, range_y + y_divs):
        for j in range(range_x, range_x + x_divs):
            if board[i][j] == to_fill:
                return 0

    return 1


def recursive_solver(board, depth=0):
    global loop, backtrack, maxdepth
    if depth > maxdepth:
        maxdepth = depth
    # print(loop)
    loop += 1

    if not next_zero(board):
        return board

    next_y, next_x = next_zero(board)

    for i in range(1, 10):
        # placer.append((next_y, next_x, i))

        # time.sleep(0.1)
        if is_possible(board, next_y, next_x, i):

            # new_board = copier(board)
            board[next_y][next_x] = i
            placer.append((next_y, next_x, i))
            # clear()
            # new_printer(board)

            if recursive_solver(board, depth + 1):
                return board

            board[next_y][next_x] = 0
    placer.append((next_y, next_x, " "))
    backtrack += 1
    return False


def visualizer(list, place):

    time.sleep(5)
    clear()
    print("\n\nSolution-")
    time.sleep(2)
    clear()

    for element in place:
        # list[element[0]][element[1]] = color_text(32) + f"\033[4m{element[2]}\033[0m"
        list[element[0]][element[1]] = color_text(32) + f"{element[2]}" + color_text(37)

        time.sleep(0.02)
        clear()
        new_printer(list)
        # if element[2] == " ":
        # print("\a")
        # winsound.Beep(2500, 100)

        # list[element[0]][element[1]] = color_text(32) + f"\033[4m{element[2]}\033[0m"

    for i in range(5):
        clear()
        new_printer(original_board)
        time.sleep(0.5)
        clear()
        new_printer(visualizer_board)
        time.sleep(0.5)


def anim(method, original, final):
    time.sleep(5)
    clear()
    print("\n\nSolution-")
    time.sleep(2)
    clear()

    method(original)
    time.sleep(2)

    for i in range(dim):
        for j in range(dim):
            if original[i][j] != final[i][j]:
                # original[i][j] == final[i][j]
                original[i][j] = color_text(32) + f"{final[i][j]}" + color_text(37)
                clear()
                method(original)
                time.sleep(0.1)


start = time.time()
solution = recursive_solver(board)
end = time.time()
timetaken = end - start
# solution = solver(board)

clear()

if len(sys.argv) == 1:
    visualizer_mode = 1

if solution:
    print(f"\n\n\n\nSolution Attained in {timetaken} seconds,")
    print(
        f"Algorithm backtracked {backtrack} times, and visited {loop} possible states."
    )
    print(f"Algorithm attained a maximum Recursion depth of {maxdepth}.")

    # new_printer(board)
    if visualizer_mode == -1:
        new_printer(board)
    elif visualizer_mode == 0:
        anim(new_printer, visualizer_board, solution)
    else:
        visualizer(visualizer_board, placer)
else:
    print("No solution/Board Invalid")
