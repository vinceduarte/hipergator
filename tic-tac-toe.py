import numpy as np

def print_board(grid):
    # using string interpolation to output the game board
    # board design credited to https://www.askpython.com/python/examples/tic-tac-toe-using-python

    board = []

    for x in grid:
        for y in x:
            if (y == 1):
                board.append('X')
            elif (y == -1):
                board.append('O')
            else:
                board.append(' ')

    print("\n")
    print("\t     |     |")
    #print("\t  {}  |  {}  |  {}".format(grid[0,0], grid[0,1], grid[0,2]))
    print("\t  {}  |  {}  |  {}".format(board[0], board[1], board[2]))
    print('\t_____|_____|_____')
    print("\t     |     |")
    #print("\t  {}  |  {}  |  {}".format(grid[1,0], grid[1,1], grid[1,2]))
    print("\t  {}  |  {}  |  {}".format(board[3], board[4], board[5]))
    print('\t_____|_____|_____')
    print("\t     |     |")
    #print("\t  {}  |  {}  |  {}".format(grid[2,0], grid[2,1], grid[2,2]))
    print("\t  {}  |  {}  |  {}".format(board[6], board[7], board[8]))
    print("\t     |     |")
    print("\n")

def check_pos(grid, x, y):
    if (grid[x][y] == 0):
        return True
    else:
        return False

def check_win(board, x, y):
    # win logic credited to https://codereview.stackexchange.com/questions/24764/tic-tac-toe-victory-check
    # returns true if columns or diagonals in matrix are same value
    if board[0][y] == board[1][y] == board [2][y]:
        return True
    if board[x][0] == board[x][1] == board [x][2]:
        return True
    if x == y and board[0][0] == board[1][1] == board [2][2]:
        return True
    if x + y == 2 and board[0][2] == board[1][1] == board [2][0]:
        return True

    return False          

def check_draw(board):
    # returns true only if all spots on the board are full
    for x in board:
        for y in x:
            if (y == 0):
                return False

    return True

def user_input(grid):
    valid = False
    while valid is False:
        pos_x = int(input("Enter a row: "))
        if (pos_x in range(0,3)):
            pos_y = int(input("Enter a col: "))
            if (pos_y in range(0,3)):
                if check_pos(grid, pos_x, pos_y):
                    valid = True
                    return pos_x, pos_y
                else:
                    print("Not allowed.")
            else:
                print("Enter a valid position.")
        else:
            print("Enter a valid position.")

def main():
    # numpy matrix for Tic-Tac-Toe board
    grid = np.zeros(shape=(3,3), dtype=int)
    
    # game logic loops while True
    game_state = True
    while game_state:
        print_board(grid)

        print("Player 1 turn.")
        x, y = user_input(grid)
        grid[x][y] = 1
        print_board(grid)
        if check_win(grid, x, y):
            print("Player 1 wins!")
            game_state = False
            break
        else: 
            if check_draw(grid):
                print("Draw!")
                game_state = False
                break   

        print("Player 2 turn.")
        x, y = user_input(grid)
        grid[x][y] = -1
        print_board(grid)
        if check_win(grid, x, y):
            print("Player 2 wins!")
            game_state = False
            break
        else: 
            if check_draw(grid):
                print("Draw!")
                game_state = False
                break

main()
