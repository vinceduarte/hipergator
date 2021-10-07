from operator import index
import numpy as np
from Classifiers_and_Regressors import runRegressor
from sklearn.model_selection import train_test_split

def print_board(grid):
    # using string interpolation to output the game board
    # board design credited to https://www.askpython.com/python/examples/tic-tac-toe-using-python

    board = []

    for x in grid:
        if (x == 1):
            board.append('X')
        elif (x == -1):
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

def check_pos(grid, pos):
    if (grid[pos] == 0):
        return True
    else:
        return False

def check_win(board):
    # win logic credited to https://codereview.stackexchange.com/questions/24764/tic-tac-toe-victory-check
    # returns true if columns or diagonals in matrix are same value
    if board[0] == board[3] == board[6] != 0:
        return True
    if board[1] == board[4] == board[7] != 0:
        return True
    if board[2] == board[5] == board[8] != 0:
        return True
    if board[0] == board[1] == board[2] != 0:
        return True
    if board[3] == board[4] == board[5] != 0:
        return True    
    if board[6] == board[7] == board[8] != 0:
        return True 
    if board[0] == board[4] == board[8] != 0:
        return True 
    if board[2] == board[4] == board[6] != 0:
        return True 

    return False          

def check_draw(board):
    # returns true only if all spots on the board are full
    for x in board:
        if (x == 0):
            return False

    return True

def user_input(grid):
    valid = False
    while valid is False:
        pos = int(input("Enter a position: "))
        if (pos in range(0,9)):
                if check_pos(grid, pos):
                    valid = True
                    return pos
                else:
                    print("Not allowed.")
        else:
            print("Not a valid position.")

def main():
    # numpy matrix for Tic-Tac-Toe board
    print("Loading...")

    multi = np.loadtxt('tictac_multi.txt')

    # optimal play multi-label
    x_multi = multi[:, :9]
    y_multi = multi[:, 9:]

    rng = np.random.RandomState(0)
    x_train_multi, x_test_multi, y_train_multi, y_test_multi = train_test_split(
        x_multi, y_multi, test_size=0.2, random_state=rng)

    reg = runRegressor(x_train_multi, x_test_multi,
                     y_train_multi, y_test_multi, 2, k=3)

    print("Press any button to start.")
    input("")
    grid = np.zeros(shape=(9), dtype=int)

    # print reference for positions
    print("\n")
    print("Position reference:")
    print("\t     |     |")
    #print("\t  {}  |  {}  |  {}".format(grid[0,0], grid[0,1], grid[0,2]))
    print("\t  {}  |  {}  |  {}".format("0", "1", "2"))
    print('\t_____|_____|_____')
    print("\t     |     |")
    #print("\t  {}  |  {}  |  {}".format(grid[1,0], grid[1,1], grid[1,2]))
    print("\t  {}  |  {}  |  {}".format("3", "4", "5"))
    print('\t_____|_____|_____')
    print("\t     |     |")
    #print("\t  {}  |  {}  |  {}".format(grid[2,0], grid[2,1], grid[2,2]))
    print("\t  {}  |  {}  |  {}".format("6", "7", "8"))
    print("\t     |     |")
    print("\n")
    
    # game logic loops while True
    game_state = True
    while game_state:

        print("Player 1 turn.")
        pos = user_input(grid)
        grid[pos] = 1

        print_board(grid)
        
        # check if previous move was a win or draw
        if check_win(grid):
            print("Player 1 wins!")
            game_state = False
            break
        else: 
            if check_draw(grid):
                print("Draw!")
                game_state = False
                break   

        print("Player 2 turn.")
        reg_predict = reg.predict([grid])
        pos = np.unravel_index(reg_predict.argmax(), reg_predict.shape)
        max = pos[1]
        while(check_pos(grid, max) is False):
            reg_predict[0][max] = 0
            pos = np.unravel_index(reg_predict.argmax(), reg_predict.shape)
            max = pos[1]
        
        grid[max] = -1

        print_board(grid)
        
        # check if previous move was a win or draw
        if check_win(grid):
            print("Player 2 wins!")
            game_state = False
            break
        else: 
            if check_draw(grid):
                print("Draw!")
                game_state = False
                break

main()
