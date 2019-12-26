#!/usr/bin/env python
# coding: utf-8

# for webscraping, downloading sudokus
from bs4 import BeautifulSoup
import urllib.request

import numpy as np
import random as random
import pandas as pd

class Solver:
    def __init__(self):
        self.current_state = np.zeros([9,9], dtype="int")
        self.candidates = np.ones([9,9,9], dtype="int") # holds all possible candidates for every square
        self.checkpoints = {
                            "candidates": [None]*100, 
                            "current_state": [None]*100, 
                            "prev_candidate": [None]*100
                           }
        self.solution = np.zeros([9,9], dtype="int")
        
    def __update_candidates(self, board_state, candidate_space):
        """ Takes the current board state and the candidate space as input. Checks every non empty 
            square of the sudoku and then removes this value from the candidate space of all squares
            in the 3x3 block, row and column. The new candidate space is then returned. """
        for i in range(9):
            for j in range(9):
                square = board_state[i,j]
                
                # check if square has a value assigned to it
                if square != 0:
                    # remove other candidates for square
                    self.candidates[:,i,j] = np.zeros(9, dtype="int")
                    # remove same candidates from row
                    self.candidates[square-1,i,:] = np.zeros(9, dtype="int") 
                    # remove same candidates from column
                    self.candidates[square-1,:,j] = np.zeros(9, dtype="int")
                    # self.candidates same candidates from 3x3 block
                    candidate_space[square-1,(int(i/3)*3):(int(i/3)*3+3),(int(j/3)*3):(int(j/3)*3+3)] = np.zeros([3,3], dtype="int")
                    # after removing all elements, from row, block and column, 
                    # the square's value is reset to the right candidate
                    self.candidates[square-1,i,j] = 1
        
        return self.candidates
    
    def __update_state(self, state, candidates):
        """ Goes through all the squares and if it's candidate space only has one value left
            sets the square to this value. This new state is then returned. """
        for i in range(9):
            for j in range(9):
                if state[i,j] == 0 and sum(candidates[:,i,j]) == 1:
                    state[i,j] = list(candidates[:,i,j]).index(1)+1 # +1 because index of 1 is 0
                    
        return state
    
    def __fill_blanks(self):
        """ Alternates between reducing the number of candidates in the candidate space and
            fixing the values of the squares if there is only one candidate left for as long
            as the board state changes. If there happens to be a square, where the number 
            of candidates becomes 0 a failure is returned. If not and all squares are filled it
            will return a success. If however there is still open squares that cannot be narrowed
            down by further reduction of the candidate space, it will signal that it is stalling. """
        previous_state = np.zeros([9,9], dtype="int")
        success = "stalling" # no more change in candidate space --> "stall"
        while not (previous_state == self.current_state).all():
            previous_state = self.current_state.copy()
            self.candidates = self.__update_candidates(self.current_state, self.candidates)
            self.current_state = self.__update_state(self.current_state, self.candidates)

        if (np.sum(self.candidates, axis=0) == 0).any(): # checks wether one square has 0 candidates left
            success = "failure"
        if self.current_state.all() != 0: # checks if all squares have a value
            success = "successful"
        return success      
        
    def solve(self, board_state):
        """ Solves the sudoku, first by reducing the candidate space and filling blanks where possible.
            If this methods stalls, a trial and error method is initiated. The square with the least
            amount of candidates is identified and the first one tried. In case the algorithm fails
            (candidate space is 0 for some square), the next candidate is tried. If the algorithm stalls
            again, it will again identify the square with the least amount of candidates and 
            try the first one. This will be repeated until solution is found or no solution is found.
            Checkpoints are created along the way, so in case of a failure while trying out a value,
            the state where this vairable was chosen can be restored and another value tried. """
        failure_depth = 0 # how many times a stall has been produced in a row
        candidate = 0 # what the previously tried candidate was (candidates are tried in ascending order)
        n_itterations = 0 # counts the number of itterations it took to solve the sudoku
        
        self.current_state = board_state
        
        while True:    
            success = self.__fill_blanks()
            print("status after itteration "+ str(n_itterations) + " =", success)
            n_itterations += 1
            
            if success == "successful":
                print("a solution was found!")
                break
            else:
                # 1. create new checkpoint
                if success == "stalling":
                    print("creating new checkpoint at position " + str(failure_depth))
                    
                    self.checkpoints["candidates"][failure_depth] = self.candidates.copy()
                    self.checkpoints["current_state"][failure_depth] = self.current_state.copy()
                    self.checkpoints["prev_candidate"][failure_depth] = candidate
                    
                    failure_depth += 1 # keep track of successive stalls
                    candidate = 0
                    
                # 1. go to previous checkpoint
                if success == "failure":
                    # try next candidate
                    if self.checkpoints["current_state"][0] is not None:
                        print("resetting to checkpoint  at position " + str(failure_depth-1))
                        
                        self.current_state = self.checkpoints["current_state"][failure_depth-1]
                        self.candidates = self.checkpoints["candidates"][failure_depth-1]
                        prev_candidate = self.checkpoints["prev_candidate"][failure_depth-1]
                    # none of the candidates works --> unsolvable
                    else:
                        print("unsolvable")
                        break

                # 2. find squares with least candidates in order
                n_candidates = np.sum(self.candidates, axis=0) # number of candidates of every square
                open_candidates = np.unique(n_candidates)[1:] # lists amounts of open candidates in total
                min_candidates_loc = np.vstack(np.where(n_candidates == min(open_candidates))) # location of squares with minimimum amount of candidates

                # 3. take first square with least candidates
                loc_min = (min_candidates_loc[0,0], min_candidates_loc[1,0])
                
                # 4. set square to first candidate
                candidate = list(self.candidates[:,loc_min[0],loc_min[1]]).index(1,candidate)+1
                print("trying out " + str(candidate) + " from " + str(list(np.arange(1,10)[self.candidates[:,loc_min[0],loc_min[1]] == 1])) + " at " + str((loc_min[0],loc_min[1])))
                self.current_state[loc_min] = candidate
                
        self.solution = self.current_state
        return self.solution


class Sudoku:
    def __init__(self):
        self.current_state = np.zeros([9,9], dtype="int")
        self.target = np.zeros([9,9], dtype="int") # if the imported sudoku has a solution it can be stored here
        self.starting_state = np.zeros([9,9], dtype="int")
        self.solution = np.zeros([9,9], dtype="int")
        
    def set_state(self, board):
        """ Takes a vector, list, string or matrix, that represents a current board state of a sudoku and 
            sets the internal variables to represent this board state. """
        if type(board) is np.ndarray:
            board = np.array(board, dtype="int")
        elif type(board) is str:
            board = np.array(list(board), dtype="int")
        else:
            board = np.array(board)
        if board.shape != (9,9):
            board = np.reshape(board, [9,9])
            
        self.starting_state = board.copy()
        self.current_state = board.copy()
                    
    def download_new(self):
        """ Reinitialises the class and scrapes https://nine.websudoku.com for
            a new sudoku. It then sets the internal current, starting and solution variables
            to match the sudoku found online. """
        self.__init__() # if multiple sudokus are being created successively, the class needs to be initialised again
        with urllib.request.urlopen("https://nine.websudoku.com") as f:
            html_doc = f.read()
        
        soup = BeautifulSoup(html_doc, 'html.parser')
        
        try:
            # scraping html file for the values of the sudoku
            for element in soup.body.table.form.find_all("input"):
                if 'input id="cheat" name="cheat"' in str(element):
                    target = element["value"] # solution of the sudoku
                if 'input id="editmask" type="hidden"' in str(element):
                    # mask is a boolean vector that decides which values 
                    # have to be removed from the target to get the starting state
                    mask = element["value"]

            # parsing target and mask string to create an array for the starting state 
            for i in range(len(mask)):
                self.solution[int(i/9),i%9] = int(target[i])
                if int(mask[i]) == 0: # 0 codes for starting state value
                    self.starting_state[int(i/9),i%9] = int(target[i])
                else:
                    self.starting_state[int(i/9),i%9] = 0

            self.current_state = self.starting_state.copy()
        except AttributeError:
            if "This IP address" in soup.text:
                print("The current IP address has been barred from making further requests to the website")
            else:
                print("something went wrong...")
        
    def solve(self):
        """ Solves the sudoku, with the help of the Solver.solve() method. """
        solver = Solver()
        self.solution = solver.solve(self.current_state)  

