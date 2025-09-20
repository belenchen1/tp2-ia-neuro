import unittest
from utils import create_board, insert_token, check_game_over
import numpy as np

class TestUtils(unittest.TestCase):

    def test_create_board_standard(self):
        b1 = create_board()
        b2 = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        self.assertTrue((b1==b2).all())
        
    def test_create_board_chico(self):
        b1 = create_board(2,4)
        b2 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        self.assertTrue((b1==b2).all())
    
    
    def test_insert_token(self):
        b1 = create_board()
        for _ in range(3):
            insert_token(b1, 0, 1)
            insert_token(b1, 0, 2)
        b2 = np.array([
            [2, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
        ])
        self.assertTrue((b1==b2).all())
        insert_token(b1, 1, 1)
        insert_token(b1, 2, 2)
        insert_token(b1, 3, 1)
        insert_token(b1, 4, 2)
        insert_token(b1, 5, 1)
        b2 = np.array([
            [2, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0],
            [1, 1, 2, 1, 2, 1, 0],
        ])
        self.assertTrue((b1==b2).all())
        for _ in range(3):
            insert_token(b1, 6, 2)
            insert_token(b1, 6, 1)
        b2 = np.array([
            [2, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 2],
            [2, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 2],
            [2, 0, 0, 0, 0, 0, 1],
            [1, 1, 2, 1, 2, 1, 2],
        ])
        self.assertTrue((b1==b2).all())

    def test_check_game_over_vacio(self):
        board = create_board()
        self.assertEqual(check_game_over(board), (False, None))

    def test_check_game_over_nadie(self):
        board = np.array([
            [2, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 2],
            [2, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 2],
            [2, 0, 0, 0, 0, 0, 1],
            [1, 1, 2, 1, 2, 1, 2],
        ])
        self.assertEqual(check_game_over(board), (False, None))

    def test_check_game_over_horizontal1(self):
        board = np.array([
            [1, 0, 0, 0, 0, 0, 1],
            [2, 0, 2, 0, 0, 0, 2],
            [2, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 2, 2, 2, 1],
            [2, 2, 2, 1, 2, 1, 1],
            [1, 2, 1, 1, 2, 1, 2],
        ])
        self.assertEqual(check_game_over(board), (True, 1))

    def test_check_game_over_horizontal2(self):
        board = np.array([
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 2],
            [2, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [2, 0, 0, 0, 0, 0, 1],
            [1, 2, 2, 2, 2, 1, 2],
        ])
        self.assertEqual(check_game_over(board), (True, 2))
    
    def test_check_game_over_diagonal1(self):
        board = np.array([
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 2, 0, 0, 0, 2],
            [2, 1, 1, 2, 1, 0, 1],
            [1, 1, 1, 2, 2, 2, 1],
            [2, 2, 2, 1, 2, 1, 1],
            [1, 2, 1, 1, 2, 1, 2],
        ])
        self.assertEqual(check_game_over(board), (True, 1))

    def test_check_game_over_vertical2(self):
        board = np.array([
            [1, 0, 0, 0, 0, 0, 1],
            [2, 0, 2, 0, 0, 0, 2],
            [2, 1, 1, 1, 2, 0, 1],
            [1, 1, 1, 2, 2, 2, 1],
            [2, 2, 2, 1, 2, 1, 1],
            [1, 2, 1, 1, 2, 1, 2],
        ])
        self.assertEqual(check_game_over(board), (True, 2))
        
    def test_check_game_over_diagonal2(self):
        board = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 0, 0, 0],
            [2, 1, 1, 2, 1, 0, 2],
            [1, 1, 1, 2, 2, 2, 1],
            [2, 2, 2, 1, 2, 1, 1],
            [1, 2, 1, 2, 2, 1, 2],
        ])
        self.assertEqual(check_game_over(board), (True, 2))
        
    def test_check_game_over_empate(self):
        board = np.array([
            [1, 1, 1, 2, 2, 1, 1],
            [2, 2, 2, 1, 1, 1, 2],
            [2, 1, 1, 2, 1, 2, 1],
            [1, 1, 1, 2, 2, 2, 1],
            [2, 2, 2, 1, 2, 1, 1],
            [1, 2, 1, 2, 2, 1, 2],
        ])
        self.assertEqual(check_game_over(board), (True, None))

if __name__ == "__main__":
    unittest.main()