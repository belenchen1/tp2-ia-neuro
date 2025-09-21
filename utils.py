import numpy as np

def create_board(rows:int=6, cols:int=7) -> np.ndarray:
    ''' Construye un tablero vacío (con todos 0) de tamaño row x cols. '''
    return np.zeros((rows, cols), dtype=int)

def insert_token(board:np.ndarray, col:int, player:int):
    ''' Coloca la ficha del jugador (1 o 2) en la columna seleccionada.
        La ficha se coloca en la primera fila vacía de la columna de board. '''
    for row in reversed(range(board.shape[0])):
        if board[row, col] == 0:
            break
    board[row, col] = player

def check_game_over(board:np.ndarray) -> tuple[bool,int]:
    ''' Revisa en el tablero si el juego terminó.
        Devuelve: (False, None) si nadie ganó aún y el juego sigue.
                  (True, None) el tablero está lleno: empate.
                  (True, jugador) si ganó el jugador (1 o 2). '''
    rows, cols = board.shape
    # Comprobación horizontal
    for row in range(rows):
        for col in range(cols - 3):
            if board[row, col] == board[row, col + 1] == board[row, col + 2] == board[row, col + 3] != 0:
                return True, board[row, col]
    # Comprobación vertical
    for col in range(cols):
        for row in range(rows - 3):
            if board[row, col] == board[row + 1, col] == board[row + 2, col] == board[row + 3, col] != 0:
                return True, board[row, col]
    # Comprobación diagonal (diagonal positiva)
    for row in range(rows - 3):
        for col in range(cols - 3):
            if board[row, col] == board[row + 1, col + 1] == board[row + 2, col + 2] == board[row + 3, col + 3] != 0:
                return True, board[row, col]
    # Comprobación diagonal (diagonal negativa)
    for row in range(3, rows):
        for col in range(cols - 3):
            if board[row, col] == board[row - 1, col + 1] == board[row - 2, col + 2] == board[row - 3, col + 3] != 0:
                return True, board[row, col]
    # Si nadie ganó, revisar ei el tablero está lleno.
    if np.all(board != 0):
        return True, None  # empate
    else:
        return False, None # el juego sigue 

def count_n_in_a_row(board, player, n):
    """
    Cuenta cuántas secuencias de longitud n tiene 'player' en el tablero.
    """
    rows, cols = board.shape
    count = 0

    # Horizontal
    for r in range(rows):
        for c in range(cols - n + 1):
            if np.all(board[r, c:c+n] == player):
                count += 1

    # Vertical
    for r in range(rows - n + 1):
        for c in range(cols):
            if np.all(board[r:r+n, c] == player):
                count += 1

    # Diagonal positiva
    for r in range(rows - n + 1):
        for c in range(cols - n + 1):
            if all(board[r+i, c+i] == player for i in range(n)):
                count += 1

    # Diagonal negativa
    for r in range(n - 1, rows):
        for c in range(cols - n + 1):
            if all(board[r-i, c+i] == player for i in range(n)):
                count += 1

    return count
