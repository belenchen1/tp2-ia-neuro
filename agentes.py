from abc import ABC, abstractmethod
from random import choice
import utils
import numpy as np

class Agent(ABC):
    @abstractmethod
    def play(self, state, valid_actions):
        '''Dado un estado y acciones válidas, retorna una acción.'''
        pass

class RandomAgent(Agent):
    ''' Agente que juega siempre al azar. '''
    def __init__(self, name: str):
        self.name = name
    def play(self, state, valid_actions):
        return choice(valid_actions)

class HumanAgent(Agent):
    ''' Agente humano que interactúa por consola. '''
    def __init__(self, name: str):
        self.name = name
    def play(self, state, valid_actions):
        return int(input("Ingrese la columna donde desea jugar: "))

class DefenderAgent(Agent):
    ''' Agente que revisa si el oponente está por ganar e intenta bloquearlo;
        si no, juega al azar. '''
    def __init__(self, name: str):
        self.name = name
    def play(self, state, valid_actions):
        me:int = state.current_player
        opponent:int = 3 - me
        for col in valid_actions:
            # Si el oponente ganará al jugar en col, elegimos col para bloquearlo.
            new_board:np.ndarray = np.array(state.board, dtype=int)
            utils.insert_token(new_board, col, opponent)
            if utils.check_game_over(new_board)[0]:
                return col
        # Si el oponente no está por ganar, elegimos al azar.
        return choice(valid_actions)
