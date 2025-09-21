from principal import Connect4State,Connect4Environment
from agentes import Agent,RandomAgent

class Connect4:
    #Clase que simula un juego de Connect4 entre dos agentes.
    def __init__(self, rows:int=6, cols:int=7, agent1:Agent=None, agent2:Agent=None): # type: ignore
        self.rows:int = rows
        self.cols:int = cols
        self.env:Connect4Environment = Connect4Environment(rows, cols)
        self.agent1:Agent = agent1 if agent1 is not None else RandomAgent("Agente 1")
        self.agent2:Agent = agent2 if agent2 is not None else RandomAgent("Agente 2")

    def play(self, render:bool=True) -> int:
        ''' Juega un juego completo de Connect4 con dos agentes. 
            Devuelve quién ganó (0==empate, 1 o 2). '''
        state:Connect4State = self.env.reset()
        done:bool = False

        while not done:
            if render: self.env.render()

            valid_actions:list[int] = self.env.available_actions()

            if self.env.state.current_player == 1:
                action = self.agent1.play(state, valid_actions)
            else:
                action = self.agent2.play(state, valid_actions)

            next_state, reward, done, info = self.env.step(action)
            state = next_state

        if render: self.env.render()

        winner:int = info["winner"]
        return winner
