import argparse
from connect4 import Connect4
from agentes import Agent, DefenderAgent, HumanAgent

def main(verbose, human_first=True):
    agente_humano:Agent = HumanAgent("Humano")
    agente_defensor:Agent = DefenderAgent("Defensor")

    if human_first:
        agent1 = agente_humano
        agent2 = agente_defensor
    else:
        agent1 = agente_defensor
        agent2 = agente_humano

    print(f"Juego: {agent1.name} (Jugador 1) vs. {agent2.name} (Jugador 2)")

    juego = Connect4(agent1=agent1, agent2=agent2)
    ganador = juego.play(render=verbose)

    print("\nResultado del juego:")
    if ganador == 0:
        print("Empate")
    elif ganador == 1:
        print(f"Ganó el jugador {agent1.name}")
    else:
        print(f"Ganó el jugador {agent2.name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Jugar una partida de Conecta 4: Humano vs Agente Defensor.")
    parser.add_argument('--human_first', action='store_true', help='El humano primero (Jugador 1)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Mostrar el tablero en cada turno')
    
    args = parser.parse_args()
    main(args.verbose, args.human_first) 
