import torch
from connect4 import Connect4
from agentes import RandomAgent, DefenderAgent
from principal_new import TrainedAgent

def evaluar(model_path, n_partidas=200, opponent="defender", device="cpu"):
    """
    Evalúa un modelo entrenado jugando Connect4 contra un oponente.
    Devuelve el winrate (victorias / n_partidas).
    """
    # elegir oponente
    if opponent == "random":
        oponente = RandomAgent("Random")
    elif opponent == "defender":
        oponente = DefenderAgent("Defender")
    else:
        raise ValueError("Oponente debe ser 'random' o 'defender'")

    # cargar agente entrenado
    agente = TrainedAgent(
        model_path=model_path,
        state_shape=(6, 7),
        n_actions=7,
        device=device
    )

    # estadísticas
    victorias, derrotas, empates = 0, 0, 0

    for i in range(n_partidas):
        if i % 2 == 0:
            juego = Connect4(agent1=agente, agent2=oponente)
            ganador = juego.play(render=False)
            if ganador == 1: victorias += 1
            elif ganador == 2: derrotas += 1
            else: empates += 1
        else:
            juego = Connect4(agent1=oponente, agent2=agente)
            ganador = juego.play(render=False)
            if ganador == 2: victorias += 1
            elif ganador == 1: derrotas += 1
            else: empates += 1

    winrate = victorias / n_partidas
    print(f"Resultados contra {opponent.capitalize()}Agent en {n_partidas} partidas:")
    print(f"  Victorias: {victorias}")
    print(f"  Derrotas: {derrotas}")
    print(f"  Empates: {empates}")
    print(f"  Winrate: {winrate:.2%}")
    return winrate
