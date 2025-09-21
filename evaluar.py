import torch
from connect4 import Connect4
from agentes import RandomAgent, DefenderAgent
from principal import TrainedAgent

def evaluar(model_path, n_partidas=200, opponent="defender", device="cpu"):
    """
    Evalúa un modelo entrenado jugando Connect4 contra un oponente.
    
    Args:
        model_path: ruta al archivo .pth del modelo entrenado
        n_partidas: cantidad de partidas a jugar
        opponent: "random" o "defender"
        device: cpu o cuda
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
        # alternar quién empieza para que sea más justo
        if i % 2 == 0:
            juego = Connect4(agent1=agente, agent2=oponente)
            ganador = juego.play(render=False)
            if ganador == 1:
                victorias += 1
            elif ganador == 2:
                derrotas += 1
            else:
                empates += 1
        else:
            juego = Connect4(agent1=oponente, agent2=agente)
            ganador = juego.play(render=False)
            if ganador == 2:
                victorias += 1
            elif ganador == 1:
                derrotas += 1
            else:
                empates += 1

    print(f"Resultados contra {opponent.capitalize()}Agent en {n_partidas} partidas:")
    print(f"  Victorias: {victorias}")
    print(f"  Derrotas: {derrotas}")
    print(f"  Empates: {empates}")
    print(f"  Winrate: {victorias/n_partidas:.2%}")
'''
if __name__ == "__main__":
    # ejemplo: cargar el modelo final entrenado contra RandomAgent
    path_vs_itself = "trained_model_vs_None_1000_0.99_1.0_0.1_0.9950.001_128_1000_100.pth"
    path_vs_random = ""
    path_vs_defender = "trained_model_vs_Defender_1000_0.99_1.0_0.1_0.9950.001_128_1000_100.pth"
    path_modelo_kaggle = "connect_four_model_fully_trained_rdm.pth"
    
    print("---------- Evaluando modelo entrenado contra sí mismo ----------")
    print("- Evaluado contra RandomAgent")
    evaluar(path_vs_itself, n_partidas=200, opponent="random", device="cuda" if torch.cuda.is_available() else "cpu")
    print("- Evaluado contra DefenderAgent")
    print("\n---------- Evaluando modelo entrenado contra RandomAgent, evaluado contra DefenderAgent? ----------")
    evaluar(path_vs_itself, n_partidas=200, opponent="defender", device="cuda" if torch.cuda.is_available() else "cpu")

    print("---------- Evaluando modelo entrenado contra RandomAgent ----------")
    print("- Evaluado contra RandomAgent")
    evaluar(path_vs_random, n_partidas=200, opponent="random", device="cuda" if torch.cuda.is_available() else "cpu")
    print("- Evaluado contra DefenderAgent")
    print("\n---------- Evaluando modelo entrenado contra RandomAgent, evaluado contra DefenderAgent? ----------")
    evaluar(path_vs_random, n_partidas=200, opponent="defender", device="cuda" if torch.cuda.is_available() else "cpu")

    print("---------- Evaluando modelo entrenado contra DefenderAgent ----------")
    print("- Evaluado contra RandomAgent")
    evaluar(path_vs_defender, n_partidas=200, opponent="random", device="cuda" if torch.cuda.is_available() else "cpu")
    print("- Evaluado contra DefenderAgent")
    print("\n---------- Evaluando modelo entrenado contra RandomAgent, evaluado contra DefenderAgent? ----------")
    evaluar(path_vs_defender, n_partidas=200, opponent="defender", device="cuda" if torch.cuda.is_available() else "cpu")
'''

if __name__ == "__main__":
    modelos = {
        "self": "trained_model_vs_None_1000_0.99_1.0_0.1_0.9950.001_128_1000_100.pth",
        "random": "trained_model_vs_Random_1000_0.99_1.0_0.1_0.9950.001_128_1000_100.pth",
        "defender": "trained_model_vs_Defender_1000_0.99_1.0_0.1_0.9950.001_128_1000_100.pth",
    }

    for nombre, path in modelos.items():
        print(f"\n===== Evaluando modelo entrenado vs {nombre.upper()} =====")
        print("-> Jugando contra RandomAgent")
        evaluar(path, n_partidas=200, opponent="random")
        print("-> Jugando contra DefenderAgent")
        evaluar(path, n_partidas=200, opponent="defender")
        #print("-> Jugando contra sí mismo")
        #evaluar(path, n_partidas=200, opponent="self")
