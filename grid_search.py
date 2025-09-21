import itertools
import torch
from principal import Connect4Environment, Connect4State, DeepQLearningAgent
from agentes import RandomAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
import numpy as np
import torch

def entrenar_y_evaluar(params, episodes=5000, eval_games=200, verbose=False, seed=42):
    """
    Entrena un agente DQN con los hiperparámetros y arquitectura dados y lo evalúa contra RandomAgent.
    
    params = (
        gamma, epsilon_decay, batch_size, memory_size, target_update_every,
        activation, width, depth, loss_name
    )
    - activation: "relu" | "tanh" | "leakyrelu" | "elu" | "gelu" | "selu"
    - width: int (p.ej. 64, 128, 256)
    - depth: int (número de capas ocultas)
    - loss_name: "mse" | "huber"
    """

    (gamma, epsilon_decay, batch_size, memory_size, target_update_every,
     activation, width, depth, loss_name) = params

    # Helper arquitectura
    def make_hidden_sizes(w, d): 
        return tuple([w] * d) if d > 0 else ()

    # Semillas para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Ambiente y oponente
    env = Connect4Environment()
    opponent = RandomAgent("Random")

    # Agente DQN
    agent = DeepQLearningAgent(
        state_shape=(env.rows, env.cols),
        n_actions=env.cols,
        device=device,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        memory_size=memory_size,
        target_update_every=target_update_every,
        hidden_sizes=make_hidden_sizes(width, depth),
        activation=activation,
        loss=loss_name  # "mse" o "huber" (usa SmoothL1Loss)
    )

    # Entrenamiento
    for episode in range(episodes):
        state = env.reset()
        done = False
        dqn_player = 1  # el agente juega como jugador 1

        while not done:
            valid_actions = env.available_actions()
            if env.state.current_player == dqn_player:
                action = agent.select_action(state, valid_actions)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step()
            else:
                action = opponent.play(state, valid_actions)
                next_state, reward, done, _ = env.step(action)

            state = next_state

        agent.update_epsilon()

        if verbose and (episode + 1) % max(1, (episodes // 10)) == 0:
            print(f"[Train] Ep {episode+1}/{episodes} | ε={agent.epsilon:.3f}")

    # Evaluación contra RandomAgent
    from connect4 import Connect4
    victorias = derrotas = empates = 0
    for i in range(eval_games):
        # alternar quién empieza
        if i % 2 == 0:
            juego = Connect4(agent1=agent, agent2=opponent)
            ganador = juego.play(render=False)
            if ganador == 1: victorias += 1
            elif ganador == 2: derrotas += 1
            else: empates += 1
        else:
            juego = Connect4(agent1=opponent, agent2=agent)
            ganador = juego.play(render=False)
            if ganador == 2: victorias += 1
            elif ganador == 1: derrotas += 1
            else: empates += 1

    winrate = victorias / eval_games
    if verbose:
        print(
            f"Params: gamma={gamma}, eps_decay={epsilon_decay}, batch={batch_size}, "
            f"mem={memory_size}, tgt_upd={target_update_every}, act={activation}, "
            f"width={width}, depth={depth}, loss={loss_name} "
            f"-> Winrate: {winrate:.2%} (W:{victorias}, L:{derrotas}, D:{empates})"
        )

    return winrate


if __name__ == "__main__":
    import itertools

    # Definición de la grilla
    gammas = [0.95, 0.99]
    epsilon_decays = [0.98, 0.99, 0.995]
    batch_sizes = [32, 64]
    memory_sizes = [5000, 20000]
    target_updates = [100, 500]

    activations = ["relu", "tanh", "leakyrelu", "gelu"]
    widths = [64, 128, 256]
    depths = [1, 2, 3]
    losses = ["mse", "huber"]

    param_grid = itertools.product(
        gammas, epsilon_decays, batch_sizes, memory_sizes, target_updates,
        activations, widths, depths, losses
    )

    resultados = []
    for params in param_grid:
        winrate = entrenar_y_evaluar(params, episodes=5000, eval_games=200, verbose=True)
        resultados.append((params, winrate))

    # mostrar las mejores configuraciones
    resultados.sort(key=lambda x: x[1], reverse=True)
    print("\nMejores configuraciones:")
    for params, winrate in resultados[:5]:
        print(f"{params} -> {winrate:.2%}")

