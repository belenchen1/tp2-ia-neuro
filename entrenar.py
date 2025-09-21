import torch
from principal import Connect4Environment, Connect4State, DeepQLearningAgent, TrainedAgent
from agentes import Agent, DefenderAgent, RandomAgent
from connect4 import Connect4
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def entrenar(episodes:int=5000,
            gamma:float=0.95, 
            epsilon_start:float=1.0, 
            epsilon_min:float=0.1, 
            epsilon_decay:float=0.98,
            alpha:float=0.001,
            batch_size:int=64, 
            memory_size:int=500,
            target_update_every:int=100,
            opponent:Agent=None,
            verbose:bool=True):

    ''' Entrenar un Agente DQN en la cantidad de episodios y con los 
        parámetros indicados. 
        Entrena jugando contra el agente opponent, si está definido.
        Si opponent==None, entrena jugando contra sí mismo. '''

    nombre_oponente:str = 'None' if opponent==None else opponent.name
    model_name:str = f"trained_model_vs_{nombre_oponente}_{episodes}_{gamma}_" + \
                     f"{epsilon_start}_{epsilon_min}_{epsilon_decay}" + \
                     f"{alpha}_{batch_size}_{memory_size}_{target_update_every}"
    if verbose: print(model_name, flush=True)
    
    # Inicialización del ambiente
    env:Connect4Environment = Connect4Environment()
    
    # Inicialización del agente
    agent:Agent = DeepQLearningAgent(
        state_shape=(env.rows, env.cols),
        n_actions=env.cols,
        device=device,
        gamma=gamma, 
        lr=alpha, 
        batch_size=batch_size, 
        target_update_every=target_update_every, 
        epsilon_decay=epsilon_decay
    )
    agent.epsilon = epsilon_start
    agent.epsilon_min = epsilon_min

    # Para guardar métricas
    history = {"episodes": [], "loss": [], "epsilon": []}

    # Entrenamiento
    for episode in range(episodes):
        state:Connect4State = env.reset()
        done:bool = False
        episode_losses = []  
        dqn_player = 1  # DQN siempre es jugador 1
    
        while not done:
            valid_actions = env.available_actions()
            if env.state.current_player==dqn_player or opponent==None:
                # Turno del DQN (o no hay oponente)
                action = agent.select_action(state, valid_actions)
                next_state, reward, done, _ = env.step(action)
                # Solo almacenar experiencias cuando DQN juega
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step() 
                if loss is not None:
                    episode_losses.append(loss)
            else: 
                # Turno del oponente
                action = opponent.play(state, valid_actions)
                next_state, reward, done, _ = env.step(action)
    
            state = next_state
    
        agent.update_epsilon() 
    
        if (episode + 1) % 100 == 0:
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
            if verbose:
                print(f"Episodio {episode + 1} | Epsilon: {agent.epsilon:.4f} | Loss promedio: {avg_loss:.6f}", flush=True)

            # Guardamos métricas
            history["episodes"].append(episode + 1)
            history["loss"].append(avg_loss)
            history["epsilon"].append(agent.epsilon)

    if verbose: print(flush=True)
    
    torch.save(agent.q_network.state_dict(), f"{model_name}.pth")

    return model_name, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenar un agente usando DQL en el ambiente de 'Connect4'.")

    # Agregar argumentos
    parser.add_argument('-n', '--episodes', type=int, default=1000, help='Número de episodios para entrenar al agente (default: 1000)')
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help='Factor de descuento (default: 0.99)')
    parser.add_argument('-es', '--epsilon_start', type=float, default=1.0, help='Valor inicial de la tasa de exploración (default: 1.0)')
    parser.add_argument('-em', '--epsilon_min', type=float, default=0.1, help='Valor mínimo de la tasa de exploración (default: 0.1)')
    parser.add_argument('-ed', '--epsilon_decay', type=float, default=0.995, help='Decay de la tasa de exploración (default: 0.995)')
    parser.add_argument('-a', '--alpha', type=float, default=0.001, help='Tasa de aprendizaje (default: 0.001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Tamaño del batch usado para aprendizaje (default: 128)')
    parser.add_argument('-ms', '--memory_size', type=int, default=1000, help='Tamaño de la memoria de experiencias del agente (default: 1000)')
    parser.add_argument('-ue', '--target_update_every', type=int, default=100, help='Cada cuánto actualizar red objetivo (default: 100)')
    parser.add_argument('-of', '--opponent_model_path', type=str, help='Archivo pth del agente DQN para usar como oponente.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Activar modo verbose para ver más detalles durante el entrenamiento')

    # Parsear los argumentos
    args = parser.parse_args()
    
    if args.opponent_model_path != None:
        agente_entrenado:Agent = TrainedAgent(
            model_path=args.opponent_model_path,
            state_shape=(6, 7),
            n_actions=7,
            device=device
        )
    else:
        # ACA VA EL OPONENTE (si es contra si mismo, dejar en None)
        # agente_entrenado = None
        # agente_entrenado = RandomAgent("Random")
        agente_entrenado = DefenderAgent("Defender")

    # Llamar a la función principal con los argumentos proporcionados
    entrenar(episodes=args.episodes, 
            gamma=args.gamma, 
            epsilon_start=args.epsilon_start, 
            epsilon_min=args.epsilon_min, 
            epsilon_decay=args.epsilon_decay,
            alpha=args.alpha,
            batch_size=args.batch_size, 
            memory_size=args.memory_size,
            target_update_every=args.target_update_every,
            opponent=agente_entrenado,
            verbose=args.verbose)
