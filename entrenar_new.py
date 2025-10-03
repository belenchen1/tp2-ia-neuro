import torch
from principal_new import Connect4Environment, Connect4State, DeepQLearningAgent
from agentes import Agent
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def entrenar(episodes=5000,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.999,
            alpha=3e-4,
            batch_size=128,
            memory_size=5000,
            target_update_every=100,
            opponent:Agent=None,
            verbose=True,
            pretrained_path=None):

    nombre_oponente = 'None' if opponent is None else opponent.name
    model_name = f"trained_model_vs_{nombre_oponente}_{episodes}_{gamma}_" + \
                 f"{epsilon_start}_{epsilon_min}_{epsilon_decay}" + \
                 f"{alpha}_{batch_size}_{memory_size}_{target_update_every}"
    if verbose: print(model_name, flush=True)

    env = Connect4Environment()
    agent = DeepQLearningAgent(
        state_shape=(env.rows, env.cols),
        n_actions=env.cols,
        device=device,
        gamma=gamma,
        lr=alpha,
        batch_size=batch_size,
        memory_size=memory_size,
        target_update_every=target_update_every,
        epsilon_decay=epsilon_decay,
        loss="huber"
    )
    agent.epsilon = epsilon_start
    agent.epsilon_min = epsilon_min

    if pretrained_path is not None:
        agent.q_network.load_state_dict(torch.load(pretrained_path, map_location=device))
        agent.target_network.load_state_dict(agent.q_network.state_dict())
        print(f"✅ Modelo pre-entrenado cargado desde {pretrained_path}")

    history = {"episodes": [], "loss": [], "epsilon": []}

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_losses = []
        dqn_player = 1
    
        while not done:
            valid_actions = env.available_actions()
            if env.state.current_player == dqn_player or opponent is None:
                action = agent.select_action(state, valid_actions)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
            else:
                action = opponent.play(state, valid_actions)
                next_state, reward, done, _ = env.step(action)

            state = next_state
    
        agent.update_epsilon()
    
        if (episode + 1) % 100 == 0:
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
            if verbose:
                print(f"Episodio {episode+1} | Epsilon: {agent.epsilon:.4f} | Loss promedio: {avg_loss:.6f}")
            history["episodes"].append(episode+1)
            history["loss"].append(avg_loss)
            history["epsilon"].append(agent.epsilon)

    if verbose: print()
    torch.save(agent.q_network.state_dict(), f"{model_name}.pth")
    return model_name, history
