import optuna
import joblib
import random, numpy as np, torch
from principal import Connect4Environment, DeepQLearningAgent
from agentes import RandomAgent
from connect4 import Connect4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_hidden_sizes(width, depth):
    return tuple([width] * depth) if depth > 0 else ()

def quick_eval(agent, games=40):
    """Evalúa rápido contra RandomAgent para usar en early stopping."""
    opponent = RandomAgent("Random")
    wins = 0
    for i in range(games):
        if i % 2 == 0:
            game = Connect4(agent1=agent, agent2=opponent)
            w = game.play(render=False)
            if w == 1: wins += 1
        else:
            game = Connect4(agent1=opponent, agent2=agent)
            w = game.play(render=False)
            if w == 2: wins += 1
    return wins / games

def entrenar_y_evaluar_fast(params, episodes=1000, eval_games=50, verbose=False, seed=42,
                            early_probe_frac=0.3, early_min_wr=0.55):
    """
    Entrena un agente DQN y lo evalúa contra RandomAgent, con early stopping.

    params = (gamma, epsilon_decay, batch_size, memory_size, target_update_every,
              activation, width, depth, loss_name)
    """
    (gamma, epsilon_decay, batch_size, memory_size, target_update_every,
     activation, width, depth, loss_name) = params

    # Semillas reproducibles
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

    env = Connect4Environment()
    opponent = RandomAgent("Random")

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
        loss=loss_name
    )

    probe_ep = max(1, int(early_probe_frac * episodes))
    for ep in range(episodes):
        state = env.reset()
        done = False
        dqn_player = 1
        while not done:
            valid = env.available_actions()
            if env.state.current_player == dqn_player:
                a = agent.select_action(state, valid)
                next_state, reward, done, _ = env.step(a)
                agent.store_transition(state, a, reward, next_state, done)
                agent.train_step()
            else:
                a = opponent.play(state, valid)
                next_state, reward, done, _ = env.step(a)
            state = next_state
        agent.update_epsilon()

        if verbose and ((ep + 1) <= 30 or (ep + 1) % 250 == 0):
            print(f"[Train] Ep {ep+1}/{episodes} | ε={agent.epsilon:.3f}")

        # early stopping check
        if (ep + 1) == probe_ep:
            wr_probe = quick_eval(agent, games=40)
            if verbose:
                print(f"[Probe @ {ep+1}/{episodes}] WR={wr_probe:.2%}")
            if wr_probe < early_min_wr:
                if verbose: print("Early stop: bajo rendimiento inicial.")
                return wr_probe

    # Evaluación completa contra RandomAgent
    wins = draws = losses = 0
    for i in range(eval_games):
        if i % 2 == 0:
            game = Connect4(agent1=agent, agent2=opponent)
            w = game.play(render=False)
            if w == 1: wins += 1
            elif w == 2: losses += 1
            else: draws += 1
        else:
            game = Connect4(agent1=opponent, agent2=agent)
            w = game.play(render=False)
            if w == 2: wins += 1
            elif w == 1: losses += 1
            else: draws += 1

    wr = wins / eval_games
    if verbose:
        print(f"Params: {params} -> Winrate: {wr:.2%} (W:{wins}, L:{losses}, D:{draws})")
    return wr

def objective(trial):
    # Espacio de búsqueda
    gamma = trial.suggest_categorical("gamma", [0.95, 0.99])
    epsilon_decay = trial.suggest_categorical("epsilon_decay", [0.98, 0.99, 0.995])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    memory_size = trial.suggest_categorical("memory_size", [5000, 20000])
    target_update_every = trial.suggest_categorical("target_update_every", [100, 300, 500])

    activation = trial.suggest_categorical("activation", ["relu", "tanh", "leakyrelu", "gelu"])
    width = trial.suggest_categorical("width", [64, 128, 256])
    depth = trial.suggest_categorical("depth", [1, 2, 3])
    loss_name = trial.suggest_categorical("loss", ["mse", "huber"])

    # Llamada a tu función rápida
    wr = entrenar_y_evaluar_fast(
        (gamma, epsilon_decay, batch_size, memory_size, target_update_every,
        activation, width, depth, loss_name),
        episodes=1000,         # << reducimos episodios por trial
        eval_games=50,         # << evaluación corta
        verbose=False,
        seed=1234 + trial.number,
        early_probe_frac=0.3,  # early stop al 30%
        early_min_wr=0.55
    )

    # Reportar a Optuna (permite pruning)
    trial.report(wr, step=1)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return wr

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=50, timeout=2*60*60)

    print("Mejor trial:")
    print("  Winrate:", study.best_value)
    print("  Params :", study.best_params)

    # Guardar el estudio para cargarlo en graficos.py
    joblib.dump(study, "optuna_study.pkl")