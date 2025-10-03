import torch
from entrenar import entrenar
from evaluar import evaluar
from agentes import RandomAgent, DefenderAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("===== Fase 1: Entrenamiento contra RandomAgent =====")
    random_opponent = RandomAgent("Random")
    model_random, history_random = entrenar(
        episodes=5000,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.999,
        alpha=0.003,
        batch_size=128,
        memory_size=5000,
        target_update_every=100,
        opponent=random_opponent,
        verbose=True
    )

    # Evaluar modelo entrenado contra Random
    wr_random_vs_random = evaluar(f"{model_random}.pth", n_partidas=200, opponent="random", device=device)
    wr_random_vs_defender = evaluar(f"{model_random}.pth", n_partidas=200, opponent="defender", device=device)

    print("\n===== Fase 2: Entrenamiento contra DefenderAgent (continuación) =====")
    defender_opponent = DefenderAgent("Defender")
    model_defender, history_defender = entrenar(
        episodes=3000,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.999,
        alpha=0.001,
        batch_size=128,
        memory_size=5000,
        target_update_every=100,
        opponent=defender_opponent,
        verbose=True
    )

    # Evaluar modelo entrenado contra Defender
    wr_defender_vs_random = evaluar(f"{model_defender}.pth", n_partidas=200, opponent="random", device=device)
    wr_defender_vs_defender = evaluar(f"{model_defender}.pth", n_partidas=200, opponent="defender", device=device)

    # Mostrar tabla comparativa en Markdown
    print("\n\n===== Resultados Comparativos =====")
    print("| Modelo entrenado contra | Eval vs Random | Eval vs Defender |")
    print("|--------------------------|----------------|------------------|")
    print(f"| RandomAgent             | {wr_random_vs_random:.2%}       | {wr_random_vs_defender:.2%}         |")
    print(f"| DefenderAgent (fase 2)  | {wr_defender_vs_random:.2%}       | {wr_defender_vs_defender:.2%}         |")

if __name__ == "__main__":
    main()
