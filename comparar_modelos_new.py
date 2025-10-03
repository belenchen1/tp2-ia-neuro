import torch
from entrenar_new import entrenar
from evaluar_new import evaluar
from agentes import RandomAgent, DefenderAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("===== Fase 1: Entrenando contra RandomAgent =====")
    model_random, _ = entrenar(
        episodes=3000,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.999,
        alpha=3e-4,
        batch_size=128,
        memory_size=5000,
        target_update_every=100,
        opponent=RandomAgent("Random"),
        verbose=True
    )
    path_random = f"{model_random}.pth"

    wr_random_vs_random = evaluar(path_random, n_partidas=200, opponent="random", device=device)
    wr_random_vs_defender = evaluar(path_random, n_partidas=200, opponent="defender", device=device)

    print("\n===== Fase 2: Continuando entrenamiento contra DefenderAgent =====")
    model_defender, _ = entrenar(
        episodes=3000,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.999,
        alpha=3e-4,
        batch_size=128,
        memory_size=5000,
        target_update_every=100,
        opponent=DefenderAgent("Defender"),
        verbose=True,
        pretrained_path=path_random   # << curriculum real
    )
    path_defender = f"{model_defender}.pth"

    wr_defender_vs_random = evaluar(path_defender, n_partidas=200, opponent="random", device=device)
    wr_defender_vs_defender = evaluar(path_defender, n_partidas=200, opponent="defender", device=device)

    print("\n\n===== Resultados Comparativos =====")
    print("| Modelo (fase)           | Eval vs Random | Eval vs Defender |")
    print("|--------------------------|----------------|------------------|")
    print(f"| Tras Fase 1 (Random)    | {wr_random_vs_random:.2%}       | {wr_random_vs_defender:.2%}         |")
    print(f"| Tras Fase 2 (Defender)  | {wr_defender_vs_random:.2%}       | {wr_defender_vs_defender:.2%}         |")

if __name__ == "__main__":
    main()
