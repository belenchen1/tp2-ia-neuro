import optuna
import joblib
import matplotlib.pyplot as plt

def main():
    # Cargar el estudio guardado en bayesian_search.py
    study = joblib.load("optuna_study.pkl")

    # ---------- Gráfico 1: Historial de optimización ----------
    fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig1.figure.set_size_inches(8, 5)
    fig1.figure.savefig("optuna_history.png")
    print("✅ Guardado: optuna_history.png")

    # ---------- Gráfico 2: Importancia de hiperparámetros ----------
    fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
    fig2.figure.set_size_inches(8, 5)
    fig2.figure.savefig("optuna_importances.png")
    print("✅ Guardado: optuna_importances.png")

    # ---------- Gráfico 3: Comparación de valores de hiperparámetros ----------
    # (scatter para ver relación entre un parámetro y el winrate)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    if "params_gamma" in trials.columns:
        plt.figure(figsize=(8,5))
        plt.scatter(trials["params_gamma"], trials["value"], alpha=0.7)
        plt.xlabel("Gamma")
        plt.ylabel("Winrate")
        plt.title("Relación entre gamma y winrate")
        plt.grid(True)
        plt.savefig("optuna_gamma_vs_winrate.png")
        print("✅ Guardado: optuna_gamma_vs_winrate.png")

    # ---------- Info en consola ----------
    print("\n=== Resultados ===")
    print("Mejor Winrate:", study.best_value)
    print("Mejores Params:", study.best_params)

if __name__ == "__main__":
    main()
