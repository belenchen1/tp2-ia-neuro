import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from agentes import Agent
import utils


class Connect4State:
    def __init__(self, rows: int = 6, cols: int = 7):
        """
        Inicializa el estado del juego Connect4.
        
        Args:
            Definir qué hace a un estado de Connect4.
        """
        self.rows = rows
        self.cols = cols
        self.board = utils.create_board(rows, cols)
        self.current_player = 1
        self.game_over = False
        self.winner = None

    def copy(self):
        """
        Crea una copia profunda del estado actual.
        
        Returns:
            Una nueva instancia de Connect4State con los mismos valores.
        """
        s = Connect4State(self.rows, self.cols)
        s.board = np.copy(self.board)
        s.current_player = int(self.current_player)
        s.game_over = bool(self.game_over)
        s.winner = None if self.winner is None else int(self.winner)
        return s

    def update_state(self):
        """
        Modifica las variables internas del estado luego de una jugada.

        Args:
            ... (_type_): _description_
            ... (_type_): _description_
        """
        over, winner = utils.check_game_over(self.board)
        self.game_over = over
        self.winner = winner

    def __eq__(self, other):
        """
        Compara si dos estados son iguales.
        
        Args:
            other: Otro estado para comparar.
            
        Returns:
            True si los estados son iguales, False en caso contrario.
        """
        if not isinstance(other, Connect4State):
            return False
        return self.current_player == other.current_player and np.array_equal(self.board, other.board)

    def __hash__(self):
        """
        Genera un hash único para el estado.
        
        Returns:
            Hash del estado basado en el tablero y jugador actual.
        """
        return hash((self.board.tobytes(), int(self.current_player)))

    def __repr__(self):
        """
        Representación en string del estado.
        """
        return f"Connect4State(player={self.current_player}\n{self.board})"


class Connect4Environment:
    def __init__(self, rows: int = 6, cols: int = 7):
        """
        Inicializa el ambiente del juego Connect4.
        Args:
            Definir las variables de instancia de un ambiente de Connect4
        """
        self.rows = rows
        self.cols = cols
        self.state: Connect4State = Connect4State(rows, cols)

    def reset(self):
        """
        Reinicia el ambiente a su estado inicial para volver a realizar un episodio.
        """
        self.state = Connect4State(self.rows, self.cols)
        return self.state

    def available_actions(self):
        """
        Obtiene las acciones válidas (columnas disponibles) en el estado actual.
        
        Returns:
            Lista de índices de columnas donde se puede colocar una ficha.
        """
        # columnas donde la fila 0 es 0
        return [c for c in range(self.cols) if self.state.board[0, c] == 0]

    def step(self, action):
        """
        Ejecuta una acción y devuelve (nuevo_estado, recompensa, done, info).
        Ahora incluye recompensas intermedias (shaping).
        """
        state = self.state
        if state.game_over:
            return state, 0, True, {"winner": state.winner}

        prev_player = state.current_player
        # Aplicar jugada
        utils.insert_token(state.board, action, prev_player)
        state.current_player = 3 - prev_player
        state.update_state()
        done = state.game_over
        winner = state.winner if done else None

        reward = 0.0

        # --- Recompensas finales ---
        if done:
            if winner is None:        # empate
                reward = 0.0
            elif winner == prev_player:
                reward = 1.0          # victoria
            else:
                reward = -1.0         # derrota
        else:
            # --- Reward shaping intermedio ---
            # 1) Bonus por ocupar columna central (estratégicamente fuerte)
            center_col = self.cols // 2
            if action == center_col:
                reward += 0.05

            # 2) Bonus por formar "tres en línea" propio
            my_threes = utils.count_n_in_a_row(state.board, prev_player, 3)
            reward += 0.1 * my_threes

            # 3) Penalización si el oponente tiene "tres en línea" (riesgo de perder)
            opp = 3 - prev_player
            opp_threes = utils.count_n_in_a_row(state.board, opp, 3)
            reward -= 0.1 * opp_threes

        info = {"winner": winner}
        return state.copy(), reward, done, info


    def render(self):
        """
        Muestra visualmente el estado actual del tablero en la consola.
        """
        b = self.state.board
        # Print board with top row first
        for r in range(self.rows):
            row = "|".join(str(int(x)) for x in b[r])
            print("|" + row + "|")
        print("-" * (self.cols * 2 + 1))


def get_activation(name: str):
    name = name.lower()
    return {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leakyrelu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "selu": nn.SELU(),
    }[name]

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(128,128), activation="relu"):
        """
        input_dim: int
        output_dim: int
        hidden_sizes: tuple/list con unidades por capa oculta, p.ej. (64,64,64)
        activation: str en {"relu","tanh","leakyrelu","elu","gelu","selu"}
        """
        super().__init__()
        act = get_activation(activation)

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act.__class__())  # nueva instancia de la misma activación
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # salida lineal

        self.net = nn.Sequential(*layers)

        # (Opcional) inicialización amigable con ReLU/LeakyReLU/ELU/GELU:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Pasa la entrada a través de la red neuronal.
        
        Args:
            x: Tensor de entrada.
            
        Returns:
            Tensor de salida con los valores Q para cada acción.
        """
        return self.net(x)


class DeepQLearningAgent(Agent):
    def __init__(self, state_shape, n_actions, device='cpu',
                 gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 lr=1e-3, batch_size=64, memory_size=1000, target_update_every=100,
                 hidden_sizes=(128,128), activation="relu", loss="mse"):
        """
        Inicializa el agente de aprendizaje por refuerzo DQN.
        
        Args:
            state_shape: Forma del estado (filas, columnas).
            n_actions: Número de acciones posibles.
            device: Dispositivo para computación ('cpu' o 'cuda').
            gamma: Factor de descuento para recompensas futuras.
            epsilon: Probabilidad inicial de exploración.
            epsilon_min: Valor mínimo de epsilon.
            epsilon_decay: Factor de decaimiento de epsilon.
            lr: Tasa de aprendizaje.
            batch_size: Tamaño del batch para entrenamiento.
            memory_size: Tamaño máximo de la memoria de experiencias.
            target_update_every: Frecuencia de actualización de la red objetivo.
        """
        self.device = torch.device(device)
        self.rows, self.cols = state_shape
        self.input_dim = self.rows * self.cols
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update_every = target_update_every
        self.train_steps = 0

        self.q_network = DQN(self.input_dim, self.n_actions).to(self.device)
        self.target_network = DQN(self.input_dim, self.n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.loss_name = loss.lower()
        self.loss_fn = nn.MSELoss() if self.loss_name == "mse" else nn.SmoothL1Loss()

        self.q_network = DQN(self.input_dim, self.n_actions,
                             hidden_sizes=hidden_sizes, activation=activation).to(self.device)
        self.target_network = DQN(self.input_dim, self.n_actions,
                                  hidden_sizes=hidden_sizes, activation=activation).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

    def preprocess(self, state: Connect4State):
        # Convertir el tablero en una copia
        board = np.array(state.board, dtype=np.int8)

        # El jugador actual es 1 o 2
        current = state.current_player
        opponent = 3 - current

        # Mapear fichas
        board = np.where(board == current, 1, board)       # fichas del jugador actual -> +1
        board = np.where(board == opponent, -1, board)     # fichas del oponente -> -1
        board = np.where(board == 0, 0, board)             # casillas vacías -> 0

        # Aplanar y pasar a tensor
        arr = board.astype(np.float32).flatten()
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return tensor


    def select_action(self, state, valid_actions):
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        Args:
            state: Estado actual del juego.
            valid_actions: Lista de acciones válidas.
            
        Returns:
            Índice de la acción seleccionada.
        """
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            s = self.preprocess(state)
            q_values = self.q_network(s).cpu().numpy().flatten()
            # mask invalid actions by setting to very low
            mask = np.full_like(q_values, -1e9)
            for a in valid_actions:
                mask[a] = q_values[a]
            return int(np.argmax(mask))

    def store_transition(self, s, a, r, s_next, done):
        """
        Almacena una transición (estado, acción, recompensa, siguiente estado, terminado) en la memoria.
        
        Args:
            s: Estado actual.
            a: Acción tomada.
            r: Recompensa obtenida.
            s_next: Siguiente estado.
            done: Si el episodio terminó.
        """
        # store shallow copies; states in this project are lightweight
        self.memory.append((s.copy(), a, r, None if s_next is None else s_next.copy(), done))

    def train_step(self):
        """
        Ejecuta un paso de entrenamiento usando experiencias de la memoria.
        
        Returns:
            Valor de la función de pérdida si se pudo entrenar, None en caso contrario.
        """
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_batch = torch.cat([self.preprocess(s) for s in states], dim=0).to(self.device)
        next_batch = torch.cat([self.preprocess(s) for s in next_states], dim=0).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_network(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q = self.target_network(next_batch)
            max_next_q, _ = next_q.max(dim=1, keepdim=True)
            target_q = reward_batch + (1 - done_batch) * (self.gamma * max_next_q)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def update_epsilon(self):
        """
        Actualiza el valor de epsilon para reducir la exploración gradualmente.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def play(self, state, valid_actions):
        """
        Wrapper para que el agente pueda ser usado en Connect4.
        Equivalente a select_action pero sin exploración.
        """
        with torch.no_grad():
            s = self.preprocess(state)
            q_values = self.q_network(s).cpu().numpy().flatten()
            # enmascarar acciones inválidas
            mask = np.full_like(q_values, -1e9)
            for a in valid_actions:
                mask[a] = q_values[a]
            return int(np.argmax(mask))



class TrainedAgent(Agent):
    def __init__(self, model_path: str, state_shape: tuple, n_actions: int, device='cpu'):
        """
        Inicializa un agente DQN pre-entrenado.
        
        Args:
            model_path: Ruta al archivo del modelo entrenado.
            state_shape: Forma del estado del juego.
            n_actions: Número de acciones posibles.
            device: Dispositivo para computación.
        """
        self.name = str(model_path)
        self.device = torch.device(device)
        self.rows, self.cols = state_shape
        self.n_actions = n_actions
        self.q_network = DQN(self.rows * self.cols, n_actions).to(self.device)
        self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
        self.q_network.eval()

    def preprocess(self, state: Connect4State):
        """
        Normaliza el estado para que la red siempre vea
        +1 = fichas del jugador actual
        -1 = fichas del oponente
        0 = casillas vacías
        """
        board = np.array(state.board, dtype=np.int8)
        current = state.current_player
        opponent = 3 - current

        board = np.where(board == current, 1, board)
        board = np.where(board == opponent, -1, board)
        board = np.where(board == 0, 0, board)

        arr = board.astype(np.float32).flatten()
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return tensor

    def play(self, state, valid_actions):
        """
        Selecciona la mejor acción según el modelo entrenado.
        
        Args:
            state: Estado actual del juego.
            valid_actions: Lista de acciones válidas.
            
        Returns:
            Índice de la mejor acción según el modelo.
        """
        with torch.no_grad():
            tensor = self.preprocess(state)
            q_values = self.q_network(tensor).cpu().numpy().flatten()
            # enmascarar acciones inválidas
            mask = np.full_like(q_values, -1e9)
            for a in valid_actions:
                mask[a] = q_values[a]
            return int(np.argmax(mask))
    def preprocess(self, state: Connect4State):
        # Convertir el tablero en una copia
        board = np.array(state.board, dtype=np.int8)

        # El jugador actual es 1 o 2
        current = state.current_player
        opponent = 3 - current

        # Mapear fichas
        board = np.where(board == current, 1, board)       # fichas del jugador actual -> +1
        board = np.where(board == opponent, -1, board)     # fichas del oponente -> -1
        board = np.where(board == 0, 0, board)             # casillas vacías -> 0

        # Aplanar y pasar a tensor
        arr = board.astype(np.float32).flatten()
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return tensor


    def select_action(self, state, valid_actions):
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        Args:
            state: Estado actual del juego.
            valid_actions: Lista de acciones válidas.
            
        Returns:
            Índice de la acción seleccionada.
        """
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            s = self.preprocess(state)
            q_values = self.q_network(s).cpu().numpy().flatten()
            # mask invalid actions by setting to very low
            mask = np.full_like(q_values, -1e9)
            for a in valid_actions:
                mask[a] = q_values[a]
            return int(np.argmax(mask))

    def store_transition(self, s, a, r, s_next, done):
        """
        Almacena una transición (estado, acción, recompensa, siguiente estado, terminado) en la memoria.
        
        Args:
            s: Estado actual.
            a: Acción tomada.
            r: Recompensa obtenida.
            s_next: Siguiente estado.
            done: Si el episodio terminó.
        """
        # store shallow copies; states in this project are lightweight
        self.memory.append((s.copy(), a, r, None if s_next is None else s_next.copy(), done))

    def train_step(self):
        """
        Ejecuta un paso de entrenamiento usando experiencias de la memoria.
        
        Returns:
            Valor de la función de pérdida si se pudo entrenar, None en caso contrario.
        """
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_batch = torch.cat([self.preprocess(s) for s in states], dim=0).to(self.device)
        next_batch = torch.cat([self.preprocess(s) for s in next_states], dim=0).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_network(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q = self.target_network(next_batch)
            max_next_q, _ = next_q.max(dim=1, keepdim=True)
            target_q = reward_batch + (1 - done_batch) * (self.gamma * max_next_q)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def update_epsilon(self):
        """
        Actualiza el valor de epsilon para reducir la exploración gradualmente.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

'''
class TrainedAgent(Agent):
    def __init__(self, model_path: str, state_shape: tuple, n_actions: int, device='cpu'):
        """
        Inicializa un agente DQN pre-entrenado.
        
        Args:
            model_path: Ruta al archivo del modelo entrenado.
            state_shape: Forma del estado del juego.
            n_actions: Número de acciones posibles.
            device: Dispositivo para computación.
        """
        self.name = str(model_path)
        self.device = torch.device(device)
        self.rows, self.cols = state_shape
        self.n_actions = n_actions
        self.q_network = DQN(self.rows * self.cols, n_actions).to(self.device)
        self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
        self.q_network.eval()

    def preprocess(self, state: Connect4State):
        """
        Normaliza el estado para que la red siempre vea
        +1 = fichas del jugador actual
        -1 = fichas del oponente
        0 = casillas vacías
        """
        board = np.array(state.board, dtype=np.int8)
        current = state.current_player
        opponent = 3 - current

        board = np.where(board == current, 1, board)
        board = np.where(board == opponent, -1, board)
        board = np.where(board == 0, 0, board)

        arr = board.astype(np.float32).flatten()
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return tensor

    def play(self, state, valid_actions):
        """
        Selecciona la mejor acción según el modelo entrenado.
        
        Args:
            state: Estado actual del juego.
            valid_actions: Lista de acciones válidas.
            
        Returns:
            Índice de la mejor acción según el modelo.
        """
        with torch.no_grad():
            tensor = self.preprocess(state)
            q_values = self.q_network(tensor).cpu().numpy().flatten()
            # enmascarar acciones inválidas
            mask = np.full_like(q_values, -1e9)
            for a in valid_actions:
                mask[a] = q_values[a]
            return int(np.argmax(mask))
'''