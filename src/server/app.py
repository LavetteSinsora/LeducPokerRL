import http.server
import json
import os
from src.engine.leduc_game import LeducGame, Action
from src.agents import registry
from src.training.training_manager import TrainingManager

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# web directory is two levels up from src/server/
WEB_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'web'))

class LeducAPIHandler(http.server.BaseHTTPRequestHandler):
    game_state_obj = None
    training_manager = None

    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_GET(self):
        if self.path == '/state':
            self._set_headers()
            self.wfile.write(json.dumps(self.format_response()).encode())
        elif self.path == '/api/agents':
            # Return list of all available agents from registry
            self._set_headers()
            agents_data = [
                {
                    "id": meta.id,
                    "displayName": meta.display_name,
                    "description": meta.description,
                    "isTrainable": meta.is_trainable,
                    "category": meta.category,
                }
                for meta in registry.list_agents()
            ]
            # Add "human" as a special option for play mode
            agents_data.insert(0, {
                "id": "human",
                "displayName": "Human Player",
                "description": "You control this player",
                "isTrainable": False,
                "category": "special"
            })
            self.wfile.write(json.dumps({"agents": agents_data}).encode('utf-8'))
        elif self.path == '/train/status':
            self._set_headers()
            self.wfile.write(json.dumps(LeducAPIHandler.training_manager.get_status()).encode())
        elif self.path == '/train/history':
            self._set_headers()
            self.wfile.write(json.dumps(LeducAPIHandler.training_manager.get_history()).encode())
        else:
            # Serve static files from 'web' directory
            path = self.path.lstrip('/')
            if path == "" or path == "web/":
                path = "index.html"
            
            # If path starts with web/, strip it since we join with WEB_DIR
            if path.startswith('web/'):
                path = path[4:]

            file_path = os.path.join(WEB_DIR, path)
            
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    
                self.send_response(200)
                if path.endswith('.html'):
                    self.send_header('Content-type', 'text/html')
                elif path.endswith('.css'):
                    self.send_header('Content-type', 'text/css')
                elif path.endswith('.js'):
                    self.send_header('Content-type', 'application/javascript')
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                self.send_error(404, "File not found")

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b''
        data = json.loads(post_data) if post_data else {}

        if self.path == '/reset':
            LeducAPIHandler.game_state_obj.game.reset()
            LeducAPIHandler.game_state_obj.last_obs = LeducAPIHandler.game_state_obj.game.get_observation()
            
            # Select agents for both players
            agent_types = data.get('agent_types', ['heuristic', 'human']) # Default: Agent 1 = heuristic, Agent 2 = human
            LeducAPIHandler.game_state_obj.agent_configs = agent_types
            
            for i, atype in enumerate(agent_types):
                if atype == 'human':
                    LeducAPIHandler.game_state_obj.agents[i] = None
                else:
                    # Use registry to create agent
                    if atype == 'value_based':
                        # Look for trained model in models/ relative to the project root
                        root_dir = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
                        model_path = os.path.join(root_dir, 'models', 'value_agent.pt')
                        if os.path.exists(model_path):
                            print(f"Loading trained model from {model_path}")
                            LeducAPIHandler.game_state_obj.agents[i] = registry.create(atype, model_path=model_path)
                        else:
                            print(f"Trained model not found at {model_path}, using initial weights.")
                            LeducAPIHandler.game_state_obj.agents[i] = registry.create(atype)
                    else:
                        LeducAPIHandler.game_state_obj.agents[i] = registry.create(atype)
                
            self._set_headers()
            self.wfile.write(json.dumps(self.format_response()).encode())

        elif self.path == '/reset_all':
            LeducAPIHandler.game_state_obj.stacks = [100, 100]
            LeducAPIHandler.game_state_obj.game.reset()
            LeducAPIHandler.game_state_obj.last_obs = LeducAPIHandler.game_state_obj.game.get_observation()
            self._set_headers()
            self.wfile.write(json.dumps(self.format_response()).encode())

        elif self.path == '/step':
            game = LeducAPIHandler.game_state_obj.game
            if game.is_finished:
                self.send_error(400, "Game finished")
                return

            curr_player = game.current_player
            agent = LeducAPIHandler.game_state_obj.agents[curr_player]

            # Determine action
            if agent is None: # Human player
                if 'action' not in data:
                    self.send_error(400, f"Action required for human Player {curr_player}")
                    return
                obs = LeducAPIHandler.game_state_obj.last_obs
                action = Action(int(data['action']))
            else:
                # Get fresh observation from current player's perspective
                # (last_obs may contain the previous player's hand)
                obs = game.get_observation(viewer_id=curr_player)
                action = agent.select_action(obs)

            LeducAPIHandler.game_state_obj.last_obs, reward, done, _ = game.step(action)
            
            # Update stacks if game finished
            if done:
                reward = game.get_reward()
                LeducAPIHandler.game_state_obj.stacks[0] += reward[0]
                LeducAPIHandler.game_state_obj.stacks[1] += reward[1]

            self._set_headers()
            self.wfile.write(json.dumps(self.format_response()).encode())
        
        elif self.path == '/train/start':
            episodes = data.get('episodes', 1000)
            batch_size = data.get('batch_size', 32)
            lr = data.get('lr', 1e-4)
            success = LeducAPIHandler.training_manager.start_training(episodes, batch_size, lr)
            self._set_headers()
            self.wfile.write(json.dumps({"success": success}).encode())

        elif self.path == '/train/stop':
            success = LeducAPIHandler.training_manager.stop_training()
            self._set_headers()
            self.wfile.write(json.dumps({"success": success}).encode())

        elif self.path == '/train/reset':
            success = LeducAPIHandler.training_manager.reset_agent()
            self._set_headers()
            self.wfile.write(json.dumps({"success": success}).encode())

        elif self.path == '/simulate/decision':
            # Run simulation for all legal actions and return values
            game = LeducAPIHandler.game_state_obj.game
            curr_player = game.current_player
            # Use the training manager's agent for evaluation to see current progress
            agent = LeducAPIHandler.training_manager.agent
            obs = game.get_observation(viewer_id=curr_player)
            
            simulator = LeducGame()
            results = []

            for action_enum in obs.legal_actions:
                # Simulation step logic similar to ValueBasedAgent.select_action
                simulator.set_state(obs)
                _, _, done, _ = simulator.step(action_enum)
                
                if done:
                    if action_enum == Action.FOLD:
                        val = -float(obs.pot[obs.current_player])
                    else:
                        from src.engine.observation import Observation
                        terminal_obs = Observation(
                            player_hand=obs.player_hand,
                            board=simulator.board,
                            pot=list(simulator.pot),
                            current_player=obs.current_player,
                            current_round=simulator.current_round,
                            legal_actions=[],
                            is_finished=True
                        )
                        val, _ = agent._evaluate_state(terminal_obs)
                else:
                    next_obs = simulator.get_observation(viewer_id=obs.current_player)
                    v_model, _ = agent._evaluate_state(next_obs)
                    val = v_model if next_obs.current_player == obs.current_player else -v_model
                
                results.append({
                    "action": action_enum.value,
                    "value": round(val, 4)
                })

            self._set_headers()
            self.wfile.write(json.dumps({"results": results}).encode())

        elif self.path == '/analyze/state':
            # Evaluate a custom state configuration based on history
            from src.engine.observation import Observation
            
            player_hand = data.get('player_hand', 'Q')
            board = data.get('board')
            history = data.get('history', []) # List of action strings/ints
            
            # Reconstruct game state from history
            try:
                game = self._get_game_from_history(history)
            except Exception as e:
                self.send_error(400, str(e))
                return

            # Construct synthetic observation for the requested analysis
            # We use the game's calculated state but override cards/board with user selection
            
            # If board is provided by user, use it. Otherwise use game's board if exists.
            # Note: In Leduc, board is revealed after round 1.
            # If user selected a board card, we force it.
            
            forced_board = board if board else game.board
            
            # Current player is determined by the history
            current_player = game.current_player
            
            obs = Observation(
                player_hand=player_hand,
                board=forced_board,
                pot=list(game.pot),
                current_player=current_player,
                current_round=game.current_round,
                legal_actions=[a for a in game.get_legal_actions()],
                is_finished=game.is_finished
            )
            
            agent = LeducAPIHandler.training_manager.agent
            simulator = LeducGame()
            results = []

            if not game.is_finished:
                for action_enum in obs.legal_actions:
                    simulator.set_state(obs)
                    _, _, done, _ = simulator.step(action_enum)
                    
                    if done:
                        if action_enum == Action.FOLD:
                            val = -float(obs.pot[obs.current_player])
                        else:
                            terminal_obs = Observation(
                                player_hand=obs.player_hand,
                                board=simulator.board,
                                pot=list(simulator.pot),
                                current_player=obs.current_player,
                                current_round=simulator.current_round,
                                legal_actions=[],
                                is_finished=True
                            )
                            val, _ = agent._evaluate_state(terminal_obs)
                    else:
                        next_obs = simulator.get_observation(viewer_id=obs.current_player)
                        v_model, _ = agent._evaluate_state(next_obs)
                        val = v_model if next_obs.current_player == obs.current_player else -v_model
                    
                    results.append({
                        "action": action_enum.value,
                        "value": round(val, 4)
                    })
            
            self._set_headers()
            self.wfile.write(json.dumps({
                "results": results, 
                "current_player": current_player,
                "is_finished": game.is_finished
            }).encode())

        elif self.path == '/analyze/calculate_state':
            # Helper to calculate state from history for the UI
            history = data.get('history', [])
            try:
                game = self._get_game_from_history(history)
                
                legal_actions = []
                if not game.is_finished:
                    legal_actions = [a.value for a in game.get_legal_actions()]
                
                # Format history for UI
                # LeducGame.history contains (player, action_string)
                history_formatted = []
                for p, a in game.history:
                    # 'a' is a string like "FOLD", "CALL", "RAISE"
                    # We need to convert it to the int value for consistency with UI
                    try:
                        action_val = Action[a].value
                    except KeyError:
                        # Fallback if somehow it's not a standard action or is already an int/enum
                        action_val = a.value if hasattr(a, 'value') else a
                        
                    history_formatted.append({"player": p, "action": action_val})

                self._set_headers()
                self.wfile.write(json.dumps({
                    "pot": game.pot,
                    "current_player": game.current_player,
                    "current_round": game.current_round,
                    "board": game.board,
                    "is_finished": game.is_finished,
                    "legal_actions": legal_actions,
                    "winner": game.winner,
                    "history": history_formatted
                }).encode())
            except Exception as e:
                self.send_error(400, str(e))
                return

        else:
            self.send_error(404)

    def _get_game_from_history(self, history):
        """Replays a sequence of actions to return a LeducGame state."""
        game = LeducGame()
        game.reset()
        
        # We need to ensure the game has enough structure to accept actions.
        # However, LeducGame randomizes cards on reset.
        # For analysis, we don't care about the actual cards held by players 
        # (except for the user's hand which we override later),
        # but we DO care if the board card matches what the user selected.
        # But here we just replay actions. Card consistency is handled by state override.
        
        for action_val in history:
            if game.is_finished:
                break
            
            # Map string/int to Action enum
            if isinstance(action_val, str):
                # Try to map string names if sent
                try:
                    action = Action[action_val.upper()]
                except KeyError:
                    action = Action(int(action_val))
            else:
                action = Action(int(action_val))
                
            # Verify legality (optional, but good for safety)
            legal = game.get_legal_actions()
            if action not in legal:
                # If we try to Check but only Call is allowed, or vice versa?
                # In Leduc: 
                # CALL = 1, RAISE = 2, FOLD = 0.
                pass
                
            game.step(action)
            
        return game

    def format_response(self):
        game = LeducAPIHandler.game_state_obj.game
        configs = LeducAPIHandler.game_state_obj.agent_configs
        is_p0_human = configs[0] == 'human'
        is_p1_human = configs[1] == 'human'

        # Logical visibility:
        # If there's at least one human, they only see their own hand unless finished.
        # If both are AI, both hands stay hidden until finished.
        # This simplifies to: if game is not finished, hand is hidden unless player is human.
        
        p0_hand = game.player_hands[0] if (game.is_finished or is_p0_human) else "HIDDEN"
        p1_hand = game.player_hands[1] if (game.is_finished or is_p1_human) else "HIDDEN"

        # Handle Action objects (enums) for JSON serialization
        history = []
        for p, a in game.history:
             history.append([p, a])

        return {
            "player_hands": [p0_hand, p1_hand],
            "board": game.board,
            "pot": game.pot,
            "stacks": LeducAPIHandler.game_state_obj.stacks,
            "current_player": game.current_player,
            "current_round_num": game.current_round,
            "current_round": "Pre-flop" if game.current_round == 0 else "Flop",
            "legal_actions": [a.value for a in game.get_legal_actions()] if not game.is_finished else [],
            "is_finished": game.is_finished,
            "winner": game.winner,
            "rewards": game.get_reward() if game.is_finished else [0, 0],
            "history": history,
            "agent_configs": configs
        }

class GlobalState:
    def __init__(self):
        self.game = LeducGame()
        self.agent_configs = ["heuristic", "human"]
        # Use registry to create initial agent
        self.agents = [registry.create("heuristic"), None]
        self.last_obs = None
        self.stacks = [100, 100]

def run(server_class=http.server.HTTPServer, handler_class=LeducAPIHandler, port=8000):
    LeducAPIHandler.game_state_obj = GlobalState()
    # Initialize TrainingManager with model path
    root_dir = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
    model_path = os.path.join(root_dir, 'models', 'value_agent.pt')
    LeducAPIHandler.training_manager = TrainingManager(model_path=model_path)
    
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting Leduc API on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
