import os
import random
from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit, join_room
from src.engine.leduc_game import LeducGame, Action

app = Flask(__name__, static_folder='../../web')
socketio = SocketIO(app, cors_allowed_origins="*")

# Game State
class MultiplayerState:
    def __init__(self):
        self.game = LeducGame()
        self.players = {}  # sid -> player_id (0 or 1)
        self.stacks = [100, 100]
        self.ready_players = set()

game_state = MultiplayerState()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'multiplayer.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('join')
def handle_join(data):
    sid = request.sid
    if sid in game_state.players:
        emit('init', {'player_id': game_state.players[sid], 'state': format_state(game_state.players[sid])})
        return

    # Assign player ID
    assigned_id = None
    existing_ids = set(game_state.players.values())
    
    if 0 not in existing_ids:
        assigned_id = 0
    elif 1 not in existing_ids:
        assigned_id = 1
    else:
        # Spectator or full
        emit('error', {'message': 'Game is full'})
        return

    game_state.players[sid] = assigned_id
    print(f"Player {assigned_id} joined: {sid}")
    
    emit('init', {'player_id': assigned_id, 'state': format_state(assigned_id)})
    broadcast_state()

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in game_state.players:
        player_id = game_state.players[sid]
        print(f"Player {player_id} disconnected: {sid}")
        del game_state.players[sid]
        broadcast_state()

@socketio.on('action')
def handle_action(data):
    sid = request.sid
    if sid not in game_state.players:
        return

    player_id = game_state.players[sid]
    if game_state.game.current_player != player_id:
        emit('error', {'message': "Not your turn"})
        return

    if game_state.game.is_finished:
        emit('error', {'message': "Game already finished"})
        return

    try:
        action_val = int(data['action'])
        action = Action(action_val)
        
        # Validate legal action
        legal_actions = game_state.game._get_legal_actions()
        if action not in legal_actions:
            emit('error', {'message': "Illegal action"})
            return

        # Perform step
        game_state.game.step(action)
        
        # Update stacks if finished
        if game_state.game.is_finished:
            rewards = game_state.game._get_reward()
            game_state.stacks[0] += rewards[0]
            game_state.stacks[1] += rewards[1]

        broadcast_state()
        
    except (ValueError, KeyError) as e:
        emit('error', {'message': str(e)})

@socketio.on('reset')
def handle_reset():
    sid = request.sid
    if sid not in game_state.players:
        return
    
    # Simple reset: anyone can reset for now
    game_state.game.reset()
    broadcast_state()

@socketio.on('reset_all')
def handle_reset_all():
    sid = request.sid
    if sid not in game_state.players:
        return
    
    game_state.stacks = [100, 100]
    game_state.game.reset()
    broadcast_state()

def format_state(viewer_id):
    """Formats the game state for a specific player view."""
    game = game_state.game
    
    # Hide other player's hand until showdown
    hands = []
    for i in range(2):
        if i == viewer_id or game.is_finished:
            hands.append(game.player_hands[i])
        else:
            hands.append("HIDDEN")

    history = [[p, a] for p, a in game.history]
    
    # Count connected players
    connected_players = list(set(game_state.players.values()))

    return {
        "player_hands": hands,
        "board": game.board,
        "pot": game.pot,
        "stacks": game_state.stacks,
        "current_player": game.current_player,
        "current_round_num": game.current_round,
        "current_round": "Pre-flop" if game.current_round == 0 else "Flop",
        "legal_actions": [a.value for a in game._get_legal_actions()] if not game.is_finished else [],
        "is_finished": game.is_finished,
        "winner": game.winner,
        "rewards": game._get_reward() if game.is_finished else [0, 0],
        "history": history,
        "connected_players": connected_players
    }

def broadcast_state():
    for sid, player_id in game_state.players.items():
        socketio.emit('state_update', format_state(player_id), room=sid)

if __name__ == '__main__':
    # Listen on all interfaces so friends can join
    socketio.run(app, debug=True, port=8000, host='0.0.0.0')
