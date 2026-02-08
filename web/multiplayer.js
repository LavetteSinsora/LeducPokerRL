const socket = io();

const elements = {
    resetBtn: document.getElementById('reset-btn'),
    resetAllBtn: document.getElementById('reset-all-btn'),
    p0Card: document.getElementById('p0-card'),
    p1Card: document.getElementById('p1-card'),
    p0Stack: document.getElementById('p0-stack'),
    p1Stack: document.getElementById('p1-stack'),
    boardCard: document.getElementById('board-card'),
    potTotal: document.getElementById('pot-total'),
    roundName: document.getElementById('round-name'),
    userActions: document.getElementById('user-actions'),
    gameLog: document.getElementById('game-log'),
    p0Name: document.getElementById('p0-name'),
    p1Name: document.getElementById('p1-name'),
    waitingOverlay: document.getElementById('waiting-overlay'),
    playerRoleDisplay: document.getElementById('player-role-display')
};

let myPlayerId = null;
let lastGameState = null;

socket.on('connect', () => {
    console.log('Connected to server');
    socket.emit('join', {});
});

socket.on('init', (data) => {
    myPlayerId = data.player_id;
    console.log('Initialized as player:', myPlayerId);
    elements.playerRoleDisplay.textContent = `Player ${myPlayerId}`;
    updateUI(data.state);
});

socket.on('state_update', (state) => {
    updateUI(state);
});

socket.on('error', (data) => {
    addLogEntry(`Error: ${data.message}`, 'error');
});

function updateUI(state) {
    lastGameState = state;

    // Manage waiting overlay
    const connectedCount = state.connected_players.length;
    if (connectedCount < 2) {
        elements.waitingOverlay.style.display = 'flex';
    } else {
        elements.waitingOverlay.style.display = 'none';

        // Populate names once players join
        if (myPlayerId === 0) {
            elements.p0Name.innerHTML = 'You <span id="player-role-tag">P0</span>';
            elements.p1Name.textContent = 'Opponent (P1)';
        } else if (myPlayerId === 1) {
            elements.p0Name.textContent = 'Opponent (P0)';
            elements.p1Name.innerHTML = 'You <span id="player-role-tag">P1</span>';
        } else {
            elements.p0Name.textContent = 'Player 0 (Spectating)';
            elements.p1Name.textContent = 'Player 1 (Spectating)';
        }
    }

    // Update cards
    elements.p0Card.textContent = state.player_hands[0] === 'HIDDEN' ? '?' : state.player_hands[0];
    elements.p0Card.className = state.player_hands[0] === 'HIDDEN' ? 'card hidden' : 'card';

    elements.p1Card.textContent = state.player_hands[1] === 'HIDDEN' ? '?' : state.player_hands[1];
    elements.p1Card.className = state.player_hands[1] === 'HIDDEN' ? 'card hidden' : 'card';

    if (state.board) {
        elements.boardCard.textContent = state.board;
        elements.boardCard.classList.remove('hidden');
    } else {
        elements.boardCard.textContent = '?';
        elements.boardCard.classList.add('hidden');
    }

    // Update Pot and Stacks
    const totalPot = state.pot.reduce((a, b) => a + b, 0);
    elements.potTotal.textContent = totalPot;
    elements.roundName.textContent = state.current_round;

    elements.p0Stack.textContent = state.stacks[0] - state.pot[0];
    elements.p1Stack.textContent = state.stacks[1] - state.pot[1];

    // Highlight current player
    const player_slot_0 = document.getElementById('player-slot-0');
    const player_slot_1 = document.getElementById('player-slot-1');
    player_slot_0.style.border = state.current_player === 0 && !state.is_finished ? '2px solid var(--accent-gold)' : 'none';
    player_slot_1.style.border = state.current_player === 1 && !state.is_finished ? '2px solid var(--accent-gold)' : 'none';

    // Update buttons
    const isMyTurn = state.current_player === myPlayerId && !state.is_finished;
    const buttons = elements.userActions.querySelectorAll('button');
    buttons.forEach(btn => {
        const actionId = parseInt(btn.getAttribute('data-action'));
        btn.disabled = !isMyTurn || !state.legal_actions.includes(actionId);
        if (actionId === 2) {
            const betAmount = state.current_round_num === 0 ? 2 : 4;
            btn.textContent = `Raise (${betAmount})`;
        }
    });

    // Update log
    renderLog(state);
}

function renderLog(state) {
    elements.gameLog.innerHTML = '';
    state.history.forEach(entry => {
        const player = entry[0];
        const action = entry[1];
        const playerName = player === myPlayerId ? 'You' : `Player ${player}`;
        elements.gameLog.innerHTML += `<div class="log-entry"><span>${playerName}</span> ${action}s</div>`;
    });

    if (state.is_finished) {
        let resultMsg = "";
        if (state.winner === -1) {
            resultMsg = "Tie Game!";
        } else {
            const winnerName = state.winner === myPlayerId ? 'You' : `Player ${state.winner}`;
            resultMsg = `${winnerName} Won!`;
            const rewards = state.rewards;
            const won = Math.max(...rewards);
            elements.gameLog.innerHTML += `<div class="log-entry">${winnerName} gained <span>${won}</span></div>`;
        }
        elements.gameLog.innerHTML += `<div class="log-entry" style="color: var(--accent-gold); font-weight: bold;">Result: ${resultMsg}</div>`;
    }
    elements.gameLog.scrollTop = elements.gameLog.scrollHeight;
}

function addLogEntry(msg, type = '') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = msg;
    elements.gameLog.appendChild(entry);
    elements.gameLog.scrollTop = elements.gameLog.scrollHeight;
}

// Event Listeners
elements.resetBtn.addEventListener('click', () => {
    socket.emit('reset');
});

elements.resetAllBtn.addEventListener('click', () => {
    socket.emit('reset_all');
});

elements.userActions.addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON') {
        const action = e.target.getAttribute('data-action');
        socket.emit('action', { action });
    }
});
