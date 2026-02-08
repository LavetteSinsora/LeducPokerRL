// If served from port 8001 (static server), fall back to port 8000 for API
const API_BASE = window.location.port === '8001'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : window.location.origin;

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
    p0Selector: document.getElementById('p0-agent-selector'),
    p1Selector: document.getElementById('p1-agent-selector')
};

let gameState = null;
let aiLoopInterval = null;

async function updateState() {
    try {
        const response = await fetch(`${API_BASE}/state`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        gameState = await response.json();
        render();
    } catch (e) {
        console.error("Failed to update state:", e);
        elements.gameLog.innerHTML += `<div class="log-entry error">Error: Cannot connect to game server.</div>`;
    }
}

async function resetGame() {
    try {
        const p0Agent = elements.p0Selector.querySelector('.active').dataset.agent;
        const p1Agent = elements.p1Selector.querySelector('.active').dataset.agent;
        const agent_types = [p0Agent, p1Agent];

        const response = await fetch(`${API_BASE}/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ agent_types })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        gameState = await response.json();

        // Clear log
        elements.gameLog.innerHTML = '<div class="log-entry">Game started! <span>' + p0Agent + '</span> vs <span>' + p1Agent + '</span></div>';

        // Stop any existing AI loop
        if (aiLoopInterval) clearInterval(aiLoopInterval);

        render();

        const isAnyHuman = agent_types.includes('human');
        if (!isAnyHuman) {
            startAiLoop();
        } else {
            // Human turn check
            checkAutoStep();
        }
    } catch (e) {
        console.error("Failed to reset game:", e);
        elements.gameLog.innerHTML += `<div class="log-entry error">Error: Failed to start new game.</div>`;
    }
}

async function resetAll() {
    try {
        const response = await fetch(`${API_BASE}/reset_all`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        gameState = await response.json();
        elements.gameLog.innerHTML = '<div class="log-entry">Stacks reset! New Session started.</div>';
        if (aiLoopInterval) clearInterval(aiLoopInterval);
        render();
        checkAutoStep();
    } catch (e) {
        console.error("Failed to reset all:", e);
        elements.gameLog.innerHTML += `<div class="log-entry error">Error: Failed to reset session.</div>`;
    }
}

async function performAction(actionId) {
    try {
        const actionInt = parseInt(actionId);

        // Optimistic UI update for Raise
        if (actionInt === 2 && gameState && !gameState.is_finished) {
            const betAmount = gameState.current_round_num === 0 ? 2 : 4;
            const other_player_pot = gameState.pot[1];
            // To raise: match opponent + add betAmount
            const new_contribution = other_player_pot + betAmount;

            // Subtract from the displayed stack immediately
            const p0_display_stack = gameState.stacks[0] - new_contribution;
            elements.p0Stack.textContent = p0_display_stack;

            // Add to the displayed pot immediately
            const new_total_pot = new_contribution + other_player_pot;
            elements.potTotal.textContent = new_total_pot;

            // Visual feedback
            elements.p0Stack.parentElement.style.transform = 'scale(1.1)';
            setTimeout(() => {
                elements.p0Stack.parentElement.style.transform = 'scale(1)';
            }, 200);
        }

        const response = await fetch(`${API_BASE}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: actionInt })
        });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        gameState = await response.json();
        addLogEntry(actionId);
        render();
        checkAutoStep();
    } catch (e) {
        console.error("Action failed:", e);
        elements.gameLog.innerHTML += `<div class="log-entry error">Error: Action failed.</div>`;
    }
}

async function autoStep() {
    try {
        const response = await fetch(`${API_BASE}/step`, { method: 'POST' });
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        gameState = await response.json();
        addLogEntry();
        render();
        checkAutoStep();
    } catch (e) {
        console.error("Auto step failed:", e);
        if (aiLoopInterval) clearInterval(aiLoopInterval);
    }
}

function checkAutoStep() {
    if (gameState.is_finished) return;

    const currPlayer = gameState.current_player;
    const isAi = gameState.agent_configs[currPlayer] !== 'human';

    if (isAi) {
        // It's the AI's turn
        setTimeout(autoStep, 1000);
    }
}

function startAiLoop() {
    aiLoopInterval = setInterval(async () => {
        if (gameState && !gameState.is_finished) {
            await autoStep();
        } else {
            clearInterval(aiLoopInterval);
        }
    }, 1500);
}

function render() {
    if (!gameState) return;

    // Update cards
    elements.p0Card.textContent = gameState.player_hands[0] === 'HIDDEN' ? '?' : gameState.player_hands[0];
    elements.p0Card.className = gameState.player_hands[0] === 'HIDDEN' ? 'card hidden' : 'card';

    elements.p1Card.textContent = gameState.player_hands[1] === 'HIDDEN' ? '?' : gameState.player_hands[1];
    elements.p1Card.className = gameState.player_hands[1] === 'HIDDEN' ? 'card hidden' : 'card';

    if (gameState.board) {
        elements.boardCard.textContent = gameState.board;
        elements.boardCard.classList.remove('hidden');
    } else {
        elements.boardCard.textContent = '?';
        elements.boardCard.classList.add('hidden');
    }

    // Update Pot and Stacks
    const totalPot = gameState.pot.reduce((a, b) => a + b, 0);
    elements.potTotal.textContent = totalPot;
    elements.roundName.textContent = gameState.current_round;

    // Display stacks as (Persistent Stack - Current Hand Contribution)
    elements.p0Stack.textContent = gameState.stacks[0] - gameState.pot[0];
    elements.p1Stack.textContent = gameState.stacks[1] - gameState.pot[1];

    // Highlight current player
    const player_slot_0 = document.getElementById('player-slot-0');
    const player_slot_1 = document.getElementById('player-slot-1');
    player_slot_0.style.boxShadow = gameState.current_player === 0 && !gameState.is_finished ? '0 0 20px var(--accent-gold)' : 'none';
    player_slot_1.style.boxShadow = gameState.current_player === 1 && !gameState.is_finished ? '0 0 20px var(--accent-gold)' : 'none';

    // Update buttons
    const currPlayer = gameState.current_player;
    const isUserTurn = gameState.agent_configs[currPlayer] === 'human' && !gameState.is_finished;
    const buttons = elements.userActions.querySelectorAll('button');
    buttons.forEach(btn => {
        const actionId = parseInt(btn.getAttribute('data-action'));
        btn.disabled = !isUserTurn || !gameState.legal_actions.includes(actionId);
        if (actionId === 2) {
            const betAmount = gameState.current_round_num === 0 ? 2 : 4;
            btn.textContent = `Raise (${betAmount})`;
        }
    });

    if (gameState.is_finished) {
        let resultMsg = "";
        let detailMsg = "";
        if (gameState.winner === -1) {
            resultMsg = "Tie Game!";
        } else {
            const rewards = gameState.rewards;
            const won = Math.max(...rewards);
            const winnerIdx = rewards.indexOf(won);
            const loserIdx = 1 - winnerIdx;

            const winnerName = gameState.agent_configs[winnerIdx] === 'human' ? 'You' : `Agent ${winnerIdx}`;
            const loserName = gameState.agent_configs[loserIdx] === 'human' ? 'You' : `Agent ${loserIdx}`;

            resultMsg = `${winnerName} Won!`;
            detailMsg = `<div class="log-entry">${winnerName} gained <span>${won}</span>, ${loserName} lost <span>${won}</span></div>`;
        }
        elements.gameLog.innerHTML += `<div class="log-entry" style="color: var(--accent-gold); font-weight: bold; font-size: 16px;">Result: ${resultMsg}</div>`;
        if (detailMsg) elements.gameLog.innerHTML += detailMsg;
        elements.gameLog.scrollTop = elements.gameLog.scrollHeight;
    }
}

function addLogEntry(actionId = null) {
    if (!gameState) return;
    const history = gameState.history;
    if (history.length === 0) return;

    const lastAction = history[history.length - 1];
    const player = lastAction[0];
    const actionName = lastAction[1];

    const playerName = gameState.agent_configs[player] === 'human' ? 'You' : `Agent ${player}`;
    const logMsg = `<div class="log-entry"><span>${playerName}</span> ${actionName}s</div>`;

    elements.gameLog.innerHTML += logMsg;
    elements.gameLog.scrollTop = elements.gameLog.scrollHeight;
}

// Event Listeners
elements.resetBtn.addEventListener('click', resetGame);
elements.resetAllBtn.addEventListener('click', resetAll);

function setupSelector(selector) {
    selector.addEventListener('click', (e) => {
        if (e.target.classList.contains('agent-opt')) {
            selector.querySelectorAll('.agent-opt').forEach(opt => opt.classList.remove('active'));
            e.target.classList.add('active');
        }
    });
}

setupSelector(elements.p0Selector);
setupSelector(elements.p1Selector);

elements.userActions.addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON') {
        performAction(e.target.getAttribute('data-action'));
    }
});

// Initial call
// updateState();
