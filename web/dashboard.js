const API_URL = window.location.port === '8001'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : window.location.origin;

let lossChart, chipsChart, matchupChart;
let smoothingAlpha = 0.6;
let matchupSmoothing = 0.6;
let rawLossData = [], rawChipsData = [];
let rawMatchupData = {};  // { opponent_id: [{x, y}, ...] }
let currentPanel = 0;
let selectedOpponents = [];
let allAgents = [];

const MATCHUP_COLORS = [
    '#f87171',  // red
    '#60a5fa',  // blue
    '#a78bfa',  // purple
    '#fb923c',  // orange
    '#2dd4bf',  // teal
    '#e879f9',  // pink
];

// EMA smoothing: alpha=0 means no smoothed line, alpha close to 1 = very smooth
function computeEMA(data, alpha) {
    if (alpha === 0 || data.length === 0) return [];
    const result = [{ x: data[0].x, y: data[0].y }];
    for (let i = 1; i < data.length; i++) {
        result.push({
            x: data[i].x,
            y: alpha * result[i - 1].y + (1 - alpha) * data[i].y
        });
    }
    return result;
}

// Chart factory for split charts
function createTrainingChart(canvasId, label, rawColor, emaColor, yAxisLabel) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: label + ' (raw)',
                    data: [],
                    borderColor: rawColor,
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    tension: 0,
                    pointRadius: 0
                },
                {
                    label: label + ' (smoothed)',
                    data: [],
                    borderColor: emaColor,
                    backgroundColor: 'transparent',
                    borderWidth: 2.5,
                    tension: 0.3,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Episode', color: '#fff' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: yAxisLabel, color: emaColor },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#fff', font: { family: 'Inter' } }
                },
                zoom: {
                    zoom: {
                        drag: { enabled: true, backgroundColor: 'rgba(244, 208, 63, 0.15)', borderColor: '#f4d03f', borderWidth: 1 },
                        mode: 'xy'
                    },
                    pan: {
                        enabled: true,
                        modifierKey: 'shift',
                        mode: 'xy'
                    },
                    limits: {
                        x: { min: 'original', max: 'original' },
                        y: { min: 'original', max: 'original' }
                    }
                }
            }
        }
    });
}

function createMatchupChart() {
    const ctx = document.getElementById('matchupChart').getContext('2d');
    matchupChart = new Chart(ctx, {
        type: 'line',
        data: { datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Episode', color: '#fff' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                },
                y: {
                    type: 'linear',
                    title: { display: true, text: 'Avg Chips/Round', color: '#4ade80' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#fff', font: { family: 'Inter' } }
                },
                zoom: {
                    zoom: {
                        drag: { enabled: true, backgroundColor: 'rgba(244, 208, 63, 0.15)', borderColor: '#f4d03f', borderWidth: 1 },
                        mode: 'xy'
                    },
                    pan: { enabled: true, modifierKey: 'shift', mode: 'xy' },
                    limits: { x: { min: 'original', max: 'original' }, y: { min: 'original', max: 'original' } }
                }
            }
        }
    });
}

function initCharts() {
    lossChart = createTrainingChart('lossChart', 'Loss', 'rgba(244, 208, 63, 0.3)', '#f4d03f', 'Loss');
    chipsChart = createTrainingChart('chipsChart', 'Chips/Round', 'rgba(74, 222, 128, 0.3)', '#4ade80', 'Chips/Round');
    createMatchupChart();
}

function initSmoothingControl() {
    const slider = document.getElementById('smoothing-slider');
    const display = document.getElementById('smoothing-value');
    slider.addEventListener('input', () => {
        smoothingAlpha = parseFloat(slider.value);
        display.textContent = smoothingAlpha.toFixed(2);
        // Recompute EMA from cached raw data (no fetch needed)
        lossChart.data.datasets[1].data = computeEMA(rawLossData, smoothingAlpha);
        chipsChart.data.datasets[1].data = computeEMA(rawChipsData, smoothingAlpha);
        lossChart.update('none');
        chipsChart.update('none');
    });
}

function initMatchupSmoothing() {
    const slider = document.getElementById('matchup-smoothing-slider');
    const display = document.getElementById('matchup-smoothing-value');
    slider.addEventListener('input', () => {
        matchupSmoothing = parseFloat(slider.value);
        display.textContent = matchupSmoothing.toFixed(2);
        updateMatchupChart();
    });
}

function initResetZoom() {
    document.getElementById('btn-reset-zoom').addEventListener('click', () => {
        lossChart.resetZoom();
        chipsChart.resetZoom();
    });
    document.getElementById('btn-reset-matchup-zoom').addEventListener('click', () => {
        matchupChart.resetZoom();
    });
}

// Carousel navigation
function initCarousel() {
    const track = document.getElementById('carousel-track');
    const prevBtn = document.getElementById('carousel-prev');
    const nextBtn = document.getElementById('carousel-next');
    const dots = document.querySelectorAll('.carousel-dot');

    const panels = document.querySelectorAll('.carousel-panel');

    function goToPanel(index) {
        currentPanel = index;
        // Show/hide panels (no flex, no transform — Chart.js can always measure)
        panels.forEach((p, i) => p.classList.toggle('active', i === index));
        prevBtn.disabled = (index === 0);
        nextBtn.disabled = (index === 1);
        dots.forEach((d, i) => d.classList.toggle('active', i === index));
        document.getElementById('carousel-title').textContent =
            index === 0 ? 'Training Progress' : 'Matchup Evaluation';

        // Resize charts after panel becomes visible
        requestAnimationFrame(() => {
            if (index === 0) { lossChart.resize(); chipsChart.resize(); }
            if (index === 1) { matchupChart.resize(); }
        });
    }

    prevBtn.addEventListener('click', () => { if (currentPanel > 0) goToPanel(currentPanel - 1); });
    nextBtn.addEventListener('click', () => { if (currentPanel < 1) goToPanel(currentPanel + 1); });
    dots.forEach(dot => {
        dot.addEventListener('click', () => goToPanel(parseInt(dot.dataset.panel)));
    });
}

// Polling for Status
async function pollStatus() {
    try {
        const res = await fetch(`${API_URL}/train/status`);
        const data = await res.json();

        const statusText = document.getElementById('train-status-text');
        const startBtn = document.getElementById('btn-start-train');
        const stopBtn = document.getElementById('btn-stop-train');

        if (data.is_training) {
            statusText.innerText = 'Training...';
            statusText.style.color = '#4ade80';
            startBtn.style.display = 'none';
            stopBtn.style.display = 'block';
        } else {
            statusText.innerText = 'Idle';
            statusText.style.color = '#f87171';
            startBtn.style.display = 'block';
            stopBtn.style.display = 'none';
            // Update button text based on whether there's training history
            startBtn.innerText = data.has_history ? 'Resume Training' : 'Start Training';
        }

        document.getElementById('stat-episode').innerText = data.stats.episode;
        document.getElementById('current-loss').innerText = data.stats.loss.toFixed(6);
        document.getElementById('current-chips-round').innerText = (data.stats.avg_chips_per_round >= 0 ? '+' : '') + data.stats.avg_chips_per_round.toFixed(2);

        // Fetch History
        const histRes = await fetch(`${API_URL}/train/history`);
        const history = await histRes.json();
        updateChart(history);

    } catch (err) {
        console.error('Polling error:', err);
    }
}

function updateChart(history) {
    rawLossData = history.filter(h => h.type === 'loss').map(h => ({ x: h.episode, y: h.loss }));
    rawChipsData = history.filter(h => h.type === 'avg_chips').map(h => ({ x: h.episode, y: h.avg_chips_per_round }));

    lossChart.data.datasets[0].data = rawLossData;
    lossChart.data.datasets[1].data = computeEMA(rawLossData, smoothingAlpha);
    lossChart.update('none');

    chipsChart.data.datasets[0].data = rawChipsData;
    chipsChart.data.datasets[1].data = computeEMA(rawChipsData, smoothingAlpha);
    chipsChart.update('none');

    // Matchup chart
    updateMatchupChart(history);
}

function updateMatchupChart(history) {
    if (history) {
        const matchupEntries = history.filter(h => h.type === 'matchup');
        rawMatchupData = {};
        matchupEntries.forEach(h => {
            if (!rawMatchupData[h.opponent_id]) rawMatchupData[h.opponent_id] = [];
            rawMatchupData[h.opponent_id].push({ x: h.episode, y: h.avg_chips_per_round });
        });
    }

    const datasets = [];
    const opponentIds = Object.keys(rawMatchupData);

    opponentIds.forEach((oppId, i) => {
        const color = MATCHUP_COLORS[i % MATCHUP_COLORS.length];
        const agent = allAgents.find(a => a.id === oppId);
        const displayName = agent ? agent.displayName : oppId;
        const raw = rawMatchupData[oppId];

        // Raw (faint)
        datasets.push({
            label: `vs ${displayName} (raw)`,
            data: raw,
            borderColor: color + '4D',
            backgroundColor: 'transparent',
            borderWidth: 1,
            tension: 0,
            pointRadius: 0
        });

        // Smoothed (bold)
        datasets.push({
            label: `vs ${displayName}`,
            data: computeEMA(raw, matchupSmoothing),
            borderColor: color,
            backgroundColor: 'transparent',
            borderWidth: 2.5,
            tension: 0.3,
            pointRadius: 0
        });
    });

    matchupChart.data.datasets = datasets;
    matchupChart.update('none');
}

// Training Actions
document.getElementById('btn-start-train').onclick = async () => {
    const lr = parseFloat(document.getElementById('train-lr').value);
    const batchSize = parseInt(document.getElementById('train-batch').value);
    const episodes = parseInt(document.getElementById('train-episodes').value);

    await fetch(`${API_URL}/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lr, batch_size: batchSize, episodes })
    });
};

document.getElementById('btn-stop-train').onclick = async () => {
    await fetch(`${API_URL}/train/stop`, { method: 'POST' });
};

document.getElementById('btn-reset-agent').onclick = async () => {
    const agentSelect = document.getElementById('train-agent-select');
    const agentId = agentSelect.value;
    const agentName = agentSelect.options[agentSelect.selectedIndex].text;

    if (confirm(`Are you sure you want to reset training for ${agentName}? This will delete the saved model for this agent and start fresh.`)) {
        await fetch(`${API_URL}/train/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ agent_id: agentId })
        });
        // Clear all charts
        rawLossData = [];
        rawChipsData = [];
        lossChart.data.datasets[0].data = [];
        lossChart.data.datasets[1].data = [];
        lossChart.update();
        chipsChart.data.datasets[0].data = [];
        chipsChart.data.datasets[1].data = [];
        chipsChart.update();
        rawMatchupData = {};
        matchupChart.data.datasets = [];
        matchupChart.update();
        selectedOpponents = [];
        populateOpponentSelector();
    }
};

// Load trainable agents and populate opponent selector
async function loadTrainableAgents() {
    try {
        const res = await fetch(`${API_URL}/api/agents`);
        const data = await res.json();
        allAgents = data.agents;
        const select = document.getElementById('train-agent-select');

        // Filter for trainable agents only
        const trainableAgents = data.agents.filter(a => a.isTrainable);

        select.innerHTML = trainableAgents.map(a =>
            `<option value="${a.id}">${a.displayName}</option>`
        ).join('');

        // Re-populate opponent checkboxes when training agent changes
        select.addEventListener('change', () => populateOpponentSelector());

        populateOpponentSelector();
    } catch (err) {
        console.error('Error loading agents:', err);
    }
}

function populateOpponentSelector() {
    const container = document.getElementById('matchup-opponent-list');
    const trainingAgentId = document.getElementById('train-agent-select').value;

    // All agents except "human" and the currently-training agent
    const opponents = allAgents.filter(a => a.id !== 'human' && a.id !== trainingAgentId);

    container.innerHTML = opponents.map((a, i) => {
        const color = MATCHUP_COLORS[i % MATCHUP_COLORS.length];
        const checked = selectedOpponents.includes(a.id) ? 'checked' : '';
        return `
            <label class="matchup-opponent-item">
                <input type="checkbox" value="${a.id}" ${checked}
                       onchange="onOpponentToggle()">
                <span class="opponent-color-swatch" style="background: ${color};"></span>
                ${a.displayName}
            </label>
        `;
    }).join('');
}

async function onOpponentToggle() {
    const checkboxes = document.querySelectorAll('#matchup-opponent-list input[type="checkbox"]');
    selectedOpponents = Array.from(checkboxes).filter(cb => cb.checked).map(cb => cb.value);

    await fetch(`${API_URL}/train/matchup-opponents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ opponent_ids: selectedOpponents })
    });
}

// --- STATE VALUE ANALYZER LOGIC ---

// State configuration
let analyzerState = {
    playerHand: 'Q',
    boardCard: null,
    history: [], // List of action ints

    // Cached state from backend
    calculatedState: null
};

// Card selector logic
function initCardSelectors() {
    // Hand selector
    document.querySelectorAll('#hand-selector .card-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#hand-selector .card-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            analyzerState.playerHand = btn.dataset.card;
            updatePreview();
        });
    });

    // Board selector
    document.querySelectorAll('#board-selector .card-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#board-selector .card-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            analyzerState.boardCard = btn.dataset.card || null;
            updatePreview();
        });
    });

    // Reset Sequence Button
    document.getElementById('btn-reset-sequence').onclick = resetSequence;
}

async function updateSequenceUI() {
    const container = document.getElementById('next-actions-container');
    const display = document.getElementById('sequence-display');

    // Fetch calculated state from backend
    try {
        const res = await fetch(`${API_URL}/analyze/calculate_state`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ history: analyzerState.history })
        });
        const state = await res.json();
        analyzerState.calculatedState = state;

        // 1. Update History Display
        display.innerHTML = '';
        if (state.history && state.history.length > 0) {
            state.history.forEach((h, index) => {
                const badge = document.createElement('div');
                badge.className = `history-item p${h.player}`;
                const actionName = getActionName(h.action);
                badge.innerText = `P${h.player}: ${actionName}`;
                display.appendChild(badge);
            });
        } else {
            display.innerHTML = '<span class="placeholder-text">Start of game (Ante 1)</span>';
        }

        // 2. Update Next Action Buttons
        container.innerHTML = '';
        if (state.is_finished) {
            container.innerHTML = `<span style="opacity: 0.7; font-size: 0.9rem;">Round Finished - Winner: Player ${state.winner}</span>`;
        } else {
            state.legal_actions.forEach(actionInt => {
                const btn = document.createElement('button');
                btn.className = 'action-btn';
                btn.innerText = getActionName(actionInt);
                btn.onclick = () => addToSequence(actionInt);
                container.appendChild(btn);
            });
        }

        updatePreview();

    } catch (err) {
        console.error("Error updating sequence UI:", err);
        container.innerHTML = '<span style="color: #f87171;">Error syncing state</span>';
    }
}

function addToSequence(action) {
    analyzerState.history.push(action);
    updateSequenceUI();
}

function resetSequence() {
    analyzerState.history = [];
    updateSequenceUI();
}

function getActionName(actionInt) {
    // 0: FOLD, 1: CALL/CHECK, 2: RAISE/BET
    // We can infer CHECK vs CALL based on context if we had it, but simpler to just use slash
    const map = { 0: 'FOLD', 1: 'CHECK/CALL', 2: 'BET/RAISE' };
    return map[actionInt] || 'UNKNOWN';
}

function updatePreview() {
    const state = analyzerState.calculatedState;

    document.getElementById('preview-hand').innerText = analyzerState.playerHand;

    // Board: Use user selection if set, otherwise use calculated state board (if valid)
    // Actually, user selection should override what the "agent" sees for evaluation,
    // but for the state preview, we might want to show what the game thinks?
    // User wants "board card... can be more automatic".
    // But they explicitly said "player hand and board card... are pretty good".
    // So we stick to user selection for those.
    // However, if the game is Pre-flop, board should be hidden?
    // The user's selection overrides everything for the *Agent's POV*.

    const boardCard = analyzerState.boardCard || (state && state.board ? state.board : '?');
    document.getElementById('preview-board').innerText = boardCard;

    if (state) {
        const totalPot = state.pot[0] + state.pot[1];
        document.getElementById('preview-pot').innerText = totalPot + ` (P0: ${state.pot[0]}, P1: ${state.pot[1]})`;

        // Also update board selector to likely board from state if user hasn't forced one?
        // No, let's keep it manual as requested.
    }
}

async function evaluateState() {
    const body = document.getElementById('action-values-body');
    body.innerHTML = '<tr><td colspan="3" style="text-align: center;"><div class="loading-spinner"></div> Evaluating...</td></tr>';

    try {
        const res = await fetch(`${API_URL}/analyze/state`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                player_hand: analyzerState.playerHand,
                board: analyzerState.boardCard,
                history: analyzerState.history
                // pot/round are inferred from history by backend now
            })
        });
        const data = await res.json();

        // Check if finished
        if (data.is_finished) {
            body.innerHTML = '<tr><td colspan="3" style="text-align: center; opacity: 0.7;">Game is finished. No actions to evaluate.</td></tr>';
            return;
        }

        renderActionValues(data.results);
    } catch (err) {
        console.error('Evaluation error:', err);
        body.innerHTML = '<tr><td colspan="3" style="text-align: center; color: #f87171;">Error evaluating state</td></tr>';
    }
}

function renderActionValues(results) {
    const body = document.getElementById('action-values-body');
    body.innerHTML = '';

    // Sort by value (descending)
    results.sort((a, b) => b.value - a.value);
    const bestValue = results[0]?.value;

    const actionNames = { 0: 'FOLD', 1: 'CALL', 2: 'RAISE' };

    results.forEach(res => {
        const isBest = res.value === bestValue;
        const tr = document.createElement('tr');
        if (isBest) tr.className = 'best-action-row';

        tr.innerHTML = `
            <td>
                <span class="action-label ${actionNames[res.action].toLowerCase()}">${actionNames[res.action]}</span>
            </td>
            <td class="${res.value >= 0 ? 'value-pos' : 'value-neg'}">${res.value.toFixed(4)}</td>
            <td>
                ${isBest ? '<span class="best-badge">BEST</span>' : ''}
            </td>
        `;
        body.appendChild(tr);
    });
}

// Wire up evaluate button
document.getElementById('btn-evaluate').onclick = evaluateState;

// Init
initCharts();
initSmoothingControl();
initMatchupSmoothing();
initResetZoom();
initCarousel();
initCardSelectors();
updateSequenceUI();
loadTrainableAgents();
setInterval(pollStatus, 2000);
pollStatus();

