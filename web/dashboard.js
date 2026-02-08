const API_URL = 'http://localhost:8000';

let chart;
let historyData = {
    loss: [],
    winRate: []
};

// Initialize Chart
function initChart() {
    const ctx = document.getElementById('trainingChart').getContext('2d');
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Batch Loss',
                    data: [],
                    borderColor: '#f4d03f',
                    backgroundColor: 'rgba(244, 208, 63, 0.1)',
                    yAxisID: 'y',
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: 'Win Rate (%)',
                    data: [],
                    borderColor: '#4ade80',
                    backgroundColor: 'rgba(74, 222, 128, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.3,
                    stepped: true
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
                    title: { display: true, text: 'Loss', color: '#f4d03f' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Win Rate (%)', color: '#4ade80' },
                    min: 0,
                    max: 100,
                    grid: { drawOnChartArea: false },
                    ticks: { color: 'rgba(255, 255, 255, 0.6)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#fff', font: { family: 'Inter' } }
                }
            }
        }
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
        }

        document.getElementById('stat-episode').innerText = data.stats.episode;
        document.getElementById('current-loss').innerText = data.stats.loss.toFixed(6);
        document.getElementById('current-win-rate').innerText = (data.stats.win_rate * 100).toFixed(1) + '%';

        // Fetch History
        const histRes = await fetch(`${API_URL}/train/history`);
        const history = await histRes.json();
        updateChart(history);

    } catch (err) {
        console.error('Polling error:', err);
    }
}

function updateChart(history) {
    const lossData = history.filter(h => h.type === 'loss').map(h => ({ x: h.episode, y: h.loss }));
    const winRateData = history.filter(h => h.type === 'win_rate').map(h => ({ x: h.episode, y: h.win_rate * 100 }));

    chart.data.datasets[0].data = lossData;
    chart.data.datasets[1].data = winRateData;
    chart.update('none'); // Update without animation for performance
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

// --- SIMULATION LOGIC ---
let simState = null;

async function resetSimulation() {
    console.log("Resetting simulation...");
    const res = await fetch(`${API_URL}/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_types: ['human', 'human'] })
    });
    simState = await res.json();
    console.log("Simulation state reset:", simState);
    updateSimUI();
}

async function stepSimulation(action) {
    if (!simState || simState.is_finished) return;

    const res = await fetch(`${API_URL}/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action })
    });
    simState = await res.json();
    updateSimUI();
}

async function updateSimUI() {
    if (!simState) return;

    document.getElementById('sim-pot').innerText = simState.pot;

    // Update Board
    const boardEl = document.getElementById('sim-board');
    boardEl.innerHTML = '';
    simState.board.forEach(card => {
        const div = document.createElement('div');
        div.className = 'community-card dealing';
        div.innerText = card;
        boardEl.appendChild(div);
    });

    // Update Hand
    const hand = simState.player_hands[simState.current_player];
    document.getElementById('sim-hand').innerText = hand === "HIDDEN" ? "?" : hand;
    document.getElementById('sim-player-label').innerText = `Player ${simState.current_player} Turn (${hand})`;

    // Fetch decisions from Agent
    if (!simState.is_finished) {
        console.log("Fetching decisions...");
        const decRes = await fetch(`${API_URL}/simulate/decision`, { method: 'POST' });
        const decData = await decRes.json();
        console.log("Decisions received:", decData);
        renderDecisions(decData.results);
    } else {
        document.getElementById('decision-body').innerHTML = `
            <tr>
                <td colspan="3" style="text-align: center; color: #f4d03f; font-weight: bold;">
                    GAME OVER - Winner: Player ${simState.winner}
                </td>
            </tr>
        `;
    }
}

function renderDecisions(results) {
    const body = document.getElementById('decision-body');
    body.innerHTML = '';

    // Sort by value (descending)
    results.sort((a, b) => b.value - a.value);
    const bestValue = results[0].value;

    results.forEach(res => {
        const isBest = res.value === bestValue;
        const tr = document.createElement('tr');
        if (isBest) tr.className = 'best-action-row';

        tr.innerHTML = `
            <td>
                <button class="reset-btn ${isBest ? 'accent' : ''}" style="padding: 5px 15px; width: 100px;" onclick="stepSimulation('${res.action}')">
                    ${res.action}
                </button>
            </td>
            <td class="${res.value >= 0 ? 'value-pos' : 'value-neg'}">${res.value.toFixed(4)}</td>
            <td>
                ${isBest ? '<span class="best-badge">RECOMMENDED</span>' : ''}
            </td>
        `;
        body.appendChild(tr);
    });
}

// Global scope for onclick
window.stepSimulation = stepSimulation;

document.getElementById('sim-reset').onclick = resetSimulation;

// Init
initChart();
setInterval(pollStatus, 2000);
resetSimulation();
pollStatus();
