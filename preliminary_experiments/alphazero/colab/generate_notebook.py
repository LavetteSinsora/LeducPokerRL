"""
Generates AlphaZero_Leduc.ipynb — a self-contained Colab notebook
with GPU support for training the AlphaZero-style Leduc Hold'em agent.

Run:  python generate_notebook.py
Output: AlphaZero_Leduc.ipynb  (in the same directory)
"""
import json
import os
from pathlib import Path

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

def code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    })


# ─────────────────────────────────────────────────────────────────────────────
md("# AlphaZero-Style Leduc Hold'em Agent\n\n"
   "Self-contained training notebook.  \n"
   "**Runtime → Change runtime type → GPU** before running.\n\n"
   "Checkpoints are saved to your Google Drive automatically.")

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
code('''\
import sys, os, json, random, copy, time, signal
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Tuple
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── GPU / Drive setup ─────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# Mount Google Drive for persistent checkpoints
try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = Path("/content/drive/MyDrive/AlphaZeroLeduc")
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH = str(DRIVE_DIR / "checkpoint.pt")
    HISTORY_PATH    = DRIVE_DIR / "train_history.json"
    print(f"Drive mounted. Checkpoint path: {CHECKPOINT_PATH}")
except ImportError:
    # Running locally
    DRIVE_DIR = Path("outputs/az_colab")
    DRIVE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH = str(DRIVE_DIR / "checkpoint.pt")
    HISTORY_PATH    = DRIVE_DIR / "train_history.json"
    print(f"No Colab Drive — saving locally to {DRIVE_DIR}")
''')

# ── Cell 2: Game Engine ────────────────────────────────────────────────────────
md("## Game Engine (Leduc Hold'em)")
code('''\
# ─── Observation ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Observation:
    player_hand: str
    board: Optional[str]
    pot: List[int]
    current_player: int
    current_round: int
    legal_actions: List[Any]
    is_finished: bool
    raises_this_round: int = 0
    opponent_stats: Optional[Any] = None
    action_history: Optional[tuple] = None


# ─── Action ───────────────────────────────────────────────────────────────────

class Action(IntEnum):
    FOLD  = 0
    CALL  = 1
    RAISE = 2


# ─── LeducGame ────────────────────────────────────────────────────────────────

class LeducGame:
    """
    Leduc Hold\'em: 6 cards (2×J, 2×Q, 2×K), 2 players, 2 rounds.
    Pre-flop bet=2, Flop bet=4, max 2 raises per round.
    """
    CARDS       = [\'J\', \'J\', \'Q\', \'Q\', \'K\', \'K\']
    BET_AMOUNTS = [2, 4]
    MAX_RAISES  = 2

    def __init__(self):
        self.reset()

    def reset(self):
        self.deck   = list(self.CARDS)
        random.shuffle(self.deck)
        self.player_hands        = [self.deck.pop(), self.deck.pop()]
        self.board               = None
        self.pot                 = [1, 1]
        self.current_round       = 0
        self.current_player      = 0
        self.raises_this_round   = 0
        self.is_finished         = False
        self.winner              = None
        self.history             = []
        self.round_betting_ended = False
        self.last_action_was_raise = False
        return self.get_observation()

    def set_state(self, observation: Observation):
        self.current_round     = observation.current_round
        self.pot               = list(observation.pot)
        self.current_player    = observation.current_player
        self.board             = observation.board
        self.is_finished       = observation.is_finished
        self.raises_this_round = observation.raises_this_round
        self.winner            = None
        self.player_hands[self.current_player] = observation.player_hand
        other = 1 - self.current_player
        if self.player_hands[other] not in self.CARDS:
            self.player_hands[other] = \'UNKNOWN\'

    def copy(self):
        return copy.deepcopy(self)

    def step(self, action):
        if self.is_finished:
            raise ValueError("Game is already finished.")
        player     = self.current_player
        bet_amount = self.BET_AMOUNTS[self.current_round]

        if action == Action.FOLD:
            self.history.append((player, "FOLD"))
            self.is_finished = True
            self.winner      = 1 - player
            return self.get_observation(), self.get_reward(), True, {}

        elif action == Action.CALL:
            self.history.append((player, "CALL"))
            other_pot = self.pot[1 - player]
            if other_pot > self.pot[player]:
                self.pot[player]         = other_pot
                self.round_betting_ended = True
            else:
                if player == 1:
                    self.round_betting_ended = True

        elif action == Action.RAISE:
            if self.raises_this_round >= self.MAX_RAISES:
                raise ValueError("Max raises reached.")
            self.history.append((player, "RAISE"))
            other_pot          = self.pot[1 - player]
            self.pot[player]   = other_pot + bet_amount
            self.raises_this_round += 1
            self.last_action_was_raise = True
            self.round_betting_ended   = False

        self.current_player = 1 - self.current_player
        if self.round_betting_ended:
            if self.current_round == 0:
                self._transition_to_flop()
            else:
                self._showdown()
        return self.get_observation(), self.get_reward(), self.is_finished, {}

    def _transition_to_flop(self):
        self.current_round       = 1
        self.board               = self.deck.pop()
        self.current_player      = 0
        self.raises_this_round   = 0
        self.round_betting_ended = False
        self.last_action_was_raise = False

    def _showdown(self):
        self.is_finished = True
        p1, p2 = self.player_hands
        if p1 == \'UNKNOWN\' or p2 == \'UNKNOWN\':
            self.winner = -2; return
        s1 = self._evaluate_hand(p1)
        s2 = self._evaluate_hand(p2)
        self.winner = 0 if s1 > s2 else (1 if s2 > s1 else -1)

    def _evaluate_hand(self, card):
        vals = {\'J\': 0, \'Q\': 1, \'K\': 2}
        return 10 + vals[card] if card == self.board else vals[card]

    def get_reward(self):
        if not self.is_finished: return [0, 0]
        if self.winner == 0:  return [ self.pot[1], -self.pot[1]]
        if self.winner == 1:  return [-self.pot[0],  self.pot[0]]
        return [0, 0]

    def get_observation(self, viewer_id=None) -> Observation:
        if viewer_id is None:
            viewer_id = self.current_player
        return Observation(
            player_hand=self.player_hands[viewer_id],
            board=self.board, pot=list(self.pot),
            current_player=self.current_player,
            current_round=self.current_round,
            legal_actions=self._get_legal_actions(),
            is_finished=self.is_finished,
            raises_this_round=self.raises_this_round,
        )

    def _get_legal_actions(self):
        actions = [Action.FOLD, Action.CALL]
        if self.raises_this_round < self.MAX_RAISES:
            actions.append(Action.RAISE)
        return actions

    def get_legal_actions(self):
        return self._get_legal_actions()


print("Game engine loaded.")
''')

# ── Cell 3: AlphaZero Modules ─────────────────────────────────────────────────
md("## AlphaZero Modules (GPU-aware)")
code('''\
# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class AZConfig:
    d_model:       int            = 8
    n_events:      int            = 9
    state_hidden:  Tuple[int,...] = (16, 16)
    belief_hidden: Tuple[int,...] = (32, 32)
    q_hidden:      Tuple[int,...] = (64, 64, 64)
    k_rollouts:    int            = 10
    temperature:   float          = 1.0
    n_episodes:    int            = 200_000
    lr:            float          = 1e-3
    lambda_belief: float          = 0.1


# ─── State encoder constants ──────────────────────────────────────────────────

CARD_TO_IDX = {\'J\': 0, \'Q\': 1, \'K\': 2}
IDX_TO_CARD = [\'J\', \'Q\', \'K\']
N_EVENTS    = 9

def action_event_id(player: int, action_value: int) -> int:
    return player * 3 + action_value

def deal_event_id(card: str) -> int:
    return 6 + CARD_TO_IDX[card]


# ─── State encoder ────────────────────────────────────────────────────────────

class StateEncoder(nn.Module):
    def __init__(self, config: AZConfig):
        super().__init__()
        d = config.d_model
        self.event_embed = nn.Embedding(config.n_events, d)
        self.cross_attn  = nn.MultiheadAttention(embed_dim=d, num_heads=1, batch_first=True)
        layers, in_dim = [], 2 * d
        for h in config.state_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]; in_dim = h
        layers.append(nn.Linear(in_dim, d))
        self.state_mlp = nn.Sequential(*layers)

    def encode_event(self, event_id: int, P_current: torch.Tensor,
                     P_history: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        eid     = torch.tensor([event_id], dtype=torch.long, device=P_current.device)
        e_base  = self.event_embed(eid)                      # (1, d)
        KV      = torch.stack(P_history, dim=0).unsqueeze(0) # (1, T, d)
        Q       = e_base.unsqueeze(0)                         # (1, 1, d)
        context, _ = self.cross_attn(Q, KV, KV)              # (1, 1, d)
        e_prime = e_base.squeeze(0) + context.squeeze(0).squeeze(0)
        P_new   = self.state_mlp(torch.cat([P_current, e_prime], dim=-1))
        return e_prime, P_new


# ─── Belief network & state ───────────────────────────────────────────────────

def informed_prior(h: str, device=None) -> torch.Tensor:
    """Marginal distribution over opponent\'s hand given own hand h."""
    if device is None: device = DEVICE
    prior = torch.full((3,), 2.0 / 5.0, device=device)
    prior[CARD_TO_IDX[h]] = 1.0 / 5.0
    return prior


class BeliefNet(nn.Module):
    def __init__(self, config: AZConfig):
        super().__init__()
        d = config.d_model
        layers, curr = [], 3 + d + d
        for h in config.belief_hidden:
            layers += [nn.Linear(curr, h), nn.ReLU()]; curr = h
        layers.append(nn.Linear(curr, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, b: torch.Tensor, P_t: torch.Tensor,
                e_prime: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.mlp(torch.cat([b, P_t, e_prime], dim=-1)), dim=-1)


@dataclass
class BeliefState:
    b_mine:    torch.Tensor
    b_opp:     List[torch.Tensor]
    P_current: torch.Tensor
    P_history: List[torch.Tensor]

    def copy(self) -> "BeliefState":
        return BeliefState(
            b_mine=self.b_mine.detach().clone(),
            b_opp=[b.detach().clone() for b in self.b_opp],
            P_current=self.P_current.detach().clone(),
            P_history=[p.detach().clone() for p in self.P_history],
        )


def make_belief_state(h_i: str, d_model: int = 8, device=None) -> BeliefState:
    if device is None: device = DEVICE
    P0 = torch.zeros(d_model, device=device)
    return BeliefState(
        b_mine=informed_prior(h_i, device),
        b_opp=[informed_prior(hj, device) for hj in IDX_TO_CARD],
        P_current=P0.clone(),
        P_history=[P0],
    )


def update_belief_state(bs: BeliefState, actor, player_i: int,
                        e_prime: torch.Tensor, P_new: torch.Tensor,
                        belief_net: BeliefNet) -> None:
    P_before = bs.P_current
    if actor is None:
        bs.b_mine = belief_net(bs.b_mine, P_before, e_prime)
        for k in range(3):
            bs.b_opp[k] = belief_net(bs.b_opp[k], P_before, e_prime)
    elif actor != player_i:
        bs.b_mine = belief_net(bs.b_mine, P_before, e_prime)
    else:
        for k in range(3):
            bs.b_opp[k] = belief_net(bs.b_opp[k], P_before, e_prime)
    bs.P_history.append(P_new)
    bs.P_current = P_new


# ─── Q-network ────────────────────────────────────────────────────────────────

class QNet(nn.Module):
    def __init__(self, config: AZConfig):
        super().__init__()
        d = config.d_model
        layers, curr = [], d + 3 + 3
        for h in config.q_hidden:
            layers += [nn.Linear(curr, h), nn.ReLU()]; curr = h
        layers.append(nn.Linear(curr, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, P_t: torch.Tensor, h_onehot: torch.Tensor,
                b: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([P_t, h_onehot, b], dim=-1))


def hand_onehot(h: str, device=None) -> torch.Tensor:
    if device is None: device = DEVICE
    v = torch.zeros(3, device=device)
    v[CARD_TO_IDX[h]] = 1.0
    return v


def masked_softmax(q_vals: torch.Tensor, legal_actions: list, T: float) -> torch.Tensor:
    mask = torch.full((3,), float(\'-inf\'), device=q_vals.device)
    for a in legal_actions:
        mask[int(a)] = 0.0
    return F.softmax((q_vals + mask) / T, dim=-1)


# ─── Rollout helpers ──────────────────────────────────────────────────────────

def _step_game(game: LeducGame, action: Action):
    actor    = game.current_player
    pre_board = game.board
    _, reward_list, done, _ = game.step(action)
    act_eid  = action_event_id(actor, int(action))
    deal_eid = None
    if game.board is not None and game.board != pre_board:
        deal_eid = deal_event_id(game.board)
    return reward_list, done, act_eid, deal_eid, actor


def _sample_action(game: LeducGame, actor: int, h_actor: str,
                   b_actor: torch.Tensor, P_current: torch.Tensor,
                   q_net: QNet, T: float) -> Action:
    legal  = game.get_legal_actions()
    q_vals = q_net(P_current, hand_onehot(h_actor, P_current.device), b_actor)
    mask   = torch.full((3,), float(\'-inf\'), device=q_vals.device)
    for a in legal:
        mask[int(a)] = 0.0
    probs = F.softmax((q_vals + mask) / T, dim=-1)
    return Action(torch.multinomial(probs, 1).item())


def _run_single_rollout(game: LeducGame, player_i: int, h_i: str, h_j: str,
                        bs_i: BeliefState, b_opp_j: torch.Tensor,
                        first_action: Action, state_enc: StateEncoder,
                        belief_net: BeliefNet, q_net: QNet, T: float) -> float:
    reward_list, done, act_eid, deal_eid, _ = _step_game(game, first_action)
    if done:
        return reward_list[player_i]

    e_prime, P_new = state_enc.encode_event(act_eid, bs_i.P_current, bs_i.P_history)
    b_opp_j = belief_net(b_opp_j, bs_i.P_current, e_prime)
    bs_i.P_history.append(P_new); bs_i.P_current = P_new

    if deal_eid is not None:
        e_prime_d, P_new_d = state_enc.encode_event(deal_eid, bs_i.P_current, bs_i.P_history)
        bs_i.b_mine = belief_net(bs_i.b_mine, bs_i.P_current, e_prime_d)
        b_opp_j     = belief_net(b_opp_j,     bs_i.P_current, e_prime_d)
        bs_i.P_history.append(P_new_d); bs_i.P_current = P_new_d

    while not game.is_finished:
        actor = game.current_player
        if actor == player_i:
            action = _sample_action(game, actor, h_i,  bs_i.b_mine, bs_i.P_current, q_net, T)
        else:
            action = _sample_action(game, actor, h_j,  b_opp_j,     bs_i.P_current, q_net, T)

        reward_list, done, act_eid, deal_eid, _ = _step_game(game, action)
        if done:
            return reward_list[player_i]

        e_prime, P_new = state_enc.encode_event(act_eid, bs_i.P_current, bs_i.P_history)
        if actor != player_i:
            bs_i.b_mine = belief_net(bs_i.b_mine, bs_i.P_current, e_prime)
        else:
            b_opp_j = belief_net(b_opp_j, bs_i.P_current, e_prime)
        bs_i.P_history.append(P_new); bs_i.P_current = P_new

        if deal_eid is not None:
            e_prime_d, P_new_d = state_enc.encode_event(deal_eid, bs_i.P_current, bs_i.P_history)
            bs_i.b_mine = belief_net(bs_i.b_mine, bs_i.P_current, e_prime_d)
            b_opp_j     = belief_net(b_opp_j,     bs_i.P_current, e_prime_d)
            bs_i.P_history.append(P_new_d); bs_i.P_current = P_new_d

    return game.get_reward()[player_i]


def pimc_search(obs: Observation, player_i: int, h_i: str,
                bs_i: BeliefState, legal_actions: list,
                state_enc: StateEncoder, belief_net: BeliefNet, q_net: QNet,
                k: int = 10, T: float = 1.0) -> torch.Tensor:
    legal_ids = {int(a) for a in legal_actions}
    dev    = bs_i.P_current.device
    q_star = torch.full((3,), float(\'-inf\'), device=dev)

    with torch.no_grad():
        game_template = LeducGame()
        game_template.set_state(obs)
        game_template.player_hands[player_i] = h_i

        for hj_idx, h_j in enumerate(IDX_TO_CARD):
            weight = bs_i.b_mine[hj_idx].item()
            if weight < 1e-8:
                continue
            b_opp_j_base   = bs_i.b_opp[hj_idx]
            action_returns = {aid: [] for aid in legal_ids}

            for action in legal_actions:
                aid = int(action)
                for _ in range(k):
                    game_copy = game_template.copy()
                    game_copy.player_hands[1 - player_i] = h_j
                    ret = _run_single_rollout(
                        game=game_copy, player_i=player_i, h_i=h_i, h_j=h_j,
                        bs_i=bs_i.copy(), b_opp_j=b_opp_j_base.clone(),
                        first_action=action, state_enc=state_enc,
                        belief_net=belief_net, q_net=q_net, T=T,
                    )
                    action_returns[aid].append(ret)

            for aid, returns in action_returns.items():
                avg = sum(returns) / len(returns)
                if q_star[aid] == float(\'-inf\'):
                    q_star[aid] = 0.0
                q_star[aid] = q_star[aid] + weight * avg

    return q_star


# ─── Episode data structures ──────────────────────────────────────────────────

@dataclass
class DecisionRecord:
    step:          int
    player:        int
    legal_actions: list
    q_star:        torch.Tensor


@dataclass
class Episode:
    hands:     List[str]
    events:    List[tuple]
    decisions: List[DecisionRecord]


# ─── Episode play (no-grad) ───────────────────────────────────────────────────

def _play_episode(state_enc: StateEncoder, belief_net: BeliefNet,
                  q_net: QNet, config: AZConfig) -> Episode:
    game = LeducGame(); game.reset()
    hands = list(game.player_hands)
    dev = next(state_enc.parameters()).device

    with torch.no_grad():
        bs = [make_belief_state(hands[p], config.d_model, dev) for p in range(2)]

    events: List[tuple] = []
    decisions: List[DecisionRecord] = []

    with torch.no_grad():
        while not game.is_finished:
            p   = game.current_player
            obs = game.get_observation(viewer_id=p)
            legal = game.get_legal_actions()

            q_star = pimc_search(obs=obs, player_i=p, h_i=hands[p], bs_i=bs[p],
                                 legal_actions=legal, state_enc=state_enc,
                                 belief_net=belief_net, q_net=q_net,
                                 k=config.k_rollouts, T=config.temperature)

            decisions.append(DecisionRecord(step=len(events), player=p,
                                            legal_actions=legal, q_star=q_star.clone()))

            probs      = masked_softmax(q_star, legal, config.temperature)
            action_idx = torch.multinomial(probs, 1).item()
            action     = Action(action_idx)

            _, done, act_eid, deal_eid, actor = _step_game(game, action)
            events.append((act_eid, actor))
            if deal_eid is not None:
                events.append((deal_eid, None))

            for pi in range(2):
                e_prime, P_new = state_enc.encode_event(act_eid, bs[pi].P_current, bs[pi].P_history)
                update_belief_state(bs[pi], actor, pi, e_prime, P_new, belief_net)
                if deal_eid is not None:
                    e_prime_d, P_new_d = state_enc.encode_event(deal_eid, bs[pi].P_current, bs[pi].P_history)
                    update_belief_state(bs[pi], None, pi, e_prime_d, P_new_d, belief_net)

    return Episode(hands=hands, events=events, decisions=decisions)


# ─── Episode replay (with-grad) ───────────────────────────────────────────────

def _replay_episode(episode: Episode, state_enc: StateEncoder,
                    belief_net: BeliefNet, q_net: QNet,
                    config: AZConfig) -> torch.Tensor:
    d   = config.d_model
    dev = next(state_enc.parameters()).device

    P_current = torch.zeros(d, device=dev)
    P_history = [torch.zeros(d, device=dev)]

    bs = [
        BeliefState(
            b_mine=informed_prior(episode.hands[p], dev),
            b_opp=[informed_prior(hj, dev) for hj in IDX_TO_CARD],
            P_current=P_current.clone(),
            P_history=[P_current.clone()],
        )
        for p in range(2)
    ]

    dec_at_step = {}
    for dp in episode.decisions:
        dec_at_step.setdefault(dp.step, []).append(dp)

    q_losses, belief_losses = [], []

    for event_idx, (event_id, actor) in enumerate(episode.events):
        for dp in dec_at_step.get(event_idx, []):
            p          = dp.player
            h_opp_idx  = CARD_TO_IDX[episode.hands[1 - p]]
            q_vals     = q_net(bs[p].P_current, hand_onehot(episode.hands[p], dev), bs[p].b_mine)
            legal_ids  = [int(a) for a in dp.legal_actions]
            q_losses.append(F.mse_loss(q_vals[legal_ids], dp.q_star[legal_ids].to(dev)))
            b_log = torch.log(bs[p].b_mine[h_opp_idx].clamp(min=1e-8))
            belief_losses.append(-b_log)

        for p in range(2):
            e_prime_p, P_new_p = state_enc.encode_event(event_id, bs[p].P_current, bs[p].P_history)
            update_belief_state(bs[p], actor, p, e_prime_p, P_new_p, belief_net)

    if not q_losses:
        return torch.tensor(0.0, device=dev)

    L_q = torch.stack(q_losses).mean()
    L_b = torch.stack(belief_losses).mean()
    return L_q + config.lambda_belief * L_b


# ─── Trainer ──────────────────────────────────────────────────────────────────

class AZTrainer:
    def __init__(self, config: AZConfig, state_enc: StateEncoder,
                 belief_net: BeliefNet, q_net: QNet):
        self.config      = config
        self.state_enc   = state_enc
        self.belief_net  = belief_net
        self.q_net       = q_net
        self.optimizer   = torch.optim.Adam(
            list(state_enc.parameters())
            + list(belief_net.parameters())
            + list(q_net.parameters()),
            lr=config.lr,
        )
        self.episode_count = 0
        self.loss_history: List[float] = []

    def train(self, log_every=1000, checkpoint_path=None,
              checkpoint_every=5000, callback=None):
        interrupted = [False]
        def _handle(sig, frame):
            print("\\n[train] Interrupt — saving checkpoint after this episode.")
            interrupted[0] = True
        prev = signal.signal(signal.SIGINT, _handle)

        try:
            start  = self.episode_count
            target = start + self.config.n_episodes
            t0     = time.time()

            for ep in range(start, target):
                loss_val = self.train_one_episode()
                self.loss_history.append(loss_val)
                if callback:
                    callback({"episode": self.episode_count, "loss": loss_val})

                if (ep - start + 1) % log_every == 0:
                    recent  = self.loss_history[-log_every:]
                    avg     = sum(recent) / len(recent)
                    elapsed = time.time() - t0
                    done    = ep - start + 1
                    eta_min = (elapsed / done) * (target - ep - 1) / 60 if done else 0
                    print(f"ep {self.episode_count:>7d}/{target} | "
                          f"avg_loss {avg:.4f} | "
                          f"elapsed {elapsed/60:.1f}m | ETA {eta_min:.1f}m")

                if checkpoint_path and self.episode_count % checkpoint_every == 0:
                    self.save(checkpoint_path)
                    print(f"  [ckpt] saved → {checkpoint_path}")

                if interrupted[0]:
                    break
        finally:
            signal.signal(signal.SIGINT, prev)
            if checkpoint_path:
                self.save(checkpoint_path)
                print(f"[train] final checkpoint → {checkpoint_path}")

    def train_one_episode(self) -> float:
        self.state_enc.eval(); self.belief_net.eval(); self.q_net.eval()
        episode = _play_episode(self.state_enc, self.belief_net, self.q_net, self.config)
        self.state_enc.train(); self.belief_net.train(); self.q_net.train()
        self.optimizer.zero_grad()
        loss = _replay_episode(episode, self.state_enc, self.belief_net, self.q_net, self.config)
        loss.backward()
        self.optimizer.step()
        self.episode_count += 1
        return loss.item()

    def save(self, path: str):
        torch.save({
            "state_enc":  self.state_enc.state_dict(),
            "belief_net": self.belief_net.state_dict(),
            "q_net":      self.q_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "episode":    self.episode_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.state_enc.load_state_dict(ckpt["state_enc"])
        self.belief_net.load_state_dict(ckpt["belief_net"])
        self.q_net.load_state_dict(ckpt["q_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.episode_count = ckpt.get("episode", 0)


def build_trainer(config: AZConfig = None):
    if config is None: config = AZConfig()
    state_enc  = StateEncoder(config).to(DEVICE)
    belief_net = BeliefNet(config).to(DEVICE)
    q_net      = QNet(config).to(DEVICE)
    return AZTrainer(config, state_enc, belief_net, q_net)


print("AlphaZero modules loaded.")
''')

# ── Cell 4: Config & Build ─────────────────────────────────────────────────────
md("## Configure & Build\n\nEdit hyperparameters here, then run the cell.")
code('''\
# ── Hyperparameters ───────────────────────────────────────────────────────────
# Edit these as needed before starting training.

config = AZConfig(
    k_rollouts    = 10,       # PIMC rollouts per action per hand
    temperature   = 1.0,      # softmax temperature for action sampling
    lr            = 1e-3,     # Adam learning rate
    lambda_belief = 0.1,      # weight for belief cross-entropy loss
    n_episodes    = 200_000,  # total training episodes
)

# ── Build trainer ─────────────────────────────────────────────────────────────
trainer = build_trainer(config)
print(f"Trainer built. Device: {DEVICE}")

# ── Resume from checkpoint (optional) ────────────────────────────────────────
if os.path.exists(CHECKPOINT_PATH):
    trainer.load(CHECKPOINT_PATH)
    print(f"Resumed from checkpoint: ep {trainer.episode_count}")
else:
    print("No checkpoint found — starting fresh.")
''')

# ── Cell 5: Train ──────────────────────────────────────────────────────────────
md("## Train\n\n"
   "Training runs for `n_episodes` episodes (starting from the current checkpoint).  \n"
   "Checkpoints are saved to Google Drive every 5000 episodes.  \n"
   "**Interrupt with the stop button** — the checkpoint is saved before exit.")
code('''\
history = []

def callback(event):
    history.append(event)
    if len(history) % 500 == 0:
        HISTORY_PATH.write_text(json.dumps(history, indent=2))

t_start = time.time()
trainer.train(
    log_every         = 1000,
    checkpoint_path   = CHECKPOINT_PATH,
    checkpoint_every  = 5000,
    callback          = callback,
)
elapsed = time.time() - t_start

HISTORY_PATH.write_text(json.dumps(history, indent=2))
print(f"\\nDone. {trainer.episode_count} episodes in {elapsed/60:.1f} min.")
print(f"Checkpoint : {CHECKPOINT_PATH}")
print(f"History    : {HISTORY_PATH}")
''')

# ── Cell 6: Plot ───────────────────────────────────────────────────────────────
md("## Loss Curve\n\nRun after training (or to inspect a resumed run).")
code('''\
import matplotlib.pyplot as plt

if HISTORY_PATH.exists():
    hist = json.loads(HISTORY_PATH.read_text())
else:
    hist = [{"episode": e["episode"], "loss": e["loss"]} for e in history]

if hist:
    episodes = [h["episode"] for h in hist]
    losses   = [h["loss"]    for h in hist]

    # Smoothed moving average
    window = max(1, len(losses) // 200)
    smooth = [sum(losses[max(0,i-window):i+1]) / min(i+1, window)
              for i in range(len(losses))]

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, losses,  alpha=0.2, color="steelblue", linewidth=0.5, label="raw")
    plt.plot(episodes, smooth,  color="steelblue", linewidth=1.5, label=f"smoothed (w={window})")
    plt.xlabel("Episode"); plt.ylabel("Loss")
    plt.title("AlphaZero Training Loss")
    plt.legend(); plt.tight_layout(); plt.show()
    print(f"Final episode: {episodes[-1]}  |  Smoothed loss: {smooth[-1]:.4f}")
else:
    print("No history data yet.")
''')

# ── Cell 7: Quick smoke test ───────────────────────────────────────────────────
md("## Smoke Test\n\nRun a tiny 10-episode training to verify the whole pipeline works before committing to the full run.")
code('''\
smoke_config = AZConfig(k_rollouts=2, n_episodes=10, temperature=1.0)
smoke_trainer = build_trainer(smoke_config)

smoke_losses = []
for i in range(10):
    loss = smoke_trainer.train_one_episode()
    smoke_losses.append(loss)
    print(f"  ep {i+1:2d}  loss={loss:.4f}")

print(f"\\nSmoke test passed. Mean loss: {sum(smoke_losses)/len(smoke_losses):.4f}")
''')


# ── Assemble notebook ──────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out_path = Path(__file__).parent / "AlphaZero_Leduc.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook written → {out_path}")
