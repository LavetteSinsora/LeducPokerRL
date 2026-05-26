"""
PIMC search and Monte Carlo rollout for the AlphaZero-style agent.

Q*(a | P_t, h_i, b_{i→j}) = Σ_{h_j} b_{i→j}(h_j) · E[return | a, h_j assumed]

For each imagined opponent hand h_j:
  For each legal action a:
    Run k independent rollouts to terminal, using Q_θ to sample actions for
    both players.  Average the returns → Q_rollout(a | h_j).
Aggregate: Q*(a) = Σ_{h_j} b_{i→j}(h_j) · Q_rollout(a | h_j)

Terminal payoff propagates directly back as the rollout return — no value head,
no discounting (single-episode game).

Community card deals are detected by comparing board state before/after each
step() call and injected as DEAL events into the belief/state update pipeline.
"""

import torch
import torch.nn.functional as F

from engine.leduc_game import LeducGame, Action
from engine.observation import Observation

from .state_encoder import (
    StateEncoder, CARD_TO_IDX, IDX_TO_CARD,
    action_event_id, deal_event_id,
)
from .belief import BeliefNet, BeliefState, update_belief_state
from .agent import QNet, hand_onehot


# ── Low-level helpers ────────────────────────────────────────────────────────

def _step_game(game: LeducGame, action: Action):
    """
    Execute action and return all relevant information.

    Returns:
        reward_list : [reward_p0, reward_p1]
        done        : bool
        act_eid     : event ID for the player action
        deal_eid    : event ID for the community card deal, or None
        actor       : player who took the action
    """
    actor = game.current_player
    pre_board = game.board

    _, reward_list, done, _ = game.step(action)

    act_eid = action_event_id(actor, int(action))
    deal_eid = None
    if game.board is not None and game.board != pre_board:
        deal_eid = deal_event_id(game.board)

    return reward_list, done, act_eid, deal_eid, actor


def _advance_belief(
    bs: BeliefState,
    actor: int | None,
    player_i: int,
    act_eid: int,
    deal_eid: int | None,
    state_enc: StateEncoder,
    belief_net: BeliefNet,
) -> None:
    """
    Apply one action event (and optional deal) to a BeliefState in-place.
    All operations run under the caller's grad context (no_grad in rollout).
    """
    e_prime, P_new = state_enc.encode_event(act_eid, bs.P_current, bs.P_history)
    update_belief_state(bs, actor, player_i, e_prime, P_new, belief_net)

    if deal_eid is not None:
        e_prime_d, P_new_d = state_enc.encode_event(deal_eid, bs.P_current, bs.P_history)
        update_belief_state(bs, None, player_i, e_prime_d, P_new_d, belief_net)


def _sample_action(
    game: LeducGame,
    actor: int,
    h_actor: str,
    b_actor: torch.Tensor,  # actor's belief about their opponent
    P_current: torch.Tensor,
    q_net: QNet,
    T: float,
) -> Action:
    """Sample an action from softmax(Q / T), masked to legal actions."""
    legal = game.get_legal_actions()
    h_oh = hand_onehot(h_actor)
    q_vals = q_net(P_current, h_oh, b_actor)

    mask = torch.full((3,), float('-inf'), device=q_vals.device)
    for a in legal:
        mask[int(a)] = 0.0
    probs = F.softmax((q_vals + mask) / T, dim=-1)
    idx = torch.multinomial(probs, 1).item()
    return Action(idx)


# ── Single rollout ───────────────────────────────────────────────────────────

def _run_single_rollout(
    game: LeducGame,        # deep copy at decision point, h_j already set
    player_i: int,
    h_i: str,
    h_j: str,               # imagined opponent hand
    bs_i: BeliefState,      # copy of player i's belief state
    b_opp_j: torch.Tensor,  # copy of j's belief about i in this h_j world
    first_action: Action,   # outer action being evaluated
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    q_net: QNet,
    T: float,
) -> float:
    """
    Run one rollout from decision point s_d.
    Player i takes first_action; then both players sample from Q until terminal.
    Returns payoff for player_i.
    """
    # ── Step 1: player i takes the outer action ───────────────────────────
    reward_list, done, act_eid, deal_eid, _ = _step_game(game, first_action)

    if done:
        return reward_list[player_i]

    # i acted: update j's hypothetical belief about i (b_opp_j)
    # and update bs_i.P (which is shared public info)
    e_prime, P_new = state_enc.encode_event(act_eid, bs_i.P_current, bs_i.P_history)
    # i acted → b_opp_j updates (j observes i's action)
    b_opp_j = belief_net(b_opp_j, bs_i.P_current, e_prime)
    bs_i.P_history.append(P_new)
    bs_i.P_current = P_new

    if deal_eid is not None:
        e_prime_d, P_new_d = state_enc.encode_event(deal_eid, bs_i.P_current, bs_i.P_history)
        bs_i.b_mine = belief_net(bs_i.b_mine, bs_i.P_current, e_prime_d)
        b_opp_j     = belief_net(b_opp_j,     bs_i.P_current, e_prime_d)
        bs_i.P_history.append(P_new_d)
        bs_i.P_current = P_new_d

    # ── Remaining rollout ─────────────────────────────────────────────────
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
            # j acted: i updates their belief about j
            bs_i.b_mine = belief_net(bs_i.b_mine, bs_i.P_current, e_prime)
        else:
            # i acted: j's hypothetical belief updates
            b_opp_j = belief_net(b_opp_j, bs_i.P_current, e_prime)

        bs_i.P_history.append(P_new)
        bs_i.P_current = P_new

        if deal_eid is not None:
            e_prime_d, P_new_d = state_enc.encode_event(deal_eid, bs_i.P_current, bs_i.P_history)
            bs_i.b_mine = belief_net(bs_i.b_mine, bs_i.P_current, e_prime_d)
            b_opp_j     = belief_net(b_opp_j,     bs_i.P_current, e_prime_d)
            bs_i.P_history.append(P_new_d)
            bs_i.P_current = P_new_d

    return game.get_reward()[player_i]


# ── PIMC search ──────────────────────────────────────────────────────────────

def pimc_search(
    obs: Observation,
    player_i: int,
    h_i: str,
    bs_i: BeliefState,      # LIVE belief state (NOT modified)
    legal_actions: list,
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    q_net: QNet,
    k: int = 10,
    T: float = 1.0,
) -> torch.Tensor:
    """
    PIMC search at decision point described by obs.

    Returns Q*(a) for all three actions, shape (3,).
    Illegal actions are set to -inf so they are never selected.

    Q*(a) = Σ_{h_j} b_{i→j}(h_j) · mean_{rollouts}[ payoff(a, h_j) ]
    """
    legal_ids = {int(a) for a in legal_actions}
    q_star = torch.full((3,), float('-inf'))

    with torch.no_grad():
        # Reconstruct the game state at the decision point using obs.
        # We use a fresh LeducGame and set it from the observation.
        game_template = LeducGame()
        game_template.set_state(obs)
        # Ensure our player's hand is set correctly (set_state only sets current_player's hand)
        game_template.player_hands[player_i] = h_i

        for hj_idx, h_j in enumerate(IDX_TO_CARD):
            weight = bs_i.b_mine[hj_idx].item()
            if weight < 1e-8:
                continue

            # b_opp_j: j's belief about i given j holds h_j (from i's portfolio)
            b_opp_j_base = bs_i.b_opp[hj_idx]

            # Accumulate returns per action
            action_returns = {aid: [] for aid in legal_ids}

            for action in legal_actions:
                aid = int(action)
                for _ in range(k):
                    # Fresh game copy with h_j set
                    game_copy = game_template.copy()
                    game_copy.player_hands[1 - player_i] = h_j

                    ret = _run_single_rollout(
                        game=game_copy,
                        player_i=player_i,
                        h_i=h_i,
                        h_j=h_j,
                        bs_i=bs_i.copy(),
                        b_opp_j=b_opp_j_base.clone(),
                        first_action=action,
                        state_enc=state_enc,
                        belief_net=belief_net,
                        q_net=q_net,
                        T=T,
                    )
                    action_returns[aid].append(ret)

            for aid, returns in action_returns.items():
                avg = sum(returns) / len(returns)
                if q_star[aid] == float('-inf'):
                    q_star[aid] = 0.0
                q_star[aid] += weight * avg

    return q_star
