"""
Optimised PIMC search — drop-in replacement for rollout.pimc_search.

Identical semantics, faster execution via four targeted optimisations:

  1. RolloutGame.pack() / restore()
       Replaces LeducGame.deepcopy() (15,570 calls, 13% of runtime) with
       a plain-tuple snapshot that restores via field assignment.

  2. Shallow BeliefState copy
       BeliefState.copy() clones every tensor with detach().clone().
       Since all belief updates rebind tensors (never mutate in-place),
       sharing tensor references between original and copy is safe.
       Only the list containers need to be new objects.
       Saves ~12% of runtime and eliminates 15,570 tensor clone operations.

  3. Cached hand one-hot vectors
       hand_onehot() creates a new 3-element tensor on every call (17K times).
       Pre-built module-level constants replace all allocations.

  4. Batched BeliefNet at deal events
       A deal event updates both b_mine and b_opp_j with identical (P_t, e).
       Batching them into a single mlp(x_batch) forward halves BeliefNet
       calls at flop-transition steps.

API:
    from preliminary_experiments.alphazero.fast_rollout import fast_pimc_search
    # Same signature as pimc_search() in rollout.py
"""

import torch
import torch.nn.functional as F

from .state_encoder import StateEncoder, CARD_TO_IDX, IDX_TO_CARD
from .belief import BeliefNet, BeliefState
from .agent import QNet
from .fast_game import RolloutGame
from engine.leduc_game import Action


# ── Module-level constants ────────────────────────────────────────────────────

# Pre-built hand one-hot vectors (J=0, Q=1, K=2).
_H_ONEHOT: dict[str, torch.Tensor] = {
    h: torch.zeros(3).scatter_(0, torch.tensor(CARD_TO_IDX[h]), 1.0)
    for h in ('J', 'Q', 'K')
}

# Legal-action masks for Q-net softmax.
# Added to q_vals before softmax; -inf blocks illegal actions.
# (created once; never modified in-place — q_vals + mask uses out-of-place +)
_MASK_ALL   = torch.zeros(3)                               # fold / call / raise
_MASK_NO_R  = torch.tensor([0., 0., float('-inf')])        # fold / call only


# ── Shallow BeliefState copy ──────────────────────────────────────────────────

def _bs_copy(bs: BeliefState) -> BeliefState:
    """
    Fast snapshot of a BeliefState for rollout use.

    Safety argument: every update to BeliefState REBINDS a variable
    (bs.b_mine = new_tensor, bs.b_opp[k] = new_tensor, etc.) rather than
    mutating a tensor in-place.  Therefore:
      - Sharing the same tensor objects between original and copy is safe;
        the copy's rebinding never affects the original's tensor.
      - Only the list containers (b_opp, P_history) must be new objects so
        that list-level mutations (index assignment, append) don't cross over.
    """
    return BeliefState(
        b_mine=bs.b_mine,
        b_opp=list(bs.b_opp),
        P_current=bs.P_current,
        P_history=list(bs.P_history),
    )


# ── Action sampling ───────────────────────────────────────────────────────────

def _sample(
    raises: int,
    h: str,
    b: torch.Tensor,
    P: torch.Tensor,
    q_net: QNet,
    T: float,
) -> Action:
    """Q-net forward + masked softmax action sampling."""
    q = q_net(P, _H_ONEHOT[h], b)
    mask = _MASK_ALL if raises < 2 else _MASK_NO_R
    return Action(torch.multinomial(F.softmax((q + mask) / T, dim=-1), 1).item())


# ── Single rollout ────────────────────────────────────────────────────────────

def _rollout(
    game: RolloutGame,
    player_i: int,
    h_i: str,
    h_j: str,
    bs: BeliefState,           # shallow copy — safe to mutate
    b_opp_j: torch.Tensor,     # j's belief about i in this h_j world (will be rebound)
    first_action: Action,
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    q_net: QNet,
    T: float,
    T_self: float | None = None,   # temperature for player_i's actions (None → T)
    T_opp:  float | None = None,   # temperature for opponent j's actions (None → T)
) -> float:
    """
    One complete rollout from a decision point.
    Same semantics as _run_single_rollout() in rollout.py.

    game must be pre-loaded (via RolloutGame.restore) to the decision-point
    state BEFORE calling this function.
    """
    t_i = T_self if T_self is not None else T   # temperature when player_i acts
    t_j = T_opp  if T_opp  is not None else T   # temperature when opponent j acts

    # ── Step 1: player i takes first_action ──────────────────────────────
    _, done, act_eid, deal_eid, _ = game.step(first_action)

    if done:
        return game.get_reward()[player_i]

    # i acted → update b_opp_j (j's hypothetical belief about i)
    e_prime, P_new = state_enc.encode_event(act_eid, bs.P_current, bs.P_history)
    b_opp_j = belief_net(b_opp_j, bs.P_current, e_prime)
    bs.P_history.append(P_new)
    bs.P_current = P_new

    if deal_eid is not None:
        e_d, P_new_d = state_enc.encode_event(deal_eid, bs.P_current, bs.P_history)
        # Deal updates both b_mine and b_opp_j with the same (P, e) → batch them.
        b_stack = torch.stack([bs.b_mine, b_opp_j])                  # (2, 3)
        P_exp   = bs.P_current.unsqueeze(0).expand(2, -1)
        e_exp   = e_d.unsqueeze(0).expand(2, -1)
        out = F.softmax(
            belief_net.mlp(torch.cat([b_stack, P_exp, e_exp], dim=-1)), dim=-1
        )                                                             # (2, 3)
        bs.b_mine, b_opp_j = out[0], out[1]
        bs.P_history.append(P_new_d)
        bs.P_current = P_new_d

    # ── Remaining rollout ─────────────────────────────────────────────────
    while not game.is_finished:
        actor = game.current_player

        if actor == player_i:
            action = _sample(game.raises_this_round, h_i, bs.b_mine,  bs.P_current, q_net, t_i)
        else:
            action = _sample(game.raises_this_round, h_j, b_opp_j,    bs.P_current, q_net, t_j)

        _, done, act_eid, deal_eid, _ = game.step(action)

        if done:
            return game.get_reward()[player_i]

        e_prime, P_new = state_enc.encode_event(act_eid, bs.P_current, bs.P_history)

        if actor != player_i:
            # Opponent acted → update my belief about them
            bs.b_mine = belief_net(bs.b_mine, bs.P_current, e_prime)
        else:
            # I acted → update j's hypothetical belief about me
            b_opp_j = belief_net(b_opp_j, bs.P_current, e_prime)

        bs.P_history.append(P_new)
        bs.P_current = P_new

        if deal_eid is not None:
            e_d, P_new_d = state_enc.encode_event(deal_eid, bs.P_current, bs.P_history)
            # Batch deal updates
            b_stack = torch.stack([bs.b_mine, b_opp_j])
            P_exp   = bs.P_current.unsqueeze(0).expand(2, -1)
            e_exp   = e_d.unsqueeze(0).expand(2, -1)
            out = F.softmax(
                belief_net.mlp(torch.cat([b_stack, P_exp, e_exp], dim=-1)), dim=-1
            )
            bs.b_mine, b_opp_j = out[0], out[1]
            bs.P_history.append(P_new_d)
            bs.P_current = P_new_d

    return game.get_reward()[player_i]


# ── PIMC search ───────────────────────────────────────────────────────────────

def fast_pimc_search(
    obs,
    player_i: int,
    h_i: str,
    bs_i: BeliefState,
    legal_actions: list,
    state_enc: StateEncoder,
    belief_net: BeliefNet,
    q_net: QNet,
    k: int = 10,
    T: float = 1.0,
    T_self: float | None = None,   # None → falls back to T
    T_opp:  float | None = None,   # None → falls back to T
) -> torch.Tensor:
    """
    PIMC search — fast drop-in replacement for pimc_search() in rollout.py.

    Q*(a) = Σ_{h_j} b_{i→j}(h_j) · mean_k[ payoff(a, h_j) ]

    Returns shape-(3,) tensor; illegal actions set to -inf.
    """
    q_star = torch.full((3,), float('-inf'))
    rg = RolloutGame()

    with torch.no_grad():
        for hj_idx, h_j in enumerate(IDX_TO_CARD):
            weight = bs_i.b_mine[hj_idx].item()
            if weight < 1e-8:
                continue

            b_opp_j_base = bs_i.b_opp[hj_idx]

            # Initialise game once per imagined opponent hand.
            rg.init_from_obs(obs, player_i, h_i, h_j)
            init_packed = rg.pack()   # snapshot of decision-point state

            for action in legal_actions:
                aid = int(action)
                total  = 0.0

                for _ in range(k):
                    rg.restore(init_packed)   # O(1) in-place reset to decision point
                    total += _rollout(
                        game=rg,
                        player_i=player_i,
                        h_i=h_i,
                        h_j=h_j,
                        bs=_bs_copy(bs_i),
                        b_opp_j=b_opp_j_base,   # rebound inside, not mutated
                        first_action=action,
                        state_enc=state_enc,
                        belief_net=belief_net,
                        q_net=q_net,
                        T=T,
                        T_self=T_self,
                        T_opp=T_opp,
                    )

                avg = total / k
                if q_star[aid] == float('-inf'):
                    q_star[aid] = 0.0
                q_star[aid] += weight * avg

    return q_star
