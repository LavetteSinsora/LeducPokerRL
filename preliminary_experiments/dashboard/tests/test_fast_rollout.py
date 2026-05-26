"""
Tests for preliminary_experiments.alphazero/fast_game.py and preliminary_experiments.alphazero/fast_rollout.py.

Three layers of correctness coverage:

  1. RolloutGame logic — game mechanics match LeducGame exactly for all
     action sequences (terminal rewards, pot accounting, win conditions).

  2. Infrastructure correctness — BeliefState shallow copy isolation;
     pack/restore round-trip; masked softmax equivalence.

  3. fast_pimc_search statistical consistency — Q* distribution over many
     calls matches the reference pimc_search within sampling tolerance.

Run with:  pytest tests/test_fast_rollout.py -v
"""

import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.leduc_game import LeducGame, Action
from engine.observation import Observation
from preliminary_experiments.alphazero.fast_game import RolloutGame
from preliminary_experiments.alphazero.fast_rollout import _bs_copy, fast_pimc_search, _MASK_ALL, _MASK_NO_R, _H_ONEHOT
from preliminary_experiments.alphazero.belief import BeliefState, make_belief_state, BeliefNet
from preliminary_experiments.alphazero.config import AZConfig
from preliminary_experiments.alphazero.state_encoder import StateEncoder
from preliminary_experiments.alphazero.agent import QNet, hand_onehot
from preliminary_experiments.alphazero.rollout import pimc_search


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_V3_CONFIG = AZConfig(
    d_model=4, state_hidden=(8,), belief_hidden=(8, 8), q_hidden=(32, 32),
    k_rollouts=20, temperature=1.0,
)


def _fresh_nets(seed: int = 0):
    torch.manual_seed(seed)
    se = StateEncoder(_V3_CONFIG)
    bn = BeliefNet(_V3_CONFIG)
    qn = QNet(_V3_CONFIG)
    se.eval(); bn.eval(); qn.eval()
    return se, bn, qn


def _make_obs(h0, h1, round_=0, pot=None, raises=0, board=None, current_player=0):
    """Construct a minimal Observation for a mid-game state."""
    if pot is None:
        pot = [1, 1]
    legal = [Action.FOLD, Action.CALL]
    if raises < 2:
        legal.append(Action.RAISE)
    hand = h0 if current_player == 0 else h1
    return Observation(
        player_hand=hand, board=board, pot=pot,
        current_player=current_player, current_round=round_,
        legal_actions=legal, is_finished=False, raises_this_round=raises,
    )


def _play_sequence_leduc(h0, h1, actions):
    """
    Play a fixed action sequence on a fresh LeducGame with forced hands.
    Returns (final_pot, reward, board).
    """
    game = LeducGame()
    game.reset()
    game.player_hands = [h0, h1]
    # Remove board card from deck to avoid issues (only needed if round 1 reached)
    for a in actions:
        if game.is_finished:
            break
        game.step(Action(a))
    return list(game.pot), game.get_reward(), game.board


def _play_sequence_rollout(h0, h1, actions, pending_board='Q'):
    """
    Play a fixed action sequence on a RolloutGame with forced hands.
    pending_board controls which board card is dealt at flop.
    Returns (final_pot, reward, board).
    """
    obs = _make_obs(h0, h1, round_=0, pot=[1, 1], raises=0)
    rg = RolloutGame()
    rg.init_from_obs(obs, player_i=0, h_i=h0, h_j=h1)
    rg._pending_board = pending_board  # fix board for deterministic test
    for a in actions:
        if rg.is_finished:
            break
        rg.step(Action(a))
    return [rg.pot0, rg.pot1], rg.get_reward(), rg.board


# ─────────────────────────────────────────────────────────────────────────────
# 1. RolloutGame — game logic correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestRolloutGameLogic:

    def test_immediate_fold_p0(self):
        """P0 folds on first action: P1 wins P0's ante."""
        pot, reward, board = _play_sequence_rollout('J', 'K', [Action.FOLD])
        assert reward == [-1, 1], f"Expected [-1, 1] got {reward}"
        assert board is None

    def test_immediate_fold_p1(self):
        """P0 checks, P1 folds: P0 wins P1's ante."""
        pot, reward, board = _play_sequence_rollout('J', 'K', [Action.CALL, Action.FOLD])
        assert reward == [1, -1], f"Expected [1, -1] got {reward}"

    def test_raise_then_fold(self):
        """P0 raises (pot=[3,1]), P1 folds: P0 wins 1 chip."""
        pot, reward, board = _play_sequence_rollout('K', 'J', [Action.RAISE, Action.FOLD])
        # P0 raised: pot = [1+2, 1] = [3, 1]; P1 folds → P1 loses pot1=1
        assert pot == [3, 1]
        assert reward == [1, -1], f"Expected [1, -1] got {reward}"

    def test_call_raise_call_reaches_flop(self):
        """P0 checks, P1 raises, P0 calls → round ends → flop dealt."""
        pot, reward, board = _play_sequence_rollout(
            'K', 'J',
            [Action.CALL, Action.RAISE, Action.CALL],
            pending_board='Q',
        )
        assert board == 'Q'
        assert pot == [3, 3], f"Expected pot [3,3] got {pot}"

    def test_max_raises_no_more_raise(self):
        """After 2 raises, only fold/call are legal."""
        rg = RolloutGame()
        obs = _make_obs('K', 'J', raises=0)
        rg.init_from_obs(obs, player_i=0, h_i='K', h_j='J')
        # P0 raises, P1 raises
        rg.step(Action.RAISE)
        rg.step(Action.RAISE)
        assert rg.raises_this_round == 2
        legal = rg.get_legal_actions()
        assert Action.RAISE not in legal

    def test_showdown_k_beats_j(self):
        """Full game to showdown: K beats J high card."""
        actions = [
            Action.CALL, Action.CALL,        # preflop: both check → flop
            Action.CALL, Action.CALL,        # postflop: both check → showdown
        ]
        pot, reward, board = _play_sequence_rollout('K', 'J', actions, pending_board='Q')
        assert pot == [1, 1]
        # K > J, P0 wins → [1, -1]
        assert reward == [1, -1], f"Expected [1,-1] got {reward} (board={board})"

    def test_showdown_pair_beats_high_card(self):
        """P0 holds K, board=K: pair wins over J."""
        actions = [Action.CALL, Action.CALL, Action.CALL, Action.CALL]
        pot, reward, board = _play_sequence_rollout('K', 'J', actions, pending_board='K')
        assert board == 'K'
        # P0 has pair (K+K), P1 has K high → P0 wins
        assert reward == [1, -1], f"{reward}"

    def test_showdown_tie(self):
        """Both players hold same rank, no pair → tie."""
        actions = [Action.CALL, Action.CALL, Action.CALL, Action.CALL]
        pot, reward, board = _play_sequence_rollout('Q', 'Q', actions, pending_board='J')
        assert board == 'J'
        assert reward == [0, 0], f"{reward}"

    @pytest.mark.parametrize("h0,h1,actions", [
        ('J', 'K', [Action.FOLD]),
        ('Q', 'J', [Action.RAISE, Action.FOLD]),
        ('K', 'Q', [Action.CALL, Action.RAISE, Action.FOLD]),
        ('J', 'Q', [Action.CALL, Action.CALL, Action.FOLD]),
        ('K', 'J', [Action.RAISE, Action.RAISE, Action.CALL, Action.CALL, Action.CALL]),
    ])
    def test_reward_matches_leduc_game(self, h0, h1, actions):
        """
        RolloutGame final rewards exactly match LeducGame for the same action sequence.
        Games that reach the flop use a fixed pending_board='Q' for both engines.
        """
        # RolloutGame
        pot_r, rew_r, board_r = _play_sequence_rollout(h0, h1, actions, pending_board='Q')

        # LeducGame — force same hands and fix the deck so board='Q' if flop reached
        game = LeducGame()
        game.reset()
        game.player_hands = [h0, h1]
        # Force board='Q': deck.pop() draws from the end, so put only 'Q' there
        game.deck = ['Q']
        for a in actions:
            if game.is_finished:
                break
            game.step(a)

        assert rew_r == game.get_reward(), (
            f"h0={h0} h1={h1} actions={[a.name for a in actions]}: "
            f"RolloutGame={rew_r}  LeducGame={game.get_reward()}"
        )

    def test_pack_restore_round_trip(self):
        """pack() / restore() preserves state exactly."""
        rg = RolloutGame()
        obs = _make_obs('K', 'J')
        rg.init_from_obs(obs, player_i=0, h_i='K', h_j='J')
        rg._pending_board = 'Q'

        state_before = rg.pack()
        # Mutate by playing an action
        rg.step(Action.RAISE)
        assert rg.pot0 != 1 or rg.pot1 != 1  # state changed

        # Restore
        rg.restore(state_before)
        assert rg.pack() == state_before

    def test_restore_identical_to_fresh_init(self):
        """After restore, game plays out identically to a fresh init."""
        rg = RolloutGame()
        obs = _make_obs('Q', 'K')
        rg.init_from_obs(obs, player_i=0, h_i='Q', h_j='K')
        rg._pending_board = 'J'
        packed = rg.pack()

        results = []
        for _ in range(3):
            rg.restore(packed)
            for a in [Action.RAISE, Action.CALL, Action.CALL, Action.CALL]:
                if rg.is_finished: break
                rg.step(a)
            results.append(rg.get_reward())

        # All three runs started from same state → same result
        assert all(r == results[0] for r in results), results


# ─────────────────────────────────────────────────────────────────────────────
# 2. _bs_copy — shallow copy isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestBeliefStateCopy:

    def _apply_fake_updates(self, bs, n=5):
        """Rebind belief tensors to new values (simulates post-copy rollout updates)."""
        for _ in range(n):
            bs.b_mine = torch.rand(3)
            for k in range(3):
                bs.b_opp[k] = torch.rand(3)
            bs.P_current = torch.rand(4)
            bs.P_history.append(torch.rand(4))

    def test_original_b_mine_unchanged_after_copy_mutated(self):
        bs = make_belief_state('J', d_model=4)
        original_b_mine = bs.b_mine.clone()
        copy = _bs_copy(bs)
        copy.b_mine = torch.rand(3)   # rebind copy's b_mine
        assert torch.allclose(bs.b_mine, original_b_mine), \
            "Original b_mine was affected by copy's rebind"

    def test_original_b_opp_unchanged_after_copy_mutated(self):
        bs = make_belief_state('Q', d_model=4)
        original_b_opps = [b.clone() for b in bs.b_opp]
        copy = _bs_copy(bs)
        for k in range(3):
            copy.b_opp[k] = torch.rand(3)
        for k in range(3):
            assert torch.allclose(bs.b_opp[k], original_b_opps[k]), \
                f"Original b_opp[{k}] was affected by copy's rebind"

    def test_original_p_history_unchanged_after_copy_append(self):
        bs = make_belief_state('K', d_model=4)
        n_before = len(bs.P_history)
        copy = _bs_copy(bs)
        copy.P_history.append(torch.rand(4))
        copy.P_history.append(torch.rand(4))
        assert len(bs.P_history) == n_before, \
            "Original P_history grew when copy was appended to"

    def test_copy_produces_same_initial_values(self):
        bs = make_belief_state('J', d_model=4)
        copy = _bs_copy(bs)
        assert torch.allclose(copy.b_mine, bs.b_mine)
        for k in range(3):
            assert torch.allclose(copy.b_opp[k], bs.b_opp[k])
        assert torch.allclose(copy.P_current, bs.P_current)
        assert len(copy.P_history) == len(bs.P_history)

    def test_full_rollout_updates_do_not_corrupt_original(self):
        """Run _apply_fake_updates on copy; original must be entirely unaffected."""
        bs = make_belief_state('Q', d_model=4)
        orig_b_mine   = bs.b_mine.clone()
        orig_b_opps   = [b.clone() for b in bs.b_opp]
        orig_P_current = bs.P_current.clone()
        orig_hist_len  = len(bs.P_history)

        copy = _bs_copy(bs)
        self._apply_fake_updates(copy, n=10)

        assert torch.allclose(bs.b_mine, orig_b_mine)
        for k in range(3):
            assert torch.allclose(bs.b_opp[k], orig_b_opps[k])
        assert torch.allclose(bs.P_current, orig_P_current)
        assert len(bs.P_history) == orig_hist_len


# ─────────────────────────────────────────────────────────────────────────────
# 3. Constants and masks
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:

    def test_hand_onehot_correct(self):
        for h, idx in [('J', 0), ('Q', 1), ('K', 2)]:
            v = _H_ONEHOT[h]
            assert v[idx].item() == 1.0
            assert v.sum().item() == 1.0

    def test_hand_onehot_matches_reference(self):
        for h in ('J', 'Q', 'K'):
            assert torch.allclose(_H_ONEHOT[h], hand_onehot(h)), \
                f"Cached onehot for {h} differs from hand_onehot()"

    def test_masks_correct_shape(self):
        assert _MASK_ALL.shape == (3,)
        assert _MASK_NO_R.shape == (3,)

    def test_mask_all_allows_all_actions(self):
        q = torch.tensor([1.0, 2.0, 3.0])
        probs = torch.softmax(q + _MASK_ALL, dim=-1)
        assert (probs > 0).all()

    def test_mask_no_raise_blocks_raise(self):
        q = torch.tensor([1.0, 2.0, 3.0])
        probs = torch.softmax(q + _MASK_NO_R, dim=-1)
        assert probs[2].item() == 0.0
        assert probs[0].item() > 0
        assert probs[1].item() > 0

    def test_masks_are_not_mutated_by_addition(self):
        """Adding to masks must not modify the module-level tensors."""
        mask_all_before  = _MASK_ALL.clone()
        mask_no_r_before = _MASK_NO_R.clone()
        q = torch.tensor([1.0, 2.0, 3.0])
        _ = q + _MASK_ALL    # out-of-place
        _ = q + _MASK_NO_R
        assert torch.allclose(_MASK_ALL,  mask_all_before)
        assert torch.allclose(_MASK_NO_R, mask_no_r_before)


# ─────────────────────────────────────────────────────────────────────────────
# 4. fast_pimc_search output properties
# ─────────────────────────────────────────────────────────────────────────────

class TestFastPimcSearch:

    @pytest.fixture(scope='class')
    def nets(self):
        return _fresh_nets(seed=42)

    def test_output_shape(self, nets):
        se, bn, qn = nets
        obs = _make_obs('K', 'J')
        bs = make_belief_state('K', d_model=4)
        q = fast_pimc_search(obs, 0, 'K', bs, obs.legal_actions, se, bn, qn, k=5)
        assert q.shape == (3,)

    def test_illegal_actions_are_neg_inf(self, nets):
        se, bn, qn = nets
        # State where raise is illegal (raises_this_round=2)
        obs = _make_obs('K', 'J', raises=2)
        bs = make_belief_state('K', d_model=4)
        q = fast_pimc_search(obs, 0, 'K', bs, obs.legal_actions, se, bn, qn, k=5)
        assert q[2].item() == float('-inf'), "Raise should be -inf when raises=2"

    def test_legal_actions_are_finite(self, nets):
        se, bn, qn = nets
        obs = _make_obs('K', 'J')
        bs = make_belief_state('K', d_model=4)
        q = fast_pimc_search(obs, 0, 'K', bs, obs.legal_actions, se, bn, qn, k=5)
        legal_ids = [int(a) for a in obs.legal_actions]
        for aid in legal_ids:
            assert torch.isfinite(q[aid]), f"Q[{aid}] should be finite"

    def test_q_star_bounded_by_game_range(self, nets):
        """Q* values must lie within Leduc's terminal reward range ±13."""
        se, bn, qn = nets
        random.seed(0); torch.manual_seed(0)
        for hand in ('J', 'Q', 'K'):
            obs = _make_obs(hand, 'Q')
            bs = make_belief_state(hand, d_model=4)
            q = fast_pimc_search(obs, 0, hand, bs, obs.legal_actions, se, bn, qn, k=10)
            for aid in [int(a) for a in obs.legal_actions]:
                val = q[aid].item()
                assert -14 <= val <= 14, f"Q*[{aid}]={val:.2f} out of game range"

    def test_player1_search_works(self, nets):
        """fast_pimc_search works correctly when player_i=1."""
        se, bn, qn = nets
        obs = _make_obs('Q', 'K', current_player=1)
        bs = make_belief_state('K', d_model=4)
        q = fast_pimc_search(obs, 1, 'K', bs, obs.legal_actions, se, bn, qn, k=5)
        assert q.shape == (3,)
        assert torch.isfinite(q[0]) and torch.isfinite(q[1])

    def test_postflop_search_works(self, nets):
        """fast_pimc_search works correctly in round 1 (flop state)."""
        se, bn, qn = nets
        obs = _make_obs('K', 'J', round_=1, pot=[3, 3], board='Q')
        bs = make_belief_state('K', d_model=4)
        q = fast_pimc_search(obs, 0, 'K', bs, obs.legal_actions, se, bn, qn, k=5)
        assert torch.isfinite(q[0]) and torch.isfinite(q[1])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Statistical consistency with reference pimc_search
# ─────────────────────────────────────────────────────────────────────────────

class TestStatisticalConsistency:

    def _collect_q_star(self, search_fn, n_trials, config, se, bn, qn, seed):
        """Run search_fn n_trials times and collect mean Q* per legal action."""
        random.seed(seed); torch.manual_seed(seed)
        results = {0: [], 1: [], 2: []}
        for _ in range(n_trials):
            hand = random.choice(['J', 'Q', 'K'])
            obs = _make_obs(hand, random.choice(['J', 'Q', 'K']))
            bs = make_belief_state(hand, d_model=config.d_model)
            q = search_fn(obs, 0, hand, bs, obs.legal_actions, se, bn, qn, k=10)
            for aid in [int(a) for a in obs.legal_actions]:
                results[aid].append(q[aid].item())
        return {aid: sum(v)/len(v) for aid, v in results.items() if v}

    def test_mean_q_star_within_tolerance(self):
        """
        Mean Q* over 200 trials must agree between fast and reference
        to within 1.0 chip (sampling noise at k=10 is high; this checks
        systematic bias, not variance).
        """
        se, bn, qn = _fresh_nets(seed=7)
        config = _V3_CONFIG

        mean_ref  = self._collect_q_star(pimc_search,      200, config, se, bn, qn, seed=0)
        mean_fast = self._collect_q_star(fast_pimc_search, 200, config, se, bn, qn, seed=0)

        for aid in [0, 1, 2]:
            if aid in mean_ref and aid in mean_fast:
                diff = abs(mean_ref[aid] - mean_fast[aid])
                assert diff < 1.0, (
                    f"Action {aid}: reference mean={mean_ref[aid]:.3f}, "
                    f"fast mean={mean_fast[aid]:.3f}, diff={diff:.3f} > 1.0"
                )
