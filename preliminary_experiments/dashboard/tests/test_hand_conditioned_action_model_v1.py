import json
import subprocess
import sys

from experiments.hand_conditioned_action_model_v1.agent import (
    HandConditionedActionModel,
    initialize_belief,
    update_belief_with_board,
)


def test_board_update_preserves_distribution():
    prior = initialize_belief("Q")
    posterior = update_belief_with_board(prior, viewer_hand="Q", board="K")
    assert abs(float(posterior.sum()) - 1.0) < 1e-6
    assert all(x >= 0 for x in posterior)


def test_hand_conditioned_action_model_smoke(tmp_path):
    output_dir = tmp_path / "hand_conditioned_action_model_v1"
    result = subprocess.run(
        [
            sys.executable,
            "experiments/hand_conditioned_action_model_v1/train.py",
            "--smoke",
            "--output-dir",
            str(output_dir),
        ],
        cwd="/Users/chrishe/Downloads/PokerRL_Vanilla",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    history = json.loads((output_dir / "train_history.json").read_text())
    assert history
    model = HandConditionedActionModel(model_path=str(output_dir / "checkpoint.pt"))
    model.set_train_mode(False)
