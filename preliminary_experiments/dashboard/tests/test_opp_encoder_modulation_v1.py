import math

from engine.poker_session import PokerSession
from experiments.opp_encoder_modulation_v1.agent import OpponentEncoderModulationAgent
from experiments.opp_encoder_modulation_v1.trainer import OpponentEncoderModulationTrainer


def test_experiment_agent_shapes():
    agent = OpponentEncoderModulationAgent()
    session = PokerSession()
    session.new_hand()
    obs = session.get_observation(viewer_id=0)

    state = agent.encode_observation(obs, viewer_id=0)
    stats = agent.encode_macro_stats(obs)
    value = agent.predict_value_from_encoded(state, stats)
    logits = agent.predict_action_logits(state, stats)

    assert tuple(state.shape) == (1, 15)
    assert tuple(stats.shape) == (4,)
    assert tuple(value.shape) == (1, 1)
    assert tuple(logits.shape) == (1, 3)


def test_experiment_trainer_collects_and_updates():
    agent = OpponentEncoderModulationAgent()
    trainer = OpponentEncoderModulationTrainer(
        agent,
        hands_per_session=2,
        rotate_every=10,
        snapshot_every=1000,
        opponent_ids=("heuristic",),
    )

    session_data = trainer.collect_episode()

    assert len(session_data) == 2
    assert "value_chain" in session_data[0]
    assert "action_examples" in session_data[0]

    loss = trainer.update_model(session_data)

    assert math.isfinite(loss)
