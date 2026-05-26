from pathlib import Path

from dashboard.server import WEB_DIR
from dashboard.tournament import TournamentRunner


def test_dashboard_static_root_exists():
    assert Path(WEB_DIR).exists()
    assert (Path(WEB_DIR) / "index.html").exists()


def test_tournament_runner_initial_state():
    runner = TournamentRunner()
    status = runner.get_status()
    assert status["is_running"] is False
