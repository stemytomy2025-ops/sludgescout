"""Audience-facing options in app_config.json."""

import pytest

import sludgescout.config as config_module
from sludgescout.config import CONFIG_NAME, AppConfig, find_root, load_config


def _app_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / CONFIG_NAME).write_text("{}", encoding="utf-8")
    return path


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """Simulate a non-editable install: the module's parents hold no app_config.json."""
    monkeypatch.setattr(config_module, "__file__", str(tmp_path / "site" / "pkg" / "config.py"))


def test_root_is_found_from_the_working_directory_when_installed(installed, tmp_path):
    app = _app_dir(tmp_path / "repo")
    assert find_root(env={}, cwd=app) == app


def test_app_root_env_var_wins(installed, tmp_path):
    chosen = _app_dir(tmp_path / "chosen")
    other = _app_dir(tmp_path / "other")
    assert find_root(env={"APP_ROOT": str(chosen)}, cwd=other) == chosen


def test_missing_config_is_a_clear_error(installed, tmp_path):
    with pytest.raises(FileNotFoundError, match="set APP_ROOT"):
        find_root(env={}, cwd=tmp_path)


def test_developer_tabs_are_shown_and_about_is_off_by_default(config):
    assert AppConfig.model_fields["show_developer_tabs"].default is True
    assert AppConfig.model_fields["about_file"].default is None
    assert config.model_copy(update={"about_file": None}).about_text() is None


def test_about_text_reads_the_file_relative_to_the_app_root(config, tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "about.md").write_text("# About\nWhat it does.", encoding="utf-8")
    cfg = config.model_copy(update={"about_file": "docs/about.md"})
    assert cfg.about_text(root=tmp_path).startswith("# About")


def test_missing_about_file_fails_loudly(config, tmp_path):
    cfg = config.model_copy(update={"about_file": "docs/nope.md"})
    with pytest.raises(FileNotFoundError, match="nope.md"):
        cfg.about_text(root=tmp_path)


def test_configured_about_file_exists():
    config = load_config()
    if config.about_file:
        assert config.about_text()
