"""Audience-facing options in app_config.json."""

import pytest

from sludgescout.config import AppConfig, load_config


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
