import pytest

from slurm_sweep.utils import ConfigValidator


def test_load_valid_config(valid_config_file):
    validator = ConfigValidator(config_path=str(valid_config_file))
    validator.load_config()
    assert "wandb" in validator.config
    assert "general" in validator.config


def test_load_invalid_config(invalid_config_file):
    validator = ConfigValidator(config_path=str(invalid_config_file))
    validator.load_config()
    with pytest.raises(ValueError, match="The `wandb` block must include `program`, `method`, and `parameters` keys."):
        validator.validate()


def test_missing_config_file():
    validator = ConfigValidator(config_path="non_existent_file.yaml")
    with pytest.raises(FileNotFoundError, match="Configuration file 'non_existent_file.yaml' not found."):
        validator.load_config()


def test_unexpected_blocks(unexpected_block_config_file):
    validator = ConfigValidator(config_path=str(unexpected_block_config_file))
    validator.load_config()
    with pytest.warns(UserWarning, match="The configuration file contains unexpected blocks: {'extra_block'}"):
        validator.validate()


def test_default_slurm_block(valid_config_file):
    validator = ConfigValidator(config_path=str(valid_config_file))
    validator.load_config()
    validator.validate()
    assert "slurm" in validator.config
    assert validator.config["slurm"] == {}


def test_mamba_and_pixi_exclusive(tmp_path):
    """Test that specifying both mamba_env and pixi_env raises ValueError."""
    pixi_toml = tmp_path / "pixi.toml"
    pixi_toml.write_text("[project]\nname = 'test'\n")

    config = {
        "wandb": {
            "program": "train.py",
            "method": "grid",
            "parameters": {"lr": {"values": [0.01, 0.1]}},
        },
        "general": {
            "entity": "test_entity",
            "project_name": "test_project",
            "mamba_env": "test_env",
            "pixi_env": str(pixi_toml),
        },
    }
    import yaml

    file_path = tmp_path / "both_envs_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config, f)

    validator = ConfigValidator(config_path=str(file_path))
    validator.load_config()
    with pytest.raises(ValueError, match="Cannot specify both `mamba_env` and `pixi_env`"):
        validator.validate()


def test_pixi_env_file_not_found(tmp_path):
    """Test that non-existent pixi_env path raises FileNotFoundError."""
    config = {
        "wandb": {
            "program": "train.py",
            "method": "grid",
            "parameters": {"lr": {"values": [0.01, 0.1]}},
        },
        "general": {
            "entity": "test_entity",
            "project_name": "test_project",
            "pixi_env": "/nonexistent/path/pixi.toml",
        },
    }
    import yaml

    file_path = tmp_path / "invalid_pixi_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config, f)

    validator = ConfigValidator(config_path=str(file_path))
    validator.load_config()
    with pytest.raises(FileNotFoundError, match="Pixi manifest file not found"):
        validator.validate()


def test_valid_pixi_env(tmp_path):
    """Test that valid pixi_env path passes validation."""
    pixi_toml = tmp_path / "pixi.toml"
    pixi_toml.write_text("[project]\nname = 'test'\n")

    config = {
        "wandb": {
            "program": "train.py",
            "method": "grid",
            "parameters": {"lr": {"values": [0.01, 0.1]}},
        },
        "general": {
            "entity": "test_entity",
            "project_name": "test_project",
            "pixi_env": str(pixi_toml),
        },
    }
    import yaml

    file_path = tmp_path / "valid_pixi_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config, f)

    validator = ConfigValidator(config_path=str(file_path))
    validator.load_config()
    validator.validate()  # Should not raise
