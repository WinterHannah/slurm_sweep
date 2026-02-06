from unittest.mock import patch

import pytest

from slurm_sweep.sweep_class import SweepManager


def test_register_sweep(sweep_config):
    manager = SweepManager()

    with patch("wandb.login") as mock_login, patch("wandb.sweep") as mock_sweep:
        mock_sweep.return_value = "test-sweep-id"

        manager.register_sweep(sweep_config, project_name="test_project", entity="test_entity")

        mock_login.assert_called_once()
        mock_sweep.assert_called_once_with(sweep=sweep_config, project="test_project", entity="test_entity")
        assert manager.sweep_id == "test-sweep-id"
        assert manager.project_name == "test_project"
        assert manager.entity == "test_entity"


def test_write_script(tmp_path, expected_slurm_script):
    manager = SweepManager()
    manager.sweep_id = "test-sweep-id"
    manager.project_name = "test_project"
    manager.entity = "test_entity"

    job_file = tmp_path / "submit.sh"

    manager.write_script(
        slurm_parameters={"time": "01:00:00", "partition": "test"},
        mamba_env="test_env",
        job_file=str(job_file),
        modules="test_module",
    )

    # Verify the script was written to the file
    assert job_file.exists()
    with open(job_file) as f:
        content = f.read()

    # Compare the generated script with the expected script
    assert content == expected_slurm_script


@pytest.mark.parametrize(
    "count, expected_command",
    [
        (None, 'wandb agent "test_entity/test_project/test-sweep-id"'),
        (1, 'wandb agent --count 1 "test_entity/test_project/test-sweep-id"'),
        (5, 'wandb agent --count 5 "test_entity/test_project/test-sweep-id"'),
    ],
)
def test_write_script_with_count(tmp_path, count, expected_command):
    manager = SweepManager()
    manager.sweep_id = "test-sweep-id"
    manager.project_name = "test_project"
    manager.entity = "test_entity"

    job_file = tmp_path / "submit_with_count.sh"

    manager.write_script(
        slurm_parameters={"time": "01:00:00", "partition": "test"},
        mamba_env="test_env",
        job_file=str(job_file),
        modules="test_module",
        count=count,
    )

    # Verify the script was written to the file
    assert job_file.exists()
    with open(job_file) as f:
        content = f.read()

    # Verify the expected command is in the script
    assert expected_command in content



def test_write_script_pixi(tmp_path, expected_slurm_script_pixi):
    """Test write_script with pixi environment."""
    expected_script, pixi_toml = expected_slurm_script_pixi

    manager = SweepManager()
    manager.sweep_id = "test-sweep-id"
    manager.project_name = "test_project"
    manager.entity = "test_entity"

    job_file = tmp_path / "submit.sh"

    manager.write_script(
        slurm_parameters={"time": "01:00:00", "partition": "test"},
        pixi_env=pixi_toml,
        job_file=str(job_file),
        modules="test_module",
    )

    assert job_file.exists()
    with open(job_file) as f:
        content = f.read()

    assert content == expected_script


def test_write_script_pixi_file_not_found(tmp_path):
    """Test write_script raises FileNotFoundError for non-existent pixi manifest."""
    manager = SweepManager()
    manager.sweep_id = "test-sweep-id"
    manager.project_name = "test_project"
    manager.entity = "test_entity"

    job_file = tmp_path / "submit.sh"

    with pytest.raises(FileNotFoundError, match="Pixi manifest file not found"):
        manager.write_script(
            slurm_parameters={"time": "01:00:00", "partition": "test"},
            pixi_env="/nonexistent/path/pixi.toml",
            job_file=str(job_file),
        )


def test_write_script_mamba_and_pixi_exclusive(tmp_path):
    """Test write_script raises ValueError when both mamba_env and pixi_env are specified."""
    pixi_toml = tmp_path / "pixi.toml"
    pixi_toml.write_text("[project]\nname = 'test'\n")

    manager = SweepManager()
    manager.sweep_id = "test-sweep-id"
    manager.project_name = "test_project"
    manager.entity = "test_entity"

    job_file = tmp_path / "submit.sh"

    with pytest.raises(ValueError, match="Cannot specify both `mamba_env` and `pixi_env`"):
        manager.write_script(
            slurm_parameters={"time": "01:00:00", "partition": "test"},
            mamba_env="test_env",
            pixi_env=str(pixi_toml),
            job_file=str(job_file),
        )
