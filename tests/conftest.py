import pytest
import yaml


@pytest.fixture
def valid_config_file(tmp_path):
    config = {
        "wandb": {
            "program": "train.py",
            "method": "grid",
            "parameters": {"lr": {"values": [0.01, 0.1]}},
        },
        "general": {
            "entity": "test_entity",
            "project_name": "test_project",
        },
    }
    file_path = tmp_path / "valid_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config, f)
    return file_path


@pytest.fixture
def invalid_config_file(tmp_path):
    config = {
        "wandb": {
            "method": "grid",
        },
        "general": {
            "entity": "test_entity",
        },
    }
    file_path = tmp_path / "invalid_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config, f)
    return file_path


@pytest.fixture
def unexpected_block_config_file(tmp_path):
    config = {
        "wandb": {
            "program": "train.py",
            "method": "grid",
            "parameters": {"lr": {"values": [0.01, 0.1]}},
        },
        "general": {
            "entity": "test_entity",
            "project_name": "test_project",
        },
        "extra_block": {"key": "value"},
    }
    file_path = tmp_path / "unexpected_block_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config, f)
    return file_path


@pytest.fixture
def sweep_config():
    return {
        "method": "grid",
        "parameters": {
            "lr": {"values": [0.01, 0.1]},
            "batch_size": {"values": [32, 64]},
        },
    }


@pytest.fixture
def expected_slurm_script():
    return """#!/bin/bash

#SBATCH --partition           test
#SBATCH --time                01:00:00

module load test_module
source $HOME/.bashrc
mamba activate test_env
wandb agent "test_entity/test_project/test-sweep-id"
"""

@pytest.fixture
def expected_slurm_script_pixi(tmp_path):
    """Return expected script and the pixi.toml path for pixi environment tests."""
    pixi_toml = tmp_path / "pixi.toml"
    pixi_toml.write_text("[project]\nname = 'test'\n")
    script = f"""#!/bin/bash

#SBATCH --partition           test
#SBATCH --time                01:00:00

module load test_module
pixi run --manifest-path {pixi_toml} wandb agent "test_entity/test_project/test-sweep-id"
"""
    return script, str(pixi_toml)


@pytest.fixture
def expected_slurm_script_uv_toml(tmp_path):
    """Return expected script and pyproject.toml path for uv environment tests (toml variant)."""
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    pyproject_toml = project_dir / "pyproject.toml"
    pyproject_toml.write_text("[project]\nname = 'test'\n")
    script = f"""#!/bin/bash

#SBATCH --partition           test
#SBATCH --time                01:00:00

module load test_module
uv run --project {project_dir} wandb agent "test_entity/test_project/test-sweep-id"
"""
    return script, str(pyproject_toml)


@pytest.fixture
def expected_slurm_script_uv_venv(tmp_path):
    """Return expected script and venv path for uv environment tests (venv variant)."""
    venv_dir = tmp_path / ".venv"
    venv_dir.mkdir()
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir()
    (bin_dir / "activate").write_text("# mock activate script")
    script = f"""#!/bin/bash

#SBATCH --partition           test
#SBATCH --time                01:00:00

module load test_module
source {venv_dir}/bin/activate
wandb agent "test_entity/test_project/test-sweep-id"
"""
    return script, str(venv_dir)


@pytest.fixture
def valid_cli_config_file(tmp_path):
    config = """
    wandb:
      program: train.py
      method: grid
      parameters:
        lr:
          values: [0.01, 0.1]
    general:
      entity: test_entity
      project_name: test_project
      mamba_env: test_env
      modules: test_module
    slurm:
      time: "01:00:00"
      partition: test
    """
    config_path = tmp_path / "cli_config.yaml"
    config_path.write_text(config)
    return str(config_path)
