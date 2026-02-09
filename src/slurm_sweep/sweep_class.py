from pathlib import Path
import wandb
from simple_slurm import Slurm


class SweepManager:
    """A class to manage wandb sweeps and submit SLURM jobs."""

    def __init__(self) -> None:
        """Initialize the SweepManager with general configuration."""
        self.project_name: str | None = None
        self.entity: str | None = None
        self.sweep_id: str | None = None

    def __repr__(self) -> str:
        """
        Return a string representation of the SweepManager instance.

        Returns
        -------
        str
            A string showing the current state of the SweepManager instance.
        """
        return f"SweepManager(project_name={self.project_name!r}, entity={self.entity!r}, sweep_id={self.sweep_id!r})"

    def register_sweep(self, sweep_config: dict, project_name: str, entity: str) -> None:
        """
        Register a sweep with wandb using the provided sweep configuration.

        Parameters
        ----------
        project_name
            The name of the wandb project.
        entity
            The name of the wandb entity or team.
        sweep_config
            The configuration dictionary for the wandb sweep.
        """
        wandb.login()

        self.project_name = project_name
        self.entity = entity

        self.sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=self.project_name,
            entity=self.entity,
        )

    def write_script(
        self,
        slurm_parameters: dict | None = None,
        mamba_env: str | None = None,
        pixi_env: str | None = None,
        uv_env: str | None = None,
        job_file: str = "submit.sh",
        convert: bool = False,
        shell: str = "/bin/bash",
        modules: str | None = None,
        count: int | None = None,  # New optional parameter
    ) -> None:
        """
        Produce a submission script to run a grid search with wandb on a SLURM cluster.

        Parameters
        ----------
        slurm_parameters
            A dictionary of SLURM parameters to pass to the `Slurm` class.
        mamba_env
            Mamba environment to activate. Mutually exclusive with ``pixi_env`` and ``uv_env``.
        pixi_env
            Path to a pixi manifest file (pixi.toml) to use for running the wandb agent.
            The command will be wrapped with ``pixi run --manifest-path <pixi_env>``.
            Mutually exclusive with ``mamba_env`` and ``uv_env``.
        uv_env
            Path to a uv environment. Can be either:
            - A path to a `pyproject.toml` file: command will be wrapped with
              ``uv run --project <parent_dir>``.
            - A path to a virtual environment directory: will source the activate script.
            Mutually exclusive with ``mamba_env`` and ``pixi_env``.
        job_file
            The slurm submission script will be written here.
        convert
            Whether to escape bash variables (see the ``sbatch`` method in ``simple_slurm``).
        shell
            The shell to use for the SLURM job.
        modules
            A space-separated string of modules to load (e.g., "stack eth_proxy").
        count
            Optional count parameter for the wandb agent command. On SLURM clusters, it is recommended
            to set this to 1 to avoid multiple agents running on the same node.
        """
        if not self.sweep_id:
            raise ValueError("Sweep ID is not set. Please register the sweep first.")

        # Validate mutual exclusivity of environment options
        env_count = sum(x is not None for x in [mamba_env, pixi_env, uv_env])
        if env_count > 1:
            raise ValueError(
                "Cannot specify more than one environment type. "
                "Please choose one of: `mamba_env`, `pixi_env`, or `uv_env`."
            )

        # Validate pixi_env path exists
        if pixi_env and not Path(pixi_env).exists():
            raise FileNotFoundError(f"Pixi manifest file not found: '{pixi_env}'")

        # Validate uv_env path exists
        if uv_env and not Path(uv_env).exists():
            raise FileNotFoundError(f"uv environment path not found: '{uv_env}'")


        # Initialize SLURM with user-provided parameters
        slurm_parameters = slurm_parameters or {}
        slurm = Slurm(**slurm_parameters)

        # Load required modules if specified
        if modules:
            slurm.add_cmd(f"module load {modules}")

        # Activate mamba/conda environment
        if mamba_env:
            slurm.add_cmd("source $HOME/.bashrc")
            slurm.add_cmd(f"mamba activate {mamba_env}")

        # Build the wandb agent command with optional --count flag for SLURM compatibility
        count_flag = f"--count {count}" if count is not None else ""
        wandb_command = (
            f'wandb agent{f" {count_flag}" if count_flag else ""} "{self.entity}/{self.project_name}/{self.sweep_id}"'
        )
       
        # Wrap command based on environment type
        if pixi_env:
            command = f"pixi run --manifest-path {pixi_env} {wandb_command}"
        elif uv_env:
            uv_path = Path(uv_env)
            if uv_path.suffix == ".toml":
                # It's a pyproject.toml file, use the parent directory as the project
                project_dir = uv_path.parent
                command = f"uv run --project {project_dir} {wandb_command}"
            else:
                # It's a virtual environment directory, source the activate script
                slurm.add_cmd(f"source {uv_path}/bin/activate")
                command = wandb_command
        else:
            command = wandb_command
        
        slurm.add_cmd(command)

        # Write to file
        with open(job_file, "w") as fid:
            fid.write(slurm.script(shell, convert))
