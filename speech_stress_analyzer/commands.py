import fire
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import mlflow
import os
import dvc.api
import sys

from . import preprocess as preprocess_module
from . import train as train_module
from . import infer as infer_module
from . import download_data as download_data_module

cs = ConfigStore.instance()

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def _preprocess_hydra_entrypoint(cfg: DictConfig) -> None:
    print("Starting preprocessing command (via _preprocess_hydra_entrypoint)...\n")
    
    if cfg.dvc.pull_data_on_train:
        print("Downloading data via DVC...")
        download_data_module.download_data(cfg)
    
    preprocess_module.run_preprocessing(cfg)
    print("Preprocessing command (via _preprocess_hydra_entrypoint) finished.\n")

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def _train_hydra_entrypoint(cfg: DictConfig, commands_instance: 'Commands') -> None:
    print("Starting training command (via _train_hydra_entrypoint)...\n")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    if cfg.dvc.pull_data_on_train:
        print("Pulling data with DVC...")
        try:
            if os.system('dvc pull') != 0:
                print("Warning: 'dvc pull' command failed.")
        except Exception as e:
            print(f"Error running dvc pull: {e}")

    mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.logging.mlflow_experiment_name)
    git_commit_id = commands_instance._get_git_commit_id()

    with mlflow.start_run(run_name=cfg.train.run_name) as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
        mlflow.log_param("git_commit_id", git_commit_id)
        print(f"Using PyTorch Lightning for training as per config: {cfg.train.framework}")
        train_module.run_training(cfg)
    print("Training command finished.\n")
    if cfg.train.get("export_onnx_after_train", False):
        print("Exporting ONNX model after training...")
        commands_instance.export_onnx(cfg)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def _infer_hydra_entrypoint(cfg: DictConfig, input_audio: str) -> None:
    print(f"Starting inference for input: {input_audio} (via _infer_hydra_entrypoint)...\n")
    if cfg.dvc.get("pull_model_on_infer", True):
        print("Pulling model/data with DVC...")
        try:
            if os.system('dvc pull') != 0:
                print("Warning: 'dvc pull' command failed during inference setup.")
        except Exception as e:
            print(f"Error running dvc pull for inference: {e}")
    infer_module.run_inference(cfg, input_audio)
    print("Inference command finished.\n")

class Commands:
    """Main CLI commands for the MLOps project."""

    def _get_git_commit_id(self):
        try:
            import subprocess
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except Exception:
            return "unknown"

    def train(self, *overrides: str) -> None:
        """Train the model using the specified configuration and optional Hydra overrides."""
        with hydra.initialize(config_path="../configs", version_base=None, job_name="train_job"):
            cfg = hydra.compose(config_name="config", overrides=list(overrides))
            
            print("Starting training command...")
            print(f"Overrides: {list(overrides)}")
            print(f"Configuration:\\n{OmegaConf.to_yaml(cfg)}")

            if cfg.dvc.get("pull_data_on_train", False):
                print("Checking for DVC repository...")
                try:
                    import subprocess
                    result = subprocess.run(['dvc', 'status'], capture_output=True, text=True)
                    if result.returncode == 0:
                        print("Pulling data with DVC...")
                        ret_code = os.system('dvc pull')
                        if ret_code != 0:
                            print("Warning: 'dvc pull' command failed.")
                    else:
                        print("Warning: Not in a DVC repository. Skipping DVC pull.")
                except Exception as e:
                    print(f"Warning: DVC not available or repository not initialized. Skipping DVC pull. Error: {e}")

            mlflow_uri = cfg.get("mlflow_tracking_uri", "file:./mlruns")
            print(f"Setting MLflow tracking URI: {mlflow_uri}")
            mlflow.set_tracking_uri(mlflow_uri)
            
            experiment_name = cfg.get("mlflow_experiment_name", "SpeechStressAnalysis_Default")
            mlflow.set_experiment(experiment_name)
            
            git_commit_id = self._get_git_commit_id()
            run_name = cfg.get("run_name", "default_training_run")

            with mlflow.start_run(run_name=run_name) as run:
                print(f"MLflow Run ID: {run.info.run_id}")
                print(f"MLflow Experiment: {experiment_name}")
                
                try:
                    mlflow.log_params(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
                    mlflow.log_param("git_commit_id", git_commit_id)
                except Exception as e:
                    print(f"Warning: Could not log parameters to MLflow: {e}")

                print(f"Using PyTorch Lightning for training as per config: {cfg.get('framework', 'PyTorch Lightning')}")
                
                try:
                    train_module.run_training(cfg)
                    print("Training command finished successfully.")
                except Exception as e:
                    print(f"Training failed with error: {e}")
                    mlflow.log_param("training_status", "failed")
                    mlflow.log_param("error_message", str(e))
                    raise

    def infer(self, input_audio: str, *overrides: str) -> None:
        """Run inference on new data, with optional Hydra overrides."""
        with hydra.initialize(config_path="../configs", version_base=None, job_name="infer_job"):
            cfg = hydra.compose(config_name="config", overrides=list(overrides))

            print(f"Starting inference for input: {input_audio}...")
            print(f"Overrides: {list(overrides)}")
            print(f"Configuration:\\n{OmegaConf.to_yaml(cfg)}")

            if cfg.dvc.get("pull_model_on_infer", True):
                print("Pulling model/data with DVC...")
                try:
                    ret_code = os.system('dvc pull')
                    if ret_code != 0:
                        print("Warning: 'dvc pull' command failed during inference setup.")
                except Exception as e:
                    print(f"Error running dvc pull for inference: {e}")
            
            infer_module.run_inference(cfg, input_audio)
            print("Inference command finished.")

    def preprocess(self, *overrides: str) -> None:
        """Preprocess the data using the specified configuration and optional Hydra overrides."""
        with hydra.initialize(config_path="../configs", version_base=None, job_name="preprocess_job"):
            cfg = hydra.compose(config_name="config", overrides=list(overrides))
            
            print("Starting preprocessing command...")
            print(f"Overrides: {list(overrides)}")
            print(f"Configuration:\\n{OmegaConf.to_yaml(cfg)}")
            preprocess_module.run_preprocessing(cfg)
            print("Preprocessing command finished.")

    def download_data(self, *overrides: str) -> None:
        """Download data from DVC remote storage, with optional Hydra overrides."""
        with hydra.initialize(config_path="../configs", version_base=None, job_name="download_data_job"):
            cfg = hydra.compose(config_name="config", overrides=list(overrides))

            print("Starting data download command...")
            print(f"Overrides: {list(overrides)}")
            print(f"Configuration:\\n{OmegaConf.to_yaml(cfg)}")
            download_data_module.download_data(cfg)
            print("Data download command finished.")

    def export_onnx(self, *overrides: str) -> None:
        """Export the trained model to ONNX format, with optional Hydra overrides."""
        with hydra.initialize(config_path="../configs", version_base=None, job_name="export_onnx_job"):
            cfg = hydra.compose(config_name="config", overrides=list(overrides))
            
            print("Exporting model to ONNX...")
            print(f"Overrides: {list(overrides)}")
            print(f"Configuration:\\n{OmegaConf.to_yaml(cfg)}")
            
            onnx_model_path = cfg.model.onnx_export.model_onnx_path
            print(f"Placeholder: ONNX export logic to be implemented. Expected output: {onnx_model_path}")
            print(f"Model export to ONNX command finished. Remember to `dvc add` and `dvc push` if successful.")

    def export_tensorrt(self, *overrides: str) -> None:
        """Convert ONNX model to TensorRT engine, with optional Hydra overrides."""
        with hydra.initialize(config_path="../configs", version_base=None, job_name="export_tensorrt_job"):
            cfg = hydra.compose(config_name="config", overrides=list(overrides))

            print("Converting ONNX model to TensorRT...")
            print(f"Overrides: {list(overrides)}")
            print(f"Configuration:\\n{OmegaConf.to_yaml(cfg)}")
            
            tensorrt_plan_path = cfg.model.tensorrt_export.get("plan_path", "models/model.plan")
            print(f"Placeholder: TensorRT conversion logic to be implemented. Expected output: {tensorrt_plan_path}")
            print(f"Model conversion to TensorRT command finished. Remember to `dvc add` and `dvc push` if successful.")

if __name__ == "__main__":
    fire.Fire(Commands) 