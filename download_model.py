from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking import MlflowClient

client = MlflowClient()
my_model = client.download_artifacts("3ab9a9afa87f4146a82ca64283fedaa6", path="model")
print(f"Placed model in: {my_model}")