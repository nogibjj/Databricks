from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking import MlflowClient

client = MlflowClient()
my_model = client.download_artifacts("b43b4e87d0664bda87e43eaa255e277d", path="model")
print(f"Placed model in: {my_model}")


