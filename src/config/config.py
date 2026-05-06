from pydantic import BaseModel, Field
from typing import List


class SparkConfig(BaseModel):
    app_name: str = "OpenFoodFactsClustering"
    master: str = "local[*]"

    driver_memory: str = "4g"
    executor_memory: str = "4g"

    shuffle_partitions: int = 8
    adaptive_enabled: bool = True
    serializer: str = "org.apache.spark.serializer.KryoSerializer"


class DataConfig(BaseModel):
    data_path: str = "data/raw/products.csv"
    model_path: str = "data/processed/kmeans_model"

    selected_columns: List[str] = Field(default_factory=lambda: [
        "energy_100g",
        "fat_100g",
        "carbohydrates_100g",
        "sugars_100g",
        "proteins_100g",
    ])

    row_limit: int = 10000


class ModelConfig(BaseModel):
    k_clusters: int = 5
    seed: int = 51

    features_col: str = "features"
    scaled_features_col: str = "scaled_features"
    prediction_col: str = "prediction"


class ProjectConfig(BaseModel):
    spark: SparkConfig = SparkConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()