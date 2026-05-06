from pyspark.sql import DataFrame
from pyspark.sql.functions import count

from src.config.config import ProjectConfig
from src.utils.logger import LoggerFactory
from src.utils.spark_manager import SparkManager
from src.data.loader import DataLoader
from src.data.preprocessing import DataPreprocessor
from src.features.builder import FeatureBuilder
from src.models.kmeans_model import KMeansClusteringModel
from src.evaluation.evaluator import ClusteringModelEvaluator


class ClusteringPipeline:
    def __init__(self):
        self.config = ProjectConfig()
        self.logger = LoggerFactory.get_logger(self.__class__.__name__)

        self.spark_manager = SparkManager()

        self.spark = None
        self.loader = None
        self.preprocessor = None
        self.feature_builder = None
        self.model = None
        self.evaluator = None

    def _initialize_components(self) -> None:
        self.logger.info("Initializing components...")

        self.spark = self.spark_manager.create_session()

        self.loader = DataLoader(
            spark=self.spark,
            file_path=self.config.data.data_path,
        )

        self.preprocessor = DataPreprocessor(
            selected_columns=self.config.data.selected_columns,
            row_limit=self.config.data.row_limit,
        )

        self.feature_builder = FeatureBuilder(
            input_columns=self.config.data.selected_columns,
            output_column=self.config.model.features_col,
            scaled_output_column=self.config.model.scaled_features_col,
        )

        self.model = KMeansClusteringModel(
            k=self.config.model.k_clusters,
            seed=self.config.model.seed,
            features_col=self.config.model.features_col,
            prediction_col=self.config.model.prediction_col,
        )

        self.evaluator = ClusteringModelEvaluator(
            features_col=self.config.model.features_col,
            prediction_col=self.config.model.prediction_col,
        )

    def run(self) -> None:
        self._initialize_components()

        self.logger.info("Loading data...")
        raw_df = self.loader.load_csv()
        self.logger.info(f"Initial row count: {raw_df.count()}")

        self.logger.info("Preprocessing data...")
        processed_df = self.preprocessor.preprocess(raw_df)
        self.logger.info(f"Processed row count: {processed_df.count()}")

        self.logger.info("Building features...")
        featured_df = self.feature_builder.build(processed_df)

        self.logger.info("Training K-Means model...")
        self.model.train(featured_df)

        self.logger.info("Making predictions...")
        predictions_df = self.model.predict(featured_df)

        self.logger.info("Evaluating clustering quality...")
        silhouette_score = self.evaluator.evaluate(predictions_df)

        self.logger.info("\nRESULTS")
        self.logger.info(f"Silhouette score: {silhouette_score:.4f}")

        self.logger.info("\nCluster centers:")
        centers = self.model.get_cluster_centers()
        for idx, center in enumerate(centers):
            self.logger.info(f"Cluster {idx}: {center}")

        self.logger.info("\nCluster sizes:")
        predictions_df.groupBy(self.config.model.prediction_col) \
            .agg(count("*").alias("cluster_size")) \
            .orderBy(self.config.model.prediction_col) \
            .show()

        self.logger.info("\nSample predictions:")
        predictions_df.select(
            *self.config.data.selected_columns,
            self.config.model.prediction_col
        ).show(20, truncate=False)

        self.logger.info("Saving model...")
        self.model.save(self.config.data.model_path)

        self.spark_manager.stop_session()