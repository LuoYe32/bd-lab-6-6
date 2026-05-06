from pyspark.sql import SparkSession

from src.config.config import ProjectConfig


class SparkManager:
    def __init__(self):
        self.config = ProjectConfig()
        self.app_name = self.config.spark.app_name
        self.master = self.config.spark.master
        self.spark = None

    def create_session(self) -> SparkSession:
        if self.spark is None:
            self.spark = (
                SparkSession.builder
                .appName(self.config.spark.app_name)
                .master(self.config.spark.master)

                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g")

                .config("spark.sql.shuffle.partitions", "8")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.sql.adaptive.enabled", "true")

                .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
                .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")

                .getOrCreate()
            )

        return self.spark

    def stop_session(self):
        if self.spark:
            self.spark.stop()