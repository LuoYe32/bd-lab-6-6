from pyspark.sql import DataFrame, SparkSession


class DataLoader:
    def __init__(self, spark: SparkSession, file_path: str):
        self.spark = spark
        self.file_path = file_path

    def load_csv(self) -> DataFrame:
        df = (
            self.spark.read
            .option("header", True)
            .option("inferSchema", True)
            .option("sep", "\t")
            .csv(self.file_path)
        )
        return df