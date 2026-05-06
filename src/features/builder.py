from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler


class FeatureBuilder:
    def __init__(
        self,
        input_columns: list[str],
        output_column: str = "features",
        scaled_output_column: str = "scaled_features",
    ):
        self.input_columns = input_columns
        self.output_column = output_column
        self.scaled_output_column = scaled_output_column

    def assemble_features(self, df: DataFrame) -> DataFrame:
        assembler = VectorAssembler(
            inputCols=self.input_columns,
            outputCol=self.output_column
        )
        return assembler.transform(df)

    def scale_features(self, df: DataFrame) -> DataFrame:
        scaler = StandardScaler(
            inputCol=self.output_column,
            outputCol=self.scaled_output_column,
            withStd=True,
            withMean=True,
        )
        scaler_model = scaler.fit(df)
        return scaler_model.transform(df)

    def build(self, df: DataFrame) -> DataFrame:
        featured_df = self.assemble_features(df)
        featured_df = self.scale_features(featured_df)
        return featured_df