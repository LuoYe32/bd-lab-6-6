from src.pipeline.clustering_pipeline import ClusteringPipeline


def main() -> None:
    pipeline = ClusteringPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()