#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Fetching artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading dataframe")
    df = pd.read_csv(artifact_local_path)

    logger.info("Starting pre-processing - drop outliers in price")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # drop rows in the dataset that are not in the proper geolocation
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(
        40.5, 41.2)
    df = df[idx].copy()

    df.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully qualified name for the artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the W&B artifact that will be created",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the artifact to create",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price for dropping outliers",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price for dropping outliers",
        required=True
    )

    args = parser.parse_args()

    go(args)
