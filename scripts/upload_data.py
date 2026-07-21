"""
Uploads the raw and labeled SCADA data to S3 so the SageMaker Pipeline and
streaming simulator Lambda can read it.

Usage:
  python scripts/upload_data.py --bucket <data-bucket-name>

The bucket name is also emitted as a CDK stack output (DataBucketName) after
`cdk deploy`, so you can do:
  python scripts/upload_data.py --bucket $(aws cloudformation describe-stacks \
      --stack-name WindTurbineAnomalyStack \
      --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" \
      --output text)
"""
import argparse
from pathlib import Path

import boto3


def upload_file(s3, local_path: Path, bucket: str, key: str) -> None:
    print(f"uploading {local_path} -> s3://{bucket}/{key}")
    s3.upload_file(str(local_path), bucket, key)


def main(args: argparse.Namespace) -> None:
    s3 = boto3.client("s3", region_name=args.region)

    raw_path = Path(args.raw_dir) / "T1.csv"
    labeled_path = Path(args.labeled_dir) / "labeled_scada.csv"
    events_path = Path(args.labeled_dir) / "events.csv"

    if raw_path.exists():
        upload_file(s3, raw_path, args.bucket, f"raw/{raw_path.name}")
    else:
        print(f"skipping raw upload, not found: {raw_path}")

    for path, prefix in [(labeled_path, "labeled"), (events_path, "labeled")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/inject_anomalies.py first."
            )
        upload_file(s3, path, args.bucket, f"{prefix}/{path.name}")

    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="Target S3 bucket name")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--labeled-dir", default="data/labeled")
    main(parser.parse_args())
