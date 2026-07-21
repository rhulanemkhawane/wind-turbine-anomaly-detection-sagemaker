#!/usr/bin/env python3
import os

import aws_cdk as cdk

from stacks.wind_turbine_stack import WindTurbineAnomalyStack

app = cdk.App()

WindTurbineAnomalyStack(
    app,
    "WindTurbineAnomalyStack",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    ),
)

app.synth()
