"""
Core infrastructure for the wind turbine anomaly detection portfolio project.

Provisions:
  - S3 data bucket (raw/, labeled/, incoming/, pipeline/, models/ prefixes)
  - SageMaker execution role (used by the Pipeline, Processing/Training jobs, endpoint)
  - Streaming simulator Lambda + EventBridge schedule (replays labeled data into incoming/)
  - Alert aggregator Lambda + S3 event notification (calls the real-time endpoint,
    tracks a rolling window in DynamoDB, publishes to SNS on a sustained anomaly)
  - SSM parameter holding the live endpoint name (updated by a post-pipeline deploy
    script once the winning model is deployed; the endpoint itself is not a static
    CDK resource because it depends on a model registered at pipeline run time)

This is a single-stack, portfolio-scale deployment: removal policies are DESTROY
and S3 auto-delete is enabled so `cdk destroy` leaves nothing behind. Do not reuse
these removal policies for anything holding real data.
"""
from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_notifications as s3n
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from aws_cdk import aws_ssm as ssm
from constructs import Construct

ENDPOINT_NAME_PARAM = "/wind-turbine-anomaly/endpoint-name"
MODEL_PACKAGE_GROUP_NAME = "wind-turbine-anomaly-detector"


class WindTurbineAnomalyStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        schedule_rate_minutes = int(self.node.try_get_context("schedule_rate_minutes") or 10)
        alert_email = self.node.try_get_context("alert_email") or ""

        # ---------------------------------------------------------------
        # S3 data bucket
        # ---------------------------------------------------------------
        data_bucket = s3.Bucket(
            self,
            "DataBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            enforce_ssl=True,
        )

        # ---------------------------------------------------------------
        # SageMaker execution role (Pipeline steps + real-time endpoint)
        # ---------------------------------------------------------------
        sagemaker_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ],
            description="Execution role for the wind turbine anomaly detection SageMaker Pipeline",
        )
        data_bucket.grant_read_write(sagemaker_role)

        # ---------------------------------------------------------------
        # DynamoDB: alert aggregator rolling window state
        # ---------------------------------------------------------------
        rolling_window_table = dynamodb.Table(
            self,
            "RollingWindowTable",
            partition_key=dynamodb.Attribute(name="turbine_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ---------------------------------------------------------------
        # SNS: alert topic
        # ---------------------------------------------------------------
        alert_topic = sns.Topic(
            self,
            "AlertTopic",
            display_name="Wind Turbine Anomaly Alerts",
        )
        if alert_email:
            alert_topic.add_subscription(subs.EmailSubscription(alert_email))

        # ---------------------------------------------------------------
        # SSM parameter: live endpoint name, filled in by the post-pipeline
        # deploy step once a model is approved and deployed.
        # ---------------------------------------------------------------
        endpoint_name_param = ssm.StringParameter(
            self,
            "EndpointNameParameter",
            parameter_name=ENDPOINT_NAME_PARAM,
            string_value="not-deployed-yet",
            description="Name of the live SageMaker real-time endpoint for anomaly scoring",
        )

        # ---------------------------------------------------------------
        # Lambda: streaming simulator
        # ---------------------------------------------------------------
        streaming_simulator_fn = _lambda.Function(
            self,
            "StreamingSimulatorFunction",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            code=_lambda.Code.from_asset("../lambda/streaming_simulator"),
            timeout=Duration.seconds(60),
            memory_size=256,
            environment={
                "DATA_BUCKET": data_bucket.bucket_name,
                "LABELED_KEY": "labeled/labeled_scada.csv",
                "INCOMING_PREFIX": "incoming/",
                "POINTER_KEY": "incoming/_pointer.json",
                "ROWS_PER_BATCH": "1",
                "TURBINE_ID": "T1",
            },
        )
        data_bucket.grant_read_write(streaming_simulator_fn)

        schedule_rule = events.Rule(
            self,
            "StreamingSimulatorSchedule",
            schedule=events.Schedule.rate(Duration.minutes(schedule_rate_minutes)),
            description="Triggers the streaming simulator Lambda to replay the next batch of SCADA rows",
        )
        schedule_rule.add_target(targets.LambdaFunction(streaming_simulator_fn))

        # ---------------------------------------------------------------
        # Lambda: alert aggregator
        # ---------------------------------------------------------------
        alert_aggregator_fn = _lambda.Function(
            self,
            "AlertAggregatorFunction",
            runtime=_lambda.Runtime.PYTHON_3_12,
            handler="handler.lambda_handler",
            code=_lambda.Code.from_asset("../lambda/alert_aggregator"),
            timeout=Duration.seconds(30),
            memory_size=256,
            environment={
                "ROLLING_WINDOW_TABLE": rolling_window_table.table_name,
                "ALERT_TOPIC_ARN": alert_topic.topic_arn,
                "ENDPOINT_NAME_PARAM": ENDPOINT_NAME_PARAM,
                "ALERT_THRESHOLD": "3",
                "TURBINE_ID": "T1",
            },
        )
        data_bucket.grant_read(alert_aggregator_fn)
        rolling_window_table.grant_read_write_data(alert_aggregator_fn)
        alert_topic.grant_publish(alert_aggregator_fn)
        endpoint_name_param.grant_read(alert_aggregator_fn)
        alert_aggregator_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["sagemaker:InvokeEndpoint"],
                resources=[
                    f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{MODEL_PACKAGE_GROUP_NAME}*"
                ],
            )
        )

        data_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(alert_aggregator_fn),
            s3.NotificationKeyFilter(prefix="incoming/", suffix=".json"),
        )

        # ---------------------------------------------------------------
        # Outputs
        # ---------------------------------------------------------------
        CfnOutput(self, "DataBucketName", value=data_bucket.bucket_name)
        CfnOutput(self, "SageMakerExecutionRoleArn", value=sagemaker_role.role_arn)
        CfnOutput(self, "RollingWindowTableName", value=rolling_window_table.table_name)
        CfnOutput(self, "AlertTopicArn", value=alert_topic.topic_arn)
        CfnOutput(self, "EndpointNameParameterName", value=ENDPOINT_NAME_PARAM)
        CfnOutput(self, "StreamingSimulatorFunctionName", value=streaming_simulator_fn.function_name)
        CfnOutput(self, "AlertAggregatorFunctionName", value=alert_aggregator_fn.function_name)
