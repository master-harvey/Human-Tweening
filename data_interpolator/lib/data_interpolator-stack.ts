import { Stack, StackProps, RemovalPolicy, CfnOutput, aws_lambda as lambda, aws_dynamodb as dynamo } from 'aws-cdk-lib';
import { Construct } from 'constructs';

export class PathDataInterpolatorStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const table = new dynamo.Table(this, "PathDataTable", {
      partitionKey: { name: "batch_id", type: dynamo.AttributeType.STRING },
      sortKey: { name: "start_timestamp", type: dynamo.AttributeType.NUMBER },
      removalPolicy: RemovalPolicy.DESTROY
    })

    const interpolateFunction = new lambda.Function(this, "PathInterpolatorFunction", {
      code: lambda.Code.fromAsset("./lambda"), runtime: lambda.Runtime.PYTHON_3_12, handler: "main.handler",
      environment: { "TABLE_NAME": table.tableName }
    })
    const functionURL = interpolateFunction.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
      cors: {
        allowedOrigins: ["https://human-tweening.vercel.app"],
        allowedHeaders: ["content-type"],
        allowedMethods: [lambda.HttpMethod.PUT]
      }
    })
    table.grantWriteData(interpolateFunction)

    new CfnOutput(this, "ENDPOINT", { value: functionURL.url })

  }
}
