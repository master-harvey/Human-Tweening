#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { PathDataInterpolatorStack } from '../lib/data_interpolator-stack';

const app = new cdk.App();
new PathDataInterpolatorStack(app, 'PathDataInterpolatorStack', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: "us-east-2" }
});