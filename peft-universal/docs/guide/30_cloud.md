# 30. Cloud: FakeCloud first, then AWS

:::why
Test cloud code locally for free before touching a real account. FakeCloud (fakecloud.dev) is a local AWS emulator on `localhost:4566`; its site lists S3, ECR, ECS, EKS, Application Auto Scaling, CloudWatch and more. This template uses the cloud for one thing, artifact storage (S3), and the guide maps the serving stack to AWS services.
:::

## 30.1 FakeCloud locally (WSL)

```wsl Install and run (official install script from fakecloud.dev)
curl -fsSL https://fakecloud.dev/install.sh | bash
fakecloud                                    # leave running; listens on http://localhost:4566
```

```wsl In a second WSL terminal
pip install awscli boto3
export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=us-east-1
export AWS_ENDPOINT_URL=http://localhost:4566
aws s3 mb s3://models
cd /mnt/c/Users/<you>/code/peft-universal
python -m peftkit.publish --src tests --s3 s3://models/smoke      # any folder works for a smoke test
aws s3 ls s3://models/smoke/ --recursive | head
```

:::honest
FakeCloud itself was not run in this guide. The S3 code path was tested with `moto` (an in-process S3 mock) in `tests/test_publish.py`: upload, skip rules and download round-trip all pass. On Windows, run FakeCloud inside WSL (the installer is a bash script) or use its Docker image per its docs.
:::

## 30.2 Real AWS: what changes

|Piece|Local / Hyperstack|AWS equivalent|Difference|
|---|---|---|---|
|artifact store|FakeCloud S3 / Hyperstack object storage|Amazon S3|unset `AWS_ENDPOINT_URL`; real IAM credentials|
|gateway image|local Docker build|Amazon ECR|`aws ecr create-repository`, `docker push`|
|serving nodes|Hyperstack VMs|EC2 GPU instances (g5/g6/p4d/p5 families) or EKS GPU node groups|quotas for GPU instances must be requested|
|autoscaling|KEDA / scaler.py|KEDA on EKS + Karpenter or Cluster Autoscaler; or ECS with Application Auto Scaling|IAM roles for controllers|
|metrics|Prometheus|Amazon Managed Prometheus / CloudWatch|scrape config|
|secrets|.env|AWS Secrets Manager / K8s secrets|no plain files in production|
|TLS + entry|reverse proxy|Application Load Balancer + ACM certificate|-|

## 30.3 Budget alert and cleanup (real AWS)

```any Budget: e-mail when actual spend passes 80% of a monthly limit
aws budgets create-budget --account-id <ACCOUNT_ID> \
  --budget '{"BudgetName":"llm-monthly","BudgetLimit":{"Amount":"100","Unit":"USD"},"TimeUnit":"MONTHLY","BudgetType":"COST"}' \
  --notifications-with-subscribers '[{"Notification":{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80,"ThresholdType":"PERCENTAGE"},"Subscribers":[{"SubscriptionType":"EMAIL","Address":"you@example.com"}]}]'
```

(The amount is a placeholder; set your own.)

```any Cleanup
aws s3 rm s3://models --recursive && aws s3 rb s3://models
aws ecr delete-repository --repository-name llm-gateway --force
kubectl delete -f serving/k8s/ ; eksctl delete cluster --name <cluster>      # if you created one
aws ec2 describe-instances --filters Name=instance-state-name,Values=running --query "Reservations[].Instances[].InstanceId"
```

On Hyperstack: delete VMs, volumes and unused floating IPs in the console; check the billing page.

:::check
- S3 upload works against FakeCloud (or moto in tests).
- A budget alert exists before any real AWS GPU is started.
:::
