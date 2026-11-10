#!/bin/sh
# Create the IAM user a Claude Code cloud environment uses for Earlybird, with only the Earlybird
# permissions (docs/cloud-env-iam-policy.json), and print an access key as env-var lines.
# Run it as an AWS admin, e.g. in AWS CloudShell (us-east-1):  sh create_cloud_iam_user.sh
# Re-running it is safe: it reuses the user and the policy, and makes a new key only if the user has
# fewer than 2 (delete an old one first with: aws iam delete-access-key --user-name earlybird-cloud --access-key-id AKIA...).
set -e
USER=earlybird-cloud
cat > /tmp/earlybird-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "Bucket", "Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetObject", "s3:PutObject"],
     "Resource": ["arn:aws:s3:::${EARLYBIRD_BUCKET}", "arn:aws:s3:::${EARLYBIRD_BUCKET}/*"]},
    {"Sid": "LaunchEvalAndGpuBoxes", "Effect": "Allow",
     "Action": ["ec2:RunInstances", "ec2:CreateTags", "ec2:Describe*"], "Resource": "*"},
    {"Sid": "ManageOnlyEarlybirdInstances", "Effect": "Allow",
     "Action": ["ec2:StartInstances", "ec2:StopInstances", "ec2:TerminateInstances", "ec2:ModifyInstanceAttribute"],
     "Resource": "arn:aws:ec2:us-east-1:${AWS_ACCOUNT_ID}:instance/*",
     "Condition": {"StringEquals": {"aws:ResourceTag/Project": "earlybird"}}},
    {"Sid": "PassBoxRole", "Effect": "Allow", "Action": "iam:PassRole",
     "Resource": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/earlybird-box"},
    {"Sid": "SsmToEarlybirdInstances", "Effect": "Allow", "Action": ["ssm:SendCommand", "ssm:StartSession"],
     "Resource": ["arn:aws:ec2:us-east-1:${AWS_ACCOUNT_ID}:instance/*"],
     "Condition": {"StringEquals": {"aws:ResourceTag/Project": "earlybird"}}},
    {"Sid": "SsmDocsAndResults", "Effect": "Allow",
     "Action": ["ssm:SendCommand", "ssm:GetCommandInvocation", "ssm:ListCommands", "ssm:ListCommandInvocations", "ssm:DescribeInstanceInformation"],
     "Resource": ["arn:aws:ssm:us-east-1::document/AWS-RunShellScript", "arn:aws:ssm:us-east-1:${AWS_ACCOUNT_ID}:*"]}
  ]
}
EOF
[ "$(aws sts get-caller-identity --query Account --output text)" = "${AWS_ACCOUNT_ID:?set AWS_ACCOUNT_ID}" ] || { echo "wrong AWS account" >&2; exit 1; }
aws iam get-user --user-name $USER >/dev/null 2>&1 || aws iam create-user --user-name $USER --tags Key=Project,Value=earlybird >/dev/null
aws iam put-user-policy --user-name $USER --policy-name earlybird-cloud --policy-document file:///tmp/earlybird-policy.json
n=$(aws iam list-access-keys --user-name $USER --query 'length(AccessKeyMetadata)' --output text)
[ "$n" -lt 2 ] || { echo "$USER already has 2 access keys; delete one first (see the top of this script)" >&2; exit 1; }
set -- $(aws iam create-access-key --user-name $USER --query 'AccessKey.[AccessKeyId,SecretAccessKey]' --output text)
rm -f /tmp/earlybird-policy.json
echo
echo "Paste these into the cloud environment's environment variables:"
echo
echo "AWS_ACCESS_KEY_ID=$1"
echo "AWS_SECRET_ACCESS_KEY=$2"
echo "AWS_DEFAULT_REGION=us-east-1"
