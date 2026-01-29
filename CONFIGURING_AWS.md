### 1. Log in to the AWS Console

Go to https://console.aws.amazon.com/iam/ and sign in with your AWS account.

### 2. Create a New IAM User

Navigate to Users → Add users

Enter a user name (e.g., s3-access-user)

Select Access key - Programmatic access

Click Next: Permissions

### 3. Attach Permissions Policy

On the Set permissions page:

Choose Attach policies directly

Search for and check AmazonS3FullAccess
🔒 Optional: For more limited access, see below.

Click Next until you reach Create user, then click Create user.

### 4. Save Access Keys

You will be shown an Access Key ID and Secret Access Key.
Important: Save these somewhere secure—this is the only time you’ll be able to view the secret key.

### 5. Set Environment Variables

Return to the [README](https://github.com/openpipe/summary-rl) instructions and set the following environment variables in your `.env` file:

```bash
AWS_ACCESS_KEY_ID=<your-access-key-id>
AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
AWS_REGION=<your-region>
```
