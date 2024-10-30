# BDA-Data-Pipeline

## Terraform
1. cd to terraform
2. Run the following commands in Terminal to set up the AWS services. <br>
`terraform init` <br>
`terraform apply` <br>
3. Run the below command in Terminal to destroy the AWS services. <br>
`terraform destroy`

## Tableau
1. Download Athena driver to connect to Tableau <br>
https://help.tableau.com/current/pro/desktop/en-us/examples_amazonathena.htm
2. Launch Tableau, select Amazon Athena under Connection and key in the following. <br>
<b>Access Key ID:</b> `AWS_ACCESS_KEY_ID` (refer to the .env file under the backend folder)
<b>Secret Access Key:</b> `AWS_SECRET_ACCESS_KEY`