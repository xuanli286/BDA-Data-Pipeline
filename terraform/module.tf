
# start s3
# resource "aws_s3_bucket" "is459-project-data" {
#   bucket = "is459-project-data"
# }

# resource "aws_s3_bucket" "is459-project-output-data" {
#   bucket = "is459-project-output-data"
# }

resource "aws_s3_bucket" "testing-athena" {
  bucket = "testing-athena-lalala"
}


# resource "aws_iam_policy_attachment" "glue_s3_access" {
#   policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
#   roles      = [aws_iam_role.example.name]
# }
# end s3


# start athena 
resource "aws_athena_database" "example" {
  name   = "athena_database"
  bucket = aws_s3_bucket.testing-athena.bucket
  comment = "our athene database"
}
# end athena
