
from aws_lambda_builders.builder import LambdaBuilder
lb = LambdaBuilder("python","pip",None)
lb.build(".", "lb_build","lb_scratch", "requirements.txt", runtime="python3.9")