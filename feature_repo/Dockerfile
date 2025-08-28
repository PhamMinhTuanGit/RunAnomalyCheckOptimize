FROM python:3.10-slim

WORKDIR /app/feature_repo

RUN pip install --upgrade pip
RUN pip install feast pandas pyarrow

COPY . /app/feature_repo

CMD ["feast", "apply"]
