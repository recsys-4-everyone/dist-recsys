#!/bin/bash

# serving
docker pull tensorflow/serving

docker run -d -p 8501:8501 --mount type=bind,source=/home/axing/savers/din,target=/models/din -e MODEL_NAME=din -t tensorflow/serving

curl -X POST http://localhost:8501/v1/models/din:predict \
  -d '{
  "signature_name": "serving_default",
  "instances":[
     {
        "user_id":["a"],
        "item_id":["item01", "item02", "item02"],
        "age": [22],
        "gender": ["1"],
        "clicked_items_15d": ["item01", "item02"]
     }]}'
