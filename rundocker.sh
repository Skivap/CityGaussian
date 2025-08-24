docker run -it --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --name original_city_gs_container \
  -v /datasets/aurickd/colmap/nthu_large:/app/data/nthu \
  -v /datasets/aurickd/outputs/original_citygs:/app/output \
  aurickd/citygs:latest
