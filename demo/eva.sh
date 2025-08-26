#!/usr/bin/env bash
python evaluation.py \
  --weights /root/autodl-tmp/ultralytics-mainPro/demo/runs/train/exp_AFPNnew/weights/best.pt \
  --data ultralytics/cfg/datasets/AAA_my_datasets.yaml \
  --imgsz 640 \
  --batch 8 \
  --split test \
  --conf 0.001 \
  --iou 0.6 \
  --save-dir /root/autodl-tmp/ultralytics-mainPro/evaluation_results \
  "$@"