#!/bin/bash
python my_main.py --model resnet50 --N 4 --pretrained --lr 0.01 0.001 0.0001 --batch-size 100 --augment-list hflip vflip affine adjust_brightness adjust_saturation --attribute attribute_pigment_network attribute_milia_like_cyst --freezing --K-models 1
