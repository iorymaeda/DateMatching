#!/bin/bash

torchserve --start \
            --ncs  \
            --model-store model-store \
            --models face_recognition