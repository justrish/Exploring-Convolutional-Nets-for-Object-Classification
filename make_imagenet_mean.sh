#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/pca_alex/data/lmdb
DATA=examples/pca_alex/data	
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_lmdb \
  $DATA/pca_alex_mean.binaryproto

echo "Done."
