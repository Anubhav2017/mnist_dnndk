#!/bin/bash

#conda activate decent_q3

echo "#####################################"
echo "EVALUATE QUANTIZED GRAPH"
echo "#####################################"


python3 eval_graph.py \
  --graph ./quantize_results/quantize_eval_model.pb \
  --input_node images_in \
  --output_node dense_1/BiasAdd \

echo "#####################################"
echo "EVALUATE QUANTIZED GRAPH COMPLETED"
echo "#####################################"

