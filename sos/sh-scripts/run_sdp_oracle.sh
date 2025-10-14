#!/bin/bash

# Run SDP Oracle script
python sos/scripts/run_sdp_oracles.py \
    --input_path "/scratch/llm/ais2t/sos/large/n6_sparse_uniform_lowrank_d10_m60/test.jsonl" \
    --run_name "sdp-oracle" \
    --oracle "newton" \
    --max_examples 5 \
    --num_variables 6 \
    --max_degree 10 \
    --use_basis_extension \
    --model_path "/scratch/htc/npelleriti/models/sos-transformer/tiny/n4_sparse_uniform_lowrank_d3_m20" \
