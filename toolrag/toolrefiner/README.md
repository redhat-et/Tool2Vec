# ToolRefiner

## Model Training

You must provide your own `train.json`, `val.json`, and `test.json` files (see the main README for format).

To train the reranker model, run:
```sh
PYTHONPATH=. python toolrag/run_full_pipeline.py \
  --train-file toolrag/train.json \
  --val-file toolrag/val.json \
  --test-file toolrag/test.json
```

This will generate embeddings, train the model, and evaluate on the test set.

Alternatively, to run training directly:
```sh
PYTHONPATH=. python toolrag/toolrefiner/train_query_nt.py \
    --model "microsoft/deberta-v3-base" \
    --lr 8e-4 \
    --use_amp \
    --std 0.2 \
    --wd 0.01 \
    --num_tools_to_be_presented 16 \
    --num_linear_warmup_steps 100 \
    --batch_size 4 \
    --num_epochs 1 \
    --training_data_dir output/train_embeddings_embedded.json \
    --test_data_dir output/val_embeddings_embedded.json \
    --tool_embedding_dir output/train_embeddings_embedded.pkl \
    --tool_name_dir toolrag/train.json \
    --train_tool_top_k_retrieval_dir train_top_k.json \
    --valid_tool_top_k_retrieval_dir val_top_k.json \
    --checkpoint_dir toolrag/toolrefiner/checkpoints/
```

## Model Evaluation

To evaluate the trained model on the test set:
```sh
PYTHONPATH=. python toolrag/toolrefiner/query_tool_selector.py \
    --query "<your query>" \
    --model_name microsoft/deberta-v3-base \
    --checkpoint_path toolrag/toolrefiner/checkpoints/model_epoch_1.pt \
    --tool_embeddings_path output/test_embeddings_embedded.pkl \
    --tool_names_path toolrag/test.json \
    --tool_embedding_dim 768
```

## Notes
- Always set `PYTHONPATH=. ` when running scripts that import from the toolrag package.
- See the main README for environment setup and data preparation.
