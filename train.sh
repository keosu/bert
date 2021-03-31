export MODEL_DIR=./models/uncased_L-2_H-128_A-2
export SQUAD_DIR=./data
export OUT_DIR=./train_output

python bert/run_squad.py \
  --vocab_file=$MODEL_DIR/vocab.txt \
  --bert_config_file=$MODEL_DIR/bert_config.json \
  --init_checkpoint=$MODEL_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=16 \
  --learning_rate=3e-5 \
  --num_train_epochs=10.0 \
  --warmup_proporion=0.0 \
  --max_seq_length=128 \
  --doc_stride=96 \
  --pre_layer_norm=True \
  --output_dir=$OUT_DIR
