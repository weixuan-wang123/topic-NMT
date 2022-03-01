SCRIPTS=/cache/code_dir/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
TRUECASE_MODEL=$SCRIPTS/recaser/train-truecaser.perl
TRUECASE=$SCRIPTS/recaser/truecase.perl

BPEROOT=/cache/code_dir/subword-nmt/subword_nmt



src=$1
tgt=$2
lang=$src-$tgt



if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi






cd /cache/code_dir
pip install --upgrade torch torchvision
pip install sacrebleu
pip install editdistance
python setup.py build develop --user
TEXT=/cache/data_dir/$src-$tgt
src=$1
tgt=$2

max_update=$3
python train.py /cache/data_dir/bin/  \
  --source-lang $src --target-lang $tgt \
  --arch enc_dec_transformer_13 --tensorboard-logdir /cache/model_dir \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 0.0007 --min-lr 1e-09 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
  --max-tokens  4096   --save-dir  /cache/model_dir --ddp-backend=no_c10d --eval-bleu \
  --update-freq 2 --no-progress-bar --log-format json --log-interval 50 \
  --save-interval-updates 1000 --keep-interval-updates 20 --max-update $max_update


python scripts/average_checkpoints.py \
--inputs /cache/model_dir/ \
--num-epoch-checkpoints  5 --output /cache/model_dir/averaged_model.pt


DATA=/cache/data_dir/bin/
OUTPUT=/cache/code_dir/output

echo "-------------------------------------translate the test set----------------------------------------------------"

# Evaluate
python fairseq_cli/generate.py $DATA \
--gen-subset test --normalized \
--path  /cache/model_dir/averaged_model.pt \
--beam 5 --batch-size 64 --remove-bpe|tee generate.out

grep ^T generate.out | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref

grep ^H generate.out |cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys

python fairseq_cli/score.py --sys generate.sys --ref generate.ref
cp generate.* /cache/model_dir