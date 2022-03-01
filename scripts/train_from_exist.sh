SCRIPTS=/cache/code_dir/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
TRUECASE_MODEL=$SCRIPTS/recaser/train-truecaser.perl
TRUECASE=$SCRIPTS/recaser/truecase.perl

BPEROOT=/cache/code_dir/subword-nmt/subword_nmt
BPE_TOKENS=40000


src=$1
tgt=$2
lang=$src-$tgt



max_update=$3
cd /cache/code_dir
pip install sacrebleu
pip install editdistance
python setup.py build develop --user

DATA=/cache/model_dir/data/bin/
OUTPUT=/cache/code_dir/output
python train.py /cache/data_dir/bin/  \
  --restore-file /cache/model_dir/checkpoint_last.pt \
  --source-lang $src --target-lang $tgt \
  --arch transformer_wmt_en_de \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 0.0007 --min-lr 1e-09 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
  --max-tokens  4096   --save-dir  /cache/model_dir \
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
--gen-subset test \
--path  /cache/model_dir/averaged_model.pt \
--beam 5 --batch-size 64 --remove-bpe|tee generate.out

grep ^T generate.out | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref

grep ^H generate.out |cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys

python fairseq_cli/score.py --sys generate.sys --ref generate.ref
cp generate.* /cache/model_dir