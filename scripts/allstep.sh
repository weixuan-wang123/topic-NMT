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

IFS=","
read -ra CORPORA <<< "$4"
read -ra DEV <<< "$5"
read -ra TEST <<< "$6"

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi


prep=/cache/data_dir/$src-$tgt
tmp=$prep/tmp
orig=/cache/data_dir/

mkdir -p $orig $tmp $prep

echo "-----------pre-processing data----------------------------------"
echo "pre-processing train data:NORM REM TOKENIZER"
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 64 -a -l $l >> $tmp/train.tags.$lang.tok_b.$l
    done
done

echo "pre-processing dev data:NORM REM TOKENIZER"
for l in $src $tgt; do
    rm $tmp/dev.tags.$lang.tok.$l
    for f in "${DEV[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 64 -a -l $l >> $tmp/dev.tags.$lang.tok_b.$l
    done
done


echo "pre-processing test data:NORM REM TOKENIZER"
for l in $src $tgt; do
    rm $tmp/test.tags.$lang.tok.$l
    for f in "${TEST[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 64 -a -l $l >> $tmp/test.tags.$lang.tok_b.$l
    done
done


echo "pre-processing train data:train_truecase_model"
for l in $src ; do
   for f in "${CORPORA[@]}"; do
       perl $TRUECASE_MODEL -corpus $tmp/train.tags.$lang.tok_b.$l -model $tmp/train_truecase_model.$src
   done
done

for l in $src ; do
    for f in "${CORPORA[@]}"; do
        perl $TRUECASE -model $tmp/train_truecase_model.$src < $tmp/train.tags.$lang.tok_b.$l > $tmp/train.tags.$lang.tok.$l
    done
done

for l in $src ; do
    for f in "${DEV[@]}"; do
        cat $orig/$f.$l | \
            perl $TRUECASE -model $tmp/train_truecase_model.$src < $tmp/dev.tags.$lang.tok_b.$l > $tmp/dev.tags.$lang.tok.$l
    done
done

for l in $src ; do
    for f in "${TEST[@]}"; do
        cat $orig/$f.$l | \
            perl $TRUECASE -model $tmp/train_truecase_model.$src < $tmp/test.tags.$lang.tok_b.$l > $tmp/test.tags.$lang.tok.$l
    done
done


mv $tmp/train.tags.$lang.tok_b.$tgt $tmp/train.tags.$lang.tok.$tgt
mv $tmp/dev.tags.$lang.tok_b.$tgt $tmp/dev.tags.$lang.tok.$tgt
mv $tmp/test.tags.$lang.tok_b.$tgt $tmp/test.tags.$lang.tok.$tgt


TRAIN=$tmp/train.$src-$tgt
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
   cat $tmp/train.tags.$lang.tok.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.tags.$lang.tok.$L dev.tags.$lang.tok.$L test.tags.$lang.tok.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done
echo "bpe finish"
perl $CLEAN -ratio 1.5 $tmp/bpe.train.tags.$lang.tok $src $tgt $prep/train 1 500
perl $CLEAN -ratio 1.5 $tmp/bpe.dev.tags.$lang.tok $src $tgt $prep/valid 1 500
echo "bpe youwanle"
for L in $src $tgt; do
    cp $tmp/bpe.test.tags.$lang.tok.$L $prep/test.$L
done

echo "prepare finish"

cd /cache/code_dir
python setup.py build develop --user
TEXT=/cache/data_dir/$src-$tgt
src=$1
tgt=$2
python preprocess.py --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir /cache/data_dir/bin/ --thresholdtgt 0 --thresholdsrc 0 --workers 64 --joined-dictionary \
  
max_update=$3
python train.py /cache/data_dir/bin/  \
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
python generate.py $DATA/ \
--gen-subset test \
--path  /cache/model_dir/averaged_model.pt \
--beam 5 --batch-size 64 --remove-bpe|tee generate.out

grep ^T generate.out | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref

grep ^H generate.out |cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys

python score.py --sys generate.sys --ref generate.ref
cp generate.* /cache/model_dir