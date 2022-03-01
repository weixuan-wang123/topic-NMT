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
read -ra CORPORA <<< "$3"
read -ra DEV <<< "$4"
read -ra TEST <<< "$5"

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
pip install sacrebleu
pip install editdistance
python setup.py build develop --user
TEXT=/cache/data_dir/$src-$tgt
src=$1
tgt=$2
python fairseq_cli/preprocess.py --source-lang $src --target-lang $tgt --dataset-impl lazy \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir /cache/data_dir/bin/ --thresholdtgt 0 --thresholdsrc 0 --workers 64 --joined-dictionary \
  