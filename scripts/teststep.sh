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
checkpoint=$8

IFS=","
read -ra TEST <<< "$3"

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi


prep=/cache/data_dir/$src-$tgt
tmp=$prep/tmp
orig=/cache/data_dir

mkdir -p $orig $tmp $prep

echo "-----------pre-processing data----------------------------------"

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


echo "apply truecase_model"



for l in $src ; do
    for f in "${TEST[@]}"; do
        cat $orig/$f.$l | \
            perl $TRUECASE -model $orig/$4 < $tmp/test.tags.$lang.tok_b.$l > $tmp/test.tags.$lang.tok.$l
    done
done


mv $tmp/test.tags.$lang.tok_b.$tgt $tmp/test.tags.$lang.tok.$tgt


BPE_CODE=$orig/$7

for L in $src $tgt; do
    for f in test.tags.$lang.tok.$L ; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

echo "bpe youwanle"
for L in $src $tgt; do
    cp $tmp/bpe.test.tags.$lang.tok.$L $prep/test.$L
done

echo "prepare finish"

cd /cache/code_dir
pip install --upgrade torch torchvision
pip install sacrebleu
pip install editdistance
python setup.py build develop --user
TEXT=/cache/data_dir/$src-$tgt
pip install sacrebleu
pip install editdistance
python fairseq_cli/preprocess.py --source-lang $src --target-lang $tgt \
  --testpref $TEXT/test --srcdict $orig/$5 --tgtdict $orig/$6 \
  --destdir /cache/data_dir/bin/ --thresholdtgt 0 --thresholdsrc 0 --workers 50  \


DATA=/cache/data_dir/bin/
OUTPUT=/cache/code_dir/output

echo "-------------------------------------translate the test set----------------------------------------------------"
#--user-dir /cache/code_dir/fairseq/models/dec_transformer_new 
# Evaluate 
python fairseq_cli/generate.py $DATA \
--gen-subset test --unnormalized \
--path  /cache/model_dir/$checkpoint \
--beam 5 --batch-size 64 --remove-bpe|tee /cache/data_dir/generate.out

cd /cache/data_dir/
grep ^T generate.out | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref

grep ^H generate.out | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys

cd /cache/code_dir
python fairseq_cli/score.py --sys /cache/data_dir/generate.sys --ref /cache/data_dir/generate.ref

python fairseq_cli/sacrebleu_wmt.py --input /cache/data_dir/generate.sys  /cache/data_dir/generate.ref