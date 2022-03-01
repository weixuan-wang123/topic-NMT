# Topic-NMT 
This repository hosts the original codes for our EMNLP 2021 Main Conference Paper: <a href="https://aclanthology.org/2021.emnlp-main.256/">Neural Machine Translation with Heterogeneous Topic Knowledge Embeddings</a>

## What is Topic-NMT 

We propose heterogeneous ways of incorporating topic information into the Transformer architecture. Specifically, the topic information can be incorporated in a heterogeneous manner, namely as pre-encoder topic embedding (`ENC_pre`), post-encoder topic embedding (`ENC_post`), and decoder topic embedding (`DEC`). Besides, the topic distribution learned for each word (as its topic embedding) is summarized at the sentence level and fed into the NMT model. The intuition is that aggregating topic distribution at the sentence level produces more accurate topic information than at the word level. This enables topic modeling to consider contexts conveyed in a sentence. Each target word is generated with the guidance of the topic information of both source and target sentences.

## Getting Started

### Requirements

This repository includes all the codes for Topic-NMT, referring to the following two sources:
<p><a href="https://github.com/pytorch/fairseq">fairseq</a></p>
<p><a href="https://github.com/adjidieng/ETM">ETM</a></p>

PyTorch version >= 1.5.0

Python version >= 3.6

Fairseq >= 0.9.0

### Method

Training the filtering model consists of two steps:
- training the ETM for topic embedding
- training the Topic-NMT model

#### Training the ETM 

To learn interpretable embeddings and topics using ETM on the 20NewsGroup dataset, run

```
python ETM/main.py --mode train --dataset 20ng --data_path data/20ng --num_topics 50 --train_embeddings 1 --epochs 1000
```
To evaluate perplexity on document completion, topic coherence, topic diversity, and visualize the topics/embeddings run

```
python ETM/main.py --mode eval --dataset 20ng --data_path data/20ng --num_topics 50 --train_embeddings 1 --tc 1 --td 1 --load_from CKPT_PATH
```
#### Training the Topic-NMT
After getting the checkpoint of ETM, you can use the topic embedding to train a Topic-NMT.

First, you should pre-process and binarize the raw data, run
```
fairseq-preprocess --source-lang en --target-lang de --trainpref $data_dir/train --validpref $data_dir/valid --testpref $data_dir/test --destdir $dest_dir  --thresholdtgt 0 --thresholdsrc 0 --workers 64 --joined-dictionary

```
Then, you can train a Topic-NMT model with the topic embedding, run
```
python train.py $data_dir/bin/ --source-lang en --target-lang de --arch ENC_pre --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000  --lr 0.0007 --min-lr 1e-09 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 --max-tokens  4096  --eval-bleu  --save-dir  $out_dir --update-freq 2 --no-progress-bar --save-interval-updates 1000 --keep-interval-updates 20 
```

Also note that the arch is specified in terms of the topic usage `(--arch)`, there are other possibe choices: `ENC_post`, `DEC`, `ENC_pre_ENC_post`, `ENC_pre_DEC`, `ENC_post_DEC`, `ENC_pre_ENC_post_DEC`.

Once your model is trained, you can generate translations, run

```
fairseq-generate --path $out_dir/checkpoint_best.pt --source-lang en --target-lang de $data_dir/bin/ --sacrebleu --quiet --skip-invalid-size-inputs-valid-test --remove-bpe --user-dir /faireseq-topic/fairseq/models/ENC_pre
```

# Citation
Please cite as:

```
@inproceedings{DBLP:conf/emnlp/WangPZL21,
  author    = {Weixuan Wang and
               Wei Peng and
               Meng Zhang and
               Qun Liu},
  title     = {Neural Machine Translation with Heterogeneous Topic Knowledge Embeddings},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2021, Virtual Event / Punta Cana, Dominican
               Republic, 7-11 November, 2021},
  pages     = {3197--3202},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://aclanthology.org/2021.emnlp-main.256}}
```
