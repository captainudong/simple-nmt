# Simple Neural Machine Translation (Simple-NMT)

김기현님의 코드를 기반으로 Transformer를 활용한 기계번역 실습 내용을 담았습니다.  
AWS ec2 t2.large(ubuntu22.04LTS, 스토리지 30GiB) 인스턴스를 활용했습니다.  
메모리 부족으로 스왑 메모리를 할당하였으며 해당 코드는 아래와 같습니다.
```bash
$ sudo dd if=/dev/zero of=/swapfile bs=128M count=16
$ sudo chmod 600 /swapfile
$ sudo mkswap /swapfile
$ sudo swapon /swapfile
$ sudo swapon -s
$ cd /
$ sudo nano /etc/fstab # 마지막에 다음과 같이 줄을 추가하고 Ctrl+X를 눌러 저장해주세요. /swapfile swap swap defaults 0 0
$ free # 잘 할당되었는지 확인
```

## Setup

```bash
$ git clone https://github.com/captainudong/simple-nmt.git
```

TorchText는 0.5 혹은 0.6 버전 설치를 권장하며 Mecab 설치 코드를 새로 작성했습니다.
아래와 같이 셸 스크립트를 실행하여 필요한 모듈을 모두 설치할 수 있습니다.
```bash
$ cd ~/simple-nmt/data
$ bash pdw_setup.sh
$ bash pdw_install_mecab.sh
```

## Data Pre-Processing
AI-Hub의 한국어-영어 병렬 데이터를 사용하였으며 1_구어체(1) 1개의 파일만 사용했습니다.  
excel파일을 tsv파일(탭으로 분리한 txt파일, 인코딩 utf-8)로 저장하여 로컬->ec2로 전송했습니다.
학습 시간이 오래 걸려 총 12,000개의 데이터만 사용하였고 Train, Valid, Test 데이터를 각각 8,000개, 2,000개 2,000개로 나눴습니다.
```bash
$ head -n 12000 1_구어체\(1\).txt > corpus.tsv
$ shuf ./corpus.tsv > corpus.shuf.tsv
$ head -n 8000 corpus.shuf.tsv > corpus.shuf.train.tsv
$ tail -n 4000 corpus.shuf.tsv | head -n 2000 > corpus.shuf.valid.tsv
$ tail -n 2000 corpus.shuf.tsv > corpus.shuf.test.tsv

$ cut -f1 corpus.shuf.train.tsv > corpus.shuf.train.ko; cut -f2 corpus.shuf.train.tsv > corpus.shuf.train.en
$ cut -f1 corpus.shuf.valid.tsv > corpus.shuf.valid.ko; cut -f2 corpus.shuf.valid.tsv > corpus.shuf.valid.en
$ cut -f1 corpus.shuf.test.tsv > corpus.shuf.test.ko; cut -f2 corpus.shuf.test.tsv > corpus.shuf.test.en
```

### Tokenize
기존의 tokenizer.py 파일로 tokenize를 실행한 결과 특수문자가 &apos와 같이 변환되는 이슈가 있어 pdw_tokenizer.py 파일을 새로 작성했습니다.
```bash
$ cat ./corpus.shuf.train.ko | mecab -O wakati | python3 ./post_tokenize.py ./corpus.shuf.train.ko > ./corpus.shuf.train.tok.ko
$ cat ./corpus.shuf.valid.ko | mecab -O wakati | python3 ./post_tokenize.py ./corpus.shuf.valid.ko > ./corpus.shuf.valid.tok.ko
$ cat ./corpus.shuf.test.ko | mecab -O wakati | python3 ./post_tokenize.py ./corpus.shuf.test.ko > ./corpus.shuf.test.tok.ko

$ cat ./corpus.shuf.train.en | python3 ./pdw_tokenizer.py | python3 ./post_tokenize.py ./corpus.shuf.train.en > ./corpus.shuf.train.tok.en
$ cat ./corpus.shuf.valid.en | python3 ./pdw_tokenizer.py | python3 ./post_tokenize.py ./corpus.shuf.valid.en > ./corpus.shuf.valid.tok.en
$ cat ./corpus.shuf.test.en | python3 ./pdw_tokenizer.py | python3 ./post_tokenize.py ./corpus.shuf.test.en > ./corpus.shuf.test.tok.en
```

### Subword Segmentation
```bash
$ python3 ./subword-nmt/learn_bpe.py --input ./corpus.shuf.train.tok.en --output bpe.en.model --symbols 20000 --verbose
$ python3 ./subword-nmt/learn_bpe.py --input ./corpus.shuf.train.tok.ko --output bpe.ko.model --symbols 30000 --verbose

$ cat ./corpus.shuf.train.tok.ko | python3 subword-nmt/apply_bpe.py -c ./bpe.ko.model > ./corpus.shuf.train.tok.bpe.ko
$ cat ./corpus.shuf.valid.tok.ko | python3 subword-nmt/apply_bpe.py -c ./bpe.ko.model > ./corpus.shuf.valid.tok.bpe.ko
$ cat ./corpus.shuf.test.tok.ko | python3 subword-nmt/apply_bpe.py -c ./bpe.ko.model > ./corpus.shuf.test.tok.bpe.ko

$ cat ./corpus.shuf.train.tok.en | python3 subword-nmt/apply_bpe.py -c ./bpe.en.model > ./corpus.shuf.train.tok.bpe.en
$ cat ./corpus.shuf.valid.tok.en | python3 subword-nmt/apply_bpe.py -c ./bpe.en.model > ./corpus.shuf.valid.tok.bpe.en
$ cat ./corpus.shuf.test.tok.en | python3 subword-nmt/apply_bpe.py -c ./bpe.en.model > ./corpus.shuf.test.tok.bpe.en
```


## Training
CrossEntropyLoss가 번역 품질을 정확하게 나타내지 않기 때문에 기존의 trainer.py 파일은 모든 에포크마다 모델을 저장하도록 되어있습니다. 스토리지 부담을 줄이기 위해 203번째 줄 save_model()함수를 26epoch부터 저장하도록 수정했습니다.

### 사용법
```bash
>> python train.py -h
usage: train.py [-h] --model_fn MODEL_FN --train TRAIN --valid VALID --lang
                LANG [--gpu_id GPU_ID] [--off_autocast]
                [--batch_size BATCH_SIZE] [--n_epochs N_EPOCHS]
                [--verbose VERBOSE] [--init_epoch INIT_EPOCH]
                [--max_length MAX_LENGTH] [--dropout DROPOUT]
                [--word_vec_size WORD_VEC_SIZE] [--hidden_size HIDDEN_SIZE]
                [--n_layers N_LAYERS] [--max_grad_norm MAX_GRAD_NORM]
                [--iteration_per_update ITERATION_PER_UPDATE] [--lr LR]
                [--lr_step LR_STEP] [--lr_gamma LR_GAMMA]
                [--lr_decay_start LR_DECAY_START] [--use_adam] [--use_radam]
                [--rl_lr RL_LR] [--rl_n_samples RL_N_SAMPLES]
                [--rl_n_epochs RL_N_EPOCHS] [--rl_n_gram RL_N_GRAM]
                [--rl_reward RL_REWARD] [--use_transformer]
                [--n_splits N_SPLITS]

optional arguments:
  -h, --help            show this help message and exit
  --model_fn MODEL_FN   Model file name to save. Additional information would
                        be annotated to the file name.
  --train TRAIN         Training set file name except the extention. (ex:
                        train.en --> train)
  --valid VALID         Validation set file name except the extention. (ex:
                        valid.en --> valid)
  --lang LANG           Set of extention represents language pair. (ex: en +
                        ko --> enko)
  --gpu_id GPU_ID       GPU ID to train. Currently, GPU parallel is not
                        supported. -1 for CPU. Default=-1
  --off_autocast        Turn-off Automatic Mixed Precision (AMP), which speed-
                        up training.
  --batch_size BATCH_SIZE
                        Mini batch size for gradient descent. Default=32
  --n_epochs N_EPOCHS   Number of epochs to train. Default=20
  --verbose VERBOSE     VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE
                        = 0, 1, 2. Default=2
  --init_epoch INIT_EPOCH
                        Set initial epoch number, which can be useful in
                        continue training. Default=1
  --max_length MAX_LENGTH
                        Maximum length of the training sequence. Default=100
  --dropout DROPOUT     Dropout rate. Default=0.2
  --word_vec_size WORD_VEC_SIZE
                        Word embedding vector dimension. Default=512
  --hidden_size HIDDEN_SIZE
                        Hidden size of LSTM. Default=768
  --n_layers N_LAYERS   Number of layers in LSTM. Default=4
  --max_grad_norm MAX_GRAD_NORM
                        Threshold for gradient clipping. Default=5.0
  --iteration_per_update ITERATION_PER_UPDATE
                        Number of feed-forward iterations for one parameter
                        update. Default=1
  --lr LR               Initial learning rate. Default=1.0
  --lr_step LR_STEP     Number of epochs for each learning rate decay.
                        Default=1
  --lr_gamma LR_GAMMA   Learning rate decay rate. Default=0.5
  --lr_decay_start LR_DECAY_START
                        Learning rate decay start at. Default=10
  --use_adam            Use Adam as optimizer instead of SGD. Other lr
                        arguments should be changed.
  --use_radam           Use rectified Adam as optimizer. Other lr arguments
                        should be changed.
  --rl_lr RL_LR         Learning rate for reinforcement learning. Default=0.01
  --rl_n_samples RL_N_SAMPLES
                        Number of samples to get baseline. Default=1
  --rl_n_epochs RL_N_EPOCHS
                        Number of epochs for reinforcement learning.
                        Default=10
  --rl_n_gram RL_N_GRAM
                        Maximum number of tokens to calculate BLEU for
                        reinforcement learning. Default=6
  --rl_reward RL_REWARD
                        Method name to use as reward function for RL training.
                        Default=gleu
  --use_transformer     Set model architecture as Transformer.
  --n_splits N_SPLITS   Number of heads in multi-head attention in
                        Transformer. Default=8
```

### Example Usage

ec2 ssh 연결이 끊어져도 실행될 수 있도록 합니다.
```bash
$ nohup python3 train.py --train ./data/corpus.shuf.train.tok.bpe --valid ./data/corpus.shuf.valid.tok.bpe --lang koen --batch_size 128 --n_epochs 10 --max_length 30 --dropout .2 --hidden_size 768 --n_layers 4 --max_grad_norm 1e+8 --iteration_per_update 32 --lr 1e-3 --lr_step 0 --use_adam --use_transformer --rl_n_epochs 0 --model_fn ./model_koen.pth &
$ disown
```

진행현황을 실시간으로 보고 싶다면 아래와 같이 명령어를 입력하여 로그를 볼 수 있습니다.
```bash
$ tail -f ./nohup.out
```


학습 진행 중...
