# dual-path-RNNs-DPRNNs-based-speech-separation
A PyTorch implementation of dual-path RNNs (DPRNNs) based speech separation on wsj0-2mix described in the amazing paper "Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation".

This implementation is based on https://github.com/kaituoxu/Conv-TasNet and https://github.com/yluo42/TAC, thanks Kaituo and Yi Luo for sharing.

Workflow:

step 1: generate jason files with wav path and length

./preprocess.py --in-dir /home/zm/deepseparation/wsj0_wav/2speakers/wav8k/min --out-dir data --sample-rate 8000

step 2: train

CUDA_VISIBLE_DEVICES=0 python train.py --train_dir data/tr --valid_dir data/cv --sample_rate 8000 --segment 4 --cv_maxlen 6 --W 2 --N 64 --K 250 --D 6 --C 2 --E 256 --H 128 --norm_type gLN --causal 0 --mask_nonlinear relu --use_cuda 1 --epochs 100 --half_lr 1 --early_stop 0 --max_norm 5 --shuffle 1 --batch_size 1 --optimizer adam --lr 1e-3 --momentum 0 --l2 0 --save_folder exp/ --checkpoint 1 --continue_from "" --print_freq 1000

step 3: separate the tt data

CUDA_VISIBLE_DEVICES=0 python separate.py --model_path exp/temp_best.pth.tar --mix_json data/tt/mix.json --out_dir exp/separate --use_cuda 1 --sample_rate 8000 --batch_size 2

Results:

We obtain SDRi 19.1017dB on wsj0-2mix with the trained model exp/temp_best.pth.tar

If you find this code is useful, please kindly cite our following new research work on speech separation based on this code. LaFurca achieved 20.55dB SDR improvement, 20.35dB SI-SDR improvement, 3.69 of PESQ, and 94.86% of ESTOI on WSJ-2mix dataset.
@article{shi2020furca,
  title={LaFurca: Iterative Multi-Stage Refined End-to-End Monaural Speech Separation Based on Context-Aware Dual-Path Deep Parallel Inter-Intra Bi-LSTM},
  author={Shi, Ziqiang and Liu, Rujie and Han, Jiqing},
  journal={arXiv preprint arXiv:2001.08998},
  year={2020}
}
