# Introduction
This repository is for **Image Captioning via Proximal Policy Optimization**.

It is based on [JDAI-CV / image-captioning](https://github.com/JDAI-CV/image-captioning). 

Please follow the same data preparation as in the repository above.
Basically PPO ([Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)) is used instead of the generally adopted 
[self-critical training](https://arxiv.org/abs/1612.00563).
The point of using PPO is its capability of enforcing trust-region constraints.
So there can be separate models for the predictor and the trainer.
The trainer is allowed the opportunity to observe various enough images and feedbacks
of CIDEr scores before substituting the predictor for generating training trajectories.
Also, in the gradient estimator of the algorithm, a word-level baseline via Monte-Carlo estimation
is implemented for replacing the sentence-level baseline, for which all words have the same baseline value. 

A pre-trained model can be downloaded [here](https://drive.google.com/file/d/1XR_Sf3c1M0UNdyMx1fWJm3g4sFYWFjnD/view?usp=sharing). 

(This model achieves a CIDEr score of
133.3% on the MSCOCO Karpathy test set, for which X-Transformer in [JDAI-CV / image-captioning](https://github.com/JDAI-CV/image-captioning) obtains 132.8%.)

## Training

The training is nearly the same as in [JDAI-CV / image-captioning](https://github.com/JDAI-CV/image-captioning).
However, we directly fine-tune over a pre-trained X-Transformer [model](https://drive.google.com/file/d/1a7aINHtpQbIw5JbAc4yvC7I1V-tQSdzb/view)
for the sake of computing resources.

### Train using PPO
Copy the pre-trained X-Transformer model into experiments/xtransformer_rl/snapshot and run the script
```
bash experiments/xtransformer_rl/train.sh
```

## Evaluation
```
CUDA_VISIBLE_DEVICES=0 python3 main_test.py --folder experiments/xtransformer_rl --resume 23
```
where the number 23 is since the checkpoint with the best performance on the Karpathy validation set
is saved as `caption_model_23.pth`.

## Acknowledgements
Thanks the contribution of [JDAI-CV / image-captioning](https://github.com/JDAI-CV/image-captioning) and 
[self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch).
