
# SignGrad: Adapting Step Size by Sign Change between Gradients

Official PyTorch implementation of SignGrad: Adapting Step Size by Sign Change between Gradients
# AngularGrad Optimizer

This repository contains the oficial implementation for [SignGrad: Adapting Step Size by Sign Change between Gradients]( todo paper link) in PyTorch.

SignGrad introduces the sign change between the current and the most recent gradient to adjust the step size of parameter update. In addition, our algorithm could also been as a plugin into the AdamW and AdamP optimizers.

You can import the optimizer as follows:
```python
from optims.SignGrad import SignGrad

...
the code of SignAdamW and SignAdamP also provided:
```python
from optims.SignAdamW import SignAdamW
from optims.SignAdamP import SignAdamP
...
model = YourModel()
optimizer = SignGrad(model.parameters())
...
for input, output in data:
  optimizer.zero_grad()
  loss = loss_function(output, model(input))
  loss.backward()
  optimizer.step()
...
```


If you have questions or suggestions, please feel free to open an issue. Please cite as:
```
(todo cite)
```
<p align="center">
<img src="figs/Rosenbrock.png" width="1000" align="center"> 
</p>

## Experiments

Experiments in the paper:

CIFAR-10/100
```
Example:
cd cifar/
python main.py --dataset cifar10 --model r50 --alg signgrad --lr 1e-3 --bs 128 --manualSeed 1  --epochs 100
```

Mini-ImageNet:
```
cd miniimagenet/
wget URL dataset to ./split_mini/
Example:
python main.py --data "./split_mini/"  --seed 1111 --batch_size 64 --workers  1 --model r50 --lr 0.001 --alg signgrad
```
Object Detection:
``` 
pip install mmdet
#pip install instaboostfast
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
``` 
``` 
cd mmdetection-master/
wget URL datasets to ./data/
python -u tools/train.py  "./configs/pascal_voc/signgradfig.py" --seed 1 --work-dir ./signgradlr1e_4eps8_bs2_numworker1
``` 

Image Retrieval:
``` 
cd Proxy-Anchor/
wget URL datasets to ./data/
cd ./code/
Example:
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --embedding-size 512 --batch-size 90 --lr 1e-4 --dataset cars --warm 1 --bn-freeze 1 --lr-decay-step 20 --optimizer signgrad --workers 1 --weight-decay 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --optimizer signgrad  --embedding-size 512 --batch-size 180 --lr 1e-4 --dataset cub --warm 1 --bn-freeze 1 --lr-decay-step 10 --workers 0
```
The experimental code is modified on the basis of the following:
https://github.com/mhaut/AngularGrad
https://github.com/yuanwei2019/EAdam-optimizer
https://github.com/tjddus9597/Proxy-Anchor-CVPR2020