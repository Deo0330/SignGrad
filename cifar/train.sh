#install for vit
#pip install einops
python main.py --dataset cifar10 --model r50 --alg adam --lr 1e-3 --bs 128 --manualSeed 1  --epochs 100
python main.py --dataset cifar10 --model r50 --alg sgd --lr 1e-1 --bs 128 --manualSeed 1  --epochs 100
python main.py --dataset cifar10 --model r50 --alg diffgrad --lr 1e-3 --bs 128 --manualSeed 1  --epochs 100
python main.py --dataset cifar10 --model r50 --alg tanangulargrad --lr 1e-3 --bs 128 --manualSeed 1  --epochs 100
python main.py --dataset cifar10 --model r50 --alg signgrad --lr 1e-3 --bs 128 --manualSeed 1  --epochs 100
python main.py --dataset cifar10 --model r50 --alg adamw --lr 1e-3 --bs 128 --manualSeed 1  --epochs 100 --weight_decay 0.05
python main.py --dataset cifar10 --model r50 --alg signadamw --lr 1e-3 --bs 128 --manualSeed 1  --epochs 100 --weight_decay 0.05
python main.py --dataset cifar10 --model r50 --alg adamp --lr 1e-3 --bs 128 --manualSeed 1  --epochs 100
python main.py --dataset cifar10 --model r50 --alg signadamp --lr 1e-3 --bs 128 --manualSeed 1  --epochs 100
