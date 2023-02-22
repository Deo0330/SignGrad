# cars
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --embedding-size 512 --batch-size 90 --lr 1e-4 --dataset cars --warm 1 --bn-freeze 1 --lr-decay-step 20 --optimizer signadamp --workers 1 --weight-decay 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --embedding-size 512 --batch-size 90 --lr 1e-4 --dataset cars --warm 1 --bn-freeze 1 --lr-decay-step 20 --optimizer adamp --workers 1 --weight-decay 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --embedding-size 512 --batch-size 90 --lr 1e-4 --dataset cars --warm 1 --bn-freeze 1 --lr-decay-step 20 --optimizer signadamw --workers 1 --weight-decay 0.05
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --embedding-size 512 --batch-size 90 --lr 1e-4 --dataset cars --warm 1 --bn-freeze 1 --lr-decay-step 20 --optimizer adamw --workers 1 --weight-decay 0.05
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --embedding-size 512 --batch-size 90 --lr 1e-4 --dataset cars --warm 1 --bn-freeze 1 --lr-decay-step 20 --optimizer sgd --workers 1 --weight-decay 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --embedding-size 512 --batch-size 90 --lr 1e-4 --dataset cars --warm 1 --bn-freeze 1 --lr-decay-step 20 --optimizer adam --workers 1 --weight-decay 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --embedding-size 512 --batch-size 90 --lr 1e-4 --dataset cars --warm 1 --bn-freeze 1 --lr-decay-step 20 --optimizer diffgrad --workers 1 --weight-decay 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --embedding-size 512 --batch-size 90 --lr 1e-4 --dataset cars --warm 1 --bn-freeze 1 --lr-decay-step 20 --optimizer tanangulargrad --workers 1 --weight-decay 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --embedding-size 512 --batch-size 90 --lr 1e-4 --dataset cars --warm 1 --bn-freeze 1 --lr-decay-step 20 --optimizer signgrad --workers 1 --weight-decay 0
# cub
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --optimizer signadamw --embedding-size 512 --batch-size 180 --lr 1e-4 --dataset cub --warm 1 --bn-freeze 1 --lr-decay-step 10 --workers 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --optimizer adamw --embedding-size 512 --batch-size 180 --lr 1e-4 --dataset cub --warm 1 --bn-freeze 1 --lr-decay-step 10 --workers 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --optimizer signadamp --embedding-size 512 --batch-size 180 --lr 1e-4 --dataset cub --warm 1 --bn-freeze 1 --lr-decay-step 10 --workers 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --optimizer adamwp --embedding-size 512 --batch-size 180 --lr 1e-4 --dataset cub --warm 1 --bn-freeze 1 --lr-decay-step 10 --workers 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --optimizer adam --embedding-size 512 --batch-size 180 --lr 1e-4 --dataset cub --warm 1 --bn-freeze 1 --lr-decay-step 10 --workers 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --optimizer sgd --embedding-size 512 --batch-size 180 --lr 1e-4 --dataset cub --warm 1 --bn-freeze 1 --lr-decay-step 10 --workers 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --optimizer diffgrad  --embedding-size 512 --batch-size 180 --lr 1e-4 --dataset cub --warm 1 --bn-freeze 1 --lr-decay-step 10 --workers 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --optimizer tanangulargrad--embedding-size 512 --batch-size 180 --lr 1e-4 --dataset cub --warm 1 --bn-freeze 1 --lr-decay-step 10 --workers 0
python train.py --gpu-id 0 --loss Proxy_Anchor --model bn_inception --optimizer signgrad  --embedding-size 512 --batch-size 180 --lr 1e-4 --dataset cub --warm 1 --bn-freeze 1 --lr-decay-step 10 --workers 0