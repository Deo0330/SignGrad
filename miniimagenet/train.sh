pip install einops
python main.py --data "./split_mini/"  --seed 1111 --batch_size 64 --workers  1 --model r50 --lr 0.1 --alg sgd
python main.py --data "./split_mini/"  --seed 1111 --batch_size 64 --workers  1 --model r50 --lr 0.001 --alg adam
python main.py --data "./split_mini/"  --seed 1111 --batch_size 64 --workers  1 --model r50 --lr 0.001 --alg diffgrad
python main.py --data "./split_mini/"  --seed 1111 --batch_size 64 --workers  1 --model r50 --lr 0.001 --alg tanangulargrad
python main.py --data "./split_mini/"  --seed 1111 --batch_size 64 --workers  1 --model r50 --lr 0.001 --alg signgrad
python main.py --data "./split_mini/"  --seed 1111 --batch_size 64 --workers  1 --model r50 --lr 0.001 --alg adamp
python main.py --data "./split_mini/"  --seed 1111 --batch_size 64 --workers  1 --model r50 --lr 0.001 --alg signadamp
python main.py --data "./split_mini/"  --seed 1111 --batch_size 64 --workers  1 --model r50 --lr 0.001 --alg adamw
python main.py --data "./split_mini/"  --seed 1111 --batch_size 64 --workers  1 --model r50 --lr 0.001 --alg signadamw

python main.py --data "./split_mini/"  --seed 1111 --batch_size 128 --workers  2 --model vit --lr 0.001 --alg diffgrad
python main.py --data "./split_mini/"  --seed 1111 --batch_size 128 --workers 2 --model vit --lr 0.001 --alg signgrad
python main.py --data "./split_mini/"  --seed 1111 --batch_size 128 --workers 2 --model vit --lr 0.001 --alg adam
python main.py --data "./split_mini/"  --seed 1111 --batch_size 128 --workers 2 --model vit --lr 0.001 --alg tanangulargrad
python main.py --data "./split_mini/"  --seed 1111 --batch_size 128 --workers 2 --model vit --lr 0.1 --alg sgd
python main.py --data "./split_mini/"  --seed 1111 --batch_size 128 --workers 2 --model vit --lr 0.001 --alg adamp
python main.py --data "./split_mini/"  --seed 1111 --batch_size 128 --workers 2 --model vit --lr 0.001 --alg adamw
python main.py --data "./split_mini/"  --seed 1111 --batch_size 128 --workers 2 --model vit --lr 0.001 --alg signadamp
python main.py --data "./split_mini/"  --seed 1111 --batch_size 128 --workers 2 --model vit --lr 0.001 --alg signadamw