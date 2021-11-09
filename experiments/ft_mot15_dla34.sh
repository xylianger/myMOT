cd src
python train.py mot --exp_id ft_mot15_dla34 --gpus 2 --batch_size 16  --data_cfg '../src/lib/cfg/mot15.json' --num_epochs 30 --lr 1e-4 --num_workers 16
cd ..
