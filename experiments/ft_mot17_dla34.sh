cd src
python train.py mot --exp_id ablation_2 --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/mot17.json' --gpus '0,2' --batch_size 12 --num_workers 12
cd ..