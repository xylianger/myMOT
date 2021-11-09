cd src
python train.py mot --exp_id mix_activate --load_model '../models/fairmot_dla34.pth' --data_cfg '../src/lib/cfg/data.json' --gpus '1' --batch_size 16 --num_workers 16
cd ..