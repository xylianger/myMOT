cd src
python train.py mot --exp_id mix_act_sphere --load_model '../models/fairmot_dla34.pth' --data_cfg '../src/lib/cfg/data.json' --gpus '0,2' --batch_size 32 --num_workers 16
cd ..