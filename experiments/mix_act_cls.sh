cd src
python train.py mot --exp_id mix_ch_cls --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/data.json' --gpus '0,1' --batch_size 12 --num_workers 12
cd ..
#没加 acticate