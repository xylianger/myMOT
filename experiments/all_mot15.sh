#pretrained model crowdhuman_dla34:  train dataset : mot17
#val:mot17
cd src
python train.py mot --exp_id mix_ch_cls --load_model '../models/crowdhuman_dla34.pth' --data_cfg '../src/lib/cfg/data.json' --gpus '0,1' --batch_size 12 --num_workers 12
cd ..