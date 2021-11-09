cd src
python train.py mot --exp_id ft_mot16_dla34 --gpus 0,1,2 --batch_size 8 --num_workers 8 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/mot16.json' --lr 1e-3 --lr_step '2,4,25'
cd ..