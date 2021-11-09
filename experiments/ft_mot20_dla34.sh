cd src
python train.py mot --exp_id ft_mot20_dla34 --gpus 0,1,2 --batch_size 8 --num_workers 8 --load_model '../models/ctdet_coco_dla_2x.pth' --data_cfg '../src/lib/cfg/mot20.json'  --lr_step '3,5,25'  --K 250 --lr 1e-3
cd ..