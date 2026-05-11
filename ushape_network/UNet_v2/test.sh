python -W ignore Test.py --data_root dataset --modal WL --model unet_v2 --pth_path outputs/UNetV2/WL/checkpoints/best.pth --save_pred True
python -W ignore Test.py --data_root dataset --modal NBI --model unet_v2 --pth_path outputs/UNetV2/NBI/checkpoints/best.pth --save_pred True
