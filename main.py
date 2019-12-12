import os

if __name__ == '__main__':
    for s in range(1,6):
        os.system(f'python pipeline.py --stride_size={s}')
        os.system(f'python train.py --model_name=base_stock_stride={s}')