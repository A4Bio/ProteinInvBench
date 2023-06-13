import shutil
import os
from tqdm import tqdm

method_name = [ 'PiFold', 'StructGNN', 'GraphTrans', 'AlphaDesign', 'ProteinMPNN', 'GVP', 'GCA']



# 将文件从源路径复制到目标路径
src_path = '/gaozhangyang/experiments/OpenCPD/results/MPNN'
dst_path = '/gaozhangyang/experiments/OpenCPD/checkpoints/PDB'
for name in tqdm(method_name):
    tag = ''
    if 'KWDesign' not in name:
        shutil.copy(os.path.join(src_path, name+tag, 'checkpoint.pth'), 
                    os.path.join(dst_path, name, 'checkpoint.pth'))
        shutil.copy(os.path.join(src_path, name+tag, 'log.log'), 
                    os.path.join(dst_path, name, 'log.log'))
        shutil.copy(os.path.join(src_path, name+tag, 'model_param.json'), 
                    os.path.join(dst_path, name, 'model_param.json'))
        shutil.copy(os.path.join(src_path, name+tag, 'results_casp15.pt'), 
                    os.path.join(dst_path, name, 'results_casp15.pt'))
        shutil.copy(os.path.join(src_path, name+tag, 'results.pt'), 
                    os.path.join(dst_path, name, 'results.pt'))
    else:
        shutil.copy(os.path.join(src_path, name+tag, 'checkpoints', 'msa2_recycle3_epoch4.pth'), 
                    os.path.join(dst_path, name, 'checkpoint.pth'))
        shutil.copy(os.path.join(src_path, name+tag, 'log.log'), 
                    os.path.join(dst_path, name, 'log.log'))
        shutil.copy(os.path.join(src_path, name+tag, 'model_param.json'), 
                    os.path.join(dst_path, name, 'model_param.json'))
        shutil.copy(os.path.join(src_path, name+tag, 'results_casp15.pt'), 
                    os.path.join(dst_path, name, 'results_casp15.pt'))
        shutil.copy(os.path.join(src_path, name+tag, 'results.pt'), 
                    os.path.join(dst_path, name, 'results.pt'))
