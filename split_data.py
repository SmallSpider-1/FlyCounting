import os
import shutil
import random

# ================= 配置参数 =================
# 原始数据集所在路径 (请确保路径正确)
src_dir = 'data'  
# 划分后的数据集保存路径
out_dir = 'dataset' 

# 划分比例: 训练集 80%, 验证集 10%, 测试集 10%
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
# ============================================

def split_dataset():
    # 1. 获取所有图片的文件名（不带后缀），确保图片和标签一一对应
    all_files = os.listdir(src_dir)
    # 筛选出所有的jpg文件，并去掉后缀拿到基础名称 (例如 '000001')
    base_names = [f.split('.jpg')[0] for f in all_files if f.endswith('.jpg')]
    
    # 2. 随机打乱数据，保证模型训练的泛化能力
    random.seed(42) # 设置随机种子，保证每次运行结果一致
    random.shuffle(base_names)
    
    # 3. 计算划分的索引点
    total_count = len(base_names)
    train_point = int(total_count * train_ratio)
    val_point = int(total_count * (train_ratio + val_ratio))
    
    # 4. 划分出三个列表
    train_names = base_names[:train_point]
    val_names = base_names[train_point:val_point]
    test_names = base_names[val_point:]
    
    # 定义一个内部函数用来创建文件夹和复制文件
    def copy_files(names, split_type):
        # 创建标准的 YOLO 目录: images/train, labels/train 等
        img_out_dir = os.path.join(out_dir, 'images', split_type)
        txt_out_dir = os.path.join(out_dir, 'labels', split_type)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(txt_out_dir, exist_ok=True)
        
        count = 0
        for name in names:
            img_src = os.path.join(src_dir, f"{name}.jpg")
            txt_src = os.path.join(src_dir, f"{name}.txt")
            
            # 确保图片和txt文件都存在才进行复制
            if os.path.exists(img_src) and os.path.exists(txt_src):
                shutil.copy(img_src, os.path.join(img_out_dir, f"{name}.jpg"))
                shutil.copy(txt_src, os.path.join(txt_out_dir, f"{name}.txt"))
                count += 1
            else:
                print(f"警告: 找不到对应的文件 {name}.jpg 或 {name}.txt")
                
        return count

    # 5. 执行复制操作
    print(f"总计找到 {total_count} 对有效数据。开始划分...")
    train_count = copy_files(train_names, 'train')
    val_count = copy_files(val_names, 'val')
    test_count = copy_files(test_names, 'test')
    
    # 6. 打印结果
    print("划分完成！")
    print(f"训练集 (Train): {train_count} 张")
    print(f"验证集 (Val): {val_count} 张")
    print(f"测试集 (Test): {test_count} 张")
    print(f"符合 YOLO 格式的数据集已保存在: {os.path.abspath(out_dir)}")

if __name__ == '__main__':
    split_dataset()