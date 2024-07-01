import numpy as np
from skimage import measure
from skimage.filters import sobel
from PIL import Image
import os


def preprocess_ground_truth(img, target_size=(768, 768)):
    """
    预处理真实值图片：
    1. 裁剪为1080x1080，取中心部分
    2. 缩放为768x768
    """
    # 确保图像是RGB模式
    img = img.convert('RGB')
    
    # 裁剪为1080x1080，取中心部分
    width, height = img.size
    left = (width - 1080) // 2
    top = (height - 1080) // 2
    right = left + 1080
    bottom = top + 1080
    img_cropped = img.crop((left, top, right, bottom))
    
    # 缩放为768x768
    img_resized = img_cropped.resize(target_size, Image.LANCZOS)
    
    return img_resized


def load_images(directory, is_ground_truth=False):
    """
    从指定目录加载图像文件
    """
    images = []
    for i in range(120):
        filename = f"{i:03d}.png"  # 生成类似 000.png, 001.png, ..., 119.png 的文件名
        img_path = os.path.join(directory, filename)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            if is_ground_truth:
                img = preprocess_ground_truth(img)
            img = img.convert('L')  # 转换为灰度图像
            img_array = np.array(img) / 255.0  # 归一化到 0-1 范围
            images.append(img_array)
        else:
            raise FileNotFoundError(f"找不到文件: {img_path}")
    return images


class MetricGRAD:
    def __init__(self, sigma=1.4):
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
        self.filter_x = torch.from_numpy(self.filter_x).unsqueeze(0).cuda()
        self.filter_y = torch.from_numpy(self.filter_y).unsqueeze(0).cuda()
    
    def __call__(self, pred, true):
        true_grad = self.gauss_gradient(true)
        pred_grad = self.gauss_gradient(pred)
        return ((true_grad - pred_grad) ** 2).sum() / 1000
    
    def gauss_gradient(self, img):
        img_filtered_x = kornia.filters.filter2D(img[None, None, :, :], self.filter_x, border_type='replicate')[0, 0]
        img_filtered_y = kornia.filters.filter2D(img[None, None, :, :], self.filter_y, border_type='replicate')[0, 0]
        return (img_filtered_x**2 + img_filtered_y**2).sqrt()
    
    @staticmethod
    def gauss_filter(sigma, epsilon=1e-2):
        half_size = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = np.int(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = MetricGRAD.gaussian(i - half_size, sigma) * MetricGRAD.dgaussian(
                    j - half_size, sigma)

        # normalize filter
        norm = np.sqrt((filter_x**2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y
        
    @staticmethod
    def gaussian(x, sigma):
        return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    
    @staticmethod
    def dgaussian(x, sigma):
        return -x * MetricGRAD.gaussian(x, sigma) / sigma**2
    

class MetricDTSSD:
    def __call__(self, pred_t, pred_tm1, true_t, true_tm1):
        dtSSD = ((pred_t - pred_tm1) - (true_t - true_tm1)) ** 2
        dtSSD = dtSSD.sum() / true_t.numel()
        dtSSD = dtSSD.sqrt()
        return dtSSD * 1e2


def calculate_metrics(predicted_frames, ground_truth_frames):
    num_frames = len(predicted_frames)
        
    # 初始化指标
    mad = 0
    mse = 0
    grad = 0
    conn = 0
    dt_ssd = 0
    
    for i in range(num_frames):
        pred = predicted_frames[i]
        gt = ground_truth_frames[i]
        
        # 计算 MAD
        mad += np.abs(pred - gt).mean()
        # mad += np.mean(np.abs(pred - gt))
        
        # 计算 MSE
        mse += ((pred - gt) ** 2).mean()
        # mse += np.mean((pred - gt) ** 2)
        
        # 计算 Grad (空间梯度)
        grad += np.mean(np.abs(sobel(pred) - sobel(gt)))
        
        # 计算 Conn (连通性)
        pred_labeled = measure.label(pred)
        gt_labeled = measure.label(gt)
        conn = np.sum(np.abs(pred_labeled.max() - gt_labeled.max()))
        
        # 计算 dtSSD (时间一致性)
        if i > 0:
            dt_ssd += np.mean((pred - predicted_frames[i-1]) ** 2)
    
    # 计算平均值
    mad /= num_frames
    mse /= num_frames
    # grad /= num_frames
    # conn /= num_frames
    dt_ssd /= (num_frames - 1)

    mad *= 1000
    mse *= 1000
    dt_ssd *= 100
    
    return {
        'MAD': mad,
        'MSE': mse,
        'Grad': grad,
        'Conn': conn,
        'dtSSD': dt_ssd
    }

# 设置数据路径
predicted_dir = "icm/"
ground_truth_dir = "alpha/"

# 加载图像
try:
    predicted_frames = load_images(predicted_dir)
    ground_truth_frames = load_images(ground_truth_dir, is_ground_truth=True)
except FileNotFoundError as e:
    print(f"错误: {e}")
    exit(1)

# 确保加载了正确数量的帧
if len(predicted_frames) != 120 or len(ground_truth_frames) != 120:
    print(f"警告: 预期120帧，但加载了 {len(predicted_frames)} 个预测帧和 {len(ground_truth_frames)} 个真实帧")

# 计算指标
metrics = calculate_metrics(predicted_frames, ground_truth_frames)
print(metrics)