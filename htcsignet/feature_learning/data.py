import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms as transforms
import pywt
from PIL import Image


class WaveletDecompose:
    """
    使用小波变换将图像分解为T个不同频率成分。
    
    小波分解会产生多个子带，对应不同的频率成分：
    - LL（近似系数）：低频成分
    - LH（水平细节）：水平高频
    - HL（垂直细节）：垂直高频  
    - HH（对角细节）：对角高频
    
    对于T个时间步，我们进行多级分解以获得足够的频率成分。
    """
    
    def __init__(self, T: int, wavelet: str = 'db1', mode: str = 'symmetric'):
        """
        Args:
            T: 时间步数/需要的频率成分数量
            wavelet: 小波类型 (如 'db1', 'haar', 'db2', 'coif1', 'sym2')
            mode: 边界扩展模式
        """
        self.T = T
        self.wavelet = wavelet
        self.mode = mode
        # 计算需要的分解级数
        self.levels = max(1, (T - 1) // 3 + 1)
    
    def __call__(self, img):
        """
        对输入图像进行小波分解
        
        Args:
            img: PIL Image 或 numpy array (H, W) 或 (C, H, W)
            
        Returns:
            list of numpy arrays: T个不同频率成分，每个与原图尺寸相同
        """
        # 转换为numpy数组
        if isinstance(img, Image.Image):
            img_array = np.array(img).astype(np.float32)
        elif isinstance(img, torch.Tensor):
            img_array = img.numpy().astype(np.float32)
        else:
            img_array = img.astype(np.float32)
        
        # 处理维度
        if len(img_array.shape) == 3:
            if img_array.shape[0] in [1, 3]:  # (C, H, W) 格式
                img_array = img_array.transpose(1, 2, 0)
            if img_array.shape[-1] == 3:  # RGB转灰度
                img_array = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
            elif img_array.shape[-1] == 1:
                img_array = img_array[:,:,0]
        
        original_shape = img_array.shape
        
        # 进行多级小波分解
        coeffs = pywt.wavedec2(img_array, self.wavelet, mode=self.mode, level=self.levels)
        
        # 收集所有频率成分
        frequency_components = []
        
        # coeffs格式: [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]
        # cAn是最低频的近似系数
        
        # 添加最低频近似系数
        cA = coeffs[0]
        cA_resized = self._resize_to_original(cA, original_shape)
        frequency_components.append(cA_resized)
        
        # 添加各级的细节系数 (从低频到高频)
        for level_detail in coeffs[1:]:
            cH, cV, cD = level_detail
            frequency_components.append(self._resize_to_original(cH, original_shape))
            frequency_components.append(self._resize_to_original(cV, original_shape))
            frequency_components.append(self._resize_to_original(cD, original_shape))
        
        # 如果成分数量超过T，截取前T个（从低频到高频）
        # 如果不足T个，循环复制或插值
        while len(frequency_components) < self.T:
            # 复制最后一个频率成分
            frequency_components.append(frequency_components[-1].copy())
        
        frequency_components = frequency_components[:self.T]
        
        # 归一化每个频率成分
        normalized_components = []
        for comp in frequency_components:
            comp_min = comp.min()
            comp_max = comp.max()
            if comp_max - comp_min > 1e-8:
                comp = (comp - comp_min) / (comp_max - comp_min)
            else:
                comp = np.zeros_like(comp)
            normalized_components.append(comp)
        
        return normalized_components
    
    def _resize_to_original(self, coeff, original_shape):
        """将小波系数调整为原始图像尺寸"""
        from scipy.ndimage import zoom
        
        if coeff.shape == original_shape:
            return coeff
        
        zoom_factors = (original_shape[0] / coeff.shape[0], 
                        original_shape[1] / coeff.shape[1])
        return zoom(coeff, zoom_factors, order=1)


class WaveletAugmentedTransform:
    """
    将小波分解的频率成分附加到图像上，为SNN不同时间步提供不同频率信息。
    
    输出格式: [T, C+1, H, W] - 每个时间步的图像包含原始通道+对应频率成分通道
    """
    
    def __init__(self, T: int, base_transform=None, wavelet: str = 'db1', 
                 append_mode: str = 'channel'):
        """
        Args:
            T: 时间步数
            base_transform: 应用于图像的基础变换 (如裁剪、缩放等)
            wavelet: 小波类型
            append_mode: 频率附加模式
                - 'channel': 将频率成分作为额外通道附加
                - 'add': 将频率成分加到原图上
                - 'multiply': 将频率成分乘到原图上
        """
        self.T = T
        self.base_transform = base_transform
        self.wavelet_decompose = WaveletDecompose(T, wavelet)
        self.append_mode = append_mode
    
    def __call__(self, img):
        """
        Args:
            img: 输入图像 (PIL Image, Tensor, 或 numpy array)
            
        Returns:
            tensor: [T, C+1, H, W] 如果 append_mode='channel'
                   [T, C, H, W] 如果 append_mode='add' 或 'multiply'
        """
        # 首先进行小波分解（在基础变换之前，使用原始图像）
        freq_components = self.wavelet_decompose(img)
        
        # 转换输入为PIL Image以统一处理
        if isinstance(img, torch.Tensor):
            if img.dim() == 2:
                img_pil = Image.fromarray((img.numpy() * 255).astype(np.uint8))
            else:
                img_pil = transforms.ToPILImage()(img)
        elif isinstance(img, np.ndarray):
            # 处理numpy数组的不同格式
            img_arr = img.copy()
            
            # 如果是 (C, H, W) 格式且 C=1，转换为 (H, W)
            if len(img_arr.shape) == 3:
                if img_arr.shape[0] == 1:
                    img_arr = img_arr.squeeze(0)  # (1, H, W) -> (H, W)
                elif img_arr.shape[0] in [3, 4]:
                    # (C, H, W) -> (H, W, C)
                    img_arr = img_arr.transpose(1, 2, 0)
                # 如果 shape[-1] == 1，也squeeze
                if len(img_arr.shape) == 3 and img_arr.shape[-1] == 1:
                    img_arr = img_arr.squeeze(-1)
            
            # 归一化到 0-255
            if img_arr.max() <= 1.0:
                img_arr = (img_arr * 255).astype(np.uint8)
            else:
                img_arr = img_arr.astype(np.uint8)
            
            img_pil = Image.fromarray(img_arr)
        else:
            img_pil = img  # 已经是PIL Image
        
        # 应用基础变换到原图（跳过ToPILImage如果已经是PIL）
        if self.base_transform is not None:
            # 过滤掉ToPILImage变换
            filtered_transforms = []
            for t in self.base_transform.transforms:
                if not isinstance(t, transforms.ToPILImage):
                    filtered_transforms.append(t)
            
            if filtered_transforms:
                img_transformed = img_pil
                for t in filtered_transforms:
                    img_transformed = t(img_transformed)
                if not isinstance(img_transformed, torch.Tensor):
                    img_transformed = transforms.ToTensor()(img_transformed)
            else:
                img_transformed = transforms.ToTensor()(img_pil)
        else:
            img_transformed = transforms.ToTensor()(img_pil)
        
        # img_transformed: [C, H, W]
        C, H, W = img_transformed.shape
        
        # 调整频率成分尺寸
        freq_tensors = []
        for freq_comp in freq_components:
            # 调整到相同的H, W
            freq_img = Image.fromarray((freq_comp * 255).astype(np.uint8))
            freq_img = freq_img.resize((W, H), Image.BILINEAR)
            freq_tensor = transforms.ToTensor()(freq_img)  # [1, H, W]
            freq_tensors.append(freq_tensor)
        
        # 构建每个时间步的输入
        time_step_inputs = []
        
        for t in range(self.T):
            if self.append_mode == 'channel':
                # 将频率成分作为额外通道附加 [C+1, H, W]
                combined = torch.cat([img_transformed, freq_tensors[t]], dim=0)
            elif self.append_mode == 'add':
                # 将频率成分加到原图上
                combined = img_transformed + 0.3 * freq_tensors[t].expand(C, -1, -1)
            elif self.append_mode == 'multiply':
                # 调制方式
                combined = img_transformed * (0.7 + 0.6 * freq_tensors[t].expand(C, -1, -1))
            else:
                raise ValueError(f"Unknown append_mode: {self.append_mode}")
            
            time_step_inputs.append(combined)
        
        # 堆叠为 [T, C', H, W]
        output = torch.stack(time_step_inputs, dim=0)
        
        return output


class WaveletTransformDataset(Dataset):
    """
    应用小波增强变换的Dataset，为SNN训练提供多时间步输入。
    """
    
    def __init__(self, dataset, T: int, base_transform=None, wavelet: str = 'db1',
                 append_mode: str = 'channel', transform_index: int = 0):
        """
        Args:
            dataset: 原始数据集
            T: 时间步数
            base_transform: 基础图像变换
            wavelet: 小波类型
            append_mode: 频率附加模式
            transform_index: 数据元组中图像的索引
        """
        self.dataset = dataset
        self.wavelet_transform = WaveletAugmentedTransform(
            T=T, 
            base_transform=base_transform, 
            wavelet=wavelet,
            append_mode=append_mode
        )
        self.transform_index = transform_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        img = data[self.transform_index]
        
        # 应用小波增强变换
        transformed_img = self.wavelet_transform(img)
        
        return tuple((transformed_img, *data[1:]))


class TransformDataset(Dataset):
    """
        Dataset that applies a transform on the data points on __get__item.
    """

    def __init__(self, dataset, transform, transform_index=0):
        self.dataset = dataset
        self.transform = transform
        self.transform_index = transform_index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        img = data[self.transform_index]

        return tuple((self.transform(img), *data[1:]))


def extract_features(x, base_model, batch_size, device, input_size=None):
    data = TensorDataset(torch.from_numpy(x))

    if input_size is not None:
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])

        data = TransformDataset(data, data_transforms)

    data_loader = DataLoader(data, batch_size=batch_size)
    result = []

    with torch.no_grad():
        for batch in data_loader:
            input = batch[0].to(device)
            #_, __, features = base_model(input)
            features = base_model(input)
            result.append(features)
            # result.append(process_function(batch))
    return torch.cat(result).cpu().numpy()
