import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import copy

import torch
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform


class TransformVisualizer:
    def __init__(self, image_path: str, label_path: str):
        """
        初始化可视化工具

        Args:
            image_path: CT图像文件路径
            label_path: 标签文件路径
        """
        self.image_path = image_path
        self.label_path = label_path
        self.original_image = None
        self.original_label = None
        self.transforms = []

        # 加载原始数据
        self.load_data()

        # 设置transforms
        self.setup_transforms()

    def load_data(self):
        """加载NIfTI格式的图像和标签"""
        try:
            # 加载图像
            img_nii = nib.load(self.image_path)
            self.original_image = torch.from_numpy(img_nii.get_fdata()).to(torch.float32)

            # 加载标签
            label_nii = nib.load(self.label_path)
            self.original_label = torch.from_numpy(label_nii.get_fdata()).to(torch.float32)

            print(f"图像形状: {self.original_image.shape}")
            print(f"标签形状: {self.original_label.shape}")

            # 标准化图像数据
            # self.original_image = self.normalize_image(self.original_image)

        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise

    def normalize_image(self, image):
        """标准化图像数据"""
        # 简单的z-score标准化
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            return (image - mean) / std
        return image

    def setup_transforms(self):
        """设置transforms列表"""
        self.transforms = [
            ("SpatialTransform", SpatialTransform(
                patch_size=[512, 350, 512],
                patch_center_dist_from_border=[0, 0, 0],
                random_crop=False,
                p_elastic_deform=0,
                elastic_deform_scale=(0, 0.2),
                elastic_deform_magnitude=(0, 0.2),
                p_rotation=0.2,
                rotation=(-0.5235987755982988, 0.5235987755982988),
                p_scaling=0.2,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                p_synchronize_def_scale_across_axes=0,
                bg_style_seg_sampling=False,
                mode_seg='bilinear',
                border_mode_seg='zeros',
                center_deformation=True,
                padding_mode_image='zeros'
            )),

            ("GaussianNoise", RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=1
        )),

            ("GaussianBlur", RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=1
        )),

            ("MultiplicativeBrightness", RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=1
        )),

            ("Contrast", RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=1
        )),

            ("SimulateLowResolution", RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=1
        )),

            ("Gamma_Invert", RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=1
        )),

            ("Gamma_Normal", RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=1
        )),

            ("Mirror", MirrorTransform(allowed_axes=(0, 1, 2))),

            ("RemoveLabel", RemoveLabelTansform(
                segmentation_channels=None,
                label_value=-1,
                set_to=0
            ))
        ]

    def prepare_data_for_transform(self, image, label):
        """为transform准备数据格式"""
        # batchgeneratorsv2期望的数据格式
        data_dict = {
            'image': image[None, ...],  # 添加channel维度
            'seg': label[None, ...] if label is not None else None
        }
        return data_dict

    def apply_single_transform(self, transform, data_dict, name):
        """应用单个transform"""
        # 深拷贝数据以避免修改原始数据
        data_copy = copy.deepcopy(data_dict)
        result = transform(**data_copy)
        print(f"成功应用 {name} transform")
        return result


    def get_middle_slice(self, volume, axis=2):
        """获取体积数据的中间切片"""
        middle_idx = volume.shape[axis] // 2
        if axis == 0:
            return volume[middle_idx, :, :]
        elif axis == 1:
            return volume[:, middle_idx, :]
        else:
            return volume[:, :, middle_idx]

    def visualize_single_transform(self, transform_name, transform, save_path=None):
        """可视化单个transform的效果"""
        # 准备数据
        data_dict = self.prepare_data_for_transform(self.original_image, self.original_label)

        # 应用transform
        transformed_data = self.apply_single_transform(transform, data_dict, transform_name)

        # 提取图像数据
        original_img = data_dict['image'][0]  # 去掉channel维度
        transformed_img = transformed_data['image'][0]

        # 获取中间切片进行可视化
        orig_slice = self.get_middle_slice(original_img)
        trans_slice = self.get_middle_slice(transformed_img)

        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        im1 = axes[0].imshow(orig_slice, cmap='gray', aspect='auto')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])

        # 变换后图像
        im2 = axes[1].imshow(trans_slice, cmap='gray', aspect='auto')
        axes[1].set_title(f'After {transform_name}')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])

        # 差异图
        # diff = trans_slice - orig_slice
        # im3 = axes[2].imshow(diff, cmap='RdBu', aspect='auto')
        # axes[2].set_title('Difference')
        # axes[2].axis('off')
        # plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/{transform_name}_comparison.png", dpi=300, bbox_inches='tight')

        plt.show()

        # 打印统计信息
        print(f"{transform_name} 统计信息:")
        print(f"  原始图像 - 均值: {orig_slice.mean():.4f}, 标准差: {orig_slice.std():.4f}")
        print(f"  变换后  - 均值: {trans_slice.mean():.4f}, 标准差: {trans_slice.std():.4f}")
        # print(f"  差异    - 均值: {diff.mean():.4f}, 标准差: {diff.std():.4f}")
        print("-" * 50)

    def visualize_all_transforms(self, save_path=None):
        """可视化所有transforms的效果"""
        print("开始可视化各个transforms的效果...")

        for i, (name, transform) in enumerate(self.transforms):
            print(f"正在处理 {i + 1}/{len(self.transforms)}: {name}")
            self.visualize_single_transform(name, transform, save_path)

    def visualize_combined_transforms(self, save_path=None):
        """可视化所有transforms组合后的效果"""
        print("开始可视化所有transforms组合的效果...")

        # 准备数据
        data_dict = self.prepare_data_for_transform(self.original_image, self.original_label)

        # 依次应用所有transforms
        current_data = data_dict
        for name, transform in self.transforms:
            print(f"应用 {name}...")
            current_data = self.apply_single_transform(transform, current_data, name)

        # 提取最终结果
        original_img = data_dict['image'][0]
        final_img = current_data['image'][0]

        # 获取中间切片
        orig_slice = self.get_middle_slice(original_img)
        final_slice = self.get_middle_slice(final_img)

        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        im1 = axes[0].imshow(orig_slice, cmap='gray', aspect='auto')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])

        # 最终结果
        im2 = axes[1].imshow(final_slice, cmap='gray', aspect='auto')
        axes[1].set_title('After All Transforms')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])

        # 差异图
        # diff = final_slice - orig_slice
        # im3 = axes[2].imshow(diff, cmap='RdBu', aspect='auto')
        # axes[2].set_title('Total Difference')
        # axes[2].axis('off')
        # plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}/combined_transforms_comparison.png", dpi=300, bbox_inches='tight')

        plt.show()

        # 打印统计信息
        print("所有transforms组合后的统计信息:")
        print(f"  原始图像 - 均值: {orig_slice.mean():.4f}, 标准差: {orig_slice.std():.4f}")
        print(f"  最终结果 - 均值: {final_slice.mean():.4f}, 标准差: {final_slice.std():.4f}")
        # print(f"  总差异   - 均值: {diff.mean():.4f}, 标准差: {diff.std():.4f}")


def main():
    """主函数"""
    # 设置文件路径
    image_path = "/home/dataset/nnUNet_raw/Dataset218_CTPelvic1k/imagesTr/dataset6_CLINIC_0079_0000.nii.gz"
    label_path = "/home/dataset/nnUNet_raw/Dataset218_CTPelvic1k/labelsTr/dataset6_CLINIC_0079.nii.gz"

    # 检查文件是否存在
    if not Path(image_path).exists():
        print(f"图像文件不存在: {image_path}")
        return

    if not Path(label_path).exists():
        print(f"标签文件不存在: {label_path}")
        return


    visualizer = TransformVisualizer(image_path, label_path)

    # 创建保存目录
    save_dir = "/home/dataset/nnUNet_raw/Dataset218_CTPelvic1k/transform_visualizations"
    Path(save_dir).mkdir(exist_ok=True)
    # 可视化各个transforms
    visualizer.visualize_all_transforms(save_dir)

    # 可视化组合效果
    visualizer.visualize_combined_transforms(save_dir)

    print(f"所有可视化结果已保存到 {save_dir} 目录")


if __name__ == "__main__":
    main()