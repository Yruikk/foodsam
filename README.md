# foodsam 项目说明（更新版）

## 1. 环境准备

进入项目目录并安装依赖：

- `cd /root/foodsam`
- `bash env_setup.sh`

## 2. 推荐用法（统一入口）

当前推荐统一使用 `RunMain.py`，一次完成检测、分割和估重。

示例：

- `python RunMain.py --image-dir ./datasets_nonlabel/Chips/薯条-狼牙薯条/1-180/ --yolo-visualize --sam-visualize`

常用参数：

- `--image-dir`：输入图片目录（递归处理）
- `--yolo-visualize`：保存检测可视化
- `--sam-visualize`：保存分割可视化
- `--weight-output-root`：每张图估重结果输出目录（默认 `weight_results`）

说明：`RunMain.py` 现在不需要强制传 `--food-class`。不传时会按路径和配置自动匹配类别并选择估重方式。

## 3. 主要输出目录

- `yolo_seg_results/.../*.npy`：检测框
- `sam_seg_results/.../*.png`：分割掩码
- `weight_results/.../*.npy`：每张图预测重量（克）
- `logs/RunMain_*.log`：运行日志

例如输入图片：

- `datasets_nonlabel/Chicken/大鸡/1-1923/094127.jpg`

对应重量输出：

- `weight_results/datasets_nonlabel/Chicken/大鸡/1-1923/094127.npy`

## 4. 重量配置

- `weight_per_object.json`：按个数估重（`g/object`）
- `weight_per_pixel.json`：按分割像素估重（`g/pixel`）

选择逻辑：

1. 命中 `weight_per_object.json` -> 用个数估重
2. 否则命中 `weight_per_pixel.json` -> 用像素估重
3. 都未命中 -> 使用 `weight_per_pixel.json` 中的 `default_weight`

## 5. calculate_accuracy.py（准确率评估）

该脚本会遍历 `weight_results` 下的 `.npy` 预测文件，并从路径中的 `xxx-yyy` 文件夹提取真实重量（`yyy`，单位克）。

逐图计算：

$$
RSE = \frac{|pred-groundtruth|}{groundtruth}
$$

全局平均后得到：

$$
Accuracy = 1 - RSE
$$

示例：

- `python calculate_accuracy.py`
- `python calculate_accuracy.py --weight-results-root weight_results --verbose`

## 6. calculate_weight 逻辑说明

仓库里当前没有独立 `calculate_weight.py`。估重逻辑已集成在 `RunMain.py` + `sam_seg.py`：

- 读取检测框和分割结果
- 按 `weight_per_object.json` 或 `weight_per_pixel.json` 计算估计重量
- 每张图保存到 `weight_results/.../*.npy`

## 7. calculate_mIoU.py（分割质量评估）

该脚本用于评估分割结果与 GT 掩码的一致性，默认对以下目录进行匹配评估：

- 预测：`sam_seg_results/datasets/images/val`
- 标注：`sam_seg_gt/datasets/images/val`

计算方式（逐图）：

$$
IoU = \frac{|Pred \cap GT|}{|Pred \cup GT|}
$$

然后对所有有效图片的 IoU 取平均，得到最终 `mIoU`。

说明：

- 掩码按二值前景处理（像素值 `> 0` 视为前景）
- 若预测图找不到对应 GT，或尺寸不一致，会提示并跳过

示例：

- `python calculate_mIoU.py`
- `python calculate_mIoU.py --pred-dir sam_seg_results/datasets/images/val --gt-dir sam_seg_gt/datasets/images/val --verbose`

## 8. 当前目录（简化）

```text
/root/foodsam
├── RunMain.py
├── yolo_detect.py
├── sam_seg.py
├── calculate_accuracy.py
├── calculate_mIoU.py
├── weight_per_object.json
├── weight_per_pixel.json
├── datasets/
├── datasets_nonlabel/
├── yolo_seg_results/
├── sam_seg_results/
├── weight_results/
└── logs/
```
