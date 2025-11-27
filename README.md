# foodsam 项目说明

## 1. 代码位置与依赖安装

建议先进入项目代码路径再运行示例命令：

```bash
cd /root/foodsam
bash env_setup.sh
```

---

## 2. 整体流程与快速开始

当前完整的处理流程是两步：

1. 使用 YOLO 对图片进行检测，生成目标框（bounding boxes）结果；
2. 使用 SAM 读取这些框作为 box prompt，进行分割，生成最终分割结果。

你可以通过一个入口脚本 `RunMain.py` 一次性完成这两步，也可以按需要单独调用 `yolo_detect.py` 和 `sam_seg.py`。

推荐的一键运行方式：

```bash
python RunMain.py --image-dir ./datasets/Danta/boat/18/ --food-class boat --yolo-visualize --sam-visualize
```

参数说明：

- `--image-dir`：待处理图片所在的目录路径（例如 `./datasets/Danta/boat/18/`）。
- `--food-class`：食物类别名称，用于在sam_seg.py中决定该类食物的重量计算方式（像素 OR 个数）。
- `--yolo-visualize`：保存yolo的可视化检测结果（会生成./yolo_seg_results_visual/）。
- `--sam-visualize`：保存sam的可视化检测结果（会生成./sam_seg_results_visual/）。

---

## 3. 运行细节说明（YOLO + SAM 两步流程）

### 3.1 第一步：YOLO 检测

脚本：`yolo_detect.py`  
示例命令：

```bash
python yolo_detect.py --image-dir ./datasets/Danta/boat/18/ --yolo-visualize
```

### 3.2 第二步：SAM 分割

脚本：`sam_seg.py`  
示例命令：

```bash
python sam_seg.py --image-dir ./datasets/Danta/boat/18/ --food-class boat --sam-visualize
```

行为说明：

- 读取第一步在 `./yolo_seg_results/` 中保存的 YOLO 检测框结果，作为 **box prompt**；
- 使用 SAM 模型对目标进行分割；
- 在 `./sam_seg_results/` 中生成最终的分割结果（包括 mask、可视化结果以及用于计算面积/重量的中间数据）。


## 4. 模型与配置文件

### 4.1 YOLO 权重文件

- 目录：`./yolo_ckpts`
- 作用：存放不同次训练得到的YOLO 权重文件，yolov5s是最小的，yolov5x是最大的，后续的v0/v1表示version（默认使用最新）。

### 4.2 每像素重量配置

- 文件：`./weight_per_pixel.json`
- 作用：为每种 `food-class` 配置“每像素重量”等信息，用于基于分割结果计算食物重量。

结构示意（示例）：

```json
{
  "danta": {
    "weight_per_pixel": 0.0012
  },
  "other_food": {
    "weight_per_pixel": 0.0008
  }
}
```

### 4.3 每个体重量配置

- 文件：`./weight_per_object.json`
- 作用：为每种 `food-class` 配置“每个体重量”等信息，用于基于分割结果计算食物重量。

结构示意（示例）：

```json
{
  "classes": {
    "蔓越莓曲奇（切片）": 17.0,
    "抹茶曲奇": 16.0,
  }
}
```

## 5. 项目目录结构示例

下方是一个简化后的目录结构示例（仅供参考，实际可略有不同）：

```text
/root/foodsam
├── RunMain.py                    # 一键串联 YOLO + SAM 的主入口脚本
├── yolo_detect.py                # YOLO 检测脚本
├── sam_seg.py                    # SAM 分割脚本
├── yolo_ckpts/                   # YOLO 权重
├── sam_ckpts/                    # SAM 权重
├── weight_per_pixel.json         # 各类食物的每像素重量配置
├── weight_per_object.json        # 各类食物的每个体重量配置
├── datasets/
│   └── Danta/
│       └── boat/
│           └── 18/               # 示例图片目录
├── yolo_seg_results/             # YOLO 检测结果输出目录（npy文件）
├── sam_seg_results/              # SAM 分割结果输出目录（二值图像）
├── yolo_seg_results_visualize/   # YOLO 可视化检测结果输出目录
├── sam_seg_results_visualize/    # SAM 可视化分割结果输出目录
├── logs/                         # RunMain的日志记录
└── README.md                     # 本说明文件
```


```
