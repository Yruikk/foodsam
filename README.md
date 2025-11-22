# foodsam 项目说明

## 1. 代码位置

建议先进入项目代码路径再运行示例命令：

```bash
cd /root/qym/foodsam
```

---

## 2. 整体流程与快速开始

当前完整的处理流程是两步：

1. 使用 YOLO 对图片进行检测，生成目标框（bounding boxes）结果；
2. 使用 SAM 读取这些框作为 box prompt，进行分割，生成最终分割结果。

你可以通过一个入口脚本 `RunMain.py` 一次性完成这两步，也可以按需要单独调用 `yolo_detect.py` 和 `sam_seg.py`。

推荐的一键运行方式：

```bash
python RunMain.py --image-dir ./datasets/Danta/boat/18/ --food-class danta
```

参数说明：

- `--image-dir`：待处理图片所在的目录路径（例如 `./datasets/Danta/boat/18/`）。
- `--food-class`：食物类别名称，用于：
  - 决定使用哪一个 YOLO 权重；
  - 决定该类食物的重量计算方式。

---

## 3. 运行细节说明（YOLO + SAM 两步流程）

无论是由 `RunMain.py` 自动串联，还是手动分步执行，逻辑都是：

### 3.1 第一步：YOLO 检测

脚本：`yolo_detect.py`  
示例命令：

```bash
python yolo_detect.py --image-dir ./datasets/Danta/boat/18/ --food-class danta
```

行为说明：

- 读取 `--image-dir` 路径下的图片；
- 根据 `--food-class`（如 `danta`）选择对应的 YOLO 权重（存放在 `./yolo_ckpts` 中）；
- 在 `./yolo_seg_results/` 下生成 YOLO 检测的结果（主要是检测框等信息）。

### 3.2 第二步：SAM 分割

脚本：`sam_seg.py`  
示例命令：

```bash
python sam_seg.py --image-dir ./datasets/Danta/boat/18/ --food-class danta
```

行为说明：

- 读取第一步在 `./yolo_seg_results/` 中保存的 YOLO 检测框结果，作为 **box prompt**；
- 使用 SAM 模型对目标进行分割；
- 在 `./sam_seg_results/` 中生成最终的分割结果（包括 mask、可视化结果以及用于计算面积/重量的中间数据）。

### 3.3 一键入口：`RunMain.py`

脚本：`RunMain.py`  
当前逻辑：**内部顺序执行上述两步**。

示例命令（等价于先跑 YOLO 再跑 SAM）：

```bash
python RunMain.py --image-dir ./datasets/Danta/boat/18/ --food-class danta
```

等价于依次执行：

```bash
python yolo_detect.py --image-dir ./datasets/Danta/boat/18/ --food-class danta
python sam_seg.py   --image-dir ./datasets/Danta/boat/18/ --food-class danta
```

---

## 4. 模型与配置文件

### 4.1 YOLO 权重文件

- 目录：`./yolo_ckpts`
- 作用：存放不同 `food-class` 对应的 YOLO 权重文件；`food-class` 影响会加载哪个权重。

示例（仅举例，实际以代码为准）：

- 当 `--food-class danta` 时，可能会选用：
  - `./yolo_ckpts/yolo_danta.pt`

未来扩展：

- 添加新食物类型时，只需：
  - 在 `./yolo_ckpts` 中放入对应权重文件；
  - 在相关逻辑中配置好 `food-class` 与权重文件的映射关系。

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

未来扩展：

- 新增食物种类时，在该文件中加入对应条目即可。

---


## 5. 项目目录结构示例

下方是一个简化后的目录结构示例（仅供参考，实际可略有不同）：

```text
/root/yrk/update_foodsam
├── RunMain.py              # 一键串联 YOLO + SAM 的主入口脚本
├── yolo_detect.py          # YOLO 检测脚本
├── sam_seg.py              # SAM 分割脚本
├── yolo_ckpts/             # 各 food-class 对应的 YOLO 权重
├── weight_per_pixel.json   # 各类食物的每像素重量配置
├── datasets/
│   └── Danta/
│       └── boat/
│           └── 18/         # 示例图片目录
├── yolo_seg_results/       # YOLO 检测结果输出目录
├── sam_seg_results/        # SAM 分割结果输出目录
└── README.md               # 本说明文件
```


```
