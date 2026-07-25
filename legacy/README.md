# 历史配置与脚本归档

本目录存放已被现行实验体系取代、但保留备查的早期配置、权重和脚本。**这里的内容不参与任何当前实验链路**,仅作历史溯源用。

现行的类别定义、数据划分和训练流程见:

- 类别与划分基准:`datasets/fruitfly_detection_v1/metadata/classes.yaml`
- 各模型训练配置:`detection_model_benchmark/<NN>_<model>/dataset_config.yaml`
- 检测缓存入口:`detection_model_benchmark/cache_detections.py`

## 重要:归档权重的类别顺序与现行定义相反

现行统一定义(`classes.yaml` 的 `canonical_yolo_and_deim_rtdetr_ids`):

```
0 = Bactrocera dorsalis
1 = Bactrocera correcta
```

而 `weights/yolo8n_two-class.pt` 内嵌的 `names` 是:

```
0 = Bactrocera correcta
1 = Bactrocera dorsalis
```

两者**完全对调**。该权重训练时使用的是 `configs/dataset_two_class.yaml`,那份配置的 `names` 列表就是反序的,顺序问题源自此处。

风险在于下游按物种名字符串匹配真值时,对调后不会报错,只会得到两个物种数值互换的结果。历史 BoxMOT 计数输出给出 correcta=93 / dorsalis=45,而对应真值是 dorsalis=61 / correcta=118,方向恰好相反,与此一致。

**不要用 `weights/` 下的权重生成检测缓存或计数结果。** 如确需复现历史结果,必须显式做类别 ID 重映射。

## 文件清单

### configs/

| 文件 | 归档原因 |
|---|---|
| `data.yaml` | 单类(仅 dorsalis)早期配置。`train`/`val`/`test` 指向的 `dataset/images/*` 已不存在;文件头注释仍残留无关的 "safety helmet" 字样。被 `datasets/fruitfly_detection_v1/` 取代。 |
| `dataset_two_class.yaml` | 两类早期配置,**类别顺序与现行定义相反**(见上)。数据路径指向另一个项目 `/home/admin1/Projects/ultralytics-yolo11-main/dataset_two_class/`,不在本仓库内。 |

### weights/

| 文件 | 归档原因 |
|---|---|
| `yolo8n_two-class.pt` | 两类工程临时权重,**类别顺序反序**(见上)。训练配置为 `configs/dataset_two_class.yaml`。已被 e100 基线的 11 个模型取代。 |
| `yolo26n.pt` | 单类工程临时权重(仅 dorsalis),顺序无问题。训练配置为 `configs/data.yaml`。同样已被 e100 基线取代。 |

权重文件受 `.gitignore` 的 `*.pt` 规则约束,不入版本控制,仅存在于本地磁盘。

### scripts/

| 文件 | 归档原因 |
|---|---|
| `split_data.py` | 早期随机 8:1:1 划分脚本,输出目录 `dataset/` 已不存在。现行划分已固化在 `datasets/fruitfly_detection_v1/` 并有指纹校验,不应再用随机重划。 |
| `train.py` | 早期单入口训练脚本,引用反序的 `dataset_two_class.yaml`,并硬编码了另一个项目的模型配置路径。现行 11 个模型各自有独立的训练环境与 RUNBOOK。 |

归档时已同步修正 `train.py` 内的相对路径,使其在新位置仍指向正确的仓库根目录;但其引用的数据配置本身仍是反序的,直接运行会产出与现行定义不一致的类别顺序。
