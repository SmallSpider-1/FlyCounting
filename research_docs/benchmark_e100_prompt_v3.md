你是一名负责可复现目标检测实验的高级 AI 工程师。请直接在服务器上构建并运行下面的实验管线，不要只给操作建议，不要让我逐条确认。

本任务预计持续数天。训练调度器必须独立于当前 agent 会话运行，不能依赖当前对话、终端连接或上下文窗口持续存活。当前会话中断后，后续会话必须能够根据磁盘中的调度状态继续监控和恢复，不能重复启动已经运行或已经完成的任务。

# 零、规则优先级与冲突仲裁

任何规则发生冲突时，按以下优先级处理：

1. 数据完整性与测试集隔离
2. 实验协议公平性
3. 可复现性与 checkpoint 可恢复性
4. 任务完成度（11 个模型全部跑通优先于严格环境一致性，见第六节环境修改政策）
5. 训练速度和资源利用率

不得为了尽快完成而违反更高优先级规则。

如果遇到未明确规定的边界情况：

- 不得现场随意选择有利于某个模型的配置；
- 必须采用对全部模型一致、可记录、可复现的规则；
- 将决策、理由和影响写入 protocol_decisions.md；
- 如果该决策会实质改变模型排名含义，停止正式训练并汇报，而不是即兴继续。

# 一、实验目标

在以下工作区完成 11 个目标检测模型的统一固定预算对比实验：

/home/admin1/Projects/ultralytics-main/detection_model_benchmark

必须严格使用：

1. YOLOv5n
2. YOLOv7-tiny
3. YOLOv8n
4. YOLOv9-T
5. YOLOv10n
6. YOLO11n
7. YOLO12n
8. YOLO26n
9. DEIM-D-FINE-N
10. RT-DETR-R18
11. Faster R-CNN R50-FPN

不得替换模型版本、规模或使用相邻型号。

本轮实验定义为：

“统一 100 epoch 固定预算、单随机种子的首轮工程筛选。”

100 epoch 是统一预算，不保证全部模型充分收敛。本轮结果不得写成具有多随机种子统计显著性的论文最终结论。

# 二、必须先读取的文件

开始任何操作前，完整读取：

1. /home/admin1/Projects/ultralytics-main/AGENTS.md（如果存在）
2. /home/admin1/Projects/ultralytics-main/research_docs/研究背景与论文主线.md
3. /home/admin1/Projects/ultralytics-main/research_docs/当前进展与下一步.md
4. /home/admin1/Projects/ultralytics-main/detection_model_benchmark/README.md
5. 每个模型目录中的 RUNBOOK.md
6. 每个模型目录中的 PROJECT_INFO.md
7. 每个模型目录中的 dataset_config.yaml
8. 每个模型的官方训练配置、优化器和调度器配置
9. /home/admin1/Projects/ultralytics-main/detection_model_benchmark/_common/environment_map.yaml
10. /home/admin1/Projects/ultralytics-main/detection_model_benchmark/_common/dataset_paths.env
11. /home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1/README.md
12. /home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1/metadata/dataset_stats.json
13. /home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1/metadata/validation_report.json
14. /home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1/metadata/classes.yaml

不要要求用户重复上述文件中已经记录的信息。

# 三、固定数据集

数据集根目录：

/home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1

YOLO 配置：

/home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1/yolo/dataset.yaml

COCO 根目录：

/home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1/coco

类别定义：

- canonical class 0 = Bactrocera dorsalis
- canonical class 1 = Bactrocera correcta

Faster R-CNN 原生类别：

- 0 = background
- 1 = Bactrocera dorsalis
- 2 = Bactrocera correcta

数据集指纹必须为：

a56fa4a6e10370e963d34e983711c3ed69b5b2f064a0d7771d976d6f1c57b5da

数据规模：

- train：8,143 张、14,343 个框
- val：1,530 张、2,491 个框
- test：1,517 张、2,706 个框
- 总计：11,190 张、19,540 个框
- 合法空背景图：3,060 张

必须保留空背景图。

Faster R-CNN 必须使用当前 train_custom.py 默认行为，保留全部训练图，不得启用：

--drop-empty-images

所有框架必须设置：

drop_last=false

确保每个 epoch 都使用全部 8,143 张训练图。

最后一个 batch 可以小于 16，但必须记录实际 batch 大小和实际 optimizer step 数。

禁止修改：

- /home/admin1/Projects/ultralytics-main/data
- /home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1

禁止重新划分、重新标注、删除空图或生成离线增强副本。

# 四、明确的 COCO 标注文件和域测试集

中央统一评测全部使用 canonical 0-based COCO 标注，不使用 `2017` 后缀版本作为主评测真值。

统一图片目录：

/home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1/coco/test

## 4.1 Mixed test

真值文件：

/home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1/coco/annotations/instances_test.json

预期：

- 图片：1,517
- 框：2,706
- class 0：1,950
- class 1：756

## 4.2 C test

真值文件：

/home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1/coco/annotations/instances_test_C.json

预期：

- 图片：152
- 框：583
- class 0：162
- class 1：421

## 4.3 R1 test

真值文件：

/home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1/coco/annotations/instances_test_R1.json

预期：

- 图片：960
- 框：723
- class 0：723
- class 1：0

R1 的 Bactrocera correcta AP 必须写为 N/A，不得填为 0。

## 4.4 R2 test

真值文件：

/home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1/coco/annotations/instances_test_R2.json

预期：

- 图片：405
- 框：1,400
- class 0：1,065
- class 1：335

## 4.5 `2017` 标注的用途

以下文件只用于 Faster R-CNN 原生 1-based 类别加载：

- instances_train2017.json
- instances_val2017.json
- instances_test2017.json
- instances_test2017_C.json
- instances_test2017_R1.json
- instances_test2017_R2.json

中央主评测不得使用这些 1-based 文件。

Faster R-CNN 预测必须先执行：

- category 1 → canonical category 0
- category 2 → canonical category 1

然后再与 canonical 0-based 真值评测。

# 五、COCO image_id 一致性

所有模型输出的预测 JSON 中：

- image_id 必须与对应 canonical COCO 真值中的 `images[].id` 完全一致；
- 禁止各框架根据文件遍历顺序自行编号；
- 禁止使用文件名哈希或顺序号替代 canonical image_id；
- 必须先通过 canonical 标注构建 `file_name → image_id` 映射；
- 每条预测根据原图片 file_name 查回 canonical image_id。

中央评测前必须检查：

- 所有 prediction image_id 都存在于对应真值；
- 没有未知 image_id；
- 没有把 mixed、C、R1、R2 的 ID 空间重新编号；
- 同一图片在四种评测视图中的 image_id 保持一致。

生成：

summary/image_id_validation.json

发现 image_id 不一致时，停止该模型的中央评测，不得继续计算错误 mAP。

# 六、固定模型、环境、权重与环境修改政策

严格使用已有 Conda 环境，不创建新环境。对环境的任何修改必须遵循 6.1 环境修改政策。

bench_detection_torch201：

- YOLOv5n
- YOLOv7-tiny
- YOLOv9-T
- YOLOv10n
- DEIM-D-FINE-N
- RT-DETR-R18

yolov26：

- YOLOv8n
- YOLO11n
- YOLO12n
- YOLO26n

pointnext：

- Faster R-CNN R50-FPN

必须使用各模型目录中已经准备的官方 COCO 预训练权重。

禁止：

- git pull；
- 联网更新仓库；
- 切换 commit；
- 替换权重；
- 替换模型；
- 让框架静默下载权重、字体或其他资产。

## 6.1 环境修改政策

总原则：以“11 个模型全部跑通”为优先。不得因为追求严格环境一致而放弃跑通某个模型，但任何环境变化都必须完整记录，且不得改变训练核心行为。

允许（无需事先请示，但必须记录）：

- 在已有 Conda 环境中新增安装纯工具类依赖，例如 pycocotools、复杂度统计工具（thop/fvcore 等）、绘图、监控、文件锁工具；
- 前提是安装过程不升级、不降级、不卸载任何已有核心训练依赖；
- 必要时使用 --no-deps 或固定版本号安装，避免依赖解析连带变动核心包。

禁止静默执行：

- 升级或降级 torch、torchvision、CUDA 相关包、numpy 等核心训练依赖；
- 卸载或替换训练框架本身；
- 因安装工具包连带变动核心包版本。

如果某模型只有变更核心训练依赖才能跑通：

- 不得静默变更；
- 将该模型标记为 env_conflict_blocked；
- 汇报冲突详情和可选方案，等待决定；
- 其他模型继续，不因此阻塞整个管线。

每次安装必须记录到 summary/environment_changes.md：

- 环境名；
- 包名和版本；
- 安装命令；
- 安装原因；
- 安装前后 pip freeze / conda list 快照路径；
- 对训练行为是否有影响的判断。

# 七、离线运行检查

正式训练前验证所有框架可以完全离线启动。

根据框架设置适用的离线环境变量，例如：

- YOLO_OFFLINE=1
- WANDB_MODE=offline 或完全禁用 W&B
- HF_HUB_OFFLINE=1
- TRANSFORMERS_OFFLINE=1

不得设置框架不识别的变量后假装生效。

禁止在正式训练进程运行期间使用：

- wget
- curl
- pip install
- conda install
- git clone
- torch.hub 在线下载
- Ultralytics HUB 联网
- 自动字体下载
- 自动权重下载

上述禁令只针对正式训练进程运行期间。在预检和故障修复阶段，允许按照第六节 6.1 环境修改政策联网安装工具类依赖；每次安装后必须重新执行受影响模型的离线启动检查，确认正式训练仍可完全离线运行。

训练前分别对 11 个入口做一次离线配置解析和单 batch 数据加载检查。

生成：

preflight/offline_readiness.csv

至少包含：

- model
- offline_environment
- local_weight_found
- local_font_or_cache_ready
- config_load
- dataloader_load
- hidden_download_attempt
- status

如果框架仍尝试联网，先通过本地配置关闭该行为，不能放任任务半夜因联网失败而退出。

# 八、实验前完整检查

执行并保存：

1. nvidia-smi
2. nvidia-smi topo -m
3. numactl --hardware（仅在已安装时）
4. df -h
5. verify_all.sh
6. 数据集 validator
7. 11 个 dataset_config.yaml 检查
8. 数据集指纹检查
9. 类别映射检查
10. COCO 子集图片数和框数检查
11. Conda 环境检查
12. 权重 SHA-256
13. 源码 commit
14. 离线运行检查

运行：

cd /home/admin1/Projects/ultralytics-main/detection_model_benchmark
./verify_all.sh

以及：

python /home/admin1/Projects/ultralytics-main/scripts/validate_fruitfly_detection_v1.py \
  --dataset-root /home/admin1/Projects/ultralytics-main/datasets/fruitfly_detection_v1

# 九、立即停止并汇报的全局条件

出现以下任一情况，停止整个正式训练管线并汇报：

1. 数据集指纹不一致；
2. 图片数、框数或类别映射不一致；
3. train/val/test 内容被修改；
4. canonical COCO image_id 异常；
5. 协议冻结后 resolved config 被静默修改；
6. 同一模型被重复启动；
7. 同一 GPU 同时出现两个正式训练进程；
8. test 在 checkpoint 冻结前被用于选模或调参；
9. checkpoint 损坏且没有可恢复副本；
10. 磁盘空间不足以安全保存 checkpoint；
11. 无法确定某模型的关键 batch/LR 策略且只能靠猜测；
12. 发现原始训练协议已经违反数据隔离。

单个模型的普通 OOM、代码兼容错误、DataLoader 错误或 env_conflict_blocked 只停止该模型任务，不立即停止其他独立模型，但必须记录并修复或汇报。

# 十、统一 100 epoch 固定预算

结果目录：

/home/admin1/Projects/ultralytics-main/detection_model_benchmark/results/fruitfly_detection_v1/benchmark_640_e100_s42

创建：

protocol.yaml

固定：

- budget_type=fixed_100_epoch_screening
- imgsz=640
- epochs=100
- seed=42
- AMP=true
- workers=8
- target_effective_batch=16
- drop_last=false
- validation_every_epoch=true
- early_stopping=false
- train images=8,143
- val images=1,530
- test images=1,517

首轮筛选不要求完全确定性。

必须：

- 设置并记录 seed=42；
- 记录 Python、NumPy、PyTorch、CUDA 随机种子；
- 记录 cudnn.benchmark；
- 记录 cudnn.deterministic；
- 记录 torch deterministic algorithms 状态；
- 记录仍然存在的非确定性来源。

不要为了完全确定性擅自开启会显著拖慢训练或导致算子失败的强制 deterministic 模式。

# 十一、effective batch、nbs 和 base LR 策略

本轮明确选择：

“强制实际 effective batch size=16。”

不接受 YOLO 框架静默使用 nbs=64 后自动 accumulate×4，形成实际 effective batch≈64。

## 11.1 YOLO 系列

对于存在 nominal batch size 或 nbs 的框架：

- 将 nbs 设置为 16；
- 或修改等价 accumulate 逻辑，使 batch=16 时 accumulation=1；
- 禁止保留 nbs=64 导致的隐藏 accumulate×4；
- 记录修改前后的 accumulate；
- 记录 weight decay 自动缩放前后值。

如果 micro-batch=16：

- accumulation=1
- effective batch=16

如果因 OOM 使用 micro-batch=8：

- accumulation=2
- effective batch=16

如果 micro-batch=4：

- accumulation=4
- effective batch=16

## 11.2 base learning rate 统一规则

每个模型先提取：

- official_base_lr
- official_reference_effective_batch

然后使用线性缩放：

target_base_lr =
official_base_lr × 16 / official_reference_effective_batch

必须记录：

- official_base_lr
- official_reference_effective_batch
- target_effective_batch
- target_base_lr
- scaling_ratio
- scaling_source

对于：

- YOLO 系列：以其官方 nbs 或官方参考总 batch 为 reference batch；
- Faster R-CNN：官方参考 effective batch=16 时，base LR 不变；
- DEIM/RT-DETR：从本地官方配置读取每 GPU batch、GPU 数或总 batch，计算官方 reference effective batch。

线性缩放规则统一适用于 SGD 和 AdamW 系模型。对 AdamW 系模型（RT-DETR、DEIM 等）同样采用线性缩放属于协议选择，必须作为显式条目写入 protocol_decisions.md。

如果官方 reference batch 无法从本地配置、README 或明确的官方脚本中确定：

- 不得猜测；
- 将该模型标记为 protocol_resolution_blocked；
- 在正式训练前汇报。

必须禁止框架在 target_base_lr 已经手工换算后再次自动缩放 LR，避免 double scaling。

## 11.3 显式固定优化器，禁止自动优化器选择

- Ultralytics 系框架（YOLOv8/11/12/26 等）必须把 optimizer 显式设置为官方默认类型（通常为 SGD），禁止保留 optimizer='auto'；
- optimizer='auto' 会忽略配置中的 lr0 和 momentum，由框架自行决定优化器和学习率，使 11.2 节的全部换算静默失效；
- 其他框架中任何等价的“自动选择优化器 / 自动改写 LR”逻辑同样必须关闭；
- 每个模型训练启动后，必须从训练日志或运行时 optimizer 对象核对实际生效的 optimizer 类型、base LR、momentum 与 resolved_config 完全一致；
- 不一致视为配置冻结失败，按第十四节的配置错误流程处理。

## 11.4 weight decay

保留官方 nominal weight decay 数值，不因为目标 batch=16 再做未经定义的二次线性缩放。

必须关闭或中和 YOLO 内部基于 nbs=64 的隐藏 weight decay 缩放，使 resolved weight decay 与记录值一致。

记录：

- official_weight_decay
- framework_auto_scaled_weight_decay
- resolved_weight_decay
- resolution_notes

## 11.5 实际 optimizer steps

设置 drop_last=false。

理论上每个 epoch：

ceil(8143/16)=509 个 effective-batch 更新单元。

必须从训练日志或 optimizer hook 记录实际 optimizer.step() 次数。

不得假设所有框架都正好是 509。

生成：

runs/<model>/optimizer_steps.csv

至少包含：

- epoch
- images_seen
- micro_batches
- optimizer_steps
- micro_batch
- accumulation
- effective_batch
- last_batch_size

# 十二、学习率调度器归一化

不得直接沿用为 26、72、160 或其他总 epoch 设计的绝对调度节点。

保留：

- 官方调度器类型；
- 相对训练阶段；
- warmup 语义；
- 主学习阶段；
- 衰减顺序；
- EMA 阶段语义。

映射规则：

mapped_epoch =
round(official_epoch / official_total_epochs × 100)

对于最后若干 epoch 的阶段：

mapped_duration =
round(official_duration / official_total_epochs × 100)

对于基于 optimizer steps 的调度器：

- 根据本轮实际预计 optimizer steps 重算；
- 不使用原数据集的绝对 step。

映射结果必须满足：

- 保持官方所有阶段的先后顺序；
- mapped_epoch 不得小于 1，不得超过 100；
- 官方本应不同的两个阶段不得因四舍五入被压缩到同一 epoch；
- warmup 映射后至少保留 1 个 epoch 或等价 step 数；
- 出现取整冲突时，做保持顺序的最小调整，并把调整前后值记录进 mapping_notes。

Faster R-CNN 示例：

- official epochs=26
- lr_steps=[16,22]
- mapped lr_steps=[62,85]

DEIM、RT-DETR 必须检查：

- scheduler milestones
- flat_epoch
- stop_epoch
- warmup
- EMA start
- EMA restart
- 数据增强阶段
- total steps

Cosine/OneCycle 必须重算：

- epochs
- T_max
- total_steps
- pct_start
- warmup duration

不缩放：

- loss 权重
- optimizer betas
- momentum
- nominal weight decay
- EMA decay 系数本身
- NMS IoU
- score threshold

# 十三、增强关闭与 no-aug 节点

本轮只保留：

- resize
- padding
- normalize
- horizontal flip，p=0.5

全程关闭：

- mosaic
- mixup
- copy-paste
- random erasing
- vertical flip
- multi-scale
- test-time augmentation
- 离线增强

如果某增强已经从 epoch 1 起全局关闭，则其对应的：

- close_mosaic
- no_aug_epoch
- stop_mixup
- stop_copy_paste

节点必须记录为：

N/A_ALREADY_DISABLED

不得再进行 epoch 映射，也不得假装存在后期 no-aug 阶段。

EMA、学习率衰减等仍有实际作用的阶段继续映射。

final_recommendation.md 必须明确声明：

“本轮为统一弱增强固定预算筛选。关闭 mosaic、mixup 等强增强可能对以强增强为官方配方前提的模型，尤其部分 YOLO 系列，造成更明显的性能损失。因此本轮结果用于统一条件筛选，不代表各模型官方最佳实践上限。”

# 十四、配置冻结

启动任何正式训练前，全部 11 个模型必须生成：

- official_schedule.yaml
- mapped_schedule_e100.yaml
- resolved_config.yaml
- lr_schedule.csv
- lr_schedule.png
- batch_lr_resolution.yaml

汇总生成：

- scheduler_mapping.csv
- batch_lr_mapping.csv
- configuration_manifest.sha256
- protocol_decisions.md

scheduler_mapping.csv 至少包含：

- model
- official_total_epochs
- official_scheduler
- official_milestones
- mapped_milestones
- official_warmup
- mapped_warmup
- official_no_aug
- mapped_no_aug（或 N/A_ALREADY_DISABLED）
- official_ema_stage
- mapped_ema_stage
- mapping_notes

全部 resolved config 生成后：

1. 运行自动化配置完整性检查脚本（覆盖本节全部产物，代替人工逐项审查）；
2. 计算 SHA-256；
3. 冻结配置；
4. 才允许启动训练。

训练开始后不得修改调度节点、batch/LR 策略或增强。

如果配置错误需要修改：

- 将原运行标记 invalid_protocol；
- 保存原日志；
- 修改配置；
- 重新计算 manifest；
- 从官方预训练权重重新开始；
- 不得从旧调度 checkpoint 接着训练。

# 十五、checkpoint 三层语义

必须实现三类 checkpoint：

## 15.1 last checkpoint

每个 epoch 结束后覆盖更新：

- last.*
- 保存模型；
- optimizer；
- scheduler；
- GradScaler；
- EMA；
- 当前 epoch；
- global optimizer step；
- 随机数状态（尽力保存；框架原生不支持时不强行改造，在 status.json 中记录 rng_state_saved=false 即可）；
- resolved config hash。

用途：断点续训。

故障恢复最多丢失一个未完成 epoch，不得最多丢 10 epoch。

## 15.2 best checkpoint

每次验证集 mAP@0.5:0.95 刷新时立即更新：

- best.*
- best epoch
- best metric
- checkpoint SHA-256

用途：最终冻结和 test。

best epoch 可以是 1～100 中任意 epoch，不要求是 10 的倍数。

## 15.3 周期审计 checkpoint

每 10 epoch 额外保存：

- epoch_010.*
- epoch_020.*
- …
- epoch_100.*

用途：审计学习过程，不用于替代 last 或 best。

必须检查每个模型是否真正实现三层语义。框架原生不支持时，增加薄适配层。

# 十六、独立于 agent 会话的训练守护管线

训练调度器必须实现为独立脚本或守护进程，例如：

- scheduler/run_scheduler.py
- scheduler/start_scheduler.sh
- scheduler/monitor_scheduler.py
- scheduler/recover_scheduler.py

调度器不得依赖当前 agent 会话持续存活。

状态唯一事实源：

- scheduler/queue.json
- scheduler/pipeline_status.json
- scheduler/gpu_0_state.json
- scheduler/gpu_1_state.json
- runs/<model>/status.json
- 实际 PID
- 实际 GPU 进程
- checkpoint 文件

agent 会话重新进入时必须：

1. 读取 queue.json；
2. 读取各 status.json；
3. 检查 PID 是否存在；
4. 检查 nvidia-smi；
5. 检查 checkpoint；
6. 检查日志更新时间；
7. 识别 running/completed/failed/resume_required；
8. 不重复启动仍在运行的任务；
9. 从磁盘事实恢复监控。

合法的阶段性汇报状态包括：

“训练管线已自主运行且可恢复，目前任务仍在后台执行。”

但这不等于最终完成。最终完成仍需满足全部完成条件。

调度器必须：

- 使用文件锁或原子状态更新；
- 防止同一模型被两个 GPU 重复领取；
- 定期写 heartbeat；
- 进程异常退出时标记 resume_required；
- 自动从 last checkpoint 恢复；
- 保存每次启动、退出、恢复日志。

# 十七、当前服务器 GPU 调度

服务器：

- 物理 GPU 0：GRID A100DX-80C，80 GB，约 76 GB 可用，NUMA 0
- 物理 GPU 1：GRID A100DX-40C，40 GB，约 38 GB 可用，NUMA 1
- 两卡拓扑：SYS

禁止：

- DDP
- DataParallel
- 单模型跨 GPU
- 单 GPU 同时两个训练进程

## 17.1 初始任务

GPU 0：

- Faster R-CNN R50-FPN

GPU 1：

- RT-DETR-R18

## 17.2 剩余 FIFO 队列

1. DEIM-D-FINE-N
2. YOLOv9-T
3. YOLOv7-tiny
4. YOLO12n
5. YOLO26n
6. YOLOv10n
7. YOLO11n
8. YOLOv8n
9. YOLOv5n

## 17.3 显存预估

正式训练前对每个模型做不写入正式结果的短 dry-run：

- 加载正式模型；
- 使用 imgsz=640；
- 使用计划 micro-batch；
- 完成前向、loss、反向；
- 额外执行一次 val batch 前向，使峰值估计覆盖训练期间的验证阶段；
- 不更新正式 checkpoint；
- 清理显存；
- 记录峰值显存。

dry-run 属于预检，不属于正式训练，允许在配置冻结完成前执行，但关键参数（imgsz、micro-batch、模型结构）必须与 resolved_config 一致。

dry-run 估计仍可能低于长期训练的真实峰值（allocator 增长、EMA 副本等），4 GB 安全余量正是为此预留，不得因 dry-run 通过就取消余量。

生成：

preflight/gpu_memory_estimates.csv

至少包含：

- model
- planned_micro_batch
- planned_accumulation
- estimated_peak_vram_gb
- gpu_used_for_estimate
- safety_margin_gb
- gpu1_eligible

GPU 1 可用显存按当前空闲显存减去至少 4 GB 安全余量判断。

## 17.4 确定性领取规则

GPU 0：

- 空闲时领取 FIFO 队首任务。

GPU 1：

1. 从 FIFO 队首开始扫描；
2. 领取第一个 `gpu1_eligible=true` 的 pending 任务；
3. 如果队首任务显存预估超过 GPU 1 安全容量，跳过该任务；
4. 被跳过任务保持 pending，等待 GPU 0；
5. 记录 skipped_for_vram；
6. 不改变其在 GPU 0 队列中的优先级；
7. 其余情况下严格 FIFO。

这样替代含糊的“GPU 0 优先重模型”。

如果 GPU 1 实际 OOM：

1. 保存 OOM 日志；
2. 降低 micro-batch；
3. 增加 accumulation；
4. 保持 effective batch=16；
5. 更新显存估计；
6. 从 last checkpoint 恢复；
7. 不直接判定模型失败。

## 17.5 CUDA_VISIBLE_DEVICES

使用 CUDA_VISIBLE_DEVICES=1 后，程序内部通常看到 cuda:0。

日志必须同时记录：

- physical_gpu_id
- CUDA_VISIBLE_DEVICES
- visible_device
- GPU profile

不得把逻辑 cuda:0 当成物理 GPU 0。

## 17.6 NUMA

numactl 已存在时：

GPU 0：

CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 <command>

GPU 1：

CUDA_VISIBLE_DEVICES=1 numactl --cpunodebind=1 --membind=1 <command>

如果 numactl 不存在或绑定失败：

- 不安装；
- 回退为 CUDA_VISIBLE_DEVICES；
- 记录原因；
- 不视为模型失败。

## 17.7 训练耗时

不同 vGPU 上的 train_hours：

- 只作为工程记录；
- 必须同时记录 GPU profile；
- 不参与模型效率排名；
- 不直接横向比较。

# 十八、输出目录

输出到：

/home/admin1/Projects/ultralytics-main/detection_model_benchmark/results/fruitfly_detection_v1/benchmark_640_e100_s42

至少包含：

benchmark_640_e100_s42/
├── protocol.yaml
├── protocol_decisions.md
├── configuration_manifest.sha256
├── scheduler_mapping.csv
├── batch_lr_mapping.csv
├── preflight/
│   ├── offline_readiness.csv
│   └── gpu_memory_estimates.csv
├── scheduler/
│   ├── queue.json
│   ├── pipeline_status.json
│   ├── gpu_0_state.json
│   ├── gpu_1_state.json
│   └── scheduler.log
├── runs/
│   ├── 01_yolov5n/
│   ├── 02_yolov7_tiny/
│   ├── 03_yolov8n/
│   ├── 04_yolov9_t/
│   ├── 05_yolov10n/
│   ├── 06_yolo11n/
│   ├── 07_yolo12n/
│   ├── 08_yolo26n/
│   ├── 09_deim_dfine_n/
│   ├── 10_rt_detr_r18/
│   └── 11_faster_rcnn_r50_fpn/
└── summary/

每个模型保存：

- command.sh
- official_schedule.yaml
- mapped_schedule_e100.yaml
- resolved_config.yaml
- batch_lr_resolution.yaml
- lr_schedule.csv
- lr_schedule.png
- environment.txt
- source_commit.txt
- weight_sha256.txt
- train.log
- metrics_history.csv
- optimizer_steps.csv
- gpu_assignment.json
- status.json
- checkpoints/last.*
- checkpoints/best.*
- checkpoints/epoch_010.*
- …
- checkpoints/epoch_100.*
- checkpoint_selection.json
- convergence_check.json
- predictions/（含 val_mixed.json 和四个 test 域的预测）
- evaluation/
- speed/

# 十九、训练和验证

1. 只使用 train 训练；
2. 每个 epoch 使用 val 评价；
3. 每个 epoch 记录 mAP@0.5:0.95；
4. 所有 mAP 统一以 0～1 尺度保存；
5. 完成 100 epoch；
6. 不使用 test；
7. best checkpoint 由 val 选择；
8. 不因某模型仍改善而延长主表训练。

# 二十、收敛检查

定义：

previous_best =
epoch 71～90 的最高 val mAP@0.5:0.95

final_best =
epoch 91～100 的最高 val mAP@0.5:0.95

delta =
final_best - previous_best

所有数值使用 0～1 尺度。

0.003 表示 0.3 个百分点。

同时记录：

global_best =
epoch 1～100 的最高 val mAP

global_best_epoch =
global_best 所在 epoch

判定：

1. delta > 0.003：
   - status=still_improving_at_e100

2. delta <= 0.003 且 global_best_epoch >= 71：
   - status=basic_converged

3. delta <= 0.003 且 global_best_epoch < 71：
   - status=early_peak_or_overfit
   - 不得仅根据 delta 声称模型已稳定收敛
   - 必须检查 71～100 是否持续退化、平台震荡或无恢复

4. validation 历史不完整：
   - status=insufficient_validation_history

convergence_check.json 至少包含：

- metric_scale
- previous_best
- previous_best_epoch
- final_best
- final_best_epoch
- delta
- global_best
- global_best_epoch
- early_global_peak
- convergence_status
- curve_interpretation
- notes

不得查看 test 后判断收敛。

# 二十一、checkpoint 冻结与延长候选时序

前提：11 个模型全部达到终态——completed，或在合理修复尝试后明确标记 failed（含 env_conflict_blocked、protocol_resolution_blocked 等已汇报的阻塞终态）。

failed 模型不参与冻结和 test，但不得阻塞其余模型进入以下流程。

达到前提后，按以下顺序执行：

1. 生成 convergence_status.csv；
2. 根据 val、复杂度和收敛状态选择 2～3 个延长候选；
3. 生成 extension_candidates_pretest.csv；
4. 记录生成时间；
5. 计算文件 SHA-256；
6. 生成 frozen_checkpoints.csv；
7. 计算 checkpoint SHA-256；
8. 写入 test_access_log.json；
9. 确认此前没有读取 test 指标；
10. 才允许开始 test。

extension_candidates_pretest.csv 不得使用 test 结果。

完成条件中必须保存上述顺序证据。

# 二十二、统一测试和 NMS 语义

每个冻结 best checkpoint 评测：

- mixed
- C
- R1
- R2

统一要求：

- 输出 canonical 0-based COCO prediction JSON；
- image_id 与真值一致；
- 中央 pycocotools evaluator；
- max_det=300；
- 低导出阈值，例如 0.001；
- 不根据 test 调参。

对于使用 NMS 的模型，记录：

- NMS IoU
- score threshold
- max_det

对于 NMS-free 模型，例如适用的：

- YOLOv10
- YOLO26
- RT-DETR
- DEIM

记录：

- nms_iou=N/A
- postprocess_type=NMS_FREE
- max_det=300
- selection_rule=top_300_scored_predictions 或等价实际规则

不得为了填写表格而给 NMS-free 模型虚构 NMS IoU。

# 二十三、统一指标

Mixed、C、R1、R2 分别输出：

- mAP@0.5:0.95
- AP50
- AP75
- AR
- 每类别 AP

另外输出：

- worst-domain mAP
- val 最佳 F1 阈值
- 冻结阈值后的 test Precision
- Recall
- F1

R1 correcta AP=N/A。

## 23.1 F1 阈值协议

- 使用冻结 best checkpoint 在 mixed val（1,530 张）上生成预测，保存为 predictions/val_mixed.json；
- 该操作只使用 val，发生在 checkpoint 冻结之后、test 之前，属于合法操作；
- 匹配判据：同类别、IoU>=0.5、按 score 降序贪心匹配；
- 置信度阈值在 0.01～0.99 范围内以不大于 0.01 的步长扫描，取 F1 最高值对应的阈值；
- 冻结该阈值后应用于 mixed test，报告 Precision、Recall、F1；
- 记录扫描粒度、IoU 判据和匹配实现。

所有主结果使用中央 evaluator，不直接混用各框架原生 mAP。

# 二十四、统一 GFLOPs 口径

不得混合使用：

- Ultralytics 内置 FLOPs
- THOP
- fvcore
- ptflops
- 各论文表格中的公开数值

作为同一主表指标。

创建一个统一复杂度统计脚本：

_common/measure_complexity.py

统一口径：

- batch=1
- 输入张量等价于 3×640×640
- 只统计模型前向
- 不统计图片解码
- 不统计 NMS
- 不统计文件 I/O
- multiply-add 按 2 FLOPs 计算
- 使用同一套统计实现和报告格式

统计实现选择：

- 首选纯 PyTorch forward-hook 计数实现，不依赖第三方库；
- 也可以统一选用某一个第三方计数工具（如 fvcore 或 thop）；若所需依赖在部分 Conda 环境缺失，按第六节 6.1 环境修改政策补装并记录版本；
- 无论哪种实现，11 个模型必须使用同一实现、同一版本、同一口径；
- 预检阶段必须先确认所选实现在三个 Conda 环境中全部可用，再进入正式统计；
- 统计脚本可以通过适配器加载不同框架模型，但核心计数逻辑必须一致。

必须报告：

- parameters
- FLOPs
- MACs（如果工具提供）
- unsupported_ops
- counted_ops_coverage
- tool_name
- tool_version
- multiply_add_convention

如果统一工具不能可靠统计某模型：

- GFLOPs=N/A
- 保存失败日志和 unsupported ops
- 不得改用另一工具单独为该模型填入主排名列
- 可以在附录另列 framework_reported_GFLOPs，但不得与统一 GFLOPs 混为一列

# 二十五、统一速度测试

所有最终速度测试都在：

- 物理 GPU 0
- GRID A100DX-80C
- CUDA_VISIBLE_DEVICES=0

测速期间：

- GPU 0 无其他任务；
- GPU 1 也停止训练、验证和数据加载；
- 两张卡均空闲；
- 不并行测速；
- 避免 CPU/磁盘高负载。

协议：

- batch=1
- imgsz=640
- warmup=50
- 正式测量至少 300 次
- CUDA synchronize
- 同一测试样本或同形状输入
- 排除磁盘读取
- 记录预处理、前向、后处理和端到端延迟
- 报告 mean、median、P90、P95、std

分别测量：

- FP32
- FP16

Faster R-CNN 的 FP16 允许使用：

torch.autocast(device_type="cuda", dtype=torch.float16)

不要求直接对整个模型调用 `.half()`。

必须记录每个模型 FP16 实现方式：

- model.half
- autocast
- framework_native_amp
- unsupported

如果某模型无法安全 FP16：

- FP16=N/A
- 保存错误；
- 不得伪造结果；
- 不影响 FP32 主测试。

禁止把 GPU 0 和 GPU 1 的 FPS 混入同一排名。

不得把服务器 A100 速度写成树莓派速度。

# 二十六、160 epoch 收敛性复验计划

100 epoch 主表不得被延长结果替换。

如果存在 still_improving_at_e100：

- 保留完整 100 epoch 主表；
- 根据 test 前冻结的 extension_candidates_pretest.csv 选择 2～3 个候选；
- 创建：

/home/admin1/Projects/ultralytics-main/detection_model_benchmark/results/fruitfly_detection_v1/e160_convergence_extension/extension_plan.yaml

当前任务只创建计划，不自动执行 160 epoch。

正式 e160 复验时：

- 从相同官方预训练权重重新开始；
- 重新把调度器映射到 160 epoch；
- 不直接续跑已经在 epoch 100 完成全部压缩调度的 checkpoint；
- 与 100 epoch 主表分开报告。

从 e100 checkpoint 继续低学习率微调只能叫：

exploratory_continuation

不能称为公平的 160 epoch 固定预算实验。

# 二十七、结果汇总

生成：

1. benchmark_results.csv
2. benchmark_results.md
3. per_domain_metrics.csv
4. per_class_metrics.csv
5. efficiency_metrics.csv
6. convergence_status.csv
7. extension_candidates_pretest.csv
8. extension_candidates_pretest.sha256
9. frozen_checkpoints.csv
10. test_access_log.json
11. image_id_validation.json
12. failures_and_repairs.md
13. environment_changes.md
14. reproducibility_manifest.json
15. training_curves.png
16. learning_rate_schedules.png
17. accuracy_efficiency_tradeoff.png
18. final_recommendation.md

主表至少包含：

- model
- status
- best_epoch
- convergence_status
- convergence_delta
- global_best_epoch
- mixed_mAP50_95
- mixed_AP50
- mixed_AP75
- C_mAP50_95
- R1_mAP50_95
- R2_mAP50_95
- worst_domain_mAP
- dorsalis_AP
- correcta_AP
- params_M
- unified_GFLOPs
- FLOPs_coverage
- checkpoint_MB
- FP32_latency_ms
- FP32_FPS
- FP16_method
- FP16_latency_ms
- FP16_FPS
- peak_VRAM_MB
- train_hours
- actual_optimizer_steps
- train_gpu_id
- train_gpu_profile
- micro_batch
- gradient_accumulation
- effective_batch
- official_reference_batch
- official_base_lr
- resolved_base_lr
- seed

train_hours 不参与不同 vGPU 间的效率排名。

# 二十八、最终报告必须声明的已知偏差

final_recommendation.md 必须明确说明：

1. 本轮只有 seed=42；
2. 本轮是固定 100 epoch 预算；
3. 100 epoch 不保证所有模型充分收敛；
4. 各模型保留官方优化器类型，但 batch、LR 和调度阶段按统一协议归一化（含对 AdamW 系模型统一采用线性 LR 缩放）；
5. 统一关闭强增强可能对部分 YOLO 系列影响更大；
6. 不同 vGPU 上的训练时间不可直接排名；
7. 推理速度统一在 GPU 0 上测量；
8. 服务器速度不是树莓派速度；
9. R1 没有 correcta，因此该域 correcta AP=N/A；
10. 当前结果用于候选筛选，不代表各模型官方最佳实践上限。

# 二十九、完成条件

只有满足以下条件才能宣布最终完成：

1. 11 个模型都有明确状态；
2. 成功模型完成 100 epoch，失败模型有明确失败原因记录；
3. 每 epoch validation 历史完整；
4. actual effective batch 已验证；
5. YOLO 隐藏 nbs/accumulation 已处理；
6. base LR 策略已冻结；
7. 每个模型实际生效的 optimizer 类型、base LR、momentum 已与 resolved_config 核对一致；
8. 调度器已映射到 100 epoch；
9. 三层 checkpoint 语义已实现；
10. last 每 epoch 更新；
11. best 在 val 刷新时更新；
12. 周期 checkpoint 每 10 epoch 保存；
13. 收敛状态已计算；
14. extension_candidates_pretest 在 test 前冻结；
15. test_access_log 能证明时序；
16. frozen checkpoints 已生成；
17. test 只在冻结后运行；
18. image_id 已通过检查；
19. mixed、C、R1、R2 使用明确 canonical 真值；
20. NMS-free 模型没有虚构 NMS 参数；
21. 统一复杂度工具已使用；
22. 最终速度全部在 GPU 0 测量；
23. Faster R-CNN FP16 实现方式已记录；
24. 没有比较不同 vGPU 的训练耗时；
25. 调度守护程序不依赖 agent 会话；
26. 会话恢复不会重复启动任务；
27. verify_all.sh 最终 PASS；
28. 数据验证最终 PASS；
29. 所有环境修改已记录在 environment_changes.md，核心训练依赖未被静默变更；
30. 没有把 e160 结果混入 e100 主表；
31. 没有把服务器结果写成树莓派结果；
32. 没有把单 seed 筛选写成论文最终结论。

# 三十、项目状态更新

完成后更新：

/home/admin1/Projects/ultralytics-main/research_docs/当前进展与下一步.md

只记录：

- 实际完成的训练；
- 真实指标；
- 收敛状态；
- 实际 GPU 调度；
- 实际失败和修复；
- checkpoint 和结果路径；
- 推荐的下一轮候选。

不得修改第一篇论文已冻结结果。

不得把运行中的任务写成已完成。

# 三十一、阶段性汇报和最终汇报

如果 agent 会话即将结束但后台任务仍在运行，阶段性汇报必须包含：

- 守护调度器 PID；
- queue.json 路径；
- pipeline_status.json 路径；
- 当前两个 GPU 的任务；
- 已完成模型；
- running 模型；
- pending 模型；
- failed/resume_required 模型；
- 最近 checkpoint；
- 后续会话恢复命令。

合法阶段性结论：

“自主训练管线已启动并可从磁盘状态恢复，当前仍在执行。”

不得在此时声称全部实验完成。

最终汇报必须包含：

1. 11 模型状态表；
2. effective batch 和实际 optimizer steps；
3. LR 与 scheduler 映射；
4. 三层 checkpoint 状态；
5. 训练 GPU 和 vGPU profile；
6. 收敛状态；
7. mixed、C、R1、R2 指标；
8. 每类别指标；
9. image_id 验证；
10. 参数量和统一 GFLOPs；
11. GPU 0 上统一 FP32/FP16 速度；
12. 失败和修复；
13. 环境修改记录；
14. 所有日志、checkpoint、预测和汇总路径；
15. 数据集及配置指纹；
16. 推荐进入 e160、多随机种子和树莓派实测的 2～3 个候选。

在完成所有安全且当前可执行的工作前，不要只回复计划。先完成预检、配置冻结和独立守护调度器，再启动训练。训练启动后允许以“后台自主运行且可恢复”作为阶段性交接，但最终完成必须等待全部实验和汇总真正结束。
