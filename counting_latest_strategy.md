# 诱集器计数最新策略

本文件是新会话优先读取的策略摘要。历史细节见：

- `counting_optimization_context_1.md`
- `findings.md`
- `progress.md`

## 计数定义

统计当前留在诱集器区域内的虫子数量：

```text
进入 +1
离开 -1
```

不是累计进入次数，也不是累计出现过的唯一虫子数。

## 当前最佳结果

| 任务 | 真值 | 当前最佳 | 误差 |
|---|---:|---:|---:|
| 单类别 | 219 | 226 | +7 |
| 双类别 class0 | 61 | 52 | -9 |
| 双类别 class1 | 118 | 112 | -6 |
| 双类别总数 | 179 | 164 | -15 |

## 单类别当前策略

单类别固定 v4，不继续改参数。

核心参数：

```bash
--reentry-cooldown-seconds 1.0
--recovery-min-score 0.30
```

推荐命令模板：

```bash
CUDA_VISIBLE_DEVICES=0 /home/admin1/.conda/envs/yolov26/bin/python main.py \
  --source <单类别视频目录> \
  --weights yolo26n.pt \
  --output <单类别输出目录> \
  --device 0 \
  --half \
  --reentry-cooldown-seconds 1.0 \
  --recovery-min-score 0.30 \
  --no-video \
  --no-event-frames
```

已验证旧数据命令：

```bash
CUDA_VISIBLE_DEVICES=0 /home/admin1/.conda/envs/yolov26/bin/python main.py \
  --source video_data/one_class_219 \
  --weights yolo26n.pt \
  --output result_one_class_v4 \
  --device 0 \
  --half \
  --reentry-cooldown-seconds 1.0 \
  --recovery-min-score 0.30 \
  --no-video \
  --no-event-frames
```

## 双类别当前策略

双类别使用 v5。

核心参数：

```bash
--class-vote-weights 0:0.18,1:1.0
--entry-detected-class-override 1
--recovery-class-match-required 1
--recovery-class-min-scores 0:0.60,1:0.97
--enable-outer-start-recovery
--outer-start-max-offset 35
--outer-start-min-outward 10
--outer-start-min-frames 2
--outer-start-max-frames 30
```

推荐命令模板：

```bash
CUDA_VISIBLE_DEVICES=1 /home/admin1/.conda/envs/yolov26/bin/python main.py \
  --source <双类别视频目录> \
  --weights yolo8n_two-class.pt \
  --output <双类别输出目录> \
  --device 0 \
  --half \
  --class-vote-weights 0:0.18,1:1.0 \
  --entry-detected-class-override 1 \
  --recovery-class-match-required 1 \
  --recovery-class-min-scores 0:0.60,1:0.97 \
  --enable-outer-start-recovery \
  --outer-start-max-offset 35 \
  --outer-start-min-outward 10 \
  --outer-start-min-frames 2 \
  --outer-start-max-frames 30 \
  --no-video \
  --no-event-frames
```

已验证旧数据命令：

```bash
CUDA_VISIBLE_DEVICES=1 /home/admin1/.conda/envs/yolov26/bin/python main.py \
  --source video_data/two_class_61_118 \
  --weights yolo8n_two-class.pt \
  --output result_two_class_v5 \
  --device 0 \
  --half \
  --class-vote-weights 0:0.18,1:1.0 \
  --entry-detected-class-override 1 \
  --recovery-class-match-required 1 \
  --recovery-class-min-scores 0:0.60,1:0.97 \
  --enable-outer-start-recovery \
  --outer-start-max-offset 35 \
  --outer-start-min-outward 10 \
  --outer-start-min-frames 2 \
  --outer-start-max-frames 30 \
  --no-video \
  --no-event-frames
```

## 后台运行建议

长视频完整运行建议使用 `tmux`，避免后台进程被会话回收。

当前仓库已有双类别 v5 启动脚本：

```bash
tmux new-session -d -s v5_counting './run_v5_two_class.sh'
```

如果换新视频目录，先复制或编辑 `run_v5_two_class.sh` 中的：

```text
--source
--output
```

## 新会话建议第一步

```bash
cat counting_latest_strategy.md
git status --short
find video_data -maxdepth 2 -type d | sort
```

然后根据新上传视频目录选择单类别或双类别命令模板运行。
