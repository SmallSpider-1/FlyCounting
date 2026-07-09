# 计数优化会话上下文

这个文件用于在新会话中继续完成诱集器虫子计数优化。

项目目录：

```text
/home/admin1/Projects/ultralytics-main
```

## 任务目标

优化视频中的诱集器区域计数。

计数定义已经确认：

```text
统计当前留在诱集器区域内有多少只虫子
进入 +1
离开 -1
```

不是累计进入次数，也不是累计出现过的唯一虫子数。

## 参考真值

| 任务 | 真值 |
|---|---:|
| 单类别 | 219 |
| 双类别 class 0 | 61 |
| 双类别 class 1 | 118 |
| 双类别总数 | 179 |

## 主要文件

| 文件 | 作用 |
|---|---|
| `main.py` | 主运行脚本，接 YOLO、ByteTrack、计数器和 CSV 输出 |
| `track_counting.py` | 新增的核心计数逻辑，包含滞回、逻辑轨迹、库存 token、恢复离开等 |
| `tests/test_track_counting.py` | 行为测试 |
| `task_plan.md` | 阶段计划 |
| `findings.md` | 诊断和实验发现 |
| `progress.md` | 过程记录 |

## 当前实现能力

- 三层区域滞回：内稳定区 / 不确定环带 / 外稳定区。
- ByteTrack ID 片段拼接为逻辑轨迹。
- 库存 token：进入创建 token，离开消费 token。
- 断轨离开恢复：`recovered_exit`。
- 类别投票权重：`--class-vote-weights`。
- 类别级恢复阈值：`--recovery-class-min-scores`。
- 只对指定类别要求恢复类别一致：`--recovery-class-match-required`。
- 环带内侧起始恢复：`--enable-band-start-recovery`。
- 外侧起始恢复：`--enable-outer-start-recovery`。
- 单类别重入冷却：`--reentry-cooldown-seconds`。
- 快速模式：`--no-video --no-event-frames`。

CSV 诊断字段包括：

- `logical_track_id`
- `event_reason`
- `recovery_mode`
- `occupancy_token_id`
- `recovered_from_logical_id`
- `recovery_score`
- `recovery_track_class_id`
- `recovery_class_match`
- `detected_cls_id`
- `voted_cls_id`

## 版本结果总览

| 版本 | 单类别 | 双类别 class0 / class1 | 主要结论 |
|---|---:|---:|---|
| 原方法 | 246 | 93 / 45 | 单类别稍高，双类别 class1 很低 |
| v1 | 256 | 215 / 91 | 边界抖动减少，但离开漏扣导致库存堆积 |
| v2 | 240 | 108 / 27 | 单类别改善，双类别 class1 被恢复离开扣太多 |
| v3 | 240 | 116 / 86 | class1 明显恢复，单类别无提升 |
| v4 | 226 | 25 / 99 | 单类别达标，双类别 class0 被扣过头 |

## 版本演进详情

### 原方法

**原逻辑**

- 直接使用 ByteTrack 的原始 ID。
- 用检测框中心点判断是否在圆内。
- 中心点跨圆边界就立即触发 enter/exit。
- 没有内外双阈值，没有不确定环带。
- 没有跨 ID 拼接。
- 双类别计数类别基本依赖当前帧检测类别。

**为什么效果不好**

- 事件点几乎都贴着圆周，检测框中心 1 到 5 px 漂移就会触发计数。
- 同一只虫在圆边界附近抖动，会反复 enter/exit。
- 虫子在区域内换 ID 后，后续离开无法和原来的进入对应。
- 双类别模型本身 class0 偏强、class1 偏弱，逐帧类别抖动会污染计数。

**完整结果**

| 任务 | 结果 |
|---|---:|
| 单类别 | 246 |
| 双类别 class0 / class1 | 93 / 45 |

**结论**

- 单类别比 219 高 27。
- 双类别 class0 偏高，class1 严重偏低。

### v1：三区域滞回 + 逻辑轨迹拼接

**新增内容**

- 新增 `track_counting.py`。
- 将区域分为：
  - 外部稳定区：距离 `> R + band`
  - 不确定环带：`R - band` 到 `R + band`
  - 内部稳定区：距离 `< R - band`
- 只有稳定外区到稳定内区才计 enter。
- 只有稳定内区到稳定外区才计 exit。
- 环带内不修改数量。
- 将 ByteTrack ID 当成短轨迹编号，新增逻辑轨迹 ID。
- 根据时间间隔、中心距离、速度方向、框尺寸、外观特征、类别投票拼接断裂轨迹。
- 进入时冻结类别，离开时扣同一类别。

**为什么这么改**

- 解决边界抖动造成的误计数。
- 解决 ByteTrack ID 不稳定导致同一虫子换 ID 后计数状态断裂。
- 解决双类别进入和离开类别不一致的问题。

**完整结果**

| 任务 | 结果 |
|---|---:|
| 单类别 | 256 |
| 双类别 class0 / class1 | 215 / 91 |

**效果**

- 事件数量确实下降：
  - 单类别：846 -> 578
  - 双类别：1703 -> 736
- 但最终计数变差，尤其双类别 class0 堆积严重。

**问题**

- v1 只允许“保有 counted_inside 状态的同一逻辑轨迹”触发 exit。
- 如果虫子进入后在区域内断 ID，离开片段没拼回原逻辑轨迹，就不会扣数。
- 结果是 enter 被保留，exit 被漏掉，库存不断累积。

### v2：库存 token + 断轨离开恢复

**新增内容**

- 引入 `OccupancyToken`。
- 每次可靠进入创建一个库存 token。
- 正常同 ID 或成功拼接的离开，优先消费自己的 token。
- 如果新轨迹首次稳定出现在内区，然后移动到外区，可以匹配并消费近期不可见的库存 token，记为 `recovered_exit`。
- 新增诊断字段：
  - `event_reason`
  - `occupancy_token_id`
  - `recovered_from_logical_id`
- 新增类别投票权重参数：
  - `--class-vote-weights`

**为什么这么改**

- v1 最大问题是“离开片段没有 counted_inside，所以不能扣”。
- token 机制把“当前占用库存”从某条轨迹身份中解耦出来。
- 即使离开时换了新 ID，也可以恢复消费已有库存。
- 初始就在区域内的虫子不会凭空扣数，因为没有 token 就不能 exit。

**完整结果**

| 任务 | 结果 |
|---|---:|
| 单类别 | 240 |
| 双类别 class0 / class1 | 108 / 27 |

**效果**

- 单类别从 v1 的 256 降到 240，有改善。
- 单类别 `recovered_exit = 16`。
- 双类别总数从 v1 的 306 降到 135，说明恢复离开确实生效。

**问题**

- 双类别 class1 被扣得太狠：
  - v2 class1 只有 27。
- 原因是 class1 本来进入就少，又被 recovered_exit 扣掉很多。
- class0/class1 分布仍受检测模型类别偏置影响。

### v3：类别保护 + 快速调参 + 环带起始恢复

**新增内容**

- 新增快速调参开关：
  - `--no-video`
  - `--no-event-frames`
- 新增恢复诊断字段：
  - `recovery_score`
  - `recovery_track_class_id`
  - `recovery_class_match`
  - `detected_cls_id`
  - `voted_cls_id`
- 新增恢复时要求类别一致：
  - `--recovery-require-class-match`
- 新增按类别设置恢复阈值：
  - `--recovery-class-min-scores`
- 新增环带内侧起始恢复：
  - `--enable-band-start-recovery`
- 尝试对双类别使用 class0 降权、class1 保护。

**为什么这么改**

- v2 中 class1 被 recovered_exit 扣过头。
- 所以 v3 对 class1 做保护：恢复离开必须类别一致，且 class1 恢复阈值更高。
- 单类别仍高估，所以尝试让从圆内半侧环带起步并离开的新 ID 也能恢复 exit。

**完整结果**

| 任务 | 结果 |
|---|---:|
| 单类别 | 240 |
| 双类别 class0 / class1 | 116 / 86 |

**效果**

- 单类别没有变化，环带内侧起始恢复没有带来额外有效扣减。
- 双类别 class1 从 27 提升到 86，说明类别保护有效。

**问题**

- 双类别 class0 仍高，class1 仍低：
  - class0 = 116，目标 61
  - class1 = 86，目标 118
- 总数 202，高于真值 179。
- 说明 v3 对 class1 保护方向正确，但 class0 还没有被压下来。

### v4：重入冷却 + 外侧起始恢复 + 类别级恢复匹配

**新增内容**

- 新增同一逻辑轨迹重入冷却：
  - `--reentry-cooldown-seconds`
- 新增外侧近边界起始并向外运动的恢复离开：
  - `--enable-outer-start-recovery`
  - `--outer-start-max-offset`
  - `--outer-start-min-outward`
  - `--outer-start-min-frames`
  - `--outer-start-max-frames`
- 新增只对指定库存类别要求恢复类别一致：
  - `--recovery-class-match-required`

**为什么这么改**

单类别 v3 仍然是 240。离线分析发现：

- 240 个最终库存中，有 54 个来自同一逻辑轨迹短时间 `exit -> enter` 后留下的库存。
- 这说明单类别剩余误差主要不是断轨离开没恢复，而是边界附近重复进入。
- 所以加了重入冷却：同一逻辑轨迹刚 exit 后，短时间内再次 enter 不计数。

双类别 v3 中：

- class1 从 27 拉到 86，方向对。
- class0 仍有 116，偏高。
- 所以 v4 允许只对 class1 要求类别一致，class0 恢复可以更宽松。

**完整结果**

| 任务 | 结果 |
|---|---:|
| 单类别 | 226 |
| 双类别 class0 / class1 | 25 / 99 |

**效果**

单类别：

```text
226 vs 219
误差 +7
约 96.8%
```

单类别首次达到 95% 以上，应暂时固定。

单类别事件变化：

```text
v3: 417 enter / 177 exit = 240
v4: 392 enter / 166 exit = 226
```

收益主要来自减少重复进入。

双类别：

- class1 从 86 提升到 99，继续变好。
- class0 从 116 被压到 25，过度扣减。
- 总数从 202 降到 124，低于真值 179。

**问题**

双类别 v4 中 class0 `recovered_exit` 太多：

```text
v3 class0 recovered_exit = 91
v4 class0 recovered_exit = 176
```

多个视频段 class0 出现负增长：

```text
09-41-28: class0 -6
10-41-29: class0 -3
13-36-17: class0 -1
```

离线重放 v4 事件发现：

```text
如果把 class0 恢复阈值从 0.30 提高到约 0.60：
class0 ≈ 63
class1 ≈ 99
```

这说明下一版 v5 应优先提高 class0 恢复阈值。

## 当前最佳策略

### 单类别

单类别先固定 v4，不要继续大改。

推荐参数：

```bash
--reentry-cooldown-seconds 1.0
--recovery-min-score 0.30
```

### 双类别

双类别继续做 v5。

当前已新增 v5 开关：

```text
--entry-detected-class-override
```

作用：指定类别在进入瞬间可用当前检测类别覆盖历史投票类别。用于处理 v4 中 `detected_cls_id=1` 但 `voted_cls_id=0`，导致 class 1 进入被计入 class 0 的情况。

优先目标：

- 保留 class1 保护。
- 提高 class0 恢复阈值，避免 class0 被扣到 25。
- 继续尝试提升 class1 进入归属。

v5 离线重放结果：

```text
基于 result_two_class_v4/count_events.csv
0:0.60,1:0.95 -> 约 63 / 99
0:0.60,1:0.97 + --entry-detected-class-override 1 -> 约 62 / 108
```

优先尝试：

```bash
--class-vote-weights 0:0.18,1:1.0
--entry-detected-class-override 1
--recovery-class-match-required 1
--recovery-class-min-scores 0:0.60,1:0.97
```

## v4 运行命令

### 单类别 v4

```bash
CUDA_VISIBLE_DEVICES=0 /home/admin1/.conda/envs/yolov11/bin/python main.py \
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

### 双类别 v4

```bash
CUDA_VISIBLE_DEVICES=1 /home/admin1/.conda/envs/yolov11/bin/python main.py \
  --source video_data/two_class_61_118 \
  --weights yolo8n_two-class.pt \
  --output result_two_class_v4 \
  --device 0 \
  --half \
  --class-vote-weights 0:0.18,1:1.0 \
  --recovery-class-match-required 1 \
  --recovery-class-min-scores 0:0.30,1:0.95 \
  --enable-outer-start-recovery \
  --outer-start-max-offset 35 \
  --outer-start-min-outward 10 \
  --outer-start-min-frames 2 \
  --outer-start-max-frames 30 \
  --no-video \
  --no-event-frames
```

## 建议新会话第一步

先读取：

```bash
tail -n 160 counting_optimization_context_1.md
tail -n 120 findings.md
tail -n 80 progress.md
cat result_one_class_v4/final_counts.csv
cat result_two_class_v4/final_counts.csv
```

然后针对双类别做 v5，优先试：

```bash
CUDA_VISIBLE_DEVICES=1 /home/admin1/.conda/envs/yolov11/bin/python main.py \
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

## v5 接续结果（2026-06-22）

### 已验证内容

- 单类别继续固定 v4，不做改动。
- 双类别按建议提高 class0 恢复阈值：

```bash
--recovery-class-min-scores 0:0.60,1:0.95
```

- 基于 `result_two_class_v4/count_events.csv` 离线重放：

| 设置 | class0 | class1 | 总数 |
|---|---:|---:|---:|
| v4 原结果 | 25 | 99 | 124 |
| class0 阈值 0.60 | 63 | 99 | 162 |
| class0 阈值 0.60 + `--entry-detected-class-override 1` | 62 | 100 | 162 |

结论：

- class0 阈值 0.60 明确有效，能把 class0 从 25 拉回到接近真值 61。
- `--entry-detected-class-override 1` 已接入代码和测试，但对最终 class1 净提升很小；因为被改成 class1 的进入，其对应离开 token 也会按 class1 扣减。
- class1 从 99 到 118 的剩余差距，主要不是恢复离开阈值问题，更可能来自检测模型/类别归属上限。

### 完整 v5 重跑状态

尝试完整重跑双类别 v5 两次都没有得到 CSV：

- 第一次 PTY 丢失日志，进程结束后 `result_two_class_v5` 为空。
- 第二次使用 `setsid` 写 `result_two_class_v5/run.log`，进程约 72 分钟后退出，日志停在模型 summary，无 traceback，无 CSV。
- 同参数 `--max-seconds 1` 烟雾测试成功，说明参数和代码可运行；完整运行失败更像外部杀进程或底层无日志崩溃。

### 当前推荐 v5 参数

```bash
CUDA_VISIBLE_DEVICES=1 /home/admin1/.conda/envs/yolov11/bin/python main.py \
  --source video_data/two_class_61_118 \
  --weights yolo8n_two-class.pt \
  --output result_two_class_v5 \
  --device 0 \
  --half \
  --class-vote-weights 0:0.18,1:1.0 \
  --recovery-class-match-required 1 \
  --recovery-class-min-scores 0:0.60,1:0.95 \
  --enable-outer-start-recovery \
  --outer-start-max-offset 35 \
  --outer-start-min-outward 10 \
  --outer-start-min-frames 2 \
  --outer-start-max-frames 30 \
  --no-video \
  --no-event-frames
```

可选诊断参数：

```bash
--entry-detected-class-override 1
```

但离线重放显示该参数只约提升 class1 一个计数，不建议把它当作主要优化手段。
