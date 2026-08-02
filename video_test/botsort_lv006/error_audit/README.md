# BoT-SORT / LV006 计数误差人工审核工作台

人工审核时先阅读：[LV006 短案例人工审核指南](REVIEW_GUIDE.md)。人工填写规则以该指南为准；本 README 主要记录工作台边界、计算规则和运行方式。

> **当前有效版本：renderer v4（2026-08-02）**。旧版短片因长视频随机按帧号跳转发生画面错位，已全部作废并重建；只能审核当前 `cases/` 和 `contact_sheet.html` 中的新版案例。旧版 2 条人工结论已从审核表清空并删除，没有进入账本。

## 1. 目的

本目录用于把 LV006 的长视频诊断压缩成一批可以人工快速复核的短案例。人工审核者不需要通看 7 段长视频，也不需要手工计算计数误差。

审核者只需要：

1. 打开 `cases/` 中的短视频；
2. 在 `review_sheet.csv` 中选择审核结论；
3. 勾选可能机制，填写审核人和置信度；
4. 仅在结论涉及漏计、方向或物种错误时，填写肉眼判断的真实物种与进出方向；
5. 看不清时选择“无法判断”，不要为了让账本闭合而猜测。

程序负责案例发现、去重、短视频生成、正式事件回填、误差计算和残差汇总。

## 2. 研究边界

- 逻辑视频：`LV006 / dorsalis_79-correcta_27`；
- 人工最终占用真值：dorsalis 79、correcta 27；
- 正式 BoT-SORT 结果：dorsalis 27、correcta 21；
- 全视频目标偏差：dorsalis -52、correcta -6；
- 正式圆：中心 `(1185, 732)`，半径 `215 px`；
- 类别语义：`class 0 = Bactrocera dorsalis`，`class 1 = Bactrocera correcta`；
- 偏差统一定义：`bias = formal_delta - expected_delta`；
- 修正量统一定义：`correction = expected_delta - formal_delta = -bias`。

本工作台只读取已经冻结的原视频、BoT-SORT 轨迹缓存和正式计数 CSV。不得重新运行或覆盖检测、跟踪和正式计数结果。候选标记只是定位线索，不能在人工复核前写成确定的 ID 断裂、漏检或遮挡结论。

片段 1 和片段 2 只有合并诊断真值 `D=11, C=8`，不能分别强行分配真值。片段 3--7 有文件级诊断真值。片段/分组真值当前只作为诊断目标；在其语义与净跨界量严格等价尚未确认时，残差应称为“相对于诊断目标的残差”。

## 3. 计划目录结构

```text
error_audit/
├── README.md
├── REVIEW_GUIDE.md             # 人工审核字段和判断规则
├── cases/                    # 待审核短视频；按优先级和片段分目录
├── thumbnails/               # 每个案例的锚点预览图
├── case_manifest.csv         # 程序生成的案例、来源和候选标记
├── review_sheet.csv          # 人工只编辑审核字段
├── error_ledger.csv          # 程序根据审核结果生成的误差账本
├── residual_summary.csv      # 分片段/分组和全视频的残差
├── generation_manifest.csv   # 输出完整性、帧数、时长和校验结果
└── generation.log            # 案例生成与验证日志
```

`README.md` 和 `REVIEW_GUIDE.md` 是人工维护文档；其余文件和子目录由案例生成程序创建或刷新。

## 4. 短案例应如何呈现

每个案例围绕一个可独立判断的真实活动窗口生成，建议包含锚点前 3 秒和后 4 秒。若同一目标在相邻时间内连续触发多个事件，应合并为一个案例，避免把同一次边界徘徊重复审核。

案例视频至少应显示：

- 原始画面及正式圆；
- 相关目标框、raw track ID、类别和置信度；
- 锚点帧、原视频帧号和时间戳；
- 窗口内全部正式事件的物种、方向和增量；
- `FIRST-SEEN-INSIDE`、`CLASS-CHANGED`、`REPEATED-CROSSING` 等候选标记；
- 锚点处短暂停留，并对关键跨界过程慢放；
- 必要时同时提供全画面和圆周区域放大画面，不能裁掉进出路径。

案例必须来自原视频帧；现有长视频 overlay 只能辅助定位，不能二次裁切后作为唯一人工判定证据。

## 5. 人工需要填写的字段

### 5.1 判定结果：互斥单选

- `formal_correct`：正式处理正确；
- `true_crossing_missed`：真实跨界但正式漏计；
- `formal_event_not_real`：正式事件并非真实跨界；
- `wrong_direction`：跨界方向错误；
- `wrong_species`：物种记录错误；
- `uncertain`：无法判断。

### 5.2 可能机制：多选

- `first_seen_inside`：首次观测已在圈内；
- `suspected_id_break`：疑似 ID 断裂或重建；
- `repeated_crossing`：边界反复穿越；
- `box_center_jitter`：框中心抖动；
- `class_change`：类别变化；
- `detection_gap`：检测中断或疑似漏检；
- `occlusion`：遮挡；
- `other`：其他。

机制字段允许为空或多选。看见“短轨迹”“新 ID”不等于已经证明 ID 断裂，应使用“疑似”表述。

### 5.3 其他人工字段

- `reviewer`：审核人；
- `confidence`：建议使用 `high / medium / low`；
- `expected_species`：需要纠错时选择 `dorsalis / correcta / none`；
- `expected_direction`：需要纠错时选择 `enter / exit / none`；
- `expected_delta_D_override`、`expected_delta_C_override`：仅当一个复杂案例中存在多个真实跨界、无法由单一物种和方向表达时填写；
- `notes`：必要的文字说明；
- `review_status`：`unreviewed / reviewed / needs_second_review`。

普通案例不需要手算 `expected_delta`、`bias`、`correction` 或 `residual`。这些字段由程序根据审核结论、真实物种和真实方向生成。复杂案例才使用 override 字段。

## 6. 自动计算规则

对每个去重案例，程序汇总窗口内归属于该案例的正式事件行，得到：

```text
formal_delta_D
formal_delta_C
```

再由人工结论推导：

- 正式处理正确：`expected_delta = formal_delta`；
- 真实跨界但漏计：按人工填写的真实物种和方向生成 `expected_delta`；
- 正式事件并非真实跨界：`expected_delta_D = 0, expected_delta_C = 0`；
- 方向错误：按人工确认的真实方向生成 `expected_delta`；
- 物种记录错误：按人工确认的真实物种和方向生成 `expected_delta`；
- 无法判断：expected、bias 和 correction 保持空值，不按 0 计入汇总；
- 复杂案例填写 override 时，以 override 为准，并在账本中记录来源。

随后自动计算：

```text
bias_D = formal_delta_D - expected_delta_D
bias_C = formal_delta_C - expected_delta_C
correction_D = expected_delta_D - formal_delta_D
correction_C = expected_delta_C - formal_delta_C
```

例：真实发生一次 dorsalis 进入，但正式没有事件：

```text
expected: D=+1, C=0
formal:   D= 0, C=0
bias:     D=-1, C=0
```

例：真实是 dorsalis 进入，却被正式记成 correcta 进入：

```text
expected: D=+1, C=0
formal:   D= 0, C=+1
bias:     D=-1, C=+1
```

## 7. 残差的直白含义

`target_bias` 是正式结果与诊断真值之间的总差额，`confirmed_bias` 是人工已审核案例能够解释的差额，`residual` 是仍未解释的部分：

```text
residual = target_bias - confirmed_bias
```

优先诊断目标如下：

| 片段或分组 | target_bias_D | target_bias_C |
|---|---:|---:|
| 片段 7 | -23 | -9 |
| 片段 5 | -12 | +5 |
| 片段 3 | -7 | +1 |
| 片段 6 | -2 | -6 |
| 片段 4 | -2 | +1 |
| 片段 1--2 合并 | -6 | +2 |
| LV006 全部 | -52 | -6 |

残差接近 0 表示已审核案例在数值上基本解释了差额，但不自动证明每个机制的因果关系。残差仍明显不为 0 时，应继续检查未产生正式事件的圆周活动窗口，或重新核验诊断真值的语义。禁止为了闭合账本给“无法判断”案例强行赋值。

## 8. 案例发现与审核顺序

第一轮优先顺序：片段 7 → 片段 5 → 片段 3 → 片段 6；之后再检查片段 4 和片段 1--2 合并组。

候选案例至少覆盖：

1. first-seen-inside 且随后发生退出；
2. 同一 raw ID 在短时间内反复进入/退出；
3. 事件附近类别发生变化；
4. 圆周带附近旧 ID 结束、新 ID 很快出现的疑似重建；
5. 框中心在圆周两侧快速抖动；
6. 检测或轨迹短暂中断；
7. 有明显圆周活动但没有正式事件的非事件候选；
8. 遮挡、多目标交汇等难判断窗口。

同一时间窗口、同一目标或同一组相互作用目标只能形成一个案例。`formal_event_rows` 应保留该案例对应的所有正式事件行号或稳定事件键，以便回溯。

## 9. 审核完成标准

- `review_sheet.csv` 中所有计划审核案例均为 `reviewed` 或明确标记为 `needs_second_review`；
- “无法判断”案例的 expected/bias/correction 为空；
- 账本不存在重复 formal event 归属；
- 每个案例都能回溯到原视频、帧范围、轨迹缓存和正式事件行；
- 分片段与全视频的 `confirmed_bias`、`residual` 可重复计算；
- 条件允许时，对约 20% 的案例进行第二人盲复核，并保留两次原始意见；
- 若高优先级案例审核后残差仍大，再补圆周带原视频活动窗口，而不是制作整段逐帧 MOT ID 真值。

## 10. 权威输入

- 原视频：`/home/admin1/Projects/ultralytics-main/video_data/dorsalis_79-correcta_27/`；
- 片段清单：`/home/admin1/Projects/ultralytics-main/video_data/manifests/segments.csv`；
- BoT-SORT 轨迹缓存：`/home/admin1/Projects/ultralytics-main/project_results/tracking_counting_baseline_v1/botsort/tracks/dorsalis_79-correcta_27/`；
- 正式计数 CSV：`/home/admin1/Projects/ultralytics-main/project_results/tracking_counting_baseline_v1/botsort/counts/dorsalis_79-correcta_27/`；
- 已有事件索引与诊断摘要：`/home/admin1/Projects/ultralytics-main/video_test/botsort_lv006/`；
- 现有只读诊断脚本：`/home/admin1/Projects/ultralytics-main/scripts/render_track_cache_diagnostics.py`。

任何生成程序都应先记录输入文件的大小、修改时间和必要哈希，结束后重新核对，证明正式输入未被改写。

## 11. 当前已生成批次与运行方式

截至 2026-08-02，首轮 100 个案例已用 renderer v4 全部覆盖重建。v4 先用 ffprobe 建立“原视频帧序号 → 显示时间戳（PTS）”索引，再按 PTS 读取画面；每个案例侧车文件都记录锚点 PTS 和原始锚点画面 SHA-256。独立 `verify` 已确认 100/100 个短片可解码、锚点画面与原视频一致、正式输入未变化；19 项自动测试全部通过。

首轮分配为 S007 36、S005 23、S003 15、S006 14、S004 6、S001-002 6；`case_manifest.csv` 还保留 2,689 个未渲染候选及其排名。100 个审核案例中，58 个含唯一归属的正式事件，42 个没有归属正式事件。旧版 100 个错误短片及其 2 条无效审核记录已删除。

建议先打开 `contact_sheet.html` 浏览缩略图，再按 `review_sheet.csv` 的顺序观看 `cases/` 中的短片。只填写人工字段；不要修改 `case_id`、路径、正式事件和 formal delta 等自动字段。全部或部分审核完成后运行：

```bash
/home/admin1/.conda/envs/bench_tracking/bin/python scripts/build_lv006_error_audit.py summarize
/home/admin1/.conda/envs/bench_tracking/bin/python scripts/build_lv006_error_audit.py verify
```

`summarize` 会校验审核组合并刷新 `error_ledger.csv`、`residual_summary.csv` 和 `validation_report.csv`；非法行不会静默进入 confirmed bias。`verify` 不重编码视频。重新运行 `generate` 会校验渲染签名并复用合格短片，同时按 `case_id` 保留已有人工字段。只有在渲染器错误导致现有审核结论失效时，才使用 `generate --invalidate-existing-reviews`；程序会先归档再清空人工字段。
