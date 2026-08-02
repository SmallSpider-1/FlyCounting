# LV006 短案例人工审核指南

本指南只说明如何填写 `review_sheet.csv`。技术实现、误差公式和运行命令见 [README.md](README.md)。

> 只审核 2026-08-02 重建后的 renderer v4 案例，即当前 `cases/` 和 `contact_sheet.html`。旧版短片画面错位，相关结论已作废，不要沿用。

## 1. 审核目标

每个案例首先判断两件事：

1. 真实实蝇是否穿过黄色圆，以及穿越的物种和方向；
2. 程序记录的正式事件是否与真实跨界一一对应。

审核时不要只看彩色框，也不要把彩色框当作真实实蝇。需要分别检查：

- 有真实实蝇跨界，但没有对应正式事件：属于真实跨界漏计；
- 没有真实实蝇跨界，但彩色框在空白、污点或反光处并产生正式事件：属于非真实正式事件；
- 同时存在真实跨界漏计和空白框假事件：两种情况都要在 `notes` 中说明，不能因为净计数碰巧抵消就判为正确；
- 彩色框确实跟踪实蝇，但正式事件的方向或物种错误：再判断为方向错误或物种错误。

彩色框表示“当前案例关联的程序轨迹”，不是真值；灰框表示同帧其他程序轨迹。框可以跟踪实蝇，也可能跟踪污点、反光或空白区域。不要根据框的位置或 D/C 标签直接判断真实物种，必须以原画面中的真实实蝇和正式事件为准。

## 2. 打开和填写

1. 在 `contact_sheet.html` 中打开短片并记住 `case_id`；
2. 用 LibreOffice Calc 打开 `review_sheet.csv`；
3. 按 `Ctrl+F` 搜索 `case_id`；
4. A--M 列是程序字段，不得修改；
5. 第一轮只填写 N--W 列中的人工字段；R、S 两列 override 留空；X--AA 二审字段留空。

固定选项必须填写下文规定的英文值。`reviewer` 和 `notes` 可以写中文。无法用固定选项准确表达时，不得自造值，应选 `uncertain` 并在 `notes` 说明。

## 3. 判断顺序

1. 先看原画面，不参考彩色框标签，确认真实实蝇的数量、物种、是否穿过黄色圆以及穿越方向；
2. 逐次记录真实跨界：物种（D/C）、方向（`enter`/`exit`）和大致发生时间；
3. 再查看视频底部显示的正式事件，记录事件发生时间、事件方向和程序判定的物种；
4. 将视频底部的正式事件与表中的 `formal_event_rows` 逐条核对，确认事件数量、方向、物种和时间是否一致；
5. 用表中的 `formal_delta_D`、`formal_delta_C` 与视频底部事件的净变化交叉检查，但不能只看净值；
6. 判断每条正式事件是否对应原画面中的同一只真实实蝇，并区分真实跨界、漏计、空白框假事件、方向错误和物种错误；
7. 最后根据全部观察结果选择一个 `review_outcome`，无法用一个结果准确表达时按复合错误处理，填写 `uncertain` 并在 `notes` 中逐条说明。

结合字段判断时：

- 原画面用于确认真实实蝇、真实物种、跨界次数和方向；
- 视频底部正式事件用于确认程序实际报告了什么事件及其发生时机；
- `formal_event_rows` 用于逐条核对正式事件，而不是只看汇总结果；
- `formal_delta_D` 和 `formal_delta_C` 用于检查 D、C 的净计数变化，只能作为对账辅助，不能替代逐条事件核查；
- 彩色框、轨迹 ID 和 D/C 标签只能作为定位线索，不能单独证明真实物种或真实跨界。

`formal_delta=0` 不等于没有正式事件：一次 `enter +1` 和一次 `exit -1` 可能互相抵消，必须同时检查 `formal_event_rows`、视频底部事件及其发生时间。即使净值与真实计数碰巧一致，也要确认每条正式事件是否确实对应真实实蝇；真实漏计与空白框假事件可能在数值上互相抵消，不能因此判为 `formal_correct`。

方向定义：

- `enter`：圆外进入圆内，真实计数 `+1`；
- `exit`：圆内离开圆外，真实计数 `-1`。

物种定义：

- `dorsalis`：橘小实蝇，D；
- `correcta`：番石榴果实蝇，C。

## 4. 审核结果

| `review_outcome` | 中文含义 | 使用条件 | `expected_species` / `expected_direction` |
|---|---|---|---|
| `formal_correct` | 正式计数正确 | 所有真实跨界均被正确记录；或没有真实跨界且没有正式事件 | 留空 |
| `true_crossing_missed` | 真实跨界漏计 | 能明确表达为一次真实跨界，但该实蝇没有对应正式事件 | 填真实物种和方向 |
| `formal_event_not_real` | 正式事件不是真实跨界 | 没有真实跨界，案例中的正式事件来自污点、空白、框抖动等 | 留空 |
| `wrong_direction` | 方向记反 | 正式事件对应同一只真实实蝇，但方向错误 | 填真实物种和真实方向 |
| `wrong_species` | 物种记错 | 正式事件对应同一只真实实蝇，但物种错误 | 填真实物种和真实方向 |
| `uncertain` | 无法确定或案例过于复杂 | 看不清、物种不明、多个错误无法用一组物种和方向表达 | 留空 |

状态填写：

- 结论明确：`review_status=reviewed`；
- 结论为 `uncertain`：`review_status=needs_second_review`、`confidence=low`。

`wrong_direction` 和 `wrong_species` 只适用于正式事件确实对应同一只真实实蝇。彩色框在空白处时，不属于“同一只实蝇方向或物种记错”。

## 5. 空白彩色框和复合错误

### 5.1 没有真实跨界，空白框产生正式事件

填写：

```text
review_status: reviewed
review_outcome: formal_event_not_real
expected_species:
expected_direction:
mechanism_flags: other
notes: 彩色框跟踪空白或污点，没有对应的真实实蝇跨界。
```

### 5.2 一次真实跨界漏计，同时空白框产生假事件

按真实跨界填写。假事件已保留在 `formal_delta` 中，程序会用真实 `expected_delta` 与全部正式事件比较。

```text
review_status: reviewed
review_outcome: true_crossing_missed
expected_species: dorsalis
expected_direction: enter
mechanism_flags: other
notes: 真实D进入没有对应正式事件；空白处彩色框同时产生了假事件。
```

即使假事件与真实跨界的物种和方向碰巧相同、净偏差为 0，也不能填 `formal_correct`；应记录为真实漏计与假事件相互抵消。

### 5.3 多次真实跨界或部分正式事件正确、部分错误

第一轮不要手算 override，也不要强行选择一个简单结果。填写：

```text
review_status: needs_second_review
review_outcome: uncertain
confidence: low
notes: 逐条写出看见的真实事件和可疑正式事件，例如“真实D enter一次；真实C exit一次；空白框假事件两次”。
```

以下情况均按复杂案例处理：

- 多次真实跨界无法用一个物种和方向表达；
- 一个正式事件正确，另一个正式事件虚假；
- 同时存在漏计、方向错误或物种错误；
- 无法确认真实物种、方向或跨界次数。

## 6. 机制字段

`mechanism_flags` 可以留空或多选；多个值用英文分号 `;` 分隔。

| 填写值 | 中文含义 |
|---|---|
| `first_seen_inside` | ID 首次出现时已在圆内 |
| `suspected_id_break` | 疑似同一只实蝇换了 ID |
| `repeated_crossing` | 短时间反复跨界 |
| `box_center_jitter` | 框中心抖动导致假跨界 |
| `class_change` | 同一目标在 D/C 之间变化 |
| `detection_gap` | 检测或轨迹中断 |
| `occlusion` | 遮挡或多目标重叠 |
| `other` | 污点误检等其他原因 |

示例：

```text
box_center_jitter;class_change;other
```

候选标记只是线索。短轨迹、新 ID 或类别变化不能直接证明 ID 断裂、漏检或物种错误。

## 7. 其他人工字段

| 字段 | 允许值或填写规则 |
|---|---|
| `reviewer` | 审核人姓名或代号，可写中文 |
| `confidence` | `high` 非常确定；`medium` 大致确定；`low` 不确定 |
| `notes` | 中文描述观察事实；使用 `other` 或复合错误时必须写清楚 |

第一轮不得填写：

```text
expected_delta_D_override
expected_delta_C_override
second_reviewer
second_review_outcome
second_confidence
adjudication_notes
```

## 8. 快速判断表

| 观察结果 | 填写结果 |
|---|---|
| 无真实跨界，也无正式事件 | `formal_correct` |
| 真实跨界的物种和方向均被正确记录 | `formal_correct` |
| 一次真实跨界没有对应正式事件 | `true_crossing_missed`，填写真实物种和方向 |
| 无真实跨界，但程序产生正式事件 | `formal_event_not_real` |
| 同一只真实实蝇的方向记反 | `wrong_direction`，填写真实物种和方向 |
| 同一只真实实蝇的物种记错 | `wrong_species`，填写真实物种和方向 |
| 多次跨界、多种错误混合或看不清 | `uncertain`，写明观察事实 |

## 9. 批量异常处理

如果连续多个案例的彩色框都在污点、反光或空白处：

1. 不要把框标签当作真实物种；
2. 按第 5 节分别记录真实跨界和假事件；
3. 先完成 5--10 个代表案例后暂停批量审核；
4. 报告出现比例和共同特征，检查首轮候选是否被系统性误检主导，再决定是否重排案例。

## 10. 当前表结构限制

- 没有独立的 `false_detection` 机制值，暂用 `other` 并在 `notes` 说明；
- 单选 `review_outcome` 不能完整表达多个真实跨界或部分事件正确、部分事件错误；此类案例第一轮标记 `uncertain`，后续二审或使用两个 override 字段处理；
- 数值残差闭合只能说明净计数对账，不能证明检测和跟踪机制正确。真实漏计与假事件可能在数值上互相抵消。
