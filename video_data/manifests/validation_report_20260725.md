# 视频目录预处理验证报告

> 验证日期：2026-07-25
> 数据范围：`video_data/` 下 6 个逻辑长视频目录及其 25 个 MP4 片段

## 验证结果

- 逻辑长视频：6；
- 物理视频片段：25；
- 总时长：179,786.922060 秒（49:56:26.922）；
- 总帧数：10,007,059；
- 总字节数：175,001,208,713；
- 视频编码：25/25 为 H.264；
- 分辨率：25/25 为 2304×1296；
- 媒体探测：25/25 可读取，帧数和时长均大于 0；
- 旧式 `output.mp4`、`two-class_*.mp4` 文件名残留：0；
- `long_videos.csv`、`segments.csv`、`ground_truth_v1.csv`、`diagnostic_subgroups_v1.csv`、`splits_v1.csv` 的行数、路径、片段编号、媒体规模、目录名真值、子组求和和物种真值交叉校验：PASS；
- `main.py`、`main_optimized.py`、`counting_experiments.py` Python 语法检查及 `run_counting_ablation.sh` Bash 语法检查：PASS。
- `tests/test_counting_experiments.py`：20/20 项行为测试通过。

## 无损重命名证明

预处理前后 MP4 数量均为 25。同一文件系统内逐文件核对设备号、inode 和字节大小，25/25 均保持一致，因此本次操作没有复制、转码、裁剪或修改视频内容。

## 尚未冻结事项

- 6 个逻辑长视频的 development/validation/test 划分仍为 `unassigned`；
- 目录级数字已记录为用户确认的跨片段累计人工总数，但累计进入、净跨界或末帧占用这三种精确定义仍待正式冻结；
- 无法从原名称恢复的片段真实拍摄先后仍未核实；当前 `segment_index` 只提供确定性执行顺序。
