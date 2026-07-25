# 项目结果目录

本目录只存放可再生成的检测、跟踪与分类计数运行结果，不存放原始视频、检测数据集、模型权重或源码。

建议的新结构为：

```text
project_results/
  detection_cache/<detector_run_id>/
  tracking_model_benchmark/<tracker_run_id>/
  counting_model_benchmark/<counting_run_id>/
```

计数消融结果应使用稳定的 `counting_run_id`，并保存检测缓存 ID、跟踪配置、计数配置、数据划分和评价输出。不要把生成结果写入 `counting_model_benchmark/` 源码目录。

2026-07-25 已按用户要求清理旧 BoxMOT 计数 CSV、已撤回 B0 轨迹缓存和旧分片诊断输出，约释放 296.6 MB。这些原始输出现已不可从本目录恢复；已提取的历史诊断摘要仍保留在 `research_docs/当前进展与下一步.md`，但不能作为当前 baseline 结果。
