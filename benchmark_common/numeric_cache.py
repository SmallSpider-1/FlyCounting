import hashlib
import json
import math
import os
from pathlib import Path


CACHE_VERSION = 2
CACHE_FIELDS = {
    "detections": ["x1", "y1", "x2", "y2", "confidence", "class_id"],
    "tracks": ["x1", "y1", "x2", "y2", "track_id", "confidence", "class_id", "detection_index"],
}

# 写入时按字段固定小数位，避免不同 NumPy/设备产生的浮点尾数差异改变 payload 哈希。
# 精度为 0 的字段本身就是整数语义，直接写成 int。
FIELD_DECIMALS = {
    "x1": 2,
    "y1": 2,
    "x2": 2,
    "y2": 2,
    "confidence": 4,
    "track_id": 0,
    "class_id": 0,
    "detection_index": 0,
}


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def quantize_value(value, decimals, field, frame_index):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"数值缓存不接受非有限数值: frame={frame_index}, field={field}, value={value!r}")
    if decimals == 0:
        rounded = math.floor(abs(number) + 0.5)
        result = int(math.copysign(rounded, number)) if rounded else 0
        if abs(number - result) > 1e-6:
            raise ValueError(
                f"数值缓存整数字段收到非整数值: frame={frame_index}, field={field}, value={value!r}"
            )
        return result
    scale = 10**decimals
    # 固定“四舍五入远离零”，不使用 Python round 的银行家进位，保证跨平台一致。
    scaled = math.floor(abs(number) * scale + 0.5)
    return math.copysign(scaled, number) / scale if scaled else 0.0


def quantize_row(row, fields, frame_index):
    return [quantize_value(value, FIELD_DECIMALS[field], field, frame_index) for value, field in zip(row, fields)]


def stable_hash(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class NumericCacheWriter:
    def __init__(self, path, kind, metadata):
        if kind not in CACHE_FIELDS:
            raise ValueError(f"不支持的数值缓存类型: {kind}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_path = self.path.with_name(f".{self.path.name}.tmp")
        self.frames_written = 0
        self.closed = False
        core_header = {
            "cache_version": CACHE_VERSION,
            "kind": kind,
            "fields": CACHE_FIELDS[kind],
            "field_decimals": {field: FIELD_DECIMALS[field] for field in CACHE_FIELDS[kind]},
            **metadata,
        }
        self.cache_id = stable_hash(core_header)
        self.header = {**core_header, "cache_id": self.cache_id}
        self.handle = open(self.temp_path, "w", encoding="utf-8")
        self.handle.write(
            json.dumps({"header": self.header}, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        )

    def write_frame(self, frame_index, rows):
        if self.closed:
            raise RuntimeError("不能向已关闭的数值缓存写入数据。")
        expected_frame_index = self.frames_written + 1
        if int(frame_index) != expected_frame_index:
            raise ValueError(f"数值缓存帧号必须连续: 期望 {expected_frame_index}，实际 {frame_index}")
        expected_columns = len(self.header["fields"])
        if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != expected_columns for row in rows):
            raise ValueError(f"数值缓存数据列数不匹配: 期望每行 {expected_columns} 列")
        fields = self.header["fields"]
        quantized = [quantize_row(row, fields, frame_index) for row in rows]
        key = "d" if self.header["kind"] == "detections" else "t"
        record = {"f": int(frame_index), key: quantized}
        self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        self.frames_written += 1

    def close(self):
        if self.closed:
            return
        footer = {"footer": {"cache_id": self.cache_id, "frames_written": self.frames_written, "complete": True}}
        self.handle.write(json.dumps(footer, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        os.replace(self.temp_path, self.path)
        self.closed = True

    def abort(self):
        if self.closed:
            return
        self.handle.close()
        self.temp_path.unlink(missing_ok=True)
        self.closed = True


def read_cache_header(path, expected_kind=None):
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        first_line = handle.readline()
    if not first_line:
        raise ValueError(f"空数值缓存: {path}")
    try:
        header = json.loads(first_line)["header"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"数值缓存头无效: {path}") from exc
    if header.get("cache_version") != CACHE_VERSION:
        raise ValueError(
            f"数值缓存版本不兼容: {path}，当前={CACHE_VERSION}，文件={header.get('cache_version')}"
        )
    kind = header.get("kind")
    if kind not in CACHE_FIELDS:
        raise ValueError(f"数值缓存类型无效: {path}, kind={kind!r}")
    if header.get("fields") != CACHE_FIELDS[kind]:
        raise ValueError(f"数值缓存字段定义不兼容: {path}")
    expected_decimals = {field: FIELD_DECIMALS[field] for field in CACHE_FIELDS[kind]}
    if header.get("field_decimals") != expected_decimals:
        raise ValueError(f"数值缓存精度策略不兼容: {path}")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"数值缓存类型不匹配: 期望 {expected_kind}，实际 {header.get('kind')}")
    header_without_id = {key: value for key, value in header.items() if key != "cache_id"}
    if header.get("cache_id") != stable_hash(header_without_id):
        raise ValueError(f"数值缓存头指纹校验失败: {path}")
    return header


def iter_cache_frames(path, expected_kind=None, require_complete=True):
    path = Path(path)
    header = read_cache_header(path, expected_kind)
    expected_key = "d" if header["kind"] == "detections" else "t"
    expected_columns = len(header["fields"])
    frames_read = 0
    footer = None
    previous_frame_index = 0
    with open(path, encoding="utf-8") as handle:
        next(handle)
        for line_number, line in enumerate(handle, 2):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"数值缓存第 {line_number} 行不是合法 JSON: {path}") from exc
            if "footer" in record:
                if footer is not None:
                    raise ValueError(f"数值缓存包含重复尾记录: {path}")
                footer = record["footer"]
                continue
            if footer is not None:
                raise ValueError(f"数值缓存尾记录后仍有数据: {path}")
            if "f" not in record or expected_key not in record:
                raise ValueError(f"数值缓存第 {line_number} 行字段不完整: {path}")
            frame_index = int(record["f"])
            if frame_index != previous_frame_index + 1:
                raise ValueError(
                    f"数值缓存帧号不连续: {path}，期望 {previous_frame_index + 1}，实际 {frame_index}"
                )
            rows = record[expected_key]
            if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != expected_columns for row in rows):
                raise ValueError(f"数值缓存第 {line_number} 行数据列数不匹配: {path}")
            frames_read += 1
            previous_frame_index = frame_index
            yield frame_index, rows
    if require_complete:
        if footer is None or not footer.get("complete"):
            raise ValueError(f"数值缓存未完整写出: {path}")
        if footer.get("cache_id") != header.get("cache_id"):
            raise ValueError(f"数值缓存头尾指纹不一致: {path}")
        if int(footer.get("frames_written", -1)) != frames_read:
            raise ValueError(f"数值缓存帧数校验失败: {path}")
