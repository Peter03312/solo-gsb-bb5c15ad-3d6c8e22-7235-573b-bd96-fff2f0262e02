"""跨审定窗口连续温度暴露复核纯函数测试（标准库 + pytest，无 IO）。

运行：PYTHONPATH=. python -m pytest tests/test_exposure.py -q
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.evaluation import evaluate
from app.exposure import ExposureReviewError, build_cross_window_review
from app.models import UNKNOWN_INSTRUMENT_FAULT, VERDICT_LABELS_ZH
from app.timeutils import parse_stored

BASE = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
TMIN, TMAX = 18.0, 22.0
MAX_GAP = 600.0


def iso_at(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def t(minute: float) -> datetime:
    return BASE + timedelta(minutes=minute)


def make_window(window_id: int, start_min: float, end_min: float,
                *, tmin=TMIN, tmax=TMAX, max_gap=MAX_GAP,
                max_exceed=300.0):
    return {
        "id": window_id,
        "name": f"case-{window_id}",
        "start_at": iso_at(t(start_min)),
        "end_at": iso_at(t(end_min)),
        "allowed_temp_min": tmin,
        "allowed_temp_max": tmax,
        "max_continuous_exceed_seconds": max_exceed,
        "max_sample_interval_seconds": max_gap,
    }


def reading(rid, minute: float, temp: float, rev: int = 1):
    return {
        "reading_id": rid, "revision": rev,
        "measured_at": iso_at(t(minute)), "temperature": temp,
        "status": "active", "created_at": iso_at(t(0)),
    }


def make_snapshot(window_id, start_min, end_min, readings, *, faults=None,
                  tmin=TMIN, tmax=TMAX, max_gap=MAX_GAP, max_exceed=300.0):
    win = make_window(window_id, start_min, end_min, tmin=tmin, tmax=tmax,
                      max_gap=max_gap, max_exceed=max_exceed)
    parsed = [(parse_stored(r["measured_at"]), r["temperature"]) for r in readings]
    ev = evaluate(parse_stored(win["start_at"]), parse_stored(win["end_at"]),
                  tmin, tmax, max_exceed, max_gap, parsed,
                  fault_periods=[
                      (parse_stored(f["start_at"]), parse_stored(f["end_at"]))
                      for f in (faults or [])])
    snap = {"window": win, "evaluation": ev, "readings": list(readings)}
    if faults is not None:
        snap["fault_periods"] = faults
    return snap


def fault(fid, s_min, e_min, reason="漂移"):
    return {
        "fault_id": fid, "start_at": iso_at(t(s_min)),
        "end_at": iso_at(t(e_min)), "reason": reason, "status": "active",
        "duration_seconds": (e_min - s_min) * 60.0,
        "created_at": iso_at(t(0)), "withdrawal": None,
    }


def make_cert(cid, window_id, snap):
    v = snap["evaluation"]["verdict"]
    return {
        "id": cid, "window_id": window_id, "version": 1,
        "verdict": v, "verdict_label_zh": VERDICT_LABELS_ZH[v],
        "issued_at": iso_at(t(0)), "note": None, "snapshot": snap,
    }


def statuses(result):
    return [(s["status"], s["duration_seconds"]) for s in result["segments"]]


def two_certs(readings_a, readings_b, *, faults_a=None, faults_b=None,
              tmin=TMIN, tmax=TMAX, max_gap=MAX_GAP, cid_a=1, cid_b=2):
    """两个首尾相接的 10 分钟窗口：[0,10) 与 [10,20)。"""
    sa = make_snapshot(10 + cid_a, 0, 10, readings_a, faults=faults_a,
                       tmin=tmin, tmax=tmax, max_gap=max_gap)
    sb = make_snapshot(10 + cid_b, 10, 20, readings_b, faults=faults_b,
                       tmin=tmin, tmax=tmax, max_gap=max_gap)
    return make_cert(cid_a, sa["window"]["id"], sa), make_cert(cid_b, sb["window"]["id"], sb)


# ------------------------------------------------------------ 结论与累计

def test_cross_boundary_exceeded_accumulates_and_equal_limit_passes():
    # 窗A 末段 [5,10) 超限 300s；窗B 首段 [10,15) 超限 300s -> 连续 600s
    ca, cb = two_certs(
        [reading("a", 0, 20.0), reading("b", 5, 23.0), reading("c", 10, 23.0)],
        [reading("a", 10, 23.0), reading("b", 15, 20.0), reading("c", 20, 20.0)],
    )
    res = build_cross_window_review([ca, cb], 600)
    assert res["verdict"] == "pass"
    assert res["max_continuous_exceeded_seconds"] == 600.0
    assert res["unknown_total_seconds"] == 0.0
    assert res["normal_total_seconds"] == 600.0
    assert res["exceeded_total_seconds"] == 600.0
    assert res["total_duration_seconds"] == 1200.0
    # 各窗自身最长均为 300s（不跨窗）
    assert [w["max_continuous_exceeded_seconds_within"] for w in res["windows"]] == [300.0, 300.0]
    # 区段带来源证书
    exc = [s for s in res["segments"] if s["status"] == "exceeded"]
    assert {s["certificate_id"] for s in exc} == {1, 2}


def test_strictly_greater_limit_fails():
    ca, cb = two_certs(
        [reading("a", 0, 20.0), reading("b", 5, 23.0), reading("c", 10, 23.0)],
        [reading("a", 10, 23.0), reading("b", 15, 20.0), reading("c", 20, 20.0)],
    )
    assert build_cross_window_review([ca, cb], 599.999)["verdict"] == "fail"
    assert build_cross_window_review([ca, cb], 600)["verdict"] == "pass"


def test_normal_at_boundary_cuts_run():
    # 窗A 末段超限 300s；窗B 首段正常 -> 最长 300s（正常切断）
    ca, cb = two_certs(
        [reading("a", 0, 20.0), reading("b", 5, 23.0), reading("c", 10, 23.0)],
        [reading("a", 10, 20.0), reading("b", 15, 20.0), reading("c", 20, 20.0)],
    )
    res = build_cross_window_review([ca, cb], 300)
    assert res["verdict"] == "pass"
    assert res["max_continuous_exceeded_seconds"] == 300.0


def test_unknown_at_boundary_cuts_run_and_pending():
    # 窗A 末段 [5,10) 超限 300s；窗B 首读在 12 -> leading 边界未知 120s 切断，
    # 随后 [12,15) 又超限 180s；最长仍为 300s，存在未知 -> pending
    ca, cb = two_certs(
        [reading("a", 0, 20.0), reading("b", 5, 23.0), reading("c", 10, 23.0)],
        [reading("a", 12, 23.0), reading("b", 15, 20.0), reading("c", 20, 20.0)],
    )
    res = build_cross_window_review([ca, cb], 300)
    assert res["verdict"] == "pending"
    assert res["max_continuous_exceeded_seconds"] == 300.0
    assert res["unknown_total_seconds"] == 120.0
    assert res["has_unknown"] is True


def test_fault_masking_near_boundary_is_unknown_and_cuts():
    # 窗A 末段本为超限 300s，被故障 [8,10) 遮蔽 120s：[5,8) 超限 180s 后
    # 被 unknown 切断；窗B 首段超限 300s。最长 = 300s，未知 120s -> pending。
    ca, cb = two_certs(
        [reading("a", 0, 20.0), reading("b", 5, 23.0), reading("c", 10, 23.0)],
        [reading("a", 10, 23.0), reading("b", 15, 20.0), reading("c", 20, 20.0)],
        faults_a=[fault("f1", 8, 10)],
    )
    res = build_cross_window_review([ca, cb], 300)
    assert res["verdict"] == "pending"
    assert res["max_continuous_exceeded_seconds"] == 300.0
    assert res["unknown_total_seconds"] == 120.0
    masked = [s for s in res["segments"]
              if s["evidence"].get("unknown_reason") == UNKNOWN_INSTRUMENT_FAULT]
    assert len(masked) == 1 and masked[0]["duration_seconds"] == 120.0
    assert masked[0]["evidence"]["fault_ids"] == ["f1"]


def test_fault_cannot_bridge_two_exceeded_runs_across_boundary():
    # 窗A 末段超限 300s；窗B 故障遮蔽首段 [10,12)，其后 [12,15) 超限 180s。
    # 故障 unknown 在边界右侧切断：A 的 300s 不与 B 的 180s 相连。
    ca, cb = two_certs(
        [reading("a", 0, 20.0), reading("b", 5, 23.0), reading("c", 10, 23.0)],
        [reading("a", 10, 23.0), reading("b", 12, 23.0),
         reading("c", 15, 20.0), reading("d", 20, 20.0)],
        faults_b=[fault("f1", 10, 12)],
    )
    res = build_cross_window_review([ca, cb], 300)
    assert res["max_continuous_exceeded_seconds"] == 300.0
    assert res["unknown_total_seconds"] == 120.0
    assert res["verdict"] == "pending"


def test_unknown_still_fails_when_limit_exceeded():
    # 既有未知，也有超限超时 -> 仍判不合格
    ca, cb = two_certs(
        [reading("a", 0, 20.0), reading("b", 4, 23.0), reading("c", 10, 23.0)],
        [reading("a", 12, 20.0), reading("b", 20, 20.0)],
    )
    res = build_cross_window_review([ca, cb], 300)
    # A 末段超限 360s > 300；B leading 未知 120s
    assert res["unknown_total_seconds"] == 120.0
    assert res["verdict"] == "fail"


def test_three_window_chain_accumulates_across_two_boundaries():
    # 三窗 [0,10)[10,20)[20,30)，首尾段依次：A 末段超限、B 全段超限、
    # C 首段超限，连续 300+600+300 = 1200s
    def snap(window_id, s, e, readings):
        return make_snapshot(window_id, s, e, readings)

    sa = snap(1, 0, 10, [reading("a", 0, 20.0), reading("b", 5, 23.0),
                         reading("c", 10, 23.0)])
    sb = snap(2, 10, 20, [reading("a", 10, 23.0), reading("b", 20, 23.0)])
    sc = snap(3, 20, 30, [reading("a", 20, 23.0), reading("b", 25, 20.0),
                          reading("c", 30, 20.0)])
    certs = [make_cert(1, 1, sa), make_cert(2, 2, sb), make_cert(3, 3, sc)]
    res = build_cross_window_review(certs, 1200)
    assert res["verdict"] == "pass"
    assert res["max_continuous_exceeded_seconds"] == 1200.0
    assert [w["window_id"] for w in res["windows"]] == [1, 2, 3]
    assert res["start_at"] == iso_at(t(0)) and res["end_at"] == iso_at(t(30))


def test_segments_partition_full_span_with_sources():
    ca, cb = two_certs(
        [reading("a", 0, 20.0), reading("b", 10, 20.0)],
        [reading("a", 10, 20.0), reading("b", 20, 20.0)],
    )
    res = build_cross_window_review([ca, cb], 300)
    assert res["verdict"] == "pass"
    segs = res["segments"]
    # 无缝拼接、覆盖全程
    assert segs[0]["start"] == iso_at(t(0))
    assert segs[-1]["end"] == iso_at(t(20))
    for x, y in zip(segs, segs[1:]):
        assert x["end"] == y["start"]
    assert sum(s["duration_seconds"] for s in segs) == 1200.0
    # 逐窗时长统计合计等于各窗时长
    assert [w["duration_seconds"] for w in res["windows"]] == [600.0, 600.0]


# ------------------------------------------------------------ 拒绝

def test_rejects_fewer_than_two_certificates():
    ca, _ = two_certs(
        [reading("a", 0, 20.0), reading("b", 10, 20.0)],
        [reading("a", 10, 20.0), reading("b", 20, 20.0)],
    )
    with pytest.raises(ExposureReviewError):
        build_cross_window_review([ca], 300)


def test_rejects_non_contiguous_windows():
    sa = make_snapshot(1, 0, 10, [reading("a", 0, 20.0), reading("b", 10, 20.0)])
    sb = make_snapshot(2, 11, 20, [reading("a", 11, 20.0), reading("b", 20, 20.0)])
    with pytest.raises(ExposureReviewError, match="不接续"):
        build_cross_window_review([make_cert(1, 1, sa), make_cert(2, 2, sb)], 300)


def test_rejects_overlapping_windows():
    sa = make_snapshot(1, 0, 10, [reading("a", 0, 20.0), reading("b", 10, 20.0)])
    sb = make_snapshot(2, 9, 20, [reading("a", 9, 20.0), reading("b", 20, 20.0)])
    with pytest.raises(ExposureReviewError):
        build_cross_window_review([make_cert(1, 1, sa), make_cert(2, 2, sb)], 300)


def test_rejects_reverse_order():
    sa = make_snapshot(1, 0, 10, [reading("a", 0, 20.0), reading("b", 10, 20.0)])
    sb = make_snapshot(2, 10, 20, [reading("a", 10, 20.0), reading("b", 20, 20.0)])
    with pytest.raises(ExposureReviewError):
        build_cross_window_review(
            [make_cert(2, 2, sb), make_cert(1, 1, sa)], 300)


def test_rejects_temp_min_mismatch():
    ca, cb = two_certs(
        [reading("a", 0, 20.0), reading("b", 10, 20.0)],
        [reading("a", 10, 20.0), reading("b", 20, 20.0)],
        tmin=18.0,
    )
    # cb 的温度区间为 [18,22]，构造一个下限不同的第二窗
    sb2 = make_snapshot(99, 10, 20,
                        [reading("a", 10, 20.0), reading("b", 20, 20.0)],
                        tmin=19.0)
    with pytest.raises(ExposureReviewError, match="温度范围不一致"):
        build_cross_window_review([ca, make_cert(2, 99, sb2)], 300)


def test_rejects_temp_max_mismatch():
    sa = make_snapshot(1, 0, 10, [reading("a", 0, 20.0), reading("b", 10, 20.0)])
    sb = make_snapshot(2, 10, 20, [reading("a", 10, 20.0), reading("b", 20, 20.0)],
                       tmax=21.5)
    with pytest.raises(ExposureReviewError, match="温度范围不一致"):
        build_cross_window_review([make_cert(1, 1, sa), make_cert(2, 2, sb)], 300)


def test_equal_temp_bounds_allowed_but_other_rules_may_differ():
    # 温度上下限一致即可；各窗采样间隔不同不拒绝（各自按快照重建）。
    # 窗A 采样间隔上限 120s：每分钟一条读数（间隔 60s 合法）；
    # 窗B 采样间隔上限 900s：仅首尾两条读数（间隔 600s 合法）。
    dense = [reading(f"r{m}", m, 20.0) for m in range(0, 11)]
    sa = make_snapshot(1, 0, 10, dense, max_gap=120.0)
    sb = make_snapshot(2, 10, 20, [reading("a", 10, 20.0), reading("b", 20, 20.0)],
                       max_gap=900.0)
    res = build_cross_window_review(
        [make_cert(1, 1, sa), make_cert(2, 2, sb)], 300)
    assert res["verdict"] == "pass"
