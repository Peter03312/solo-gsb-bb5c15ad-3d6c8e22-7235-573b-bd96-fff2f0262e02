"""跨审定窗口连续温度暴露复核 HTTP API 集成测试。

运行：PYTHONPATH=. python -m pytest tests/test_exposure_review_api.py -q
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.config import settings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(settings, "db_path", str(db_file))
    monkeypatch.setattr(settings, "db_timeout", 5)
    from app.main import app

    with TestClient(app) as c:
        yield c


def iso(minute: float, base_day: int = 1) -> str:
    t = datetime(2026, 10, base_day, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return t.isoformat().replace("+00:00", "Z")


def make_window(client, start, end, *, day=1, tmin=18.0, tmax=22.0,
                max_exceed=300.0, max_gap=600.0, name="case"):
    r = client.post("/api/v1/windows", json={
        "name": name, "start_at": iso(start, day), "end_at": iso(end, day),
        "allowed_temp_min": tmin, "allowed_temp_max": tmax,
        "max_continuous_exceed_seconds": max_exceed,
        "max_sample_interval_seconds": max_gap})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def submit(client, wid, rid, minute, temp, rev=1, *, day=1):
    r = client.post(f"/api/v1/windows/{wid}/readings", json={
        "reading_id": rid, "revision": rev,
        "measured_at": iso(minute, day), "temperature": temp})
    assert r.status_code == 201, r.text
    return r.json()


def issue(client, wid):
    r = client.post(f"/api/v1/windows/{wid}/certificates/issue", json={})
    assert r.status_code == 201, r.text
    return r.json()


def review(client, ids, limit):
    return client.get(
        "/api/v1/certificates/exposure-review",
        params=[*[("certificate_id", i) for i in ids],
                ("max_continuous_exceed_seconds", limit)])


def two_contiguous_windows(client):
    """窗A [0,10) 末段超限 300s；窗B [10,20) 首段超限 300s（跨边界=600s）。"""
    wa = make_window(client, 0, 10, name="A")
    for rid, m, t in [("a", 0, 20.0), ("b", 5, 23.0), ("c", 10, 23.0)]:
        submit(client, wa, rid, m, t)
    ca = issue(client, wa)["id"]
    wb = make_window(client, 10, 20, name="B")
    for rid, m, t in [("a", 10, 23.0), ("b", 15, 20.0), ("c", 20, 20.0)]:
        submit(client, wb, rid, m, t)
    cb = issue(client, wb)["id"]
    return wa, wb, ca, cb


# ------------------------------------------------------------ 主流程

def test_review_pass_at_exact_limit_and_sources(client):
    wa, wb, ca, cb = two_contiguous_windows(client)
    r = review(client, [ca, cb], 600)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["verdict"] == "pass"
    assert d["verdict_label_zh"] == "合格"
    assert d["max_continuous_exceeded_seconds"] == 600.0
    assert d["max_continuous_exceed_limit_seconds"] == 600.0
    assert d["unknown_total_seconds"] == 0.0
    assert d["total_duration_seconds"] == 1200.0
    assert d["start_at"] == iso(0) and d["end_at"] == iso(20)
    # 逐窗与来源证书
    assert [w["window_id"] for w in d["windows"]] == [wa, wb]
    assert [w["certificate_id"] for w in d["windows"]] == [ca, cb]
    for seg in d["segments"]:
        assert seg["status"] in ("normal", "exceeded", "unknown")
        assert seg["certificate_id"] in (ca, cb)
        assert "evidence" in seg
    # 全程区段无缝拼接
    for x, y in zip(d["segments"], d["segments"][1:]):
        assert x["end"] == y["start"]


def test_review_fail_when_strictly_over_limit(client):
    _, _, ca, cb = two_contiguous_windows(client)
    r = review(client, [ca, cb], 599.999)
    assert r.status_code == 200
    d = r.json()
    assert d["verdict"] == "fail" and d["verdict_label_zh"] == "不合格"
    assert "不合格" in d["conclusion_zh"]


def test_review_pending_with_unknown(client):
    # 窗B 首读在 12：leading 边界未知 120s 切断 A 末段的 300s 超限
    wa = make_window(client, 0, 10, name="A")
    for rid, m, t in [("a", 0, 20.0), ("b", 5, 23.0), ("c", 10, 23.0)]:
        submit(client, wa, rid, m, t)
    ca = issue(client, wa)["id"]
    wb = make_window(client, 10, 20, name="B")
    for rid, m, t in [("a", 12, 23.0), ("b", 15, 20.0), ("c", 20, 20.0)]:
        submit(client, wb, rid, m, t)
    cb = issue(client, wb)["id"]

    d = review(client, [ca, cb], 300).json()
    assert d["verdict"] == "pending" and d["verdict_label_zh"] == "待定"
    assert d["max_continuous_exceeded_seconds"] == 300.0
    assert d["unknown_total_seconds"] == 120.0
    leading = [s for s in d["segments"]
               if s["evidence"].get("unknown_reason") == "boundary_gap"]
    assert len(leading) == 1 and leading[0]["duration_seconds"] == 120.0


def test_review_uses_immutable_snapshot_not_current_state(client):
    """复核只依证书快照：签发后改读数并登记故障，复核结果不变（不读当前状态）。"""
    wa, wb, ca, cb = two_contiguous_windows(client)
    before = review(client, [ca, cb], 600).json()

    # 签发之后：修订当前读数（全部正常）、登记故障遮蔽——不重新签发
    submit(client, wa, "b", 6, 20.0, rev=2)
    r = client.post(f"/api/v1/windows/{wa}/fault-periods", json={
        "fault_id": "f1", "start_at": iso(0), "end_at": iso(10),
        "reason": "当前状态：仪器漂移"})
    assert r.status_code == 201
    # 窗口当前审定已变为 pending/正常，但旧证复核结果应原样
    after = review(client, [ca, cb], 600).json()
    assert after["segments"] == before["segments"]
    assert after["max_continuous_exceeded_seconds"] == \
        before["max_continuous_exceeded_seconds"] == 600.0
    assert after["verdict"] == "pass"


def test_review_is_read_only(client):
    wa, wb, ca, cb = two_contiguous_windows(client)
    review(client, [ca, cb], 600)
    # 证书未被撤销、窗口读数条数不变、移交单等数据不受影响
    cert = client.get(f"/api/v1/certificates/{ca}").json()
    assert cert["revoked"] is False
    readings = client.get(f"/api/v1/windows/{wa}/readings").json()
    assert len(readings) == 3


# ------------------------------------------------------------ 拒绝路径

def test_reject_duplicate_certificate_ids(client):
    _, _, ca, _ = two_contiguous_windows(client)
    r = review(client, [ca, ca], 300)
    assert r.status_code == 422
    assert "重复" in r.json()["detail"]


def test_reject_single_certificate(client):
    _, _, ca, _ = two_contiguous_windows(client)
    r = review(client, [ca], 300)
    assert r.status_code == 422


def test_reject_non_positive_limit(client):
    _, _, ca, cb = two_contiguous_windows(client)
    r = review(client, [ca, cb], 0)
    assert r.status_code == 422
    r = review(client, [ca, cb], -1)
    assert r.status_code == 422


def test_reject_missing_certificate_404(client):
    _, _, ca, _ = two_contiguous_windows(client)
    r = review(client, [ca, 99999], 300)
    assert r.status_code == 404


def test_reject_revoked_certificate_409(client):
    wa, wb, ca, cb = two_contiguous_windows(client)
    r = client.post(f"/api/v1/certificates/{cb}/revocation",
                    json={"reason": "签发依据有误"})
    assert r.status_code == 201
    r = review(client, [ca, cb], 300)
    assert r.status_code == 409
    assert "撤销" in r.json()["detail"]
    # 撤销后重新签发窗B，新版本成为当前可用证书 -> 通过
    cb2 = issue(client, wb)["id"]
    assert review(client, [ca, cb2], 600).status_code == 200


def test_reject_non_latest_version_409(client):
    wa, wb, ca, cb = two_contiguous_windows(client)
    # 窗A 再签一版（ca 未撤销但已非最新）
    ca2 = issue(client, wa)["id"]
    r = review(client, [ca, cb], 300)
    assert r.status_code == 409
    assert "最新签发版本" in r.json()["detail"]
    # 换成最新版即通过
    assert review(client, [ca2, cb], 600).status_code == 200


def test_reject_revoked_latest_then_reissue(client):
    """撤销最新版后不退回旧版：旧版复核仍 409；重签新版后恢复。"""
    wa, wb, ca, cb = two_contiguous_windows(client)
    ca2 = issue(client, wa)["id"]  # ca2 最新，ca 为旧版
    # 撤销最新版 ca2：窗口无当前可用证书
    r = client.post(f"/api/v1/certificates/{ca2}/revocation",
                    json={"reason": "依据有误"})
    assert r.status_code == 201
    assert review(client, [ca2, cb], 300).status_code == 409
    # ca 虽是未撤销旧版，但非最新 -> 同样 409
    assert review(client, [ca, cb], 300).status_code == 409
    # 再签发新版 -> 成为当前可用
    ca3 = issue(client, wa)["id"]
    assert review(client, [ca3, cb], 600).status_code == 200


def test_reject_non_contiguous_windows_422(client):
    _, _, ca, _ = two_contiguous_windows(client)
    wc = make_window(client, 60, 70, name="C")
    submit(client, wc, "a", 60, 20.0)
    submit(client, wc, "b", 70, 20.0)
    cc = issue(client, wc)["id"]
    r = review(client, [ca, cc], 300)
    assert r.status_code == 422
    assert "不接续" in r.json()["detail"]


def test_reject_reverse_order_422(client):
    _, _, ca, cb = two_contiguous_windows(client)
    r = review(client, [cb, ca], 300)
    assert r.status_code == 422


def test_reject_temp_range_mismatch_422(client):
    _, _, ca, _ = two_contiguous_windows(client)
    # 窗口与 A 不接续（起点 20），但温度范围也与 A 不一致（下限 19）：
    # 接续校验先于温度范围校验 -> 422 不接续
    wc = make_window(client, 20, 30, tmin=19.0, name="C")
    submit(client, wc, "a", 20, 20.0)
    submit(client, wc, "b", 30, 20.0)
    cc = issue(client, wc)["id"]
    r = review(client, [ca, cc], 300)
    assert r.status_code == 422
    assert "不接续" in r.json()["detail"]

    # 与 A、B 接续（窗D [10,20)）但温度下限不同 -> 422 温度范围不一致
    wd = make_window(client, 10, 20, tmin=19.0, name="D")
    submit(client, wd, "a", 10, 20.0)
    submit(client, wd, "b", 20, 20.0)
    cd = issue(client, wd)["id"]
    r = review(client, [ca, cd], 300)
    assert r.status_code == 422
    assert "温度范围不一致" in r.json()["detail"]


def test_reject_no_partial_results(client):
    """任一输入不合法即整体拒绝：响应不含任何分段结果。"""
    _, _, ca, cb = two_contiguous_windows(client)
    r = review(client, [ca, cb], 0)  # 非正上限
    assert r.status_code == 422
    body = r.json()
    assert "segments" not in body and "verdict" not in body


def test_three_window_chain(client):
    wa, wb, ca, cb = two_contiguous_windows(client)
    wc = make_window(client, 20, 30, name="C")
    for rid, m, t in [("a", 20, 23.0), ("b", 25, 20.0), ("c", 30, 20.0)]:
        submit(client, wc, rid, m, t)
    cc = issue(client, wc)["id"]
    # A[5,10) 超 + B[10,15) 超 + C[20,25) 超：B 末段 [15,20) 正常会切断，
    # 故最长为 600s（跨 A/B 边界），不是 900s。
    d = review(client, [ca, cb, cc], 600).json()
    assert len(d["windows"]) == 3
    assert d["max_continuous_exceeded_seconds"] == 600.0
    assert d["verdict"] == "pass"
    assert d["total_duration_seconds"] == 1800.0


def test_same_window_two_versions_rejected_as_duplicate(client):
    """同一件展品在同一审定窗口的两张版本证书：标识不同但窗口相同，
    窗口不可能首尾相接（区间相同），应拒绝（不接续）。"""
    wa, wb, ca, cb = two_contiguous_windows(client)
    ca2 = issue(client, wa)["id"]
    r = review(client, [ca, ca2], 300)
    # ca 已非最新 -> 先命中 409
    assert r.status_code == 409
