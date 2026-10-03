"""跨审定窗口连续温度暴露复核（纯函数，无 IO，可独立测试）。

馆员复核**一件展品跨多个审定窗口**的连续温度暴露时，输入按时间排列的至少
两张证书（必须分别是所属窗口的当前可用版本）与正数“全程连续超限上限（秒）”。
本模块从各证书的**不可变快照**（window / readings / fault_periods）还原半开
审定区段——复用 ``app.report.build_evidence_segments`` 的采样保持、采样间隔
超限未知、边界空白未知与仪器故障遮蔽语义——**不采用当前读数或当前规则**。

拒绝条件（任一不满足即明确拒绝，不出具部分结果）：

- 证书少于两张、证书标识重复（由调用方先行校验）；
- 快照窗口必须**首尾相接**：前一张窗口的终点（半开，不含）恰为后一张窗口的
  起点（含），且整体按时间升序排列；
- 各快照的允许温度**上下限必须分别一致**（其余规则如各窗采样间隔可不同，
  重建时各窗按各自快照规则）。

连续累计口径：把各窗重建区段按时间首尾相接后逐段扫描，**跨窗边界两侧均为
超限时连续累计**；正常或未知（含仪器故障遮蔽）一律切断累计。全程最长连续
超限时长**严格大于**全程上限 -> 不合格（fail）；否则只要存在未知区段 ->
待定（pending）；其余 -> 合格（pass）。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from .models import VERDICT_LABELS_ZH, Verdict
from .report import EvidenceSeg, build_evidence_segments
from .timeutils import parse_stored, to_iso


class ExposureReviewError(Exception):
    """跨窗复核输入不满足接续/温度范围等一致性约束（调用方映射为 422）。"""


def _seg_duration(seg: EvidenceSeg) -> float:
    return (seg.end - seg.start).total_seconds()


def _merge_same_evidence(segs: Sequence[EvidenceSeg]) -> list[EvidenceSeg]:
    """合并相邻且状态与证据完全相同的区段（零时长段丢弃）。

    与修订影响报告的区间合并同口径：仅首尾相接且 evidence_key 一致才合并。
    状态相同但证据不同（如边界空白未知与采样间隔超限未知相邻）仍分段输出，
    使每段状态都可追溯到快照证据；是否合并不影响全程连续累计与总时长。
    """
    merged: list[EvidenceSeg] = []
    for seg in segs:
        if seg.end <= seg.start:
            continue
        if (merged and merged[-1].end == seg.start
                and merged[-1].evidence_key() == seg.evidence_key()):
            merged[-1] = replace(merged[-1], end=seg.end)
        else:
            merged.append(seg)
    return merged


def _status_totals(segs: Sequence[EvidenceSeg]) -> tuple[float, float, float]:
    normal = exceeded = unknown = 0.0
    for seg in segs:
        dur = _seg_duration(seg)
        if seg.status == "normal":
            normal += dur
        elif seg.status == "exceeded":
            exceeded += dur
        else:
            unknown += dur
    return normal, exceeded, unknown


def _max_continuous_within(segs: Sequence[EvidenceSeg]) -> float:
    longest = current = 0.0
    for seg in segs:
        if seg.status == "exceeded":
            current += _seg_duration(seg)
            longest = max(longest, current)
        else:
            current = 0.0
    return longest


def _segment_out(seq: int, seg: EvidenceSeg, cert: dict[str, Any]) -> dict[str, Any]:
    return {
        "seq": seq,
        "start": to_iso(seg.start),
        "end": to_iso(seg.end),
        "duration_seconds": round(_seg_duration(seg), 3),
        "status": seg.status,
        "evidence": seg.evidence(),
        "window_id": cert["window_id"],
        "certificate_id": cert["id"],
        "certificate_version": cert["version"],
    }


def _cert_ref(cert: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": cert["id"],
        "window_id": cert["window_id"],
        "version": cert["version"],
        "verdict": cert["verdict"],
        "verdict_label_zh": VERDICT_LABELS_ZH.get(cert["verdict"], cert["verdict"]),
        "issued_at": cert["issued_at"],
        "note": cert.get("note"),
        "revoked": bool(cert.get("revoked")),
        "revocation": cert.get("revocation"),
    }


def build_cross_window_review(
    certificates: Sequence[dict[str, Any]],
    max_continuous_exceed_seconds: float,
) -> dict[str, Any]:
    """组装跨审定窗口连续温度暴露复核结果。

    certificates: 按输入顺序给出的证书 dict（须含 snapshot；调用方负责核验
                  各证存在、未撤销且为所属窗口最新签发版本，以及标识不重复）。
    max_continuous_exceed_seconds: 全程连续超限上限（秒，正数；严格大于才判
                  不合格，等于不算超过）。
    """
    if len(certificates) < 2:
        raise ExposureReviewError(
            f"跨窗复核至少需要 2 张证书，实际收到 {len(certificates)} 张。"
        )

    # ---- 1) 从各证书不可变快照重建半开区段（不读当前读数/规则） ----
    meta: list[dict[str, Any]] = []
    per_segments: list[list[EvidenceSeg]] = []
    for cert in certificates:
        snap = cert["snapshot"]
        win = snap["window"]
        win_start = parse_stored(win["start_at"])
        win_end = parse_stored(win["end_at"])
        if win_end <= win_start:
            raise ExposureReviewError(
                f"证书 {cert['id']} 快照窗口非正时长区间"
                f"[{to_iso(win_start)}, {to_iso(win_end)})，拒绝复核。"
            )
        meta.append({"cert": cert, "win": win, "start": win_start, "end": win_end})
        per_segments.append(_merge_same_evidence(build_evidence_segments(snap)))

    # ---- 2) 窗口首尾相接（半开：前窗终点 == 后窗起点），整体按时间升序 ----
    for i in range(1, len(meta)):
        prev, cur = meta[i - 1], meta[i]
        if prev["end"] != cur["start"]:
            raise ExposureReviewError(
                "审定窗口不接续，拒绝复核："
                f"证书 {prev['cert']['id']}（窗口 {prev['cert']['window_id']}）的窗口终点 "
                f"{to_iso(prev['end'])} 与证书 {cur['cert']['id']}"
                f"（窗口 {cur['cert']['window_id']}）的窗口起点 {to_iso(cur['start'])} "
                "不一致；证书必须按时间排列且相邻窗口首尾相接"
                "（前一窗口终点须恰为后一窗口起点）。"
            )

    # ---- 3) 允许温度上下限各窗一致 ----
    base_min = float(meta[0]["win"]["allowed_temp_min"])
    base_max = float(meta[0]["win"]["allowed_temp_max"])
    for item in meta[1:]:
        tmin = float(item["win"]["allowed_temp_min"])
        tmax = float(item["win"]["allowed_temp_max"])
        if tmin != base_min or tmax != base_max:
            raise ExposureReviewError(
                "各审定窗口允许温度范围不一致，拒绝复核："
                f"证书 {meta[0]['cert']['id']} 的允许温度区间为 "
                f"[{base_min:g}, {base_max:g}]°C，证书 {item['cert']['id']} 为 "
                f"[{tmin:g}, {tmax:g}]°C；跨窗连续复核要求温度上下限一致。"
            )

    limit = float(max_continuous_exceed_seconds)

    # ---- 4) 逐窗统计 + 跨窗连续累计 ----
    windows_out: list[dict[str, Any]] = []
    segments_out: list[dict[str, Any]] = []
    normal_total = exceeded_total = unknown_total = 0.0
    longest = 0.0          # 全程最长连续超限（跨窗边界两侧超限连续累计）
    current_run = 0.0      # 当前连续超限累计（正常/未知切断，含跨窗边界处）
    seq = 0
    for idx, (item, segs) in enumerate(zip(meta, per_segments)):
        cert = item["cert"]
        w_normal, w_exceeded, w_unknown = _status_totals(segs)
        normal_total += w_normal
        exceeded_total += w_exceeded
        unknown_total += w_unknown
        for seg in segs:
            segments_out.append(_segment_out(seq, seg, cert))
            seq += 1
            if seg.status == "exceeded":
                # 前一窗口末段与本窗首段在边界两侧均超限时 current_run 不归零，
                # 自然跨窗连续累计；正常/未知段（含故障遮蔽）在此切断。
                current_run += _seg_duration(seg)
                longest = max(longest, current_run)
            else:
                current_run = 0.0

        windows_out.append({
            "seq": idx,
            "window_id": cert["window_id"],
            "window_start_at": to_iso(item["start"]),
            "window_end_at": to_iso(item["end"]),
            "duration_seconds": round((item["end"] - item["start"]).total_seconds(), 3),
            "allowed_temp_min": base_min,
            "allowed_temp_max": base_max,
            "certificate_id": cert["id"],
            "certificate_version": cert["version"],
            "segment_count": len(segs),
            "normal_seconds": round(w_normal, 3),
            "exceeded_seconds": round(w_exceeded, 3),
            "unknown_seconds": round(w_unknown, 3),
            "max_continuous_exceeded_seconds_within":
                round(_max_continuous_within(segs), 3),
        })

    longest = round(longest, 3)
    normal_total = round(normal_total, 3)
    exceeded_total = round(exceeded_total, 3)
    unknown_total = round(unknown_total, 3)
    has_unknown = unknown_total > 0
    total_seconds = round(
        (meta[-1]["end"] - meta[0]["start"]).total_seconds(), 3)

    # 严格大于才不合格；等于限值不算超过。
    if longest > limit:
        verdict = Verdict.FAIL.value
    elif has_unknown:
        verdict = Verdict.PENDING.value
    else:
        verdict = Verdict.PASS.value

    label = VERDICT_LABELS_ZH[verdict]
    n = len(meta)
    if verdict == Verdict.FAIL.value:
        head = (
            f"跨窗复核结论：不合格（fail）：全程最长连续超限 {longest:g}s "
            f"严格超过全程上限 {limit:g}s（跨窗边界两侧均超限时已连续累计）。"
        )
    elif verdict == Verdict.PENDING.value:
        head = (
            f"跨窗复核结论：待定（pending）：全程最长连续超限 {longest:g}s "
            f"未超过全程上限 {limit:g}s（等于不算超过），但全程存在未知区段，"
            f"未知总时长 {unknown_total:g}s（含采样间隔超限、边界空白与仪器故障遮蔽）。"
        )
    else:
        head = (
            f"跨窗复核结论：合格（pass）：全程最长连续超限 {longest:g}s "
            f"未超过全程上限 {limit:g}s（等于不算超过），且全程无未知区段。"
        )
    tail = (
        f"共复核 {n} 张当前可用证书、{n} 个首尾相接审定窗口"
        f"（{to_iso(meta[0]['start'])} ~ {to_iso(meta[-1]['end'])}，"
        f"半开），全程 {total_seconds:g}s，允许温度区间 "
        f"[{base_min:g}, {base_max:g}]°C；结论仅依据各证书不可变快照，"
        "不采用当前读数或规则。"
    )

    return {
        "certificates": [_cert_ref(cert) for cert in certificates],
        "windows": windows_out,
        "segments": segments_out,
        "segment_count": len(segments_out),
        "start_at": to_iso(meta[0]["start"]),
        "end_at": to_iso(meta[-1]["end"]),
        "total_duration_seconds": total_seconds,
        "normal_total_seconds": normal_total,
        "exceeded_total_seconds": exceeded_total,
        "unknown_total_seconds": unknown_total,
        "has_unknown": has_unknown,
        "max_continuous_exceeded_seconds": longest,
        "max_continuous_exceed_limit_seconds": limit,
        "verdict": verdict,
        "verdict_label_zh": label,
        "conclusion_zh": head + tail,
    }
