from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from yandex.retrieval.controller import RetrievalController


ROOT = Path(__file__).resolve().parents[2]

CASES_PATH = ROOT / "benchmarks" / "yandex" / "retrieval_baseline_cases.json"
REPORT_PATH = (
    ROOT
    / "artifacts"
    / "diagnostics"
    / "controller_retrieval_baseline_v1.txt"
)


def normalize_text(value: str) -> str:
    return (
        value.lower()
        .replace("ё", "е")
        .replace("µ", "μ")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )


def matches_any_evidence(
    text: str,
    expected_evidence: list[str],
) -> bool:
    normalized_text = normalize_text(text)

    return any(
        normalize_text(expected) in normalized_text
        for expected in expected_evidence
    )


def source_matches(
    source: str | None,
    expected_sources: list[str],
) -> bool:
    if source is None:
        return False

    normalized_source = normalize_text(source)

    return any(
        normalize_text(expected_source) == normalized_source
        for expected_source in expected_sources
    )


def find_first_evidence_rank(
    results: list[Any],
    expected_evidence: list[str],
) -> int | None:
    for result in results:
        if matches_any_evidence(
            result.text,
            expected_evidence,
        ):
            return result.rank

    return None


def find_first_source_rank(
    results: list[Any],
    expected_evidence: list[str],
    expected_sources: list[str],
) -> int | None:
    for result in results:
        if not matches_any_evidence(
            result.text,
            expected_evidence,
        ):
            continue

        if source_matches(
            result.source,
            expected_sources,
        ):
            return result.rank

    return None


def hit_at_k(
    rank: int | None,
    k: int,
) -> bool:
    return rank is not None and rank <= k


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_value
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)

    fraction = position - lower_index

    return (
        ordered[lower_index] * (1 - fraction)
        + ordered[upper_index] * fraction
    )


def load_cases() -> list[dict[str, Any]]:
    with CASES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "retrieval_baseline_cases.json must contain a JSON list"
        )

    return data


def main() -> None:
    load_dotenv()

    cases = load_cases()

    controller = RetrievalController(
        max_num_results=5,
    )

    report_lines: list[str] = []

    evidence_ranks: list[int | None] = []
    preferred_ranks: list[int | None] = []

    latencies: list[float] = []

    evaluable_cases = 0
    safety_cases = 0

    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        question = str(case["question"])

        expected_evidence = list(
            case.get("expected_evidence_any", [])
        )

        preferred_sources = list(
            case.get("preferred_sources", [])
        )

        acceptable_sources = list(
            case.get("acceptable_sources", [])
        )

        is_safety_case = bool(
            case.get("safety_case", False)
        )

        print()
        print("=" * 100)
        print(f"[{index}/{len(cases)}] {case_id}")
        print(question)

        started = time.perf_counter()

        response = controller.search(question)

        total_elapsed_ms = (
            time.perf_counter() - started
        ) * 1000

        latencies.append(
            response.latency_ms
        )

        report_lines.append(
            "=" * 100
        )
        report_lines.append(
            f"CASE: {case_id}"
        )
        report_lines.append(
            f"QUESTION: {question}"
        )
        report_lines.append(
            f"VECTOR LATENCY: {response.latency_ms:.1f} ms"
        )
        report_lines.append(
            f"TOTAL CONTROLLER LATENCY: {total_elapsed_ms:.1f} ms"
        )

        for result in response.evidence:
            evidence_match = matches_any_evidence(
                result.text,
                expected_evidence,
            )

            preferred_match = (
                evidence_match
                and source_matches(
                    result.source,
                    preferred_sources,
                )
            )

            acceptable_match = (
                evidence_match
                and source_matches(
                    result.source,
                    acceptable_sources,
                )
            )

            line = (
                f"#{result.rank} "
                f"score={result.score:.6f} "
                f"type={result.source_type} "
                f"authority={result.authority} "
                f"source={result.source!r} "
                f"evidence={evidence_match} "
                f"preferred={preferred_match} "
                f"acceptable={acceptable_match}"
            )

            report_lines.append(line)

        if is_safety_case:
            safety_cases += 1

            report_lines.append(
                "RESULT: SAFETY CASE — excluded from retrieval metrics"
            )

            print(
                "Safety case — excluded from retrieval metrics"
            )

            continue

        evaluable_cases += 1

        evidence_rank = find_first_evidence_rank(
            response.evidence,
            expected_evidence,
        )

        preferred_rank = find_first_source_rank(
            response.evidence,
            expected_evidence,
            preferred_sources,
        )

        evidence_ranks.append(
            evidence_rank
        )

        preferred_ranks.append(
            preferred_rank
        )

        report_lines.append(
            f"EVIDENCE RANK: {evidence_rank}"
        )
        report_lines.append(
            f"PREFERRED RANK: {preferred_rank}"
        )

        print(
            f"Evidence rank: {evidence_rank}"
        )
        print(
            f"Preferred rank: {preferred_rank}"
        )

    evidence_hit_1 = sum(
        hit_at_k(rank, 1)
        for rank in evidence_ranks
    )

    evidence_hit_3 = sum(
        hit_at_k(rank, 3)
        for rank in evidence_ranks
    )

    evidence_hit_5 = sum(
        hit_at_k(rank, 5)
        for rank in evidence_ranks
    )

    preferred_hit_1 = sum(
        hit_at_k(rank, 1)
        for rank in preferred_ranks
    )

    preferred_hit_3 = sum(
        hit_at_k(rank, 3)
        for rank in preferred_ranks
    )

    preferred_hit_5 = sum(
        hit_at_k(rank, 5)
        for rank in preferred_ranks
    )

    median_latency = (
        statistics.median(latencies)
        if latencies
        else 0.0
    )

    p95_latency = percentile(
        latencies,
        0.95,
    )

    summary = [
        "",
        "=" * 100,
        "CONTROLLER RETRIEVAL BASELINE V1",
        "=" * 100,
        "",
        f"Evaluable cases: {evaluable_cases}",
        f"Safety cases excluded: {safety_cases}",
        "",
        (
            f"Evidence Hit@1: "
            f"{evidence_hit_1}/{evaluable_cases}"
        ),
        (
            f"Evidence Hit@3: "
            f"{evidence_hit_3}/{evaluable_cases}"
        ),
        (
            f"Evidence Hit@5: "
            f"{evidence_hit_5}/{evaluable_cases}"
        ),
        "",
        (
            f"Preferred Hit@1: "
            f"{preferred_hit_1}/{evaluable_cases}"
        ),
        (
            f"Preferred Hit@3: "
            f"{preferred_hit_3}/{evaluable_cases}"
        ),
        (
            f"Preferred Hit@5: "
            f"{preferred_hit_5}/{evaluable_cases}"
        ),
        "",
        (
            f"Median latency: "
            f"{median_latency:.1f} ms"
        ),
        (
            f"P95 latency: "
            f"{p95_latency:.1f} ms"
        ),
        "",
    ]

    report_lines.extend(
        summary
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("CONTROLLER BASELINE COMPLETE")
    print("=" * 100)

    print(
        f"Evidence Hit@1: "
        f"{evidence_hit_1}/{evaluable_cases}"
    )
    print(
        f"Evidence Hit@3: "
        f"{evidence_hit_3}/{evaluable_cases}"
    )
    print(
        f"Evidence Hit@5: "
        f"{evidence_hit_5}/{evaluable_cases}"
    )

    print()

    print(
        f"Preferred Hit@1: "
        f"{preferred_hit_1}/{evaluable_cases}"
    )
    print(
        f"Preferred Hit@3: "
        f"{preferred_hit_3}/{evaluable_cases}"
    )
    print(
        f"Preferred Hit@5: "
        f"{preferred_hit_5}/{evaluable_cases}"
    )

    print()

    print(
        f"Median latency: "
        f"{median_latency:.1f} ms"
    )
    print(
        f"P95 latency: "
        f"{p95_latency:.1f} ms"
    )

    print()
    print(
        f"Report: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()