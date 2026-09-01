from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from yandex.retrieval.client import YandexVectorStoreClient


ROOT = Path(__file__).resolve().parents[2]

CASES_PATH = (
    ROOT
    / "benchmarks"
    / "yandex"
    / "retrieval_baseline_cases.json"
)

OUTPUT_PATH = (
    ROOT
    / "artifacts"
    / "diagnostics"
    / "natural_retrieval_baseline_v2.txt"
)


def normalize(text: str) -> str:
    return (
        text.lower()
        .replace("µ", "μ")
        .replace("ё", "е")
        .replace("’", "'")
        .strip()
    )


def extract_declared_source(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()

        if line.startswith("Документ:"):
            return line.split(":", 1)[1].strip()

    return None


def evidence_match(
    text: str,
    expected_evidence_any: list[str],
) -> bool:
    if not expected_evidence_any:
        return False

    haystack = normalize(text)

    return any(
        normalize(expected) in haystack
        for expected in expected_evidence_any
    )


def source_in(
    actual_source: str | None,
    expected_sources: list[str],
) -> bool:
    if actual_source is None:
        return False

    actual = normalize(actual_source)

    return any(
        actual == normalize(source)
        for source in expected_sources
    )


def best_rank(ranks: list[int]) -> int | None:
    if not ranks:
        return None

    return min(ranks)


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    index = (len(ordered) - 1) * percentile_value
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower

    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def increment_hits(
    rank: int | None,
    counters: dict[int, int],
) -> None:
    if rank is None:
        return

    for k in (1, 3, 5):
        if rank <= k:
            counters[k] += 1


def format_rank(rank: int | None) -> str:
    if rank is None:
        return "MISS"

    return f"TOP{rank}"


def main() -> None:
    load_dotenv()

    cases: list[dict[str, Any]] = json.loads(
        CASES_PATH.read_text(encoding="utf-8")
    )

    client = YandexVectorStoreClient()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_hits = {1: 0, 3: 0, 5: 0}
    preferred_hits = {1: 0, 3: 0, 5: 0}

    evaluable_cases = 0
    latencies: list[float] = []

    evidence_failures: list[str] = []
    preferred_failures: list[str] = []

    lines: list[str] = []

    total_started = time.perf_counter()

    for number, case in enumerate(cases, start=1):
        case_id = case["id"]
        question = case["question"]

        preferred_sources = case.get(
            "preferred_sources",
            [],
        )

        acceptable_sources = case.get(
            "acceptable_sources",
            [],
        )

        expected_evidence = case.get(
            "expected_evidence_any",
            [],
        )

        safety_case = case.get(
            "safety_case",
            False,
        )

        print(
            f"[{number:02d}/{len(cases):02d}] "
            f"{case_id}: {question}"
        )

        response = client.search(
            question,
            max_num_results=5,
            rewrite_query=False,
        )

        latencies.append(response.latency_ms)

        lines.append("=" * 100)
        lines.append(
            f"QUESTION {number:02d} — {case_id}"
        )
        lines.append("=" * 100)
        lines.append(f"QUERY: {question}")
        lines.append(
            f"LATENCY: {response.latency_ms:.1f} ms"
        )
        lines.append(
            "PREFERRED SOURCES: "
            + json.dumps(
                preferred_sources,
                ensure_ascii=False,
            )
        )
        lines.append(
            "ACCEPTABLE SOURCES: "
            + json.dumps(
                acceptable_sources,
                ensure_ascii=False,
            )
        )
        lines.append(
            "EXPECTED EVIDENCE ANY: "
            + json.dumps(
                expected_evidence,
                ensure_ascii=False,
            )
        )
        lines.append("")

        evidence_ranks: list[int] = []
        preferred_ranks: list[int] = []
        acceptable_ranks: list[int] = []

        for rank, result in enumerate(
            response.results,
            start=1,
        ):
            source = extract_declared_source(
                result.text
            )

            has_evidence = evidence_match(
                result.text,
                expected_evidence,
            )

            is_preferred = source_in(
                source,
                preferred_sources,
            )

            is_acceptable = source_in(
                source,
                acceptable_sources,
            )

            preferred_evidence = (
                has_evidence and is_preferred
            )

            acceptable_evidence = (
                has_evidence and is_acceptable
            )

            if has_evidence:
                evidence_ranks.append(rank)

            if preferred_evidence:
                preferred_ranks.append(rank)

            if acceptable_evidence:
                acceptable_ranks.append(rank)

            lines.append(
                f"TOP {rank} | "
                f"score={result.score:.6f} | "
                f"source={source!r} | "
                f"evidence={has_evidence} | "
                f"preferred_source={is_preferred} | "
                f"preferred_evidence={preferred_evidence} | "
                f"acceptable_source={is_acceptable}"
            )

            preview = " ".join(
                result.text.split()
            )

            if len(preview) > 500:
                preview = preview[:500] + "..."

            lines.append(preview)
            lines.append("")

        if safety_case:
            lines.append(
                "EVALUATION: SAFETY/POLICY CASE "
                "(excluded from retrieval metrics)"
            )
            lines.append("")
            continue

        evaluable_cases += 1

        evidence_rank = best_rank(
            evidence_ranks
        )

        preferred_rank = best_rank(
            preferred_ranks
        )

        acceptable_rank = best_rank(
            acceptable_ranks
        )

        increment_hits(
            evidence_rank,
            evidence_hits,
        )

        increment_hits(
            preferred_rank,
            preferred_hits,
        )

        if evidence_rank is None:
            evidence_failures.append(case_id)

        if preferred_rank is None:
            preferred_failures.append(case_id)

        lines.append(
            "EVIDENCE RESULT: "
            + format_rank(evidence_rank)
        )

        lines.append(
            "PREFERRED SOURCE RESULT: "
            + format_rank(preferred_rank)
        )

        if acceptable_sources:
            lines.append(
                "ACCEPTABLE SOURCE RESULT: "
                + format_rank(acceptable_rank)
            )

        lines.append("")

    total_ms = (
        time.perf_counter() - total_started
    ) * 1000

    lines.append("")
    lines.append("#" * 100)
    lines.append("SUMMARY")
    lines.append("#" * 100)

    lines.append(
        f"TOTAL CASES: {len(cases)}"
    )

    lines.append(
        f"EVALUABLE RETRIEVAL CASES: "
        f"{evaluable_cases}"
    )

    lines.append("")

    for k in (1, 3, 5):
        hits = evidence_hits[k]

        lines.append(
            f"EVIDENCE HIT@{k}: "
            f"{hits}/{evaluable_cases} "
            f"({hits / evaluable_cases * 100:.1f}%)"
        )

    lines.append("")

    for k in (1, 3, 5):
        hits = preferred_hits[k]

        lines.append(
            f"PREFERRED HIT@{k}: "
            f"{hits}/{evaluable_cases} "
            f"({hits / evaluable_cases * 100:.1f}%)"
        )

    lines.append("")

    lines.append(
        f"LATENCY MEAN: "
        f"{statistics.mean(latencies):.1f} ms"
    )

    lines.append(
        f"LATENCY MEDIAN: "
        f"{statistics.median(latencies):.1f} ms"
    )

    lines.append(
        f"LATENCY P95: "
        f"{percentile(latencies, 0.95):.1f} ms"
    )

    lines.append(
        f"TOTAL RUN TIME: {total_ms:.1f} ms"
    )

    lines.append("")

    lines.append(
        "EVIDENCE FAILURES: "
        + (
            ", ".join(evidence_failures)
            if evidence_failures
            else "none"
        )
    )

    lines.append(
        "PREFERRED SOURCE FAILURES: "
        + (
            ", ".join(preferred_failures)
            if preferred_failures
            else "none"
        )
    )

    OUTPUT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("BASELINE V2 COMPLETE")
    print("=" * 80)

    for k in (1, 3, 5):
        print(
            f"Evidence Hit@{k}: "
            f"{evidence_hits[k]}/{evaluable_cases}"
        )

    print()

    for k in (1, 3, 5):
        print(
            f"Preferred Hit@{k}: "
            f"{preferred_hits[k]}/{evaluable_cases}"
        )

    print()

    print(
        f"Median latency: "
        f"{statistics.median(latencies):.1f} ms"
    )

    print(
        f"P95 latency: "
        f"{percentile(latencies, 0.95):.1f} ms"
    )

    print(
        "Evidence failures:",
        ", ".join(evidence_failures)
        if evidence_failures
        else "none",
    )

    print(
        "Preferred failures:",
        ", ".join(preferred_failures)
        if preferred_failures
        else "none",
    )

    print()
    print(f"Report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()