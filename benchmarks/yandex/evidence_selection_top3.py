from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from yandex.retrieval.controller import RetrievalController
from yandex.retrieval.models import EvidenceChunk


ROOT = Path(__file__).resolve().parents[2]

CASES_PATH = (
    ROOT
    / "benchmarks"
    / "yandex"
    / "retrieval_baseline_cases.json"
)

REPORT_PATH = (
    ROOT
    / "artifacts"
    / "diagnostics"
    / "evidence_selection_top3.txt"
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
        normalize_text(expected) == normalized_source
        for expected in expected_sources
    )


def find_evidence_rank(
    evidence: list[EvidenceChunk],
    expected_evidence: list[str],
) -> int | None:
    for item in evidence:
        if matches_any_evidence(
            item.text,
            expected_evidence,
        ):
            return item.rank

    return None


def find_preferred_rank(
    evidence: list[EvidenceChunk],
    expected_evidence: list[str],
    preferred_sources: list[str],
) -> int | None:
    for item in evidence:
        if not matches_any_evidence(
            item.text,
            expected_evidence,
        ):
            continue

        if source_matches(
            item.source,
            preferred_sources,
        ):
            return item.rank

    return None


def load_cases() -> list[dict[str, Any]]:
    with CASES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "retrieval_baseline_cases.json must contain a list"
        )

    return data


def main() -> None:
    load_dotenv()

    controller = RetrievalController(
        max_num_results=5,
    )

    cases = load_cases()

    evaluable = 0

    full_top5_hits = 0
    selected_top3_hits = 0

    preferred_top5_hits = 0
    preferred_top3_hits = 0

    lost_cases: list[str] = []
    preferred_lost_cases: list[str] = []

    report: list[str] = []

    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        question = str(case["question"])

        expected_evidence = list(
            case.get("expected_evidence_any", [])
        )

        preferred_sources = list(
            case.get("preferred_sources", [])
        )

        is_safety_case = bool(
            case.get("safety_case", False)
        )

        print()
        print("=" * 100)
        print(f"[{index}/{len(cases)}] {case_id}")
        print(question)

        response = controller.search(question)

        full_evidence = response.evidence

        # Experimental selection:
        # keep only semantic TOP3.
        selected_evidence = full_evidence[:3]

        report.append("=" * 100)
        report.append(f"CASE: {case_id}")
        report.append(f"QUESTION: {question}")

        for item in full_evidence:
            selected = item in selected_evidence

            evidence_match = matches_any_evidence(
                item.text,
                expected_evidence,
            )

            report.append(
                f"#{item.rank} "
                f"selected={selected} "
                f"score={item.score:.6f} "
                f"type={item.source_type} "
                f"authority={item.authority} "
                f"source={item.source!r} "
                f"evidence={evidence_match}"
            )

        if is_safety_case:
            report.append(
                "RESULT: SAFETY CASE — excluded"
            )

            print(
                "Safety case — excluded from metrics"
            )

            continue

        evaluable += 1

        full_rank = find_evidence_rank(
            full_evidence,
            expected_evidence,
        )

        selected_rank = find_evidence_rank(
            selected_evidence,
            expected_evidence,
        )

        full_preferred_rank = find_preferred_rank(
            full_evidence,
            expected_evidence,
            preferred_sources,
        )

        selected_preferred_rank = find_preferred_rank(
            selected_evidence,
            expected_evidence,
            preferred_sources,
        )

        if full_rank is not None:
            full_top5_hits += 1

        if selected_rank is not None:
            selected_top3_hits += 1

        if full_preferred_rank is not None:
            preferred_top5_hits += 1

        if selected_preferred_rank is not None:
            preferred_top3_hits += 1

        lost = (
            full_rank is not None
            and selected_rank is None
        )

        preferred_lost = (
            full_preferred_rank is not None
            and selected_preferred_rank is None
        )

        if lost:
            lost_cases.append(case_id)

        if preferred_lost:
            preferred_lost_cases.append(case_id)

        report.append(
            f"FULL TOP5 EVIDENCE RANK: {full_rank}"
        )

        report.append(
            f"SELECTED TOP3 EVIDENCE RANK: {selected_rank}"
        )

        report.append(
            f"FULL TOP5 PREFERRED RANK: {full_preferred_rank}"
        )

        report.append(
            f"SELECTED TOP3 PREFERRED RANK: "
            f"{selected_preferred_rank}"
        )

        report.append(
            f"EVIDENCE LOST: {lost}"
        )

        print(
            f"TOP5 evidence rank: {full_rank}"
        )

        print(
            f"TOP3 evidence rank: {selected_rank}"
        )

        if lost:
            print("!!! EVIDENCE LOST BY TOP3 !!!")

    summary = [
        "",
        "=" * 100,
        "EVIDENCE SELECTION EXPERIMENT — NAIVE TOP3",
        "=" * 100,
        "",
        f"Evaluable cases: {evaluable}",
        "",
        (
            f"Full TOP5 Evidence Hit: "
            f"{full_top5_hits}/{evaluable}"
        ),
        (
            f"Selected TOP3 Evidence Hit: "
            f"{selected_top3_hits}/{evaluable}"
        ),
        "",
        (
            f"Full TOP5 Preferred Hit: "
            f"{preferred_top5_hits}/{evaluable}"
        ),
        (
            f"Selected TOP3 Preferred Hit: "
            f"{preferred_top3_hits}/{evaluable}"
        ),
        "",
        (
            "Evidence lost cases: "
            + (
                ", ".join(lost_cases)
                if lost_cases
                else "none"
            )
        ),
        (
            "Preferred evidence lost cases: "
            + (
                ", ".join(preferred_lost_cases)
                if preferred_lost_cases
                else "none"
            )
        ),
        "",
    ]

    report.extend(summary)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("TOP3 SELECTION EXPERIMENT COMPLETE")
    print("=" * 100)

    print(
        f"Full TOP5 Evidence Hit: "
        f"{full_top5_hits}/{evaluable}"
    )

    print(
        f"Selected TOP3 Evidence Hit: "
        f"{selected_top3_hits}/{evaluable}"
    )

    print()

    print(
        f"Full TOP5 Preferred Hit: "
        f"{preferred_top5_hits}/{evaluable}"
    )

    print(
        f"Selected TOP3 Preferred Hit: "
        f"{preferred_top3_hits}/{evaluable}"
    )

    print()

    print(
        "Evidence lost cases:",
        ", ".join(lost_cases)
        if lost_cases
        else "none",
    )

    print(
        "Preferred evidence lost cases:",
        ", ".join(preferred_lost_cases)
        if preferred_lost_cases
        else "none",
    )

    print()
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()