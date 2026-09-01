from __future__ import annotations

import json
import re
import statistics
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
    / "evidence_compaction_v1.txt"
)


MAX_FRAGMENT_CHARS = 700
CONTEXT_LINES = 1


STOP_WORDS = {
    "какой",
    "какая",
    "какие",
    "какое",
    "можно",
    "нужно",
    "ли",
    "для",
    "при",
    "перед",
    "после",
    "во",
    "в",
    "на",
    "из",
    "по",
    "и",
    "или",
    "у",
    "с",
    "со",
    "к",
    "от",
    "до",
    "используется",
    "поддерживает",
    "поддерживает ли",
    "angioPlus".lower(),
    "core",
}


def normalize_text(value: str) -> str:
    return (
        value.lower()
        .replace("ё", "е")
        .replace("µ", "μ")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )


def tokenize(value: str) -> set[str]:
    normalized = normalize_text(value)

    tokens = set(
        re.findall(
            r"[a-zа-я0-9μ][a-zа-я0-9μ.+_-]*",
            normalized,
            flags=re.IGNORECASE,
        )
    )

    return {
        token
        for token in tokens
        if len(token) >= 2
        and token not in STOP_WORDS
    }


def split_lines(text: str) -> list[str]:
    lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        lines.append(line)

    return lines


def line_score(
    line: str,
    query_tokens: set[str],
) -> int:
    if not query_tokens:
        return 0

    line_tokens = tokenize(line)

    overlap = query_tokens & line_tokens

    score = len(overlap) * 10

    normalized_line = normalize_text(line)

    for token in query_tokens:
        if token in normalized_line:
            score += 1

    return score


def compact_text(
    *,
    query: str,
    text: str,
    max_chars: int = MAX_FRAGMENT_CHARS,
) -> str:
    if len(text) <= max_chars:
        return text

    lines = split_lines(text)

    if not lines:
        return text[:max_chars]

    query_tokens = tokenize(query)

    scored = [
        (
            line_score(line, query_tokens),
            index,
        )
        for index, line in enumerate(lines)
    ]

    scored.sort(
        key=lambda item: (
            item[0],
            -item[1],
        ),
        reverse=True,
    )

    best_score, best_index = scored[0]

    # Если вообще нет lexical match,
    # ничего "умного" не угадываем.
    # Сохраняем начало chunk как безопасный fallback.
    if best_score <= 0:
        return text[:max_chars]

    selected_indexes: set[int] = set()

    for index in range(
        max(0, best_index - CONTEXT_LINES),
        min(
            len(lines),
            best_index + CONTEXT_LINES + 1,
        ),
    ):
        selected_indexes.add(index)

    # Добавим ещё наиболее релевантные строки,
    # пока не достигнем лимита.
    for score, index in scored[1:]:
        if score <= 0:
            break

        selected_indexes.add(index)

        candidate = "\n".join(
            lines[i]
            for i in sorted(selected_indexes)
        )

        if len(candidate) > max_chars:
            selected_indexes.remove(index)
            break

    fragment = "\n".join(
        lines[index]
        for index in sorted(selected_indexes)
    )

    if len(fragment) > max_chars:
        fragment = fragment[:max_chars]

    return fragment


def matches_any_evidence(
    text: str,
    expected_evidence: list[str],
) -> bool:
    normalized_text = normalize_text(text)

    return any(
        normalize_text(expected) in normalized_text
        for expected in expected_evidence
    )


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

    full_hits = 0
    compact_hits = 0

    lost_cases: list[str] = []

    full_sizes: list[int] = []
    compact_sizes: list[int] = []

    report: list[str] = []

    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        question = str(case["question"])

        expected_evidence = list(
            case.get("expected_evidence_any", [])
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

        compact_evidence: list[tuple[EvidenceChunk, str]] = []

        for item in full_evidence:
            fragment = compact_text(
                query=question,
                text=item.text,
            )

            compact_evidence.append(
                (item, fragment)
            )

        full_payload_chars = sum(
            len(item.text)
            for item in full_evidence
        )

        compact_payload_chars = sum(
            len(fragment)
            for _, fragment in compact_evidence
        )

        full_sizes.append(full_payload_chars)
        compact_sizes.append(compact_payload_chars)

        full_match = any(
            matches_any_evidence(
                item.text,
                expected_evidence,
            )
            for item in full_evidence
        )

        compact_match = any(
            matches_any_evidence(
                fragment,
                expected_evidence,
            )
            for _, fragment in compact_evidence
        )

        report.append("=" * 100)
        report.append(f"CASE: {case_id}")
        report.append(f"QUESTION: {question}")
        report.append(
            f"FULL PAYLOAD CHARS: {full_payload_chars}"
        )
        report.append(
            f"COMPACT PAYLOAD CHARS: {compact_payload_chars}"
        )

        if full_payload_chars:
            reduction = (
                1
                - compact_payload_chars
                / full_payload_chars
            ) * 100
        else:
            reduction = 0.0

        report.append(
            f"REDUCTION: {reduction:.1f}%"
        )

        for item, fragment in compact_evidence:
            original_match = matches_any_evidence(
                item.text,
                expected_evidence,
            )

            compact_item_match = matches_any_evidence(
                fragment,
                expected_evidence,
            )

            report.append(
                f"#{item.rank} "
                f"source={item.source!r} "
                f"original_chars={len(item.text)} "
                f"compact_chars={len(fragment)} "
                f"original_evidence={original_match} "
                f"compact_evidence={compact_item_match}"
            )

            report.append("COMPACT:")
            report.append(fragment)
            report.append("-" * 80)

        if is_safety_case:
            print(
                "Safety case — excluded from retrieval metrics"
            )

            report.append(
                "RESULT: SAFETY CASE — excluded"
            )

            continue

        evaluable += 1

        if full_match:
            full_hits += 1

        if compact_match:
            compact_hits += 1

        lost = full_match and not compact_match

        if lost:
            lost_cases.append(case_id)

        print(
            f"Full evidence: {full_match}"
        )
        print(
            f"Compact evidence: {compact_match}"
        )
        print(
            f"Payload: {full_payload_chars} -> "
            f"{compact_payload_chars} chars "
            f"(-{reduction:.1f}%)"
        )

        if lost:
            print(
                "!!! EVIDENCE LOST BY COMPACTION !!!"
            )

    median_full = statistics.median(
        full_sizes
    )

    median_compact = statistics.median(
        compact_sizes
    )

    total_full = sum(full_sizes)
    total_compact = sum(compact_sizes)

    total_reduction = (
        (
            1
            - total_compact
            / total_full
        )
        * 100
        if total_full
        else 0.0
    )

    summary = [
        "",
        "=" * 100,
        "EVIDENCE COMPACTION V1 COMPLETE",
        "=" * 100,
        "",
        f"Evaluable cases: {evaluable}",
        "",
        (
            f"Full TOP5 Evidence Hit: "
            f"{full_hits}/{evaluable}"
        ),
        (
            f"Compact TOP5 Evidence Hit: "
            f"{compact_hits}/{evaluable}"
        ),
        "",
        (
            f"Median full payload: "
            f"{median_full:.0f} chars"
        ),
        (
            f"Median compact payload: "
            f"{median_compact:.0f} chars"
        ),
        "",
        (
            f"Total full payload: "
            f"{total_full} chars"
        ),
        (
            f"Total compact payload: "
            f"{total_compact} chars"
        ),
        (
            f"Total reduction: "
            f"{total_reduction:.1f}%"
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
    print("EVIDENCE COMPACTION V1 COMPLETE")
    print("=" * 100)

    print(
        f"Full TOP5 Evidence Hit: "
        f"{full_hits}/{evaluable}"
    )

    print(
        f"Compact TOP5 Evidence Hit: "
        f"{compact_hits}/{evaluable}"
    )

    print()

    print(
        f"Median full payload: "
        f"{median_full:.0f} chars"
    )

    print(
        f"Median compact payload: "
        f"{median_compact:.0f} chars"
    )

    print()

    print(
        f"Total full payload: "
        f"{total_full} chars"
    )

    print(
        f"Total compact payload: "
        f"{total_compact} chars"
    )

    print(
        f"Total reduction: "
        f"{total_reduction:.1f}%"
    )

    print()

    print(
        "Evidence lost cases:",
        ", ".join(lost_cases)
        if lost_cases
        else "none",
    )

    print()

    print(
        f"Report: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()