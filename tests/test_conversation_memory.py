# -*- coding: utf-8 -*-
import time

from app.conversation_memory import ConversationMemory, Turn


def test_isolation_by_conversation_and_user():
    mem = ConversationMemory(ttl_seconds=1800, max_turns=3)
    mem.append_turn("chatA", "user1", "вопрос A1", "standalone A1")
    mem.append_turn("chatB", "user2", "вопрос B1", "standalone B1")

    snap_a = mem.get_snapshot("chatA", "user1")
    snap_b = mem.get_snapshot("chatB", "user2")

    assert len(snap_a.turns) == 1
    assert len(snap_b.turns) == 1
    assert snap_a.turns[0].standalone_question == "standalone A1"
    assert snap_b.turns[0].standalone_question == "standalone B1"


def test_same_conversation_different_user_isolated():
    mem = ConversationMemory(ttl_seconds=1800, max_turns=3)
    mem.append_turn("chatX", "user1", "q1", "s1")
    mem.append_turn("chatX", "user2", "q2", "s2")

    snap_1 = mem.get_snapshot("chatX", "user1")
    snap_2 = mem.get_snapshot("chatX", "user2")
    assert len(snap_1.turns) == 1
    assert snap_1.turns[0].standalone_question == "s1"
    assert snap_2.turns[0].standalone_question == "s2"


def test_max_turns_bounded_to_three():
    mem = ConversationMemory(ttl_seconds=1800, max_turns=3)
    for i in range(6):
        mem.append_turn("c", "u", f"n{i}", f"s{i}")

    snap = mem.get_snapshot("c", "u")
    assert len(snap.turns) == 3
    # Only the last three survive.
    assert [t.standalone_question for t in snap.turns] == ["s3", "s4", "s5"]


def test_ttl_expires_context():
    mem = ConversationMemory(ttl_seconds=0, max_turns=3)  # everything stale
    mem.append_turn("c", "u", "n1", "s1")
    snap = mem.get_snapshot("c", "u")
    assert len(snap.turns) == 0


def test_ttl_keeps_fresh_context():
    mem = ConversationMemory(ttl_seconds=3600, max_turns=3)
    mem.append_turn("c", "u", "n1", "s1")
    snap = mem.get_snapshot("c", "u")
    assert len(snap.turns) == 1


def test_empty_turn_not_stored():
    mem = ConversationMemory(ttl_seconds=1800, max_turns=3)
    mem.append_turn("c", "u", "", "")
    snap = mem.get_snapshot("c", "u")
    assert len(snap.turns) == 0


def test_turn_is_frozen_dataclass_without_was_secret():
    import dataclasses
    fields = {f.name for f in dataclasses.fields(Turn)}
    assert fields == {"normalized_question", "standalone_question", "timestamp"}
    assert dataclasses.is_dataclass(Turn)
    assert Turn.__dataclass_params__.frozen is True
