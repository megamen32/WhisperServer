"""Lightweight transcription contracts shared by the local model adapters."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ParakeetWord:
    start: float
    end: float
    word: str


@dataclass(frozen=True)
class ParakeetSegment:
    start: float
    end: float
    text: str
    words: tuple[ParakeetWord, ...] = ()


@dataclass(frozen=True)
class ParakeetInfo:
    language: str
    language_probability: float
