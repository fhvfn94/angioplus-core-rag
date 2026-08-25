# -*- coding: utf-8 -*-
"""LiveKit voice component boundary for AngioPlus Core RAG.

This package contains ONLY the thin client boundary between the future
LiveKit voice component and the existing RAG API. It must never import
app.main / app.* RAG implementation and must never access Qdrant directly.
"""
