#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import asyncio
from unittest.mock import MagicMock

import pytest

from rag.graphrag.search import KGSearch


def _make_kgsearch(capture):
    # Bypass Dealer.__init__ (which builds a FulltextQueryer); we only need
    # dataStore + get_vector for these helpers.
    kg = object.__new__(KGSearch)

    def _search(*args, **kwargs):
        # match_expressions is the 4th positional arg to dataStore.search
        capture["match"] = args[3]
        return MagicMock()

    kg.dataStore = MagicMock()
    kg.dataStore.search.side_effect = _search
    return kg


SENTINEL = object()


async def _fake_get_vector(*args, **kwargs):
    return SENTINEL


@pytest.mark.asyncio
async def test_get_relevant_ents_by_keywords_awaits_vector(monkeypatch):
    # Regression for #14711: the async get_vector must be awaited before being
    # handed to dataStore.search, not passed through as a bare coroutine.
    capture = {}
    kg = _make_kgsearch(capture)
    monkeypatch.setattr(kg, "get_vector", _fake_get_vector)
    monkeypatch.setattr(kg, "_ent_info_from_", lambda es_res, sim_thr: {})

    await kg.get_relevant_ents_by_keywords(["k"], {}, ["idx"], ["kb"], MagicMock())

    match = capture["match"][0]
    assert not asyncio.iscoroutine(match), "un-awaited coroutine leaked into search"
    assert match is SENTINEL


@pytest.mark.asyncio
async def test_get_relevant_relations_by_txt_awaits_vector(monkeypatch):
    capture = {}
    kg = _make_kgsearch(capture)
    monkeypatch.setattr(kg, "get_vector", _fake_get_vector)
    monkeypatch.setattr(kg, "_relation_info_from_", lambda es_res, sim_thr: {})

    await kg.get_relevant_relations_by_txt("q", {}, ["idx"], ["kb"], MagicMock())

    match = capture["match"][0]
    assert not asyncio.iscoroutine(match), "un-awaited coroutine leaked into search"
    assert match is SENTINEL
