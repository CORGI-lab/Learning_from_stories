# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import textworld
from textworld.challenges.coin_collector import make


def test_making_coin_collector():
    expected = {
        1: {"quest_length": 1, "nb_rooms": 1},
        100: {"quest_length": 100, "nb_rooms": 100},
        101: {"quest_length": 1, "nb_rooms": 2},
        200: {"quest_length": 100, "nb_rooms": 200},
        201: {"quest_length": 1, "nb_rooms": 3},
        300: {"quest_length": 100, "nb_rooms": 300},
    }
    for level in [1, 100, 101, 200, 201, 300]:
        options = textworld.GameOptions()
        options.seeds = 1234

        settings = {"level": level, "force_entity_numbering": True}
        game = make(settings, options)
        assert len(game.quests[0].commands) == expected[level]["quest_length"]
        assert len(game.world.rooms) == expected[level]["nb_rooms"]
