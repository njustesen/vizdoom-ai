#!/usr/bin/env python
# -*- coding: utf-8 -*-

from vizdoom import *

class DoomServer:

    def __init__(self, screen_resolution, config_file_path, deathmatch=False, bots=7, visual=False, async=True):
        self.screen_resolution = screen_resolution
        self.deathmatch = deathmatch
        self.bots = bots
        self.visual = visual
        self.async = async
        self.config_file_path = config_file_path

    def start_game(self):
        print("Initializing doom...")
        game = DoomGame()
        game.load_config(self.config_file_path)
        game.set_window_visible(self.visual)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_screen_resolution(self.screen_resolution)
        if self.deathmatch:
            #self.game.set_doom_map("map01")  # Limited deathmatch.
            game.set_doom_map("map02")  # Full deathmatch.
            # Start multiplayer game only with your AI (with options that will be used in the competition, details in cig_host example).
            game.add_game_args("-host 1 -deathmatch +timelimit 2.0 "
                               "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")
            # Name your agent and select color
            # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
            game.add_game_args("+name AI +colorset 0")

        if self.async:
            game.set_mode(Mode.ASYNC_PLAYER)
        else:
            game.set_mode(Mode.PLAYER)

        game.init()

        if self.deathmatch:
            game.send_game_command("removebots")
            for i in range(self.bots):
                game.send_game_command("addbot")

        #self.game.new_episode()

        print("Doom initialized.")
        return game

    def restart_game(self, game):
        if self.deathmatch:
            game.close()
            return self.start_game()
        game.new_episode()
        return game