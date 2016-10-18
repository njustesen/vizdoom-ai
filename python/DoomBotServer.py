from vizdoom import *
from vizdoom import ScreenResolution as res

class DoomBotServer(object):

    def __DoomGame__(self, graphics, fast, bots):

        self.game = DoomGame()

        self.game.set_vizdoom_path("../../bin/vizdoom")

        # Use CIG example config or your own.
        self.game.load_config("../config/cig_train.cfg")
        self.game.set_screen_resolution(res.RES_320X240)
        self.game.set_window_visible(graphics)

        # Select game and map you want to use.
        self.game.set_doom_game_path("../../scenarios/freedoom2.wad")
        # game.set_doom_game_path("../../scenarios/doom2.wad")  # Not provided with environment due to licences

        self.game.set_doom_map("map01")  # Limited deathmatch.
        # game.set_doom_map("map02")  # Full deathmatch.

        # Start multiplayer game only with your AI (with options that will be used in the competition, details in cig_host example).
        self.game.add_game_args("-host 1 -deathmatch +timelimit 2.0 "
                           "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")

        # Name your agent and select color
        # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
        self.game.add_game_args("+name AI +colorset 0")

        # Multiplayer requires the use of asynchronous modes, but when playing only with bots, synchronous modes can also be used.
        if fast:
            self.game.set_mode(Mode.PLAYER)
        else:
            self.game.set_mode(Mode.ASYNC_PLAYER)

        self.game.init()

        # Three example sample actions
        self.actions = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],    # TURN_LEFT
            [0, 1, 0, 0, 0, 0, 0, 0, 0],    # TURN_RIGHT
            [0, 0, 1, 0, 0, 0, 0, 0, 0],    # ATTACK
            [0, 0, 0, 1, 0, 0, 0, 0, 0],    # MOVE_RIGHT
            [0, 0, 0, 0, 1, 0, 0, 0, 0],    # MOVE_LEFT
            [0, 0, 0, 0, 0, 1, 0, 0, 0],    # MOVE_FORWARD
            [0, 0, 0, 0, 0, 0, 1, 0, 0]     # MOVE_BACKWARD
        ]

        # Play with this many bots
        self.bots = bots

    def reset(self, bots):
        #print("Starting a new game.\n")
        self.game.send_game_command("removebots")
        for i in range(self.bots):
            self.game.send_game_command("addbot")
        self.game.new_episode()
        #print("New game started with " + str(bots) + " bots\n")
