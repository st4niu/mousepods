# SPDX-License-Identifier: Apache-2.0
"""Glue for the Games page: Steam discovery, games outside Steam (any Wine prefix), prefix installer
and saved per-game settings."""
import json
import os
import re
import threading
import unicodedata

import bridge_install
import game_bridge
import steam_games
from game_profiles import effective_profile, normalize_baseline, normalize_profile

HERE = os.path.dirname(os.path.abspath(__file__))
DLL_DIRS = (os.path.join(HERE, "native", "out"), "/app/share/mousepods/native")


WINE_ID = re.compile(r"^wine-[a-z0-9]+(?:-[0-9]+)?$")       # same shape as game_bridge._APPID


def is_running(appid, cfg_path=None):
    """A Steam game by its appid; a game outside Steam by a process using its Wine prefix."""
    if WINE_ID.match(str(appid)):
        game = _wine_game(str(appid), wine_entries(cfg_path).get(str(appid))) if cfg_path else None
        return bool(game and game.prefix) and os.path.realpath(game.prefix) in bridge_install.running_prefixes()
    return bridge_install.process_running(appid)


def steam_roots():
    """Every Steam install, native first (a game in both counts as the native one)."""
    forced = os.environ.get("MOUSEPODS_STEAM_ROOT")
    if forced:
        return [steam_games.SteamRoot(os.path.realpath(forced), False)]
    roots = steam_games.find_steam_roots(os.path.expanduser("~"))
    return sorted(roots, key=lambda r: r.flatpak)


# Steam's cached library art, wide header first (old layout <appid>_<name>, new <appid>/<name>
# or <appid>/<hash>/<name>). No logo/icon: transparent or tiny, not a cover.
COVER_NAMES = ("header.jpg", "library_header.jpg", "library_hero.jpg", "library_600x900.jpg",
               "header.png", "library_header.png", "library_hero.png", "library_600x900.png")


def cover_path(appid, roots=None):
    """The cached cover image of a game (real path inside a librarycache dir), or None. Read-only."""
    appid = str(appid)
    if not (0 < len(appid) <= 12 and appid.isascii() and appid.isdigit()):   # "²" is a digit too
        return None
    for root in steam_roots() if roots is None else roots:
        cache = os.path.realpath(os.path.join(root.path, "appcache", "librarycache"))
        if not os.path.isdir(cache):
            continue
        own = os.path.join(cache, appid)
        dirs = [own]
        try:
            dirs += sorted(e.path for e in os.scandir(own) if e.is_dir())
        except OSError:
            pass
        for name in COVER_NAMES:
            for p in [os.path.join(cache, f"{appid}_{name}")] + [os.path.join(d, name) for d in dirs]:
                real = os.path.realpath(p)
                if os.path.commonpath([real, cache]) == cache and os.path.isfile(real):
                    return real
    return None


def dll_dir():
    forced = os.environ.get("MOUSEPODS_DLL_DIR")
    if forced:
        return forced
    for d in DLL_DIRS:
        if os.path.isfile(os.path.join(d, "NPClient64.dll")):
            return d
    return DLL_DIRS[0]


def shm_dir():
    return os.environ.get("MOUSEPODS_SHM_DIR", "/dev/shm")


def _saved(cfg_path):
    try:
        with open(cfg_path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def saved_games(cfg_path):
    games = _saved(cfg_path).get("games")
    return games if isinstance(games, dict) else {}


def saved_baseline(cfg_path):
    return normalize_baseline(_saved(cfg_path).get("game_baseline"))


# ---- games outside Steam: a name and a Wine prefix (Lutris, Heroic, Bottles, plain Wine, the LUG
# Helper for Star Citizen). Saved under "wine_games" as {id: {"name", "prefix"}}; a removed one is null.

def wine_entries(cfg_path):
    games = _saved(cfg_path).get("wine_games") if cfg_path else None
    if not isinstance(games, dict):
        return {}
    return {k: v for k, v in games.items() if WINE_ID.match(str(k)) and isinstance(v, dict)
            and isinstance(v.get("name"), str) and isinstance(v.get("prefix"), str)}


def _wine_game(gid, entry):
    if not entry:
        return None
    prefix = entry["prefix"]
    has = os.path.isfile(os.path.join(prefix, "user.reg"))
    return steam_games.Game(appid=gid, name=entry["name"], library="", prefix=prefix if has else None,
                            proton=None, launch_options="", flatpak=False)


def wine_games(cfg_path):
    return [_wine_game(k, v) for k, v in wine_entries(cfg_path).items()]


def in_flatpak():
    return os.path.exists("/.flatpak-info")


def _slug(name):
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", plain)[:24] or "game"


def resolve_prefix(path):
    """The Wine prefix (folder holding user.reg) a user pointed at: the prefix itself, or a folder
    with it in pfx/ (Proton layout). Raises BridgeError."""
    path = os.path.realpath(os.path.expanduser((path or "").strip()))
    if not path or path == "/":
        raise bridge_install.BridgeError("bad_prefix")
    for candidate in (path, os.path.realpath(os.path.join(path, "pfx"))):
        if os.path.isfile(os.path.join(candidate, "user.reg")):
            return candidate
    if not os.path.exists(path) and in_flatpak():
        raise bridge_install.BridgeError("prefix_hidden", path)    # outside what the sandbox may see
    raise bridge_install.BridgeError("bad_prefix")


def add_wine(cfg_path, name, prefix):
    """New game outside Steam; returns (id, entry) to save under "wine_games". Raises BridgeError."""
    name = " ".join(str(name or "").split())[:80]
    if not name:
        raise bridge_install.BridgeError("bad_name")
    prefix = resolve_prefix(prefix)
    taken = wine_entries(cfg_path)
    steam = [g.prefix for copies in _installs()[1].values() for g in copies.values() if g.prefix]
    if any(os.path.realpath(p) == prefix for p in [e["prefix"] for e in taken.values()] + steam):
        raise bridge_install.BridgeError("already_added")
    base, n = "wine-" + _slug(name), 2
    gid = base
    while gid in taken or gid in saved_games(cfg_path):
        gid, n = f"{base}-{n}", n + 1
    return gid, {"name": name, "prefix": prefix}


def remove_wine(cfg_path, game):
    """Forget a game outside Steam; only while it is off, so no bridge stays behind in its prefix.
    A prefix that is gone holds no bridge any more: then it goes even if it was on."""
    if not game.prefix:
        return
    if (saved_games(cfg_path).get(game.appid) or {}).get("enabled") or bridge_install.status(game) == "on":
        raise bridge_install.BridgeError("switch_off_first")


def needs_closed_confirmation(appid):
    """In the Flatpak, MousePods sees only its own processes: whether a game outside Steam runs
    cannot be known, so the user confirms it is closed before its prefix is changed."""
    return bool(WINE_ID.match(str(appid))) and in_flatpak()


def refresh_tables(cfg_path):
    """Games switched on before their TrackIR table existed get it now, without an off / on.
    Returns the names of the games whose prefix was updated."""
    saved, done = saved_games(cfg_path), []
    for appid, entry in saved.items():
        if not (entry or {}).get("enabled"):
            continue
        if needs_closed_confirmation(appid):
            continue                     # cannot tell whether it runs: only on the user's word
        game = find(appid, cfg_path)
        if game is None or not game.prefix:
            continue
        try:
            with _PREFIX_LOCK:
                changed = bridge_install.refresh_tables(game, lambda a: is_running(a, cfg_path))
            if changed:
                done.append(game.name)
        except OSError:
            continue                     # unreadable prefix: the switch shows it on the next try
    return done


def _installs():
    """(roots, {appid: {flatpak: Game}}) from every Steam install. All installed games: one never
    started with Proton has no prefix yet and shows a hint."""
    roots, installs = steam_roots(), {}
    for root in roots:
        for g in steam_games.list_games(root):
            installs.setdefault(g.appid, {}).setdefault(g.flatpak, g)
    return roots, installs


def _pick(copies, entry):
    """The install the user chose for a game in both Steams (saved "flatpak"), native otherwise."""
    want = bool((entry or {}).get("flatpak"))
    return copies.get(want) or copies.get(False) or copies.get(True)


def _games(saved=None):
    roots, installs = _installs()
    saved = saved or {}
    games = [_pick(copies, saved.get(appid)) for appid, copies in installs.items()]
    return roots, sorted(games, key=lambda g: g.name.lower()), installs


def find(appid, cfg_path=None):
    if WINE_ID.match(str(appid)):
        return _wine_game(str(appid), wine_entries(cfg_path).get(str(appid))) if cfg_path else None
    _, installs = _installs()
    copies = installs.get(str(appid))
    if not copies:
        return None
    return _pick(copies, saved_games(cfg_path).get(str(appid)) if cfg_path else None)


def list_payload(cfg_path):
    refresh_tables(cfg_path)
    saved = saved_games(cfg_path)
    roots, games, installs = _games(saved)
    baseline = saved_baseline(cfg_path)
    running = bridge_install.running_appids()        # one /proc pass for the whole list
    wine = wine_games(cfg_path)
    prefixes = bridge_install.running_prefixes() if wine else set()
    out = []
    adopted = {}
    for g in games:
        entry = saved.get(g.appid) or {}
        status = bridge_install.status(g)
        if status == "on" and not entry.get("enabled"):
            # our bridge is in the prefix but this config doesn't know (set up by another
            # MousePods, e.g. a distrobox): show it as on and feed it, so the switch tells the truth
            entry = adopted[g.appid] = {"enabled": True, "flatpak": g.flatpak,
                                        "profile": normalize_profile(entry.get("profile"))}
        out.append({"appid": g.appid, "name": g.name, "has_prefix": bool(g.prefix), "proton": g.proton,
                    "flatpak": g.flatpak, "both": len(installs[g.appid]) > 1, "status": status,
                    "foreign_tracker": bridge_install.foreign_tracker(g),
                    "enabled": bool(entry.get("enabled")), "running": g.appid in running,
                    "launch_option_hint": "opentrack-launcher" in g.launch_options,
                    "cover": cover_path(g.appid, roots) is not None,
                    "profile": effective_profile(normalize_profile(entry.get("profile")), baseline)})
    for g in sorted(wine, key=lambda g: g.name.lower()):
        entry = saved.get(g.appid) or {}
        prefix = wine_entries(cfg_path)[g.appid]["prefix"]
        out.append({"appid": g.appid, "name": g.name, "has_prefix": bool(g.prefix), "proton": None,
                    "flatpak": False, "both": False, "status": bridge_install.status(g),
                    "foreign_tracker": bridge_install.foreign_tracker(g), "wine": True, "prefix": prefix,
                    "enabled": bool(entry.get("enabled")),
                    "running": os.path.realpath(prefix) in prefixes,
                    "launch_option_hint": False, "cover": False,
                    "profile": effective_profile(normalize_profile(entry.get("profile")), baseline)})
    if adopted:
        from airpods_mouse_controller import update_config_file
        update_config_file(cfg_path, {"games": adopted})
    missing = [p for r in roots for p in steam_games.missing_libraries(r)]
    return {"steam_found": bool(roots), "baseline": baseline, "steam_flatpak": False, "games": out,
            "flatpak_sandbox": in_flatpak(),
            "missing_libraries": list(dict.fromkeys(missing))}


_PREFIX_LOCK = threading.Lock()      # one prefix edit at a time (web requests run in parallel)


def set_enabled(cfg_path, game, enabled):
    """Install or remove the bridge in the game's prefix; returns the entry to save.
    Raises bridge_install.BridgeError."""
    with _PREFIX_LOCK:
        return _set_enabled(cfg_path, game, enabled)


def _set_enabled(cfg_path, game, enabled):
    entry = dict(saved_games(cfg_path).get(game.appid) or {})
    running = lambda appid: is_running(appid, cfg_path)      # noqa: E731
    if enabled:
        pose = game_bridge.pose_path(game.appid, shm_dir(), game.flatpak)
        bridge_install.install(game, dll_dir(), pose, running)
    elif bridge_install.status(game) == "on" or bridge_install.has_our_files(game):
        bridge_install.uninstall(game, running)
    entry["enabled"] = bool(enabled)
    entry["flatpak"] = game.flatpak                 # the bridge writes where that Steam looks
    entry["profile"] = normalize_profile(entry.get("profile"))
    return entry


def set_steam(cfg_path, game, flatpak):
    """For a game in both Steams: use the Flatpak one (True) or the native one. Only while it is off,
    so the bridge never stays behind in the other prefix."""
    entry = dict(saved_games(cfg_path).get(game.appid) or {})
    if entry.get("enabled"):
        raise bridge_install.BridgeError("switch_off_first")
    entry["flatpak"] = bool(flatpak)
    entry["profile"] = normalize_profile(entry.get("profile"))
    return entry


def set_profile(cfg_path, game, profile):
    entry = dict(saved_games(cfg_path).get(game.appid) or {})
    entry["enabled"] = bool(entry.get("enabled"))
    entry["profile"] = normalize_profile(profile)
    return entry
