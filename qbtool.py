import logging
import re
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from os import fspath
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import bencodepy
import qbittorrentapi
from genutility.args import is_dir, is_file
from genutility.json import json_lines, read_json
from genutility.torrent import get_torrent_hash

import common

DEFAULT_CONFIG = {
    "category-name": "_Unregistered",
    "host": "localhost:8080",
    "scrapes-file": "scrapes.jl",
}

filtermap = {
    "paused": ("pausedDL", "pausedUP"),
    "completed": ("uploading", "stalledUP", "pausedUP", "forcedUP"),
    "errored": {"missingFiles"},
    "stalled": {"stalledUP", "stalledDL"},
    "downloading": {"downloading", "pausedDL", "forcedDL"},
}


def get_private_torrents(client: qbittorrentapi.Client, **kwargs) -> Iterator[Tuple[Any, List[dict]]]:
    for torrent in client.torrents_info(**kwargs):
        trackers = client.torrents_trackers(torrent.hash)
        assert trackers[0]["url"] == "** [DHT] **"
        assert trackers[1]["url"] == "** [PeX] **"
        assert trackers[2]["url"] == "** [LSD] **"
        is_private = trackers[0]["status"] == 0

        if is_private:
            yield torrent, trackers[3:]


def get_bad_torrents(client: qbittorrentapi.Client) -> Iterator[Dict[str, Any]]:
    for torrent, trackers in get_private_torrents(client, filter="seeding"):
        for tracker in trackers:
            if tracker["status"] == 4:
                yield {
                    "hash": torrent.hash,
                    "name": torrent.name,
                    "msg": tracker["msg"],
                }


def paused_private(client: qbittorrentapi.Client) -> Iterator[Any]:
    for torrent, trackers in get_private_torrents(client, filter="paused"):
        if torrent.state == "pausedUP":
            yield torrent


def categorize_private_with_failed_trackers(client: qbittorrentapi.Client, args: Namespace):
    new = {t["hash"]: t["name"] for t in get_bad_torrents(client)}
    old = {t["hash"]: t["name"] for t in client.torrents_info(category=args.category_name)}

    add = new.keys() - old.keys()
    remove = old.keys() - new.keys()

    if args.do_move:
        try:
            client.torrents_create_category(args.category_name)
        except qbittorrentapi.exceptions.Conflict409Error:
            pass  # ignore existing categories

        client.torrents_set_category(args.category_name, add)
        client.torrents_set_category("", remove)
    else:
        for hash_ in add:
            print(f"Add {hash_} {new[hash_]}")
        for hash_ in remove:
            print(f"Remove {hash_} {old[hash_]}")

    print(f"Total: {len(new)}, add: {len(add)}, remove: {len(remove)}")


def torrent_exists(client: qbittorrentapi.Client, hash: str) -> bool:

    try:
        client.torrents_properties(hash)
        return True
    except qbittorrentapi.NotFound404Error:
        return False


def remove_loaded_torrents(client: qbittorrentapi.Client, args: Namespace) -> None:

    num_remove = 0
    num_keep = 0

    for path in common._iter_torrent_files(args.path, args.recursive):
        try:
            hash = get_torrent_hash(fspath(path))
        except bencodepy.exceptions.BencodeDecodeError:
            logging.error("Failed to read torrent file: <%s>", path)
            continue

        # hash = bytes.fromhex(hash)
        exists = torrent_exists(client, hash)
        if exists:
            logging.info("Removed %s <%s>", hash, path)
            num_remove += 1
            if args.do_remove:
                try:
                    path.unlink()
                except PermissionError as e:
                    logging.error("Failed to remove %s: %s", path, e)
        else:
            num_keep += 1
            logging.debug("Keep %s <%s>", hash, path)

    print(f"Remove {num_remove} files, keep {num_keep} files")


def list_paused_private(client: qbittorrentapi.Client, args: Namespace) -> None:
    torrents = list(paused_private(client))
    for torrent in torrents:
        print(torrent.hash, torrent.name)


def scrape_loaded(client: qbittorrentapi.Client, args: Namespace) -> None:
    batchsize = 50
    delay = 5.0

    all = defaultdict(set)
    for torrent, trackers in get_private_torrents(client):
        all[torrent["tracker"]].add(torrent.hash)

    with json_lines.from_path(args.out, "wt") as fw:
        for obj in common._scrape_all(all, batchsize, delay):
            fw.write(obj)


def move_by_availability(client: qbittorrentapi.Client, args: Namespace) -> None:

    import pandas as pd

    tuples = []
    with json_lines.from_path(args.scrapes_file, "rt") as jl:
        for tracker_url, hash, info in jl:
            try:
                tuples.append((tracker_url, hash, info["complete"]))
            except KeyError:
                logging.warning("num_complete not found for %s, %s: %s", tracker_url, hash, info)

    df = pd.DataFrame.from_records(tuples, columns=["tracker", "hash", "complete"])

    print(df.head())
    print(df.describe())
    print(df[df["complete"] == 1])


def _move_by_rename(
    torrents_info: Iterable[qbittorrentapi.AttrDict],
    src: str,
    dest: str,
    regex: bool,
    match_start: bool,
    match_end: bool,
    case_sensitive: bool,
) -> Iterator[Tuple[str, qbittorrentapi.AttrDict]]:
    flags = re.IGNORECASE if not case_sensitive else 0

    if not regex:
        src = re.escape(src)
        dest = dest.replace("\\", "\\\\")

    if match_start:
        src = f"^{src}"

    if match_end:
        src = f"{src}$"

    src_p = re.compile(src, flags)

    for torrent in torrents_info:
        if not torrent.content_path.startswith(torrent.save_path):  # file might already be queued for moving
            logging.error(f"{torrent.content_path} doesn't start with {torrent.save_path}")
            continue

        new, n = src_p.subn(dest, torrent.save_path)
        renamed = n > 0

        if renamed:
            yield new, torrent


def move_by_rename(client: qbittorrentapi.Client, args: Namespace) -> None:
    torrents_info = client.torrents_info()

    for new_location, torrent in _move_by_rename(
        torrents_info, args.src, args.dest, args.regex, args.match_start, args.match_end, args.case_sensitive
    ):
        print(f"Moving `{torrent.name}` from <{torrent.save_path}> to <{new_location}>.")
        if args.do_move:
            client.torrents_set_location(new_location, torrent.hash)


def find_torrents(client: qbittorrentapi.Client, args: Namespace) -> None:
    num_add_try = 0
    num_add_fail = 0

    for torrent_file, match in common._find_torrents(
        args.torrents_dirs, args.data_dirs, args.ignore_top_level_dir, args.follow_symlinks, args.recursive
    ):
        print(f"Found possible match for {torrent_file}: {match}")
        num_add_try += 1
        if args.do_add:

            if args.ignore_top_level_dir:
                content_layout = "NoSubfolder"
            else:
                content_layout = "Original"

            result = client.torrents_add(
                torrent_files=fspath(torrent_file),
                save_path=fspath(match),
                is_skip_checking=False,
                is_paused=False,
                content_layout=content_layout,
            )
            if result == "Fails.":
                logging.error("Failed to add %s", torrent_file)
                num_add_fail += 1

    print(f"Tried to add {num_add_try} torrents, {num_add_fail} failed")


def get_config():
    conf = DEFAULT_CONFIG
    try:
        file_config = read_json(common.config_dir / "config.json")
        conf.update(file_config)
    except FileNotFoundError:
        pass

    return conf


def print_full_help(client: qbittorrentapi.Client, args: Namespace):
    print("\n".join(common.full_help(args.all_parsers)))


def main():
    conf = get_config()

    parser = ArgumentParser()
    parser.add_argument("--host", default=conf.get("host"), help="qBittorrent web interface host and port")
    parser.add_argument("--username", default=conf.get("username"), help="qBittorrent web interface username")
    parser.add_argument("--password", default=conf.get("password"), help="qBittorrent web interface password")
    parser.add_argument("--verbose", action="store_true", help="Show debug output")
    subparsers = parser.add_subparsers(dest="action", required=True)

    parser_a = subparsers.add_parser(
        "categorize-failed-private",
        help="Move private torrents with failed tracker announces to a special category. This can help to find torrents which were removed from a private tracker, or anounce urls with outdated API keys.",
    )
    parser_a.add_argument(
        "--do-move", action="store_true", help="Actually move them, otherwise the moves are only printed"
    )
    parser_a.add_argument(
        "--category-name", default=conf.get("category-name"), help="Name of category to assign torrents to"
    )
    parser_a.set_defaults(func=categorize_private_with_failed_trackers)

    parser_b = subparsers.add_parser("list-paused-private", help="List all paused private torrents")
    parser_b.add_argument("path", type=is_dir, help="Input directory")
    parser_b.set_defaults(func=list_paused_private)

    parser_c = subparsers.add_parser("scrape-loaded", help="Scrape all torrents and output results to file")
    parser_c.add_argument("path", type=is_dir, help="Input directory")
    parser_c.add_argument("--out", default=Path(conf["scrapes-file"]), help="Path to write scraped file info to")
    parser_c.set_defaults(func=scrape_loaded)

    parser_d = subparsers.add_parser(
        "move-by-availability",
        help="Moves torrents around on the harddrive based on number of seeders. NOT FULLY IMPLEMENTED YET!",
    )
    parser_d.add_argument("path", type=is_dir, help="Input directory")
    parser_d.add_argument(
        "--scrapes-file", type=is_file, default=Path(conf["scrapes-file"]), help="Created by the scrape-loaded command."
    )
    parser_d.add_argument(
        "--do-move", action="store_true", help="Actually move them, otherwise the moves are only printed"
    )
    parser_d.set_defaults(func=move_by_availability)

    parser_e = subparsers.add_parser(
        "remove-loaded-torrents", help="Delete torrents files from directory if they are already loaded in qBittorrent"
    )
    parser_e.add_argument("path", nargs="+", type=is_dir, help="Input directory")
    parser_e.add_argument("--do-remove", action="store_true", help="Remove the file from disk")
    parser_e.add_argument(
        "--recursive",
        action="store_true",
        help="Scan for torrent files recursively.",
    )
    parser_e.set_defaults(func=remove_loaded_torrents)

    parser_f = subparsers.add_parser("move-by-rename", help="b help")
    parser_f.add_argument("--src", type=str, help="Source path pattern", required=True)
    parser_f.add_argument("--dest", type=str, help="Destination path pattern", required=True)
    parser_f.add_argument(
        "--do-move", action="store_true", help="Actually move them, otherwise the moves are only printed"
    )
    parser_f.add_argument("--case-sensitive", action="store_true")
    parser_f.add_argument("--regex", action="store_true", help="Use regex for src and dest")
    parser_f.add_argument(
        "--match-start", action="store_true", help="Only match if src is found at the start of the path"
    )
    parser_f.add_argument("--match-end", action="store_true", help="Only match if src is found at the end of the path")
    parser_f.set_defaults(func=move_by_rename)

    parser_g = subparsers.add_parser(
        "find-torrents",
        help="Load torrent files from directory, find the associated files on the harddrive and load the torrents.",
    )
    parser_g.add_argument("--torrents-dirs", nargs="+", type=is_dir, help="Directory with torrent files", required=True)
    parser_g.add_argument(
        "--data-dirs", nargs="+", type=is_dir, help="Directory to look for torrent data", required=True
    )
    parser_g.add_argument(
        "--do-add",
        action="store_true",
        help="Actually add the torrents to qBittorrent, otherwise just print found ones",
    )
    parser_g.add_argument(
        "--ignore-top-level-dir",
        action="store_true",
        help="Ignore the name of the top level dir. This will help to find torrents where no sub-folder was created.",
    )
    parser_g.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinks (and junctions)",
    )
    parser_g.add_argument(
        "--recursive",
        action="store_true",
        help="Scan for torrent files recursively.",
    )
    parser_g.set_defaults(func=find_torrents)

    parser_help = subparsers.add_parser("full-help", help="Show full help, including subparsers")
    parser_help.set_defaults(func=print_full_help)

    args = parser.parse_args()
    args.all_parsers = [parser, parser_a, parser_b, parser_c, parser_d, parser_e, parser_f, parser_g]

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    client = qbittorrentapi.Client(host=args.host, username=args.username, password=args.password)
    args.func(client, args)


if __name__ == "__main__":
    main()
