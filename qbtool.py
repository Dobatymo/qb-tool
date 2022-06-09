import logging
import re
from collections import defaultdict
from os import fspath
from typing import Any, Dict, Iterator

import bencodepy
import qbittorrentapi
import requests
from genutility.args import is_dir
from genutility.iter import batch, retrier
from genutility.json import json_lines
from genutility.time import TakeAtleast
from genutility.torrent import ParseError, get_torrent_hash, scrape

DEFAUL_CAT_NAME = "_Unregistered"


def get_private_torrents(client, **kwargs):
    for torrent in client.torrents_info(**kwargs):
        trackers = client.torrents_trackers(torrent.hash)
        assert trackers[0]["url"] == "** [DHT] **"
        assert trackers[1]["url"] == "** [PeX] **"
        assert trackers[2]["url"] == "** [LSD] **"
        is_private = trackers[0]["status"] == 0

        if is_private:
            yield torrent, trackers[3:]


def get_bad_torrents(client) -> Iterator[Dict[str, Any]]:
    for torrent, trackers in get_private_torrents(client, filter="seeding"):
        for tracker in trackers:
            if tracker["status"] == 4:
                yield {
                    "hash": torrent.hash,
                    "name": torrent.name,
                    "msg": tracker["msg"],
                }


def categorize_private_with_failed_trackers(client, cat_name: str = DEFAUL_CAT_NAME):
    new = {t["hash"] for t in get_bad_torrents(client)}
    old = {t["hash"] for t in client.torrents_info(category=cat_name)}

    add = new - old
    remove = old - new

    try:
        client.torrents_create_category(cat_name)
    except qbittorrentapi.exceptions.Conflict409Error:
        pass  # ignore existing categories

    client.torrents_set_category(cat_name, add)
    client.torrents_set_category("", remove)

    return len(new), len(add), len(remove)


def paused_private(client):
    for torrent, trackers in get_private_torrents(client, filter="paused"):
        if torrent.state == "pausedUP":
            yield torrent


def scrape_all(client, batchsize: int = 50, delay: float = 5):
    """Larger batch sizes than 50 lead to issues with many trackers"""

    all = defaultdict(set)
    for torrent, trackers in get_private_torrents(client):
        all[torrent["tracker"]].add(torrent.hash)

    for tracker_url, hashes in all.items():

        if not tracker_url:
            logging.warning("Could not find working tracker for: %s", hashes)
            continue

        try:
            for hash_batch in batch(hashes, batchsize, list):
                for _ in retrier(60, attempts=3):
                    try:
                        with TakeAtleast(delay):
                            for hash, info in scrape(tracker_url, hash_batch).items():
                                yield tracker_url, hash, info
                        break
                    except requests.ConnectionError as e:
                        logging.warning("%s (%s)", e, tracker_url)
                else:
                    logging.error("Skipping remaining hashes for %s", tracker_url)
                    break
        except ValueError as e:
            logging.warning("%s (%s)", e, tracker_url)
        except ParseError as e:
            logging.error("%s (%s): %s", e, tracker_url, e.data)


filtermap = {
    "paused": ("pausedDL", "pausedUP"),
    "completed": ("uploading", "stalledUP", "pausedUP", "forcedUP"),
    "errored": {"missingFiles"},
    "stalled": {"stalledUP", "stalledDL"},
    "downloading": {"downloading", "pausedDL", "forcedDL"},
}


def torrent_exists(client, hash: str) -> bool:

    try:
        client.torrents_properties(hash)
        return True
    except qbittorrentapi.NotFound404Error:
        return False


def remove_loaded_torrents(client, args) -> None:

    if args.recursive:
        it = args.path.rglob("*.torrent")
    else:
        it = args.path.glob("*.torrent")

    for path in it:
        try:
            hash = get_torrent_hash(fspath(path))
        except bencodepy.exceptions.BencodeDecodeError:
            logging.error("Failed to read torrent file: <%s>", path)
            continue

        # hash = bytes.fromhex(hash)
        exists = torrent_exists(client, hash)
        if args.do_remove:
            if exists:
                logging.info("Removed %s <%s>", hash, path)
                path.unlink()
            else:
                logging.info("Keep %s <%s>", hash, path)
        else:
            logging.info(hash, exists, path)


def categorize_failed_private(client, args) -> None:
    total, add, remove = categorize_private_with_failed_trackers(client, args.category_name)
    print(total, add, remove)


def list_paused_private(client, args) -> None:
    torrents = list(paused_private(client))
    for torrent in torrents:
        print(torrent.hash, torrent.name)


def scrape_loaded(client, args) -> None:
    with json_lines.from_path(args.out, "wt") as fw:
        for obj in scrape_all(client):
            fw.write(obj)


def move_by_availability(client, args) -> None:

    import pandas as pd

    tuples = []
    with json_lines.from_path("scrapes.jl", "rt") as jl:
        for tracker_url, hash, info in jl:
            try:
                tuples.append((tracker_url, hash, info["complete"]))
            except KeyError:
                logging.warning("num_complete not found for %s, %s: %s", tracker_url, hash, info)

    df = pd.DataFrame.from_records(tuples, columns=["tracker", "hash", "complete"])

    print(df.head())
    print(df.describe())
    print(df[df["complete"] == 1])


def move_by_rename(client, args) -> None:

    flags = re.IGNORECASE if not args.case_sensitive else 0

    if args.regex:
        src = args.src
        dest = args.dest
    else:
        src = re.escape(args.src)
        dest = args.dest.replace("\\", "\\\\")

    if args.match_start:
        src = f"^{src}"

    if args.match_end:
        src = f"{src}$"

    src = re.compile(src, flags)

    for torrent in client.torrents_info():
        assert torrent.content_path.startswith(torrent.save_path)

        new, n = src.subn(dest, torrent.save_path)
        renamed = n > 0

        if renamed:
            print(f"Moving `{torrent.name}` from <{torrent.save_path}> to <{new}>.")
            if args.do_move:
                client.torrents_set_location(new, torrent.hash)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path

    DEFAULT_USERNAME = "admin"
    DEFAULT_PASSWORD = "password"

    parser = ArgumentParser()
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--recursive", action="store_true")
    subparsers = parser.add_subparsers(dest="action", required=True)

    parser_a = subparsers.add_parser(
        "categorize-failed-private", help="Move private torrents with failed tracker announces to a special category"
    )
    parser_a.add_argument("--category-name", default=DEFAUL_CAT_NAME, help="Name of category to assign torrents to")
    parser_a.set_defaults(func=categorize_failed_private)

    parser_b = subparsers.add_parser("list-paused-private", help="List all paused private torrents")
    parser_b.add_argument("path", type=is_dir, help="Input directory")
    parser_b.set_defaults(func=list_paused_private)

    parser_c = subparsers.add_parser("scrape-loaded", help="Scrape all torrents and output results to file")
    parser_c.add_argument("path", type=is_dir, help="Input directory")
    parser_c.add_argument("--out", default=Path("out.jl"))
    parser_c.set_defaults(func=scrape_loaded)

    parser_d = subparsers.add_parser(
        "move-by-availability", help="Moves torrents around on the harddrive based on number of seeders"
    )
    parser_d.add_argument("path", type=is_dir, help="Input directory")
    parser_d.add_argument(
        "--do-move", action="store_true", help="Actually move them, otherwise the moves are only printed"
    )
    parser_d.set_defaults(func=move_by_availability)

    parser_e = subparsers.add_parser(
        "remove-loaded-torrents", help="Delete torrents files from directory if they are already loaded in qBittorrent"
    )
    parser_e.add_argument("path", type=is_dir, help="Input directory")
    parser_e.add_argument("--do-remove", action="store_true")
    parser_e.set_defaults(func=remove_loaded_torrents)

    parser_e = subparsers.add_parser("move-by-rename", help="b help")
    parser_e.add_argument("--src", type=str, help="Source path pattern", required=True)
    parser_e.add_argument("--dest", type=str, help="Destination path pattern", required=True)
    parser_e.add_argument(
        "--do-move", action="store_true", help="Actually move them, otherwise the moves are only printed"
    )
    parser_e.add_argument("--case-sensitive", action="store_true")
    parser_e.add_argument("--regex", action="store_true", help="Use regex for src and dest")
    parser_e.add_argument(
        "--match-start", action="store_true", help="Only match if src is found at the start of the path"
    )
    parser_e.add_argument("--match-end", action="store_true", help="Only match if src is found at the end of the path")
    parser_e.set_defaults(func=move_by_rename)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    client = qbittorrentapi.Client(host="localhost:8080", username=args.username, password=args.password)
    args.func(client, args)

    """
    client.torrents_reannounce(torrent_hashes)
    client.sync_torrent_peers(hash)
    client.torrents_recheck(torrent_hashes)

    torrents_set_location(location=None, torrent_hashes=None


    for torrent in qbt_client.torrents_info(filter="seeding", sort="num_complete", limit=3):
        print(f'{torrent.hash[-6:]}: {torrent.name} {torrent.num_complete} ({torrent.state})')
    """
