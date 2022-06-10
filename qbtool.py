import logging
import re
from collections import defaultdict
from os import fspath
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import bencodepy
import qbittorrentapi
import requests
from genutility.args import is_dir
from genutility.iter import batch, retrier
from genutility.json import json_lines
from genutility.time import TakeAtleast
from genutility.torrent import ParseError, get_torrent_hash, read_torrent_info_dict, scrape
from genutility.tree import SequenceTree

DEFAUL_CAT_NAME = "_Unregistered"


def get_private_torrents(client, **kwargs) -> Iterator[Tuple[Any, List[dict]]]:
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


def paused_private(client) -> Iterator[Any]:
    for torrent, trackers in get_private_torrents(client, filter="paused"):
        if torrent.state == "pausedUP":
            yield torrent


def scrape_all(client, batchsize: int = 50, delay: float = 5) -> Iterator[Tuple[str, str, Any]]:
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

    num_remove = 0
    num_keep = 0

    for path in it:
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
                path.unlink()
        else:
            num_keep += 1
            logging.debug("Keep %s <%s>", hash, path)

    print(f"Remove {num_remove} files, keep {num_keep} files")


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


class InversePathTree:

    endkey = "\0"

    def __init__(self):
        self.trie = SequenceTree(endkey=self.endkey)

    def add(self, path: Path, size: int) -> None:
        self.trie[reversed(path.parts)] = size

    def find(self, path: Path) -> Dict[Path, int]:
        try:
            node = self.trie.get_node(reversed(path.parts))
        except KeyError:
            return {}

        paths = {Path(*reversed(parts)): size for parts, size in self.trie.iter_node(node)}

        return paths

    def __len__(self) -> int:
        return self.trie.calc_leaves()


def _build_inverse_tree(basepath: Path) -> InversePathTree:

    tree = InversePathTree()
    for path in basepath.rglob("*"):
        if path.is_file():
            tree.add(path, path.stat().st_size)

    return tree


def _load_torrent_info(path: str) -> Optional[dict]:
    info = read_torrent_info_dict(path)

    if "files" not in info:
        return {"path": info["name"], "size": info["length"]}

    return None


def find_torrents(client, args) -> None:
    infos = {}
    for file in args.torrents_dir.rglob("*.torrent"):
        info = _load_torrent_info(file)
        if info is None:
            continue

        infos[fspath(file)] = info

    logging.info("Loaded %s torrents from <%s>", len(infos), args.torrents_dir)

    for dir in args.data_dirs:
        invtree = _build_inverse_tree(dir)
        num_add_try = 0
        num_add_fail = 0
        logging.info("Built filesystem tree with %s files from <%s>", len(invtree), dir)

        # stage 1: match paths and sizes

        for torrent_file, info in infos.items():
            single_file = Path(info["path"])
            path_matches = invtree.find(single_file)
            if not path_matches:
                continue
            meta_matches = []
            for path, size in path_matches.items():
                if size == info["size"]:
                    meta_matches.append(path)

            if len(meta_matches) == 0:
                logging.debug("Found path, but no size matches for %s", info)
            elif len(meta_matches) == 1:
                full_path = meta_matches[0] / single_file
                assert full_path.exists()
                print(f"Found possible match for {torrent_file}: {full_path}")
                if args.do_add:
                    result = client.torrents_add(
                        torrent_files=torrent_file,
                        save_path=fspath(meta_matches[0]),
                        is_skip_checking=False,
                        is_paused=False,
                    )
                    num_add_try += 1
                    if result == "Fails.":
                        logging.error("Failed to add %s", torrent_file)
                        num_add_fail += 1
            else:
                logging.warning("Found multiple possible matches for %s: %s", torrent_file, meta_matches)

        print(f"Tried to add {num_add_try} torrents, {num_add_fail} failed")

        # stage 2: match size and hashes (for renames)


if __name__ == "__main__":
    from argparse import ArgumentParser

    DEFAULT_USERNAME = "admin"
    DEFAULT_PASSWORD = "password"

    parser = ArgumentParser()
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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
        "find-torrents", help="Delete torrents files from directory if they are already loaded in qBittorrent"
    )
    parser_g.add_argument("--torrents-dir", type=is_dir, help="Directory with torrent files", required=True)
    parser_g.add_argument(
        "--data-dirs", nargs="+", type=is_dir, help="Directory to look for torrent data", required=True
    )
    parser_g.add_argument(
        "--do-add",
        action="store_true",
        help="Actually add the torrents to qBittorrent, otherwise just print found ones",
    )
    parser_g.set_defaults(func=find_torrents)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
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
