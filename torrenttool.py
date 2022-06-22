import logging
import shutil
from argparse import ArgumentParser, Namespace
from itertools import chain
from os import fspath
from pathlib import Path
from urllib.parse import urlsplit

from genutility.args import is_dir
from genutility.torrent import read_torrent

import common


def move_torrent_files_by_tracker(args: Namespace) -> None:

    if args.do_move and not args.out_dir:
        raise RuntimeError("Didn't provide output directory")

    if args.do_move:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for file in common._iter_torrent_files(args.torrents_dirs, args.recursive):
        d = read_torrent(fspath(file))

        announces = set()
        try:
            announces.add(d["announce"])
        except KeyError:
            pass
        announces.update(chain.from_iterable(d.get("announce-list", [])))

        for announce in announces:
            sr = urlsplit(announce)
            found = any(hostname.lower() in sr.hostname.lower() for hostname in args.tracker_hostnames)
            if found:
                print(f"Found <{file}> with tracker <{sr.hostname}>")
                if args.do_move:
                    outpath = args.out_dir / file.name
                    if outpath.exists():
                        logging.warning("Skipping <%s>. File already exists.", outpath)
                        continue
                    shutil.move(fspath(file), outpath)
                break


def print_full_help(args: Namespace):
    print("\n".join(common.full_help(args.all_parsers)))


def main():

    parser = ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Show debug output")
    subparsers = parser.add_subparsers(dest="action", required=True)

    parser_a = subparsers.add_parser(
        "move-torrent-files-by-tracker", help="Scan the directories for torrent files from certain trackers."
    )
    parser_a.add_argument("--torrents-dirs", nargs="+", type=is_dir, help="Directory with torrent files", required=True)
    parser_a.add_argument("--tracker-hostnames", nargs="+", type=str, help="tracker domain to check for", required=True)
    parser_a.add_argument("--do-move", action="store_true", help="Move the files from disk")
    parser_a.add_argument("--out-dir", type=Path, help="Directory to move files to")
    parser_a.add_argument(
        "--recursive",
        action="store_true",
        help="Scan for torrent files recursively.",
    )
    parser_a.set_defaults(func=move_torrent_files_by_tracker)

    parser_help = subparsers.add_parser("full-help", help="Show full help, including subparsers")
    parser_help.set_defaults(func=print_full_help)

    args = parser.parse_args()
    args.all_parsers = [parser, parser_a]

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    args.func(args)


if __name__ == "__main__":
    main()
