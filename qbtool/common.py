import logging
from argparse import ArgumentParser
from collections import defaultdict
from operator import itemgetter
from os import fspath
from pathlib import Path
from typing import Any, Collection, Dict, FrozenSet, Iterable, Iterator, List, Optional, Set, Tuple, Union

import requests
from genutility.filesystem import scandir_rec_simple
from genutility.iter import batch, retrier
from genutility.time import TakeAtleast
from genutility.torrent import ParseError, get_torrent_hash, read_torrent_info_dict, scrape
from genutility.tree import SequenceTree
from platformdirs import user_data_dir

APP_NAME = "qb-tool"
AUTHOR = "Dobatymo"
config_dir = Path(user_data_dir(APP_NAME, AUTHOR))


class InversePathTree:
    endkey = "\0"

    def __init__(self) -> None:
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


def recstr(obj: Any) -> str:
    if isinstance(obj, list):
        it: Iterable[str] = map(recstr, obj)
        return f"[{', '.join(it)}]"
    elif isinstance(obj, dict):
        it = (f"{recstr(k)}: {recstr(k)}" for k, v in obj.items())
        return f"{{{', '.join(it)}}}"
    else:
        return str(obj)


def full_help(parsers: Iterable[ArgumentParser]) -> Iterator[str]:
    for parser in parsers:
        yield parser.format_help()


def _iter_torrent_files(dirs: Collection[Path], recursive: bool) -> Iterator[Path]:
    for dir in dirs:
        if recursive:
            globfunc = dir.rglob
        else:
            globfunc = dir.glob
        yield from globfunc("*.torrent")


def _load_torrent_info(path: Path, ignore_top_level_dir: bool) -> List[Dict[str, Any]]:
    info = read_torrent_info_dict(path, normalize_string_fields=True)

    if "files" not in info:
        return [{"path": Path(info["name"]), "size": info["length"]}]
    else:
        if ignore_top_level_dir:
            return [{"path": Path(*file["path"]), "size": file["length"]} for file in info["files"]]
        else:
            return [{"path": Path(info["name"], *file["path"]), "size": file["length"]} for file in info["files"]]


def _build_inverse_tree(basepath: Path, follow_symlinks: bool) -> InversePathTree:
    tree = InversePathTree()
    for entry in scandir_rec_simple(fspath(basepath), dirs=False, follow_symlinks=follow_symlinks):
        tree.add(Path(entry.path), entry.stat().st_size)

    return tree


def _build_size_map(basepath: Path, follow_symlinks: bool) -> Dict[int, Set[Path]]:
    sizemap = defaultdict(set)
    for entry in scandir_rec_simple(fspath(basepath), dirs=False, follow_symlinks=follow_symlinks):
        sizemap[entry.stat().st_size].add(Path(entry.path))

    return sizemap


def check_exists(_meta_matches: Path, info: List[Dict[str, Any]]) -> bool:
    for file in info:
        full_path = _meta_matches / file["path"]
        if not full_path.exists():
            logging.error("File does not exist: <%s>", full_path)
            return False

    return True


def _get_unique_paths(sizemap: Dict[int, Set[Path]], all_sizes: Collection[int]):
    unique_paths = []
    try:
        for size in all_sizes:
            paths = sizemap[size]
            if len(paths) == 1:
                unique_paths.append(paths.pop())
            else:
                return None
    except KeyError:
        return None

    return unique_paths


def _find_torrents(
    torrents_dirs: Collection[Path],
    data_dirs: Collection[Path],
    ignore_top_level_dir: bool = False,
    follow_symlinks: bool = False,
    recursive: bool = True,
    modes: Union[FrozenSet[str], Set[str]] = frozenset(),
) -> Iterator[Tuple[Path, str, Path, Optional[Dict[Path, Path]]]]:
    infos: Dict[Path, Tuple[str, List[Dict[str, Any]]]] = {}
    for file in _iter_torrent_files(torrents_dirs, recursive):
        try:
            # fixme: file is read twice from disk
            hash = get_torrent_hash(file)
            files = _load_torrent_info(file, ignore_top_level_dir)
            infos[file] = (hash, files)
        except TypeError as e:
            logging.error("Skipping %s: %s", file, e)

    if modes - {"paths-and-sizes", "sizes"}:
        raise ValueError(f"Invalid modes: {modes}")

    dirs = ", ".join(f"<{dir}>" for dir in torrents_dirs)
    logging.info("Loaded %s torrents from %s", len(infos), dirs)

    for dir in data_dirs:
        if not modes or "paths-and-sizes" in modes:
            # stage 1: match paths and sizes

            invtree = _build_inverse_tree(dir, follow_symlinks)
            logging.info("Built filesystem tree with %s files from <%s>", len(invtree), dir)

            for torrent_file, (infohash, files_info) in infos.items():
                path_matches = defaultdict(list)
                all_sizes = [_file["size"] for _file in files_info]
                for _file in files_info:
                    for path, size in invtree.find(_file["path"]).items():
                        path_matches[path].append(size)

                meta_matches = []
                partial_matches_sizes = []
                partial_matches_paths = []
                for path, sizes in path_matches.items():
                    if sizes == all_sizes:
                        meta_matches.append(path)
                    elif len(sizes) == len(all_sizes):
                        num_same = sum(1 for s1, s2 in zip(sizes, all_sizes) if s1 == s2)
                        partial_matches_sizes.append((path, num_same))
                    else:
                        partial_matches_paths.append((path, len(sizes)))

                if len(meta_matches) == 0:
                    num_partial_matches = len(partial_matches_sizes) + len(partial_matches_paths)

                    if len(files_info) == 1:
                        logging.debug(
                            "Found path, but no size matches for <%s>: %s", torrent_file, files_info[0]["path"]
                        )
                    elif partial_matches_sizes:
                        best_path, best_num = max(partial_matches_sizes, key=itemgetter(1))
                        logging.info(
                            "Found %d partial matches for <%s>. <%s> matches %d out of %d file sizes and all paths.",
                            num_partial_matches,
                            torrent_file,
                            best_path,
                            best_num,
                            len(all_sizes),
                        )
                    elif partial_matches_paths:
                        best_path, best_num = max(partial_matches_paths, key=itemgetter(1))
                        logging.info(
                            "Found %d partial matches for <%s>. <%s> matches %d out of %d file paths.",
                            num_partial_matches,
                            torrent_file,
                            best_path,
                            best_num,
                            len(all_sizes),
                        )

                elif len(meta_matches) == 1:
                    _meta_matches = meta_matches[0]

                    if not check_exists(_meta_matches, files_info):
                        continue

                    yield torrent_file, infohash, _meta_matches, None

                else:
                    logging.warning(
                        "Found %d possible matches for %s: %s", len(meta_matches), torrent_file, recstr(meta_matches)
                    )

        if not modes or "sizes" in modes:
            # stage 2: match size and hashes (for renames)

            sizemap = _build_size_map(dir, follow_symlinks)
            logging.info("Built filesystem size map with %s files from <%s>", len(sizemap), dir)

            import os.path

            for torrent_file, (infohash, files_info) in infos.items():
                all_sizes = [_file["size"] for _file in files_info]
                unique_paths = _get_unique_paths(sizemap, all_sizes)
                if unique_paths is None:
                    continue
                assert len(all_sizes) == len(unique_paths)

                save_path = os.path.commonpath(p.parent for p in unique_paths)
                renamed_files = {old["path"]: new.relative_to(save_path) for old, new in zip(files_info, unique_paths)}
                yield torrent_file, infohash, save_path, renamed_files


def _scrape_all(tracker_hashes: Dict[str, Set[str]], batchsize: int, delay: float) -> Iterator[Tuple[str, str, Any]]:
    for tracker_url, hashes in tracker_hashes.items():
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
