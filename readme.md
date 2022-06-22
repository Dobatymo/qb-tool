# qb-tool

## Usage

```
usage: qbtool.py [-h] [--host HOST] [--username USERNAME]
                 [--password PASSWORD] [--verbose]
                 {categorize-failed-private,list-paused-private,scrape-loaded,move-by-availability,remove-loaded-torrents,move-by-rename,find-torrents,full-help}
                 ...

positional arguments:
  {categorize-failed-private,list-paused-private,scrape-loaded,move-by-availability,remove-loaded-torrents,move-by-rename,find-torrents,full-help}
    categorize-failed-private
                        Move private torrents with failed tracker announces to
                        a special category. This can help to find torrents
                        which were removed from a private tracker, or anounce
                        urls with outdated API keys.
    list-paused-private
                        List all paused private torrents
    scrape-loaded       Scrape all torrents and output results to file
    move-by-availability
                        Moves torrents around on the harddrive based on number
                        of seeders. NOT FULLY IMPLEMENTED YET!
    remove-loaded-torrents
                        Delete torrents files from directory if they are
                        already loaded in qBittorrent
    move-by-rename      b help
    find-torrents       Load torrent files from directory, find the associated
                        files on the harddrive and load the torrents.
    full-help           Show full help, including subparsers

optional arguments:
  -h, --help            show this help message and exit
  --host HOST           qBittorrent web interface host and port
  --username USERNAME   qBittorrent web interface username
  --password PASSWORD   qBittorrent web interface password
  --verbose             Show debug output

usage: qbtool.py categorize-failed-private [-h] [--do-move]
                                           [--category-name CATEGORY_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --do-move             Actually move them, otherwise the moves are only
                        printed
  --category-name CATEGORY_NAME
                        Name of category to assign torrents to

usage: qbtool.py list-paused-private [-h] path

positional arguments:
  path        Input directory

optional arguments:
  -h, --help  show this help message and exit

usage: qbtool.py scrape-loaded [-h] [--out OUT] path

positional arguments:
  path        Input directory

optional arguments:
  -h, --help  show this help message and exit
  --out OUT   Path to write scraped file info to

usage: qbtool.py move-by-availability [-h] [--scrapes-file SCRAPES_FILE]
                                      [--do-move]
                                      path

positional arguments:
  path                  Input directory

optional arguments:
  -h, --help            show this help message and exit
  --scrapes-file SCRAPES_FILE
                        Created by the scrape-loaded command.
  --do-move             Actually move them, otherwise the moves are only
                        printed

usage: qbtool.py remove-loaded-torrents [-h] [--do-remove] [--recursive]
                                        path [path ...]

positional arguments:
  path         Input directory

optional arguments:
  -h, --help   show this help message and exit
  --do-remove  Remove the file from disk
  --recursive  Scan for torrent files recursively.

usage: qbtool.py move-by-rename [-h] --src SRC --dest DEST [--do-move]
                                [--case-sensitive] [--regex] [--match-start]
                                [--match-end]

optional arguments:
  -h, --help        show this help message and exit
  --src SRC         Source path pattern
  --dest DEST       Destination path pattern
  --do-move         Actually move them, otherwise the moves are only printed
  --case-sensitive
  --regex           Use regex for src and dest
  --match-start     Only match if src is found at the start of the path
  --match-end       Only match if src is found at the end of the path

usage: qbtool.py find-torrents [-h] --torrents-dirs TORRENTS_DIRS
                               [TORRENTS_DIRS ...] --data-dirs DATA_DIRS
                               [DATA_DIRS ...] [--do-add]
                               [--ignore-top-level-dir] [--follow-symlinks]
                               [--recursive]

optional arguments:
  -h, --help            show this help message and exit
  --torrents-dirs TORRENTS_DIRS [TORRENTS_DIRS ...]
                        Directory with torrent files
  --data-dirs DATA_DIRS [DATA_DIRS ...]
                        Directory to look for torrent data
  --do-add              Actually add the torrents to qBittorrent, otherwise
                        just print found ones
  --ignore-top-level-dir
                        Ignore the name of the top level dir. This will help
                        to find torrents where no sub-folder was created.
  --follow-symlinks     Follow symlinks (and junctions)
  --recursive           Scan for torrent files recursively.

```

## Config

Some of the main arguments like username and password can be put into a config file at `<user_data_dir>/Dobatymo/qb-tool/config.json`. For Windows this defaults to `C:\Users\<user>\AppData\Local\Dobatymo\qb-tool\config.json`

## Usage notes

- `py qbtool.py move-by-rename --src "/movie" --dest "/film"` will match `/movie/`, but also `/movies-old/` and others. So either use `--src /movie/` or add `--match-end`
