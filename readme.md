# qb-tool

## usage notes

- `py qbtool.py move-by-rename --src "/movie" --dest "/film"` will match `/movie/`, but also `/movies-old/` and others. So either use `--src /movie/` or add `--match-end`
