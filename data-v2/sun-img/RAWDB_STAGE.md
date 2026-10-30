# Local raw snapshots

The central RawDB service is the authority. To refresh the regenerable local
Parquet inputs used by existing builders, run:

```bash
/home/t-lab01/spaceweather-collector/.venv/bin/python /home/t-lab01/sunrun/sun-img/rawdb_stage.py
```

This exports `raw.solar` to `~/.local/state/sundb/stage/ace_solar/year=YYYY/part.parquet`
and `raw.suvi_hourly` to `stage/suvi_hourly/year=YYYY/part.parquet`. It writes each
partition through a temporary file and atomic replacement, with a SHA-256 receipt.
Use `--dataset ace_solar`, `--dataset suvi_hourly`, `--year YYYY`, or `--stage PATH`
to restrict output. Old unselected partitions are retained. The operation has no
Google Drive, PostgreSQL, or other external publishing step, and does not modify
pinned experiment inputs. It is an explicit export, not a collector startup hook.
