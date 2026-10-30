# lane11 source provenance handoff (ACE / SUVI, base111)

Read-only audit; no downloads, no source admission change, no champion input,
no official evaluation, no upload.

- Full report: `research/hi-lane-20261024/HANDOFF_SOURCE_PROVENANCE.md`
- Per-origin JSON (13,521 origins): `research/hi-lane-20261024/source_provenance_base111.json`
- Human-readable summary: `research/hi-lane-20261024/source_provenance_base111.md`
- Tool: `research/hi-lane-20261024/verify_source_provenance.py`
- Tests: `research/hi-lane-20261024/test_verify_source_provenance.py` (24 pass)

## Key results

- Sidecar `sources.json`: 3 file entries pass SHA/receipt; 1 fail
  (`raw-ace-snapshot` 2026 copy 280,498 vs admitted 292,059 bytes, different
  SHA256); 3 directory entries unknown (directory SHA recipe absent).
- ACE: 13,521/13,521 pass the existing `obs_end + 2h` rule; admitted
  `ace-snapshot` year files 5/5 SHA+bytes. Daily-list publication:
  234 `available_at <= origin`, 5,470 late, 7,817 absent (2h-proxy-only).
- SUVI: 13,047/13,521 pass `obs_end <= origin-1h`, `S3 LastModified <= origin`,
  `status=ok`, exact local bytes; 0 fail; 474 unknown (424 2022 no-slot because
  S3 LastModified is 2025; 50 `nc_size` missing). No byte mismatch.

The gap is per-origin publication/byte evidence and the raw snapshot 2026
vintage, not the sidecar file hashes. Missing metadata remains `unknown`; no
default constant or boolean was substituted.
