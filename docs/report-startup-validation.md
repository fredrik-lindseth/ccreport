# Report startup validation

Measured on 2026-10-28 using a read-only SQLite backup of the local database:
1,194,447 records, 18,000 cached files and 5,503 prepared rollup rows. The
original database and agent-tool logs were not modified.

Each cell is the median of three separate Python processes. The original
comparison alternated before/after order; the revised implementation was
measured afterward against the same frozen corpus. Imports, database reads,
aggregation and rendering to a terminal-like sink are included. Database
copying, live log discovery/parsing and exchange-rate network calls are
excluded. These are local reporting measurements, not an end-to-end guarantee
for another machine or a slow terminal connection.

| Scenario / metric | Before PR (`0d69e9f`) | Initial PR (`c9b387c`) | Revised PR |
| --- | ---: | ---: | ---: |
| Warm cache: first table | 1.081 s | 9.601 s | 0.084 s |
| Warm cache: total | 1.485 s | 14.117 s | 1.375 s |
| Warm cache: peak process memory | 217 MiB | 2,021 MiB | 217 MiB |
| Rollup rebuild: first table | 9.747 s | 9.273 s | 0.087 s |
| Rollup rebuild: total | 12.872 s | 25.613 s | 13.314 s |
| Rollup rebuild: peak process memory | 1,961 MiB | 2,032 MiB | 1,963 MiB |
| Simulated 5 s refresh: first table | 6.004 s | 9.472 s | 0.089 s |
| Simulated 5 s refresh: total | 6.398 s | 19.813 s | 6.424 s |
| Simulated 5 s refresh: peak process memory | 217 MiB | 2,022 MiB | 218 MiB |

“First table” in the PR columns can be a clearly dated saved report. Before
this PR, the first table is the freshly calculated one. The 5 s refresh is an
injected delay, not a measurement of actual log parsing. The rebuild scenario
clears the rollup fingerprint and performs an actual rebuild from cached
records. Full rebuilds remain expensive: this change does not optimize them.

The initial PR loaded the entire cached history to draw its preview, defeating
the normal rollup optimization. The revision saves the final rendering and
shows that small snapshot on the next matching invocation. Its warm-cache
preview phase took 16 ms median. The first status message appeared after
68 ms. With no saved snapshot, a separate first run showed status after
81 ms, its first updated table after 0.960 s, and finished in 1.353 s.

The revised total runtime ranges were 1.366–1.426 s warm, 12.985–13.948 s for
a rebuild and 6.416–6.425 s with the injected delay. The rebuild median was
0.442 s (3.4%) above the baseline median. With three runs and normal system
load variation this does not establish the cause of that difference; there
is no claim that indexing or the final report calculation became faster.
The improvement is useful, labelled information during the wait.

All 28 measured final report bodies (including the first run without a saved
snapshot) had the same SHA-256:
`4b76be4455b1e55fa4590c208957f2d342d31299b251e6e3454a3359cd393987`.
This checks the textual final report on this corpus, in addition to regression
tests for JSON, filters, error handling and source policy.

The costs and limitations are:

- One additional rendering snapshot, 66,312 bytes for this report and limited
  to 1 MiB, plus a small timing file (442 bytes for the measured run).
- Extra terminal output: about 131 kB with a saved preview versus 66 kB before.
  Slow terminal connections can make this more expensive than the local sink.
- A dated snapshot can be stale. It is displayed only for the same report
  options, enabled tools and terminal width. With a different selection or
  no saved snapshot, startup shows feedback and waits for the updated report.
- The saved report contains local report data, including project/account names,
  and is written atomically with mode 0600. It preserves plain text rather
  than the original colours.
- No extra model calls or exchange-rate network requests are introduced.
  A separate 50-run microbenchmark of the timing diagnostics measured about
  9 ms median for 19 atomic diagnostic writes; that isolates the bookkeeping,
  not a complete report run.
