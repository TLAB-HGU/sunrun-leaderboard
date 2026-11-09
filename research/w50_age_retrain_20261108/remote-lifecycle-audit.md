# Remote training interruption: infrastructure evidence

The original two training processes and detached tmux scheduler disappeared
without normal exit receipts. Read-only diagnostics confirmed repeated Ubuntu
distribution teardown: systemd-journald received SIGTERM from PID1
`systemd-shutdow`, followed by EXT4 unmount/remount. This is an infrastructure
failure, not evidence about the image-age hypothesis or training objective.

WSL repeatedly reported `/sbin/init failed to start within 10000ms` and
`CreateLoginSession ... Timed out waiting for user session for uid=1000`.
Initial loginctl/systemctl queries could not reach the system bus. The host
has `systemd=true` in `/etc/wsl.conf`. The inspected logind configuration only
contained the commented default `KillUserProcesses=no`; a user-session kill
policy was not established. No OOM event was found in the captured kernel-log
tail, which does not prove exhaustive absence.

The leader changed execution to keep the actual scheduler running in the
foreground of the SSH/WSL connection. At remote **22:16:04 KST**, PID1 still
had its **22:14:19** start time; scheduler PID448 and seed42 PID454 had run
since **22:14:48**, and seed43 PID478 since **22:14:50**. dbus and logind were
now running. The journal reported completion of system startup at22:15:40,
after **1min20.809s**, with cloud-init falling back to DataSourceNone.
This snapshot establishes that recovery was alive, not that training finished.

**Timestamp correction:** initial `dmesg --time-format iso` timestamps appeared
to place another teardown after the foreground restart. Fresh current process
IDs/start times contradicted that interpretation, and clock-change messages
were logged. Those reconstructed dmesg wall times must not be used to claim
the foreground recovery died. Current ps and journal evidence take precedence.

The precise trigger of distribution teardown and underlying slow startup
remain unproven. The smallest justified recovery is the active foreground
scheduler connection, then verification of stable PID1/training IDs beyond the
previous failure interval and advancing training artifacts. No host-wide
configuration change is justified while that recovery is progressing.

Relevant code: finite SSH/WSL helper in
`../rerun_df_ffill_20261103/golab2_transport.py:6`, per-run detached tmux launch
in `host_scheduler.py:101`. The earlier campaign uses the same detached
service pattern in `../drop035_v650_confirm_20261108/launch_services.py:21`
through its finite remote call at line27. This identifies the lifecycle
exposure, not proof that all earlier campaigns failed.

Compact observations and exact diagnostic boundaries are preserved in
`remote-lifecycle-evidence.json`. The debugger performed read-only remote
inspection and did not launch, terminate, or reconfigure remote services.
