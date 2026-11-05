#!/bin/bash
set -u
N="$1"
DIR="/home/t-lab01/sunrun/research/arxiv-bottlenecks-123-20261024"
LOG="$DIR/logs/lane${N}.log"
cd /home/t-lab01/sunrun || exit 1
echo "LANE=${N} START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ) TMUX_PANE=${TMUX_PANE:-unknown} PID=$$" | tee "$LOG"
gjc -p "You are the lane ${N} literature-research worker for the sunrun project. Read ${DIR}/lanes/lane${N}.md and execute that assignment exactly. Read ${DIR}/REQUEST.md first. Use web_search and arXiv full text; verify first-submission dates and IDs. Write only the two files named in the lane assignment (lane report and ACK) under ${DIR}. Do not edit RECEIPT.md, SYNTHESIS.md, papers.csv, other lanes, experiments, protected evaluators, or run new experiments/official evaluations/uploads. This is a literature and validation-design task. End your final message with a single line: ACK lane${N} <status> <paper_count>." >> "$LOG" 2>&1
RC=$?
echo "LANE=${N} EXIT=${RC} END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
sleep 3600
