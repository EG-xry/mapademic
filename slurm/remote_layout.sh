#!/bin/bash
# Run the GPU layout for one release on a Slurm cluster and fetch coords.parquet.
#
#   bash slurm/remote_layout.sh data/2026-09-23 [ssh-host] [remote-dir]
#
# Run from the repo root. Uploads <release>/graph/ and this checkout's
# slurm/run_layout.py, submits slurm/layout-expanse.sbatch with
# DATA=<remote-dir>, and copies coords.parquet back into <release>/.
# graph/init_pos.parquet (from `pipeline warmstart`) is passed as
# --init-pos when present. ITERS, CHUNK and LAYOUT_ARGS are forwarded to the
# job (defaults 1000, 100, none).
#
# A job that ends on its wall-clock limit is resubmitted (run_layout.py
# resumes from layout_ckpt/); any other failure stops the script with the
# log's tail, since rerunning a deterministic failure only burns allocation.
# One SSH master connection is reused, so password/Duo is asked for once.
# Safe to re-run: a job already queued or running for this release (job name
# mapademic-layout-<release>) is waited on, never doubled.
set -euo pipefail

LOCAL=${1:?usage: remote_layout.sh <local release dir> [ssh host] [remote dir]}
HOST=${2:-expanse}
REL=$(basename "$LOCAL")
REMOTE=${3:-/expanse/lustre/scratch/egao2/temp_project/mapademic-$REL}
JOBNAME="mapademic-layout-$REL"
MAX_SUBMITS=${MAX_SUBMITS:-5}
POLL=${POLL:-120}
ITERS=${ITERS:-1000}
CHUNK=${CHUNK:-100}
LAYOUT_ARGS=${LAYOUT_ARGS:-}

[ -f "$LOCAL/graph/edges_int32.parquet" ] || { echo "no $LOCAL/graph; run prep first" >&2; exit 1; }
[ -f slurm/run_layout.py ] || { echo "run this from the mapademic repo root" >&2; exit 1; }
if [ -f "$LOCAL/graph/init_pos.parquet" ]; then
    LAYOUT_ARGS="$LAYOUT_ARGS --init-pos $REMOTE/graph/init_pos.parquet"
fi

SOCK="$HOME/.ssh/mapademic-cm-%r@%h:%p"
SSH=(ssh -o ControlMaster=auto -o "ControlPath=$SOCK" -o ControlPersist=12h
     -o ServerAliveInterval=60 -o ServerAliveCountMax=10)
# Close the master on exit only if this run opened it: a connection someone
# opened by hand (and other runs may be sharing) is left alone.
OWN_MASTER=1
if "${SSH[@]}" -O check "$HOST" 2>/dev/null; then OWN_MASTER=0; fi
trap 'if [ "$OWN_MASTER" = 1 ]; then "${SSH[@]}" -O exit "$HOST" 2>/dev/null || true; fi' EXIT
remote() { "${SSH[@]}" "$HOST" "$@"; }

# Prints the id of this release's queued/running job, or nothing; fails (rc 1)
# only when the cluster could not be asked, so a dropped connection is never
# mistaken for "the job has finished".
active_job() {
    local out
    out=$(remote "squeue -h -u \$USER -n '$JOBNAME' -o %i") || return 1
    printf '%s\n' "$out" | head -n 1
}

# Final Slurm state of a job (COMPLETED, TIMEOUT, FAILED, ...); sacct can lag
# the queue by a few seconds, so ask until it reports a finished state.
final_state() {
    local st=""
    for _ in 1 2 3 4 5 6; do
        st=$(remote "sacct -j $1 -n -X -o State%20" | awk 'NF{print $1; exit}') || st=""
        case "$st" in ""|PENDING|RUNNING|COMPLETING|REQUEUED) sleep 10 ;; *) break ;; esac
    done
    printf '%s\n' "$st"
}

remote "mkdir -p '$REMOTE/graph' \"\$HOME/mapademic/slurm\""
rsync -a --info=progress2 -e "${SSH[*]}" "$LOCAL/graph/" "$HOST:$REMOTE/graph/"
rsync -a -e "${SSH[*]}" slurm/run_layout.py slurm/layout-expanse.sbatch "$HOST:mapademic/slurm/"

submits=0
while ! remote "test -f '$REMOTE/coords.parquet'"; do
    job=$(active_job)
    if [ -z "$job" ]; then
        if [ "$submits" -ge "$MAX_SUBMITS" ]; then
            echo "coords.parquet not produced after $MAX_SUBMITS submissions;" \
                 "see $JOBNAME-*.log in ~/mapademic on $HOST" >&2
            exit 1
        fi
        submits=$((submits + 1))
        job=$(remote "cd \"\$HOME/mapademic\" && DATA='$REMOTE' ITERS='$ITERS' CHUNK='$CHUNK'" \
                     "LAYOUT_ARGS='$LAYOUT_ARGS' sbatch --parsable -J '$JOBNAME' slurm/layout-expanse.sbatch")
        echo "submitted job $job (submission $submits/$MAX_SUBMITS; iters $ITERS;$LAYOUT_ARGS)"
    else
        echo "waiting on job $job, already queued or running for $REL"
    fi
    while true; do
        sleep "$POLL"
        if ! state=$(active_job); then
            echo "$(date +%H:%M) cluster unreachable; retrying" >&2
            continue
        fi
        [ -n "$state" ] || break
        remote "tail -n 1 \"\$HOME/mapademic/$JOBNAME-$job.log\" 2>/dev/null" || true
    done
    st=$(final_state "$job")
    echo "job $job ended: ${st:-unknown}"
    case "$st" in
        COMPLETED|TIMEOUT|PREEMPTED|NODE_FAIL) ;;
        *)
            remote "tail -n 25 \"\$HOME/mapademic/$JOBNAME-$job.log\" 2>/dev/null" || true
            echo "stopping: job $job ended $st, which a resubmission would repeat" >&2
            exit 1 ;;
    esac
done

rsync -a --info=progress2 -e "${SSH[*]}" "$HOST:$REMOTE/coords.parquet" "$LOCAL/"
echo "layout done -> $LOCAL/coords.parquet"
