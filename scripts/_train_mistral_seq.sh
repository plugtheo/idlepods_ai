#!/bin/bash
set -e
for cap in planning research criticism; do
    echo "==============================="
    echo "Starting ${cap} at $(date)"
    echo "==============================="
    python /app/training/train_gpu_simple.py --capability "${cap}" --fresh "${cap}" --regen-data "${cap}" --validate > /tmp/train_${cap}_v2.log 2>&1
    rc=$?
    echo "Finished ${cap}: exit=${rc}"
    if [ "${rc}" -ne 0 ]; then
        echo "ABORT: ${cap} training failed, stopping sequence"
        exit 1
    fi
done
echo "ALL DONE"
