#!/bin/bash
# Chunked NPU full-dataset evaluation with per-chunk result persistence.
# Each chunk runs format_only (saves results_nusc.json), so a crash in one
# chunk loses only that chunk. After all chunks, merge + NuScenesEval.
#
# Usage (detached):  setsid bash run_npu_test_chunks.sh 3 >/tmp/npu_test_chunks/driver.log 2>&1
# Monitor (after):   grep -E "=== Chunk|exit:|Merged|FAILED" /tmp/npu_test_chunks/driver.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"
CONFIG="$ROOT/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py"
CKPT="$SCRIPT_DIR/models/bevfusion_official.pth"
TEST_PY="$SCRIPT_DIR/npu_test.py"
WORK=/tmp/npu_test_chunks
NCHUNKS=${1:-3}
TOTAL=${2:-6019}

mkdir -p "$WORK"
CHUNK_SIZE=$(( (TOTAL + NCHUNKS - 1) / NCHUNKS ))

for ((c=0; c<NCHUNKS; c++)); do
    START=$(( c * CHUNK_SIZE ))
    END=$(( START + CHUNK_SIZE ))
    if (( START >= TOTAL )); then break; fi
    if (( END > TOTAL )); then END=$TOTAL; fi
    N=$(( END - START ))
    PREFIX="$WORK/chunk${c}"
    LOG="$WORK/chunk${c}.log"
    echo "=== Chunk $c: frames [$START, $END) count=$N prefix=$PREFIX ==="
    rm -f /dev/shm/psm_* 2>/dev/null
    python "$TEST_PY" "$CONFIG" "$CKPT" \
        --work-dir "$WORK/chunk${c}_work" \
        --start-idx "$START" --max-frames "$N" \
        --format-only --jsonfile-prefix "$PREFIX" \
        > "$LOG" 2>&1
    rc=$?
    echo "Chunk $c exit: $rc"
    if [ $rc -ne 0 ]; then
        echo "Chunk $c FAILED - inspect $LOG"
        exit $rc
    fi
    ls -la "$PREFIX/pred_instances_3d/" 2>/dev/null | grep results_nusc
done

echo "=== All chunks done. Merging results ==="
python3 - "$WORK" <<'HEREDOC_PY'
import json, glob, os, sys
work = sys.argv[1]
merged = None
for f in sorted(glob.glob(f'{work}/chunk*/pred_instances_3d/results_nusc.json')):
    with open(f) as fp:
        d = json.load(fp)
    if merged is None:
        merged = {'meta': d.get('meta', {}), 'results': {}}
    merged['results'].update(d['results'])
    print(f'  {f}: {len(d["results"])} samples')
out = f'{work}/results_nusc_merged.json'
if merged is not None:
    with open(out, 'w') as fp:
        json.dump(merged, fp)
    print(f'Merged {len(merged["results"])} samples -> {out}')
else:
    print('ERROR: no chunk results found')
    sys.exit(1)
HEREDOC_PY
echo "=== Driver complete ==="