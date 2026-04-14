#!/bin/bash
# Quick profiling script for DSAC

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export DSAC_PROFILE=1
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/.."

SCENARIO="${1:-discovery_phase2/curvature_invariant.yaml}"
STEPS="${2:-500}"
WORKERS="${3:-9}"

echo "Profiling DSAC..."
echo "Scenario: $SCENARIO"
echo "Steps: $STEPS"
echo "Workers: $WORKERS"
echo ""

python3 -m delta_machine.cli headless \
    --scenario "$SCENARIO" \
    --scenario-dir scenarios \
    --steps "$STEPS" \
    --workers "$WORKERS"

# Print timing stats if available
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    from DSAC_tools.timing_instrumentation import print_timing_stats
    print_timing_stats()
except Exception as e:
    print(f'Could not print timing stats: {e}')
"

