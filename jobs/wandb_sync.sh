#!/bin/bash
# sync_wandb_runs.sh - Syncs all wandb offline runs in the current directory

echo "🔎 Searching for WandB offline runs..."

# Find all directories that match the wandb offline run pattern
OFFLINE_DIRS=$(find . -path "$SCRATCH/wandb/offline-run-*" -type d)

if [ -z "$OFFLINE_DIRS" ]; then
    echo "❌ No WandB offline runs found."
    exit 0
fi

# Count total runs
TOTAL_RUNS=$(echo "$OFFLINE_DIRS" | wc -l)
echo "🔍 Found $TOTAL_RUNS WandB offline runs."

# Counter for synced runs
SYNCED=0
FAILED=0

echo "🔄 Starting sync process..."
echo "-----------------------------"

# Process each run directory
echo "$OFFLINE_DIRS" | while read -r dir; do
    echo "⏳ Syncing run: $dir"
    if wandb sync "$dir"; then
        SYNCED=$((SYNCED + 1))
        echo "✅ Successfully synced: $dir"
    else
        FAILED=$((FAILED + 1))
        echo "❌ Failed to sync: $dir"
    fi
    echo "-----------------------------"
    echo "Progress: $SYNCED/$TOTAL_RUNS completed"
done

echo "🏁 Sync complete!"
echo "✅ Successfully synced: $SYNCED runs"
echo "❌ Failed to sync: $FAILED runs"