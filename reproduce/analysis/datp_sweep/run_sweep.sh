#!/bin/bash
cd "$(dirname "$0")/../../.."
for ds in complextr:baseline_complextr_dir:delta_complextr_fair; do
  IFS=: read name src dst <<< "$ds"
  for w in 0 0.5 1 2 2.83; do
    STATS_OUT=reproduce/analysis/datp_sweep/${name}_wt${w}.json /opt/miniconda3/bin/python reproduce/rebuild_delta_state_v2.py --src reproduce/$src --dst reproduce/$dst --stats-only --temporal-weight $w > /dev/null 2>&1
  done
  for dcy in 0.0005 0.001 0.002 0.01; do
    STATS_OUT=reproduce/analysis/datp_sweep/${name}_decay${dcy}.json /opt/miniconda3/bin/python reproduce/rebuild_delta_state_v2.py --src reproduce/$src --dst reproduce/$dst --stats-only --decay $dcy > /dev/null 2>&1
  done
  STATS_OUT=reproduce/analysis/datp_sweep/${name}_oldspan.json /opt/miniconda3/bin/python reproduce/rebuild_delta_state_v2.py --src reproduce/$src --dst reproduce/$dst --stats-only --no-core-span > /dev/null 2>&1
done
echo SWEEP_DONE
