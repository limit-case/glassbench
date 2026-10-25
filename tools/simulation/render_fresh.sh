#!/bin/sh
# Fresh held-out evaluation scenes in thermally limited batches: 10 scenes per batch, 3-minute pause between batches.
B=/Applications/Blender.app/Contents/MacOS/Blender
OUT=synth/fresh
first=1
for spec in "d415 5000 5009" "d415 5010 5019" "d435 5000 5009" "d435 5010 5019" "l515 5000 5009" "l515 5010 5019" "l515 5020 5029" "l515 5030 5039"; do
  set -- $spec
  [ $first -eq 0 ] && { echo "pause 180s"; sleep 180; }
  first=0
  echo "BATCH $1 $2-$3 start $(date +%T)"
  $B -b -P tools/simulation/render_scene.py -- --variant $1 --seeds $(seq $2 $3) --out $OUT 2>&1 | grep -E "^SCENE|Traceback|Error"
  echo "BATCH $1 $2-$3 end $(date +%T)"
done
echo ALLDONE
