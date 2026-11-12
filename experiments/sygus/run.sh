#!/bin/bash
# The SyGuS comparison through the Try page's own tools: generate-csv.ts (expression.ts) makes every episode and scores
# every answer file; Teenie answers the files; cvc5 gets PBE problems built from the same files.
#
#   experiments/sygus/run.sh fetch        the Hacker's Delight files from SyGuS-Org/benchmarks at 13c8deb
#   experiments/sygus/run.sh episodes     write the episodes and check them against the SMT-LIB references
#   experiments/sygus/run.sh teenie-hd    Teenie answers the Hacker's Delight episodes, generate-csv scores them
#   experiments/sygus/run.sh cvc5-deep    cvc5 on the deep-carry PBE problems (background-friendly; 300 s each)
#   experiments/sygus/run.sh teenie-deep  Teenie on the deep problems cvc5 timed out on, generate-csv scores them
#   experiments/sygus/run.sh tabpfn-hd | tabpfn-deep    the same with TabPFN (GPU; needs TABPFN_TOKEN or ~/.tabpfn_token)
#
# GEN may point at another copy of generate-csv.ts (default: site/scripts/generate-csv.ts).
set -euo pipefail
cd "$(dirname "$0")/../.."
GEN=${GEN:-site/scripts/generate-csv.ts}
E=experiments/sygus
mkdir -p $E/episodes/hd $E/episodes/deep $E/checks

formula() { uv run python -c "import json,sys; print(next(r['formula'] for r in json.load(open('$E/rules.json')) if r['name']==sys.argv[1]))" "$1"; }

case ${1:-} in
fetch)
  mkdir -p $E/benchmarks/hd
  for n in 01-d1 01-d5 02-d0 02-d1 02-d5 03-d0 03-d1 03-d5 04-d0 04-d1 04-d5 05-d0 05-d1 05-d5 06-d0 06-d1 06-d5 \
           07-d0 07-d1 07-d5 08-d0 08-d1 08-d5 09-d0 09-d1 09-d5 13-d0 13-d1 13-d5 14-d0 14-d1 14-d5 15-d0 15-d1 \
           15-d5 17-d0 17-d1 17-d5 19-d0 19-d1 19-d5 20-d0 20-d1 20-d5; do
    curl -sfL "https://raw.githubusercontent.com/SyGuS-Org/benchmarks/13c8deb68a873635879c9a69bc78caebd340f646/lib/General_Track/from_2018/hd-$n-prog.sl" \
      -o "$E/benchmarks/hd/hd-$n-prog.sl"
  done
  ls $E/benchmarks/hd | wc -l
  ;;
cvc5-hd)
  mkdir -p $E/cvc5-hd && rm -f $E/cvc5-hd/results.tsv
  ls $E/benchmarks/hd/*.sl | xargs -P 4 -I{} $E/run_cvc5.sh {} $E/cvc5-hd 300
  uv run python -m teenie.bits.sygus check-cvc5
  ;;
episodes)
  uv run python -m teenie.bits.sygus rules > /dev/null
  uv run python -c "import json; [print(r['set'], r['name'], r['formula'], sep='\t') for r in json.load(open('$E/rules.json')) if 'formula' in r]" |
  while IFS=$'\t' read -r set name f; do
    if [ "$set" = hd ]; then seeds="0 1 2 3 4"; else seeds="0"; fi
    for s in $seeds; do
      bun "$GEN" "$f" --seed "$(echo -n "$name-s$s" | cksum | cut -d' ' -f1)" --questions 1000 \
        -o "$E/episodes/$set/$name-s$s.csv" 2>/dev/null
    done
  done
  uv run python -m teenie.bits.sygus verify $E/episodes/hd/*.csv $E/episodes/deep/*.csv | grep -v ' 0 of ' || echo "all episodes match the SMT-LIB references"
  ;;
teenie-hd | teenie-deep | tabpfn-hd | tabpfn-deep)
  set_name=${1#*-}
  method=${1%-*}
  if [ "$method" = teenie ]; then kind=answers; suffix=""; else kind=tabpfn; suffix="-tabpfn"; fi
  if [ "$set_name" = hd ]; then files=$(ls $E/episodes/hd/*-s?.csv); else
    files=$(awk -F'\t' '$2=="timeout"{print "'$E'/episodes/deep/" $1 ".csv"}' $E/cvc5-deep/results.tsv | sort); fi
  if [ "$method" = teenie ]; then uv run python -m teenie.bits.sygus answer $files
  else uv run --extra tabpfn python -m teenie.bits.sygus tabpfn $files; fi
  for f in $files; do
    stem=$(basename "$f" .csv)
    bun "$GEN" "$(formula "${stem%-s*}")" --check "${f%.csv}-$kind.csv" --json > "$E/checks/$stem$suffix.json"
  done
  echo "scored $(echo $files | wc -w) files into $E/checks/"
  ;;
cvc5-deep)
  uv run python -m teenie.bits.sygus pbe $E/episodes/deep/*-s0.csv --folder $E/deep
  mkdir -p $E/cvc5-deep && rm -f $E/cvc5-deep/results.tsv
  ls $E/deep/*.sl | xargs -P 4 -I{} $E/run_cvc5.sh {} $E/cvc5-deep 300
  awk -F'\t' '{c[$2]++} END {for (k in c) print k, c[k]}' $E/cvc5-deep/results.tsv
  ;;
*) sed -n '2,13p' "$0"; exit 1 ;;
esac
