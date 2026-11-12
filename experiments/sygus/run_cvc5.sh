#!/bin/bash
# Run cvc5 (SyGuS, default options) on one .sl file with a wall-clock timeout; one line of tab-separated results.
# usage: run_cvc5.sh <file.sl> <outdir> [timeout seconds]
f=$1; out=$2; limit=${3:-300}
name=$(basename "$f" .sl)
ulimit -v "${CVC5_MEMORY_KB:-3000000}"  # unlimited to lift the cap
start=$(date +%s%N)
timeout "$limit" /old-arch/home/ash/Projects/teenie/tools/cvc5-Linux-x86_64-static/bin/cvc5 --lang=sygus2 "$f" > "$out/$name.out" 2> "$out/$name.err"
code=$?
end=$(date +%s%N)
if [ $code -eq 124 ]; then status=timeout; elif [ $code -eq 0 ] && grep -q 'define-fun' "$out/$name.out"; then status=solved; else status="error($code)"; fi
printf "%s\t%s\t%s\n" "$name" "$status" "$(( (end - start) / 1000000 ))" >> "$out/results.tsv"  # milliseconds
