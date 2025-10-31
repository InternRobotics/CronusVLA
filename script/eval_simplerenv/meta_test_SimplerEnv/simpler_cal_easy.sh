ckpt_path=$3

if [ -z "$1" ]; then
    NUM=$1
    CUR=$2
else
    NUM=1
    CUR=0
fi

TOTAL_PORTS=15520
PORTS_PER_GROUP=$((TOTAL_PORTS / ($NUM * 8)))
START_PORT=$((50000 + (PORTS_PER_GROUP * $CUR * 8)))


cd ./experiments/SimplerEnv
base_dir=$(dirname "$ckpt_path")
base_dir=$(dirname "$base_dir")
file_name=$(basename "$ckpt_path" .pt)
result_path="${base_dir}/results_${file_name}"

if [ ! -d "$result_path" ]; then
    mkdir -p "$result_path"
fi
log_path="${result_path}/log/"
if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
fi

echo "get...final_result.log"
python ./simpler_env/calc_metrics_evaluation_easy.py --ckpt-mapping $(basename "$ckpt_path") --log-dir-root $result_path  > "${log_path}/final_result.log" 2>&1