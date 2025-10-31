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

bash ./scripts_self/server_speed_scripts/A1_cronusvla_bridge-4.sh $ckpt_path 0 $((START_PORT+PORTS_PER_GROUP*5)) $((START_PORT+PORTS_PER_GROUP*6-1)) > "${log_path}/log.log" 2>&1 &
pid1=$!
echo "A1_cronusvla_bridge-4: $pid1"

wait $pid1

echo "get...final_result.log"
python ./simpler_env/calc_metrics_evaluation_easy.py --ckpt-mapping $(basename "$ckpt_path") --log-dir-root $result_path  > "${log_path}/final_result.log" 2>&1