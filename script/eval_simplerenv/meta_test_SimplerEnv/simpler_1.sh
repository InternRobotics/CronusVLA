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

bash ./scripts_self/server_speed_scripts/1_cronusvla_drawer_variant_agg-12.sh $ckpt_path 0 $((START_PORT+PORTS_PER_GROUP*0)) $((START_PORT+PORTS_PER_GROUP*1-1)) > "${log_path}/log1.log" 2>&1 &
pid1=$!
echo "1_cronusvla_drawer_variant_agg-12: $pid1"

bash ./scripts_self/server_speed_scripts/1_cronusvla_drawer_visual_matching-12.sh $ckpt_path 1 $((START_PORT+PORTS_PER_GROUP*1)) $((START_PORT+PORTS_PER_GROUP*2-1)) > "${log_path}/log2.log" 2>&1 &
pid2=$!
echo "1_cronusvla_drawer_visual_matching-12: $pid2"

bash ./scripts_self/server_speed_scripts/1_cronusvla_pick_coke_can_variant_agg-12.sh $ckpt_path 2 $((START_PORT+PORTS_PER_GROUP*2)) $((START_PORT+PORTS_PER_GROUP*3-1)) > "${log_path}/log3.log" 2>&1 &
pid3=$!
echo "1_cronusvla_pick_coke_can_variant_agg-12: $pid3"

bash ./scripts_self/server_speed_scripts/1_cronusvla_put_in_drawer_variant_agg-13.sh $ckpt_path 3 $((START_PORT+PORTS_PER_GROUP*3)) $((START_PORT+PORTS_PER_GROUP*4-1)) > "${log_path}/log4.log" 2>&1 &
pid4=$!
echo "1_cronusvla_put_in_drawer_variant_agg-13: $pid4"

bash ./scripts_self/server_speed_scripts/2_cronusvla_drawer_variant_agg-4A1_cronusvla_pick_coke_can_visual_matching-8.sh $ckpt_path 4 $((START_PORT+PORTS_PER_GROUP*4)) $((START_PORT+PORTS_PER_GROUP*5-1)) > "${log_path}/log5.log" 2>&1 &
pid5=$!
echo "2_cronusvla_drawer_variant_agg-4A1_cronusvla_pick_coke_can_visual_matching-8: $pid5"

bash ./scripts_self/server_speed_scripts/2_cronusvla_drawer_visual_matching-12A1_cronusvla_bridge-4.sh $ckpt_path 5 $((START_PORT+PORTS_PER_GROUP*5)) $((START_PORT+PORTS_PER_GROUP*6-1)) > "${log_path}/log6.log" 2>&1 &
pid6=$!
echo "2_cronusvla_drawer_visual_matching-12A1_cronusvla_bridge-4: $pid6"

bash ./scripts_self/server_speed_scripts/2_cronusvla_pick_coke_can_variant_agg-2A1_cronusvla_move_near_visual_matching-4A1_cronusvla_move_near_variant_agg-7.sh $ckpt_path 6 $((START_PORT+PORTS_PER_GROUP*6)) $((START_PORT+PORTS_PER_GROUP*7-1)) > "${log_path}/log7.log" 2>&1 &
pid7=$!
echo "2_cronusvla_pick_coke_can_variant_agg-2A1_cronusvla_move_near_visual_matching-4A1_cronusvla_move_near_variant_agg-7: $pid7"

bash ./scripts_self/server_speed_scripts/2_cronusvla_put_in_drawer_variant_agg-2A1_cronusvla_put_in_drawer_visual_matching-8.sh $ckpt_path 7 $((START_PORT+PORTS_PER_GROUP*7)) $((START_PORT+PORTS_PER_GROUP*8-1)) > "${log_path}/log8.log" 2>&1 &
pid8=$!
echo "2_cronusvla_put_in_drawer_variant_agg-2A1_cronusvla_put_in_drawer_visual_matching-8: $pid8"


wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6 $pid7 $pid8

echo "get...final_result.log"
python ./simpler_env/calc_metrics_evaluation_easy.py --ckpt-mapping $(basename "$ckpt_path") --log-dir-root $result_path  > "${log_path}/final_result.log" 2>&1