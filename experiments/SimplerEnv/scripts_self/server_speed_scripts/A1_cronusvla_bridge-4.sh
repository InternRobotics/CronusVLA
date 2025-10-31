gpu_id=$2
policy_model=cronusvla
ckpt_path=$1

START_PORT=$3
END_PORT=$4

get_random_port() {
    local port=$((RANDOM % (END_PORT - START_PORT + 1) + START_PORT))
    while lsof -i:$port &>/dev/null; do
        port=$((RANDOM % (END_PORT - START_PORT + 1) + START_PORT))
    done
    echo $port
}

AVAILABLE_PORT=$(get_random_port)
AVAILABLE_HOST='127.0.0.1'

wait_for_server() {
    while ! lsof -i:$AVAILABLE_PORT > /dev/null 2>&1; do
        echo "Waiting for server to start..."
        sleep 5
    done
    echo "Server is ready, starting client..."
}

echo "Using available host: $AVAILABLE_HOST ,available port: $AVAILABLE_PORT"
CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_server.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model cronusvla --ckpt-path $1 --policy-setup widowx_bridge --env-name None &
SERVER_PID=$!
echo "Server started with PID $SERVER_PID"

wait_for_server

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028


CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name PutSpoonOnTableClothInScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;


scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference_client.py --host $AVAILABLE_HOST --port $AVAILABLE_PORT --policy-model ${policy_model} --ckpt-path ${ckpt_path} \
  --robot ${robot} --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
  --env-name PutEggplantInBasketScene-v0 --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x ${robot_init_x} ${robot_init_x} 1 --robot-init-y ${robot_init_y} ${robot_init_y} 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1;


echo "Shutting down server with PID $SERVER_PID..."
kill $SERVER_PID
echo "Server shut down."