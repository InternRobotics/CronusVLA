NUM=1
bash ./script/eval_simplerenv/meta_test_SimplerEnv/simpler_0.sh $NUM 0 path/to/cronusvla_7B_bridge_rt_1/checkpoints/step-055000-epoch-04-loss=0.0286.pt & 
pid1=$!
wait $pid1