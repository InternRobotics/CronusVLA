NUM=2 # total number of ckpts

# paremeters-> total num; current id; ckpt path (of .pt file)
bash ./script/eval_simplerenv/meta_test_SimplerEnv/simpler_0.sh $NUM 0 path/to/cronusvla_0.5B_bridge_rt_1/checkpoints/step-042500-epoch-07-loss=0.0587.pt & 
pid1=$!

bash ./script/eval_simplerenv/meta_test_SimplerEnv/simpler_1.sh $NUM 1 path/to/cronusvla_7B_bridge_rt_1/checkpoints/step-055000-epoch-04-loss=0.0286.pt &
pid2=$!

wait $pid1 $pid2