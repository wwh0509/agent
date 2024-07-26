


python -u point_nav_ppo_train.py \
    --root_dir "test" \
    --config_file "gibson_extension/examples/configs/turtlebot_multi_point_nav.yaml" \
    --agent_config_file "../configs/agent.yaml" \
    --initial_collect_steps 1024 \
    --collect_steps_per_iteration 1 \
    --batch_size 1024 \
    --train_steps_per_iteration 1 \
    --replay_buffer_capacity 10000 \
    --num_eval_episodes 10 \
    --eval_interval 10000000 \
    --gpu_c 0 \
    --gpu_g 0 \
    --generate_data True \
    --num_parallel_environments 2 \
    --model_ids "Ihlen_0_int, Merom_0_int" \
    --num_episodes 5 \
    2>&1 | grep -v "Created DrawableGroup" > generate_data.log \