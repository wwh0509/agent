


python -u point_nav_ppo_train.py \
    --model_ids Rs_int \
    2>&1 | grep -v "Created DrawableGroup" > output_clamp.log \