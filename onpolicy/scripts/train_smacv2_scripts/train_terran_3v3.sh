env="StarCraft2v2"
map="10gen_terran"
algo="rmappo"
units="3v3"

exp="test1"
seed_max=3

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES="" python ../train/train_multi_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed}  --units ${units}  --n_training_threads 1 --n_rollout_threads 1 --num_mini_batch 1 --episode_length 2000 \
    --num_env_steps 60000000 --ppo_epoch 5 --use_value_active_masks --use_eval --eval_episodes 32
done
