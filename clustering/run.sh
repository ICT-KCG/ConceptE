mkdir -p logs

nohup python main.py --dataset=ace --seed=1 --root ../data/ace/ --known_class_filename train_concept_Qwen3-32B.json --new_class_filename train_concept_Qwen3-32B.json --test_class_filename test_dev_concept_Qwen3-32B.json --b_size 64 --max_len 240  --warmup 60 -gpu_ids 0 --cuda --fuse_concept > logs/ace_clustering_train.log 2>&1 &
