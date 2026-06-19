mkdir -p logs

nohup python main.py --dataset=ace --seed=1 --root ../data/ace/ --known_class_filename train_concept_Qwen3-32B.json --new_class_filename train_concept_Qwen3-32B.json --test_class_filename test_dev_concept_Qwen3-32B.json --b_size 256 --max_len 240 --load_checkpoint_path ../clustering/model/model_ace_best.pt --lr 1e-3 --epochs 100 --score_pos_loss 2.0 --score_neg_loss 2.0 --noise_ratio 0.0 --aggregator_loss 0.1 --gpu_ids 0 --cuda --use_trg_concept --train_hierarchy_only --freeze_taxonomy_embeddings > logs/ace_linking_train.log 2>&1 &
