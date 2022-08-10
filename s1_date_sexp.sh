python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 False --use_planet False --use_s1 True --use_s2 False --batch_size 2 --date_pred_for sowing --s1_bands [0,1,2] --use_testing True --epochs 20;

python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 False --use_planet False --use_s1 True --use_s2 False --batch_size 2 --date_pred_for sowing --s1_bands [2] --use_testing True --epochs 20;

python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 False --use_planet False --use_s1 True --use_s2 False --batch_size 2 --date_pred_for sowing --s1_bands [0] --use_testing True --epochs 20;

python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 False --use_planet False --use_s1 True --use_s2 False --batch_size 2 --date_pred_for sowing --s1_bands [1] --use_testing True --epochs 20;


