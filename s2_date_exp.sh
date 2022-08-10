python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 False --use_planet False --use_s1 False --use_s2 True --batch_size 2 --date_pred_for sowing --s2_bands [0,1,2,3,4,5,6,7,8,9] --use_testing True --epochs 20;

python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 False --use_planet False --use_s1 False --use_s2 True --batch_size 2 --date_pred_for sowing --s2_bands [2,1,0,3] --use_testing True --epochs 20;

python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 False --use_planet False --use_s1 False --use_s2 True --batch_size 2 --date_pred_for sowing --s2_bands [2,1,0,4] --use_testing True --epochs 20;

python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 False --use_planet False --use_s1 False --use_s2 True --batch_size 2 --date_pred_for sowing --s2_bands [2,1,0,5] --use_testing True --epochs 20;

python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 False --use_planet False --use_s1 False --use_s2 True --batch_size 2 --date_pred_for sowing --s2_bands [2,1,0,7] --use_testing True --epochs 20;

python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 False --use_planet False --use_s1 False --use_s2 True --batch_size 2 --date_pred_for sowing --s2_bands [2,1,0,3,4,5,7] --use_testing True --epochs 20;

python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 False --use_planet False --use_s1 False --use_s2 True --batch_size 2 --date_pred_for sowing --s2_bands [3,4,5,7] --use_testing True --epochs 20;
