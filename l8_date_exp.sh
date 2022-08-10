python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 True --use_planet False --use_s1 False --use_s2 False --batch_size 2 --date_pred_for sowing --l8_bands [0,1,2,3,4,5,6] --use_testing True --epochs 20;

python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 True --use_planet False --use_s1 False --use_s2 False --batch_size 2 --date_pred_for sowing --l8_bands [0,1,2,3,4,5,6,7] --use_testing True --epochs 20;

python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 True --use_planet False --use_s1 False --use_s2 False --batch_size 2 --date_pred_for sowing --l8_bands [3,2,1,0] --use_testing True --epochs 20;

python train_date_prediction.py --path_to_cauvery ../../Documents/ --model_name unet-fc --num_timesteps 184 --include_indices True --use_l8 True --use_planet False --use_s1 False --use_s2 False --batch_size 2 --date_pred_for sowing --l8_bands [0,7] --use_testing True --epochs 20;

