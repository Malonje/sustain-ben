

python transfer_crop_type_mapping.py --model_name=unet3d --country=cauvery --num_timesteps=40 --lr=0.0003 --s2_agg=False --planet_agg=False --name=cauvery_3dunet_use_planet_noagg --epochs=130 --batch_size=32 --optimizer=adam --weight_decay=0 --loss_weight=False --weight_scale=1 --dropout=0.5 --clip_val=True --hidden_dims=128 --resize_planet=True --device=cuda --include_indices=True --path_to_cauvery_images ../Dataset --use_planet=False --use_testing True --seed=1   --use_s2=False --use_s1=False --use_l8=True  --l8_bands=[0,7]  --run_name='L8(UB,T)' --split='regionwise';
python transfer_crop_type_mapping.py --model_name=unet3d --country=cauvery --num_timesteps=40 --lr=0.0003 --s2_agg=False --planet_agg=False --name=cauvery_3dunet_use_planet_noagg --epochs=130 --batch_size=32 --optimizer=adam --weight_decay=0 --loss_weight=False --weight_scale=1 --dropout=0.5 --clip_val=True --hidden_dims=128 --resize_planet=True --device=cuda --include_indices=True --path_to_cauvery_images ../Dataset --use_planet=False --use_testing True --seed=1   --use_s2=False --use_s1=False --use_l8=True --l8_bands=[0,7]  --run_name='L8(UB,T)' --split='plotwise';


python transfer_crop_type_mapping.py --model_name=unet3d --country=cauvery --num_timesteps=40 --lr=0.0003 --s2_agg=False --planet_agg=False --name=cauvery_3dunet_use_planet_noagg --epochs=130 --batch_size=32 --optimizer=adam --weight_decay=0 --loss_weight=False --weight_scale=1 --dropout=0.5 --clip_val=True --hidden_dims=128 --resize_planet=True --device=cuda --include_indices=True --path_to_cauvery_images ../Dataset --use_planet=False --use_testing True --seed=1   --use_s2=False --use_s1=True --use_l8=False  --s1_bands=[1,2]  --run_name='S1(VH Angle)' --split='regionwise';
python transfer_crop_type_mapping.py --model_name=unet3d --country=cauvery --num_timesteps=40 --lr=0.0003 --s2_agg=False --planet_agg=False --name=cauvery_3dunet_use_planet_noagg --epochs=130 --batch_size=32 --optimizer=adam --weight_decay=0 --loss_weight=False --weight_scale=1 --dropout=0.5 --clip_val=True --hidden_dims=128 --resize_planet=True --device=cuda --include_indices=True --path_to_cauvery_images ../Dataset --use_planet=False --use_testing True --seed=1   --use_s2=False --use_s1=True --use_l8=False --s1_bands=[1,2]  --run_name='S1(VH Angle)' --split='plotwise';




