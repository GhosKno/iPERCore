rlaunch --cpu=3 --gpu=1 --memory=8192 -- python demo/motion_imitate.py --gpu_ids 0 \
   --image_size 512 \
   --num_source 2   \
   --output_dir "./results/results_multiview" \
   --model_id   "afan_6" \
   --src_path   "path?=./assets/samples/sources/afan_6/afan_6=ns=2,name?=afan_6,bg_path?=./assets/samples/sources/afan_6/IMG_7217.JPG" \
   --ref_path   "path?=/data/jupyter/dy_dance_video/gallery/11/a_合格_455e9a4a248db073002505f028c5ef92_mp4_act11.mp4,name?=a_合格_455e9a4a248db073002505f028c5ef92_mp4_act11,pose_fc?=300"
