random_scale=0.15
rlaunch --cpu=2 --gpu=1 --memory=4096 -- python demo/motion_imitate.py --gpu_ids 0 \
   --image_size 512 \
   --num_source 2   \
   --random_scale=$random_scale \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "custom1_015" \
   --src_path   "path?=./assets/samples/sources/custom/doclggcocv.jpeg,name?=custom1_015" \
   --ref_path   "path?=./assets/samples/references/akun_2.mp4,name?=akun_2,pose_fc?=300"

random_scale=0.20
rlaunch --cpu=2 --gpu=1 --memory=4096 -- python demo/motion_imitate.py --gpu_ids 0 \
   --image_size 512 \
   --num_source 2   \
   --random_scale=$random_scale \
   --output_dir "./results" \
   --assets_dir "./assets"  \
   --model_id   "custom1_02" \
   --src_path   "path?=./assets/samples/sources/custom/doclggcocv.jpeg,name?=custom1_02" \
   --ref_path   "path?=./assets/samples/references/akun_2.mp4,name?=akun_2,pose_fc?=300"
