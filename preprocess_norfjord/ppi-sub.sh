
#$ -S /bin/bash
#$ -l h_rt=48:00:00
#$ -q research-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G 
#$ -l h_data=8G
#$ -wd /lustre/storeA/users/mateuszm/NF160/


bash -l
#conda activate myenv
source /modules/rhel8/mamba-mf3/etc/profile.d/ppimam.sh
conda activate 2025-01-production
python /home/mateuszm/downscaling/preprocess_norfjord/transform.py -s 2024-01-01 -e 2024-12-31 -d A01

