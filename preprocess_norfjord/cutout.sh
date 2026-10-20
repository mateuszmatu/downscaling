
#$ -S /bin/bash
#$ -l h_rt=72:00:00
#$ -q research-r8.q
#$ -l h_rss=20G
#$ -l mem_free=20G 
#$ -l h_data=20G
#$ -wd /home/mateuszm/downscaling/preprocess_norfjord/
#$ -o logs/
#$ -e logs/
#$ -N cutoutA12


bash -l
#conda activate myenv
source /modules/rhel8/mamba-mf3/etc/profile.d/ppimam.sh
conda activate 2025-01-production
python /home/mateuszm/downscaling/preprocess_norfjord/cutout.py -s 2024-01-01 -e 2024-12-31 -d A12
