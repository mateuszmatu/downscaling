
#$ -S /bin/bash
#$ -l h_rt=48:00:00
#$ -q nuclear-r8.q
#$ -l h_rss=8G
#$ -l mem_free=8G 
#$ -l h_data=8G
#$ -wd /lustre/storeA/users/mateuszm/NF160/


bash -l
#conda activate myenv
source /modules/rhel8/mamba-mf3/etc/profile.d/ppimam.sh
conda activate 2025-01-production
python /home/mateuszm/wp3/NorFjordToZDepth/transform.py -s 2022-05-01 -e 2022-05-31 -d A05

