#$ -S /bin/bash
#$ -l h_rt=72:00:00
#$ -pe shmem-4 4
#$ -q gpu-r8.q
#$ -l mem_free=50G,h_rss=50G
#$ -wd /home/mateuszm/downscaling/
#$ -o logs/
#$ -e logs/

conda activate myenv

python train.py