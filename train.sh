#$ -S /bin/bash
#$ -l h_rt=72:00:00
#$ -q gpu-r8.q
#$ -l h=sm-nx10077659-bc-compute.int.met.no
##$ -l h_rss=200G,mem_free=200G,h=sm-nx10077659-bc-compute.int.met.no
#$ -wd /home/mateuszm/downscaling/
#$ -o output/
#$ -e output/

conda activate myenv

python train.py