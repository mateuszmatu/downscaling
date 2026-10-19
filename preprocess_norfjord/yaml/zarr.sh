#$ -S /bin/bash
#$ -l h_rt=72:00:00
#$ -q bigmem-r8.q
#$ -l h_rss=80G,mem_free=80G,h_data=80G
#$ -o /lustre/storeB/users/mateuszm/NF160/logs/
#$ -e /lustre/storeB/users/mateuszm/NF160/logs/
#$ -N anemoi-datasets-A09_2


DOMAIN=A09_2



OUTDIR=/lustre/storeB/users/mateuszm/NF160/zarr/
ZARRFILE=$OUTDIR/$DOMAIN.zarr
YAMLFILE=/home/mateuszm/downscaling/preprocess_norfjord/yaml/$DOMAIN.yaml   

conda deactivate
source /lustre/storeB/project/fou/hi/foccus/python-envs/anemoi-env-9-6-26/bin/activate

# No more parallell, because there are some buggs where it doesn't finish. 
# This is more consistent. 


anemoi-datasets init $YAMLFILE $ZARRFILE --overwrite
for i in {1..10}
do
    anemoi-datasets load $ZARRFILE --part $i/10
done

anemoi-datasets finalise $ZARRFILE
