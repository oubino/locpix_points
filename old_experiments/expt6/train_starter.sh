#Python train script with modified k

#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=05:30:00

# GPU/CPU request
#$ -l coproc_v100=1

#Get email at start and end of the job
#$ -m be

#Now run the job
python locpix_points/src/locpix_points/scripts/train.py -i /nobackup/scou/output/c15/expt6 -c locpix_points/experiments/expt6/starter.yaml
