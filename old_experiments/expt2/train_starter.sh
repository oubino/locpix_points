#Python train script with modified k

#Run with current environment (-V) and in the current directory (-cwd)
#$ -V -cwd

#Request some time- min 15 mins - max 48 hours
#$ -l h_rt=02:30:00

# GPU/CPU request
#$ -l coproc_v100=1

#Get email at start and end of the job
#$ -m be

#Now run the job
python locpix_points/src/locpix_points/scripts/train.py -i /nobackup/scou/output/nieves/expt2 -c locpix_points/experiments/expt2/starter.yaml