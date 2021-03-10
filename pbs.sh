#PBS -N T5_Java
#PBS -A lisch175
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l pmem=2gb
#PBS -l walltime=10:00
#PBS -q testflight-gpu
#PBS -j oe
#PBS -o T5_Java.out

set -e
export LOGFILE=$PBS_O_WORKDIR/$PBS_JOBNAME"."$PBS_JOBID".log"

SCRATCHDIR=/scratch_gs/$USER/$PBS_JOBID
mkdir -p "$SCRATCHDIR" 


cd $PBS_O_WORKDIR

echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" START" > $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE 


# module load pytorch/1.7
# module load transformers/4.1.1
# module load pytorch-lightning/1.1.3
# module load tokenizers/0.9.4
# module load sentencepiece/0.1.94
# module load tqdm/4.56.0

cp -r $PBS_O_WORKDIR/* $SCRATCHDIR/.
cd $SCRATCHDIR
rm $PBS_JOBNAME"."$PBS_JOBID".log"

echo >> $LOGFILE
qstat -f $PBS_JOBID >> $LOGFILE  


wget https://s3.us-east-2.amazonaws.com/leclair.tech/data/funcom/funcom_filtered.tar.gz
tar xfvz funcom_filtered.tar.gz

# !pip install --quiet transformers==4.1.1
# !pip install --quiet pytorch-lightning==1.1.3
# !pip install --quiet tokenizers==0.9.4 
# !pip install --quiet sentencepiece==0.1.94
# !pip install --quiet tqdm==4.56.0
# !pip install --quiet torch==1.7.0

pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de transformers==4.1.1
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de pytorch-lightning==1.1.3
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de tokenizers==0.9.4
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de sentencepiece==0.1.94
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de tqdm==4.56.0
pip install --user -i http://pypi.repo.test.hhu.de/simple/ --trusted-host pypi.repo.test.hhu.de torch==1.7.0





python train_t5.py


echo "$PBS_JOBID ($PBS_JOBNAME) @ `hostname` at `date` in "$RUNDIR" END" >> $LOGFILE
echo "`date +"%d.%m.%Y-%T"`" >> $LOGFILE