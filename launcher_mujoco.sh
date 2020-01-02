#!/usr/bin/env bash

seed=(0 1 2 3 4)
mainlr=(1e-4)
#intfclr=(1e-4 3e-4 5e-4 7e-4 9e-4)
intfclr=(5e-4)
#piolr=(7e-4 9e-4 3e-4 5e-4)
piolr=(3e-4)

port=($(seq 4000 1 4100))

envname="HalfCheetahDir-v1"

count=0

for _piolr in ${piolr[@]}
do
    for _intfclr in ${intfclr[@]}
    do
        for _mainlr in ${mainlr[@]}
        do
            for _seed in ${seed[@]}
            do
                if [ -f temprun.sh ] ; then
                    rm temprun.sh
                fi
                echo "#!/bin/bash" >> temprun.sh
                echo "#SBATCH --account=addaccounthere" >> temprun.sh
                echo "#SBATCH --output=\"/scratch/username/maml/Maml_seed${_seed}_mainlr${_mainlr}_intfclr_${_intfclr}_piolr_${_piolr}-%j.out\"" >> temprun.sh
                echo "#SBATCH --job-name=Maml_seed${_seed}_mainlr${_mainlr}_intfclr_${_intfclr}_piolr_${_piolr}" >> temprun.sh
                echo "#SBATCH --gres=gpu:0" >> temprun.sh
                echo "#SBATCH --mem=5G" >> temprun.sh
                echo "#SBATCH --time=1:00:00" >> temprun.sh
                echo "source $HOME/miniconda3/etc/profile.d/conda.sh" >> temprun.sh
                echo "conda activate intfc" >> temprun.sh
                echo "cd $HOME/ioc/control/baselines/ppoc_int/" >> temprun.sh
                k="xvfb-run -n "${port[$count]}" -s \"-screen 0 1024x768x24 -ac +extension GLX +render -noreset\" python run_mujoco.py --env "$envname" --saves --opt 2 --seed ${_seed} --mainlr ${_mainlr} --intlr ${_intfclr} --piolr ${_piolr} --switch --wsaves"
                echo $k >> temprun.sh
                echo $k
                eval "sbatch temprun.sh"
                rm temprun.sh
                count=$((count + 1))
            done
        done
    done
done
