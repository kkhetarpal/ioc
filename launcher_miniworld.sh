#!/usr/bin/env bash

main_lr=(1e-4) #(1e-4 3e-4 7e-4 5e-4)
int_lr=(9e-5)  #(1e-4 3e-3 8e-4 8e-5 5e-4 3e-4 9e-5) #(3e-3 8e-5 8e-4)
seed=(0)
port=({4000..4020})
envname="MiniWorld-OneRoom-v0"  #"MiniWorld-PickupObjs-v0" #MiniWorld-PutNext-v0
numoption=2

count=0
for _main_lr in ${main_lr[@]}
do
   for _int_lr in ${int_lr[@]}
   do
        for _seed in ${seed[@]}
        do
                if [ -f temprun.sh ] ; then
                        rm temprun.sh
                fi

                echo "#!/bin/bash" >> temprun.sh
                echo "#SBATCH --account=addccaccounthere" >> temprun.sh
                echo "#SBATCH --output=\"/scratch/username/slurm-%j.out\"" >> temprun.sh
                echo "#SBATCH --gres=gpu:1" >> temprun.sh
                echo "#SBATCH --mem=30G" >> temprun.sh
                echo "#SBATCH --time=10:00:00" >> temprun.sh
                echo "source $HOME/intf/bin/activate" >> temprun.sh
                echo "cd $HOME/ioc/miniworld/baselines/ppoc_int/" >> temprun.sh
                k="xvfb-run -n "${port[$count]}" -s \"-screen 0 1024x768x24 -ac +extension GLX +render -noreset\" python run_miniw.py --env "$envname" --seed $_seed --opt $numoption --saves --mainlr $_main_lr --intlr $_int_lr --switch --wsaves"
                echo $k >> temprun.sh
                echo $k
                eval "sbatch temprun.sh"
                rm temprun.sh
                count=$((count + 1))
        done
   done
done
