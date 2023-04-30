method=all

hist_file="./res/sensors/_hist.txt"
echo "$(date +"%T") run history" > $hist_file

# 1. generate MCMC samples
sensors_location.R

CUDA_VISIBLE_DEVICES="" taskset -c 11-15 python3 experiment_sensor.py --METHOD=$method --RAM_SCALE=0.1 &
CUDA_VISIBLE_DEVICES="" taskset -c 16-20 python3 experiment_sensor.py --METHOD=$method --RAM_SCALE=0.3 &
CUDA_VISIBLE_DEVICES="" taskset -c 21-25 python3 experiment_sensor.py --METHOD=$method --RAM_SCALE=0.5 &
wait
echo "$(date +"%T") finished 1/2" >> $hist_file

CUDA_VISIBLE_DEVICES="" taskset -c 11-15 python3 experiment_sensor.py --METHOD=$method --RAM_SCALE=0.7 &
CUDA_VISIBLE_DEVICES="" taskset -c 16-20 python3 experiment_sensor.py --METHOD=$method --RAM_SCALE=0.9 &
CUDA_VISIBLE_DEVICES="" taskset -c 21-25 python3 experiment_sensor.py --METHOD=$method --RAM_SCALE=1.08 &
CUDA_VISIBLE_DEVICES="" taskset -c 26-30 python3 experiment_sensor.py --METHOD=$method --RAM_SCALE=1.3 &
wait
echo "$(date +"%T") finished 2/2" >> $hist_file

# # sample size
# for n in 500 1000 2000
# do
#     CUDA_VISIBLE_DEVICES="" taskset -c 1-15 python3 experiment_sensor.py --METHOD=$method --RAM_SCALE=0.1 --n=$n &
# done
# wait
# echo "$(date +"%T") finished sample size" >> $hist_file