method=all

hist_file="./res/t-banana/_hist.txt"
echo "$(date +"%T") run history" > $hist_file

# for t_std in 0.01 0.05 0.1 0.5 1. 2. 5.
# for t_std in 10. 15. 20.
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 experiments.py --method=$method \
#   --model=t-banana --dim=10 --nmodes=20 --nbanana=10 --T=100 --n=1000 --ratio_s_var=1. --t_std=$t_std --nrep=100 --rand_start=20. &
# done
# wait

# echo "$(date +"%T") finished t_std" >> $hist_file

# for dim in 2 5 10 15 20 25 30 35 40 45 50
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 experiments.py --method=$method \
#   --model=t-banana --dim=$dim --nmodes=20 --nbanana=10 --T=100 --n=1000 --ratio_s_var=5. --t_std=0.1 --nrep=100 --rand_start=20. &
# done

# wait
# echo "$(date +"%T") finished dims" >> $hist_file

# for ratio_s_var in 0.01 0.05 0.1 0.5 1. 2. 5. 10. 15. 20.
for ratio_s_var in 0.
do
  CUDA_VISIBLE_DEVICES="" taskset -c 11-30 python3 experiments.py --method=$method \
  --model=t-banana --dim=10 --nmodes=20 --nbanana=10 --T=100 --n=1000 --ratio_s_var=$ratio_s_var --t_std=0.1 --nrep=100 --rand_start=20. &
done

wait
echo "$(date +"%T") finished ratio_s_var" >> $hist_file
  