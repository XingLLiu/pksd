# 1h30min at 2%
# new: 1h25min at 2%
# method=spksd # 48mins at 4%
# method=pksd
# method=all # 2h2min at 1%
# CUDA_VISIBLE_DEVICES="" taskset -c 1-50 python3  \
#     experiments.py --model=rbm --dim=50 --dh=10 --shift=1. --T=50 --n=1000 --nrep=100 --rand_start=10. --method=$method


hist_file="./res/rbm/_hist.txt"
echo "$(date +"%T") run history" > $hist_file

method=all

# # latent dim
# for dh in 2 5 8 10 # 20 30 40
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python3 \
#     experiments.py --model=rbm --dim=50 --dh=$dh --shift=1. --T=50 --n=1000 --nrep=100 --rand_start=10. --method=$method &
# done
# wait
# echo "$(date +"%T") finished latent dim 1/2" >> $hist_file
# for dh in 20 30 40
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python3 \
#     experiments.py --model=rbm --dim=50 --dh=$dh --shift=1. --T=50 --n=1000 --nrep=100 --rand_start=10. --method=$method &
# done
# wait
# echo "$(date +"%T") finished latent dim 2/2" >> $hist_file

# # # perturbation to mixing ratios
# for shift in 0. 0.01 0.05 #0.1 0.5 1.
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python3 \
#     experiments.py --model=rbm --dim=50 --dh=5 --shift=$shift --T=50 --n=1000 --nrep=100 --rand_start=10. --method=$method &
# done
# wait
# echo "$(date +"%T") finished mixing ratios 1/2" >> $hist_file
# for shift in 0.1 0.5 1.
# do
#   CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python3 \
#     experiments.py --model=rbm --dim=50 --dh=5 --shift=$shift --T=50 --n=1000 --nrep=100 --rand_start=10. --method=$method &
# done
# wait
# echo "$(date +"%T") finished mixing ratios 2/2" >> $hist_file

# # # sample size
# for n in 200 500 1000 #1500 2000
# do
#   # power test
#   CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 \
#     experiments.py --model=rbm --dim=50 --dh=5 --shift=1. --T=50 --n=$n --nrep=100 --rand_start=10. --method=$method &
  
#   # level test
#   CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 \
#     experiments.py --model=rbm --dim=50 --dh=5 --shift=0. --T=50 --n=$n --nrep=100 --rand_start=10. --method=$method &
# done
# wait
# echo "$(date +"%T") finished sample size 1/2" >> $hist_file
# for n in 1500 2000
# do
#   # power test
#   CUDA_VISIBLE_DEVICES="" taskset -c 0-10 python3 \
#     experiments.py --model=rbm --dim=50 --dh=5 --shift=1. --T=50 --n=$n --nrep=100 --rand_start=10. --method=$method &
  
#   # level test
#   CUDA_VISIBLE_DEVICES="" taskset -c 11-20 python3 \
#     experiments.py --model=rbm --dim=50 --dh=5 --shift=0. --T=50 --n=$n --nrep=100 --rand_start=10. --method=$method &
# done
# wait
# echo "$(date +"%T") finished sample size 2/2" >> $hist_file

# perturbation to mode separation
# for B_scale in 0.2 0.3 0.4 #0. 0.1 0.5 1. 5. 10.
# for B_scale in 2. #3. 4.
# do
#   # CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python3 \
#     # experiments.py --model=rbm --dim=50 --dh=5 --shift=1. --T=50 --n=1000 --nrep=100 --rand_start=10. --B_scale=$B_scale --method=$method &
#   CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python3 \
#     experiments.py --model=rbm --dim=50 --dh=5 --shift=1. --T=50 --n=1000 --nrep=100 --rand_start=5. --B_scale=$B_scale --method=$method &
# done
# wait
# echo "$(date +"%T") finished B scale" >> $hist_file

# perturbation to B
for std in 0. 0.01 0.02 0.04 0.06 0.08 0.1 0.5 1.0 1.2
do
  CUDA_VISIBLE_DEVICES="" taskset -c 0-20 python3 \
    experiments.py --model=rbmStd --dim=50 --dh=5 --shift=1. --T=50 --n=1000 --nrep=100 --rand_start=10. --noise_std=$std --method=$method &
done
wait
echo "$(date +"%T") finished B scale" >> $hist_file
