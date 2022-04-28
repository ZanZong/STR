nohup python ../run_training.py --model-name transformer --epoch 1 --batch-size 210 --strategy none > none-b210.log 2>&1
nohup python ../run_training.py --model-name transformer --epoch 1 --batch-size 210 --strategy str-app > str-app-b210.log 2>&1

nohup python ../run_training.py --model-name transformer --epoch 1 --batch-size 230 --strategy none > none-b230.log 2>&1
nohup python ../run_training.py --model-name transformer --epoch 1 --batch-size 230 --strategy str-app > str-app-b230.log 2>&1

nohup python ../run_training.py --model-name transformer --epoch 1 --batch-size 250 --strategy none > none-b250.log 2>&1
nohup python ../run_training.py --model-name transformer --epoch 1 --batch-size 250 --strategy str-app > str-app-b250.log 2>&1

nohup python ../run_training.py --model-name transformer --epoch 1 --batch-size 270 --strategy none > none-b270.log 2>&1
nohup python ../run_training.py --model-name transformer --epoch 1 --batch-size 270 --strategy str-app > str-app-b270.log 2>&1

nohup python ../run_training.py --model-name transformer --epoch 1 --batch-size 290 --strategy none > none-b290.log 2>&1
nohup python ../run_training.py --model-name transformer --epoch 1 --batch-size 290 --strategy str-app > str-app-b290.log 2>&1
