# nohup python ../run_training.py --model-name ResNet152 --epoch 1 --batch-size 40 --strategy none > none-b40.log 2>&1
# nohup python ../run_training.py --model-name ResNet152 --epoch 1 --batch-size 40 --strategy str-app > str-app-b40.log 2>&1

nohup python ../run_training.py --model-name ResNet152 --epoch 1 --batch-size 50 --strategy none > none-b50.log 2>&1
nohup python ../run_training.py --model-name ResNet152 --epoch 1 --batch-size 50 --strategy str-app > str-app-b50.log 2>&1

nohup python ../run_training.py --model-name ResNet152 --epoch 1 --batch-size 60 --strategy none > none-b60.log 2>&1
nohup python ../run_training.py --model-name ResNet152 --epoch 1 --batch-size 60 --strategy str-app > str-app-b60.log 2>&1

nohup python ../run_training.py --model-name ResNet152 --epoch 1 --batch-size 70 --strategy none > none-b70.log 2>&1
nohup python ../run_training.py --model-name ResNet152 --epoch 1 --batch-size 70 --strategy str-app > str-app-b70.log 2>&1

nohup python ../run_training.py --model-name ResNet152 --epoch 1 --batch-size 80 --strategy none > none-b80.log 2>&1
nohup python ../run_training.py --model-name ResNet152 --epoch 1 --batch-size 80 --strategy str-app > str-app-b80.log 2>&1

nohup python ../run_training.py --model-name ResNet152 --epoch 1 --batch-size 90 --strategy none > none-b90.log 2>&1
nohup python ../run_training.py --model-name ResNet152 --epoch 1 --batch-size 90 --strategy str-app > str-app-b90.log 2>&1