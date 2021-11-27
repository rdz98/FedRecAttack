# FedRecAttack

Model Poisoning Attack to Federated Recommendation.

This is the pytorch implementation of FedRecAttack.

## Environment
+ Python 3.7.9
+ numpy==1.18.5
+ torch==1.7.0+cu101


## Usage

To run FedRecAttack on MovieLens-100K with ![](http://latex.codecogs.com/svg.latex?\rho=5\%,\kappa=60,\xi=1\%):

`python main.py --dataset=ml-100k/ --attack=FedRecAttack --clients_limit=0.05 --items_limit=60 --part_percent=1`

There are three choices on dataset:

`--dataset=ml-100k/`, `--dataset=ml-1m/` and `--dataset=steam/`.

There are four choices on attack:

`--attack=FedRecAttack`, `--attack=Random`, `--attack=Bandwagon` and `--attack=Popular`.

## Output
```
Arguments: attack=FedRecAttack,dim=32,path=Data/,dataset=ml-100k/,device=cuda,lr=0.01,epochs=200,batch_size=256,grad_limit=1.0,clients_limit=0.05,items_limit=60,part_percent=1,attack_lr=0.01,attack_batch_size=256
Load data done [5.4 s]. #user=990, #item=1682, #train=99056, #test=943
Target items: [894].
output format: ({Sampled HR@10}), ({ER@5},{ER@10},{NDCG@10})
Iteration 0(init), (0.0848) on test, (0.0021, 0.0054, 0.0025) on target. [1.1s]
Iteration 1, loss = 72.84028 [3.2s], (0.0732) on test, (0.3869, 0.4019, 0.3799) on target. [1.0s]
Iteration 2, loss = 72.82960 [2.0s], (0.0764) on test, (0.5198, 0.5220, 0.5133) on target. [1.1s]
Iteration 3, loss = 72.82328 [2.0s], (0.0817) on test, (0.6045, 0.6077, 0.6001) on target. [1.1s]
Iteration 4, loss = 72.81755 [2.0s], (0.1166) on test, (0.7170, 0.7192, 0.7155) on target. [1.0s]
Iteration 5, loss = 72.80792 [2.0s], (0.2174) on test, (0.7814, 0.7824, 0.7809) on target. [1.0s]
... ...
```

## License

The codes are currently for presentation to the reviewers only.
Please do not use the codes for other purposes.

## Fix

There are several writing errors in our paper, which are corrected here:

+ In Eq. (14), when ![](http://latex.codecogs.com/svg.latex?x<0,g(x)=e^x-1).