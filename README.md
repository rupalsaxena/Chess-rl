# Chess-rl
This repository consists of project of Introduction to Reinforcement Learning course of University of Zurich.

Following are the ways to run different Reinforcement Learning algorithms.
If you are Linux and MacOS user, do the following:

To run sarsa:
```
# only one time command
chmod +x sarsa.sh
# run sarsa
./sarsa.sh
```

To run q-learning:
```
# only one time command
chmod +x q-learning.sh
# run q-learning
./q-learning.sh
```

To run random:
```
# only one time command
chmod +x random.sh
# run random 
./random.sh
```

To run sarsa with exploding gradients:
```
# only one time command
chmod +x sarsa_explode.sh
# run sarsa explode gradients 
./sarsa_explode.sh
```

Alternate method to run the code is as follows:
Go to src/main/main.py file, choose your config path of choice. 
Please make sure to comment 
```
#CONFIG_PATH = os.environ["CONFIG_PATH"]
```
Please make sure to uncomment your config of choice:
```
CONFIG_PATH = "src/main/configs/sarsa.yaml"
#CONFIG_PATH = "src/main/configs/q-learning.yaml"
#CONFIG_PATH = "src/main/configs/random.yaml"
#CONFIG_PATH = "src/main/configs/sarsa_explode_gradients.yaml"
```
Save the changes and then run the following:
```
python3 src/main/main.py
```

