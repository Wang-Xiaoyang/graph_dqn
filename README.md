# graph_dqn

## This project is about designing a graph-q-learning agent, in which the Q-network is a graph convolutional network (GCN).

## The environment is a vector bin packing problem (VBPP), which simulates the RU-DU association problem in the Open-RAN.

Profile a python script

`python -m cProfile -s cumulative -o ./profile.csv python_script.py`

Check the profile

`cprofilev -f ./profile.csv`
