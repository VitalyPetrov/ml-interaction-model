# ml-interaction-model
Machine learning (ML)-based framework to train interaction models between atoms

Typical usage scenario:

- create directories storing the simulations i/o data

'''bash
python init_random_structures.py
'''

- (optional) specify/implement your interaction model in interaction_models.py file

- process the simulation data output and train interaction model:

'''bash
python train_model.py
'''
