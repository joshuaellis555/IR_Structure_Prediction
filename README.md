# IR_Structure_Prediction

Install the Conda environment through the environment_IR.yml file.

Requires SpecJar.pickle from https://www.dropbox.com/sh/kvqmvuqa6lw4a5p/AAD27Rqbm8WCO3VLaFHXi6D1a?dl=0

•	Add the file to the same folder as train_dqn.py and evaluate_dqn.py

Training can be restarted by running train_dqn.py

•	Change the  * *fold* * parameter (indexed 0 to 4) to train a model on a different fold. Be sure to also change the * *fold* * parameter in evaluate_dqn.py. Results will be erroneous if training and testing occur on different folds.

To run evaluation on the trained model, run evaluate_dqn.py.

Trained model reported in the paper can be downloaded from: https://www.dropbox.com/sh/kvqmvuqa6lw4a5p/AAD27Rqbm8WCO3VLaFHXi6D1a?dl=0

•	Training state (the 13GB file) is not needed to evaluate the model.