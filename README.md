# IR_Structure_Prediction

Install the Conda environment through the environment_IR.yml file.

The trained model can be found in the States folder.

To run evaluation on the trained model, run evaluate_dqn.py.

Training can be restarted by running train_dqn.py
•	Change the  * *fold* * parameter (indexed 0 to 4) to train a model on a different fold. Be sure to also change the fold parameter in evaluate_dqn.py. Results will be erroneous if training and testing occur on different folds.
