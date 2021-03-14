# TEAM 17

## Team and project overview

Team members are Selma, Raphaël and Victor. Thank you guys !

### General strategy

We train a Deep Q-network, the agent finds the best course of action for reducing the cost of each building. The learning model prioritizes high value data and can decide when to stop training. We used data augmentation to highlight periodic effects.

## Scientific approach

### Approach description

Describe the approach(es) you adopted to solve the problem raised in this hackaton. Strive to justify your choices.

As a very standard start, we looked into the data. It is clear that a lot of the building behavor is predetermined. The buildings simulate expected behaviors, such as load reduction on weekends or in the midst of summer. There is no extra load during the winter, which could happen due to warming the place up, but one can guess that for the sake of the exercise, the load ought to follow the solar illumination a bit.

Since time was of the essence, we chose a DQN network with prioritized replay. Still, due to predetermined behaviors a large enough replay memory could have suffice. In the spirit of energy and calculation efficieny, the priority replay helps reducing the training time.

We initially put all our calculations in Foat16 (half precision) in reduce the calculation footprint. Since the evaluation is done on CPU time, we had to stick with Float32 precision since most 16-bits operations are not supported on CPU.

As an immediate way to reduce unecessary calculation, we introduced early stopping according to two strategies : when the training score is on a plateau, or when we get close to the perfect score.

Lighter networks were of great help for us, especially due to overfitting caused by, again, the predetermined buildings behaviors. But our algorithm is robust to a non-fixed seed building behavior.

Normalisation is a necessary step in most tasks. We brought all parameters to a 0..1 range. Additionnaly, we introduced two extra features; the week progress and the year progress. They encode how close we are to the weekend and the seasons respectively.

### Future improvement

Deepen the graph analysis and produce a rule based approach

Can be usefull for pretraining ? (can be ecological)

Policy distillation

## Project usage

The code is either run through the notebook, or by executing the `train_per.py` file.

The prerequisite are in `requirements.txt`, namely PyTorch.

In `wrappers.py`, we have an observation wrapper for data augmentation
