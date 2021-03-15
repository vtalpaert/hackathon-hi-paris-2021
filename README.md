# Hi! Paris Hackathon 2021

https://hackathon-hi-paris.fr/

[Event instructions](https://drive.google.com/file/d/1i724c0-8ZfwtsJas0qenCKDTph8w7Dp6/view)

## The event

First Data / AI event of Hi! Paris, the new interdisciplinary research and teaching Center for Data Analytics and Artificial Intelligence

## The theme

AI for Energy Efficiency Beyond producing a Data / AI model, the competition will ask to realistically project the solutions in a market context

## When?

From Friday March 12, 6:30pm to Sunday March 14, 5pm

## Mission

- You have 3 buildings which act as Smart grids. Each building is equipped with a PV (solar power) and a battery, and the third building is additionally equipped with a fuel-based generator.
- The buildings must try to stay as independent as possible energy-wise from the public grid, but they may need at some moments to import electricity from the public grid, which generates additional costs. Also, if the fuel-based generator (or genset) is used, additional costs are incurred because using fuel is expensive.
- Every hour, the 3 buildings need to be supplied with energy and you will have to decide how to supply them:
    - Should I import all the energy needed from the grid ?
    - Should I use the energy I have already stored in my battery to avoid importation costs ?
    - Is it the right time to use the fuel-based generator ?

## Team and project overview

Team members are Selma Azennar, Raphaël Teitgen and Victor Talpaert.
Selma is a ENSAE and HEC student, passionate about yoga and data science. Victor likes recycling and half baked AI. Raphaël enjoys building computers and pushing ideas to their limits. Raphaël and Victor are both doctoral candidates from ENSTA Paris.

Thank you guys !

### General strategy

We train a Deep Q-network, the agent finds the best course of action for reducing the cost of each building. The learning model prioritizes high value data and can decide when to stop training. We used data augmentation to highlight periodic effects.

## Scientific approach

### Approach description

As a very standard start, we looked into the data. It is clear that a lot of the building behavor is predetermined. The buildings simulate expected behaviors, such as load reduction on weekends or in the midst of summer. There is no extra load during the winter, which could happen due to warming the place up, but one can guess that for the sake of the exercise, the load ought to follow the solar illumination a bit.

Since time was of the essence, we chose a DQN network with prioritized replay [1]. Still, due to predetermined behaviors a large enough replay memory could have suffice. In the spirit of energy and calculation efficieny, the priority replay helps reducing the training time. Credit for the base code goes to [2].

We initially put all our calculations in Foat16 (half precision) in reduce the calculation footprint. Since the evaluation is done on CPU time, we had to stick with Float32 precision since most 16-bits operations are not supported on CPU.

As an immediate way to reduce unecessary calculation, we introduced early stopping according to two strategies : when the training score is on a plateau, or when we get close to the perfect score.

Lighter networks were of great help for us, especially due to overfitting caused by, again, the predetermined buildings behaviors. But our algorithm is robust to a non-fixed seed building behavior.

Normalisation is a necessary step in most tasks. We brought all parameters to a 0..1 range. Additionnaly, we introduced two extra features; the week progress and the year progress. They encode how close we are to the weekend and the seasons respectively.

### Future improvement

The building 3 is not as similar as building 1 is to 2. For this, different networks sizes should be chosen, instead of a one-size-fit-all (not pun intended).

Still, with a common network, pretrained network could fasten the training. As well as feeding data from the rule based approaches [3].

Policy distillation would be a nice way to simplify a network [4].

## Project usage

The code is either run through the notebook where our results are already visible, or by executing the `train_per.py` file which will log the experiments to tensorboard.

The prerequisite are in `requirements.txt`, namely PyTorch.

In `wrappers.py`, we have an observation wrapper for data augmentation.

## References

1. Schaul, T. et al. “Prioritized Experience Replay.” CoRR abs/1511.05952 (2016): n. pag.
1. https://github.com/stormont/per
1. Hu, Weihua et al. “Strategies for Pre-training Graph Neural Networks.” arXiv: Learning (2020): n. pag.
1. Rusu, Andrei A. et al. “Policy Distillation.” CoRR abs/1511.06295 (2016): n. pag.
