# A Fairness-Oriented Reinforcement Learning Approach for the Operation and Control of Shared Micromobility Services

## How to train and/or evaluate the algorithm

1. Clone the repository

2. Train all the scenarios across different seeds:
   ```
   ./training.sh
   ```
3. After training, evaluate the algorithm. You can also just evaluate the algorithm by using our pre-trained models.
   ```
   # The different scripts correspond to the different scenarios: from 2 to 5 categories
   python evaluation_2.py
   python evaluation_3.py
   python evaluation_4.py
   python evaluation_5.py
   ```
4. Plot the results:
   ```
   # Here, x must be substituted with the number of categories
   python boxplots.py --cat 5 --save
   python paretoplots_new.py --cat x --save
   ```
