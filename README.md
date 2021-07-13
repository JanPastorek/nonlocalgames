# NonlocalGames

Framework for nonlocalgames

Webpage of the project:
https://janpastorek.github.io/Bachelor-Thesis/

Few words towards this project:

* #### The first goal was to build a scalable non local game environment with local database for storing important information from the search. (we have built scalable environment and we have marked methods which are not scalable and need separate implementation)

* #### The second goal was to try different approaches that will be able to search through the strategies for non local game and output/learn the paths to bigger winning probability (CHSH value).  

NonLocalGame provides abstract framework for nonlocalgames.

There are so far six implementations of non local environments.

* CHSHPrototype - is just a prototype that uses fixed gates(path) to illustrate how CHSH game (basic nonlocal game) works.

For the below ones we have implemented methods to search for the best strategies

* NlgDeterministic - is a way to generate best strategies for 2-player, 2-question games classicaly

* NlgParalelClassical - is a way to generate best strategies for 2-player, n-questions games classicaly

* NlgDiscreteStatesActions - is fully functioning nonlocal game discrete environment optimized for discrete states and actions to be used on this environment. Can either use discrete gates as input, or can use simulated annealing on each step's action's gate which always chooses suboptimal gate as action.

* NlgGeneticOptimalization - is fully functioning nonlocal game environment where Genetic algorithm optimizes input rotation gates to learn the best possible way to maximize win_accuracy,CHSH value. (could be set also for complex gates) Works best for small games where each has 1 qubit.

* NlgContinuousOptimalization - environment uses NlgGeneticOptimalization to optimize gates that the agent chose each step globally. (very slow)

I built two Reinforcement learning agents to search these environments:

* Basic REGRESSION agent

* DQN agent

* For games with only one qubit per player, one can use another approach - optimalizing already chosen gates through genetic algortihm. 

On top of those I built Genetic algorithm that is able to optimalize agents hyperparameters (and also choose the best reward function)

#### dev
* clone this repository into your developer folder

* open src folder in your favourite IDE (works with PyCharm)

* install missing libraries - tensorflow, keras, PyTorch, numpy, pandas, scikit-learn, qiskit

* find docummentation at > src/doc

Find results for all 2-player, 2-question games with 1-epr pair at  > src/non_local_games_evaluated.html

# Acknowledgments

We are thankful to Daniel Nagaj for supervising this bachelor thesis, introducing us to the quantum computing and for useful discussions. We are also thankful to
Pavel Petrovic for useful discussion about the RL. 

