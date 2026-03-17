# Create a Foosball simulation

i want to make a foosball simulation to start, and then scale that up; i have watched some youtube on monte carlo simulations, but haven't myself implemented any. I am comfortable with Python and programming. I would like help implementing and learning throughout this project. 

## Objectives

- Determine the best foosball strategy
- Determine the scoring probability depending on the position
- Be like a chess bot but for foosball
- Model the game with probabilities of actions (not rely on physics)
    - Collision and bounding will help though

## Statistical / probability model

- ball direction/speed after hit, and interception chance vary (e.g. based on normal distribution)
- intended direction may initially be the goal, but with more advanced strategy there will be passing and dribbling
- probabilities offer abstraction because not every degree of freedom is necessary
- each hit has a certain probability of passing (i.e. intersection or overlapping) position (x, y) (could also help to include speed at that position)
    - if it passes through a position, 
- probabilities are determined from data (through simulation) or optionally by inputting probabilities (i.e. if this player is good at this type of shot)

## Gameplay simulation (simplifications) and evolution

- more skill leads to higher percentage shots (shots more likely to go where you want it)
- more consistency leads to a thinner distribution curve (less randomness)
- different models play against each other many times
- add randomness and noise to prevent 'overfitting' and model nonideality
- 2D field (x axis is goal to goal, y axis is parallel to rod direction): each player has some x distance in front and behind them where they can be and where they can hit

## Rules and simplifications

- each point is independent: winning the point means winning the game (run multiple rounds of the same matchup to get an average)
- 4 hands: you can control any player at any time, no switching hands required
- full field vision: you can see and focus on the entire field
- turn-based, simultaneous decision (simulates the defender anticipating)
- state: ball position, speed, which rod has possession, position (x, y) of each rod
- each side has 3 goalies (instead of the usual 1 goalie), being able to reach the field corners so that we can ignore edge and corner ramps, keeping it 2D
- collision detection: ball bounces off walls and players (including players can kick diagonally)

## Visualization

- basic foosball field in bird's eye view
- all players on the same rod must have equal x and 
- highlight occupiable space in front / behind each player (toggleable option)
- heatmap of expected ball position after hit (toggleable option)
- display strategy strength and weaknesses

## Notes

- Values should be parameterized where possible
- Goal is for this to be scalable to other sports like soccer and basketball, and maybe even life
- Next level is improvement (also just evolution) - which area you choose to improve will have the greatest impact on team success

## Foosball is a great starting point

* Fully observable — you can see everything
* Discrete-ish — rods can rotate/slide, relatively constrained
* Small state space — ~8 rods, ~22 players (fixed positions per rod), 1 ball
* Fast outcomes — goals happen frequently, so learning signal is dense (unlike soccer's 2.7 goals/90min)
* Scales conceptually — Monte Carlo → rule-based AI → RL → self-play is a natural progression

## Build order

1. Data structures        — Field, Rod, Player, BallState
2. Single hit function    — sample (angle, speed) from normal dist given skill params
3. Interception check     — does ball pass through rod coverage at that x?
4. Point simulation       — chain hits until goal or out-of-bounds
5. Monte Carlo runner     — run N points, return win probability
6. Strategy interface     — agents choose rod y-positions and target angles
7. Strategy comparison    — run Strategy A vs Strategy B, compare win rates