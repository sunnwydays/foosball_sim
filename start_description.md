# Create a Foosball simulation

i want to make a foosball simulation to start, and then scale that up; i have watched some youtube on monte carlo simulations, but haven't myself implemented any. I am comfortable with Python and programming. I would like help implementing and learning throughout this project. 

## Objectives

- Determine the best foosball strategy
- Determine the scoring probability depending on the position
- Be like a chess bot but for foosball
- Model the game with probabilities of actions (not rely on physics)
    - Collision and bounding will help though

## Statistical / probability model

- direction & speed vary based on normal distribution
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

## Simplifications

- each point is independent: winning the point means winning the game (run multiple rounds of the same matchup to get an average)
- 4 hands: you can control any player at any time, no switching hands required
- full field vision: you can see and focus on the entire field

## Visualization

- basic foosball field in bird's eye view
- 3 goalie version so that we can ignore edge and corner ramps, keeping it 2D
- all players on the same rod must have equal x and 
- highlight occupiable space in front / behind each player (toggleable option)
- heatmap of expected ball position after hit (toggleable option)

## Notes

- Values should be parameterized where possible
- Goal is for this to be scalable to other sports like soccer and basketball, and maybe even life
- Next level is improvement (also just evolution) - which area you choose to improve will have the greatest impact on team success