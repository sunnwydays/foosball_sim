from perspective of a person, 

you can move and swing however you like

usually based on ball position / velo and opponent positions, and your own rod's pos

give a delta or and absolute? absolute.

---

full physics (swing and trajectory must follow line made by player and ball) - as much freedom as real life. implement prediction by literally rotating 

vs.

swing simplification:
- hit any direction - would also scale up (real life soccer players can hit any direction, i guess you can read them in real life too but too complicated for now)
- much simpler and faster
- uses reaction delay
- implement prediction by simulating swing based on estimated velo and dir, and then apply result velo if contact with any player on rod 


do you agree that keeping swing simplification and allowing players to make a premove at any time makes sense? 
IRL when you try to premove, you have a somewhat intended velocity and direction, which we can match with our skill genes
- if ball is within the rod's reach at any time during the swing phase (assume swing takes a constant amount of time), then it will be equivalent to the deflecting player reacting instantly and hitting the ball. however of course they are not reacting instantly, they are predicting - they made their intention of where to hit the ball prior to the ball making contact
- however, then there is no disadvantage to constantly predicting
- makes sense as pro players will often quickly rotate their players within a small angle so that if the ball makes contact, it will deflect forward
- but on the backward swing of that, then there should be a delay (ball passively bounces off the player if makes contact) equivalent to the amount of time they committed into their swing premove? and we can't allow the swing premove to go on forever, it should be fixed

another question would be possibly horizontal predictions, but that might already be possible (choose position doesn't depend on ball)