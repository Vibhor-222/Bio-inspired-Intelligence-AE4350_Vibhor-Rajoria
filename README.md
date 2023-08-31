# Bio-inspired-Intelligence-AE4350_Vibhor-Rajoria

ABOUT:

The Lunar Lander simulation replicates the challenge of safely landing a miniature rocket on
the lunar terrain. The environment used to evaluate the algorithm is accessible through the
Gymnasium platform created by Oleg Klimov. 

Within this simulation, the spacecraft is equipped with a primary engine and dual lateral boosters, allowing
precise control over its descent and orientation. The spacecraft operates under the gravitational
influence of the moon, and its engines possess an infinite fuel supply. The primary goal is to
navigate the spacecraft to the designated landing area located between two flags at coordinates
(0,0), all while averting potential collisions. It’s important to note that landing beyond the de-
fined landing pad is a feasible outcome. The simulation commences with the lander positioned
at the upper central part of the viewport, having been subjected to an initial random force
applied to its center of mass.

The state comprises an 8-dimensional vector, as listed in the table. Following each step in the simulation,
a corresponding reward is given. The total reward of an episode is the sum of the rewards for
all the steps within that episode. An episode is considered a solution if it scores at least 200
points.
