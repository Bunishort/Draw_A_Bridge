This project aims at creating a bridge building simulation game, similar to World Of Goo, but with science.
It is currently at a very early stage.

To try the game, launch tests/test_pygame.py. Click to draw a bridge or anything, and press the space bar to start the simulation. You can alternate between draw mode and simulation mode using the space bar. 

The homemade engine is a kind of explicit Finite Element Method in plane strain (so, 2 dimensions only). It simulates a linear viscoelastic material in small strain/ small displacements, with inertia. It makes use of Numpy and ModernGL libraries for the engine, most of the computing being done on GPU. Pygame and ModernGL are used for the interface and rendering.

Please reach out if you want to contribute, I really do not know how to manage a public open source repository.

Have fun.
