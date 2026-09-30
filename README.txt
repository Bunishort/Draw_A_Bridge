This project aims at creating a bridge building simulation game, similar to World Of Goo, but different.
It is currently at a very early stage.

To try the game, you can use the precompiled versions in the Release tab, or download the source code and launch tests/test_pygame.py. Click to draw a bridge or anything, and press the space bar to start the simulation. Click on the buttons from the button bar below and see what happens. Click on the screen during simulation to attract the pixels to your cursor.

The homemade engine is a kind of explicit Finite Element Method in plane strain (so, 2 dimensions only). It simulates a linear viscoelastic material in small strain/ small displacements, with inertia. It makes use of Numpy and ModernGL libraries for the engine, most of the computing being done on GPU. Pygame and ModernGL are used for the interface and rendering.

Have fun.
