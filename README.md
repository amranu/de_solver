# PDE Solver

This repository contains an interactive 2‑D partial differential equation (PDE) solver written in Python. The application provides a PyQt5 interface with an OpenGL view for visualizing solutions to several PDEs in real time.

## Supported equations
- Heat equation
- Schrödinger equation (explicit and Crank–Nicolson schemes)
- Wave equation
- Burgers' equation
- Navier–Stokes equations
- Reaction–diffusion system

## Repository layout
- `main.py` – launches the Qt application.
- `pde_solver/gui.py` – implementation of the GUI, OpenGL widget and control panel.
- `pde_solver/solvers.py` – numerical update routines, many accelerated with Numba.
- `pde_solver/__init__.py` – exposes the window class and solver functions.

## Installation
The solver requires Python 3 with the following packages:

```
PyQt5
numpy
scipy
PyOpenGL
numba
```

Install them using `pip`:

```bash
pip install PyQt5 numpy scipy PyOpenGL numba
```

## Running the application
Execute the main script:

```bash
python main.py
```

## Usage
The main window shows a grid where the chosen equation is solved. Use the control panel on the right to:

- Select the PDE and boundary conditions.
- Change parameters such as diffusion coefficients, wave speed, viscosity and time step.
- Start, pause or reset the simulation.
- Save and load parameter configurations (stored as JSON files).

Initial conditions can be drawn directly on the grid with the left mouse button. Use the middle mouse button to pan and the mouse wheel to zoom.

### Adaptive time stepping
For the explicit Schrödinger solver the application adjusts the time step automatically based on the simulation stability. The current value is displayed at the bottom of the control panel.

## License
This repository does not specify a license. Please consult the repository owner if you wish to use the code.
