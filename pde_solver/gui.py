# Import necessary modules
import sys
import numpy as np
import json
import logging  # Added for debugging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QOpenGLWidget, QWidget, QHBoxLayout,
    QVBoxLayout, QLabel, QSlider, QPushButton, QDoubleSpinBox, QComboBox, QFileDialog, QSpinBox
)
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QColor
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.sparse import diags, identity, kron, csc_matrix
from scipy.sparse.linalg import spsolve

from .solvers import heat_equation_update, schrodinger_equation_update, schrodinger_equation_cn_update, wave_equation_update, burgers_equation_update, navier_stokes_update, reaction_diffusion_update
# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
PI = np.pi


# Define the main window class
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('2D PDE Solver with Adaptive Time Stepping')
        self.setGeometry(100, 100, 1500, 800)  # Increased width for better layout

        # Create central widget and set layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Create OpenGL widget
        self.glWidget = GLWidget(self)
        main_layout.addWidget(self.glWidget, stretch=3)

        # Create control panel with color scale
        self.controlPanel = ControlPanel(self.glWidget, self)
        main_layout.addWidget(self.controlPanel, stretch=1)

    def closeEvent(self, event):
        # Handle any cleanup if necessary
        logging.info('Application is closing.')
        # Stop the simulation timer if running
        self.glWidget.simulation_timer.stop()
        event.accept()


# Define the OpenGL widget class
class GLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super(GLWidget, self).__init__(parent)
        self.grid_size = 100
        self.alpha = 0.01  # Thermal diffusivity constant for Heat Equation
        self.c = 1.0       # Wave speed for Wave Equation
        self.viscosity = 0.1  # Viscosity for Burgers' and Navier-Stokes Equation
        self.Du = 0.16     # Diffusion coefficient for u in Reaction-Diffusion
        self.Dv = 0.08     # Diffusion coefficient for v in Reaction-Diffusion
        self.feed_rate = 0.035  # Feed rate for Reaction-Diffusion
        self.kill_rate = 0.065  # Kill rate for Reaction-Diffusion
        self.dx = 1.0 / self.grid_size
        self.dy = 1.0 / self.grid_size
        self.dt = 0.0001  # Initial Time step
        self.equation = 'Heat Equation'  # Default equation
        self.is_running = False
        self.is_paused = False
        self.boundary_condition = 'Dirichlet'  # Default boundary condition

        self.schrodinger_method = 'Explicit'  # Default method for Schrödinger Equation

        self.nit = 50         # Number of iterations for pressure Poisson equation
        self.rho = 1.0        # Density
        self.Fx = 0.0         # External force in x-direction
        self.Fy = 0.0         # External force in y-direction
        self.frequency = 1.0  # Frequency for oscillating external force
        self.time = 0.0       # Initialize time

        # Adaptive Time Stepping Variables
        self.min_dt = 1e-6  # Minimum allowed dt
        self.max_dt = 0.01   # Maximum allowed dt
        self.adaptive_dt = self.dt  # Current adaptive dt
        self.dt_adjustment_factor = 0.5  # Factor to adjust dt
        self.tolerance = 1e-3  # Tolerance for max change
        self.previous_psi = None  # To store previous psi for comparison
        # Cached matrices for Crank-Nicolson Schrödinger solver
        self.cn_A = None
        self.cn_B = None

        # Mouse interaction variables
        self.drawing = False
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.last_mouse_pos = None
        self.last_drawing_pos = None  # Added for drawing direction

        # Initialize data structures
        self.initialize_data()

        # Reference to control panel
        self.controlPanel = None

        # Create a simulation timer
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.update_simulation)

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Set background color
        glEnable(GL_DEPTH_TEST)
        self.update_projection()

    def resizeGL(self, w, h):
        # Set the viewport to cover the new window size
        glViewport(0, 0, w, h)
        self.update_projection()

    def update_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        zoom_factor = self.zoom
        pan_x = self.pan.x()
        pan_y = self.pan.y()
        gluOrtho2D(pan_x, pan_x + self.grid_size * zoom_factor,
                   pan_y, pan_y + self.grid_size * zoom_factor)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        self.draw_grid()

    def draw_grid(self):
        glBegin(GL_QUADS)
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size - 1):
                if self.equation == 'Heat Equation':
                    # Map temperature to color
                    temp = self.u[i, j]
                    color = self.temperature_to_color(temp)
                elif self.equation == 'Schrödinger Equation':
                    # Map probability density to color
                    prob_density = np.abs(self.psi[i, j]) ** 2
                    # Update max_prob_density for normalization
                    if prob_density > self.max_prob_density:
                        self.max_prob_density = prob_density
                    color = self.probability_to_color(prob_density)
                elif self.equation == 'Wave Equation':
                    # Map displacement to color
                    displacement = self.u_current[i, j]
                    color = self.displacement_to_color(displacement)
                elif self.equation == "Burgers' Equation":
                    # Visualize the magnitude of the velocity
                    velocity_magnitude = np.sqrt(self.u[i, j] ** 2 + self.v[i, j] ** 2)
                    color = self.velocity_to_color(velocity_magnitude)
                elif self.equation == 'Navier-Stokes':
                    # Visualize the magnitude of the velocity
                    velocity_magnitude = np.sqrt(self.u[i, j] ** 2 + self.v[i, j] ** 2)
                    color = self.velocity_to_color(velocity_magnitude)
                elif self.equation == 'Reaction-Diffusion':
                    # Map concentration of v to color
                    conc = self.v[i, j]
                    color = self.concentration_to_color(conc)
                glColor3f(*color)
                # Draw the cell
                x = i
                y = j
                glVertex2f(x, y)
                glVertex2f(x + 1, y)
                glVertex2f(x + 1, y + 1)
                glVertex2f(x, y + 1)
        glEnd()

        # Draw vector field for Burgers' Equation and Navier-Stokes Equation
        if self.equation == "Burgers' Equation" or self.equation == 'Navier-Stokes':
            self.draw_vector_field()

    def draw_vector_field(self):
        glColor3f(1.0, 1.0, 1.0)  # White color for vectors
        glLineWidth(1.0)
        scale = 0.5  # Adjust the scale of the arrows
        glBegin(GL_LINES)
        skip = max(1, self.grid_size // 20)  # Skip some vectors for clarity
        for i in range(0, self.grid_size, skip):
            for j in range(0, self.grid_size, skip):
                x = i + 0.5
                y = j + 0.5
                u = self.u[i, j] * scale
                v = self.v[i, j] * scale
                glVertex2f(x, y)
                glVertex2f(x + u, y + v)
        glEnd()

    def temperature_to_color(self, temp):
        # Map temperature to RGB color (blue to red)
        # Clamp temperature between 0.0 and 1.0
        temp = np.clip(temp, 0.0, 1.0)
        return (temp, 0.0, 1.0 - temp)

    def probability_to_color(self, prob_density):
        # Map probability density to color using grayscale
        # Normalize prob_density
        normalized = prob_density / self.max_prob_density
        normalized = np.clip(normalized, 0.0, 1.0)
        return (normalized, normalized, normalized)

    def displacement_to_color(self, displacement):
        # Map displacement to RGB color (blue for negative, red for positive)
        # Normalize displacement based on maximum absolute displacement
        max_disp = np.max(np.abs(self.u_current))
        if max_disp == 0:
            max_disp = 1e-10  # Prevent division by zero
        normalized = displacement / max_disp
        normalized = np.clip(normalized, -1.0, 1.0)
        if normalized < 0:
            # Blue to white for negative displacements
            return (0.0, 0.0, 1.0 + normalized)
        else:
            # White to red for positive displacements
            return (normalized, 0.0, 1.0)

    def velocity_to_color(self, velocity):
        # Normalize velocity
        max_velocity = np.max(np.sqrt(self.u ** 2 + self.v ** 2))
        if max_velocity == 0:
            max_velocity = 1e-10  # Prevent division by zero
        normalized = velocity / max_velocity
        normalized = np.clip(normalized, 0.0, 1.0)

        if self.equation == 'Navier-Stokes':
            # Define color stops for Navier-Stokes
            color_stops = [0.0, 0.25, 0.5, 0.75, 1.0]
            colors = [
                (0.0, 0.0, 1.0),    # Blue
                (0.0, 1.0, 1.0),    # Cyan
                (0.0, 1.0, 0.0),    # Green
                (1.0, 1.0, 0.0),    # Yellow
                (1.0, 0.0, 0.0)     # Red
            ]

            # Find the interval in which the normalized velocity falls
            for i in range(len(color_stops) - 1):
                if color_stops[i] <= normalized <= color_stops[i+1]:
                    # Calculate the fraction within the current interval
                    fraction = (normalized - color_stops[i]) / (color_stops[i+1] - color_stops[i])
                    # Linearly interpolate between the two colors
                    r = colors[i][0] + fraction * (colors[i+1][0] - colors[i][0])
                    g = colors[i][1] + fraction * (colors[i+1][1] - colors[i][1])
                    b = colors[i][2] + fraction * (colors[i+1][2] - colors[i][2])
                    break
            else:
                # If normalized velocity is exactly 1.0, assign the last color
                r, g, b = colors[-1]
        else:
            # Original mapping for other equations (blue to red)
            r = normalized
            g = 0.0
            b = 1.0 - normalized

        return (r, g, b)


    def concentration_to_color(self, conc):
        # Map concentration to RGB color (blue to red)
        conc = np.clip(conc, 0.0, 1.0)
        return (conc, 0.0, 1.0 - conc)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_drawing_pos = event.pos()  # Start tracking mouse position
            self.set_initial_condition(event)
        elif event.button() == Qt.MiddleButton:
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing and self.last_drawing_pos is not None:
            self.set_initial_condition(event, prev_event_pos=self.last_drawing_pos)
            self.last_drawing_pos = event.pos()
        elif event.buttons() & Qt.MiddleButton and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            self.pan -= QPointF(delta.x() * self.zoom, -delta.y() * self.zoom)
            self.last_mouse_pos = event.pos()
            self.update_projection()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.last_drawing_pos = None  # Reset the last drawing position
        elif event.button() == Qt.MiddleButton:
            self.last_mouse_pos = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        zoom_factor = 0.9 if delta > 0 else 1.1
        self.zoom *= zoom_factor
        self.zoom = np.clip(self.zoom, 0.1, 10.0)
        self.update_projection()
        self.update()

    def set_initial_condition(self, event, prev_event_pos=None):
        # Get mouse position
        x = event.x()
        y = event.y()
        width = self.width()
        height = self.height()
        # Convert to world coordinates
        x_world = self.pan.x() + (x / width) * self.grid_size * self.zoom
        y_world = self.pan.y() + ((height - y) / height) * self.grid_size * self.zoom
        # Grid indices
        i = int(x_world)
        j = int(y_world)

        if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
            if self.equation == 'Navier-Stokes':
                if prev_event_pos is not None:
                    # Get previous mouse position
                    x_prev = prev_event_pos.x()
                    y_prev = prev_event_pos.y()
                    x_world_prev = self.pan.x() + (x_prev / width) * self.grid_size * self.zoom
                    y_world_prev = self.pan.y() + ((height - y_prev) / height) * self.grid_size * self.zoom
                    # Compute delta
                    delta_x = x_world - x_world_prev
                    delta_y = y_world - y_world_prev
                    norm = np.hypot(delta_x, delta_y)
                    if norm > 0:
                        # Compute normalized injection direction
                        dir_x = delta_x / norm
                        dir_y = delta_y / norm
                        # Compute magnitude (clamped)
                        scaling_factor = 5.0  # Adjust as needed
                        max_magnitude = 5.0   # Adjust as needed
                        magnitude = min(norm * scaling_factor, max_magnitude)
                        # Build a small velocity impulse proportional to dt to avoid blow-up
                        u_impulse = dir_x * magnitude * self.dt
                        v_impulse = dir_y * magnitude * self.dt
                        # Number of points along the line for smooth injection
                        num_points = int(norm * 2) + 1
                        xs = np.linspace(x_world_prev, x_world, num_points)
                        ys = np.linspace(y_world_prev, y_world, num_points)
                        is_ = np.clip(xs.astype(int), 0, self.grid_size - 1)
                        js = np.clip(ys.astype(int), 0, self.grid_size - 1)
                        # Add impulse to velocities along the path
                        for idx in range(len(is_)):
                            i_idx = is_[idx]
                            j_idx = js[idx]
                            self.u[i_idx, j_idx] += u_impulse
                            self.v[i_idx, j_idx] += v_impulse
                        # Redraw to show injected velocity
                        self.update()
                    else:
                        # Zero movement, do nothing
                        pass
                else:
                    # No previous position, can't compute direction
                    pass
            else:
                # Existing code for other equations
                if self.equation == 'Heat Equation':
                    self.u[i, j] = 1.0  # Maximum temperature
                elif self.equation == 'Schrödinger Equation':
                    # Add a Gaussian packet
                    self.add_gaussian_packet(i, j)
                elif self.equation == 'Wave Equation':
                    # Set initial displacement
                    self.u_current[i, j] = 1.0
                    self.u_prev[i, j] = 0.0  # Assuming initial velocity is zero
                elif self.equation == "Burgers' Equation":
                    # Set initial velocity
                    self.u[i, j] = 1.0  # Example value; adjust as needed
                    self.v[i, j] = 0.0
                elif self.equation == 'Reaction-Diffusion':
                    # Introduce a small disturbance
                    radius = 5
                    self.u[max(i - radius, 0):min(i + radius, self.grid_size),
                           max(j - radius, 0):min(j + radius, self.grid_size)] = 0.50
                    self.v[max(i - radius, 0):min(i + radius, self.grid_size),
                           max(j - radius, 0):min(j + radius, self.grid_size)] = 0.25
                self.update()

    def display_warning(self, message):
        self.controlPanel.warning_label.setStyleSheet("color: red;")  # Set text color to red
        self.controlPanel.warning_label.setText(message)

    def display_info(self, message):
        self.controlPanel.warning_label.setStyleSheet("color: black;")  # Set text color to black
        self.controlPanel.warning_label.setText(message)

    def update_simulation(self):
        if self.is_paused:
            return
        try:
            if self.equation == 'Heat Equation':
                self.update_heat_equation()
            elif self.equation == 'Schrödinger Equation':
                if self.schrodinger_method == 'Explicit':
                    # Update using the Explicit method with adaptive_dt
                    self.psi = schrodinger_equation_update(
                        self.psi, self.adaptive_dt, self.dx, self.dy, self.boundary_condition
                    )
                    # Normalize and adjust dt
                    self.normalize_wave_function()
                    self.adaptive_time_step_adjustment()
                    self.controlPanel.current_dt_label.setText(f'Current dt: {self.adaptive_dt:.6f}')
                    # Update previous_psi after dt adjustment
                    self.previous_psi = self.psi.copy()
                elif self.schrodinger_method == 'Crank-Nicolson':
                    # Update using cached Crank-Nicolson matrices
                    psi_flat = self.psi.flatten()
                    psi_flat_new = spsolve(self.cn_A, self.cn_B.dot(psi_flat))
                    self.psi = psi_flat_new.reshape((self.grid_size, self.grid_size))
                    # Normalize to prevent overflow
                    self.normalize_wave_function()
                    self.controlPanel.current_dt_label.setText(f'Current dt: {self.dt:.6f}')
            elif self.equation == 'Wave Equation':
                self.update_wave_equation()
            elif self.equation == "Burgers' Equation":
                self.update_burgers_equation()
            elif self.equation == 'Navier-Stokes':
                self.update_navier_stokes()
            elif self.equation == 'Reaction-Diffusion':
                self.update_reaction_diffusion()
            self.update()
        except Exception as e:
            self.is_running = False
            self.display_warning(f'Error: {str(e)}')

    def update_heat_equation(self):
        # Ensure stability (CFL condition)
        max_dt = (self.dx ** 2) / (4 * self.alpha)
        if self.dt > max_dt:
            self.dt = max_dt
            self.display_warning('Warning: dt adjusted for stability in Heat Equation.')
        else:
            self.display_warning('')

        # Update using Numba-optimized function
        self.u = heat_equation_update(self.u, self.alpha, self.dt, self.dx, self.dy, self.boundary_condition)

    def adaptive_time_step_adjustment(self):
        if self.equation != 'Schrödinger Equation' or self.schrodinger_method != 'Explicit':
            return
        if self.previous_psi is None:
            self.previous_psi = self.psi.copy()
            return
        max_change = np.max(np.abs(self.psi - self.previous_psi))
        if max_change > self.tolerance and self.adaptive_dt > self.min_dt:
            # Decrease dt
            self.adaptive_dt *= self.dt_adjustment_factor
            self.display_warning('Warning: dt decreased for stability.')
            logging.info(f'Dt decreased to {self.adaptive_dt}')
        elif max_change < self.tolerance / 2 and self.adaptive_dt < self.max_dt:
            # Increase dt
            self.adaptive_dt /= self.dt_adjustment_factor
            self.display_info('Info: dt increased for efficiency.')
            logging.info(f'Dt increased to {self.adaptive_dt}')
        else:
            self.display_warning('')

        # Update previous_psi for next comparison
        self.previous_psi = self.psi.copy()

    def normalize_wave_function(self):
        # Normalize the wave function to ensure total probability is 1
        total_prob = np.sum(np.abs(self.psi) ** 2) * self.dx * self.dy
        if total_prob > 0:
            self.psi /= np.sqrt(total_prob)
            self.max_prob_density = np.max(np.abs(self.psi) ** 2)
        else:
            self.max_prob_density = 1e-10  # Prevent division by zero
        if self.equation == 'Schrödinger Equation' and self.schrodinger_method == 'Explicit':
            self.previous_psi = self.psi.copy()

    def update_wave_equation(self):
        # Ensure stability (CFL condition for wave equation)
        max_dt = self.dx / self.c / np.sqrt(2)
        if self.dt > max_dt:
            self.dt = max_dt
            self.display_warning('Warning: dt adjusted for stability in Wave Equation.')
        else:
            self.display_warning('')

        # Update using Numba-optimized function
        self.u_prev, self.u_current = wave_equation_update(self.u_prev, self.u_current, self.c, self.dt, self.dx, self.dy, self.boundary_condition)

    def update_burgers_equation(self):
        # Update using Numba-optimized function
        self.u, self.v = burgers_equation_update(self.u, self.v, self.viscosity, self.dt, self.dx, self.dy)

    def update_navier_stokes(self):
        # Stability check (CFL condition and diffusion limit)
        try:
            u_max = np.max(np.abs(self.u))
            v_max = np.max(np.abs(self.v))
            cfl_dt = min(self.dx / (u_max + 1e-8), self.dy / (v_max + 1e-8))
            diff_dt = min(self.dx ** 2, self.dy ** 2) / (4 * self.viscosity + 1e-8)
            max_dt = min(cfl_dt, diff_dt)
            if self.dt > max_dt:
                self.dt = max_dt
                self.display_warning('Warning: dt adjusted for stability in Navier-Stokes.')
            else:
                self.display_warning('')
            self.controlPanel.current_dt_label.setText(f'Current dt: {self.dt:.6f}')
        except Exception:
            pass
        # Update time
        self.time += self.dt
        # Apply oscillating external forces if frequency is set
        if self.Fx != 0.0:
            Fx_time = self.Fx * np.sin(2 * np.pi * self.frequency * self.time)
        else:
            Fx_time = 0.0
        if self.Fy != 0.0:
            Fy_time = self.Fy * np.sin(2 * np.pi * self.frequency * self.time)
        else:
            Fy_time = 0.0
        # Update using Numba-optimized function
        self.u, self.v, self.p = navier_stokes_update(
            self.u, self.v, self.p, self.b, self.viscosity, self.dt, self.dx, self.dy, self.boundary_condition, self.nit, self.rho, Fx_time, Fy_time)

    def update_reaction_diffusion(self):
        # Update using Numba-optimized function
        self.u, self.v = reaction_diffusion_update(self.u, self.v, self.Du, self.Dv, self.feed_rate, self.kill_rate, self.dt, self.dx, self.dy)

    def add_gaussian_packet(self, x0, y0, sigma=2.0):
        X, Y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        gauss_packet = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2))
        self.psi += gauss_packet.astype(np.complex128)
        self.normalize_wave_function()  # Normalize after adding a packet
        if self.schrodinger_method == 'Explicit':
            self.previous_psi = self.psi.copy()

    def start_simulation(self):
        logging.info('Simulation started.')
        self.is_running = True
        self.is_paused = False
        self.simulation_timer.start(30)  # Timer interval in ms

    def pause_simulation(self):
        logging.info('Simulation paused.')
        self.is_paused = True
        self.simulation_timer.stop()

    def reset_simulation(self):
        # Stop the simulation timer if running
        self.simulation_timer.stop()
        self.is_running = False
        self.initialize_data()
        if self.equation == 'Schrödinger Equation' and self.schrodinger_method == 'Explicit':
            self.initialize_adaptive_time_step()
        elif self.equation == 'Schrödinger Equation' and self.schrodinger_method == 'Crank-Nicolson':
            self.initialize_schrodinger_cn()
        self.update()
        self.controlPanel.update_current_dt_display()

    def initialize_data(self):
        # Initialize data structures based on the selected equation
        if self.equation == 'Heat Equation':
            self.u = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)  # Use float64 for Numba compatibility
        elif self.equation == 'Schrödinger Equation':
            self.psi = np.zeros((self.grid_size, self.grid_size), dtype=np.complex128)  # Wave function
            self.max_prob_density = 1e-10  # For normalization in visualization
            if self.schrodinger_method == 'Explicit':
                self.adaptive_dt = self.dt
                self.previous_psi = self.psi.copy()
        elif self.equation == 'Wave Equation':
            # For Wave Equation, we need to store previous and current states
            self.u_prev = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)  # Previous displacement
            self.u_current = np.zeros_like(self.u_prev)  # Current displacement
        elif self.equation == "Burgers' Equation":
            self.u = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
            self.v = np.zeros_like(self.u)
        elif self.equation == 'Navier-Stokes':
            self.u = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
            self.v = np.zeros_like(self.u)
            self.p = np.zeros_like(self.u)
            self.b = np.zeros_like(self.u)
        elif self.equation == 'Reaction-Diffusion':
            # Initialize concentrations with small random noise
            self.u = np.ones((self.grid_size, self.grid_size), dtype=np.float64)
            self.v = np.zeros_like(self.u)
            self.u += 0.01 * np.random.randn(self.grid_size, self.grid_size)
            self.v += 0.01 * np.random.randn(self.grid_size, self.grid_size)

    def initialize_adaptive_time_step(self):
        self.adaptive_dt = self.dt
        if self.equation == 'Schrödinger Equation' and self.schrodinger_method == 'Explicit':
            self.previous_psi = self.psi.copy()

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_c(self, c):
        self.c = c

    def set_dt(self, dt):
        self.dt = dt
        if self.equation == 'Schrödinger Equation' and self.schrodinger_method == 'Explicit':
            self.adaptive_dt = dt
        logging.info(f'Dt set to {self.dt}')

    def set_viscosity(self, viscosity):
        self.viscosity = viscosity

    def set_Du(self, Du):
        self.Du = Du

    def set_Dv(self, Dv):
        self.Dv = Dv

    def set_feed_rate(self, feed_rate):
        self.feed_rate = feed_rate

    def set_kill_rate(self, kill_rate):
        self.kill_rate = kill_rate

    def set_equation(self, equation):
        self.equation = equation
        self.reset_simulation()
        # Update color scale based on the selected equation
        self.controlPanel.color_scale.set_equation(equation)

    def set_boundary_condition(self, condition):
        self.boundary_condition = condition
        self.reset_simulation()

    def set_schrodinger_method(self, method):
        self.schrodinger_method = method
        if method == 'Explicit':
            self.initialize_adaptive_time_step()
        logging.info(f'Schrödinger method set to {method}')
        self.reset_simulation()
    def initialize_schrodinger_cn(self):
        """Build and cache matrices for Crank-Nicolson Schrödinger solver."""
        Nx = self.grid_size
        Ny = self.grid_size
        N = Nx * Ny
        hbar = 1.0
        m = 1.0
        dx = self.dx
        dy = self.dy
        dt = self.dt
        r_x = 1j * hbar * dt / (4 * m * dx ** 2)
        r_y = 1j * hbar * dt / (4 * m * dy ** 2)
        # Build 1D Laplacian with boundary conditions
        e = np.ones(Nx)
        diagonals = [-2 * e, e[:-1], e[:-1]]
        L1D = diags(diagonals, [0, -1, 1], shape=(Nx, Nx), format='csc')
        if self.boundary_condition == 'Neumann':
            arr = L1D.toarray()
            arr[0, 0] = -1
            arr[-1, -1] = -1
            L1D = csc_matrix(arr)
        elif self.boundary_condition == 'Periodic':
            arr = L1D.toarray()
            arr[0, -1] = 1
            arr[-1, 0] = 1
            L1D = csc_matrix(arr)
        # Build 2D Laplacian
        I = identity(Nx, format='csc')
        Lx = kron(I, L1D, format='csc')
        Ly = kron(L1D, I, format='csc')
        Lap = Lx + Ly
        # Construct A and B matrices
        A = identity(N, dtype=complex, format='csc') - (r_x + r_y) * Lap
        B = identity(N, dtype=complex, format='csc') + (r_x + r_y) * Lap
        # Cache for use in update
        self.cn_A = A
        self.cn_B = B

    def set_rho(self, rho):
        self.rho = rho

    def set_nit(self, nit):
        self.nit = nit

    def set_Fx(self, Fx):
        self.Fx = Fx

    def set_Fy(self, Fy):
        self.Fy = Fy

    def set_frequency(self, frequency):
        self.frequency = frequency

# Numba-optimized functions


# Define the Control Panel class
class ControlPanel(QWidget):
    def __init__(self, glWidget, parent=None):
        super(ControlPanel, self).__init__(parent)
        self.glWidget = glWidget
        self.initUI()
        self.glWidget.controlPanel = self  # Reference for accessing control panel elements

    def initUI(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Warning Label for user feedback
        self.warning_label = QLabel('')
        self.warning_label.setStyleSheet("color: red;")  # Make warnings red
        self.main_layout.addWidget(self.warning_label)

        # Equation Selection
        equation_label = QLabel('Select Equation:')
        self.main_layout.addWidget(equation_label)

        self.equation_combo = QComboBox()
        self.equation_combo.addItems([
            'Heat Equation',
            'Schrödinger Equation',
            'Wave Equation',
            "Burgers' Equation",
            'Reaction-Diffusion',
            'Navier-Stokes'
        ])
        self.equation_combo.currentTextChanged.connect(self.equation_changed)
        self.equation_combo.setToolTip('Choose between different PDEs.')
        self.main_layout.addWidget(self.equation_combo)

        # Boundary Condition Selection
        boundary_label = QLabel('Select Boundary Condition:')
        self.main_layout.addWidget(boundary_label)

        self.boundary_combo = QComboBox()
        self.boundary_combo.addItems(['Dirichlet', 'Neumann', 'Periodic'])
        self.boundary_combo.currentTextChanged.connect(self.boundary_changed)
        self.boundary_combo.setToolTip('Choose the boundary condition for the simulation.')
        self.main_layout.addWidget(self.boundary_combo)

        # Create control panes
        self.create_heat_controls()
        self.create_wave_controls()
        self.create_viscosity_controls()
        self.create_reaction_diffusion_controls()
        self.create_schrodinger_controls()
        self.create_dt_controls()
        self.create_simulation_controls()
        self.create_navier_stokes_controls()  # Added for Navier-Stokes parameters

        # Save and Load Buttons
        self.create_save_load_controls()

        # Spacer to push widgets to the top
        self.main_layout.addStretch()

        # Add current dt display
        self.current_dt_label = QLabel(f'Current dt: {self.glWidget.dt:.6f}')
        self.main_layout.addWidget(self.current_dt_label)

        # Add Color Scale
        color_scale_label = QLabel('Color Scale:')
        self.main_layout.addWidget(color_scale_label)

        self.color_scale = ColorScaleWidget()
        self.main_layout.addWidget(self.color_scale)

        # Initially show controls relevant to the default equation
        self.update_controls_visibility('Heat Equation')

    def equation_changed(self, text):
        self.glWidget.set_equation(text)
        self.update_controls_visibility(text)

    def boundary_changed(self, text):
        self.glWidget.set_boundary_condition(text)

    def update_controls_visibility(self, equation):
        # Remove all control panes
        self.remove_control_panes()

        # Add control panes relevant to the selected equation
        if equation == 'Heat Equation':
            self.main_layout.insertWidget(5, self.heat_controls)
        elif equation == 'Wave Equation':
            self.main_layout.insertWidget(5, self.wave_controls)
        elif equation == "Burgers' Equation":
            self.main_layout.insertWidget(5, self.viscosity_controls)
        elif equation == 'Navier-Stokes':
            self.main_layout.insertWidget(5, self.viscosity_controls)
            self.main_layout.insertWidget(6, self.navier_stokes_controls)
        elif equation == 'Reaction-Diffusion':
            self.main_layout.insertWidget(5, self.reaction_diffusion_controls)
        elif equation == 'Schrödinger Equation':
            self.main_layout.insertWidget(5, self.schrodinger_controls)
        # Add dt controls and simulation controls after the option panes
        self.main_layout.insertWidget(7, self.dt_controls)
        self.main_layout.insertWidget(8, self.simulation_controls)
        self.main_layout.insertWidget(9, self.save_load_controls)
        # Update color scale
        self.color_scale.set_equation(equation)

    def remove_control_panes(self):
        for pane in [self.heat_controls, self.wave_controls, self.viscosity_controls,
                     self.reaction_diffusion_controls, self.schrodinger_controls,
                     self.navier_stokes_controls,  # Include Navier-Stokes controls
                     self.dt_controls, self.simulation_controls, self.save_load_controls]:
            self.main_layout.removeWidget(pane)
            pane.setParent(None)

    def create_heat_controls(self):
        self.heat_controls = QWidget()
        layout = QVBoxLayout()
        self.heat_controls.setLayout(layout)

        # Thermal Diffusivity (alpha) Control
        alpha_label = QLabel('Thermal Diffusivity (α):')
        layout.addWidget(alpha_label)

        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(1)    # Represents 0.01
        self.alpha_slider.setMaximum(100)  # Represents 1.00
        self.alpha_slider.setValue(int(self.glWidget.alpha * 100))
        self.alpha_slider.valueChanged.connect(self.alpha_changed)
        self.alpha_slider.setToolTip('Adjust the thermal diffusivity (α) for the Heat Equation.')
        layout.addWidget(self.alpha_slider)

        self.alpha_spinbox = QDoubleSpinBox()
        self.alpha_spinbox.setDecimals(3)
        self.alpha_spinbox.setRange(0.01, 1.0)
        self.alpha_spinbox.setSingleStep(0.01)
        self.alpha_spinbox.setValue(self.glWidget.alpha)
        self.alpha_spinbox.valueChanged.connect(self.alpha_spinbox_changed)
        self.alpha_spinbox.setToolTip('Set the thermal diffusivity (α) value.')
        layout.addWidget(self.alpha_spinbox)

    def create_wave_controls(self):
        self.wave_controls = QWidget()
        layout = QVBoxLayout()
        self.wave_controls.setLayout(layout)

        # Wave Speed (c) Control for Wave Equation
        wave_label = QLabel('Wave Speed (c):')
        layout.addWidget(wave_label)

        self.wave_slider = QSlider(Qt.Horizontal)
        self.wave_slider.setMinimum(1)    # Represents 0.1
        self.wave_slider.setMaximum(200)  # Represents 20.0
        self.wave_slider.setValue(int(self.glWidget.c * 10))
        self.wave_slider.valueChanged.connect(self.wave_changed)
        self.wave_slider.setToolTip('Adjust the wave speed (c) for the Wave Equation.')
        layout.addWidget(self.wave_slider)

        self.wave_spinbox = QDoubleSpinBox()
        self.wave_spinbox.setDecimals(1)
        self.wave_spinbox.setRange(0.1, 20.0)
        self.wave_spinbox.setSingleStep(0.1)
        self.wave_spinbox.setValue(self.glWidget.c)
        self.wave_spinbox.valueChanged.connect(self.wave_spinbox_changed)
        self.wave_spinbox.setToolTip('Set the wave speed (c) value.')
        layout.addWidget(self.wave_spinbox)

    def create_viscosity_controls(self):
        self.viscosity_controls = QWidget()
        layout = QVBoxLayout()
        self.viscosity_controls.setLayout(layout)

        # Viscosity (nu) Control for Burgers' Equation and Navier-Stokes
        viscosity_label = QLabel('Viscosity (ν):')
        layout.addWidget(viscosity_label)

        self.viscosity_slider = QSlider(Qt.Horizontal)
        self.viscosity_slider.setMinimum(1)    # Represents 0.01
        self.viscosity_slider.setMaximum(100)  # Represents 1.00
        self.viscosity_slider.setValue(int(self.glWidget.viscosity * 100))
        self.viscosity_slider.valueChanged.connect(self.viscosity_changed)
        self.viscosity_slider.setToolTip('Adjust the viscosity (ν) for the equation.')
        layout.addWidget(self.viscosity_slider)

        self.viscosity_spinbox = QDoubleSpinBox()
        self.viscosity_spinbox.setDecimals(3)
        self.viscosity_spinbox.setRange(0.01, 1.0)
        self.viscosity_spinbox.setSingleStep(0.01)
        self.viscosity_spinbox.setValue(self.glWidget.viscosity)
        self.viscosity_spinbox.valueChanged.connect(self.viscosity_spinbox_changed)
        self.viscosity_spinbox.setToolTip('Set the viscosity (ν) value.')
        layout.addWidget(self.viscosity_spinbox)

    def create_navier_stokes_controls(self):
        self.navier_stokes_controls = QWidget()
        layout = QVBoxLayout()
        self.navier_stokes_controls.setLayout(layout)

        # Density (rho)
        rho_label = QLabel('Density (ρ):')
        layout.addWidget(rho_label)

        self.rho_spinbox = QDoubleSpinBox()
        self.rho_spinbox.setDecimals(2)
        self.rho_spinbox.setRange(0.1, 10.0)
        self.rho_spinbox.setSingleStep(0.1)
        self.rho_spinbox.setValue(self.glWidget.rho)
        self.rho_spinbox.valueChanged.connect(self.rho_changed)
        self.rho_spinbox.setToolTip('Set the fluid density (ρ) value.')
        layout.addWidget(self.rho_spinbox)

        # Number of iterations (nit)
        nit_label = QLabel('Pressure Solver Iterations:')
        layout.addWidget(nit_label)

        self.nit_spinbox = QSpinBox()
        self.nit_spinbox.setRange(10, 500)
        self.nit_spinbox.setValue(self.glWidget.nit)
        self.nit_spinbox.valueChanged.connect(self.nit_changed)
        self.nit_spinbox.setToolTip('Set the number of iterations for the pressure Poisson solver.')
        layout.addWidget(self.nit_spinbox)

        # External Force Fx
        fx_label = QLabel('External Force Fx:')
        layout.addWidget(fx_label)

        self.fx_spinbox = QDoubleSpinBox()
        self.fx_spinbox.setDecimals(3)
        self.fx_spinbox.setRange(-10.0, 10.0)
        self.fx_spinbox.setSingleStep(0.1)
        self.fx_spinbox.setValue(self.glWidget.Fx)
        self.fx_spinbox.valueChanged.connect(self.fx_changed)
        self.fx_spinbox.setToolTip('Set the external force in x-direction (Fx).')
        layout.addWidget(self.fx_spinbox)

        # External Force Fy
        fy_label = QLabel('External Force Fy:')
        layout.addWidget(fy_label)

        self.fy_spinbox = QDoubleSpinBox()
        self.fy_spinbox.setDecimals(3)
        self.fy_spinbox.setRange(-10.0, 10.0)
        self.fy_spinbox.setSingleStep(0.1)
        self.fy_spinbox.setValue(self.glWidget.Fy)
        self.fy_spinbox.valueChanged.connect(self.fy_changed)
        self.fy_spinbox.setToolTip('Set the external force in y-direction (Fy).')
        layout.addWidget(self.fy_spinbox)

        # External Force Frequency
        frequency_label = QLabel('External Force Frequency:')
        layout.addWidget(frequency_label)

        self.frequency_spinbox = QDoubleSpinBox()
        self.frequency_spinbox.setDecimals(2)
        self.frequency_spinbox.setRange(0.1, 10.0)
        self.frequency_spinbox.setSingleStep(0.1)
        self.frequency_spinbox.setValue(self.glWidget.frequency)
        self.frequency_spinbox.valueChanged.connect(self.frequency_changed)
        self.frequency_spinbox.setToolTip('Set the frequency for the oscillating external force.')
        layout.addWidget(self.frequency_spinbox)

    def create_reaction_diffusion_controls(self):
        self.reaction_diffusion_controls = QWidget()
        layout = QVBoxLayout()
        self.reaction_diffusion_controls.setLayout(layout)

        # Diffusion Coefficient Du
        Du_label = QLabel('Diffusion Coefficient (D₁):')
        layout.addWidget(Du_label)

        self.Du_slider = QSlider(Qt.Horizontal)
        self.Du_slider.setMinimum(1)    # Represents 0.01
        self.Du_slider.setMaximum(100)  # Represents 1.00
        self.Du_slider.setValue(int(self.glWidget.Du * 100))
        self.Du_slider.valueChanged.connect(self.Du_changed)
        self.Du_slider.setToolTip('Adjust the diffusion coefficient D₁.')
        layout.addWidget(self.Du_slider)

        self.Du_spinbox = QDoubleSpinBox()
        self.Du_spinbox.setDecimals(3)
        self.Du_spinbox.setRange(0.01, 1.0)
        self.Du_spinbox.setSingleStep(0.01)
        self.Du_spinbox.setValue(self.glWidget.Du)
        self.Du_spinbox.valueChanged.connect(self.Du_spinbox_changed)
        self.Du_spinbox.setToolTip('Set the diffusion coefficient D₁ value.')
        layout.addWidget(self.Du_spinbox)

        # Diffusion Coefficient Dv
        Dv_label = QLabel('Diffusion Coefficient (D₂):')
        layout.addWidget(Dv_label)

        self.Dv_slider = QSlider(Qt.Horizontal)
        self.Dv_slider.setMinimum(1)    # Represents 0.01
        self.Dv_slider.setMaximum(100)  # Represents 1.00
        self.Dv_slider.setValue(int(self.glWidget.Dv * 100))
        self.Dv_slider.valueChanged.connect(self.Dv_changed)
        self.Dv_slider.setToolTip('Adjust the diffusion coefficient D₂.')
        layout.addWidget(self.Dv_slider)

        self.Dv_spinbox = QDoubleSpinBox()
        self.Dv_spinbox.setDecimals(3)
        self.Dv_spinbox.setRange(0.01, 1.0)
        self.Dv_spinbox.setSingleStep(0.01)
        self.Dv_spinbox.setValue(self.glWidget.Dv)
        self.Dv_spinbox.valueChanged.connect(self.Dv_spinbox_changed)
        self.Dv_spinbox.setToolTip('Set the diffusion coefficient D₂ value.')
        layout.addWidget(self.Dv_spinbox)

        # Feed Rate (F)
        feed_label = QLabel('Feed Rate (F):')
        layout.addWidget(feed_label)

        self.feed_slider = QSlider(Qt.Horizontal)
        self.feed_slider.setMinimum(1)    # Represents 0.01
        self.feed_slider.setMaximum(100)  # Represents 1.00
        self.feed_slider.setValue(int(self.glWidget.feed_rate * 100))
        self.feed_slider.valueChanged.connect(self.feed_changed)
        self.feed_slider.setToolTip('Adjust the feed rate (F).')
        layout.addWidget(self.feed_slider)

        self.feed_spinbox = QDoubleSpinBox()
        self.feed_spinbox.setDecimals(3)
        self.feed_spinbox.setRange(0.01, 1.0)
        self.feed_spinbox.setSingleStep(0.01)
        self.feed_spinbox.setValue(self.glWidget.feed_rate)
        self.feed_spinbox.valueChanged.connect(self.feed_spinbox_changed)
        self.feed_spinbox.setToolTip('Set the feed rate (F) value.')
        layout.addWidget(self.feed_spinbox)

        # Kill Rate (k)
        kill_label = QLabel('Kill Rate (k):')
        layout.addWidget(kill_label)

        self.kill_slider = QSlider(Qt.Horizontal)
        self.kill_slider.setMinimum(1)    # Represents 0.01
        self.kill_slider.setMaximum(100)  # Represents 1.00
        self.kill_slider.setValue(int(self.glWidget.kill_rate * 100))
        self.kill_slider.valueChanged.connect(self.kill_changed)
        self.kill_slider.setToolTip('Adjust the kill rate (k).')
        layout.addWidget(self.kill_slider)

        self.kill_spinbox = QDoubleSpinBox()
        self.kill_spinbox.setDecimals(3)
        self.kill_spinbox.setRange(0.01, 1.0)
        self.kill_spinbox.setSingleStep(0.01)
        self.kill_spinbox.setValue(self.glWidget.kill_rate)
        self.kill_spinbox.valueChanged.connect(self.kill_spinbox_changed)
        self.kill_spinbox.setToolTip('Set the kill rate (k) value.')
        layout.addWidget(self.kill_spinbox)

    def create_schrodinger_controls(self):
        self.schrodinger_controls = QWidget()
        layout = QVBoxLayout()
        self.schrodinger_controls.setLayout(layout)

        # Schrödinger Method Selection
        schrodinger_label = QLabel('Schrödinger Method:')
        layout.addWidget(schrodinger_label)

        self.schrodinger_combo = QComboBox()
        self.schrodinger_combo.addItems(['Explicit', 'Crank-Nicolson'])
        self.schrodinger_combo.currentTextChanged.connect(self.schrodinger_method_changed)
        self.schrodinger_combo.setToolTip('Choose the numerical method for the Schrödinger Equation.')
        layout.addWidget(self.schrodinger_combo)

    def create_dt_controls(self):
        self.dt_controls = QWidget()
        layout = QVBoxLayout()
        self.dt_controls.setLayout(layout)

        # Time Step (dt) Control
        dt_label = QLabel('Time Step (Δt):')
        layout.addWidget(dt_label)

        self.dt_slider = QSlider(Qt.Horizontal)
        self.dt_slider.setMinimum(1)    # Represents 0.0001
        self.dt_slider.setMaximum(1000) # Represents 0.1
        self.dt_slider.setValue(int(self.glWidget.dt * 10000))
        self.dt_slider.valueChanged.connect(self.dt_changed)
        self.dt_slider.setToolTip('Adjust the time step (Δt) for the simulation.')
        layout.addWidget(self.dt_slider)

        self.dt_spinbox = QDoubleSpinBox()
        self.dt_spinbox.setDecimals(5)
        self.dt_spinbox.setRange(0.0001, 0.1)
        self.dt_spinbox.setSingleStep(0.0001)
        self.dt_spinbox.setValue(self.glWidget.dt)
        self.dt_spinbox.valueChanged.connect(self.dt_spinbox_changed)
        self.dt_spinbox.setToolTip('Set the time step (Δt) value.')
        layout.addWidget(self.dt_spinbox)

    def create_simulation_controls(self):
        self.simulation_controls = QWidget()
        layout = QVBoxLayout()
        self.simulation_controls.setLayout(layout)

        # Simulation Controls
        controls_label = QLabel('Simulation Controls:')
        layout.addWidget(controls_label)

        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.glWidget.start_simulation)
        self.start_button.setToolTip('Start the simulation.')
        layout.addWidget(self.start_button)

        self.pause_button = QPushButton('Pause')
        self.pause_button.clicked.connect(self.glWidget.pause_simulation)
        self.pause_button.setToolTip('Pause the simulation.')
        layout.addWidget(self.pause_button)

        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.glWidget.reset_simulation)
        self.reset_button.setToolTip('Reset the simulation to its initial state.')
        layout.addWidget(self.reset_button)

    def create_save_load_controls(self):
        self.save_load_controls = QWidget()
        layout = QVBoxLayout()
        self.save_load_controls.setLayout(layout)

        # Save and Load Buttons
        self.save_button = QPushButton('Save Configuration')
        self.save_button.clicked.connect(self.save_configuration)
        self.save_button.setToolTip('Save the current simulation settings.')
        layout.addWidget(self.save_button)

        self.load_button = QPushButton('Load Configuration')
        self.load_button.clicked.connect(self.load_configuration)
        self.load_button.setToolTip('Load simulation settings from a file.')
        layout.addWidget(self.load_button)

    # Control change methods
    def alpha_changed(self, value):
        alpha = value / 100.0
        self.glWidget.set_alpha(alpha)
        self.alpha_spinbox.blockSignals(True)
        self.alpha_spinbox.setValue(alpha)
        self.alpha_spinbox.blockSignals(False)

    def alpha_spinbox_changed(self, value):
        self.glWidget.set_alpha(value)
        self.alpha_slider.blockSignals(True)
        self.alpha_slider.setValue(int(value * 100))
        self.alpha_slider.blockSignals(False)

    def wave_changed(self, value):
        c = value / 10.0
        self.glWidget.set_c(c)
        self.wave_spinbox.blockSignals(True)
        self.wave_spinbox.setValue(c)
        self.wave_spinbox.blockSignals(False)

    def wave_spinbox_changed(self, value):
        self.glWidget.set_c(value)
        self.wave_slider.blockSignals(True)
        self.wave_slider.setValue(int(value * 10))
        self.wave_slider.blockSignals(False)

    def viscosity_changed(self, value):
        viscosity = value / 100.0
        self.glWidget.set_viscosity(viscosity)
        self.viscosity_spinbox.blockSignals(True)
        self.viscosity_spinbox.setValue(viscosity)
        self.viscosity_spinbox.blockSignals(False)

    def viscosity_spinbox_changed(self, value):
        self.glWidget.set_viscosity(value)
        self.viscosity_slider.blockSignals(True)
        self.viscosity_slider.setValue(int(value * 100))
        self.viscosity_slider.blockSignals(False)

    def Du_changed(self, value):
        Du = value / 100.0
        self.glWidget.set_Du(Du)
        self.Du_spinbox.blockSignals(True)
        self.Du_spinbox.setValue(Du)
        self.Du_spinbox.blockSignals(False)

    def Du_spinbox_changed(self, value):
        self.glWidget.set_Du(value)
        self.Du_slider.blockSignals(True)
        self.Du_slider.setValue(int(value * 100))
        self.Du_slider.blockSignals(False)

    def Dv_changed(self, value):
        Dv = value / 100.0
        self.glWidget.set_Dv(Dv)
        self.Dv_spinbox.blockSignals(True)
        self.Dv_spinbox.setValue(Dv)
        self.Dv_spinbox.blockSignals(False)

    def Dv_spinbox_changed(self, value):
        self.glWidget.set_Dv(value)
        self.Dv_slider.blockSignals(True)
        self.Dv_slider.setValue(int(value * 100))
        self.Dv_slider.blockSignals(False)

    def feed_changed(self, value):
        feed_rate = value / 100.0
        self.glWidget.set_feed_rate(feed_rate)
        self.feed_spinbox.blockSignals(True)
        self.feed_spinbox.setValue(feed_rate)
        self.feed_spinbox.blockSignals(False)

    def feed_spinbox_changed(self, value):
        self.glWidget.set_feed_rate(value)
        self.feed_slider.blockSignals(True)
        self.feed_slider.setValue(int(value * 100))
        self.feed_slider.blockSignals(False)

    def kill_changed(self, value):
        kill_rate = value / 100.0
        self.glWidget.set_kill_rate(kill_rate)
        self.kill_spinbox.blockSignals(True)
        self.kill_spinbox.setValue(kill_rate)
        self.kill_spinbox.blockSignals(False)

    def kill_spinbox_changed(self, value):
        self.glWidget.set_kill_rate(value)
        self.kill_slider.blockSignals(True)
        self.kill_slider.setValue(int(value * 100))
        self.kill_slider.blockSignals(False)

    def dt_changed(self, value):
        dt = value / 10000.0
        self.glWidget.set_dt(dt)
        self.dt_spinbox.blockSignals(True)
        self.dt_spinbox.setValue(dt)
        self.dt_spinbox.blockSignals(False)
        self.current_dt_label.setText(f'Current dt: {self.glWidget.dt:.6f}')

    def dt_spinbox_changed(self, value):
        self.glWidget.set_dt(value)
        self.dt_slider.blockSignals(True)
        self.dt_slider.setValue(int(value * 10000))
        self.dt_slider.blockSignals(False)
        self.current_dt_label.setText(f'Current dt: {self.glWidget.dt:.6f}')

    def schrodinger_method_changed(self, text):
        self.glWidget.set_schrodinger_method(text)

    def rho_changed(self, value):
        self.glWidget.set_rho(value)

    def nit_changed(self, value):
        self.glWidget.set_nit(value)

    def fx_changed(self, value):
        self.glWidget.set_Fx(value)

    def fy_changed(self, value):
        self.glWidget.set_Fy(value)

    def frequency_changed(self, value):
        self.glWidget.set_frequency(value)

    def save_configuration(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "JSON Files (*.json)", options=options)
        if filename:
            config = {
                'equation': self.equation_combo.currentText(),
                'boundary_condition': self.boundary_combo.currentText(),
                'alpha': self.alpha_spinbox.value(),
                'c': self.wave_spinbox.value(),
                'viscosity': self.viscosity_spinbox.value(),
                'Du': self.Du_spinbox.value(),
                'Dv': self.Dv_spinbox.value(),
                'feed_rate': self.feed_spinbox.value(),
                'kill_rate': self.kill_spinbox.value(),
                'dt': self.glWidget.dt,  # Save adaptive dt
                'schrodinger_method': self.schrodinger_combo.currentText(),
                'rho': self.rho_spinbox.value(),
                'nit': self.nit_spinbox.value(),
                'Fx': self.fx_spinbox.value(),
                'Fy': self.fy_spinbox.value(),
                'frequency': self.frequency_spinbox.value()
            }
            with open(filename, 'w') as f:
                json.dump(config, f)

    def load_configuration(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)", options=options)
        if filename:
            with open(filename, 'r') as f:
                config = json.load(f)
            # Update UI elements and GLWidget accordingly
            self.equation_combo.setCurrentText(config.get('equation', 'Heat Equation'))
            self.boundary_combo.setCurrentText(config.get('boundary_condition', 'Dirichlet'))
            self.alpha_spinbox.setValue(config.get('alpha', 0.01))
            self.wave_spinbox.setValue(config.get('c', 1.0))
            self.viscosity_spinbox.setValue(config.get('viscosity', 0.1))
            self.Du_spinbox.setValue(config.get('Du', 0.16))
            self.Dv_spinbox.setValue(config.get('Dv', 0.08))
            self.feed_spinbox.setValue(config.get('feed_rate', 0.035))
            self.kill_spinbox.setValue(config.get('kill_rate', 0.065))
            self.dt_spinbox.setValue(config.get('dt', 0.0001))
            self.schrodinger_combo.setCurrentText(config.get('schrodinger_method', 'Explicit'))
            self.rho_spinbox.setValue(config.get('rho', 1.0))
            self.nit_spinbox.setValue(config.get('nit', 50))
            self.fx_spinbox.setValue(config.get('Fx', 0.0))
            self.fy_spinbox.setValue(config.get('Fy', 0.0))
            self.frequency_spinbox.setValue(config.get('frequency', 1.0))
            self.glWidget.dt = config.get('dt', 0.0001)
            self.glWidget.reset_simulation()
            # Update the current dt display
            self.update_current_dt_display()

    def update_current_dt_display(self):
        self.current_dt_label.setText(f'Current dt: {self.glWidget.dt:.6f}')


# Define the Color Scale Widget
class ColorScaleWidget(QWidget):
    def __init__(self, parent=None):
        super(ColorScaleWidget, self).__init__(parent)
        self.equation = 'Heat Equation'  # Default equation

    def set_equation(self, equation):
        self.equation = equation
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())

        if self.equation == 'Heat Equation':
            # Blue (cold) to Red (hot)
            gradient.setColorAt(0.0, QColor(0, 0, 255))   # Blue
            gradient.setColorAt(1.0, QColor(255, 0, 0))   # Red
            title = "Temperature"
        elif self.equation == 'Schrödinger Equation':
            # Black (low) to White (high)
            gradient.setColorAt(0.0, QColor(0, 0, 0))     # Black
            gradient.setColorAt(1.0, QColor(255, 255, 255))  # White
            title = "Probability Density"
        elif self.equation == 'Wave Equation':
            # Blue (negative displacement) to White (zero) to Red (positive displacement)
            gradient.setColorAt(0.0, QColor(0, 0, 255))     # Blue
            gradient.setColorAt(0.5, QColor(255, 255, 255)) # White
            gradient.setColorAt(1.0, QColor(255, 0, 0))     # Red
            title = "Displacement"
        elif self.equation == "Burgers' Equation" or self.equation == 'Navier-Stokes':
            if self.equation == 'Navier-Stokes':
                # Enhanced gradient for Navier-Stokes: Blue → Cyan → Green → Yellow → Red
                gradient.setColorAt(0.0, QColor(0, 0, 255))       # Blue
                gradient.setColorAt(0.25, QColor(0, 255, 255))    # Cyan
                gradient.setColorAt(0.5, QColor(0, 255, 0))       # Green
                gradient.setColorAt(0.75, QColor(255, 255, 0))    # Yellow
                gradient.setColorAt(1.0, QColor(255, 0, 0))       # Red
            else:
                # Original gradient for Burgers' Equation
                gradient.setColorAt(0.0, QColor(0, 0, 255))   # Blue
                gradient.setColorAt(1.0, QColor(255, 0, 0))   # Red
            title = "Velocity Magnitude"
        elif self.equation == 'Reaction-Diffusion':
            # Blue (low concentration) to Red (high concentration)
            gradient.setColorAt(0.0, QColor(0, 0, 255))   # Blue
            gradient.setColorAt(1.0, QColor(255, 0, 0))   # Red
            title = "Concentration"

        painter.setBrush(gradient)
        painter.drawRect(10, 30, self.width() - 20, self.height() - 40)

        # Draw labels
        painter.setPen(Qt.white)
        painter.drawText(5, 25, title + " Scale")
        if self.equation == 'Wave Equation':
            painter.drawText(5, self.height() - 10, "Negative")
            painter.drawText(self.width() - 50, self.height() - 10, "Positive")
        else:
            painter.drawText(5, self.height() - 10, "Low")
            painter.drawText(self.width() - 35, self.height() - 10, "High")



# Main execution
