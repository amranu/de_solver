import numpy as np
from numba import njit, prange
from scipy.sparse import diags, identity, kron, csc_matrix
from scipy.sparse.linalg import spsolve

@njit(parallel=True)
def heat_equation_update(u, alpha, dt, dx, dy, boundary_condition):
    u_new = u.copy()
    dx2 = dx ** 2
    dy2 = dy ** 2

    for i in prange(1, u.shape[0] - 1):
        for j in range(1, u.shape[1] - 1):
            u_new[i, j] = u[i, j] + alpha * dt * (
                (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx2 +
                (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy2
            )

    # Apply boundary conditions
    if boundary_condition == 'Dirichlet':
        u_new[0, :] = 0
        u_new[-1, :] = 0
        u_new[:, 0] = 0
        u_new[:, -1] = 0
    elif boundary_condition == 'Neumann':
        u_new[0, :] = u_new[1, :]
        u_new[-1, :] = u_new[-2, :]
        u_new[:, 0] = u_new[:, 1]
        u_new[:, -1] = u_new[:, -2]
    elif boundary_condition == 'Periodic':
        u_new[0, :] = u_new[-2, :]
        u_new[-1, :] = u_new[1, :]
        u_new[:, 0] = u_new[:, -2]
        u_new[:, -1] = u_new[:, 1]

    return u_new

@njit
def schrodinger_equation_update(psi, dt, dx, dy, boundary_condition):
    psi_new = psi.copy()
    dx2 = dx ** 2
    dy2 = dy ** 2

    # Compute the laplacian
    for i in range(1, psi.shape[0] - 1):
        for j in range(1, psi.shape[1] - 1):
            laplacian = (
                (psi[i + 1, j] - 2 * psi[i, j] + psi[i - 1, j]) / dx2 +
                (psi[i, j + 1] - 2 * psi[i, j] + psi[i, j - 1]) / dy2
            )
            # Time evolution using the Schrödinger equation (Explicit method)
            psi_new[i, j] = psi[i, j] + 1j * dt * 0.5 * laplacian

    # Apply boundary conditions
    if boundary_condition == 'Dirichlet':
        psi_new[0, :] = 0
        psi_new[-1, :] = 0
        psi_new[:, 0] = 0
        psi_new[:, -1] = 0
    elif boundary_condition == 'Neumann':
        psi_new[0, :] = psi_new[1, :]
        psi_new[-1, :] = psi_new[-2, :]
        psi_new[:, 0] = psi_new[:, 1]
        psi_new[:, -1] = psi_new[:, -2]
    elif boundary_condition == 'Periodic':
        psi_new[0, :] = psi_new[-2, :]
        psi_new[-1, :] = psi_new[1, :]
        psi_new[:, 0] = psi_new[:, -2]
        psi_new[:, -1] = psi_new[:, 1]

    return psi_new

def schrodinger_equation_cn_update(psi, dt, dx, dy, boundary_condition):
    Nx, Ny = psi.shape
    N = Nx * Ny  # Total number of grid points

    # Constants
    hbar = 1.0  # Reduced Planck's constant
    m = 1.0     # Particle mass

    # Precompute coefficients
    r_x = 1j * hbar * dt / (4 * m * dx ** 2)
    r_y = 1j * hbar * dt / (4 * m * dy ** 2)

    # Create 1D Laplacian operators with appropriate boundary conditions
    e = np.ones(Nx)
    diagonals = [-2 * e, e[:-1], e[:-1]]
    L1D = diags(diagonals, [0, -1, 1], shape=(Nx, Nx))

    if boundary_condition == 'Dirichlet':
        # Dirichlet boundary conditions: already handled by L1D
        pass
    elif boundary_condition == 'Neumann':
        # Modify diagonals for Neumann boundary conditions
        L1D = L1D.toarray()
        L1D[0, 0] = -1
        L1D[-1, -1] = -1
        L1D = csc_matrix(L1D)
    elif boundary_condition == 'Periodic':
        # Modify diagonals for Periodic boundary conditions
        L1D = L1D.toarray()
        L1D[0, -1] = 1
        L1D[-1, 0] = 1
        L1D = csc_matrix(L1D)

    # Create 2D Laplacian operator using Kronecker products
    I = identity(Nx)
    Lx = kron(I, L1D)
    Ly = kron(L1D, I)
    Laplacian = Lx + Ly

    # Construct the A and B matrices
    A = identity(N, dtype=complex) - (r_x + r_y) * Laplacian
    B = identity(N, dtype=complex) + (r_x + r_y) * Laplacian

    # Flatten the psi array for matrix operations
    psi_flat = psi.flatten()

    # Solve the linear system A * psi_new = B * psi_old
    psi_new_flat = spsolve(A, B.dot(psi_flat))

    # Reshape back to 2D array
    psi_new = psi_new_flat.reshape((Nx, Ny))

    return psi_new

@njit(parallel=True)
def wave_equation_update(u_prev, u_current, c, dt, dx, dy, boundary_condition):
    u_next = np.zeros_like(u_current)
    dx2 = dx ** 2
    dy2 = dy ** 2

    for i in prange(1, u_current.shape[0] - 1):
        for j in range(1, u_current.shape[1] - 1):
            laplacian = (
                (u_current[i + 1, j] - 2 * u_current[i, j] + u_current[i - 1, j]) / dx2 +
                (u_current[i, j + 1] - 2 * u_current[i, j] + u_current[i, j - 1]) / dy2
            )
            u_next[i, j] = (2 * u_current[i, j] - u_prev[i, j] +
                            (c * dt) ** 2 * laplacian)

    # Apply boundary conditions
    if boundary_condition == 'Dirichlet':
        u_next[0, :] = 0
        u_next[-1, :] = 0
        u_next[:, 0] = 0
        u_next[:, -1] = 0
    elif boundary_condition == 'Neumann':
        u_next[0, :] = u_next[1, :]
        u_next[-1, :] = u_next[-2, :]
        u_next[:, 0] = u_next[:, 1]
        u_next[:, -1] = u_next[:, -2]
    elif boundary_condition == 'Periodic':
        u_next[0, :] = u_next[-2, :]
        u_next[-1, :] = u_next[1, :]
        u_next[:, 0] = u_next[:, -2]
        u_next[:, -1] = u_next[:, 1]

    # Update previous and current states
    u_prev = u_current.copy()
    u_current = u_next.copy()

    return u_prev, u_current

@njit(parallel=True)
def burgers_equation_update(u, v, viscosity, dt, dx, dy):
    u_new = u.copy()
    v_new = v.copy()
    Nx, Ny = u.shape

    for i in prange(Nx):
        for j in range(Ny):
            ip = (i + 1) % Nx
            im = (i - 1 + Nx) % Nx
            jp = (j + 1) % Ny
            jm = (j - 1 + Ny) % Ny

            u_x = (u[ip, j] - u[im, j]) / (2 * dx)
            u_y = (u[i, jp] - u[i, jm]) / (2 * dy)
            v_x = (v[ip, j] - v[im, j]) / (2 * dx)
            v_y = (v[i, jp] - v[i, jm]) / (2 * dy)

            u_xx = (u[ip, j] - 2 * u[i, j] + u[im, j]) / dx ** 2
            u_yy = (u[i, jp] - 2 * u[i, j] + u[i, jm]) / dy ** 2
            v_xx = (v[ip, j] - 2 * v[i, j] + v[im, j]) / dx ** 2
            v_yy = (v[i, jp] - 2 * v[i, j] + v[i, jm]) / dy ** 2

            u_new[i, j] = u[i, j] - dt * (u[i, j] * u_x + v[i, j] * u_y) + viscosity * dt * (u_xx + u_yy)
            v_new[i, j] = v[i, j] - dt * (u[i, j] * v_x + v[i, j] * v_y) + viscosity * dt * (v_xx + v_yy)

    return u_new, v_new

@njit
def navier_stokes_update(u, v, p, b, viscosity, dt, dx, dy, boundary_condition, nit, rho, Fx, Fy):
    Nx, Ny = u.shape
    # Step 1: Tentative velocity fields (u*, v*)

    u_star = u.copy()
    v_star = v.copy()

    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):

            # Advection terms (central differencing)
            conv_u = u[i, j] * (u[i, j] - u[i - 1, j]) / dx + v[i, j] * (u[i, j] - u[i, j - 1]) / dy
            conv_v = u[i, j] * (v[i, j] - v[i - 1, j]) / dx + v[i, j] * (v[i, j] - v[i, j - 1]) / dy

            # Diffusion terms
            diff_u = viscosity * (
                (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) / dx ** 2 +
                (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) / dy ** 2
            )

            diff_v = viscosity * (
                (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) / dx ** 2 +
                (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) / dy ** 2
            )

            u_star[i, j] = u[i, j] + dt * (-conv_u + diff_u + Fx / rho)
            v_star[i, j] = v[i, j] + dt * (-conv_v + diff_v + Fy / rho)

    # Apply boundary conditions to u_star, v_star (No-slip boundaries)
    u_star[0, :] = 0
    u_star[-1, :] = 0
    u_star[:, 0] = 0
    u_star[:, -1] = 0

    v_star[0, :] = 0
    v_star[-1, :] = 0
    v_star[:, 0] = 0
    v_star[:, -1] = 0

    # Step 2: Compute RHS of Pressure Poisson Equation

    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            b[i, j] = (rho * ((u_star[i + 1, j] - u_star[i - 1, j]) / (2 * dx) +
                              (v_star[i, j + 1] - v_star[i, j - 1]) / (2 * dy))) / dt

    # Step 3: Solve Pressure Poisson Equation
    for iteration in range(nit):
        p_old = p.copy()
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                p[i, j] = (((p_old[i + 1, j] + p[i - 1, j]) * dy ** 2 +
                            (p_old[i, j + 1] + p[i, j - 1]) * dx ** 2) -
                           b[i, j] * dx ** 2 * dy ** 2) / (2 * (dx ** 2 + dy ** 2))

        # Apply Neumann boundary conditions (dp/dn = 0)
        p[0, :] = p[1, :]     # dp/dx = 0 at x=0
        p[-1, :] = p[-2, :]   # dp/dx = 0 at x=Lx
        p[:, 0] = p[:, 1]     # dp/dy = 0 at y=0
        p[:, -1] = p[:, -2]   # dp/dy = 0 at y=Ly

    # Step 4: Update velocities using pressure gradient
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            u[i, j] = u_star[i, j] - dt / (2 * rho * dx) * (p[i + 1, j] - p[i - 1, j])
            v[i, j] = v_star[i, j] - dt / (2 * rho * dy) * (p[i, j + 1] - p[i, j - 1])

    # Apply boundary conditions to u, v (No-slip boundaries)
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0

    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0

    return u, v, p
    
@njit(parallel=True)
def reaction_diffusion_update(u, v, Du, Dv, F, k, dt, dx, dy):
    u_new = u.copy()
    v_new = v.copy()
    Nx, Ny = u.shape

    for i in prange(Nx):
        for j in range(Ny):
            ip = (i + 1) % Nx
            im = (i - 1 + Nx) % Nx
            jp = (j + 1) % Ny
            jm = (j - 1 + Ny) % Ny

            u_xx = (u[ip, j] - 2 * u[i, j] + u[im, j]) / dx ** 2
            u_yy = (u[i, jp] - 2 * u[i, j] + u[i, jm]) / dy ** 2
            v_xx = (v[ip, j] - 2 * v[i, j] + v[im, j]) / dx ** 2
            v_yy = (v[i, jp] - 2 * v[i, j] + v[i, jm]) / dy ** 2

            uvv = u[i, j] * v[i, j] * v[i, j]

            u_new[i, j] = u[i, j] + dt * (Du * (u_xx + u_yy) - uvv + F * (1 - u[i, j]))
            v_new[i, j] = v[i, j] + dt * (Dv * (v_xx + v_yy) + uvv - (F + k) * v[i, j])

    return u_new, v_new
