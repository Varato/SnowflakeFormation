"""
Implementation of Reiter's model of cellular automata.

Reiter's model is a hexagonal automata which can be described as follows. Tessellate the
plane into hexagonal cells. Each cell has six nearest neighbors. The state variable 's(t, z)' of cell
'z' at time 't' represents the amount of water stored in cell. The cells are divided into three types:
    - A cell is called frozen if s(t, z) >= 1. 
    - If a cell is not frozen itself but at least one of the nearest neighbors is frozen, the cell is called a boundary cell. 
    - A cell that is neither frozen nor boundary is called nonreceptive. 
The union of frozen and boundary cells are called receptive cells.

Beside s(t, z), there are two more variables for each cell:
    - u(t, z): represents the amount of water that participates in diffusion
    - v(t, z): represents the amount of water that doesn't participate in diffusion

There are three parameters for the system:
    - beta: represents a fixed constant background vapor level
    - alpha: the diffusion constant.
    - gamma: the amount of vapor added in each simulation step

Updating rule of the cellular automata:

1. Constant addition. For any receptive cell z:
    v(t+1, z) = v(t, z) + gamma

2. Diffusion. For any cell z:
    u(t+1, z) = u(t, z) + alpha/2 * [
        1/6 * (sum of u(t, z') for all neighbors z' of z)
      - u(t, z) 
    ]

3. Updating s(z):
    s(t+1, z) = u(t+1, z) + v(t+1, z)
"""

import math
import matplotlib.pyplot as plt
import numpy as np


# First implement the hexegon grid function
def get_hexegon_center(r: int, c: int, L: float):
    """
    Given the indexing of a hexegon of a hexegon grid, return the center of the hexegon

    The hexegons are alined as follows
     _   _   _
    / \_/ \_/ \_ 
    \_/ \_/ \_/ \ 
    / \_/ \_/ \_/
    \_/ \_/ \_/
    
    The x axis is torward right.
    The y axis is torward bottom.
    The left top hexegon is indexed by r=0, c =0, coordinate is (0, 0).
    The one below the left top hexegon is index by r=1, c=0, coordinate is (0, sqrt(3)L)
    """
    x = c * 3/2 * L
    y = r * math.sqrt(3) * L + (c % 2) * math.sqrt(3)/2 * L
    return (x, -y)
    
def get_hexegon_vertices(x: float, y: float, L: float):
    """
    Given the center of a hexegon, return the vertices of the hexegon

    The hexegons are alined as follows
     _   _   _
    / \_/ \_/ \_ 
    \_/ \_/ \_/ \ 
    / \_/ \_/ \_/
    \_/ \_/ \_/
    
    The x axis is torward right.
    The y axis is torward bottom.
    The left top hexegon is indexed by r=0, c =0, coordinate is (0, 0).
    The one below the left top hexegon is index by r=1, c=0, coordinate is (0, sqrt(3)L)
    """
    vertices = []
    vertices.append((x - L, y))
    vertices.append((x - L/2, y + math.sqrt(3)/2 * L))
    vertices.append((x + L/2, y + math.sqrt(3)/2 * L))
    vertices.append((x + L, y))
    vertices.append((x + L/2, y - math.sqrt(3)/2 * L))
    vertices.append((x - L/2, y - math.sqrt(3)/2 * L))
    return vertices

def get_neighbors(r: int, c: int):
    neighbors = []
    neighbors.append((r, c + 1))
    neighbors.append((r, c - 1))
    neighbors.append((r + 1, c))
    neighbors.append((r - 1, c))
    if c % 2 == 0:
        neighbors.append((r - 1, c - 1))
        neighbors.append((r - 1, c + 1))
    else:
        neighbors.append((r + 1, c - 1))
        neighbors.append((r + 1, c + 1))
    return neighbors

class ReiterCellularAutomata:
    def __init__(self, grid_size: int, alpha: float, beta: float, gamma: float) -> None:
        self.grid_size = grid_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # 1. +2 to give the grid edges around so that every cell has 6 neighbors
        self.s = np.ones((grid_size + 2, grid_size + 2)) * beta
        self.s[grid_size//2 + 1, grid_size//2 + 1] = 1.0

        self.xmin, self.ymax = get_hexegon_center(1, 1, 1)
        self.xmax, self.ymin = get_hexegon_center(grid_size, grid_size, 1)

    def frozen(self) -> np.ndarray: # returns NxN
        return self.s[1:-1, 1:-1] >= 1
    
    def have_frozen_neighbors(self) -> np.ndarray:
        u = self.s[:-2, 1:-1] >= 1  # NxN
        d = self.s[2:,  1:-1] >= 1  # NxN 
        l = self.s[1:-1, :-2] >= 1  # NxN
        r = self.s[1:-1,  2:] >= 1  # NxN

        # left diagonal
        ld = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        ld[:, 0::2] = self.s[2:,  0:-2:2] >= 1.0  # for odd  columns' left diagonal neighbors
        ld[:, 1::2] = self.s[:-2, 1:-2:2] >= 1.0  # for even columns' left diagonal neighbors

        # right diagonal
        rd = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        rd[:, 0::2] = self.s[2:,  2::2] >= 1.0 # for odd  columns' right diagonal neighbors
        rd[:, 1::2] = self.s[:-2, 3::2] >= 1.0 # for odd  columns' right diagonal neighbors

        return np.logical_or.reduce((u, d, l, r, ld, rd))

    def receptive(self) -> np.ndarray:
        return np.logical_or(self.frozen(), self.have_frozen_neighbors())
    
    def compute_mean(self, u: np.ndarray) -> np.ndarray:
        # u: N+2 by N+2
        # returns: N x N
        u_u = u[:-2, 1:-1]
        u_d = u[2:,  1:-1] 
        u_l = u[1:-1, :-2]
        u_r = u[1:-1,  2:]

        u_ld = np.zeros((self.grid_size, self.grid_size))
        u_ld[:, 0::2] = u[2:,  0:-2:2]
        u_ld[:, 1::2] = u[:-2, 1:-2:2]

        u_rd = np.zeros((self.grid_size, self.grid_size))
        u_rd[: ,0::2] = u[2:,  2::2] # for odd  columns' right diagonal neighbors
        u_rd[: ,1::2] = u[:-2, 3::2] # for even  columns' right diagonal neighbors

        return (u_u + u_d + u_l + u_r + u_ld + u_rd) / 6.0
    
    def compute_mean_loop(self, u: np.ndarray) -> np.ndarray:
        # u: N+2 by N+2
        # returns: N x N
        u_mean = np.zeros((self.grid_size, self.grid_size))
        for r in range(self.grid_size + 2):
            for c in range(self.grid_size + 2):
                if r > 0 and r <= self.grid_size and c > 0 and c <= self.grid_size:
                    neighbors = get_neighbors(r, c)
                    
                    mean_val = 0
                    for rr, cc in neighbors:
                        mean_val += u[rr, cc]
                    mean_val /= 6

                    u_mean[r-1, c-1] = mean_val
        return u_mean

    def update(self) -> None:
        if self.diameter() > self.grid_size - 3:
            return
        receptive = self.receptive()

        rep = np.zeros([self.grid_size + 2, self.grid_size + 2])
        nrep = np.zeros([self.grid_size + 2, self.grid_size + 2])

        nrep[0, :] = self.beta
        nrep[-1, :] = self.beta
        nrep[:, 0] = self.beta
        nrep[:, -1] = self.beta

        rep[1:-1, 1:-1][receptive] = self.s[1:-1, 1:-1][receptive] + self.gamma
        nrep[1:-1, 1:-1][~receptive] = self.s[1:-1, 1:-1][~receptive]

        self.s[1:-1, 1:-1] = rep[1:-1, 1:-1] + self.compute_mean(nrep)

    def diameter(self):
        frozen_grid = self.frozen()
        return np.sum(frozen_grid[:, self.grid_size//2])
    
    def draw(self, ax: plt.Axes):

        ca_max = 2.0
        ca_min = self.beta

        ax.set_aspect('equal')
        ax.set_ylim([self.ymin, self.ymax])
        ax.set_xlim([self.xmin, self.xmax])
        ax.axis('off')
        for r in range(N+2):
            for c in range(N+2):
                x, y = get_hexegon_center(r, c, 1)
                s = np.clip(self.s[r, c], ca_min, ca_max)
                # normalize the s value to [0, 1]
                s = (s - ca_min) / (ca_max - ca_min) 
                s = s**0.5
                vertices = get_hexegon_vertices(x, y, 1)
                # draw the polygon grid by ax.fill
                ax.fill(*zip(*vertices), facecolor=plt.cm.gray(s), edgecolor=plt.cm.gray(s))
                # fill the hexegon with gray color mapped by the value s
        # ax.text(0.5*(self.xmax - self.xmin), -5, 
        #         f'a={self.alpha:.2f}, b={self.beta:f}, g={self.gamma:f}', 
        #         ha='center', va='center', fontsize=10)
        # ax.text(0.5*(self.xmax - self.xmin), -10, f'diameter = {self.diameter():d}', ha='center', va='center', fontsize=10)
    


if __name__ == "__main__":
    nrows = 2
    ncols = 2

    alpha = np.ones([nrows, ncols]) * 1.0
    beta =  np.array([[0.13, 0.5], [0.9, 0.7]])
    gamma = np.array([[0.01, 0.001], [0.001, 0.001]])

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(8.6,10))
    plt.subplots_adjust(wspace=0, hspace=0)

    N = 61
    ca_matrix = [[ReiterCellularAutomata(N, alpha[i, j], beta[i, j], gamma[i, j]) for j in range(ncols)] for i in range(nrows)]
    for t in range(400):
        for i in range(nrows):
            for j in range(ncols):
                print("t={} ca[{}, {}] updating, s lim: [{:.3f}, {:.3f}] ".format(t, i, j, np.min(ca_matrix[i][j].s), np.max(ca_matrix[i][j].s)))
                ca_matrix[i][j].update()
                ca_matrix[i][j].draw(axes[i, j])
        plt.savefig(f'anims/ca_{t:03d}.png', dpi=200, transparent=True)
        print('saved at {}'.format(f'anims/ca_{t:03d}.png'))
    # plt.show()
    # draw a N by N hexegon grid

    # r = 3
    # c = 5
    # neighbors = get_neighbors(r, c)
    # x, y = get_hexegon_center(r, c, 1)
    # ax.plot(x, y, 'bo')
    # for rr, cc in neighbors:
    #     x, y = get_hexegon_center(rr, cc, 1)
    #     ax.plot(x, y, 'ro')

    # plt.savefig('snowflake.png', dpi=200, transparent=True)            

    
