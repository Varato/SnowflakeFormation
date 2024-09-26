"""
Implementation of Reiter's model of cellular automata.

Reiter's model is a hexagonal automata which can be described as follows. Tessellate the
plane into hexagonal cells. Each cell has six nearest neighbors. The state variable 's(t, z)' of cell
'z' at time 't' represents the amount of water stored in cell. The cells are divided into three types:
    - A cell is called frozen if s(t, z) >= 1. 
    - If a cell is not frozen itself but at least one of the nearest neighbors is frozen, the cell is called a boundary cell. 
    - A cell that is neither frozen nor boundary is called nonreceptive. 
The union of frozen and boundary cells are called receptive cells.
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation


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
    The left top hexegon is indexed by r=0, c=0, coordinate is (0, 0).
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
        self.s[grid_size//2+1, grid_size//2+1] = 1.0
        # self.s[5*grid_size//9, 5*grid_size//9] = 1.0

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

        # adding constant for receptive cells
        rep[1:-1, 1:-1][receptive] = self.s[1:-1, 1:-1][receptive] + self.gamma

        # diffusion
        nrep[0, :] = self.beta
        nrep[-1, :] = self.beta
        nrep[:, 0] = self.beta
        nrep[:, -1] = self.beta        
        nrep[1:-1, 1:-1][~receptive] = self.s[1:-1, 1:-1][~receptive]
        nrep[1:-1, 1:-1] = (1 - 0.5*self.alpha)*nrep[1:-1, 1:-1] + 0.5*self.alpha*self.compute_mean(nrep)

        self.s[1:-1, 1:-1] = rep[1:-1, 1:-1] + nrep[1:-1, 1:-1]

    def diameter(self):
        frozen_grid = self.frozen()
        return np.sum(frozen_grid[:, self.grid_size//2])
    
    def draw(self, ax: plt.Axes):
        ax.set_aspect('equal')
        ax.set_ylim([self.ymin, self.ymax])
        ax.set_xlim([self.xmin, self.xmax])
        ax.axis('off')
        for r in range(N+2):
            for c in range(N+2):
                x, y = get_hexegon_center(r, c, 1)
                s = self.s[r, c]
                # normalize the s value to [0, 1]
                vertices = get_hexegon_vertices(x, y, 1)
                # draw the polygon grid by ax.fill
                color = self.get_color(s)
                ax.fill(*zip(*vertices), facecolor=color, edgecolor=color)
                # fill the hexegon with gray color mapped by the value s
        # ax.text(0.5*(self.xmax - self.xmin), -5, 
        #         f'a={self.alpha:.2f}, b={self.beta:f}, g={self.gamma:f}', 
        #         ha='center', va='center', fontsize=10)
        # ax.text(0.5*(self.xmax - self.xmin), -10, f'diameter = {self.diameter():d}', ha='center', va='center', fontsize=10)

    def draw_fast(self, ax: plt.Axes):
       
        R, C = np.meshgrid(np.arange(0, self.grid_size+2), np.arange(0, self.grid_size+2), indexing="ij")
        X = C * 1.5
        Y = R * math.sqrt(3) + (C % 2) * math.sqrt(3)/2
        ax.scatter(X.flatten(), -Y.flatten(), c=self.water_color_map(self.s.flatten()), s=12)

    def water_color_map(self, x:np.ndarray, a:float=0.6):
        sigmoid = 0.6/(1+np.exp(-(x-1)*a)) #(n,)
        # y = 0.8*x
        # y[x>=1] = sigmoid[x>=1]
        frozen_colors = plt.cm.ocean_r(sigmoid) #(n, 4)
        nonfrozen_colors = plt.cm.gray(x*0.8)  #(n, 4)

        cm = np.zeros([x.shape[0], 4])
        cm[x>=1] = frozen_colors[x>=1]
        cm[x<1] = nonfrozen_colors[x<1]
        return cm


if __name__ == "__main__":

    alpha = 1

    # 1
    beta = 0.9
    gamma = 0.05

    #2
    # beta = 0.4
    # gamma = 0.001

    N = 151
    ca = ReiterCellularAutomata(N, alpha, beta, gamma)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 8))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, right=1,left=0)


    # def run_ca(t):
    #     ca.draw_fast(ax)
    #     print("t={}, s lim: [{:.3f}, {:.3f}] ".format(t, np.min(ca.s), np.max(ca.s)))
    #     ca.update()

    # anim = FuncAnimation(fig, func=run_ca, frames=range(800), interval=200, repeat=False, cache_frame_data=False)
    # plt.show()

    for t in range(1, 70+1):
        save_name = f'anims/flake_{t:03d}.png'
        if not os.path.isfile(save_name):
            plt.cla() # important to be fast
            ax.set_aspect('equal')
            ax.set_ylim([ca.ymin, ca.ymax])
            ax.set_xlim([ca.xmin, ca.xmax])
            ax.axis('off')   
            ca.draw_fast(ax)
            plt.savefig(save_name, dpi=120, transparent=True)
            print('saved at {}'.format(save_name))
        print("t={}, s lim: [{:.3f}, {:.3f}] ".format(t, np.min(ca.s), np.max(ca.s)))
        ca.update()

  
        

    
