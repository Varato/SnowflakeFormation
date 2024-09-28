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

def contrast(s, a: float=1):
    # y = (np.exp(2*s) - 1) / (np.exp(2*s) + 1)
    # cm = plt.cm.gray(y)
    
    # s: 0-1
    vapor_contrast = 0.8*s # s<1
    vapor_color = plt.cm.gray(vapor_contrast)
    
    # s > 1
    ice_contrast = (np.exp(2*(s-0.9)*a) - 1) / (np.exp(2*(s-0.9)*a) + 1)
    ice_color = plt.cm.ocean_r(ice_contrast)
    
    y = np.zeros_like(s)
    y[s<1] = vapor_contrast[s<1]
    y[s>=1] = ice_contrast[s>=1]
    
    cm = np.zeros([s.shape[0], 4]) # 4: rgba
    cm[s<1] = vapor_color[s<1]
    cm[s>=1] = ice_color[s>=1]
    
    return y, cm

def find_six_neighbors(s: np.ndarray):
    """
    Parameters
    ----------
        s: (N+2) x (N+2) array

    Returns
    -------
        s1, s2, s3, s4, s5, s6: NxN array
    """
    # 1. find the eight neighbors (including diagonal ones) in a square grid
    s_u = s[:-2, 1:-1]
    s_d = s[2:,  1:-1] 
    s_l = s[1:-1, :-2]
    s_r = s[1:-1,  2:]

    s_lu = s[:-2, :-2]
    s_ru = s[:-2, 2:]
    s_ld = s[2:, :-2]
    s_rd = s[2:, 2:]

    # 2. construct the 6 neighbors of a hexagon grid
    s1 = s_u
    s2 = s_d
    s3 = s_l
    s4 = s_r

    s5 = np.empty([s.shape[0]-2, s.shape[1]-2])
    s6 = np.empty([s.shape[0]-2, s.shape[1]-2])

    s5[:, 0::2] = s_ld[:, 0::2]
    s6[:, 0::2] = s_rd[:, 0::2]

    s5[:, 1::2] = s_lu[:, 1::2]
    s6[:, 1::2] = s_ru[:, 1::2]

    return (s1, s2, s3, s4, s5, s6)


class ReiterCellularAutomata:
    def __init__(self, grid_size: int, alpha: float, beta: float, gamma: float) -> None:
        self.grid_size = grid_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # 1. +2 to give the grid edges around so that every cell has 6 neighbors
        self.s = np.ones((grid_size + 2, grid_size + 2)) * beta
        # self.s[grid_size//2+1, grid_size//2+1] = 1.0
        self.s[4*grid_size//9, 4*grid_size//9] = 1.0
        self.s[5*grid_size//9, 5*grid_size//9] = 1.0

        self.xmin, self.ymax = get_hexegon_center(1, 1, 1)
        self.xmax, self.ymin = get_hexegon_center(grid_size, grid_size, 1)

        self.boundary_mask = np.ones(self.s.shape, dtype=np.int32)
        boundary_size = 50
        self.boundary_mask[boundary_size:-boundary_size, boundary_size:-boundary_size] = 0


    def frozen(self) -> np.ndarray: # returns NxN
        return self.s[1:-1, 1:-1] >= 1
    
    def have_frozen_neighbors(self) -> np.ndarray:
        s1, s2, s3, s4, s5, s6 = find_six_neighbors(self.s)
        return np.logical_or.reduce((
            s1>=1, 
            s2>=1,
            s3>=1,
            s4>=1, 
            s5>=1,
            s6>=1
        ))

    def receptive(self) -> np.ndarray:
        return np.logical_or(self.frozen(), self.have_frozen_neighbors())
    
    def compute_mean(self, u: np.ndarray) -> np.ndarray:
        # u: N+2 by N+2
        # returns: N x N
        u1, u2, u3, u4, u5, u6 = find_six_neighbors(u)
        return (u1 + u2 + u3 + u4 + u5 + u6) / 6.0
        
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
        if self.edge_touched():
            return
        receptive = self.receptive() # boolean array
        
        v = np.zeros([self.grid_size+2, self.grid_size+2])
        v[1:-1,1:-1][receptive] = self.s[1:-1,1:-1][receptive] + self.gamma
        
        u = np.zeros([self.grid_size+2, self.grid_size+2])
        u[0, :] = self.beta
        u[-1, :] = self.beta
        u[:, 0] = self.beta
        u[:, -1] = self.beta
        u[1:-1, 1:-1][~receptive] = self.s[1:-1,1:-1][~receptive]
        u1, u2, u3, u4, u5, u6 = find_six_neighbors(u)
        u_mean = (u1 + u2 + u3 + u4 + u5 + u6)/6.0
        u[1:-1, 1:-1] = u[1:-1, 1:-1] + self.alpha * (u_mean - u[1:-1, 1:-1])
        
        self.s[1:-1, 1:-1] = u[1:-1, 1:-1] + v[1:-1, 1:-1]

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

    def draw_fast(self, ax: plt.Axes, a=1.0):
       
        R, C = np.meshgrid(np.arange(0, self.grid_size+2), np.arange(0, self.grid_size+2), indexing="ij")
        X = C * 1.5
        Y = R * math.sqrt(3) + (C % 2) * math.sqrt(3)/2
        _, cm = contrast(self.s.flatten(), a=a)
        ax.scatter(X.flatten(), -Y.flatten(), c=cm, s=12)
    
    def edge_touched(self):
        if np.any(self.s[self.boundary_mask] >= 1):
            return True
        else:
            return False


if __name__ == "__main__":

    alpha = 1

    1
    beta = 0.4
    gamma = 0.001
    a = 0.7

    # 2
    # beta = 0.9
    # gamma = 0.0001
    # a = 1.3

    # 3
    # beta = 0.7
    # gamma = 0.01
    # a=0.4

    # 4
    # beta = 0.3
    # gamma = 0.05
    # a = 0.02

    ca = ReiterCellularAutomata(151, alpha, beta, gamma)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 8))
    plt.subplots_adjust(wspace=0, hspace=0, top=1, bottom=0, right=1,left=0)

    # def run_ca(t):
    #     ca.draw_fast(ax)
    #     print("t={}, s lim: [{:.3f}, {:.3f}] ".format(t, np.min(ca.s), np.max(ca.s)))
    #     ca.update()

    # anim = FuncAnimation(fig, func=run_ca, frames=range(800), interval=200, repeat=False, cache_frame_data=False)
    # plt.show()

    
    N = 800
    for t in range(1, N+1):
        save_name = f'anims/flake_{t:03d}.png'
        if not os.path.isfile(save_name):
            plt.cla() # important to be fast
            ax.set_aspect('equal')
            ax.set_ylim([ca.ymin, ca.ymax])
            ax.set_xlim([ca.xmin, ca.xmax])
            ax.axis('off')   
            ca.draw_fast(ax, a=a)
            plt.savefig(save_name, dpi=120, transparent=True)
            print('saved at {}'.format(save_name))
        print("t={}, s lim: [{:.3f}, {:.3f}]  boundary: {}".format(t, np.min(ca.s), np.max(ca.s), ca.edge_touched()))
        ca.update()
