"""Solve MOOG FBD Wind fold problem."""
# -*- coding: utf-8 -*-
import pdb
import numpy as np
import pandas as pd
from dataclasses import dataclass
import sympy
from functools import cached_property

import seaborn as sns
import matplotlib.pyplot as plt
from plotfuncs import multiline

# %% Determine coordinates of point G via rotation
def rotate(vector: np.ndarray,
           angle: float,
           origin: np.ndarray) -> np.ndarray:
    """Rotate vector about origin (point B)"""
    v1 = vector - origin
    
    x = v1[0]
    y = v1[1]
    x1 = x * np.cos(angle) - y * np.sin(angle)
    y1 = x * np.sin(angle) + y * np.sin(angle)
    return np.array([x1, y1]) + origin


def unit_vector(r: np.ndarray):
    """Unit vector from vector."""
    mag = np.sum(r**2)**0.5
    return r / mag


def magnitude(r : np.ndarray):
    """Calculate magnitude of vector"""
    return np.sum(r**2)**0.5


@dataclass
class Point:
    """Point dataclass."""
    name : str
    loc : np.ndarray
    
    
@dataclass
class Element:
    """Element data storage."""
    name : str
    point1 : Point
    point2 : Point
    
    
    @cached_property
    def delta(self):
        return self.point2.loc - self.point1.loc
    
    
    @cached_property
    def length(self):
        d = self.delta
        return np.sqrt(np.sum(d**2))
        
    
    @cached_property
    def tangent(self):
        return self.delta / self.length
    
    
    def tension1(self, force: np.ndarray):
        """Calculate tension given a force acting on point1."""
        t = self.tangent
        return -(t[0]*force[0] + t[1]*force[1])
    
    
    def tension2(self, force: np.ndarray):
        """Calculate tension given a force acting on point2."""
        t = self.tangent
        return t[0]*force[0] + t[1]*force[1]


@dataclass
class Force:
    """Force dataclass."""
    point: Point
    link: str
    component: str
    index: int
    

    def __repr__(self):
        pname = self.point.name
        link = self.link
        return f'Force({pname}, {link})'
    
    
    def symbol(self):
        """Symbolic str name."""
        pname = self.point.name
        link = self.link
        component = self.component
        return f'F_{pname}_{link}_{component}'
    
    
    def vector_name(self):
        pname = self.point.name
        link = self.link    
        return f'F_{pname}_{link}'
    
    
class Moment(Force):
    """Moment dataclass"""
    def __repr__(self):
        pname = self.point.name
        link = self.link
        return f'Moment({pname}, {link})'

    
    def symbol(self):
        """Symbolic str name."""
        pname = self.point.name
        link = self.link
        component = self.component
        return f'M_{pname}_{link}_{component}'


    def vector_name(self):
        pname = self.point.name
        link = self.link    
        return f'M_{pname}_{link}'
    
    
class Solver:
    """Solve system of equations Ax = b for x."""
    def __init__(self, forces: list[Force]):
        self.forces = forces
        symbol_names = [f.symbol() for f in forces]
        self.force_dict = dict(zip(symbol_names, forces))  
        self.equations = []
        self.symbols = sympy.symbols(symbol_names)
        
        
    def add_equation(self, coeffs: dict, rhs: float):
        """Add linear equation. 
        
        Parameters
        ----------
        coeffs : dict
            Left hand side of the equation's coefficients for each force variable.
            Each key is the force name. Each value is the coefficient 
            amplitude. 
        rhs : float
            Right hand side value.
        """
        eqn = LinearEquation(self, coeffs, rhs)
        self.equations.append(eqn)
        
        
    def solve(self):
        """Solve system of equations Ax = b for x."""
        coeffs = [eqn.array() for eqn in self.equations]
        coeffs = np.array(coeffs)
        
        rhs = [eqn.value for eqn in self.equations]
        rhs = np.row_stack(rhs)
        
        # This performs x = inv(A)*b 
        out = np.linalg.inv(coeffs) @ rhs
        self.solution = out
        return out
    
    
    @cached_property
    def vector_dict(self):
        """Force vectors of solution for each force name."""
        force_values = self.solution.flatten()
        vector_names = [f.vector_name() for f in self.forces]
        vector_names = np.unique(vector_names)
        
        out = {}
        for name in vector_names:
            xname = name + '_x'
            yname = name + '_y'
            zname = name + '_z'
            try:
                ii = self.force_dict[xname].index
                fx = force_values[ii]
            except KeyError:
                fx = 0.0
                
            try:
                ii = self.force_dict[yname].index
                fy = force_values[ii]
            except KeyError:
                fy = 0.0          

            try:
                ii = self.force_dict[zname].index
                fz = force_values[ii]
            except KeyError:
                fz = 0.0      
            out[name] = np.array([fx, fy, fz])
        return out
    
    
    @cached_property
    def magnitude_dict(self):
        """Force magnitudes of the solution for each force name."""
        out = {}
        for name, values in self.vector_dict.items():
            magnitude = (values[0]**2 + values[1]**2 + values[2]**2)**0.5
            out[name] = magnitude
        return out
    
    
    @cached_property
    def elements(self):
        """dict of Element, indexed by element name."""
        links = {}
        for force in self.forces:
            point = force.point
            link = force.link
            try:
                links[link].append(point)
            except KeyError:
                links[link] = [point]
            
        out = {}
        for name, points in links.items():
            element = Element(name, points[0], points[1])
            out[name] = element
        return out
    
    
    @cached_property
    def points(self):
        """dict of Point, indexed by point name."""
        points = {}
        for force in self.forces:
            point = force.point
            points[point.name] = point
        return points
            
    
class LinearEquation:
    """Create a linear equation."""
    def __init__(self, solver: Solver, coeffs: dict, value: float):
        self.forces = solver.forces
        self.force_dict = solver.force_dict
        self.symbols = solver.symbols
        self.coeffs = coeffs
        self.value = value
        
        
    def __repr__(self):
        coeffs = self.coeffs
        eqn = np.zeros(18)
        for key, value  in coeffs.items():
            force = self.force_dict[key]
            eqn[force.index] = value
        
        return repr(sum(eqn * self.symbols))
    
    
    def array(self):
        """Construct coefficients array."""
        coeffs = self.coeffs
        eqn = np.zeros(18)
        for key, value  in coeffs.items():
            force = self.force_dict[key]
            eqn[force.index] = value
        return eqn


# %% Load element position data

df_locs = pd.read_csv('locations.csv', index_col=0)
force1x = 0
force1y = 100

# %% Construct vector to force point "G"

rb = df_locs.loc['B'].values[0:2]

rg_0 = np.array([15, -4])
rg_45 = rotate(rg_0, np.radians(45), rb)
rg_90 = rotate(rg_0, np.radians(90), rb)

# Add rg data to dataframe
data = np.concatenate([rg_0, rg_45, rg_90])
series = pd.Series(data, index=df_locs.columns, name='G')
df_locs = df_locs.append(series)

# %% Function to solve problem

def solve_eqn(angle):
    """Create system of equations for wing fold and solve. 

    Parameters
    ----------
    angle : float
        Angle of wing fold.

    Returns
    -------
    out : Solver
    """
    
    # Define position vectors
    if angle == 0:
        df = df_locs.iloc[:, 0:2]
    elif angle == 45:
        df = df_locs.iloc[:, 2:4]
    elif angle == 90:
        df = df_locs.iloc[:, 4:6]
    
    ra = df.loc['A'].values
    rb = df.loc['B'].values
    rc = df.loc['C'].values
    rd = df.loc['D'].values
    re = df.loc['E'].values
    rf = df.loc['F'].values
    rg = df.loc['G'].values
    
    # Relative position vectors, needed for moment equilibrium
    r_ba = rb - ra
    r_da = rd - ra
    r_ac = ra - rc
    r_bg = rb - rg
    r_bf = rb - rf
    r_fd = rf - rd
    r_ec = re - rc

    # Define forces
            
    # Generate points
    points = {}
    for (key, val) in df.iterrows():
        points[key] = Point(key, val.values)
        
    # Generate forces
    forces = []
    
    # X-components
    forces.append(Force(points['A'], '2',  'x', 0))
    forces.append(Force(points['A'], '3',  'x', 1))
    forces.append(Force(points['B'], 'BE', 'x', 2))
    forces.append(Force(points['B'], 'BF', 'x', 3))
    forces.append(Force(points['C'], '2',  'x', 4))
    forces.append(Force(points['D'], '3',  'x', 5))
    forces.append(Force(points['E'], 'BE', 'x', 6))
    forces.append(Force(points['F'], 'BF', 'x', 7))
    
    # Y-components
    forces.append(Force(points['A'], '2',  'y', 8))
    forces.append(Force(points['A'], '3',  'y', 9))
    forces.append(Force(points['B'], 'BE', 'y', 10))
    forces.append(Force(points['B'], 'BF', 'y', 11))
    forces.append(Force(points['C'], '2',  'y', 12))
    forces.append(Force(points['D'], '3',  'y', 13))
    forces.append(Force(points['E'], 'BE', 'y', 14))
    forces.append(Force(points['F'], 'BF', 'y', 15))
    
    # Moments
    forces.append(Moment(points['A'], '2', 'z', 16))
    forces.append(Moment(points['A'], '3', 'z', 17))
    
    symbol_names = [f.symbol() for f in forces]
    force_dict = dict(zip(symbol_names, forces))
    
    # Construct equilibrium equations
    symbols = sympy.symbols(symbol_names)
    for ii, symbol in enumerate(symbols):
        print(ii, symbol)
    
    # 0 F_A_2
    # 1 F_A_3
    # 2 F_B_BE
    # 3 F_B_BF
    # 4 F_C_2
    # 5 F_D_3
    # 6 F_E_BE
    # 7 F_F_BF
    # 8 M_A_2
    # 9 M_A_3
    
    solver = Solver(forces)

    # Construct force equilibrium equations
    coeffs1 = {}
    coeffs1['F_B_BE_x'] = 1
    coeffs1['F_B_BF_x'] = 1
    coeffs1['F_A_3_x'] = 1
    coeffs1['F_A_2_x'] = 1
    solver.add_equation(coeffs1, 0)
    
    coeffs1 = {}
    coeffs1['F_B_BE_y'] = 1
    coeffs1['F_B_BF_y'] = 1
    coeffs1['F_A_3_y'] = 1
    coeffs1['F_A_2_y'] = 1
    solver.add_equation(coeffs1, 0)
    
    coeffs2 = {}
    coeffs2['F_D_3_x'] = 1
    coeffs2['F_A_3_x'] = -1
    solver.add_equation(coeffs2, 0)
    
    coeffs2 = {}
    coeffs2['F_D_3_y'] = 1
    coeffs2['F_A_3_y'] = -1
    solver.add_equation(coeffs2, 0)
    
    coeffs3 = {}
    coeffs3['F_A_2_x'] = -1
    coeffs3['F_C_2_x'] = 1
    solver.add_equation(coeffs3, 0)
    
    coeffs3 = {}
    coeffs3['F_A_2_y'] = -1
    coeffs3['F_C_2_y'] = 1
    solver.add_equation(coeffs3, 0)
    
    coeffs4 = {}
    coeffs4['F_D_3_x'] = -1
    coeffs4['F_F_BF_x'] = 1
    solver.add_equation(coeffs4, 0)
    
    coeffs4 = {}
    coeffs4['F_D_3_y'] = -1
    coeffs4['F_F_BF_y'] = 1
    solver.add_equation(coeffs4, 0)
    
    coeffs5 = {}
    coeffs5['F_E_BE_x'] = 1
    coeffs5['F_C_2_x'] = -1
    solver.add_equation(coeffs5, 0)
    
    coeffs5 = {}
    coeffs5['F_E_BE_y'] = 1
    coeffs5['F_C_2_y'] = -1
    solver.add_equation(coeffs5, 0)
    
    coeffs6 = {}
    coeffs6['F_B_BF_x'] = 1
    coeffs6['F_F_BF_x'] = 1
    solver.add_equation(coeffs6, force1x)
    
    
    coeffs6 = {}
    coeffs6['F_B_BF_y'] = 1
    coeffs6['F_F_BF_y'] = 1
    solver.add_equation(coeffs6, force1y)
    
    # Construct moment equilibrium equations
    
    coeffs7 = {}
    coeffs7['F_A_2_x'] = -r_ba[1]
    coeffs7['F_A_2_y'] = r_ba[0]
    coeffs7['F_A_3_x'] = -r_ba[1]
    coeffs7['F_A_3_y'] = r_ba[0]
    solver.add_equation(coeffs7, 0)
    
    
    coeffs8 = {}
    coeffs8['M_A_3_z'] = 1
    coeffs8['F_A_3_x'] = r_da[1]
    coeffs8['F_A_3_y'] = -r_da[0]
    solver.add_equation(coeffs8, 0)
    
    coeffs9 = {}
    coeffs9['M_A_2_z'] = 1
    coeffs9['F_C_2_x'] = -r_ac[1]
    coeffs9['F_C_2_y'] = r_ac[0]
    solver.add_equation(coeffs9, 0)
    
    
    coeffs10 = {}
    coeffs10['F_D_3_x'] = -r_fd[1]
    coeffs10['F_D_3_y'] = r_fd[0]
    solver.add_equation(coeffs10, 0)
    
    coeffs11 = {}
    coeffs11['F_C_2_x'] = -r_ec[1]
    coeffs11['F_C_2_y'] = r_ec[0]
    solver.add_equation(coeffs11, 0)
    
    coeffs12 = {}
    coeffs12['F_F_BF_x'] = -r_bf[1]
    coeffs12['F_F_BF_y'] = r_bf[0]
    rhs = r_bg[0] * force1y - r_bg[1] * force1x
    
    solver.add_equation(coeffs12, rhs)
    solver.solve()
    return solver


def calc_tension(solver: Solver):
    """Calculate tension for each link element."""
    
    vec = solver.vector_dict
    
    # Get link tension
    points = solver.points
    
    # Build new elements for ones that weren't created from solver. 
    solver.elements['1'] = Element('1', points['A'], points['B'])
    solver.elements['4'] = Element('4', points['D'], points['F'])
    solver.elements['5'] = Element('5', points['C'], points['E'])
    
    t1 = solver.elements['1'].tension1(vec['F_A_2'] + vec['F_A_3'])
    t2 = solver.elements['2'].tension2(vec['F_C_2'])
    t3 = solver.elements['3'].tension2(vec['F_D_3'])
    t4 = solver.elements['4'].tension2(vec['F_F_BF'])
    t5 = solver.elements['5'].tension2(vec['F_E_BE'])
    
    tensions = {}
    tensions['1'] = t1
    tensions['2'] = t2
    tensions['3'] = t3
    tensions['4'] = t4
    tensions['5'] = t5
    return tensions



def plot(solver: Solver, tensions: dict, kwargs=None):
    
    # colors = plt.cm.coolwarm_r(np.linspace(0, 1, 100))
    if kwargs is None:
        kwargs = {}        
    kwargs['lw'] = 5
    
    
    # PLOT ELEMENTS
    x_points = []
    y_points = []
    values = []
    for name in tensions:
        
        tension = tensions[name]
        element = solver.elements[name]
        point1 = element.point1.loc
        point2 = element.point2.loc
        
        x_points.append([point1[0], point2[0]])
        y_points.append([point1[1], point2[1]])
        values.append(tension)
        
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    values = np.array(values)
    fig, ax = plt.subplots()
    # plt.ylim(-1, 5)
    # ax.set_adjustable('datalim')
    # ax.set_ylim(-5, 5)
    lc = multiline(x_points, y_points, values, **kwargs)
    
    axcb = fig.colorbar(lc)    
    axcb.set_label('Tension (lb)')
    plt.grid()
    
    # PLOT NODE & NODE LABELS
    for name, point in solver.points.items():
        plt.plot(point.loc[0], point.loc[1], 'o',
                 markersize=8,
                 color='r'
                 )
        plt.text(point.loc[0], point.loc[1], 
                 '  ' + point.name,
                 ha='left',
                 va='center',
                 color='r')
        
    # Annotate tensions 
    x_mids = x_points.mean(axis=1)
    y_mids = y_points.mean(axis=1)
    mid_points = np.column_stack((x_mids, y_mids))
    for name, tension, loc in zip(tensions, values, mid_points):
        label = f'L{name}'
        plt.text(loc[0], loc[1], label, color='b',
                 ha='left',
                 va='center',)
    plt.xlabel('X-Position (in)')
    plt.ylabel('Y-Position (in)')
    plt.axis((-1, 5, -5, 3))
    ax.set_aspect('equal')


def sanity_check(solver):
    """Validate equilibrium using another net equilibrium equation."""
    F_B_BE = solver.vector_dict['F_B_BE']
    F_E_BE = solver.vector_dict['F_E_BE']
    fnet = F_B_BE + F_E_BE
    # pdb.set_trace()
    assert np.isclose(fnet[0], -force1x)
    assert np.isclose(fnet[1], -force1y)
    
    
    

# %% Calculate for all angles 

solver = solve_eqn(0)
tensions = calc_tension(solver)
plot(solver, tensions)
plt.title('0 deg')
plt.savefig('0-deg.png', transparent=True)
s1 = pd.Series(tensions)
sanity_check(solver)

solver = solve_eqn(45)
tensions = calc_tension(solver)
plot(solver, tensions)
plt.title('45 deg')
plt.savefig('45-deg.png', transparent=True)
s2 = pd.Series(tensions)
sanity_check(solver)

solver = solve_eqn(90)
tensions = calc_tension(solver)
plot(solver, tensions)
plt.title('90 deg')
plt.savefig('90-deg.png', transparent=True)
s3 = pd.Series(tensions)
sanity_check(solver)

series = {'0-deg' : s1, '45-deg' : s2, '90-deg' : s3}
dft = pd.DataFrame(series)
dft.index.name = 'Link'
dft.to_csv('tension-output.csv')