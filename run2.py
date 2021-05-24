# -*- coding: utf-8 -*-
import pdb
import numpy as np
import pandas as pd
from dataclasses import dataclass
import sympy
from functools import cached_property

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


@dataclass
class Point:
    """Point dataclass."""
    name : str
    loc : np.ndarray
    
    
@dataclass
class Element:
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
            Left hand side of the equation's coefficients for each variable.
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
        """Force vectors of solution."""
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
        out = {}
        for name, values in self.vector_dict.items():
            magnitude = (values[0]**2 + values[1]**2 + values[2]**2)**0.5
            out[name] = magnitude
        return out
    
    
    # @cached_property
    # def node_magnitude_dict(self) -> dict:
    #     """Retrieve internal force magnitudes for each link."""
    #     out = {}
    #     for name, values in self.vector_dict.items():
    #         ftype, node, link = name.split('_')
    #         newname = ftype + '_' + node
    #         try:
    #             out[newname] = out[newname] + values
    #         except KeyError:
    #             out[newname] = values
        
    #     for name, value in out.items():
    #         out[name] = (value[0]**2 + value[1]**2 + value[2]**2)**0.5
    #     return out
    
    
    @cached_property
    def elements(self):
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
            
    
    # @cached_property
    # def element_tensions(self) -> dict:
    #     vectors = self.vector_dict.copy()
    #     elements = self.elements
        
    #     tensions1 = {}
    #     tensions2 = {}
    #     for vector_name, vector_value in vectors.items():
            
    #         ftype, node, link = vector_name.split('_')
    #         element = self.elements[link]
    #         if ftype == 'F':
    #             if node == element.point1.name:
    #                 tension = element.tension1(vector_value)
    #                 tensions1[link] = tension
                    
    #             elif node == element.point2.name:
    #                 tension = element.tension2(vector_value)
    #                 tensions2[link] = tension
                    
    #             else:
    #                 raise ValueError('Some weird error has happened')
                
        
    #     for key in tensions1:
    #         assert np.isclose(tensions1[key], tensions2[key])
    #     return tensions1
            
        
            
    
    
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
angle = 0

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
    
    # %% Define position vectors
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
    
    rba = rb - ra
    rda = rd - ra
    rac = ra - rc
    rbg = rb - rg
    rbf = rb - rf
    rfd = rf - rd
    rec = re - rc
    
    
    
    # %% Define forces
    
        
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
    
    
    #%% Construct equilibrium equations
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
    coeffs7['F_A_2_x'] = rba[0]
    coeffs7['F_A_2_y'] = rba[1]
    coeffs7['F_A_3_x'] = rba[0]
    coeffs7['F_A_3_y'] = rba[1]
    solver.add_equation(coeffs7, 0)
    
    
    coeffs8 = {}
    coeffs8['M_A_3_z'] = 1
    coeffs8['F_A_3_x'] = -rda[0]
    coeffs8['F_A_3_y'] = -rda[1]
    solver.add_equation(coeffs8, 0)
    
    coeffs9 = {}
    coeffs9['M_A_2_z'] = 1
    coeffs9['F_C_2_x'] = rac[0]
    coeffs9['F_C_2_y'] = rac[1]
    solver.add_equation(coeffs9, 0)
    
    
    coeffs10 = {}
    coeffs10['F_D_3_x'] = rfd[0]
    coeffs10['F_D_3_y'] = rfd[1]
    solver.add_equation(coeffs10, 0)
    
    coeffs11 = {}
    coeffs11['F_C_2_x'] = rec[0]
    coeffs11['F_C_2_y'] = rec[1]
    solver.add_equation(coeffs11, 0)
    
    
    coeffs12 = {}
    coeffs12['F_F_BF_x'] = rbf[0]
    coeffs12['F_F_BF_y'] = rbf[1]
    rhs = rbg[1] * force1y
    solver.add_equation(coeffs12, rhs)
    solver.solve()
    return solver





solver = solve_eqn(0)
mag = solver.magnitude_dict
t = solver.element_tensions