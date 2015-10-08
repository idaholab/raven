#/*************************************************/
#/*           DO NOT MODIFY THIS HEADER           */
#/*                                               */
#/*                     BISON                     */
#/*                                               */
#/*    (c) 2015 Battelle Energy Alliance, LLC     */
#/*            ALL RIGHTS RESERVED                */
#/*                                               */
#/*   Prepared by Battelle Energy Alliance, LLC   */
#/*     Under Contract No. DE-AC07-05ID14517      */
#/*     With the U. S. Department of Energy       */
#/*                                               */
#/*     See COPYRIGHT for full restrictions       */
#/*************************************************/

#!/Usr/bin/env python2.5


# Pellet Type 1
# Obligatory parameters
Pellet1= {}
Pellet1['type'] = 'discrete'
Pellet1['quantity'] = 1
Pellet1['mesh_density'] = 'coarse'

Pellet1['outer_radius'] = 0.0041
Pellet1['inner_radius'] = 0
Pellet1['height'] = 2*5.93e-3
Pellet1['dish_spherical_radius'] = 1.01542e-2
Pellet1['dish_depth'] = 3e-4
Pellet1['chamfer_width'] = 5.0e-4
Pellet1['chamfer_height'] = 1.6e-4

# Pellet Collection

pellets = [Pellet1]

# Stack options
pellet_stack = {}
pellet_stack['default_parameters'] = False

pellet_stack['merge_pellets'] = 'point'   # choose between 'yes', 'no', 'point' or 'surface'
pellet_stack['higher_order'] = True
pellet_stack['angle'] = 0

# Clad: Geometry of the clad
clad = {}
clad['mesh_density'] = 'coarse'
clad['gap_width'] = 8e-5
clad['bot_gap_height'] = 1e-3
clad['clad_thickness'] = 5.6e-4
clad['top_bot_clad_height'] = 2.24e-3
clad['plenum_fuel_ratio'] = 0.15

clad['with_liner'] = False
clad['liner_width'] = 5.0e-5


# Meshing parameters
mesh = {}
mesh['default_parameters'] = False

# Parameters of mesh density 'coarse'
coarse = {}
coarse['pellet_r_interval'] = 3
coarse['pellet_z_interval'] = 2
coarse['pellet_dish_interval'] = 2
coarse['pellet_flat_top_interval'] = 1
coarse['pellet_chamfer_interval'] = 1
coarse['clad_radial_interval'] = 2
coarse['clad_sleeve_scale_factor'] = 1
coarse['cap_radial_interval'] = 3
coarse['cap_vertical_interval'] = 3
coarse['pellet_slices_interval'] = 4
coarse['pellet_angular_interval'] = 3
coarse['clad_angular_interval'] = 6
