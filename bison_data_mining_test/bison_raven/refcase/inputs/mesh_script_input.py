#/*************************************************/
#/*           DO NOT MODIFY THIS HEADER           */
#/*                                               */
#/*                     BISON                     */
#/*                                               */
#/*    (c) 2014 Battelle Energy Alliance, LLC     */
#/*            ALL RIGHTS RESERVED                */
#/*                                               */
#/*   Prepared by Battelle Energy Alliance, LLC   */
#/*     Under Contract No. DE-AC07-05ID14517      */
#/*     With the U. S. Department of Energy       */
#/*                                               */
#/*     See COPYRIGHT for full restrictions       */
#/*************************************************/

#!/Usr/bin/env python2.5

""" Pellet default geometry
Pellet1['outer_radius'] = 0.0045
Pellet1['inner_radius'] = 0.0030
Pellet1['height'] = 23.5e-03
"""

# PK6-3 fuel rod from the Super-Ramp experiment. Ref:
# Djurle, S., 1984. The Super-Ramp Project. Final Report STUDSVIK-STSR-32.

Pellet= {}
Pellet['type'] = 'smeared'
Pellet['quantity'] = 28
Pellet['mesh_density'] = 'mdensity'
Pellet['outer_radius'] = 0.004573
Pellet['inner_radius'] = 0.
Pellet['height'] = 1.125e-02
Pellet['dish_spherical_radius'] = 0.
Pellet['dish_depth'] = 0.
Pellet['chamfer_width'] = 0.
Pellet['chamfer_height'] = 0.


# Pellet Collection
pellets = [Pellet]

# Stack options
pellet_stack = {}
pellet_stack['default_parameters'] = False
pellet_stack['merge_pellets'] = 'point'   # choose between 'yes', 'no', 'point' or 'surface'
pellet_stack['higher_order'] = True
pellet_stack['angle'] = 0

"""Pellet stack default parameters:
 pellet_stack['merge_pellets'] = 'yes'
 pellet_stack['higher_order'] = False
 pellet_stack['angle'] = 0
"""

# Clad: Geometry of the clad
clad = {}
clad['mesh_density'] = 'mdensity'
clad['gap_width'] = 7.3e-05
clad['bot_gap_height'] = 4.e-03
clad['clad_thickness'] = 7.265e-04
clad['top_bot_clad_height'] = 1.15e-02
#clad['plenum_fuel_ratio'] = 0.12
clad['top_gap_height'] = 3.3e-02

clad['with_liner'] = False
clad['liner_width'] = 0.

""" Clad default geometry:
    clad_dictionary['gap_width'] = 1.5e-04
    clad_dictionary['bot_gap_width'] = 1.5e-02
    clad_dictionary['clad_width'] = 7.25e-04
    clad_dictionary['top_bot_clad_width'] = 2.e-03
    clad_dictionary['plenum_fuel_ratio'] = 0.045 ###########

    clad_dictionary['with_liner'] = False
"""

# Meshing parameters
mesh = {}
mesh['default_parameters'] = False

# Parameters of mesh density 'mdensity'
mdensity = {}
mdensity['pellet_r_interval'] = 15
mdensity['pellet_z_interval'] = 1
mdensity['pellet_dish_interval'] = 0
mdensity['pellet_flat_top_interval'] = 0
mdensity['pellet_chamfer_interval'] = 0
mdensity['clad_radial_interval'] = 4
mdensity['clad_sleeve_scale_factor'] = 1
mdensity['cap_radial_interval'] = 6
mdensity['cap_vertical_interval'] = 3
mdensity['pellet_slices_interval'] = 4
mdensity['pellet_angular_interval'] = 6
mdensity['clad_angular_interval'] = 12

# Parameters of the mesh density 'medium'
#medium = {}
#medium['pellet_r_interval'] = 11
#medium['pellet_z_interval'] = 3
#medium['pellet_dish_interval'] = 6
#medium['pellet_flat_top_interval'] = 3
#medium['pellet_chamfer_interval'] = 2
#medium['clad_radial_interval'] = 4
#medium['clad_sleeve_scale_factor'] = 0.5
#medium['cap_radial_interval'] = 4
#medium['cap_vertical_interval'] = 3
#medium['pellet_slices_interval'] = 16
#medium['pellet_angular_interval'] = 12
#medium['clad_angular_interval'] = 16

# Parameter of the mesh density 'fine'
#fine = {}
#fine['pellet_r_interval'] = 22
#fine['pellet_z_interval'] = 5
#fine['pellet_flat_top_interval'] = 6
#fine['pellet_chamfer_interval'] = 4
#fine['pellet_dish_interval'] = 12
#fine['clad_radial_interval'] = 5
#fine['clad_sleeve_scale_factor'] = 0.5
#fine['cap_radial_interval'] = 8
#fine['cap_vertical_interval'] = 5
#fine['pellet_slices_interval'] = 32
#fine['pellet_angular_interval'] = 24
#fine['clad_angular_interval'] = 16



""" Meshing default parameters:

    coarse['pellet_r_size'] = 0.000683
    coarse['pellet_z_size'] = 0.00593
    coarse['pellet_dish_interval'] = 3
    coarse['pellet_flat_top_interval'] = 2
    coarse['pellet_chamfer_interval'] = 1
    coarse['clad_radial_interval'] = 3
    coarse['clad_sleeve_scale_factor'] = 1
    coarse['radial_interval_cap'] = 6
    coarse['vertical_interval_cap'] = 3
    coarse['pellet_slices_interval'] = 4


    medium['pellet_r_size'] = 0.000372
    medium['pellet_z_size'] = 0.00148
    medium['pellet_dish_interval'] = 6
    medium['pellet_flat_top_interval'] = 3
    medium['pellet_chamfer_interval'] = 2
    medium['clad_radial_interval'] = 4
    medium['clad_sleeve_scale_factor'] = 0.5
    medium['radial_interval_cap'] = 4
    medium['vertical_interval_cap'] = 3
    medium['pellet_slices_interval'] = 16


    fine['pellet_r_size'] = 0.0001863
    fine['pellet_z_size'] = 0.0007413
    fine['pellet_flat_top_interval'] = 6
    fine['pellet_chamfer_interval'] = 4
    fine['pellet_dish_interval'] = 12
    fine['clad_radial_interval'] = 4
    fine['clad_sleeve_scale_factor'] = 0.5
    fine['radial_interval_cap'] = 4
    fine['vertical_interval_cap'] = 5
    fine['pellet_slices_interval'] = 32
"""
