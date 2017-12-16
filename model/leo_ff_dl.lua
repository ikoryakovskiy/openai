--[[
--  LEO rigid body model
--  Based on leoconfig.xml by Erik Schuitema
--
--  Note:
--    Erik's XML allows child nodes to be connected at
--    a random anchor, instead of just {0, 0, 0}. To move the
--    anchor to {0, 0, 0}, it has to be subtracted from the
--    com and the children's joint_frame.
--
--  Author:
--    Wouter Caarls <wouter@caarls.org>
--    Manuel Kudruss <manuel.kudruss@iwr.uni-heidelberg.de>
--]]

-- strict checks for undefined variables
-- require 'SRC.strict'

-- load module with convenience functions
-- utils = require 'SRC.utils'

-- initial
armICangle = -0.26 -- = -15*degtorad

-- for model
torsoBoomMass = 0.91326
torsoCMX = -0.00102
torsoCMZ = 0.009945
torsoIYY = 0.004676374
torsoheight = (0.24155)
torsoHipDistX = (0.00273)
torsoHipDistZ = (-torsoheight/2)
torsoMass = 0.94226
boomMass = 0.860
boomCMY = 0.835
boomLength = 1.70
boomIZZ = 0.31863
boomVirtualMassX = 0 --(boomCMY^2*boomMass + boomIZZ)/(boomLength^2)
--boomVirtualMassZ = boomMass*boomCMY/boomLength
--boomExtForce = (boomVirtualMassZ - boomVirtualMassX)*(-9.81)
torsoBoomMass = torsoMass + boomVirtualMassX
torsoCMX = -0.00102
torsoCMZ = 0.009945
torsoIYY = 0.004676374
torsoBoomCMX = (torsoMass*torsoCMX + boomVirtualMassX*torsoHipDistX)/(torsoMass + boomVirtualMassX)
torsoBoomCMZ = (torsoMass*torsoCMZ + boomVirtualMassX*torsoHipDistZ)/(torsoMass + boomVirtualMassX)
torsoBoomIYY = torsoIYY + torsoMass*((torsoCMX-torsoBoomCMX)^2 + (torsoCMZ-torsoBoomCMZ)^2) + boomVirtualMassX*((torsoHipDistX-torsoBoomCMX)^2 + (torsoHipDistZ-torsoBoomCMZ)^2)

shoulderoffsety = 0.11975
armMass = 0.095
armCMX = 0.05308
armCMZ = -0.14260
armIYY = 0.00087318718
armJointX = (-0.00543)
armJointY = (shoulderoffsety + 0.005)
armJointZ = (0.091275)

upleglength = (0.116)
interlegdist = (0.06390)

uplegMass = 0.19978
uplegCMX = 0.00285
uplegCMZ = -0.00481 - upleglength/2
uplegIYY = 0.000273133

uplegLeftJointX = (torsoHipDistX)
uplegLeftJointY = (interlegdist/2)
uplegLeftJointZ = (torsoHipDistZ)

uplegRightJointX = (torsoHipDistX)
uplegRightJointY = (-interlegdist/2)
uplegRightJointZ = (torsoHipDistZ)

loleglength = 0.1085

lolegMass = 0.13691
lolegCMX = 0.00804 + 0.00405
lolegCMZ = -0.00867 - loleglength/2
lolegIYY = 0.000153379

lolegLeftJointX = (0.0)
lolegLeftJointY = (0.0)
lolegLeftJointZ = (-upleglength)

lolegRightJointX = (0.0)
lolegRightJointY = (0.0)
lolegRightJointZ = (-upleglength)

footMass = 0.07319
footCMX = 0.00048+0.009
footCMZ = 0.00461-0.03559
footIYY = 0.000048812

footLeftJointX = (0. + 0.00405)
footLeftJointY = (0.)
footLeftJointZ = (-loleglength)

footRightJointX = (0.+0.00405)
footRightJointY = (0.0)
footRightJointZ = (-loleglength)

-- for visuals
dxlheight = 0.051
dxlwidth = 0.036
dxldepth = 0.036
torsowidth = 0.160
handoffsetx = 0.11730
handoffsetz = -0.26947
handwheelradius = 0.013
footwheelradius = 0.013
kneewheeloffsetx = 0.031
kneewheeloffsetz = 0.0165
footlength = 0.081

local function iyymatrix(iyy)
  local inertiadontcare = 1.000
  return {
           {inertiadontcare, 0., 0.},
           {0.0, iyy, 0.},
           {0., 0., inertiadontcare}
         }
end

local function rotymatrix(angle)
  return {
           {  math.cos(angle), 0.0, math.sin(angle)},
           {  0.0,             1.0, 0.0            },
           { -math.sin(angle), 0.0, math.cos(angle)}
         }
end

joints = {
  fixed = {},
  boom = {
    { 0., 0., 0., 1., 0., 0.},
    { 0., 0., 0., 0., 0., 1.},
    { 0., 1., 0., 0., 0., 0.},
  },
  lhinge = {
    { 0., 1., 0., 0., 0., 0.}
  },
  hinge = {
    { 0., -1., 0., 0., 0., 0.}
  }
}

colors = {
  bracket = { 0.8, 0.8, 0.8 },
  wheel = { 0.1, 0.1, 0.1 },
  dynamixels = { 0.5, 0.5, 0.5 }
}

-- **************************************************************************
-- *** HELPERS **************************************************************
-- **************************************************************************

local function get_point_by_name (container, name)
    for key, value in ipairs(container) do
        if value["name"] == name then
            -- value["coordinates"] = value["point"]
            return value
        end
    end
    print('container does not contain point with name: ', name)
end

function control(state, action)
    -- Convert voltage to torque
    Kt = 0.006325081
    G = 212.6
    R = 4.6
    dof = 9
    for ii = 0, dof-1 do
--      print(action[ii])
      action[ii] = Kt*G*(action[ii] - Kt*G*state[dof + ii])/R;
--      print(action[ii])
      -- action[ii] = action[ii] - 0.0*state[dof + ii]; -- Friction
    end
    return {0.0, 0.0, 0.0, action[3], action[4], action[5], action[6], action[7], action[8]}
end

-- **************************************************************************
-- *** VISUALS **************************************************************
-- **************************************************************************

visuals = {
  torso = {
    {
      name = "main_body",
      dimensions = { 0.010, torsowidth, torsoheight },
      color = colors.bracket,
      mesh_center = { (-0.02316),(0.), (0.) },
      src = "meshes/unit_cube.obj",
    },
    {
      name = "upper_bracket",
      dimensions = { 0.03, torsowidth + (0.02), 0.03 },
      color = colors.bracket,
      mesh_center = { (0.), (0.), (0.08) },
      src = "meshes/unit_cube.obj",
    },
    {
      name = "lower_bracket",
      dimensions = { 0.02, torsowidth + (0.02), 0.025 },
      color = colors.bracket,
      mesh_center = { (0.), (0.), (-0.04) },
      src = "meshes/unit_cube.obj",
    },
    {
      name = "shoulder_dynamixel",
      dimensions = { (dxldepth), (dxlwidth), (dxlheight) },
      color = colors.dynamixel,
      mesh_center = {
        (-0.00543),
        (shoulderoffsety - dxlwidth/2),
        torsoheight/2 - (0.0175 + dxlheight/2)
      },
      src = "meshes/unit_cube.obj",
    },

  },
  arm = {
    {
      -- Should be rotated
      name = "upper_arm_box",
      dimensions = { (0.020), (0.010), (0.140) },
      color = { 0.8, 0.8, 0.8},
      mesh_center = {
        (0.01527), 0.0, (-0.05699)
      },
      src = "meshes/unit_cube.obj",
    },
    {
      -- Should be rotated
      name = "lower_arm_box",
      dimensions = { (0.020), (0.010), (0.150) },
      color = colors.bracket,
      mesh_center = {
        (0.06989), 0.0, (-0.18714)
      },
      src = "meshes/unit_cube.obj",
    },
    {
      name = "hand_wheel",
      dimensions = {
        (2*handwheelradius), (2*handwheelradius), (2*handwheelradius)
      },
      color = colors.wheel,
      mesh_center = {
        (handoffsetx), 0.0, (handoffsetz)
      },
      src = "meshes/unit_sphere_lowres.obj"
    }
  },
  upper_arm = {
    {
      -- Should be rotated
      name = "upper_arm_box",
      dimensions = { (0.020), (0.010), (0.140) },
      color = { 0.8, 0.8, 0.8},
      mesh_center = {
        (0.01527), 0.005, (-0.05699)
      },
      src = "meshes/unit_cube.obj",
    },
  },
  lower_arm = {
    {
      -- Should be rotated
      name = "lower_arm_box",
      dimensions = { (0.020), (0.010), (0.150) },
      color = colors.bracket,
      mesh_center = {
        (0.06989), 0.005, (-0.18714)
      },
      src = "meshes/unit_cube.obj",
    },
    {
      name = "hand_wheel",
      dimensions = {
        (2*handwheelradius), (2*handwheelradius), (2*handwheelradius)
      },
      color = colors.wheel,
      mesh_center = {
        (handoffsetx), 0.005, (handoffsetz)
      },
      src = "meshes/unit_sphere_lowres.obj"
    }
  },
  upperleg = {
    {
      name = "bracket",
      dimensions = { (0.00755), (0.034), (upleglength + 0.02) },
      color = colors.bracket,
      mesh_center = { (dxldepth/2), (0.), (-upleglength/2) },
      src = "meshes/unit_cube.obj"
    },
    {
      name = "hip_dynamixel",
      dimensions = { (dxldepth), (dxlwidth), (dxlheight) },
      color = colors.dynamixel,
      mesh_center = {
        (0.0), (0.), upleglength/2 - (0.015) - upleglength/2
      },
      src = "meshes/unit_cube.obj"
    },
    {
      name = "knee_dynamixel",
      dimensions = { (dxldepth), (dxlwidth), (dxlheight) },
      color = colors.dynamixel,
      mesh_center = {
        (0.0), (0.), -upleglength/2 + (0.015) -upleglength/2
      },
      src = "meshes/unit_cube.obj"
    }
  },
  lowerleg = {
    {
      name = "bracket",
      dimensions = { (0.00755), (0.034), loleglength + (0.02) },
      color = colors.bracket,
      mesh_center = {
        (dxldepth/2 +0.00405), (0.), (-loleglength/2)
      },
      src = "meshes/unit_cube.obj"
    },
    {
      name = "ankle_dynamixel",
      dimensions = { (dxldepth), (dxlwidth), (dxlheight) },
      color = colors.dynamixel,
      mesh_center = {
        (0.0+0.00405), (0.), -loleglength/2 + (0.015) -loleglength/2
      },
      src = "meshes/unit_cube.obj"
    },
    {
      name = "knee_wheel",
      dimensions = {
        (2*footwheelradius), (2*footwheelradius), (2*footwheelradius)
      },
      color = colors.wheel,
      mesh_center = {
        (kneewheeloffsetx), (0.), loleglength/2 + (kneewheeloffsetz) -loleglength/2
      },
      src = "meshes/unit_sphere_lowres.obj"
    }
  },
  foot = {
    {
      name = "box",
      dimensions = { (0.070), (0.019), (0.0175) },
      color = colors.bracket,
      mesh_center = { (0. + 0.009), (0.), (0. - 0.03559) },
      src = "meshes/unit_cube.obj"
    },
    {
      name = "ankle_wheel",
      dimensions = {
        (2*footwheelradius), (2*footwheelradius), (2*footwheelradius)
      },
      color = colors.wheel,
      mesh_center = { (-footlength/2 + 0.009), (0.), (0. - 0.03559) },
      src = "meshes/unit_sphere_lowres.obj"
    },
    {
      name = "toe_wheel",
      dimensions = {
        (2*footwheelradius), (2*footwheelradius), (2*footwheelradius) },
      color = colors.wheel,
      mesh_center = { (footlength/2 + 0.009), (0.), (0. - 0.03559) },
      src = "meshes/unit_sphere_lowres.obj"
    }
  }
}

-- **************************************************************************
-- *** CONTACT POINTS *******************************************************
-- **************************************************************************

contact_points = {
  {
    name = "root",
    point = { 0.0, 0.0, 0.0 },
    coordinates = { 0.0, 0.0, 0.0 },
    body = "torso",
    color = {1., 1., 1.}
  },
  {
    name = "heel_right",
    point = { (-footlength/2 + 0.009), (0.), (0. - 0.03559 - footwheelradius) },
    coordinates = { (-footlength/2 + 0.009), (0.), (0. - 0.03559 - footwheelradius) },
    body = "footright",
    color = {1., 1., 1.}
  },
  {
    name = "tip_right",
    point = { (footlength/2 + 0.009), (0.), (0. - 0.03559 - footwheelradius) },
    coordinates = { (footlength/2 + 0.009), (0.), (0. - 0.03559 - footwheelradius) },
    body = "footright",
    color = {1., 1., 1.}
  },
  {
    name = "heel_left",
    point = { (-footlength/2 + 0.009), (0.), (0. - 0.03559 - footwheelradius) },
    coordinates = { (-footlength/2 + 0.009), (0.), (0. - 0.03559 - footwheelradius) },
    body = "footleft",
    color = {1., 1., 1.}
  },
  {
    name = "tip_left",
    point = { (footlength/2 + 0.009), (0.), (0. - 0.03559 - footwheelradius) },
    coordinates = { (footlength/2 + 0.009), (0.), (0. - 0.03559 - footwheelradius) },
    body = "footleft",
    color = {1., 1., 1.}
  },
}

-- **************************************************************************
-- *** CONSTRAINT SETS
-- **************************************************************************

constraint_sets = {
-- Single contacts
  single_support_heel_right = {
     { point = "heel_right", normal = { 1, 0, 0,}, },
     { point = "heel_right", normal = { 0, 0, 1,}, },
   },
  single_support_heel_left = {
     { point = "heel_left", normal = { 1, 0, 0,}, },
     { point = "heel_left", normal = { 0, 0, 1,}, },
   },
  single_support_tip_right = {
     { point = "tip_right", normal = { 1, 0, 0,}, },
     { point = "tip_right", normal = { 0, 0, 1,}, },
   },
  single_support_tip_left = {
     { point = "tip_left", normal = { 1, 0, 0,}, },
     { point = "tip_left", normal = { 0, 0, 1,}, },
   },
-- Double contacts
  single_support_flat_right = {
     { point = "heel_right", normal = { 1, 0, 0,}, },
     { point = "heel_right", normal = { 0, 0, 1,}, },
     { point = "tip_right", normal = { 0, 0, 1,}, },
   },
  single_support_flat_left = {
     { point = "heel_left", normal = { 1, 0, 0,}, },
     { point = "heel_left", normal = { 0, 0, 1,}, },
     { point = "tip_left", normal = { 0, 0, 1,}, },
   },
  double_support_heel = {
     { point = "heel_left", normal = { 1, 0, 0,}, },
     { point = "heel_left", normal = { 0, 0, 1,}, },
     { point = "heel_right", normal = { 1, 0, 0,}, },
     { point = "heel_right", normal = { 0, 0, 1,}, },
   },
  double_support_tip = {
     { point = "tip_left", normal = { 1, 0, 0,}, },
     { point = "tip_left", normal = { 0, 0, 1,}, },
     { point = "tip_right", normal = { 1, 0, 0,}, },
     { point = "tip_right", normal = { 0, 0, 1,}, },
   },
  double_support_hr_tl = {
     { point = "tip_left", normal = { 1, 0, 0,}, },
     { point = "tip_left", normal = { 0, 0, 1,}, },
     { point = "heel_right", normal = { 1, 0, 0,}, },
     { point = "heel_right", normal = { 0, 0, 1,}, },
   },
  double_support_hl_tr = {
     { point = "heel_left", normal = { 1, 0, 0,}, },
     { point = "heel_left", normal = { 0, 0, 1,}, },
     { point = "tip_right", normal = { 1, 0, 0,}, },
     { point = "tip_right", normal = { 0, 0, 1,}, },
   },
-- Triple contacts
  double_support_fl_hr = {
    { point = "heel_right", normal = { 1, 0, 0,}, },
    { point = "heel_right", normal = { 0, 0, 1,}, },

    { point = "heel_left", normal = { 1, 0, 0,}, },
    { point = "heel_left", normal = { 0, 0, 1,}, },
    { point = "tip_left", normal = { 0, 0, 1,}, },
  },
  double_support_fl_tr = {
    { point = "tip_right", normal = { 1, 0, 0,}, },
    { point = "tip_right", normal = { 0, 0, 1,}, },

    { point = "heel_left", normal = { 1, 0, 0,}, },
    { point = "heel_left", normal = { 0, 0, 1,}, },
    { point = "tip_left", normal = { 0, 0, 1,}, },
  },
  double_support_fr_hl = {
    { point = "heel_right", normal = { 1, 0, 0,}, },
    { point = "heel_right", normal = { 0, 0, 1,}, },
    { point = "tip_right", normal = { 0, 0, 1,}, },

    { point = "heel_left", normal = { 1, 0, 0,}, },
    { point = "heel_left", normal = { 0, 0, 1,}, },
  },
  double_support_fr_tl = {
    { point = "heel_right", normal = { 1, 0, 0,}, },
    { point = "heel_right", normal = { 0, 0, 1,}, },
    { point = "tip_right", normal = { 0, 0, 1,}, },

    { point = "tip_left", normal = { 1, 0, 0,}, },
    { point = "tip_left", normal = { 0, 0, 1,}, },
  },
-- All four in contact
  double_support = {
  --  -- NOTE: one has to be careful with multiple position contacts that should
  --  --       are supposed to keep a body straight on the floor
    { point = "heel_right", normal = { 1, 0, 0,}, },
    { point = "heel_right", normal = { 0, 0, 1,}, },
    { point = "tip_right", normal = { 0, 0, 1,}, },

    { point = "heel_left", normal = { 1, 0, 0,}, },
    { point = "heel_left", normal = { 0, 0, 1,}, },
    { point = "tip_left", normal = { 0, 0, 1,}, },
  },
}

-- **************************************************************************
-- *** MODEL
-- **************************************************************************

model = {
    -- assign variable to model
    points = contact_points,
    constraint_sets = constraint_sets,

    -- model definition
    gravity = {0., 0., -9.81},
    -- configuration of the environment
    configuration = {
    axis_front = { 1, 0, 0},
    axis_up    = { 0, 0, 1},
    axis_right = { 0, -1, 0},
    },
    frames = {
    {
      -- Note: torso is unmoved from original position
      name = "torso",
      parent = "ROOT",
      body = {
        mass = torsoBoomMass,
          com = {torsoBoomCMX, 0., torsoBoomCMZ},
          inertia = iyymatrix(torsoBoomIYY)
      },
      joint = joints.boom,
      visuals = visuals.torso,
      points = { -- draw contact points
        root = get_point_by_name(contact_points, "root"),
      }
    },
    {
      name = "arm",
      parent = "torso",
      body = {
        mass = armMass,
        com = {armCMX, 0., armCMZ},
        inertia = iyymatrix(armIYY)
      },
      joint = joints.fixed,
      joint_frame = {
        r = {armJointX, armJointY, armJointZ},
        E = {
            { 1, 0 , 0},
            { 0, 1,  0},
            { 0, 0,  1},
        },
      },
      visuals = visuals.arm
    },
    {
      name = "upperlegleft",
      parent = "torso",
      body = {
        mass = uplegMass,
        com = {uplegCMX, 0.0, uplegCMZ},
        inertia = iyymatrix(uplegIYY),
      },
      joint = joints.hinge,
      joint_frame = {
        r = {uplegLeftJointX, uplegLeftJointY, uplegLeftJointZ}
      },
      visuals = visuals.upperleg
    },
    {
      name = "upperlegright",
      parent = "torso",
      body = {
        mass = uplegMass,
        com = {uplegCMX, 0., uplegCMZ},
        inertia = iyymatrix(uplegIYY),
      },
      joint = joints.hinge,
      joint_frame = {
        r = {uplegRightJointX, uplegRightJointY, uplegRightJointZ}
      },
      visuals = visuals.upperleg
    },
    {
      name = "lowerlegleft",
      parent = "upperlegleft",
      body = {
        mass = lolegMass,
        com = {lolegCMX, 0.0, lolegCMZ},
        inertia = iyymatrix(lolegIYY)
      },
      joint = joints.hinge,
      joint_frame = {
        r = {lolegLeftJointX, lolegLeftJointY, lolegLeftJointZ}
      },
      visuals = visuals.lowerleg
    },
    {
      name = "lowerlegright",
      parent = "upperlegright",
      body = {
        mass = lolegMass,
        com = {lolegCMX, 0., lolegCMZ},
        inertia = iyymatrix(lolegIYY)
      },
      joint = joints.hinge,
      joint_frame = {
        r = {lolegRightJointX, lolegRightJointY, lolegRightJointZ}
      },
      visuals = visuals.lowerleg
    },
    {
      name = "footleft",
      parent = "lowerlegleft",
      body = {
        mass = footMass,
        com = {footCMX, 0.0, footCMZ},
        inertia = iyymatrix(footIYY)
      },
      joint = joints.hinge,
      joint_frame = {
        r = {footLeftJointX, footLeftJointY, footLeftJointZ}
      },
      visuals = visuals.foot,
      points = { -- draw contact points
        heel_left = get_point_by_name(contact_points, "heel_left"),
        tip_left = get_point_by_name(contact_points, "tip_left"),
      }
    },
    {
      name = "footright",
      parent = "lowerlegright",
      body = {
        mass = footMass,
        com = {footCMX, 0., footCMZ},
        inertia = iyymatrix(footIYY)
      },
      joint = joints.hinge,
      joint_frame = {
        r = {footRightJointX, footRightJointY, footRightJointZ}
      },
      visuals = visuals.foot,
      points = { -- draw contact points
        heel_right = get_point_by_name(contact_points, "heel_right"),
        tip_right = get_point_by_name(contact_points, "tip_right"),
      }
    }
  }, -- end frames
} -- end model

-- NOTE: it is important to return the created model
return model
