# 2d coordinate transformation, N x 3 Numpy array as input,
# z coordinate is untouched
# The coordinate frame rotates counter-clock-wise by angle_rad
import math
import numpy as np


def coord_transform_v2(angle_rad, x_shift, y_shift, input):
    # Rotational
    # Angle in radians
    out = np.zeros((input.shape[0], 3))
    # print(angle_rad)
    out[:, 0] = input[:, 0] * np.cos(angle_rad) + input[:, 1] * np.sin(angle_rad)
    out[:, 1] = input[:, 1] * np.cos(angle_rad) - input[:, 0] * np.sin(angle_rad)
    # print(out)
    # Translational transformation
    out[:, 0] = out[:, 0] - x_shift
    out[:, 1] = out[:, 1] + y_shift
    return out


def get_mtransform(angle_rad, x_shift, y_shift):
    return (np.array([[math.cos(angle_rad), -math.sin(angle_rad), 0],
                     [math.sin(angle_rad), math.cos(angle_rad), 0],
                      [0, 0, 1]]), np.array([-x_shift, y_shift, 0]))

# M(x+M_inv*a)=Mx+a


def coord_transform(m_rotate, m_shift, input):
    return np.matmul(input, m_rotate) + m_shift
