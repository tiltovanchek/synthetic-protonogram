import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import random
import time
start = time.time()

def Ef(r, vect, Q_density):
    beg = vect[:-1] # нач элты улитки
    end = vect[1:]
    Q_dr = np.sqrt(np.sum(np.power((end - beg), 2), axis=1)) * Q_density
    coord_diff = r - 0.5 * (end + beg)
    Q_dr = np.stack((Q_dr, Q_dr, Q_dr), axis=1)
    numerator = Q_dr * coord_diff
    denominator = np.power(np.sum(np.power(coord_diff, 2), axis=1), 1.5)  # возведение в квадрат разностей координат, суммирование x,y,z поточечно, степень 3/2, получили массив куба расстояний до точек
    denominator = np.stack((denominator, denominator, denominator), axis=1)
    E_points = np.divide(numerator, denominator)
    E = np.sum(E_points, axis=0)
    return E

def H(r, vect, I):
    beg = vect[:-1] # нач элты улитки
    end = vect[1:]
    I = np.stack((I, I, I), axis=1)
    dr = (end - beg) * I
    coord_diff = r - 0.5 * (end + beg)
    numerator = np.cross(dr, coord_diff)
    denominator = np.power(np.sum(np.power(coord_diff, 2), axis=1), 1.5)  # возведение в квадрат разностей координат, суммирование x,y,z поточечно, степень 3/2, получили массив куба расстояний до точек
    denominator = np.stack((denominator, denominator, denominator), axis=1)
    H_points = np.divide(numerator, denominator)
    H = np.sum(H_points, axis=0)
    #Hx = I[i] * ((end[1][i] - beg[1][i]) * (r[2] - 0.5*(end[2][i] + beg[2][i])) - (end[2][i] - beg[2][i]) * (r[1] - 0.5*(end[1][i] + beg[1][i]))) / \
         #np.power((r[0] - 0.5*(end[0][i] + beg[0][i]))**2 + (r[1] - 0.5*(end[1][i] + beg[1][i]))**2 + (r[2] - 0.5*(end[2][i] + beg[2][i]))**2, 1.5)
    return H

def Runge(x, y, z, vx, vy, vz, t, t_end, dt):
    x_values = []
    y_values = []
    z_values = []
    vx_values = []
    vy_values = []
    vz_values = []
    time_values = []
    while t < t_end:
        B = H([x,y,z], vect, I) * 1e-7
        E = Ef([x,y,z], vect, Q_density) * 9 * 1e9
        dvx1 = dt * (E[0] + (vy * B[2] - vz * B[1])) * koef
        dvy1 = dt * (E[1] + (-vx * B[2] + vz * B[0])) * koef
        dvz1 = dt * (E[2] + (vx * B[1] - vy * B[0])) * koef

        dvx2 = dt * (E[0] + ((vy + dvy1 / 2) * B[2] - (vz + dvz1 / 2) * B[1])) * koef
        dvy2 = dt * (E[1] + (-(vx + dvx1 / 2) * B[2] + (vz + dvz1 / 2) * B[0])) * koef
        dvz2 = dt * (E[2] + ((vx + dvx1 / 2) * B[1] - (vy + dvy1 / 2) * B[0])) * koef

        dvx3 = dt * (E[0] + ((vy + dvy2 / 2) * B[2] - (vz + dvz2 / 2) * B[1])) * koef
        dvy3 = dt * (E[1] + (-(vx + dvx2 / 2) * B[2] + (vz + dvz2 / 2) * B[0])) * koef
        dvz3 = dt * (E[2] + ((vx + dvx2 / 2) * B[1] - (vy + dvy2 / 2) * B[0])) * koef

        dvx4 = dt * (E[0] + ((vy + dvy3) * B[2] - (vz + dvz3) * B[1])) * koef
        dvy4 = dt * (E[1] + (-(vx + dvx3) * B[2] + (vz + dvz3) * B[0])) * koef
        dvz4 = dt * (E[2] + ((vx + dvx3) * B[1] - (vy + dvy3) * B[0])) * koef

        vx += (dvx1 + dvx2 * 2 + dvx3 * 2 + dvx4) / 6
        vy += (dvy1 + dvy2 * 2 + dvy3 * 2 + dvy4) / 6
        vz += (dvz1 + dvz2 * 2 + dvz3 * 2 + dvz4) / 6

        x += vx * dt
        y += vy * dt
        z += vz * dt

        time_values.append(t)
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        vx_values.append(vx)
        vy_values.append(vy)
        vz_values.append(vz)

        t += dt
    res = [x_values, y_values, z_values, vx_values, vy_values, vz_values]
    return res

def snail():
    '''
    y = np.array([174, 185, 192, 192, 188, 181, 168, 152, 137, 126, 120, 117, 118, 125,
                  134, 152, 168, 189, 202, 217, 233, 249, 263, 277, 293, 305, 318, 330,
                  342, 354, 363, 372, 377, 378, 377, 372, 367, 360, 351, 339, 327, 312,
                  296, 284, 271, 252, 232, 210, 193, 190, 186, 183, 180, 178, 176, 174])
    y = (y - 174) * 1e-6 * 91.2 / 180
    z = np.array([330, 315, 300, 289, 277, 266, 253, 239, 226, 211, 195, 177, 162, 147,
                  137, 122, 113, 105, 103, 103, 104, 107, 112, 115, 122, 129, 139, 149,
                  163, 178, 197, 218, 238, 266, 287, 309, 328, 347, 366, 389, 409, 428,
                  446, 457, 467, 480, 490, 502, 509, 481, 452, 425, 401, 377, 357, 330])
    z = (-z + 330) * 1e-6 * 91.2 / 180

    angle = -55 * np.pi / 180
    angle_arr = angle * np.ones(len(y))
    y_new = y * np.cos(angle_arr) - z * np.sin(angle_arr)
    z_new = y * np.sin(angle_arr) + z * np.cos(angle_arr)
    y = y_new
    z = z_new

    x = np.zeros(len(y))
    vect = np.stack((x, y, z), axis=-1)
    '''
    x = np.zeros(56)
    fi = np.linspace(0, 2 * np.pi, 56)
    r = 1e-4
    y = r * np.cos(-fi + np.pi / 2)
    z = r * np.sin(-fi + np.pi / 2)
    vect = np.stack((x, y, z), axis=-1)
    return vect

def plot():
    plt.figure(figsize=(6, 6))
    plt.title('I = 10 kA, Q = 800 нКл, '
              'Qpl, N = 3000')
    plt.grid(lw=0.5, ls='--')
    plt.xlabel("x", fontsize=15)
    plt.ylabel("z", fontsize=15)
    plt.tick_params(labelsize=10)
    plt.xlim(-0.01, 0.01)
    plt.ylim(-0.01, 0.01)

def loop():

    for i in range(0, N_particles):
        angle_fi = random.uniform(0, 2 * np.pi)
        angle_theta = random.uniform(1e-2, 10 * 1e-2)
        v0 = 1.38 * 1e7 * np.sqrt(random.randint(1,1))
        x0 = S * np.sin(angle_theta) * np.cos(angle_fi)
        y0 = S - 2.8 * 1e-3
        z0 = S * np.sin(angle_theta) * np.sin(angle_fi)
        vx0 = v0 * np.sin(angle_theta) * np.cos(angle_fi)
        vy0 = v0
        vz0 = v0 * np.sin(angle_theta) * np.sin(angle_fi)

        t_end = 2 * 1e-3 / v0
        res = Runge(x0, y0, z0, vx0, vy0, vz0, t0, t_end, dt)
        x, y, z, vx, vy, vz = res
        #ax3D.plot(x,y,z)

        t1 = (l - y[-1]) / vy[-1]
        x_new = vx[-1]*t1 + x[-1]
        z_new = vz[-1]*t1 + z[-1]
        plt.scatter(x_new, z_new, s = 0.7, color = 'b')
        #ax.scatter(x_new, y_new, z_new, color='tab:red')

plot()
vect = snail()
N_segments = 55

beg = vect[:-8]  # нач элты улитки
end = vect[1:-7]
dr = end - beg
l_snail = np.sum(np.sqrt(np.sum(np.power(dr, 2), axis=1)))

Q = 0.3*1e-8
p = Q / l_snail
Q_density = np.array([p]*N_segments)
#Q_density[48:] = 0
I = np.array([-0.3*1e5]*N_segments)
koef = 0.9576 * 1e7

N_particles = 3000

S = 1.8 * 1e-3
l = 80 * 1e-3
#v0 = 4.2 * 1e7

dt = 1e-12
t0 = 0
#t_end = 0.5*1e-10
#t = np.arange(0, 0.5*1e-10, dt)



#fig1 = plt.figure(figsize=(10,10))
#ax3D = fig1.add_subplot(projection='3d')
loop()

beg = vect[:-1]  # нач элты улитки
end = vect[1:]
dr = end - beg

l = np.sum(np.power(dr, 2), axis=1)

print('Длина спирали: ', np.sum(np.sqrt(l) * 1e6))

end = time.time()
print(end - start)

plt.show()