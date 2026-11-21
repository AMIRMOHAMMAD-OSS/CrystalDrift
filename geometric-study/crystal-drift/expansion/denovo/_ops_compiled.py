from math import *
import numpy as np
from numba import njit
PARAMS=['IGW_EP', 'IGW_SPP', 'IGW_TA', 'IGW_TA1', 'IGW_TA2', 'TA', 'TA1', 'TA2', 'TA3', 'exp_cos_AA', 'exp_cos_AA1', 'exp_cos_AA2', 'fac_AA', 'fac_AA1', 'fac_AA2', 'min_SPP', 'shift_AA2', 'w_SPP', 2, 4, 5, 6, 7]
TYPES=['T', 'bcc', 'bent', 'cuboct', 'cuboct_max', 'hex_bipyr', 'hex_plan_max', 'hex_pyr', 'oct', 'oct_legacy', 'oct_max', 'pent_bipyr', 'pent_plan', 'pent_plan_max', 'pent_pyr', 'see_saw_rect', 'sq_bipyr', 'sq_face_cap_trig_pris', 'sq_plan', 'sq_plan_max', 'sq_pyr', 'sq_pyr_legacy', 'tet', 'tet_max', 'tri_bipyr', 'tri_plan', 'tri_plan_max', 'tri_pyr']
@njit(cache=True)
def gramschmidt(v,u):
    return v-(np.dot(v,u)/np.dot(u,u))*u

@njit(cache=True)
def geom(v,typ,params,comp_azi):
    n_neighbors=len(v)
    dist=np.empty(n_neighbors)
    rij_norm=np.empty_like(v)
    for i in range(n_neighbors):
        dist[i]=np.sqrt(np.dot(v[i],v[i]))
        rij_norm[i]=v[i]/dist[i]
    left_of_unity=1.-1.e-12
    very_small=1.e-12
    fac_bcc=1./exp(-0.5)
    gaussthetak=np.zeros(1)
    ops=np.zeros(1)
    qsp_theta=np.zeros((1,n_neighbors,n_neighbors-1))
    norms=np.zeros((1,n_neighbors,n_neighbors-1))
    ipi = 1 / pi
    piover2 = pi / 2.0
    onethird = 1 / 3
    twothird = 2 / 3.0
    for j in range(n_neighbors):
        zaxis = rij_norm[j]
        kc = 0
        for k in range(n_neighbors):
            if j != k:
                for idx in range(1):
                    pass
                    pass
                tmp = max(-1.0, min(np.dot(zaxis, rij_norm[k]), 1.0))
                thetak = acos(tmp)
                xaxis = gramschmidt(rij_norm[k], zaxis)
                if np.linalg.norm(xaxis) < very_small:
                    flag_xaxis = True
                else:
                    xaxis = xaxis / np.linalg.norm(xaxis)
                    flag_xaxis = False
                if comp_azi:
                    flag_yaxis = True
                    yaxis = np.cross(zaxis, xaxis)
                    if np.linalg.norm(yaxis) > very_small:
                        yaxis = yaxis / np.linalg.norm(yaxis)
                        flag_yaxis = False
                for idx in range(1):
                    if typ in [2, 21]:
                        tmp = params[2] * (thetak * ipi - params[5])
                        qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp)
                        norms[idx][j][kc] += 1
                    elif typ in [25, 26, 22, 23]:
                        tmp = params[2] * (thetak * ipi - params[5])
                        gaussthetak[idx] = exp(-0.5 * tmp * tmp)
                        if typ in [26, 23]:
                            qsp_theta[idx][j][kc] += gaussthetak[idx]
                            norms[idx][j][kc] += 1
                    elif typ in [0, 27, 20, 14, 7]:
                        tmp = params[0] * (thetak * ipi - 0.5)
                        qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp)
                        norms[idx][j][kc] += 1
                    elif typ in [18, 8, 9, 3, 4]:
                        if thetak >= params[15]:
                            tmp = params[1] * (thetak * ipi - 1.0)
                            qsp_theta[idx][j][kc] += params[17] * exp(-0.5 * tmp * tmp)
                            norms[idx][j][kc] += params[17]
                    elif typ in [15, 24, 16, 11, 5, 10, 19, 6]:
                        if thetak < params[15]:
                            tmp = params[0] * (thetak * ipi - 0.5) if typ != 6 else params[2] * (fabs(thetak * ipi - 0.5) - params[5])
                            qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp)
                            norms[idx][j][kc] += 1
                    elif typ in [12, 13]:
                        tmp = 0.4 if thetak <= params[5] * pi else 0.8
                        tmp2 = params[2] * (thetak * ipi - tmp)
                        gaussthetak[idx] = exp(-0.5 * tmp2 * tmp2)
                        if typ == 13:
                            qsp_theta[idx][j][kc] += gaussthetak[idx]
                            norms[idx][j][kc] += 1
                    elif typ == 1 and j < k:
                        if thetak >= params[15]:
                            tmp = params[1] * (thetak * ipi - 1.0)
                            qsp_theta[idx][j][kc] += params[17] * exp(-0.5 * tmp * tmp)
                            norms[idx][j][kc] += params[17]
                    elif typ == 17 and thetak < params[8]:
                        tmp = params[3] * (thetak * ipi - params[6])
                        qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp)
                        norms[idx][j][kc] += 1
                for m in range(n_neighbors):
                    if m != j and m != k and (not flag_xaxis):
                        tmp = max(-1.0, min(np.dot(zaxis, rij_norm[m]), 1.0))
                        thetam = acos(tmp)
                        x_two_axis_tmp = gramschmidt(rij_norm[m], zaxis)
                        norm = np.linalg.norm(x_two_axis_tmp)
                        if norm < very_small:
                            flag_xtwoaxis = True
                        else:
                            xtwoaxis = x_two_axis_tmp / norm
                            phi = acos(max(-1.0, min(np.dot(xtwoaxis, xaxis), 1.0)))
                            flag_xtwoaxis = False
                            if comp_azi:
                                phi2 = atan2(np.dot(xtwoaxis, yaxis), np.dot(xtwoaxis, xaxis))
                        if typ in [24, 16, 11, 5, 10, 19, 6, 15] and thetam >= params[15]:
                            tmp = params[1] * (thetam * ipi - 1.0)
                            qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp)
                            norms[idx][j][kc] += 1
                        if not flag_xaxis and (not flag_xtwoaxis):
                            for idx in range(1):
                                if typ in [25, 26, 22, 23]:
                                    tmp = params[2] * (thetam * ipi - params[5])
                                    tmp2 = cos(params[12] * phi) ** params[9]
                                    tmp3 = 1 if typ in [26, 23] else gaussthetak[idx]
                                    qsp_theta[idx][j][kc] += tmp3 * exp(-0.5 * tmp * tmp) * tmp2
                                    norms[idx][j][kc] += 1
                                elif typ in [12, 13]:
                                    tmp = 0.4 if thetam <= params[5] * pi else 0.8
                                    tmp2 = params[2] * (thetam * ipi - tmp)
                                    tmp3 = cos(phi)
                                    tmp4 = 1 if typ == 13 else gaussthetak[idx]
                                    qsp_theta[idx][j][kc] += tmp4 * exp(-0.5 * tmp2 * tmp2) * tmp3 * tmp3
                                    norms[idx][j][kc] += 1
                                elif typ in [0, 27, 20, 14, 7]:
                                    tmp = cos(params[12] * phi) ** params[9]
                                    tmp3 = params[0] * (thetam * ipi - 0.5)
                                    qsp_theta[idx][j][kc] += tmp * exp(-0.5 * tmp3 * tmp3)
                                    norms[idx][j][kc] += 1
                                elif typ in [18, 8, 9]:
                                    if thetak < params[15] and thetam < params[15]:
                                        tmp = cos(params[12] * phi) ** params[9]
                                        tmp2 = params[0] * (thetam * ipi - 0.5)
                                        qsp_theta[idx][j][kc] += tmp * exp(-0.5 * tmp2 * tmp2)
                                        if typ == 9:
                                            qsp_theta[idx][j][kc] -= tmp * params[21] * params[22]
                                        norms[idx][j][kc] += 1
                                elif typ in [24, 16, 11, 5, 10, 19, 6]:
                                    if thetam < params[15] and thetak < params[15]:
                                        tmp = cos(params[12] * phi) ** params[9]
                                        tmp2 = params[0] * (thetam * ipi - 0.5) if typ != 6 else params[2] * (fabs(thetam * ipi - 0.5) - params[5])
                                        qsp_theta[idx][j][kc] += tmp * exp(-0.5 * tmp2 * tmp2)
                                        norms[idx][j][kc] += 1
                                elif typ == 1 and j < k:
                                    if thetak < params[15]:
                                        fac = 1 if thetak > piover2 else -1
                                        tmp = (thetam - piover2) / asin(1 / 3)
                                        qsp_theta[idx][j][kc] += fac * cos(3 * phi) * fac_bcc * tmp * exp(-0.5 * tmp * tmp)
                                        norms[idx][j][kc] += 1
                                elif typ == 15:
                                    if thetam < params[15] and thetak < params[15] and (phi < 0.75 * pi):
                                        tmp = cos(params[12] * phi) ** params[9]
                                        tmp2 = params[0] * (thetam * ipi - 0.5)
                                        qsp_theta[idx][j][kc] += tmp * exp(-0.5 * tmp2 * tmp2)
                                        norms[idx][j][kc] += 1.0
                                elif typ in [3, 4]:
                                    if thetam < params[15] and params[19] < thetak < params[18]:
                                        if params[19] < thetam < params[18]:
                                            tmp = cos(phi)
                                            tmp2 = params[20] * (thetam * ipi - 0.5)
                                            qsp_theta[idx][j][kc] += tmp * tmp * exp(-0.5 * tmp2 * tmp2)
                                            norms[idx][j][kc] += 1.0
                                        elif thetam < params[19]:
                                            tmp = 0.0556 * (cos(phi - 0.5 * pi) - 0.81649658)
                                            tmp2 = params[21] * (thetam * ipi - onethird)
                                            qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp) * exp(-0.5 * tmp2 * tmp2)
                                            norms[idx][j][kc] += 1.0
                                        elif thetam > params[18]:
                                            tmp = 0.0556 * (cos(phi - 0.5 * pi) - 0.81649658)
                                            tmp2 = params[21] * (thetam * ipi - twothird)
                                            qsp_theta[idx][j][kc] += exp(-0.5 * tmp * tmp) * exp(-0.5 * tmp2 * tmp2)
                                            norms[idx][j][kc] += 1.0
                                elif typ == 17 and (not flag_yaxis) and (thetak < params[8]):
                                    if thetam < params[8]:
                                        tmp = cos(params[13] * phi2) ** params[10]
                                        tmp2 = params[3] * (thetam * ipi - params[6])
                                    else:
                                        tmp = cos(params[14] * (phi2 + params[16])) ** params[11]
                                        tmp2 = params[4] * (thetam * ipi - params[7])
                                    qsp_theta[idx][j][kc] += tmp * exp(-0.5 * tmp2 * tmp2)
                                    norms[idx][j][kc] += 1
                kc += 1
    for idx in range(1):
        if typ in [25, 22, 2, 18, 8, 9, 3, 12]:
            ops[idx] = tmp_norm = 0.0
            for j in range(n_neighbors):
                ops[idx] += sum(qsp_theta[idx][j])
                tmp_norm += float(sum(norms[idx][j]))
            ops[idx] = ops[idx] / tmp_norm if tmp_norm > 1e-12 else 0.0
        elif typ in [0, 27, 15, 20, 24, 16, 14, 7, 11, 5, 10, 26, 23, 19, 13, 4, 6, 17]:
            ops[idx] = 0.0
            if n_neighbors > 1:
                for j in range(n_neighbors):
                    for k in range(len(qsp_theta[idx][j])):
                        qsp_theta[idx][j][k] = qsp_theta[idx][j][k] / norms[idx][j][k] if norms[idx][j][k] > 1e-12 else 0.0
                    ops[idx] = max(qsp_theta[idx][j]) if j == 0 else max(ops[idx], np.max(qsp_theta[idx][j]))
        elif typ == 1:
            ops[idx] = 0.0
            for j in range(n_neighbors):
                ops[idx] += sum(qsp_theta[idx][j])
            if n_neighbors > 3:
                ops[idx] = ops[idx] / float(0.5 * float(n_neighbors * (6 + (n_neighbors - 2) * (n_neighbors - 3))))
            else:
                ops[idx] = 0.0
        elif typ == 21:
            if n_neighbors > 1:
                dmean = np.mean(dist)
                acc = 0.0
                for d in dist:
                    tmp = params[18] * (d - dmean)
                    acc = acc + exp(-0.5 * tmp * tmp)
                for j in range(n_neighbors):
                    ops[idx] = max(qsp_theta[idx][j]) if j == 0 else max(ops[idx], np.max(qsp_theta[idx][j]))
                ops[idx] = acc * ops[idx] / float(n_neighbors)
            else:
                ops[idx] = 0.0
    return ops[0]
