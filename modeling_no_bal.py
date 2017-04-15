import sys
import math as mth
from scipy import *
from sympy import *
import random as rand
import numpy as np
from scipy.stats import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize
####################################
import time as t
from mpi4py import MPI
#################################################
import reading as r
import writing as w

class Modeling:
    def __init__(self): #конструктор       
        self.gamma1 = .0
        self.gamma2 = .0
        self.sigma = .0
        self.T = []
        self.K = 9
#######################################
    def from_read(self, a):
        self.gamma1 = a.gamma1
        self.gamma2 = a.gamma2
        self.sigma = a.sigma
        self.T.append(a.time)
#######################################
    def func_trend(self, time): #функция тренда
        return time ** self.gamma2
#######################################

    def rnd_volatility_model(self):
        xi = 1.
        eta = 0.8
        sigm = [rand.gammavariate(xi, eta) for i in range(self.K)]
        return sigm
####################################################
    def param(self, time): #вычисление параметров
        #gamm = self.rnd_drifts_model()
        sigm = self.rnd_volatility_model()
        a = list(map(lambda x, y: self.gamma1 * (self.func_trend(x) - self.func_trend(y)), time[1:], time[:-1]))
        b = list(map(lambda x, y, z: z * (self.func_trend(x) - self.func_trend(y)) ** 2, time[1:], time[:-1], sigm))
        # b = list(map(lambda x, y: self.sigma * (self.func_trend(x) - self.func_trend(y)) ** 2, time[1:], time[:-1]))
        return a, b
#######################################
    def delta_Z(self, time, N): #вычисление приращений
        a, b= self.param(time)
        delta_z = []
        for i in range(N - 1):
            delta_z.append(invgauss.rvs(a[i], b[i], size = self.K))
        return delta_z
#######################################
    def sum_delt_Z(self, deltaZ, N, i):
            sum_delt_Z = .0
            for j in range(i):
                sum_delt_Z += deltaZ[j]
            return sum_delt_Z

#######################################
    #показатель деградации
    def rate_of_degrad(self, deltaZ, N):  
        tmp_z = []
        tmp_z.append(deltaZ[0]) 
        for i in range(1, N):
            sum = self.sum_delt_Z(deltaZ, N, i)
            tmp_z.append(sum)
        return tmp_z
######################################################
    #функция правдоподобия
    def likelihood_function(self, est_theta, time, N, deltaZ):
        k = self.K
        temp = list(map(lambda x, y: x ** est_theta[1] - y ** est_theta[1], time[1:], time[:-1]))
        tmp = list(map(lambda x, y, z: (z - est_theta[0] * (x ** est_theta[1] - y ** est_theta[1])) ** 2, time[1:], time[:-1], deltaZ))
        dT = np.log(np.array(temp))
        ds = np.sum(np.array(tmp) / deltaZ)
        dzlog = np.sum(np.log(deltaZ))
        #################################################
        lnL = (N - 1) * (k) * np.log(est_theta[2]) / 2 + \
            (k) * np.sum(dT) - (N - 1) * (k) * np.log(2 * mth.pi) / 2 - \
            3 * dzlog / 2 - (est_theta[2] / (2 * est_theta[0] ** 2)) * ds
        return -lnL

####################################################
    #функция правдоподобия для модели со случайным эффектом
    def likelihood_function_rnd_eff(self, est_theta, time, N, deltaZ):
        k = self.K

        tmp = list(map(lambda x, y, z: est_theta[3] * (z - est_theta[0] * (x ** est_theta[1] - y ** est_theta[1])) ** 2 + \
            (2 * (est_theta[0]** 2) * z ), time[1:], time[:-1], deltaZ))

        temp = list(map(lambda x, y: x ** est_theta[1] - y ** est_theta[1], time[1:], time[:-1]))
        try:
            dT = np.sum(np.log(np.array(temp)))
            dslog1 = (N - 1) * k * (mth.log(2) + 2 * mth.log(est_theta[0] if est_theta[0]>0 else 1) + \
                mth.log(est_theta[3] if est_theta[3]>0 else 1)) + np.sum(np.log(np.array(deltaZ)))
            dslog = np.sum(np.log(np.array(tmp)))
            lnL = (N - 1) * (k) * mth.log(mth.gamma(est_theta[2] + 1/2)) +\
                (est_theta[2] + 0.5) * dslog1 + k * dT - \
               (N - 1) * (k) * mth.log(mth.gamma(est_theta[2])) - \
              (N - 1) * (k)  * mth.log(2 * mth.pi) / 2 - \
              3 * np.sum(np.log(np.array(deltaZ))) / 2 - \
              (est_theta[2] + 0.5) * dslog -\
            (N - 1) * (k) * est_theta[2] * mth.log(est_theta[3] if est_theta[3]>0 else 1) 
            
        except:
            print('EXCEPTION')
            lnL = 0
        else:
            return -lnL
        return -lnL
#####################################################
    #минимизация функции правдоподобия
    def max_likelihood_estimation(self, N, time, deltaZ):
       est_theta0 = np.array([0.5, 0.5, 0.8])
       lam_fun = lambda est_theta: self.likelihood_function(est_theta, time, N, deltaZ)

       tmp = minimize(lam_fun, est_theta0, method='nelder-mead', options={'xtol': 1e-12, 'disp': True})
       res = tmp.x
       return res
#####################################################
#минимизация функции правдоподобия со случайным эффектом
    def max_likelihood_estimation_rnd_eff(self, N, time, deltaZ):
       est_theta0 = np.array([0.5, 0.5, 1., 0.8 ])
       lam_fun = lambda est_theta: self.likelihood_function_rnd_eff(est_theta, time, N, deltaZ)
       tmp = minimize(lam_fun, est_theta0, method='nelder-mead', options={'xtol': 1e-12, 'maxfev':1000, 'disp': True})
       res = tmp.x
       return res

#######################################
    #Монте-Карло для данных с фиксированным эффектом
    def Monte_Karlo_method(self, M, gamma1, gamma2, sigma, N, time):
        write_data = w.WriteData()
        #######################
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        ############################
        t0 = t.time()
        if size != 0:
            delta = int(M / size)
            deltaZ1 = [self.delta_Z(time, N) for i in range(delta * rank, delta * (rank + 1))]

            deltaZ = [self.delta_Z(time, N) for i in range(M)]
            deltaZ_rank = deltaZ[delta*rank : delta*(rank + 1)]

            write_data.WritingInFile(['deltaZ'], [deltaZ], 'deltaZ' + str(rank) + '.txt')
            # оценивание параметров модели
            est_param_maxlnL = np.array(list(map(lambda x: self.max_likelihood_estimation(N, time, x), deltaZ)))
            write_data.WritingInFile(['est_param'], [est_param_maxlnL], 'est_param' + str(rank) + '.txt')
            MG1, MG2, MS = est_param_maxlnL[:, 0], est_param_maxlnL[:, 1], est_param_maxlnL[:, -1]
            discrepancy_gam1 = np.sum(abs(MG1 - self.gamma1) / self.gamma1) / int(M / size)
            discrepancy_gam2 = np.sum(abs(MG2 - self.gamma2) / self.gamma2) / int(M / size)
            discrepancy_sigm = np.sum(abs(MS - self.sigma) / self.sigma) / int(M / size)
            discr_est = np.array([discrepancy_gam1, discrepancy_gam2, discrepancy_sigm])
            data = np.empty(3, dtype=double)
            comm.Allreduce(discr_est, data, op=MPI.SUM)
            print("data on rank: %d is: "%rank, discr_est)
            t1 = t.time()
            time_local = np.array([float(format(t1 - t0))])
            time = np.empty(1, dtype=double)
            comm.Allreduce(time_local, time, op=MPI.MAX)
        MPI.Finalize()
        print("data on rank: %d is: "%rank, data/size)
        
        print ("time: ", format(time))
        return data/size
    #Монте-Карло для данных со случайным эффектом
####################################
    def Monte_Karlo_method1(self, M, N, time):
        a = 0.5
        al = 0.5
        xi = 1.
        eta = 0.8
        write_data = w.WriteData()

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        t0 = t.time()
        if size != 0:
            deltaZ = [self.delta_Z(time, N) for i in range(int(M/size))]
            write_data.WritingInFile(['deltaZ'], [deltaZ], 'deltaZ' + str(rank) + '.txt')
            # оценивание параметров модели
            est_param_maxlnL = np.array(list(map(lambda x: self.max_likelihood_estimation_rnd_eff(N, time, x), deltaZ)))
            write_data.WritingInFile(['est_param'], [est_param_maxlnL], 'est_param' + str(rank) + '.txt')
            MG1, MG2, MXiEta = np.array(est_param_maxlnL[:,0]), np.array(est_param_maxlnL[:,1]), np.array(est_param_maxlnL[:,2] * est_param_maxlnL[:,-1])
            discrepancy_gam1 = np.sum(abs(MG1 - self.gamma1) / self.gamma1) / int(M / size)
            discrepancy_gam2 = np.sum(abs(MG2 - self.gamma2) / self.gamma2) / int(M / size)
            discrepancy_sigm = np.sum(abs(MXiEta - xi * eta) / (xi * eta)) / int(M / size)
            discr_est = np.array([discrepancy_gam1, discrepancy_gam2, discrepancy_sigm])
            data = np.empty(3, dtype=double)
            comm.Allreduce(discr_est, data, op=MPI.SUM)
            print("data on rank: %d is: "%rank, discr_est)
            t1 = t.time()
            time_local = np.array([float(format(t1 - t0))])
            time = np.empty(1, dtype=double)
            comm.Allreduce(time_local, time, op=MPI.MAX)
        MPI.Finalize()
        print("data on rank: %d is: "%rank, data/size)
        print ("time: ", format(t1 - t0))
        return data/size
##############################################
    # def plot_graph(self, z, time):
        # a = 0.5
        # al = 2.0
        # xi = 1.0
        # eta = 0.8
        # plt.xlabel('time')
        # plt.ylabel('Rate of degradation')
        # #plt.axis([0, 10, 0, 20])
        # plt.ylim(0,100)
        # y = np.array([90]*10)
        # plt.text(1, 91, r'$Threshold$')
        # #plt.text(0.5, 85, r'$\gamma_1 = 0.5,\ \gamma_2 = 2,\ \xi = 0.85,\ \eta = 0.01$')
        # plt.text(0.5, 85, r'$\gamma_1 = 0.5,\ \gamma_2 = 2,\ \sigma = 0.8$')
        # plt.text(0.5, 80, r'$m(t) = t^{\gamma_2}$', fontsize = 15, color = 'red')
        # h = plt.plot(z, color = 'grey')
        # p = [self.func_trend(el) for el in time]
        # k = plt.plot(p, 'r', linewidth = 4)
        # f = plt.plot(y, '')
        # plt.show()

    # def plot_ecdf(self, x, M):
        
        # sorted_data = np.sort(x)
        # yvals=np.arange(len(sorted_data))/float(len(sorted_data))
        # plt.plot(sorted_data,yvals)
        # plt.show()


    # def plot_f(self, est_param, M, time):
        # eG1 = est_param[0]
        # eG2 = est_param[1]
        # eS = est_param[2]
        # plt.xlabel('t')
        # plt.ylabel('S(t)')
        # plt.ylim(0, 1)
        # plt.xlim(5, 10)
        # z = 90.0
        # est_F = []
        # t = np.linspace(0, 10, 100)
        # # вероятность безотказной работы за время
        # for i in range(100):
            # est_F.append([])
            # est_F[i].append(1.0)
            # for j in range(1,100):
                # est_F[i].append(invgauss.cdf(z, eG1[i] * t[j] ** eG2[i], eS[i] * (t[j] ** eG2[i]) ** 2))
            # p = plt.plot(t, est_F[i], color = 'grey')
        # F = []
        # F.append(1.0)
        # for j in range(1,100):
            # F.append(invgauss.cdf(z, self.gamma1 * t[j] **self.gamma2, self.sigma * (t[j] **self.gamma2) ** 2))
        # h = plt.plot(t, F, 'r', linewidth = 4)
        # plt.show()