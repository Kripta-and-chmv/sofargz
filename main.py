import reading as r
import modeling_proc as m
import writing as w
import numpy as np
if __name__ == '__main__':
    print("1")
    M = 10
    init_data = r.ReadInitData()
    write_data = w.WriteData()
    model = m.Modeling()
    init_data.Reading('input.txt')
    gamma1, gamma2, sigma, N, time = init_data.GetData()
    N = int(N)
    model.from_read(init_data)
    #############################3
    #arg = model.integr(time, N)
    ###############################
    #discr = model.Monte_Karlo_method1(M, N, time)
    discr = model.Monte_Karlo_method(M, gamma1, gamma2, sigma, N, time)
    # write_data.WritingInFile(['est_param'], [est_param], 'estimation.txt')
    write_data.WritingInFile(['discr'], [discr], 'discr.txt')
    ###################################
    #G1, G2, S = init_data.GetEst('estimation.txt')
    #est_param = np.array([G1, G2, S])
    #model.plot_f(est_param, M, time)
    #deltaZ = model.delta_Z(time, N)
    #est_param_maxlnL = model.max_likelihood_estimation_rnd_eff(N, time, deltaZ)
    #############################
    #z = model.rate_of_degrad(deltaZ, N)
    #model.plot_graph(z, time)
    #model.plot_ecdf(est_param, M)
    #print()