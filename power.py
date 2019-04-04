def power(effect_size):
    alpha = 0.05
    power = 0.8
    from statsmodels.stats.power import  tt_ind_solve_power
    n = tt_ind_solve_power(effect_size = effect_size,
                                  alpha = alpha,
                                  power = power)
    return n
