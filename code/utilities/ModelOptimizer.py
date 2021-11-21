# imports
import numpy as np

class ModelOptimizer:

    seed = 123

    def __init__(self, seed=123):
        self.seed = seed

    def __scores(self, rmsecs, rmsecvs):
        rmsec_arr = np.array(rmsecs)
        rmsecv_arr = np.array(rmsecvs)
        deviations = rmsec_arr - rmsecv_arr
        abs_deviations = abs(deviations)
        scores = 0.5*abs_deviations + 0.3*deviations + 0.1*rmsec_arr + 0.1*rmsecv_arr
        return [round(s, 4) for s in scores]
    
    def opt_params(self, rmsecs, rmsecvs, param_combos):
        '''
            Returns a dict of the best optimal parameters
        '''
        param_scores = self.__scores(rmsecs, rmsecvs)
        min_score = min(param_scores)
        max_score = max(param_scores)
        opt_ix = param_scores.index(min_score)
        opt_params = param_combos[opt_ix]
        opt_dict = {'index': opt_ix,
                    'rmsec': rmsecs[opt_ix],
                    'rmsecv': rmsecvs[opt_ix],
                    'params': opt_params,
                    'score': min_score,
                    'highest_possible_score': max_score}
        return opt_dict
