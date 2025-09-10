import numpy as np

from rail.core.stage import RailStage
from rail.estimation.estimator import CatEstimator


class CatEstimatorWrapper:
    """Helper class to use a CatEstimator with in memory data
    """    
    def __init__(
        self,
        cat_estimator: CatEstimator,
        param_names: list[str],
        point_estimate: str='zmode',
    ):
        """Constructor
        
        Parameters
        ----------
        cat_estimator:
            CatEstimator to wrap
            
        param_names:
            Parameter names

        point_estimate:
            Which point estimate to use
        """
        self._estimator = cat_estimator
        self._param_names = param_names
        self._point_estimate = point_estimate        
        self._estimator.open_model(**self._estimator.config)

    def __call__(self, vals):
        """Evaluation function

        Parameters
        ----------
        vals: 
            Should have shape [N_params, N_values]
            
        Returns
        -------
        np.ndarray[..., N_params, N_values] estimates using different values of parameter
        """        
        n_objs = vals.shape[-1]
        table_like = {key_: val_.flatten() for key_, val_ in zip(self._param_names, vals)}
        RailStage.data_store.clear()
        self._estimator._input_length = n_objs
        self._estimator._initialize_run()
        self._estimator._process_chunk(0, n_objs, table_like, True)
        estimates = self._estimator._output_handle.data
        return estimates.ancil[self._point_estimate]

    

class CatEstimatorDerivativeWrapper:
    """Helper class to compute derivatives using scipy stats

    This will compute d z / d alpha for one input parameter, keeping the others fixed
    """    
    def __init__(
        self,
        cat_estimator: CatEstimator,
        free_param: str,
        init_values: dict[str, float],
        point_estimate: str='zmode',
    ):
        """Constructor
        
        Parameters
        ----------
        cat_estimator:
            CatEstimator to wrap
            
        free_param:
            Parameter to evaluate derivative w.r.t

        init_values:
            Initial values for all paramters

        point_estimate:
            Which point estimate to use
        """
        self._estimator = cat_estimator
        self._free_param = free_param
        self._init_values = init_values
        self._point_estimate = point_estimate
        self._estimator.open_model(**self._estimator.config)
        
    def __call__(self, vals: np.ndarray) -> np.ndarray:
        """Evaluation function

        Parameters
        ----------
        vals: 
            Should have shape [..., N_values]
            
        Returns
        -------
        np.ndarray[..., N_values] estimates using different values of free parameter
        """
        n_vals = vals.shape[-1]
        ones = np.ones(vals.shape)
        table_like = {key_: np.squeeze(ones.flatten()*vals_) for key_, vals_ in self._init_values.items()}
        if vals.size > 1:
            table_like[self._free_param] = np.squeeze(vals.flatten())
        else:
            table_like[self._free_param] = vals.flatten()
        RailStage.data_store.clear()
        self._estimator._input_length = n_vals
        self._estimator._initialize_run()
        self._estimator._process_chunk(0, n_vals, table_like, True)
        estimates = self._estimator._output_handle.data
        return np.reshape(estimates.ancil[self._point_estimate], vals.shape)
    

class CatEstimatorJacobianWrapper:
    """Helper class to compute Jacobians using scipy stats

    This will compute d z / d alpha for all the input parameters
    """    

    def __init__(
        self,
        cat_estimator: CatEstimator,
        param_names: list[str],
        point_estimate: str='zmode',
    ):
        """Constructor
        
        Parameters
        ----------
        cat_estimator:
            CatEstimator to wrap
            
        param_names:
            Parameter names

        point_estimate:
            Which point estimate to use
        """
        self._estimator = cat_estimator
        self._param_names = param_names
        self._point_estimate = point_estimate        
        self._estimator.open_model(**self._estimator.config)

    def __call__(self, vals):
        """Evaluation function

        Parameters
        ----------
        vals: 
            Should have shape [..., N_params, N_values]
            
        Returns
        -------
        np.ndarray[..., N_params, N_values] estimates using different values of parameter
        """        
        n_objs = vals.shape[-1]
        table_like = {key_: val_.flatten() for key_, val_ in zip(self._param_names, vals)}
        RailStage.data_store.clear()
        self._estimator._input_length = n_objs
        self._estimator._initialize_run()
        self._estimator._process_chunk(0, n_objs, table_like, True)
        estimates = self._estimator._output_handle.data
        return estimates.ancil[self._point_estimate]


class CatEstimatorHessianWrapper:
    """Helper class to compute Hessians using scipy stats

    This will compute d^2 z / d alpha_1 d_alpha_2 for all the input parameters
    """    

    def __init__(
        self,
        cat_estimator: CatEstimator,
        param_names: list[str],
        point_estimate: str='zmode',
    ):
        """Constructor
        
        Parameters
        ----------
        cat_estimator:
            CatEstimator to wrap
            
        param_names:
            Parameter names

        point_estimate:
            Which point estimate to use
        """
        self._estimator = cat_estimator
        self._param_names = param_names
        self._point_estimate = point_estimate                
        self._estimator.open_model(**self._estimator.config)

    def __call__(self, vals):
        """Evaluation function

        Parameters
        ----------
        vals: 
            Should have shape [..., N_params, N_params, N_values]
            
        Returns
        -------
        np.ndarray[..., N_params, N_params, N_values] estimates using different values of parameter
        """
        n_objs = vals.size
        table_like = {key_: val_.flatten() for key_, val_ in zip(self._param_names, vals)}
        RailStage.data_store.clear()
        self._estimator._input_length = n_objs
        self._estimator._initialize_run()
        self._estimator._process_chunk(0, n_objs, table_like, True)
        estimates = self._estimator._output_handle.data
        return np.reshape(estimates.ancil[self._point_estimate], vals.shape[1:])
