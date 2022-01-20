from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, make_scorer
import numpy as np
import pandas as pd
from math import log
import xarray as xr
from sklearn.utils import resample
from numpy.random import uniform

from functools import partial
from sklearn.metrics._base import _average_binary_score
from sklearn.utils.multiclass import type_of_target


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def brier_score(y, predictions):
    return np.mean((predictions - y) ** 2)

def brier_skill_score(y, predictions):
    return 1.0 - brier_score(y, predictions) / brier_score(y, y.mean())

def max_csi(y, predictions, known_skew):
    """
    Compute normalized maximum CSI 
    """
    sr, pod, _ = precision_recall_curve(y, predictions)
    sr[sr==0] = 0.0001
    pod[pod==0] = 0.0001
    
    csi = calc_csi(sr, pod)
    idx = np.argmax(csi)
    
    max_csi = csi[idx]
    norm_max_csi = norm_csi(y, predictions, known_skew)
    bias = pod / sr

    return max_csi, norm_max_csi, bias[idx]

def bss_reliability(y, predictions):
    """
    Reliability component of BSS. Weighted MSE of the mean forecast probabilities
    and the conditional event frequencies. 
    """
    mean_fcst_probs, event_frequency, indices = reliability_curve(y, predictions, n_bins=10, return_indices=True)
    # Add a zero for the origin (0,0) added to the mean_fcst_probs and event_frequency
    counts = [1e-5]
    for i in indices:
        if i is np.nan:
            counts.append(1e-5)
        else:
            counts.append(len(i[0]))

    mean_fcst_probs[np.isnan(mean_fcst_probs)] = 1e-5
    event_frequency[np.isnan(event_frequency)] = 1e-5

    diff = (mean_fcst_probs-event_frequency)**2
    return np.average(diff, weights=counts)


def modified_precision(precision, known_skew, new_skew): 
    """
    Modify the success ratio according to equation (3) from 
    Lampert and Gancarski (2014). 
    """
    precision[precision<1e-5] = 1e-5
    term1 = new_skew / (1.0-new_skew)
    term2 = ((1/precision) - 1.0)
    
    denom = known_skew + ((1-known_skew)*term1*term2)
    
    return known_skew / denom 
    
def calc_sr_min(skew):
    pod = np.linspace(0,1,100)
    sr_min = (skew*pod) / (1-skew+(skew*pod))
    return sr_min 

def _binary_uninterpolated_average_precision(
            y_true, y_score, known_skew, new_skew, pos_label=1, sample_weight=None):
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
        # Return the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        if known_skew is not None:
            precision = modified_precision(precision, known_skew, new_skew)
        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

def min_aupdc(y_true, pos_label, average, sample_weight=None, known_skew=None, new_skew=None):
    """
    Compute the minimum possible area under the performance 
    diagram curve. Essentially, a vote of NO for all predictions. 
    """
    min_score = np.zeros((len(y_true)))
    average_precision = partial(_binary_uninterpolated_average_precision,
                                known_skew=known_skew,
                                new_skew=new_skew,
                                pos_label=pos_label)
    ap_min = _average_binary_score(average_precision, y_true, min_score,
                                 average, sample_weight=sample_weight)

    return ap_min


def calc_csi(precision, recall):
    """
    Compute the critical success index
    """
    precision[precision<1e-5] = 1e-3
    recall[recall<1e-5] = 1e-3
    
    csi = 1.0 / ((1/precision) + (1/recall) - 1.0)
    
    return csi 

def norm_csi(y_true, y_score, known_skew, pos_label=1, sample_weight=None):
    """
    Compute the normalized modified critical success index. 
    """
    new_skew = np.mean(y_true)
    precision, recall, _ = precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    if known_skew is not None:
        precision = modified_precision(precision, known_skew, new_skew)
    
    csi = calc_csi(precision, recall)
    max_csi = np.max(csi)
    ncsi = (max_csi - known_skew) / (1.0 - known_skew)
    
    return ncsi 
    


def norm_aupdc(y_true, y_score, known_skew, *, average="macro", pos_label=1,
                            sample_weight=None, min_method='random'):
    """
    Compute the normalized modified average precision. Normalization removes 
    the no-skill region either based on skew or random classifier performance. 
    Modification alters success ratio to be consistent with a known skew. 
  
    Parameters:
    -------------------
        y_true, array of (n_samples,)
            Binary, truth labels (0,1)
        y_score, array of (n_samples,)
            Model predictions (either determinstic or probabilistic)
        known_skew, float between 0 and 1 
            Known or reference skew (# of 1 / n_samples) for 
            computing the modified success ratio.
        min_method, 'skew' or 'random'
            If 'skew', then the normalization is based on the minimum AUPDC 
            formula presented in Boyd et al. (2012).
            
            If 'random', then the normalization is based on the 
            minimum AUPDC for a random classifier, which is equal 
            to the known skew. 
    
    
    Boyd, 2012: Unachievable Region in Precision-Recall Space and Its Effect on Empirical Evaluation, ArXiv
    """
    new_skew = np.mean(y_true)

    y_type = type_of_target(y_true)
    if y_type == "multilabel-indicator" and pos_label != 1:
        raise ValueError("Parameter pos_label is fixed to 1 for "
                         "multilabel-indicator y_true. Do not set "
                         "pos_label or set pos_label to 1.")
    elif y_type == "binary":
        # Convert to Python primitive type to avoid NumPy type / Python str
        # comparison. See https://github.com/numpy/numpy/issues/6784
        present_labels = np.unique(y_true).tolist()
        if len(present_labels) == 2 and pos_label not in present_labels:
            raise ValueError(
                f"pos_label={pos_label} is not a valid label. It should be "
                f"one of {present_labels}"
            )
    average_precision = partial(_binary_uninterpolated_average_precision,
                                known_skew=known_skew,
                                new_skew=new_skew,
                                pos_label=pos_label)
    
    ap = _average_binary_score(average_precision, y_true, y_score,
                                 average, sample_weight=sample_weight)
    
    if min_method == 'random':
        ap_min = known_skew 
    elif min_method == 'skew':
        ap_min = min_aupdc(y_true, 
                       pos_label, 
                       average,
                       sample_weight=sample_weight,
                       known_skew=known_skew, 
                       new_skew=new_skew)
    
    naupdc = (ap - ap_min) / (1.0 - ap_min)

    return naupdc


# ESTABLISH THE SCORING METRICS FOR THE CROSS-VALIDATION
scorer_dict = {'auc': make_scorer(score_func=roc_auc_score,
                                     greater_is_better=True,
                                     needs_proba=True,
                                    )    ,
           'aupdc': make_scorer(score_func=average_precision_score,
                                     greater_is_better=True,
                                     needs_proba=True,
                                    )    ,
           'aupdc_norm': make_scorer(score_func=norm_aupdc,
                                     greater_is_better=True,
                                     needs_proba=True,
                                    ),
           'bss' : make_scorer(score_func=brier_skill_score,
                                     greater_is_better=True,
                                     needs_proba=True,
                                    ),
           }

class ContingencyTable:
    ''' Calculates the values of the contingency table.
    param: y, True binary labels. shape = [n_samples] 
    param: predictions, predictions binary labels. shape = [n_samples]
    ContingencyTable calculates the components of the contigency table, but ignoring correct negatives. 
    Can use determinstic and probabilistic input.     
    '''
    def __init__( self, y, predictions ):
        hits = np.sum( np.where(( y == 1) & ( predictions == 1 ), 1, 0 ) )
        false_alarms = np.sum( np.where(( y == 0) & ( predictions == 1 ), 1, 0 ) )
        misses = np.sum( np.where(( y == 1) & ( predictions == 0), 1, 0 ) )
        corr_negs = np.sum( np.where(( y == 0) & ( predictions == 0 ),  1, 0 ) )

        self.table = np.array( [ [hits, misses], [false_alarms, corr_negs]], dtype=float)
        # Hit: self.table[0,0]
        # Miss: self.table[0,1]
        # False Alarms: self.table[1,0]
        # Corr Neg. : self.table[1,1] 

    def calc_pod(self):
        '''
        Probability of Detection (POD) or Hit Rate. 
        Formula: hits / hits + misses
        '''
        ##print self.table[0,0] / (self.table[0,0] + self.table[0,1])
        return self.table[0,0] / (self.table[0,0] + self.table[0,1])

    def calc_pofd(self):
        '''
        Probability of False Detection.
        Formula: false alarms / false alarms + correct negatives
        '''
        return self.table[1,0] / (self.table[1,0] + self.table[1,1])

    def calc_sr(self):
        '''
        Success Ratio (1 - FAR).
        Formula: hits / (hits+false alarms)
        '''
        if self.table[0,0] + self.table[1,0] == 0.0:
            return 1.
        else:
            return self.table[0,0] / (self.table[0,0] + self.table[1,0])

    @staticmethod
    def calc_bias(pod, sr):
        '''
        Frequency Bias.
        Formula: POD / SR ; (hits + misses) / (hits + false alarms)  
        '''
        sr[np.where(sr==0)] = 1e-5
        return pod / sr

    @staticmethod
    def calc_csi(pod, sr):
        '''
        Critical Success Index.
        Formula: Hits / ( Hits+Misses+FalseAlarms)
        '''
        sr[np.where(sr==0)] = 1e-5
        pod[np.where(pod==0)] = 1e-5
        return 1. /((1./sr) + (1./pod) - 1.)

def performance_curve(y, predictions, bins=np.arange(0, 1, 0.005), deterministic=False ):
    ''' 
    Generates the POD and SR for a series of probability thresholds 
    to produce performance diagram (Roebber 2009) curves
    '''
    if deterministic:
        table = ContingencyTable( y, predictions )
        pod = table.calc_pod( )
        sr = table.calc_sr( )
    else:
        tables = [ ContingencyTable(y.astype(int), np.where(np.round(predictions,10)
                >= round(p,5),1,0).astype(int)) for p in bins]

        pod = np.array([t.calc_pod() for t in tables])
        sr = np.array([t.calc_sr() for t in tables])
        pofd = np.array([t.calc_pofd() for t in tables])
    return pod, pofd, sr

def reliability_curve(y, predictions, n_bins=10, return_indices=False):
    """
    Generate a reliability (calibration) curve. 
    Bins can be empty for both the mean forecast probabilities 
    and event frequencies and will be replaced with nan values. 
    Unlike the scikit-learn method, this will make sure the output
    shape is consistent with the requested bin count. The output shape
    is (n_bins+1,) as I artifically insert the origin (0,0) so the plot
    looks correct. 
    """
    bin_edges = np.linspace(0,1, n_bins+1)
    bin_indices = np.clip(
                np.digitize(predictions, bin_edges, right=True) - 1, 0, None
                )

    indices = [np.where(bin_indices==i+1)
               if len(np.where(bin_indices==i+1)[0]) > 0 else np.nan for i in range(n_bins) ]

    mean_fcst_probs = [np.nan if i is np.nan else np.mean(predictions[i]) for i in indices]
    event_frequency = [np.nan if i is np.nan else np.sum(y[i]) / len(i[0]) for i in indices]

    # Adding the origin to the data
    mean_fcst_probs.insert(0,0)
    event_frequency.insert(0,0)
    
    if return_indices:
        return np.array(mean_fcst_probs), np.array(event_frequency), indices 
    else:
        return np.array(mean_fcst_probs), np.array(event_frequency) 

    
def reliability_uncertainty(y_true, y_pred, n_iter = 1000, n_bins=10 ):
    '''
    Calculates the uncertainty of the event frequency based on Brocker and Smith (WAF, 2007)
    '''
    mean_fcst_probs, event_frequency = reliability_curve(y_true, y_pred, n_bins=n_bins)

    event_freq_err = [ ]
    for i in range( n_iter ):
        Z     = uniform( size = len(y_pred) )
        X_hat = resample( y_pred )
        Y_hat = np.where( Z < X_hat, 1, 0 )
        _, event_freq = reliability_curve(X_hat, Y_hat, n_bins=n_bins)
        event_freq_err.append(event_freq)

    ef_low = np.nanpercentile(event_freq_err, 2.5, axis=0)
    ef_up  = np.nanpercentile(event_freq_err, 97.5, axis=0)

    return mean_fcst_probs, event_frequency, ef_low, ef_up


def _get_binary_xentropy(target_values, forecast_probabilities):
    """Computes binary cross-entropy.

    This function satisfies the requirements for `cost_function` in the input to
    `run_permutation_test`.

    E = number of examples

    :param: target_values: length-E numpy array of target values (integer class
        labels).
    :param: forecast_probabilities: length-E numpy array with predicted
        probabilities of positive class (target value = 1).
    :return: cross_entropy: Cross-entropy.
    """
    MIN_PROBABILITY = 1e-15
    MAX_PROBABILITY = 1. - MIN_PROBABILITY
    forecast_probabilities[
        forecast_probabilities < MIN_PROBABILITY] = MIN_PROBABILITY
    forecast_probabilities[
        forecast_probabilities > MAX_PROBABILITY] = MAX_PROBABILITY

    return -1 * np.nanmean(
        target_values * np.log2(forecast_probabilities) +
        (1 - target_values) * np.log2(1 - forecast_probabilities))

def compute_multiple_metrics(y, predictions, n_boot, metrics, metric_names, 
        metric_dimensions, alpha=0.95, forecast_time_indices=None, known_skew=None,):
    """
    Compute multi-metrics and bootstrap for confidence intervals
    """
    base_random_state = np.random.RandomState(22)
    random_num_set = base_random_state.choice(10000, size=n_boot, replace=False)
    even_or_odd = base_random_state.choice([2, 3], size=n_boot)

    scores = {n : [ ] for n in metric_names}
    for j in range(n_boot):
        new_random_state = np.random.RandomState(random_num_set[j])
        these_idxs = new_random_state.choice(len(y), size= len(y))
    
        if forecast_time_indices is not None:
            # val with either be a 2, 3, or 4 and then only resample from those samples. 
            val = even_or_odd[j]
            # Find those forecast time indices that are even or odd 
            where_is_fti = np.where(forecast_time_indices%val==0)[0]
            # Find the "where_is_item" has the same indices as "where_is_fti"
            idxs_subset = list(set(these_idxs).intersection(where_is_fti))
            # Resample idxs_subset for each iteration 
            these_idxs = new_random_state.choice(idxs_subset, size=len(idxs_subset),)

        y_temp=y[these_idxs]
        prediction_temp = predictions[these_idxs]
        
        for m, n in zip(metrics, metric_names):
            if n in ['AUPDC_NORM', 'MAX_CSI']:
                vals = m(y_temp, prediction_temp, known_skew=known_skew)
            else:
                vals = m(y_temp, prediction_temp)
            if type(vals) is tuple:
                vals = np.array(vals)
            scores[n].append(vals)
        
    p_high = ((1.0-alpha)/2.0) * 100
    p_low = (alpha+((1.0-alpha)/2.0)) * 100
    
    scores_ci = {}
    vars_with_dims = list(metric_dimensions.keys())
    
    for key in scores.keys():
        axis = 0 if np.ndim(scores[key]) > 1 else None

        if key in vars_with_dims:
            dim_name = metric_dimensions[key]['dim_names']
            for i in range(np.shape(scores[key])[1]):
                name = metric_dimensions[key]['names'][i]
                score = np.array(scores[key])[:,i]

                if dim_name is not None:
                    scores_ci[name+'_mean'] = ([dim_name], np.nanmean(score, axis=axis))
                    scores_ci[name+'_lowerbound'] = ([dim_name], np.nanpercentile(score, p_low, 
                                                                              axis=axis, interpolation='nearest'))
                    scores_ci[name+'_upperbound'] = ([dim_name], np.nanpercentile(score, p_high, axis=axis, 
                                                                 interpolation='nearest')) 
                else:
                    scores_ci[name+'_mean'] = np.nanmean(score, axis=axis)
                    scores_ci[name+'_lowerbound'] = np.nanpercentile(score, p_low, axis=axis, interpolation='nearest')
                    scores_ci[name+'_upperbound'] = np.nanpercentile(score, p_high, axis=axis, interpolation='nearest')

                    scores_ci[name] = (['n_iter'], score) 

        else:
            scores_ci[key+'_mean'] = np.nanmean(scores[key], axis=axis)
            scores_ci[key+'_lowerbound'] = np.nanpercentile(scores[key], p_low, axis=axis, interpolation='nearest')
            scores_ci[key+'_upperbound'] = np.nanpercentile(scores[key], p_high, axis=axis, interpolation='nearest')

            scores_ci[key] = (['n_iter'], scores[key])

    score_ds = xr.Dataset(scores_ci)
    
    return score_ds

