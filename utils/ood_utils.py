from typing import List, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy.core.multiarray import array as array
from sklearn.metrics import pairwise_distances
import torch

from utils import write_json, read_json
from constants import OOD_METHODS

# Requires python 3.10 or higher
#@dataclass(slots=True)
class OODMethod(ABC):
    
    name: str
    distance_method: bool
    per_class: bool
    per_stride: bool
    thresholds: List[float] or List[List[float]]
    # The threshold to use when matching the predicted boxes to the ground truth boxes.
    #   All boxes with an IoU lower than this threshold will be considered bad predictions
    iou_threshold_for_matching: float
    # Define the minimum threshold to output a box when predicting. All boxe with
    #   a confidence lower than this threshold will be discarded automatically
    min_conf_threshold: float
    which_internal_activations: str  # Where to extract internal activations from
    norm: int  # Norm to use when computing the distance in forward dynamics methods

    def __init__(self, name: str, distance_method: bool, per_class: bool, which_internal_activations: str):
        self.name = name
        self.distance_method = distance_method
        self.per_class = per_class
        self.thresholds = None
        self.which_internal_activations = which_internal_activations

    @abstractmethod
    def format_one_step_activations(self, activations: torch.Tensor, **kwargs) -> np.array or float:
        """
        Function to be overriden by each method to format the internal activations of the model.
        The extracted activations will be stored in the list all_activations
        """
        pass

    def append_one_step_activations_to_list(self, all_activations: List, one_step_activations: torch.Tensor, actions: np.array, **kwargs):
        """
        Append the activations of one step to the list of activations of all steps
        """
        if self.per_class:
            # Use actions to append the activations to the corresponding class
            all_activations[actions[0]].append(self.format_one_step_activations(one_step_activations, actions=actions))
        elif self.which_internal_activations == 'observations':
            # Use actions to predict the next state
            all_activations.append(self.format_one_step_activations(one_step_activations, actions=actions))
        else:
            # No need to use actions in the other cases
            all_activations.append(self.format_one_step_activations(one_step_activations))

    def compute_ood_score_on_one_step(self, activations: torch.Tensor, actions: np.array, **kwargs) -> float:
        """
        Compute the OOD score for one step
        """
        # Introduce the activations in a list to be compatible with the rest of the code
        activations = [self.format_one_step_activations(activations, actions)]
        score = self.compute_scores(activations, actions)
        return float(score[0])
    
    def compute_ood_decision_on_one_step(self, activations: torch.Tensor, actions: np.array, **kwargs) -> bool:
        """
        Return True if the state is OOD, False otherwise
        """
        score = self.compute_ood_score_on_one_step(activations, actions)
        return self.compute_ood_decision_on_one_score(score, actions)
    
    @abstractmethod
    def compute_ood_decision_on_one_score(self, score, actions: np.array, **kwargs) -> bool:
        """
        Function to be overriden by each method and contains the logic to compute the OOD decision.
        If its per_class and/or the method is a distance method, all the logic should be implemented here.
        """
        pass

    #@abstractmethod
    def compute_ood_decision_on_one_episode(activations: np.array, actions: np.array, **kwargs) -> List[bool]:
        """
        Function to be overriden by each method to compute the OOD decision for one episode
        Vectorized implementation:
            This function will not call the compute_ood_decision_on_one_step function, instead it will
            try to compute the decision for the whole episode at once
        """
        pass

    def save_ood_detection_results(self, data: dict, save_path: Path, model_name=''):
        """
        """
        import pandas as pd
        save_path = Path(save_path)  # In case we receive a string
        save_path.mkdir(parents=False, exist_ok=True)
        ## Transform the data to a pandas dataframe
        # First, transform into a list of lists
        data_list = []
        for k, v in data.items():
            k = k.split('_')
            data_list.append(k+list(v))
        df = pd.DataFrame(data=data_list, columns=['lvl_id', 'ind_or_ood_eps', 'num_oods_seen', 'num_oods_correctly_detected', 'num_oods_detected', 'steps'])
        
        if model_name:
            df.to_csv(save_path / f'ood_detection_results_{self.name}_{model_name}.csv')
        else:
            df.to_csv(save_path / f'ood_detection_results_{self.name}.csv')

    def save_ood_method_info(self, save_path: Path or str, model_name=''):
        """
        Save the information of the OODMethod in the path
        """
        save_path = Path(save_path)  # In case we receive a string
        save_path.mkdir(parents=False, exist_ok=True)
        data = {}
        data['thresholds'] = self.thresholds
        if model_name:
            write_json(data, save_path / f'thresholds_{self.name}_{model_name}.json')
        else:
            write_json(data, save_path / f'thresholds_{self.name}.json')
        # Distance methods need to add the clusters 
        if isinstance(self, DistanceMethod):
            if model_name:
                torch.save(self.clusters, save_path / f'clusters_{self.name}_{model_name}.pt')
                #write_npy(self.clusters, save_path / f'clusters_{self.name}_{model_name}.npy')
            else:
                torch.save(self.clusters, save_path / f'clusters_{self.name}.pt')
                #write_npy(self.clusters, save_path / f'clusters_{self.name}.npy')
        

    def load_ood_method_info(self, save_path: Path, model_name=''):
        """
        Load the information of the OODMethod in the path
        """
        if model_name:
            thresholds = read_json(save_path / f'thresholds_{self.name}_{model_name}.json')
        else:
            thresholds = read_json(save_path / f'thresholds_{self.name}.json')
        self.thresholds = thresholds['thresholds']
        # Distance methods need to add the clusters 
        if isinstance(self, DistanceMethod):
            if model_name:
                self.clusters = torch.load(save_path / f'clusters_{self.name}_{model_name}.pt')
                #self.clusters = read_npy(save_path / f'clusters_{self.name}_{model_name}.npy')
            else:
                self.clusters = torch.load(save_path / f'clusters_{self.name}.pt')
                #self.clusters = read_npy(save_path / f'clusters_{self.name}.npy')

    def save_activations(self, all_activations: List, save_path: Path, model_name=''):
        """
        Save the activations in the path
        """
        if self.distance_method:
            activations_name = 'feature_maps'
        else:
            activations_name = 'logits'
        if model_name:
            torch.save(all_activations, save_path / f'{activations_name}_{self.name}_{model_name}.pt')
        else:
            torch.save(all_activations, save_path / f'{activations_name}_{self.name}.pt')

    def load_activations(self, save_path: Path, model_name='') -> List:
        """
        Load the activations in the path
        """
        if self.distance_method:
            activations_name = 'feature_maps'
        else:
            activations_name = 'logits'
        if model_name:
            return torch.load(save_path / f'{activations_name}_{self.name}_{model_name}.pt')
        else:
            return torch.load(save_path / f'{activations_name}_{self.name}.pt')
            
    @abstractmethod
    def compute_scores(self, activations: List, *args, **kwargs) -> List[np.array] or np.array:
        """
        Function to be overriden by each method to compute the scores
        """
        pass
    
    def generate_thresholds(self, all_activations: List, tpr: float) -> List[float] or List[List[float]]:
        """
        Generate the thresholds for each class using the in-distribution scores.
        If per_class=True, in_scores must be a list of lists,
        where each list is the list of scores for each class.
        tpr must be in the range [0, 1]
        """
        all_scores = self.compute_scores(all_activations)

        if self.distance_method:
            # If the method measures distance, the higher the score, the more OOD. Therefore
            # we need to get the upper bound, the tpr*100%
            used_tpr = 100*tpr
        else:            
            # As the method is a similarity method, the higher the score, the more IND. Therefore
            # we need to get the lower bound, the (1-tpr)*100%
            used_tpr = (1 - tpr)*100

        sufficient_samples = 10
        good_number_of_samples = 50

        if self.per_class:
            thresholds = [0 for _ in range(len(all_scores))]
            for idx, cl_scores in enumerate(all_scores):
                if len(cl_scores) > sufficient_samples:
                    thresholds[idx] = float(np.percentile(cl_scores, used_tpr, method='lower'))
                    if len(cl_scores) < good_number_of_samples:
                        print(f"Class {idx}: {len(cl_scores)} samples. The threshold may not be accurate")
                else:
                    print(f"Class {idx} has less than {sufficient_samples} samples. No threshold is generated")
                        
        else:
            thresholds = np.quantile(all_scores, 0.95)
        
        return thresholds

#################################################################################
# Create classes for each method. Methods will inherit from OODMethod,
#   will override the abstract methods and also any other function that is needed.
#################################################################################

### Methods using the forward dynamics module of the model ###
class ForwardDynamicsMethods(OODMethod):
    def __init__(self, name, norm: int, **kwargs):
        distance_method = False
        which_internal_activations = 'observations'
        per_class = False
        im_module = None  # Loaded in the code
        super().__init__(name, distance_method, per_class, which_internal_activations)
        self.norm = norm

    def format_one_step_activations(self, activations: torch.Tensor, actions: np.array) -> float:
        """
        Return the logits of the predicted class
        """
        current_obs, next_obs = activations
        # get tensor shape [batch,3,7,7]
        input_obs = self.im_module.preprocess_observations(current_obs)
        input_next_obs = self.im_module.preprocess_observations(next_obs)
        # get s and s'
        state_emb = self.im_module.state_embedding(input_obs)
        next_state_emb = self.im_module.state_embedding(input_next_obs)
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions)
        pred_next_state_emb = self.im_module.forward_dynamics(state_emb, actions)
        forward_dynamics_loss = torch.norm(pred_next_state_emb - next_state_emb, dim=1, p=self.norm)
        return forward_dynamics_loss.item()
    
    def compute_ood_decision_on_one_score(self, score, actions: np.array, *args, **kwargs) -> bool:
        """
        True (OOD detected) if the score is lower than the threshold. If per_class=True, the threshold
        will be the one for the predicted class.
        """
        if self.per_class:
            return score > self.thresholds[actions[0]]
        else:
            return score > self.thresholds
        
    def compute_scores(self, activations: List, *args, **kwargs) -> np.array:
        return np.array(activations)  # No need to compute anything, as the scores are already computed


class ForwardDynamicsL1(ForwardDynamicsMethods):
    def __init__(self, **kwargs):
        name = 'forward_dynamics_l1'
        norm = 1
        super().__init__(name, norm, **kwargs)

class ForwardDynamicsL2(ForwardDynamicsMethods):
    def __init__(self, **kwargs):
        name = 'forward_dynamics_l2'
        norm = 2
        super().__init__(name, norm, **kwargs)

### Logits based methods for methods using logits of the model ###
class LogitsMethod(OODMethod):
    
    def __init__(self, name: str, per_class: bool):
        distance_method = False
        which_internal_activations = 'logits'
        super().__init__(name, distance_method, per_class, which_internal_activations)

    def format_one_step_activations(self, activations: torch.Tensor, actions: np.array) -> float:
        """
        Return the logits of the predicted class
        """
        #return activations[0, actions[0]].item()  # Return the logits of the predicted class
        return activations  # Return the logits of all
    
    def compute_ood_decision_on_one_score(self, score, actions: np.array, *args, **kwargs) -> bool:
        """
        True (OOD detected) if the score is lower than the threshold. If per_class=True, the threshold
        will be the one for the predicted class.
        """
        if self.per_class:
            return score < self.thresholds[actions[0]]
        else:
            return score < self.thresholds

class MSP(LogitsMethod):

    def __init__(self, **kwargs):
        name = 'msp'
        per_class = True
        super().__init__(name, per_class=per_class, **kwargs)
    
    def compute_scores(self, activations: List, *args, **kwargs) -> List[np.array]:
        return [torch.softmax(activations[0], dim=1).max()]  # No need to compute anything, as the scores are already computed

    
class Energy(LogitsMethod):

    temper: float

    def __init__(self, temper: float, **kwargs):
        name = 'energy'
        per_class = True
        super().__init__(name, per_class=per_class, **kwargs)
        self.temper = temper
    
    def compute_scores(self, activations: List, *args, **kwargs) -> List[np.array]:
        #return self.temper * torch.logsumexp(torch.tensor(activations) / self.temper, dim=0).numpy()
        #return np.array([self.temper * torch.logsumexp(torch.tensor(activations) / self.temper, dim=0).numpy()])
        #return [self.temper * torch.logsumexp(torch.tensor(activations) / self.temper, dim=0).numpy()]
        return [self.temper * torch.logsumexp(activations[0] / self.temper, dim=1)]
    

### Methods using feature maps of the model ###
class DistanceMethod(OODMethod):
    
    agg_method: Callable
    cluster_method: str
    cluster_optimization_metric: str
    available_cluster_methods: List[str]
    available_cluster_optimization_metrics: List[str]
    clusters: np.array

    # name: str, distance_method: bool, per_class: bool, per_stride: bool, iou_threshold_for_matching: float, min_conf_threshold: float
    def __init__(self, name: str, agg_method: str, per_class: bool, cluster_method: str, cluster_optimization_metric: str, **kwargs):
        distance_method = True  # Always True for distance methods
        which_internal_activations = 'ftmaps'  # This could be changed in subclasses
        super().__init__(name, distance_method, per_class, which_internal_activations=which_internal_activations, **kwargs)
        self.available_cluster_methods = ['one','all','DBSCAN', 'KMeans', 'GMM', 'HDBSCAN', 'OPTICS', 'SpectralClustering', 'AgglomerativeClustering']
        self.available_cluster_optimization_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        self.cluster_method = self.check_cluster_method_selected(cluster_method)
        self.cluster_optimization_metric = self.check_cluster_optimization_metric_selected(cluster_optimization_metric)
        if agg_method == 'mean':
            self.agg_method = np.mean
        elif agg_method == 'median':
            self.agg_method = np.median
        else:
            raise NameError(f"The agg_method argument must be one of the following: 'mean', 'median'. Current value: {agg_method}")
        
    def check_cluster_method_selected(self, cluster_method: str) -> str:
        assert cluster_method in self.available_cluster_methods, f"cluster_method must be one of {self.available_cluster_methods}, but got {cluster_method}"
        return cluster_method

    def check_cluster_optimization_metric_selected(self, cluster_optimization_metric: str) -> str:
        assert cluster_optimization_metric in self.available_cluster_optimization_metrics, f"cluster_method must be one" \
          f"of {self.available_cluster_optimization_metrics}, but got {cluster_optimization_metric}"
        return cluster_optimization_metric
    
    def format_one_step_activations(self, activations: torch.Tensor, *args, **kwargs) -> np.array:
        """
        Return the feature maps with the appropriate shape
        """
        return activations[0].cpu().numpy()
    
    @abstractmethod
    def compute_distance(self, activations: np.array, clusters: np.array) -> np.array:
        """
        Function that will take two tensors and compute the distance between them, pairwise.
        To be overriden by each method.
        """
        pass

    def compute_scores(self, activations: List[np.array], actions: np.array = None, *args, **kwargs) -> List[np.array] or np.array:
        """
        This function has the logic of looping over the clusters to then call the function that computes the scores.
        """
        if self.per_class:
            # Case where we receive the activations sorted by class
            if len(activations) > 1:
                scores = []
                for idx, ftmaps_one_cls in enumerate(activations):
                    # If there are no feature maps for this class, we skip it
                    if len(ftmaps_one_cls) > 0:
                        scores.append(self.compute_distance(np.stack(ftmaps_one_cls), self.clusters[idx]).min(axis=1))
                    else:
                        scores.append(np.array([]))
                return scores  # List of arrays, one per class
                
            else:
                if actions is None:
                    raise ValueError("The actions argument must be provided when per_class=True and only one activation is provided")
                ## Check if there is a cluster for the class
                if self.clusters[actions[0]].shape[0] == 0:
                    # Case where we receive the activations of one class that has not been seen during training and therefore
                    #   we do not have any cluster for it. We return a score higher than the threshold to indicate that it is OOD
                    return np.array([max(self.thresholds) + 10])
                else:
                    # Normal case, where a cluster exists for the class
                    scores = self.compute_distance(np.stack(activations), self.clusters[actions[0]])
                    return scores.min(axis=1)
            
        else:
            activations = np.stack(activations)
            scores = self.compute_distance(activations, self.clusters)
        
            return scores.min(axis=1)  # Min distance to any cluster

    def compute_ood_decision_on_one_score(self, score, actions: np.array, **kwargs) -> bool:
        """
        True (OOD detected) if the score (distance) is higher than the threshold
        """
        if self.per_class:
            return score > self.thresholds[actions[0]]
        else:
            return score > self.thresholds

    def generate_clusters(self, ind_tensors: List[np.array]):
        """
        Generate the clusters for each class using the in-distribution tensors (usually feature maps).
        self.clusters must be a numpy array of shape (n_clusters, n_features)
        """
        if self.per_class:
            
            clusters_per_class = [[] for _ in range(len(ind_tensors))]
            if self.cluster_method == 'one':
                for idx, ftmaps_one_cls in enumerate(ind_tensors):
                    if len(ftmaps_one_cls) > 0:
                        clusters_per_class[idx] = self.agg_method(ftmaps_one_cls, axis=0, keepdims=True)
                    else:
                        clusters_per_class[idx] = np.array([])
                
            elif self.cluster_method in self.available_cluster_methods:
                raise NotImplementedError("Not implemented yet")
            elif self.cluster_method == 'all':
                raise NotImplementedError("As the amount of In-Distribution data is too big," \
                                        "ir would be intractable to treat each sample as a cluster")
            else:
                raise NameError(f"The clustering_opt must be one of the following: 'one', 'all', or one of {self.available_cluster_methods}." \
                                f"Current value: {self.cluster_method}")            
            self.clusters = clusters_per_class
        
        else:

            ind_tensors = np.stack(ind_tensors, axis=0)
            if self.cluster_method == 'one':
                # newaxis to add the first dimension, as we need to have a shape of (n_clusters, n_features) and the mean
                #   operation will return a shape of (n_features) only
                clusters = self.agg_method(ind_tensors, axis=0, keepdims=True)

            elif self.cluster_method in self.available_cluster_methods:
                raise NotImplementedError("Not implemented yet")
            
            elif self.cluster_method == 'all':
                raise NotImplementedError("As the amount of In-Distribution data is too big," \
                                        "ir would be intractable to treat each sample as a cluster")           
            else:
                raise NameError(f"The clustering_opt must be one of the following: 'one', 'all', or one of {self.available_cluster_methods}." \
                                f"Current value: {self.cluster_method}")

            self.clusters = clusters


class L1(DistanceMethod):
    
    # name: str, agg_method: str, per_class: bool, per_stride: bool, cluster_method: str, cluster_optimization_metric: str
    def __init__(self, agg_method, **kwargs):
        name = 'l1'
        per_class = False
        cluster_method = 'one'
        cluster_optimization_metric = 'silhouette'
        super().__init__(name, agg_method, per_class, cluster_method, cluster_optimization_metric, **kwargs)
    
    def compute_distance(self, activations: np.array, clusters: np.array) -> np.array:
        distances = pairwise_distances(
            X=activations,
            Y=clusters,
            metric='l1'
            )
        return distances
        
        
class L1PerAction(L1):
        
    def __init__(self, agg_method, **kwargs):
        super().__init__(agg_method, **kwargs)
        self.per_class = True
        self.name = 'l1_per_action'


class L2(DistanceMethod):
    
    def __init__(self, agg_method, **kwargs):
        name = 'l2'
        per_class = False
        cluster_method = 'one'
        cluster_optimization_metric = 'silhouette'
        super().__init__(name, agg_method, per_class, cluster_method, cluster_optimization_metric, **kwargs)
    
    def compute_distance(self, activations: np.array, clusters: np.array) -> np.array:
        distances = pairwise_distances(
            clusters,
            activations,
            metric='l2'
            )
        return distances
    
class L2PerAction(L2):
            
    def __init__(self, agg_method, **kwargs):
        super().__init__(agg_method, **kwargs)
        self.per_class = True
        self.name = 'l2_per_action'


### Global methods ###
def save_ood_detection_results(data: dict, save_path: Path, name: str, model_name=''):
    """
    Save the results of the OOD detection in a csv file
    """
    save_path = Path(save_path)  # In case we receive a string
    save_path.mkdir(parents=False, exist_ok=True)
    
    if model_name:
        torch.save(data, save_path / f'ood_detection_results_{name}_{model_name}.pt')
    else:
        torch.save(data, save_path / f'ood_detection_results_{name}.pt')

def save_performance_metrics(env_names, seed, data, save_path: str, model_name: str):
    from csv import writer
    save_path = Path(save_path)
    model_name = Path(model_name).name
    # Remove seed from the name
    model_name = model_name[:-2]

    columns = ['env', 'seed', 'num_episodes_played','num_episodes_solved','success_rate','sum_of_returns','mean_return', 'mean_return_in_solved_eps']

    # Create list of lists with the data
    data_list = []
    for i in range(len(env_names)):
        data_list.append([env_names[i], seed] + data[i])

    # csv path
    csv_path = save_path / f'performance_evaluation_{model_name}.csv'
    if csv_path.exists():
        print('CSV file already exists, appending to it')
        # Open our existing CSV file in append mode
        # Create a file object for this file
        with open(csv_path, 'a') as f_object:
            # Pass this file object to csv.writer() and get a writer object
            writer_object = writer(f_object)
            # Pass the list as an argument into the writerow()
            writer_object.writerows(data_list)
            # Close the file object
            f_object.close()
    else:
        print('CSV file does not exist, creating it')
        # Create a file object for this file
        with open(csv_path, 'w') as f_object:
            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = writer(f_object)
            # Create the headers row and pass the list as an argument into the writerow()
            writer_object.writerow(columns)
            writer_object.writerows(data_list)
            # Close the file object
            f_object.close()    


def select_ood_method(name: str) -> DistanceMethod or LogitsMethod:
    
    assert name in OOD_METHODS, f"The name of the method must be one of {OOD_METHODS}, but got {name}"

    if name == "msp":
        ood_method = MSP()
    
    elif name == "energy":
        ood_method = Energy(temper=1)

    elif name == "l1":
        ood_method = L1(agg_method='mean')

    elif name == "l2":
        ood_method = L2(agg_method='mean')

    elif name == "l1_per_action":
        ood_method = L1PerAction(agg_method='mean')
    
    elif name == "l2_per_action":
        ood_method = L2PerAction(agg_method='mean')

    elif name == "forward_dynamics_l1":
        ood_method = ForwardDynamicsL1()
    
    elif name == "forward_dynamics_l2":
        ood_method = ForwardDynamicsL2()

    else:
        raise NameError(f"The name of the method must be one of {OOD_METHODS}, but got {name}")
    
    return ood_method

