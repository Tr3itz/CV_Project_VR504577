import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from ptflops import get_model_complexity_info # get FLOPs info


class SuperModule(nn.Module):
    def __init__(self, alternatives: nn.ModuleList, path: str=None):
        """
        SuperNet Module managing multiple alternative blocks.
        ---------
        Args:
        - alternatives: nn.ModuleList - list of alternative layers/blocks
        - path: str                   - path to the SuperModule
        ---------
        Attributes:
        - self.alternatives - list of alternative layers/blocks
        - self.weights      - trainable parameters (one for each branch)
        """

        super().__init__()

        # Location of the SuperModule in the seed network
        self.path = path

        # Alternatives and parameters
        self.alternatives = alternatives
        self.weights = nn.parameter.Parameter(data=torch.ones(size=(len(self.alternatives),)) * 1/len(self.alternatives), requires_grad=True)

        # Number of parameters for each alternative 
        self.params_count = torch.tensor(
            data=[sum(p.numel() for p in alt.parameters() if p.requires_grad) for alt in self.alternatives], 
            dtype=torch.float, 
            requires_grad=True # Requires gradient for backprogagation of the regularizer
        )
        self.params_max = torch.max(self.params_count).detach()

        # Number of FLOPs for each alternative - updated in self._check_sanity()
        self.flops_count, self.flops_max = None, None
        
        self.checked = False


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.checked: return self._check_sanity(x)

        # Compute the Gumbel-Softmax on SuperModule's parameters
        gs_weights = F.gumbel_softmax(self.weights, tau=1.0, hard=False)
        
        # Compute the weighted sum of alternatives' outputs
        out = [gs_weights[idx]*alt(x) for idx, alt in enumerate(self.alternatives)]
        out = torch.stack(out)
        out = torch.sum(out, dim=0)
        
        # Needed if batch has 1 element
        if out.dim() != 4:
            out = out.unsqueeze(dim=0)

        return out
        

    def _stable_weighted_cost(self, device: torch.device, type_cost: str='params'):
        if self.params_count.device.type != device.type:
            self.params_count, self.params_max = self.params_count.to(device), self.params_max.to(device)
            self.flops_count, self.flops_max = self.flops_count.to(device), self.flops_max.to(device) 

        # Weighted sum of normalized number of parameters
        gs_weights = F.gumbel_softmax(self.weights, tau=1.0, hard=False)
        
        if type_cost == 'params':
            cost_norm = self.params_count / self.params_max
        elif type_cost == 'flops':
            cost_norm = self.flops_count / self.flops_max
        else:
            raise ValueError(f"Cost type {type_cost} not implemented. Select either 'params' or 'flops'.")

        return torch.sum(gs_weights * cost_norm)
    

    def _target_weighted_cost(self, device: torch.device, type_cost: str='params'):
        if self.params_count.device.type != device.type:
            self.params_count, self.params_max = self.params_count.to(device), self.params_max.to(device)
            self.flops_count, self.flops_max = self.flops_count.to(device), self.flops_max.to(device)  

        # Weighted sum of number of parameters
        gs_weights = F.gumbel_softmax(self.weights, tau=1.0, hard=False)
        
        if type_cost == 'params':
            return torch.sum(gs_weights * self.params_count)
        elif type_cost == 'flops':
            return torch.sum(gs_weights * self.flops_count)
        else:
            raise ValueError(f"Cost type {type_cost} not implemented. Select either 'params' or 'flops'.")


    def _check_sanity(self, x: torch.Tensor) -> torch.Tensor:
        output = [alt(x) for alt in self.alternatives]

        try:
            out_tensor = torch.cat(output, dim=0)
        except RuntimeError: # tensor can be concatenated only if they share the size
            out_shape = output[0].shape
            for idx, out in enumerate(output):
                if out.shape != out_shape:
                    err_msg += f'Alternatives in SuperModule have different output shapes. Expected shape {out_shape} but got {out.shape} from alternative number {idx}.'
                    err_msg = f' (@ {self.path})' if self.path is not None else ''
                    raise RuntimeError(err_msg)

        self.checked = True

        # Update number of FLOPs per alternative
        self.flops_count = torch.tensor(
            data=[get_model_complexity_info(model=alt, input_res=tuple(x.squeeze().size()), print_per_layer_stat=False, as_strings=False, backend='pytorch')[0] for alt in self.alternatives],                                        
            dtype=torch.float, 
            requires_grad=True # Requires gradient for backpropagation of the regularizer
        )        
        self.flops_max = torch.max(self.flops_count).detach()

        return torch.sum(out_tensor, dim=0).unsqueeze(dim=0)



class SuperNet(nn.Module):
    def __init__(self, seed: nn.Module, branches: nn.ModuleDict, input_shape: torch.Size):
        """
        SuperNet implementation for local Differeintiable Neural Achitecture Search!
        ---------
        Args:
        - seed: nn.Module             - original network
        - branches: nn.ModuleDict     - dictionary of alternatives
            -> key: str               - path-like to the chosen block
            -> value: nn.ModuleList   - list of alternatives
        - input_shape: torch.Size     - shape of the input data (used for sanity check)
        ---------
        Attributes:
        - self.seed    - network with alternatives
        - self.pahts   - paths to SuperModules
        """

        super().__init__()

        # Baseline network
        self.seed = seed
        # Paths to SuperModules
        self.paths = []

        # Set alternatives to the chosen modules 
        for at, alternatives in branches.items():
            at = at.split('/')
            at = '.'.join(at) if len(at) > 1 else at[0]
            self.paths.append(at)
            self.seed.set_submodule(at, SuperModule(alternatives, path=at))

        # Check sanity of the chosen alternatives
        self._check_sanity(input_shape)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seed(x)
    
    
    def export(self, verbose: bool=False) -> nn.Module:
        """Extract the best option from the alternatives"""

        for path in self.paths:
            # SuperModule
            super_module = self.seed.get_submodule(path)

            # Best alternative
            block_idx = torch.argmax(super_module.weights)
            block = super_module.alternatives[block_idx]
            
            if verbose:
                print(f'[BEST ALTERNATIVE for {path}]:\n{block}\n')

            # Remove FLOPs counting attrs for model reusability
            delattr(block, 'start_flops_count')
            delattr(block, 'stop_flops_count')
            delattr(block, 'reset_flops_count')
            delattr(block, 'compute_average_flops_cost')
            
            # Set the best alternative
            self.seed.set_submodule(path, block)
        
        return self.seed
    
    
    def get_weighted_cost(self, stable: bool, target: int=None, type_cost: str='params') -> tuple[torch.Tensor]:
        """Compute the weighted cost of the alternatives"""

        device = next(self.parameters()).device

        if stable:
            stable_cost = [self.seed.get_submodule(path)._stable_weighted_cost(device, type_cost) for path in self.paths]
            stable_cost = torch.sum(torch.stack(stable_cost), dim=0)
        else:
            stable_cost = torch.zeros(1, dtype=torch.float, requires_grad=True).squeeze()
        
        if target is not None:
            target_cost = [self.seed.get_submodule(path)._target_weighted_cost(device, type_cost) for path in self.paths]
            target_cost = torch.sum(torch.stack(target_cost), dim=0)
        else:
            target_cost = torch.zeros(1, dtype=torch.float, requires_grad=True).squeeze()

        # Soft-Constraint: compare with a target quantity
        if target is not None:
            target_cost = torch.max(torch.zeros(1, dtype=torch.float, requires_grad=True).squeeze(), target_cost - target)
            
        return stable_cost, target_cost         
    
    
    def get_supermodules_weights(self) -> dict[str, list[float]]:
        weights = {}

        for path in self.paths:
            super_module = self.seed.get_submodule(path)
            weights[path] = super_module.weights.tolist()

        return weights

    
    def _check_sanity(self, input_shape: torch.Size):
        """
        Feed a tensor with the same shape of input data in order to check if
        the alternatives in the SuperModules share the same output shape
        """

        sanity_tensor = torch.unsqueeze(torch.rand(input_shape), dim=0)
        with torch.no_grad():
            self.seed(sanity_tensor)



class SuperRegularizer(nn.Module):
    def __init__(self, type_cost: str, stable_alpha: float=1.0, target: int=None, target_gamma: float=0.0):
        """
        Regularizer class for SuperNet!
        ---------
        Args:
        - type_cost: str        - type of the regularizer ('params' or 'flops')
        - stable_alpha: float   - weight for the stable regularizer
        - target: int           - qunatity to compare the selected cost with
        - target_gamma: float   - weight for the target regularizer
        """

        super().__init__()       
        
        if stable_alpha == 0.0 and target is None:
            raise RuntimeError('No regularization selected. At least stable=True or define target quantity.') 
        
        if target_gamma > 0.0 and target is None:
            raise RuntimeError('target_gamma > 0, but no target quantity defined.')
        
        if target is not None and target_gamma == 0.0:
            raise RuntimeError('Target quantity defined but target_gamma is 0.')
        
        # Cost type
        self.type_cost = type_cost
        
        # Stable loss
        self.stable_alpha = stable_alpha
        self.stable = stable_alpha > 0.0
        
        # Soft constraint
        self.target = target
        self.target_gamma = target_gamma


    def __call__(self, model: SuperNet):        
        reg_loss = model.get_weighted_cost(self.stable, self.target, self.type_cost)
        return self.stable_alpha*reg_loss[0] + self.target_gamma*reg_loss[1]


    def forward(self, model: SuperNet):       
        reg_loss = model.get_weighted_cost(self.stable, self.target, self.type_cost)    
        return self.stable_alpha*reg_loss[0] + self.target_gamma*reg_loss[1]
        
    
    def split_costs(self, model: SuperNet):
        reg_loss = model.get_weighted_cost(self.stable, self.target, self.type_cost)
        return self.stable_alpha*reg_loss[0], self.target_gamma*reg_loss[1]
