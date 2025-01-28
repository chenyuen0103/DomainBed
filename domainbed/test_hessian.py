import torch
import torch.nn.functional as F

class HessianPenModel:
    def __init__(self):
        pass

    def hessian_pen_mem(self, x, logits, envs):
        unique_envs = envs.unique()
        num_envs = len(unique_envs)
        
        x_outer_envs = [torch.einsum('bi,bj->bij', x[envs == e], x[envs == e]) for e in unique_envs]
        diff_envs = []
        for e in unique_envs:
            mask = envs == e
            logits_env = logits[mask]
            p = F.softmax(logits_env, dim=1)
            diff_envs.append(torch.diag_embed(p) - torch.einsum('bi,bj->bij', p, p))
        
        # Compute H_H_f with vectorized operations
        H_H_f = torch.zeros(num_envs, num_envs, device=x.device, dtype=torch.float16)
        for e1 in range(num_envs):
            for e2 in range(e1, num_envs):
                prob_trace = torch.einsum('bik,cjk->bcij', diff_envs[e1], diff_envs[e2]).diagonal(dim1=-2, dim2=-1).sum(-1)  # [B_e1, B_e2]
                x_trace = torch.einsum('bik,cjk->bcij', x_outer_envs[e1], x_outer_envs[e2]).diagonal(dim1=-2, dim2=-1).sum(-1)  # [B_e1, B_e2]

                H_H_f[e1, e2] = (prob_trace * x_trace).sum() / (x_outer_envs[e1].shape[0] * x_outer_envs[e2].shape[0])
                H_H_f[e2, e1] = H_H_f[e1, e2]  # Symmetry
        
        f_norm_env = H_H_f.diagonal()
        shared_term = H_H_f.sum() / (num_envs ** 2)
        individual_term = 2 * H_H_f.sum(dim=1) / num_envs
        sum_h_minus_h_bar_sq = torch.sum(f_norm_env + shared_term - individual_term) / num_envs

        sum_h_minus_h_bar_sq /= (logits.shape[1] ** 2)
        return f_norm_env, sum_h_minus_h_bar_sq, H_H_f

    def hessian_pen_mem_optimized(self, x, logits, envs):
        unique_envs, inverse_indices = envs.unique(return_inverse=True)
        num_envs = unique_envs.size(0)
        
        # Count number of samples per environment
        env_counts = torch.bincount(inverse_indices)
        
        # Split x and logits based on environment counts
        x_split = torch.split(x, env_counts.tolist())
        logits_split = torch.split(logits, env_counts.tolist())
        
        # Compute softmax probabilities for each environment
        p_split = [F.softmax(logit, dim=1) for logit in logits_split]  # list of [B_e, C]
        
        # Compute diff_envs: list of [B_e, C, C]
        diff_envs = [torch.diag_embed(p) - torch.einsum('bi,bj->bij', p, p) for p in p_split]
        
        # Compute x_outer_envs: list of [B_e, D, D]
        x_outer_envs = [torch.einsum('bi,bj->bij', xi, xi) for xi in x_split]
        
        # Initialize H_H_f
        H_H_f = torch.zeros((num_envs, num_envs), device=x.device, dtype=torch.float16)
        
        # Compute H_H_f using nested loops
        for e1 in range(num_envs):
            for e2 in range(e1, num_envs):
                # Compute prob_trace: [B_e1, B_e2]
                prob_trace = torch.einsum('bik,cjk->bcij', diff_envs[e1], diff_envs[e2]).diagonal(dim1=-2, dim2=-1).sum(-1)  # [B_e1, B_e2]
                
                # Compute x_trace: [B_e1, B_e2]
                x_trace = torch.einsum('bik,cjk->bcij', x_outer_envs[e1], x_outer_envs[e2]).diagonal(dim1=-2, dim2=-1).sum(-1)  # [B_e1, B_e2]
                
                # Compute H_H_f[e1, e2]
                H_H_f[e1, e2] = (prob_trace * x_trace).sum() / (env_counts[e1] * env_counts[e2])
                H_H_f[e2, e1] = H_H_f[e1, e2]  # Symmetry
        
        # Compute the final terms
        f_norm_env = H_H_f.diagonal()
        shared_term = H_H_f.sum() / (num_envs ** 2)
        individual_term = 2 * H_H_f.sum(dim=1) / num_envs
        sum_h_minus_h_bar_sq = torch.sum(f_norm_env + shared_term - individual_term) / num_envs
        
        sum_h_minus_h_bar_sq /= (logits.size(1) ** 2)
        
        return f_norm_env, sum_h_minus_h_bar_sq, H_H_f

def test_hessian_pen_mem_functions():
    torch.manual_seed(42)  # For reproducibility

    # Define input dimensions
    batch_size = 32
    num_classes = 10
    feature_dim = 64
    num_envs = 5

    # Generate random input data
    # Ensure that environments have varying batch sizes
    # For example, envs: [0]*5 + [1]*9 + [2]*6 + [3]*7 + [4]*5 = total 32
    envs = torch.tensor([0]*5 + [1]*9 + [2]*6 + [3]*7 + [4]*5)
    if envs.size(0) != batch_size:
        raise ValueError(f"Total number of samples in envs ({envs.size(0)}) does not match batch_size ({batch_size})")

    x = torch.randn(batch_size, feature_dim, requires_grad=True)
    logits = torch.randn(batch_size, num_classes, requires_grad=True)

    # Initialize the model
    model = HessianPenModel()

    # Compute outputs using the original function
    f_norm_original, sum_h_original, H_H_f_original = model.hessian_pen_mem(x, logits, envs)

    # Compute outputs using the optimized function
    f_norm_optimized, sum_h_optimized, H_H_f_optimized = model.hessian_pen_mem_optimized(x, logits, envs)
    breakpoint()
    # Define tolerances for comparison
    atol = 1e-3
    rtol = 1e-3

    # Compare f_norm_env
    if torch.allclose(f_norm_original, f_norm_optimized, atol=atol, rtol=rtol):
        print("f_norm_env matches between original and optimized functions.")
    else:
        print("f_norm_env does NOT match between original and optimized functions.")
        print("Original f_norm_env:", f_norm_original)
        print("Optimized f_norm_env:", f_norm_optimized)

    # Compare sum_h_minus_h_bar_sq
    if torch.allclose(sum_h_original, sum_h_optimized, atol=atol, rtol=rtol):
        print("sum_h_minus_h_bar_sq matches between original and optimized functions.")
    else:
        print("sum_h_minus_h_bar_sq does NOT match between original and optimized functions.")
        print("Original sum_h_minus_h_bar_sq:", sum_h_original)
        print("Optimized sum_h_minus_h_bar_sq:", sum_h_optimized)

    # Compare H_H_f
    if torch.allclose(H_H_f_original, H_H_f_optimized, atol=atol, rtol=rtol):
        print("H_H_f matches between original and optimized functions.")
    else:
        print("H_H_f does NOT match between original and optimized functions.")
        print("Original H_H_f:", H_H_f_original)
        print("Optimized H_H_f:", H_H_f_optimized)

    # Perform backward pass to check gradients
    # Zero gradients first
    if x.grad is not None:
        x.grad.zero_()
    if logits.grad is not None:
        logits.grad.zero_()

    # Compute loss and backward for original function
    loss_original = sum_h_original
    loss_original.backward(retain_graph=True)
    grad_x_original = x.grad.clone()
    grad_logits_original = logits.grad.clone()

    # Zero gradients again
    x.grad.zero_()
    logits.grad.zero_()

    # Compute loss and backward for optimized function
    loss_optimized = sum_h_optimized
    loss_optimized.backward()
    grad_x_optimized = x.grad.clone()
    grad_logits_optimized = logits.grad.clone()

    # Compare gradients w.r.t x
    if torch.allclose(grad_x_original, grad_x_optimized, atol=atol, rtol=rtol):
        print("Gradients w.r.t x match between original and optimized functions.")
    else:
        print("Gradients w.r.t x do NOT match between original and optimized functions.")
        print("Original grad_x:", grad_x_original)
        print("Optimized grad_x:", grad_x_optimized)

    # Compare gradients w.r.t logits
    if torch.allclose(grad_logits_original, grad_logits_optimized, atol=atol, rtol=rtol):
        print("Gradients w.r.t logits match between original and optimized functions.")
    else:
        print("Gradients w.r.t logits do NOT match between original and optimized functions.")
        print("Original grad_logits:", grad_logits_original)
        print("Optimized grad_logits:", grad_logits_optimized)

if __name__ == "__main__":
    test_hessian_pen_mem_functions()
