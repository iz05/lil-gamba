import torch
from torch.nn import functional as F
from sample_utils import geometric_dist


def complex_log(input, eps=1e-12):
    eps = input.new_tensor(eps)
    real = input.abs().maximum(eps).log()
    imag = (input < 0).to(input.dtype) * torch.pi
    return torch.complex(real, imag)


def selective_scan(u, dt, A, B, C, D, mode='logcumsumexp'):
    dA = torch.einsum('bld,dn->bldn', dt, A)
    dB_u = torch.einsum('bld,bld,bln->bldn', dt, u, B)
    dA = dA.clamp(min=-20)
    
    padding =  (0, 0, 0, 0, 1, 0)
    
    match mode:
        case 'cumsum':            
            dA_cumsum = F.pad(dA[:, 1:], padding).cumsum(1).exp()
            x = dB_u / (dA_cumsum + 1e-12)
            x = x.cumsum(1) * dA_cumsum
            y = torch.einsum('bldn,bln->bld', x, C)
        
        case 'logcumsumexp':  # more numerically stable (Heisen sequence)
            dB_u_log = complex_log(dB_u)
            dA_star = F.pad(dA[:, 1:].cumsum(1), padding)
            x_log = torch.logcumsumexp(dB_u_log - dA_star, 1) + dA_star
            y = torch.einsum('bldn,bln->bld', x_log.real.exp() * torch.cos(x_log.imag), C)
            
    return y + u * D

def selective_scan_naive(u, dt, A, B, C, D, eps=1e-8):
    """
    Naive selective scan for shapes:
        u:  (B, L, D)
        dt: (B, L, D)
        A:  (D, N)
        B:  (B, L, N)
        C:  (B, L, N)
        D:  (D,)
    
    Returns:
        y: (B, L, D)
    """
    B_size, L, D_in = u.shape

    N = A.shape[1]

    h = torch.zeros(B_size, D_in, N, device=u.device, dtype=u.dtype)
    outputs = []

    # Precompute identity for A_bar clipping
    I = torch.zeros_like(A)  # Not needed since elementwise

    for t in range(L):
        u_t = u[:, t, :]   # (B,D)
        dt_t = dt[:, t, :] # (B,D)

        # ZOH discretization for A
        deltaA = dt_t.unsqueeze(-1) * A.unsqueeze(0)  # (B,D,N)
        A_bar = deltaA.exp()                          # (B,D,N)

        # ZOH discretization for B elementwise
        B_bar = torch.where(
            deltaA.abs() > eps,
            ((deltaA.exp() - 1) / deltaA) * B[:, t, :].unsqueeze(1),  # (B,1,N) broadcast over D
            B[:, t, :].unsqueeze(1)
        )  # (B,D,N)

        # Input modulation
        dB_u = u_t.unsqueeze(-1) * B_bar  # (B,D,N)

        # Update hidden state
        h = h * A_bar + dB_u  # (B,D,N)

        # Compute output
        y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1) + (u_t * D)  # (B,D)

        outputs.append(y_t)

    return torch.stack(outputs, dim=1)  # (B,L,D)

def selective_scan_multi(u, dt, A_list, B, C, D, decay=1.0, eps=1e-8):
    """
    Selective scan for shapes:
        u:  (B, L, D)
        dt: (B, L, D)
        A:  List[Tensor(D, N)]
        B:  (B, L, N)
        C:  (B, L, N)
        D:  (D,)
    
    Returns:
        y: (B, L, D)
    """
    B_size, L, D_in = u.shape

    N = A_list[0].shape[1]

    outputs = []

    # hidden state queue, starts with one vector of all 0s
    queue = [torch.zeros(B_size, D_in, N, device=u.device, dtype=u.dtype), ]

    # Precompute identity for A_bar clipping
    I = torch.zeros_like(A_list[0])  # Not needed since elementwise

    for t in range(L):
        u_t = u[:, t, :]   # (B,D)
        dt_t = dt[:, t, :] # (B,D)
        
        # compute new hidden state
        hidden = torch.zeros(B_size, D_in, N, device=u.device, dtype=u.dtype)

        for idx in range(min(len(A_list), len(queue))):
            h = queue[idx]
            A = A_list[idx]

            A = A.to(u.device)

            # ZOH discretization for A
            deltaA = dt_t.unsqueeze(-1) * A.unsqueeze(0)  # (B,D,N)
            A_bar = deltaA.exp()                          # (B,D,N)

            hidden += h * A_bar

            # update input based only for the first (most recent) hidden state
            if idx == 0:
                # ZOH discretization for B elementwise
                B_bar = torch.where(
                    deltaA.abs() > eps,
                    ((deltaA.exp() - 1) / deltaA) * B[:, t, :].unsqueeze(1),  # (B,1,N) broadcast over D
                    B[:, t, :].unsqueeze(1)
                )  # (B,D,N)

                # Input modulation
                dB_u = u_t.unsqueeze(-1) * B_bar  # (B,D,N)

                hidden += dB_u
            
        # update queue
        if len(queue) < len(A_list):
            queue.insert(0, hidden)
        
        else:
            prob_dist = geometric_dist(
                v=1.0 - 1.0 * decay * t / (t + 1),
                N=len(A_list)
            )
            idx_to_replace = torch.multinomial(
                torch.tensor(prob_dist, device=hidden.device),
                num_samples=1
            ).item()
            queue.pop(idx_to_replace)
            queue.insert(0, hidden)

        # Compute output
        y_t = (hidden * C[:, t, :].unsqueeze(1)).sum(dim=-1) + (u_t * D)  # (B,D)

        outputs.append(y_t)

    return torch.stack(outputs, dim=1)  # (B,L,D)


# the mismatch between the cumsum and logcumsumexp modes will grow quickly as sequence length scales up
if __name__ == "__main__":
    for length in [4, 8, 16, 32, 64, 128, 256]:
        u = -1 + 2 * torch.rand(2, length, 32)
        dt = torch.ones(2, length, 32)
        A =  -torch.rand(32, 16)
        B = torch.rand(2, length, 16)
        C = torch.rand(2, length, 16)
        D = torch.rand(32)
        
        output_cumsum = selective_scan(u, dt, A, B, C, D, mode='cumsum')
        output_logcumsumexp = selective_scan(u, dt, A, B, C, D, mode='logcumsumexp')
    
        print(f"mismatch at length {length} is {(output_cumsum - output_logcumsumexp).abs().max()}")
    