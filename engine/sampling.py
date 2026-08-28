import torch
import torch.nn.functional as F


def get_probs_from_logits(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor
) -> torch.Tensor:
    trail = (1,) * (logits.ndim - temperature.ndim)
    temp = temperature.view(*temperature.shape, *trail)
    top_p = top_p.view(*top_p.shape, *trail)
    greedy = temp == 0
    scaled = logits / temp.masked_fill(greedy, 1.0)

    s_logits, s_idx = torch.sort(scaled, descending=True, dim=-1)
    s_probs = torch.softmax(s_logits, dim=-1)
    remove = (s_probs.cumsum(dim=-1) - s_probs) > top_p
    s_logits = s_logits.masked_fill(remove, float("-inf"))
    scaled.scatter_(-1, s_idx, s_logits)

    probs = torch.softmax(scaled, dim=-1)
    probs.masked_fill_(greedy, 0.0)
    argmax = logits.argmax(-1, keepdim=True)
    probs.scatter_add_(-1, argmax, greedy.to(probs.dtype).expand_as(argmax))
    return probs

def sample_next(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor
) -> torch.Tensor:
    probs = get_probs_from_logits(logits, temperature, top_p)
    return torch.multinomial(probs, num_samples=1)

def rejection_sample(
    drafts: torch.LongTensor,
    draft_probs: torch.Tensor,
    verify_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, k = drafts.shape
    p = verify_probs[:, :k].gather(-1, drafts.unsqueeze(-1)).squeeze(-1)
    q = draft_probs.gather(-1, drafts.unsqueeze(-1)).squeeze(-1)
    accept = torch.rand_like(q) * q < p
    n_acc = accept.cumprod(1).sum(1)
    q_probs = F.pad(draft_probs, (0, 0, 0, 1)) # we want the k + 1 position to be 0 in case all drafts accepted
    b = torch.arange(B, device=drafts.device)
    q_n = q_probs[b, n_acc]
    p_n = verify_probs[b, n_acc]
    residual_probs = F.normalize(torch.relu(p_n - q_n), p=1, dim=-1) # L1 norm along vocab
    committed = F.pad(drafts, (0, 1)) # [B, k + 1] with drafts filled
    replacement = torch.multinomial(residual_probs, num_samples=1)
    committed.scatter_(-1, n_acc[:, None], replacement)
    return n_acc, committed
