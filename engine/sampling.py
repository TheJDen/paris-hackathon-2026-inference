import torch

def sample_next(logits: torch.Tensor, temperature: torch.Tensor, top_p: torch.Tensor):
    temp = temperature.view(-1, 1)
    greedy = temp == 0
    safe = torch.where(greedy, torch.ones_like(temp), temp)
    scaled = logits / safe

    s_logits, s_idx = torch.sort(scaled, descending=True, dim=-1)
    cum = torch.softmax(s_logits, dim=-1).cumsum(dim=-1)
    remove = cum > top_p.view(-1, 1)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    s_logits = s_logits.masked_fill(remove, float("-inf"))
    scaled = torch.full_like(scaled, float("-inf")).scatter(-1, s_idx, s_logits)

    probs = torch.softmax(scaled, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    greedy_tok = torch.argmax(logits, dim=-1, keepdim=True)
    return torch.where(greedy, greedy_tok, sampled)

