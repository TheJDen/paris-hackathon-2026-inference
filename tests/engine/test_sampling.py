import pytest
import torch

from engine.sampling import get_probs_from_logits, rejection_sample, sample_next

device = "cuda"

def test_greedy_is_argmax():
    logits = torch.randn(100, 100, device=device)
    out = sample_next(logits, logits.new_zeros(100), logits.new_ones(100))
    assert torch.equal(out.squeeze(-1), logits.argmax(-1))

def test_top_p_zero_is_argmax():
    logits = torch.randn(100, 100, device=device)
    out = sample_next(logits, logits.new_ones(100), logits.new_zeros(100))
    assert torch.equal(out.squeeze(-1), logits.argmax(-1))

@pytest.mark.parametrize("T", [0.1, 1.0, 2.0])
def test_distribution_matches_softmax(T):
    torch.manual_seed(93)
    V, N = 100, 200_000
    logits = torch.randn(V, device=device)
    temp, top_p = logits.new_full((N,), T), logits.new_ones(N)
    batch = logits.expand(N, V)
    out = sample_next(batch, temp, top_p)
    hist = torch.bincount(out.flatten(), minlength=V) / N
    torch.testing.assert_close(hist, torch.softmax(logits / T, -1), atol=0.01, rtol=0.)

@pytest.mark.parametrize("dist", ["identical", "disjoint", "random"])
@pytest.mark.parametrize("T", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("k", [1, 2, 5])
def test_rejection_dist_matches_target(dist, T, k):
    torch.manual_seed(93)
    V, N = 100, 200_000
    p_logits = torch.randn(V, device=device)
    q_logits = torch.randn(V, device=device)
    if dist == "identical":
        q_logits = p_logits.clone()
    elif dist == "disjoint":
        p_logits[V//2:] = float("-inf")
        q_logits[:V//2] = float("-inf")
    temp, top_p = q_logits.new_full((N,), T), q_logits.new_ones(N)
    p_batch = p_logits.expand(N, V)
    ref_samples = sample_next(p_batch, temp, top_p)
    ref = torch.bincount(ref_samples.flatten(), minlength=V) / N

    p_probs = get_probs_from_logits(p_logits.expand(N, V), temp, top_p)
    verify_probs = p_probs.unsqueeze(1).expand(N, k + 1, V)
    q_probs = get_probs_from_logits(q_logits.expand(N, V), temp, top_p)
    drafts = torch.multinomial(q_probs, k, replacement=True)
    draft_probs = q_probs.unsqueeze(1).expand(N, k, V)

    n_acc, commited = rejection_sample(drafts, draft_probs, verify_probs)
    valid = torch.arange(k + 1, device=device)[None] <= n_acc[:, None]
    pooled = commited[valid]
    hist = torch.bincount(pooled, minlength=V) / pooled.numel()
    torch.testing.assert_close(hist, ref, atol=0.01, rtol=0.)

