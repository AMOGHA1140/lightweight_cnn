"""Synthetic unit checks for GAConv (no dataset required).

Run:  python -m tests.test_gaconv
"""

import torch
import torch.nn.functional as F

from common.gaconv import GAConv


def test_output_shape():
    gaconv = GAConv(128)
    x = torch.randn(2, 128, 64, 64)
    y = gaconv(x)
    assert y.shape == x.shape, y.shape
    print("output_shape: ok", tuple(y.shape))


def test_identity_init():
    # At init the geometry is identity (theta=0, sigma=1) so the offsets are
    # zero and the deform conv collapses to a plain depthwise 3x3. We compare
    # against that depthwise conv (pw_proj excluded).
    gaconv = GAConv(32).eval()
    x = torch.randn(2, 32, 16, 16)
    theta, smaj, smin = gaconv.geometry(x)
    offsets = gaconv._compute_offsets(theta, smaj, smin)
    assert offsets.abs().max().item() < 1e-5, offsets.abs().max().item()

    from torchvision.ops import deform_conv2d
    deformed = deform_conv2d(x, offsets, gaconv.dw_weight, padding=1)
    ref = F.conv2d(x, gaconv.dw_weight, padding=1, groups=32)
    assert torch.allclose(deformed, ref, atol=1e-5), (deformed - ref).abs().max().item()
    print("identity_init: ok (offsets~0, deform==grouped conv)")


def test_gradients():
    gaconv = GAConv(16)
    x = torch.randn(2, 16, 24, 24, requires_grad=True)
    gaconv(x).pow(2).mean().backward()
    for name in ("geometry_pred", "pw_proj"):
        w = getattr(gaconv, name).weight
        assert w.grad is not None and torch.isfinite(w.grad).all(), name
    assert gaconv.dw_weight.grad is not None and torch.isfinite(gaconv.dw_weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("gradients: ok (geometry_pred, dw_weight, pw_proj, input)")


def test_offset_orientation():
    # Force a +90 degree rotation and check the offsets rotate the sampling
    # grid as expected: a point on the +x axis (gx=1, gy=0) should map toward
    # the +y axis, i.e. acquire a large +dy and cancel its dx.
    gaconv = GAConv(8)
    B, H, W = 1, 4, 4
    theta = torch.full((B, 1, H, W), 3.141592653589793 / 2)
    smaj = torch.ones(B, 1, H, W)
    smin = torch.ones(B, 1, H, W)
    offsets = gaconv._compute_offsets(theta, smaj, smin)  # [B,18,H,W]
    # cell index 5 is (kh=1, kw=2) -> base (gy=0, gx=1); channels (dy,dx)=(10,11)
    dy = offsets[0, 10, 0, 0].item()
    dx = offsets[0, 11, 0, 0].item()
    assert dy > 0.9 and abs(dx + 1.0) < 1e-4, (dy, dx)
    print(f"offset_orientation: ok (+90deg maps (gx=1,gy=0) -> dy={dy:.3f}, dx={dx:.3f})")


def main():
    torch.manual_seed(0)
    test_output_shape()
    test_identity_init()
    test_gradients()
    test_offset_orientation()
    if torch.cuda.is_available():
        gaconv = GAConv(64).cuda()
        y = gaconv(torch.randn(1, 64, 32, 32, device="cuda"))
        assert y.shape == (1, 64, 32, 32)
        print("cuda_forward: ok")
    print("\nall GAConv tests passed")


if __name__ == "__main__":
    main()
