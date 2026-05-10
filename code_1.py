"""
Geometric Invariance Test for Kinematic Decoder
================================================
Experiment accompanying the ФО 78.0 (Philosophy of Awareness 78.0) framework.

Core question: is semantic distinction stored in the geometry of latent space,
or only in specific weight coordinates?

Two decoders are compared under geometric transformations of latent space:
  - BaselineDecoder: reads latent coordinates directly (coordinate-dependent)
  - KinematicDecoder: reads trajectory shape (velocity, acceleration, cosine)

If the kinematic decoder is invariant to rotation while the baseline degrades,
distinction is encoded in trajectory shape, not coordinate position.

Note on projection invariance: KinematicDecoder uses F.normalize internally,
which makes it partially invariant to scaling by construction. Projection
robustness should be interpreted as an architectural artifact, not a
meaningful invariant. The informative tests are orthogonal rotation and
affine distortion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 5
D_MODEL = 16
EPOCHS  = 600
EPS     = 1e-4

# ==========================================
# 1. TWO DECODERS — HONEST COMPARISON
# ==========================================

class BaselineDecoder(nn.Module):
    """
    Standard decoder: reads latent coordinates directly.
    Coordinate-dependent — expected to degrade under rotation.
    """
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, h0, h1, h2):
        # Uses only h0 — fully coordinate-dependent
        return torch.sigmoid(self.linear(h0))


class KinematicDecoder(nn.Module):
    """
    Kinematic decoder: reads the shape of the recursive trajectory.
    Uses velocity, acceleration, and their cosine similarity.
    Invariant to orthogonal transformations by design (normalized inputs).
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, h0, h1, h2):
        # Normalize to remove scale dependence
        h0_n = F.normalize(h0, dim=-1)
        h1_n = F.normalize(h1, dim=-1)
        h2_n = F.normalize(h2, dim=-1)

        v = h1_n - h0_n                      # velocity: first difference
        a = h2_n - 2 * h1_n + h0_n          # acceleration: second difference

        norm_v = v.norm(dim=-1, keepdim=True)
        norm_a = a.norm(dim=-1, keepdim=True)
        cos_va = F.cosine_similarity(v, a, dim=-1).unsqueeze(-1)

        features = torch.cat([norm_v, norm_a, cos_va], dim=-1)
        return torch.sigmoid(self.linear(features))


# ==========================================
# 2. SHARED TRANSFORMER BACKBONE
# ==========================================

class CognitiveTransformer(nn.Module):
    """
    Micro-transformer with recursive echo steps.
    Shared backbone for both decoders — only the decoder differs.
    """
    def __init__(self, decoder, seq_len=SEQ_LEN, d_model=D_MODEL):
        super().__init__()
        self.encoder   = nn.Linear(seq_len, d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=2, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2   = nn.LayerNorm(d_model)
        self.decoder = decoder

    def _step(self, h):
        """Single recursive echo step."""
        hr = h.unsqueeze(1)
        hr = self.norm1(hr + self.attention(hr, hr, hr)[0])
        return self.norm2(hr + self.ffn(hr)).squeeze(1)

    def forward(self, x, transform=None):
        """
        Forward pass with optional geometric transformation applied
        to latent vectors before the decoder.
        transform: [D, D] matrix applied as h @ transform
        """
        h = self.encoder(x).unsqueeze(1)
        h = self.norm1(h + self.attention(h, h, h)[0])
        h0 = self.norm2(h + self.ffn(h)).squeeze(1)
        h1 = self._step(h0)
        h2 = self._step(h1)

        # Apply geometric transformation before decoding
        if transform is not None:
            h0 = h0 @ transform
            h1 = h1 @ transform
            h2 = h2 @ transform

        return self.decoder(h0, h1, h2)


# ==========================================
# 3. DATA GENERATION
# ==========================================

def get_data_firstmin_hard(batch=1000):
    """
    FirstMin task with hard negatives.
    Label = 1 if first element is strictly minimum, 0 otherwise.
    Hard negatives: first element placed at EPS above/below the minimum
    of remaining elements — forces learning a sharp decision boundary.
    """
    # Easy examples: random sequences
    x_norm = torch.rand(batch // 2, SEQ_LEN, device=device)
    y_norm = torch.zeros(batch // 2, 1, device=device)

    # Hard negatives: first element near the boundary
    x_hard = torch.rand(batch // 2, SEQ_LEN, device=device)
    min_others = x_hard[:, 1:].min(dim=1, keepdim=True)[0]
    coin = torch.randint(0, 2, (batch // 2, 1), device=device).float()
    x_hard[:, 0:1] = min_others + (coin * EPS) - ((1 - coin) * EPS)
    y_hard = (x_hard[:, 0:1] < x_hard[:, 1:]).all(dim=1, keepdim=True).float()

    return torch.cat([x_norm, x_hard]), torch.cat([y_norm, y_hard])


# ==========================================
# 4. TRAINING AND EVALUATION
# ==========================================

def train(model, x, y, epochs=EPOCHS, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    return model


def accuracy(model, x, y, transform=None):
    model.eval()
    with torch.no_grad():
        pred = model(x, transform=transform)
        return ((pred > 0.5) == (y > 0.5)).float().mean().item()


# ==========================================
# 5. GEOMETRIC TRANSFORMATIONS
# ==========================================

def make_transforms(d_model=D_MODEL):
    """
    Four transformations of increasing severity:
      R — orthogonal rotation (isometry, preserves distances)
      S — isotropic scaling (preserves angles)
      A — affine distortion (rotation + stretching)
      P — projection (irreversible information loss)

    P is a control case: both decoders should degrade.
    If KinematicDecoder is robust to P, this is an architectural
    artifact (normalization), not a meaningful invariant.
    """
    # Orthogonal rotation via QR decomposition
    R, _ = torch.linalg.qr(
        torch.randn(d_model, d_model, device=device)
    )

    # Isotropic scaling
    S = 3.0 * torch.eye(d_model, device=device)

    # Affine: rotation composed with random stretching
    A = R @ (torch.eye(d_model, device=device) +
             0.5 * torch.randn(d_model, d_model, device=device))

    # Projection: zero out half the directions (irreversible)
    P = torch.randn(d_model, d_model, device=device)
    P[:, :d_model // 2] = 0.0

    return {
        "Orthogonal rotation":          R,
        "Isotropic scaling (x3.0)":     S,
        "Affine distortion":            A,
        "Projection (info loss)":       P,
    }


# ==========================================
# 6. RUN EXPERIMENT
# ==========================================

print("Generating data...")
x_train, y_train = get_data_firstmin_hard(1000)
x_test,  y_test  = get_data_firstmin_hard(500)

print(f"\nTraining Baseline decoder ({EPOCHS} epochs)...")
model_base = CognitiveTransformer(BaselineDecoder()).to(device)
train(model_base, x_train, y_train)

print(f"Training Kinematic decoder ({EPOCHS} epochs)...")
model_kine = CognitiveTransformer(KinematicDecoder()).to(device)
train(model_kine, x_train, y_train)

transforms = make_transforms()

acc_base_clean = accuracy(model_base, x_test, y_test)
acc_kine_clean = accuracy(model_kine, x_test, y_test)

print("\n" + "=" * 80)
print("GEOMETRIC INVARIANCE TEST — FirstMin(Hard)")
print(f"{'Transformation':<35} | {'Baseline':>10} | {'Kinematic':>10} | {'Delta':>8}")
print("-" * 80)
print(f"{'No transformation':<35} | "
      f"{acc_base_clean:>10.4f} | {acc_kine_clean:>10.4f} | "
      f"{acc_kine_clean - acc_base_clean:>+8.4f}")

for name, T in transforms.items():
    ab = accuracy(model_base, x_test, y_test, transform=T)
    ak = accuracy(model_kine, x_test, y_test, transform=T)
    flag = " <-- INVARIANT" if abs(ak - acc_kine_clean) < 0.05 else ""
    print(f"{name:<35} | {ab:>10.4f} | {ak:>10.4f} | "
          f"{ak - ab:>+8.4f}{flag}")

print("=" * 80)
print("""
INTERPRETATION:
  Baseline   — reads latent coordinates directly (coordinate-dependent).
  Kinematic  — reads trajectory shape: velocity, acceleration, cosine.

  Invariant confirmed if:
    Kinematic loses < 5% accuracy after transformation
    Baseline degrades significantly more

  Invariant disconfirmed if:
    Both decoders degrade similarly
    (kinematic features are also coordinate-dependent)

  Projection is a control case: information is irreversibly lost.
  Both decoders should degrade. Kinematic robustness to projection
  is an artifact of internal normalization, not a meaningful invariant.
  The informative tests are orthogonal rotation and affine distortion.
""")
