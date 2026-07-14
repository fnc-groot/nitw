import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

warnings.filterwarnings("ignore")

SEED = 42
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


class ChestXrayDataset:
    """
    Loads 'NORMAL' and 'PNEUMONIA' chest X-ray images for binary classification.
    Label: 0 -> NORMAL, 1 -> PNEUMONIA
    """

    def __init__(self, root, classes, transform=None, max_per_class=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.classes = classes

        for label, cls in enumerate(classes):
            cls_dir = Path(root) / cls
            imgs = sorted(cls_dir.glob("*.*"))
            imgs = [p for p in imgs if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            if max_per_class:
                imgs = imgs[:max_per_class]
            for img_path in imgs:
                self.samples.append(str(img_path))
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


def build_pqc(n_qubits: int):
    """
    Single-stage PQC matching the notebook implementation.

    Data encoding : Ry(theta_i)
    Trainable     : Rx(alpha_i), Rz(phi_i), Ry(lam_i)
    Entanglement  : U-shaped CNOT ladder
    """
    qc = QuantumCircuit(n_qubits)
    theta = ParameterVector("theta", n_qubits)
    alpha = ParameterVector("alpha", n_qubits)
    phi = ParameterVector("phi", n_qubits)
    lam = ParameterVector("lam", n_qubits)

    for i in range(n_qubits):
        qc.ry(theta[i], i)

    for i in range(n_qubits):
        qc.rx(alpha[i], i)

    for i in range(n_qubits):
        qc.rz(phi[i], i)

    for i in range(n_qubits):
        qc.ry(lam[i], i)

    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    for i in range(n_qubits - 1, 0, -1):
        qc.cx(i, i - 1)

    return qc, theta, alpha, phi, lam


def build_z_observables(n_qubits):
    observables = []
    for i in range(n_qubits):
        pauli_str = "I" * (n_qubits - 1 - i) + "Z" + "I" * i
        observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))
    return observables


def quantum_forward_single(theta_vals, alpha_vals, phi_vals, lam_vals, qc, theta_p, alpha_p, phi_p, lam_p, z_obs):
    n_qubits = len(theta_vals)

    param_dict = {}
    for i, p in enumerate(theta_p):
        param_dict[p] = float(theta_vals[i])
    for i, p in enumerate(alpha_p):
        param_dict[p] = float(alpha_vals[i])
    for i, p in enumerate(phi_p):
        param_dict[p] = float(phi_vals[i])
    for i, p in enumerate(lam_p):
        param_dict[p] = float(lam_vals[i])

    bound_qc = qc.assign_parameters(param_dict)
    sv = Statevector(bound_qc)
    probs = sv.probabilities()

    exp_vals = np.zeros(n_qubits)
    for i in range(n_qubits):
        p0, p1 = 0.0, 0.0
        for state_idx in range(2 ** n_qubits):
            bit_i = (state_idx >> i) & 1
            if bit_i == 0:
                p0 += probs[state_idx]
            else:
                p1 += probs[state_idx]
        exp_vals[i] = p0 - p1

    return exp_vals


class SingleStagePQCFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta_t, alpha_t, phi_t, lam_t, qc, theta_p, alpha_p, phi_p, lam_p, z_obs):
        ctx.save_for_backward(theta_t, alpha_t, phi_t, lam_t)
        ctx.qc = qc
        ctx.theta_p = theta_p
        ctx.alpha_p = alpha_p
        ctx.phi_p = phi_p
        ctx.lam_p = lam_p
        ctx.z_obs = z_obs

        batch_size = theta_t.shape[0]
        n_qubits = theta_t.shape[1]

        theta_np = theta_t.detach().cpu().numpy()
        alpha_np = alpha_t.detach().cpu().numpy()
        phi_np = phi_t.detach().cpu().numpy()
        lam_np = lam_t.detach().cpu().numpy()

        outputs = np.zeros((batch_size, n_qubits))
        for b in range(batch_size):
            outputs[b] = quantum_forward_single(
                theta_np[b], alpha_np, phi_np, lam_np, qc, theta_p, alpha_p, phi_p, lam_p, z_obs
            )

        return torch.tensor(outputs, dtype=torch.float32, device=theta_t.device)

    @staticmethod
    def backward(ctx, grad_output):
        theta_t, alpha_t, phi_t, lam_t = ctx.saved_tensors
        qc = ctx.qc
        theta_p = ctx.theta_p
        alpha_p = ctx.alpha_p
        phi_p = ctx.phi_p
        lam_p = ctx.lam_p
        z_obs = ctx.z_obs

        shift = np.pi / 2
        batch_size = theta_t.shape[0]
        n_qubits = theta_t.shape[1]

        theta_np = theta_t.detach().cpu().numpy()
        alpha_np = alpha_t.detach().cpu().numpy()
        phi_np = phi_t.detach().cpu().numpy()
        lam_np = lam_t.detach().cpu().numpy()
        grad_np = grad_output.detach().cpu().numpy()

        def fwd(th, a, p, l):
            return quantum_forward_single(th, a, p, l, qc, theta_p, alpha_p, phi_p, lam_p, z_obs)

        grad_alpha = np.zeros_like(alpha_np)
        for j in range(n_qubits):
            acc = 0.0
            for b in range(batch_size):
                a_p = alpha_np.copy()
                a_m = alpha_np.copy()
                a_p[j] += shift
                a_m[j] -= shift
                e_p = fwd(theta_np[b], a_p, phi_np, lam_np)
                e_m = fwd(theta_np[b], a_m, phi_np, lam_np)
                acc += np.sum(grad_np[b] * (e_p - e_m) / 2)
            grad_alpha[j] = acc / batch_size

        grad_phi = np.zeros_like(phi_np)
        for j in range(n_qubits):
            acc = 0.0
            for b in range(batch_size):
                p_p = phi_np.copy()
                p_m = phi_np.copy()
                p_p[j] += shift
                p_m[j] -= shift
                e_p = fwd(theta_np[b], alpha_np, p_p, lam_np)
                e_m = fwd(theta_np[b], alpha_np, p_m, lam_np)
                acc += np.sum(grad_np[b] * (e_p - e_m) / 2)
            grad_phi[j] = acc / batch_size

        grad_lam = np.zeros_like(lam_np)
        for j in range(n_qubits):
            acc = 0.0
            for b in range(batch_size):
                l_p = lam_np.copy()
                l_m = lam_np.copy()
                l_p[j] += shift
                l_m[j] -= shift
                e_p = fwd(theta_np[b], alpha_np, phi_np, l_p)
                e_m = fwd(theta_np[b], alpha_np, phi_np, l_m)
                acc += np.sum(grad_np[b] * (e_p - e_m) / 2)
            grad_lam[j] = acc / batch_size

        grad_theta = np.zeros_like(theta_np)
        for b in range(batch_size):
            for i in range(n_qubits):
                th_p = theta_np[b].copy()
                th_m = theta_np[b].copy()
                th_p[i] += shift
                th_m[i] -= shift
                e_p = fwd(th_p, alpha_np, phi_np, lam_np)
                e_m = fwd(th_m, alpha_np, phi_np, lam_np)
                grad_theta[b, i] = np.sum(grad_np[b] * (e_p - e_m) / 2)

        return (
            torch.tensor(grad_theta, dtype=torch.float32, device=theta_t.device),
            torch.tensor(grad_alpha, dtype=torch.float32, device=alpha_t.device),
            torch.tensor(grad_phi, dtype=torch.float32, device=phi_t.device),
            torch.tensor(grad_lam, dtype=torch.float32, device=lam_t.device),
            None, None, None, None, None, None,
        )


class SingleStagePQCLayer(nn.Module):
    def __init__(self, n_qubits, qc, theta_p, alpha_p, phi_p, lam_p, z_obs):
        super().__init__()
        self.n_qubits = n_qubits
        self.qc = qc
        self.theta_p = theta_p
        self.alpha_p = alpha_p
        self.phi_p = phi_p
        self.lam_p = lam_p
        self.z_obs = z_obs

        self.alpha_weights = nn.Parameter(torch.randn(n_qubits) * 0)
        self.phi_weights = nn.Parameter(torch.randn(n_qubits) * 0)
        self.lam_weights = nn.Parameter(torch.randn(n_qubits) * 0)

    def forward(self, theta_tensor):
        return SingleStagePQCFunction.apply(
            theta_tensor,
            self.alpha_weights,
            self.phi_weights,
            self.lam_weights,
            self.qc,
            self.theta_p,
            self.alpha_p,
            self.phi_p,
            self.lam_p,
            self.z_obs,
        )


class CNNFeatureExtractor(nn.Module):
    """
    CNN Feature Extractor:
      Input  : (B, 3, H, W)
      Output : (B, n_features) — features scaled to [-pi, pi]
    """

    def __init__(self, n_features=6, img_size=64):
        super().__init__()
        self.n_features = n_features

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        flat_size = 64 * (img_size // 8) * (img_size // 8)

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_features),
        )

        self.angle_scale = nn.Parameter(torch.tensor(np.pi), requires_grad=False)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        x = torch.tanh(x) * self.angle_scale
        return x


class HQNNModel(nn.Module):
    def __init__(self, n_features, n_qubits, qc, theta_p, alpha_p, phi_p, lam_p, z_obs, img_size):
        super().__init__()
        self.cnn = CNNFeatureExtractor(n_features=n_features, img_size=img_size)
        self.quantum = SingleStagePQCLayer(
            n_qubits=n_qubits,
            qc=qc,
            theta_p=theta_p,
            alpha_p=alpha_p,
            phi_p=phi_p,
            lam_p=lam_p,
            z_obs=z_obs,
        )
        self.classical_head = nn.Sequential(nn.Linear(n_qubits, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print("CNN START")
        features = self.cnn(x)
        print("CNN DONE")
        return torch.zeros((x.shape[0],), device=x.device)


def load_config(base_dir: Path):
    config_path = base_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json at {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_checkpoint_path(base_dir: Path):
    return base_dir / "hqnn_checkpoint.pth"


def load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and any(k.startswith("cnn.") or k.startswith("quantum.") for k in checkpoint.keys()):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and len(checkpoint) > 0 and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise RuntimeError(f"Unsupported checkpoint format in {checkpoint_path}")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch. Missing: {missing}; Unexpected: {unexpected}")

    return model


def preprocess_image(image_path, img_size, mean, std):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transform(image).unsqueeze(0)


def main():
    base_dir = Path(__file__).resolve().parent

    if len(sys.argv) != 2:
        print("Usage:")
        print("python main.py image.jpg")
        return 1

    image_path = Path(sys.argv[1])
    if not image_path.exists() or not image_path.is_file():
        print(f"Error: wrong image path or file does not exist: {image_path}")
        return 1

    checkpoint_path = resolve_checkpoint_path(base_dir)
    if not checkpoint_path.exists():
        print(f"Error: missing checkpoint: {checkpoint_path}")
        return 1

    try:
        image = Image.open(image_path)
        image.verify()
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"Error: invalid image: {image_path} ({exc})")
        return 1

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA unavailable. Using CPU fallback.")
        print("STEP A")

    try:
        config = load_config(base_dir)
        print("STEP B")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    img_size = int(config.get("img_size", 28))
    n_features = int(config.get("n_features", 4))
    n_qubits = int(config.get("n_qubits", 4))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    qc, theta_params, alpha_params, phi_params, lam_params = build_pqc(n_qubits)
    print("STEP C")
    z_observables = build_z_observables(n_qubits)

    model = HQNNModel(
        n_features=n_features,
        n_qubits=n_qubits,
        qc=qc,
        theta_p=theta_params,
        alpha_p=alpha_params,
        phi_p=phi_params,
        lam_p=lam_params,
        z_obs=z_observables,
        img_size=img_size,
    )
    print("STEP D")
    model = model.to(device)

    try:
        model = load_checkpoint(model, checkpoint_path, device)
        print("STEP E")
    except Exception as exc:
        print(f"Error: unable to load checkpoint: {exc}")
        return 1

    model.eval()

    with torch.no_grad():
        try:
            input_tensor = preprocess_image(image_path, img_size, mean, std).to(device)
            print("STEP F")
            probs = model(input_tensor)
            print("STEP G")
        except Exception as exc:
            print(f"Error: invalid image preprocessing: {exc}")
            return 1

    prob = probs[0].item()
    if prob >= 0.5:
        prediction = "PNEUMONIA"
    else:
        prediction = "NORMAL"

    confidence = prob * 100.0
    print("----------------------------------")
    print(f"Prediction : {prediction}")
    print(f"Confidence : {confidence:.2f}%")
    print("----------------------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
