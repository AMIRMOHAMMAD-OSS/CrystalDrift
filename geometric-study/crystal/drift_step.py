"""One training update on the two observed MgZn2 assignments in orbit_probe.json."""
import json
import math
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


@torch.no_grad()
def batch_field(generated, real, temperatures):
    """TokenDrift squared distances and Deng et al.'s two-axis kernel normalization."""
    if len(generated) < 2 or len(real) == 0:
        raise ValueError("The minibatch needs a real reference and two generated examples.")
    references = torch.cat((real, generated), dim=0)
    offsets = references.unsqueeze(0) - generated.unsqueeze(1)
    distances = offsets.square().sum(dim=-1)
    rows = torch.arange(len(generated), device=generated.device)
    distances[rows, len(real) + rows] = torch.inf
    fields = []
    for temperature in temperatures:
        log_affinity = -distances / temperature
        affinity = (0.5 * (log_affinity.log_softmax(0) + log_affinity.log_softmax(1))).exp()
        attractive, repulsive = affinity.split((len(real), len(generated)), dim=1)
        attractive_offsets, repulsive_offsets = offsets.split((len(real), len(generated)), dim=1)
        toward_data = (attractive.unsqueeze(-1) * attractive_offsets).sum(1)
        toward_model = (repulsive.unsqueeze(-1) * repulsive_offsets).sum(1)
        field = toward_data * repulsive.sum(1, keepdim=True) - toward_model * attractive.sum(1, keepdim=True)
        rms = (field.square().sum(-1).mean() + 1e-8).sqrt()
        fields.append(field / rms)
    return torch.stack(fields).mean(0)


def one_update(state_features, second_state_count, split_initialization=False):
    # Both branches preserve Mg4Zn8. Zn has either one 8g orbit or two 4c orbits.
    # The first three logits index 4c repeat counts 0, 1, 2. The last two index 8g counts 0, 1.
    logits = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    initial_offsets = torch.zeros(10, dtype=logits.dtype)
    if split_initialization:
        initial_offsets[:5] = -math.log(3)
        initial_offsets[5:] = math.log(3)

    def assignment_probabilities(z):
        first = (z[0] + z[4]).expand_as(initial_offsets)
        second = z[2] + z[3] + initial_offsets
        return torch.stack((first, second), dim=1).softmax(1)

    probabilities = assignment_probabilities(logits)
    soft = probabilities @ state_features
    real = torch.cat((state_features[0].repeat(10 - second_state_count, 1),
                      state_features[1].repeat(second_state_count, 1)))
    direction = batch_field(soft.detach(), real, (0.25, 1.0))
    target = (soft + 0.1 * direction).detach()
    loss = 0.5 * (soft - target).square().sum(-1).mean()
    gradient, = torch.autograd.grad(loss, logits)
    after = assignment_probabilities(logits.detach() - 0.5 * gradient)
    return {"target_fraction_two_4c": second_state_count / len(real),
            "probability_two_4c_before": probabilities[:, 1].mean().item(),
            "probability_two_4c_after": after[:, 1].mean().item(),
            "logit_gradient": gradient.tolist()}


def main():
    probe = json.loads((HERE / "orbit_probe.json").read_text(encoding="utf-8"))
    example = probe["categorical_gradient_probe"]
    tables = {"orbit": example["hard_species_orbit_features_Mg_Zn"],
              "composition": example["hard_species_atom_counts_Mg_Zn"]}
    results = {name: [one_update(torch.tensor(table, dtype=torch.float64), n) for n in (2, 5, 8)]
               for name, table in tables.items()}
    after = [run["probability_two_4c_after"] for run in results["orbit"]]
    assert after[0] < 0.5 < after[2] and abs(after[1] - 0.5) < 1e-10
    assert all(run["probability_two_4c_after"] == 0.5 for run in results["composition"])
    assert all(run["logit_gradient"][1] == 0 for group in results.values() for run in group)
    # Distinct soft features exercise repulsion at each query, using one shared trainable head.
    orbit_table = torch.tensor(tables["orbit"], dtype=torch.float64)
    split_runs = [one_update(orbit_table, n, split_initialization=True) for n in (2, 5, 8)]
    split_after = [run["probability_two_4c_after"] for run in split_runs]
    assert split_after[0] < 0.5 < split_after[2] and abs(split_after[1] - 0.5) < 1e-10
    output = {"materials_in_state_order": example["state_material_ids"], "torch": torch.__version__,
              "batch_size": 10, "temperatures": [0.25, 1.0], "drift_scale": 0.1, "learning_rate": 0.5,
              "comparison": results, "orbit_initial_probabilities_half_0.25_half_0.75": split_runs}
    (HERE / "drift_step.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Probability of the two-4c assignment after one update")
    for name, group in results.items():
        print(name, [round(run["probability_two_4c_after"], 6) for run in group])
    print("orbit, split initialization", [round(p, 6) for p in split_after])


if __name__ == "__main__":
    main()
