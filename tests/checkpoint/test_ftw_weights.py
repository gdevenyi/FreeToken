"""FTW replay: dropping entries by name before their bytes are read, and the vision-tower presence check."""

import torch

from freetoken.checkpoint.ftw import FTWReader, FTWWriter, ftw_tensor_names, iter_ftw_weights
from freetoken.models.weight import ftw_lacks_vision, load_weight


def _write_ftw(out_dir, names):
    writer = FTWWriter(str(out_dir))
    tensors = {name: torch.full((4, 8), float(i), dtype=torch.bfloat16) for i, name in enumerate(names)}
    for name, tensor in tensors.items():
        writer.add_tensor(name, tensor)
    writer.finalize({})
    return tensors


def test_keep_drops_entries_before_their_bytes_are_read(tmp_path, monkeypatch):
    tensors = _write_ftw(tmp_path, ["model.a.weight", "visual.b.weight", "model.c.weight"])
    read = []
    original = FTWReader.read_into

    def spy(self, dest, entry, **kwargs):
        read.append(entry["name"])
        return original(self, dest, entry, **kwargs)

    monkeypatch.setattr(FTWReader, "read_into", spy)
    got = dict(iter_ftw_weights(str(tmp_path), keep=lambda name: not name.startswith("visual.")))
    assert list(got) == ["model.a.weight", "model.c.weight"]
    assert read == ["model.a.weight", "model.c.weight"]
    for name, tensor in got.items():
        assert torch.equal(tensor, tensors[name])


def test_no_keep_replays_every_entry(tmp_path):
    tensors = _write_ftw(tmp_path, ["model.a.weight", "visual.b.weight"])
    got = dict(iter_ftw_weights(str(tmp_path)))
    assert list(got) == list(tensors)
    assert torch.equal(got["visual.b.weight"], tensors["visual.b.weight"])


def test_load_weight_text_only_skips_the_tower(tmp_path):
    names = ["model.a.weight", "visual.b.weight", "vision_tower.c.weight"]
    _write_ftw(tmp_path, names)
    cpu = torch.device("cpu")
    assert [n for n, _ in load_weight(str(tmp_path), cpu, include_vision=False)] == ["model.a.weight"]
    assert [n for n, _ in load_weight(str(tmp_path), cpu)] == names


def test_ftw_lacks_vision(tmp_path):
    _write_ftw(tmp_path / "text", ["model.a.weight"])
    _write_ftw(tmp_path / "vl", ["model.a.weight", "visual.b.weight"])
    assert ftw_lacks_vision(str(tmp_path / "text"))
    assert not ftw_lacks_vision(str(tmp_path / "vl"))
    assert not ftw_lacks_vision(str(tmp_path))
    assert ftw_tensor_names(str(tmp_path / "vl"), "weight") == ["model.a.weight", "visual.b.weight"]


def test_ftw_banks_place_expert_rows_before_the_reads(tmp_path, monkeypatch):
    """The FTW fast path splits every expert's rows over the NUMA nodes like the safetensors
    path, for both row layouts, before the O_DIRECT reads fault the pages in."""
    import freetoken.moe.host_banks as hb
    from freetoken.checkpoint.ftw import layer_bank_entry_name, load_ftw_banks

    L, E = 2, 4
    writer = FTWWriter(str(tmp_path))
    for layer in range(L):
        writer.add_tensor(layer_bank_entry_name("gate_up_packed", layer),
                          torch.ones(E, 64, 32, dtype=torch.uint8), kind="experts_bank")
    # flat layout: layer 1 starts mid-page, so its rows are a carved view of the read window
    writer.add_tensor("down_packed", torch.ones(L * E, 32, 16, dtype=torch.uint8), kind="experts_bank")
    writer.finalize({"quant_format": "nvfp4", "expert_bank_num_layers": L})

    events = []
    original = FTWReader.read_into

    def spy(self, dest, entry, **kwargs):
        events.append("read")
        return original(self, dest, entry, **kwargs)

    monkeypatch.setattr(FTWReader, "read_into", spy)
    monkeypatch.setattr(hb, "numa_placement_nodes", lambda: [0, 1])
    monkeypatch.setattr(hb, "place_expert_rows", lambda tensors, nodes: events.append(
        ("place", [(t.data_ptr(), tuple(t.shape)) for t in tensors], nodes)))
    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    monkeypatch.delenv("FREETOKEN_BANK_CUDA_ALLOC", raising=False)

    banks = load_ftw_banks(str(tmp_path), num_layers=L)
    placed = [ev for ev in events if ev != "read"]
    assert placed and events.index("read") > events.index(placed[-1]), events
    assert all(nodes == [0, 1] for _, _, nodes in placed)
    got = sorted(t for _, tensors, _ in placed for t in tensors)
    want = sorted((t.data_ptr(), tuple(t.shape)) for per_layer in banks.sources.values() for t in per_layer)
    assert got == want
    for per_layer in banks.sources.values():
        for t in per_layer:
            assert bool((t == 1).all())
