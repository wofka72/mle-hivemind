import json
import os.path

import torch
from tqdm import trange

from lib.models.gpt import LeanGPTConfig, LeanGPTForPreTraining

HERE = os.path.abspath(os.path.dirname(__file__)) + "/"


def test_gpt_forward_backward(filename: str = HERE + "gpt_test_data.pth"):
    """
    tests the correctness model outputs, loss and gradients for a small GPT model with all the features:
    - rotary embeddings
    - sandwich norm
    - parameter sharing with adapters
    - pixelated butterfly
    - reversible stem

    :note: see generate_gpt_test_data below for the training procedure and references
    """
    torch.manual_seed(1337)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    atol, rtol = 1e-4, float("inf")
    test_data = torch.load(filename)

    config = LeanGPTConfig(**json.loads(test_data["config"]), out_proj_bias=True, tie_embedding_hidden_mapping=True)
    # note: the keyword arguments in the previous line are for backward compatibility with the checkpoint
    model = LeanGPTForPreTraining(config).train(False)

    report = model.load_state_dict(test_data["state"])
    assert not report.missing_keys and not report.unexpected_keys, f"Checkpoint keys mismatch: {report}"

    model.zero_grad()
    output = model(**test_data["batch"])
    output.loss.backward()

    assert torch.allclose(output["loss"], test_data["loss"], rtol=rtol, atol=atol), "Loss does not match reference"
    assert torch.allclose(output["logits"], test_data["logits"], rtol=rtol, atol=atol), "Logits do not match reference"
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Param {name} has no grad"
        assert torch.allclose(param.grad, test_data["grads"][name], rtol=rtol, atol=atol), \
            f"Grad w.r.t. {name} does not match reference"


def generate_gpt_test_data(
    filename: str = HERE + "gpt_test_data.pth", num_steps: int = 250, loss_threshold: float = 1.0
):
    torch.manual_seed(1337)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    config_json = b"""{
      "architectures": ["LeanGPTForPretraining"],
      "model_type": "lean_gpt",
      "num_hidden_layers": 8,
      "num_hidden_groups": 4,
      "num_inner_groups": 2,
      "num_attention_heads": 4,
      "vocab_size": 10000,
      "hidden_size": 256,
      "intermediate_size": 1024,
      "embedding_size": 64,
      "block_size": 8,
      "lowrank_dim": 60,
      "adapter_dim": 4,
      "share_large_matrices": true,
      "reversible": true,
      "hidden_act": "gelu_new",
      "hidden_act_gated": true,
      "sandwich_norm": true,
      "position_embedding_type": "rotary",
      "hidden_dropout_prob": 0,
      "classifier_dropout_prob": 0.0,
      "attention_probs_dropout_prob": 0,
      "layer_norm_eps": 1e-12,
      "type_vocab_size": 2,
      "pad_token_id": 0,
      "bos_token_id": 2,
      "eos_token_id": 3
    }"""
    batch = {
        "input_ids": torch.tensor(
            [
                [
                    610,
                    6245,
                    133,
                    25,
                    7106,
                    11,
                    2329,
                    11,
                    290,
                    9552,
                    6127,
                    198,
                    198,
                    24,
                    25,
                    1129,
                    3001,
                    6185,
                    642,
                    9502,
                    628,
                    628,
                    198,
                    198,
                    4342,
                    318,
                    644,
                    468,
                    587,
                    1016,
                    319,
                    428,
                    1285,
                    379,
                    1438,
                    397,
                    3880,
                    3799,
                    13,
                    628,
                    628,
                    198,
                    198,
                    3952,
                    352,
                    13,
                    20,
                    13,
                    16,
                    350,
                    7474,
                    198,
                    198,
                    5962,
                    572,
                    11,
                    356,
                    1392,
                    4296,
                    352,
                    13,
                    20,
                    13,
                    16,
                ],
                [
                    357,
                    2288,
                    8,
                    2817,
                    13,
                    198,
                    198,
                    38,
                    13,
                    41,
                    13,
                    5498,
                    11,
                    371,
                    13,
                    35,
                    13,
                    2644,
                    11,
                    8687,
                    13,
                    5416,
                    13,
                    360,
                    7632,
                    357,
                    1113,
                    8,
                    4353,
                    4790,
                    13,
                    198,
                    198,
                    7841,
                    1548,
                    6060,
                    4912,
                    11,
                    564,
                    251,
                    4832,
                    286,
                    2142,
                    1548,
                    3123,
                    447,
                    251,
                    11,
                    6554,
                    13,
                    8687,
                    13,
                    449,
                    13,
                    327,
                    513,
                    357,
                    1113,
                    8,
                    352,
                    13,
                    198,
                    198,
                    44,
                ],
                [
                    5694,
                    3414,
                    326,
                    262,
                    8545,
                    561,
                    1441,
                    329,
                    257,
                    2368,
                    1622,
                    286,
                    511,
                    2277,
                    905,
                    13,
                    564,
                    250,
                    6385,
                    2486,
                    1839,
                    302,
                    12,
                    4300,
                    11,
                    340,
                    691,
                    2331,
                    3148,
                    326,
                    356,
                    561,
                    1577,
                    7383,
                    1222,
                    350,
                    6213,
                    1194,
                    1622,
                    11,
                    447,
                    251,
                    2329,
                    5694,
                    1182,
                    286,
                    2656,
                    8300,
                    290,
                    3227,
                    8758,
                    2770,
                    805,
                    531,
                    13,
                    256,
                    9999,
                    9999,
                    9999,
                    9999,
                    9999,
                    9999,
                    9999,
                    9999,
                ],
                [
                    1722,
                    262,
                    2431,
                    4378,
                    276,
                    3371,
                    5896,
                    11,
                    4662,
                    278,
                    4671,
                    287,
                    3162,
                    3751,
                    645,
                    5895,
                    286,
                    7163,
                    848,
                    1612,
                    625,
                    4581,
                    290,
                    360,
                    2260,
                    198,
                    198,
                    1212,
                    2708,
                    318,
                    517,
                    621,
                    362,
                    812,
                    1468,
                    198,
                    198,
                    1212,
                    2708,
                    318,
                    517,
                    621,
                    362,
                    812,
                    1468,
                    198,
                    198,
                    464,
                    1294,
                    1230,
                    319,
                    3217,
                    2318,
                    260,
                    992,
                    3812,
                    663,
                    717,
                    8325,
                    287,
                    517,
                    621,
                    1440,
                    812,
                ],
            ]
        ),
        "attention_mask": torch.tensor(
            [
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
            ]
        ),
        "labels": torch.tensor(
            [
                [
                    610,
                    6245,
                    133,
                    25,
                    7106,
                    11,
                    2329,
                    11,
                    290,
                    9552,
                    6127,
                    198,
                    198,
                    24,
                    25,
                    1129,
                    3001,
                    6185,
                    642,
                    9502,
                    628,
                    628,
                    198,
                    198,
                    4342,
                    318,
                    644,
                    468,
                    587,
                    1016,
                    319,
                    428,
                    1285,
                    379,
                    1438,
                    397,
                    3880,
                    3799,
                    13,
                    628,
                    628,
                    198,
                    198,
                    3952,
                    352,
                    13,
                    20,
                    13,
                    16,
                    350,
                    7474,
                    198,
                    198,
                    5962,
                    572,
                    11,
                    356,
                    1392,
                    4296,
                    352,
                    13,
                    20,
                    13,
                    16,
                ],
                [
                    357,
                    2288,
                    8,
                    2817,
                    13,
                    198,
                    198,
                    38,
                    13,
                    41,
                    13,
                    5498,
                    11,
                    371,
                    13,
                    35,
                    13,
                    2644,
                    11,
                    8687,
                    13,
                    5416,
                    13,
                    360,
                    7632,
                    357,
                    1113,
                    8,
                    4353,
                    4790,
                    13,
                    198,
                    198,
                    7841,
                    1548,
                    6060,
                    4912,
                    11,
                    564,
                    251,
                    4832,
                    286,
                    2142,
                    1548,
                    3123,
                    447,
                    251,
                    11,
                    6554,
                    13,
                    8687,
                    13,
                    449,
                    13,
                    327,
                    513,
                    357,
                    1113,
                    8,
                    352,
                    13,
                    198,
                    198,
                    44,
                ],
                [
                    5694,
                    3414,
                    326,
                    262,
                    8545,
                    561,
                    1441,
                    329,
                    257,
                    2368,
                    1622,
                    286,
                    511,
                    2277,
                    905,
                    13,
                    564,
                    250,
                    6385,
                    2486,
                    1839,
                    302,
                    12,
                    4300,
                    11,
                    340,
                    691,
                    2331,
                    3148,
                    326,
                    356,
                    561,
                    1577,
                    7383,
                    1222,
                    350,
                    6213,
                    1194,
                    1622,
                    11,
                    447,
                    251,
                    2329,
                    5694,
                    1182,
                    286,
                    2656,
                    8300,
                    290,
                    3227,
                    8758,
                    2770,
                    805,
                    531,
                    13,
                    256,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                    -100,
                ],
                [
                    1722,
                    262,
                    2431,
                    4378,
                    276,
                    3371,
                    5896,
                    11,
                    4662,
                    278,
                    4671,
                    287,
                    3162,
                    3751,
                    645,
                    5895,
                    286,
                    7163,
                    848,
                    1612,
                    625,
                    4581,
                    290,
                    360,
                    2260,
                    198,
                    198,
                    1212,
                    2708,
                    318,
                    517,
                    621,
                    362,
                    812,
                    1468,
                    198,
                    198,
                    1212,
                    2708,
                    318,
                    517,
                    621,
                    362,
                    812,
                    1468,
                    198,
                    198,
                    464,
                    1294,
                    1230,
                    319,
                    3217,
                    2318,
                    260,
                    992,
                    3812,
                    663,
                    717,
                    8325,
                    287,
                    517,
                    621,
                    1440,
                    812,
                ],
            ]
        ),
    }

    torch.manual_seed(1337)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    config = LeanGPTConfig(**json.loads(config_json))
    model = LeanGPTForPreTraining(config).to(device).train(True)
    train_batch = {key: value.to(device) for key, value in batch.items()}

    # overfit to a given batch
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for step in trange(num_steps):
        opt.zero_grad()
        output = model(**train_batch)
        output.loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"step={step}, loss={output.loss.item()}")

        if step == num_steps - 1:
            print(f"step={step}, loss={output.loss.item()}")
            assert output.loss.item() < loss_threshold, "Training loss is unexpectedly high"

    # compute reference outputs, loss and gradients
    model = model.cpu().train(False)
    model.zero_grad(set_to_none=True)
    output = model(**batch)
    output.loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Param {name} has no grad"

    test_data = dict(
        config=config_json,
        state=model.state_dict(),
        batch=batch,
        logits=output.logits,
        loss=output.loss,
        grads={name: param.grad.data.clone() for name, param in model.named_parameters()},
    )
    torch.save(test_data, filename)
