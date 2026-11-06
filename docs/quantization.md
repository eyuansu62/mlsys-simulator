# Quantized Weight and KV Storage (FP8 / FP4 / INT8 / INT4)

Version 1.15 models quantized storage as a **declared effective byte width**, not
as a quantization format parser. The user supplies the per-element bytes actually
resident in memory (including any block/scale metadata) and an explicit dequant
work model. The simulator never infers either from a framework configuration, and
it does not model quantization accuracy: results are analytical/synthetic
predictions of bytes and work, not statements about output quality.

## Declared byte widths

`strategy.model.weight_bytes`, `kv_bytes` and `activation_bytes` accept any
positive finite value up to `8.0` (bytes per element). The value is the source of
truth for all memory and traffic calculations; block/scale metadata is **not**
auto-modeled in v1 and must already be folded into the declared number.

Effective widths round up exactly: a storage calculation uses
`ceil(elements × width)`, so fractional widths never understate physical bytes
while exact 2/4/8-byte models keep their historical integer products unchanged.

Common effective widths and their derivations:

| Format | Raw bytes | Effective bytes/element | Derivation |
| --- | --- | --- | --- |
| FP8 E4M3 / E5M2 | 1.0 | 1.0 | W8A8 or W8A16; no per-tensor metadata in the macro model |
| INT8 | 1.0 | 1.0 | W8A16; per-tensor scales are negligible at macro scale |
| INT4, GPTQ/AWQ group 128 | 0.5 | 0.515625 | 4-bit weight + FP16 scale per 128: `0.5 + 2/128` |
| INT4, GPTQ/AWQ group 64 | 0.5 | 0.53125 | 4-bit weight + FP16 scale per 64: `0.5 + 2/64` |
| FP4 E2M1, NVFP4 | 0.5 | 0.5625 | 4-bit weight + FP8-E4M3 scale per 16: `0.5 + 1/16` |
| FP4 E2M1, MXFP4 | 0.5 | 0.53125 | 4-bit weight + E8M0 scale per 32: `0.5 + 1/32` |
| FP8 KV cache | 1.0 | 1.0 | Mainstream KV quantization; no accuracy modeled |
| INT8 / INT4 / FP4 KV | 1.0 / 0.5 / 0.5 | as declared | KV dequant scales with the declared bytes only |

## Dtype labels

Labels remain advisory and must agree with the bytes when set:

| Label | Bytes |
| --- | --- |
| `float16`, `bfloat16` | 2.0 |
| `float8_e4m3`, `float8_e5m2`, `int8` | 1.0 |
| `int4`, `float4_e2m1` | 0.5 |
| `float32` / `float64` | 4.0 / 8.0 |

Any other value is `auto`. An explicit label that disagrees with the declared
bytes is a configuration error, and an effective width that includes block
metadata (for example `0.515625`) does not match a raw label — declare the bytes
and leave the label `auto`.

## Dequant work model

`strategy.model.weight_dequant` selects how packed weights are consumed:

| Mode | Model |
| --- | --- |
| `none` | Weights are read at `weight_bytes` once per GEMM (the default, unquantized contract) |
| `fused` | Packed weights are read once, traffic is unchanged, and the conversion is charged to the GEMM at `dequant_flops_per_weight` FLOPs per weight element |
| `materialized` | A separate pass reads the packed storage and writes expanded tensors, then the GEMM reads them: `params × (weight_bytes + 2 × activation_bytes)` total traffic and one extra kernel launch per layer |

A non-`none` mode is only valid for packed storage: `weight_bytes` must be
strictly below `activation_bytes`.

KV dequant is always modeled fused on the fly — every loaded KV element costs
`dequant_flops_per_weight` FLOPs inside the attention macro — and no expanded KV
storage is materialized.

`calibration.dequant_flops_per_weight` (default `1.0`) is the uncalibrated
first-order rate applied to both weight and KV dequant FLOPs. Calibrate it per
format/hardware with measured or reference evidence before treating the numbers
as accurate.

## Example

```json
{
  "strategy": {
    "model": {
      "name": "Qwen3-1.7B-GPTQ128",
      "revision": "gptq-128-fp16-scale",
      "weight_bytes": 0.515625,
      "weight_dequant": "fused",
      "kv_bytes": 1.0,
      "kv_dtype": "int8"
    }
  },
  "calibration": {"dequant_flops_per_weight": 1.0}
}
```

The FP8 weights fit where the BF16 copy does not; the fused dequant cost shows up
as extra FLOPs on the linear macros, and the 1-byte KV cache halves attention KV
traffic with on-the-fly dequant FLOPs added to attention.

## Profile identity and compatibility

`weight_dequant` participates in the Profile identity. Quantized experiments
therefore cannot silently reuse non-quantized Profiles, and a
`none`/`fused`/`materialized` change invalidates an otherwise identical Profile
(`CALIBRATION_MISMATCH`). Profile files written before 1.15 omit the field; they
load unchanged and still match the non-quantized experiments they were captured
for.

## Boundaries

- The Qwen3 importer still rejects `quantization_config`; the error now points
  here for the explicit declaration path.
- Live vLLM capture still rejects quantized models; all live-validation evidence
  in this repository remains unquantized.
- Byte calculations replaced the historical `int()` truncation with exact
  ceiling arithmetic; in the 2-byte domain the results are bit-for-bit
  unchanged, which the regression suite locks.
- This is an analytical/synthetic storage model. It makes no claim about
  quantization error, calibration data, or the accuracy of any specific FP8/INT8/
  INT4/FP4 kernel.
