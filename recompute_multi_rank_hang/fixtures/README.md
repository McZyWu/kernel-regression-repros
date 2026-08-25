# Captured input fixtures

The four production-derived tensor captures are intentionally not committed.
Place them in this directory with the following names:

```text
gdn_recompute_call160_npu0_20260824.pt
gdn_recompute_call160_npu1_20260824.pt
gdn_recompute_call160_npu2_20260824.pt
gdn_recompute_call160_npu3_20260824.pt
```

Expected SHA256 values:

```text
3c3da2664b8c0456c6f65d9beeb8eaceb3f40698c4f62b8d109f6a951c846676  gdn_recompute_call160_npu0_20260824.pt
a7b3eb0d5a5d55b73a7324988ead7fbdf7fbedef66fa5b4fd90ddaf6d0fdca94  gdn_recompute_call160_npu1_20260824.pt
7784986441905abc5ead48b47d7a326dde4dd63a66d44669a8309e3cc4af2ef1  gdn_recompute_call160_npu2_20260824.pt
83fcca04f0aa34d11639cc4745d3e3842e2f07cd89df11fa095a4c5468480b8b  gdn_recompute_call160_npu3_20260824.pt
```

Each file is approximately 65 MB.  They are ignored by the repository's
`.gitignore`; pass an external absolute path with `--input` if they are stored
elsewhere.
