# Checkpoints

Large model files are not copied into this source directory.

The training command needs:

1. The MedSAM ViT-B base checkpoint, `medsam_vit_b.pth`.
2. The MPI-tuned MedSAM-LoRA checkpoint, `best_model.pth`.

Inference needs the MedSAM ViT-B base checkpoint and a complete T4FNet fold checkpoint. The complete T4FNet checkpoint already contains the frozen MedSAM-LoRA state.

Use command-line arguments to provide the file paths. Do not commit third-party or experimental weights before checking their distribution terms.
