---
title: KG-Enhanced ViT5 QA
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# KG-Enhanced ViT5 for Vietnamese Question Answering

Vietnamese Question Answering model with optional Knowledge Graph enhancement.

## Model Architecture

### Base Model
- **Pre-trained Model**: `VietAI/vit5-base` (T5ForConditionalGeneration)
- **Architecture**: Encoder-Decoder Transformer
- **Tokenizer**: ViT5 SentencePiece tokenizer

### Model Specifications
- **d_model**: 768
- **num_heads**: 12
- **num_encoder_layers**: 12
- **num_decoder_layers**: 12
- **vocab_size**: 32,000

### Knowledge Graph Enhancement (Optional)
When `use_kg=True`:
- **GNN Encoder**: GCN/GAT/GraphSAGE
  - Input: KG node features (300-dim embeddings)
  - Hidden: 256
  - Output: 768 (matches T5 d_model)
  - Layers: 2
- **Cross-Attention**: Multi-head attention to fuse KG embeddings into encoder output
- **Projection Layer**: Linear layer to match T5 hidden size

## Usage

1. Enter a question in Vietnamese
2. (Optional) Provide context or relevant information
3. Adjust generation parameters in the sidebar
4. Click "Generate Answer"

## Generation Parameters

- **Max Length**: Maximum length of generated answer (20-100 tokens)
- **Repetition Penalty**: Penalty for repeating tokens (1.0-2.0)
- **Temperature**: Sampling temperature (0.1-2.0)
- **Beam Size**: Number of beams for beam search (1-5)

## Model Files

To use a trained model, place `model_state_dict.pt` in the root directory of this space.

The checkpoint should contain:
- `model_state_dict`: Model weights (required)
- Or the full checkpoint dictionary with `model_state_dict` key

## Dependencies

See `requirements.txt` for full list of dependencies.

## Citation

If you use this model, please cite:
- ViT5: [VietAI/vit5-base](https://huggingface.co/VietAI/vit5-base)
- T5: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)

