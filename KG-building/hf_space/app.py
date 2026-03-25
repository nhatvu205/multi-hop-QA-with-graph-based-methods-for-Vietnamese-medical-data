"""
Streamlit app for KG-Enhanced ViT5 Question Answering
HuggingFace Space deployment
"""

import streamlit as st
import torch
from transformers import T5Tokenizer
import sys
import os
import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
import re

# Add current directory to path to import model modules (for local testing if needed)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# If you also copy `modeling_vit5_kg.py` into this folder, you can import the class directly.
# For HuggingFace Space, we will load the model from the Hub repo instead.
try:
    from modeling_vit5_kg import KGEnhancedViT5  # optional local override
except ImportError:
    KGEnhancedViT5 = None

# Page config
st.set_page_config(
    page_title="Hệ thống trả lời câu hỏi y tế tiếng Việt",
    layout="wide"
)

# Title
st.title("Hệ thống trả lời câu hỏi y tế tiếng Việt")
st.markdown("""
*Mô hình QA đa bước y tế tiếng Việt dựa trên mô hình Transformer kết hợp Knowledge Graph Embedding**

Mô hình sử dụng pretrained ViT5, huấn luyện trên hơn 6000 mẫu QA đa bước kết hợp cơ chế encoding bằng GNN và cross-attention của Transformer.
""")

# Sidebar for model configuration
with st.sidebar:
    st.header("Cấu hình mô hình")
    
    # Model selection (will be auto-detected from checkpoint if available)
    use_kg = st.checkbox("Sử dụng KGE để tăng cường?", value=True, help="Cải thiện khả năng trả lơi của mô hình bằng đồ thị")
    
    # Generation parameters (matching training config)
    st.subheader("Tham số tùy chỉnh")
    st.caption("⚠️ Khuyến nghị: Giữ nguyên giá trị mặc định để có kết quả tốt nhất")
    
    max_length = st.slider(
        "Max Length (độ dài tối đa câu trả lời)", 
        min_value=20, max_value=128, value=50, step=5,
        help="Độ dài tối đa của câu trả lời (tokens). Mặc định: 50 (khớp với training/evaluation)"
    )
    repetition_penalty = st.slider(
        "Repetition Penalty", 
        min_value=1.0, max_value=2.0, value=1.2, step=0.1,
        help="Hệ số phạt lặp lại từ. Mặc định: 1.2 (khớp với training). Giá trị cao hơn = ít lặp hơn"
    )
    temperature = st.slider(
        "Temperature", 
        min_value=0.1, max_value=2.0, value=1.0, step=0.1,
        help="Nhiệt độ sampling. 1.0 = greedy (khuyến nghị), >1.0 = random. Mặc định: 1.0 (khớp với training)"
    )
    num_beams = st.slider(
        "Beam Size", 
        min_value=1, max_value=5, value=1, step=1,
        help="Số lượng beams cho beam search. 1 = greedy decoding (khuyến nghị). Mặc định: 1 (khớp với training)"
    )
    
    # Input length settings (matching training config)
    st.markdown("---")
    st.subheader("Tham số Input")
    max_seq_len = st.slider(
        "Max Sequence Length (input)", 
        min_value=128, max_value=512, value=384, step=64,
        help="Độ dài tối đa của input (question + context). Mặc định: 384 (khớp với training config.max_seq_len)"
    )
    
    st.markdown("---")
    st.markdown("### Thông tin mô hình sử dụng")
    st.info("""
    **Base Model**: VietAI/vit5-base
    
    **Kiến trúc**:
    - Encoder-Decoder Transformer (T5)
    - Optional GNN (GCN/GAT/GraphSAGE) for KG
    - Cross-Attention fusion
    
    **Tokenizer**: ViT5 SentencePiece
    """)

# Initialize session state
HF_MODEL_ID = "nhatvu205/vit5_kg_medqa"


@st.cache_resource
def load_model(use_kg_param=True):
    """Load model and tokenizer from HuggingFace Hub (nhatvu205/vit5_kg_medqa)
    
    Args:
        use_kg_param: Kept for backward compatibility (not used to change remote weights)
    
    Returns:
        model, tokenizer, device, actual_use_kg
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer directly from model repo (use T5Tokenizer for ViT5)
    tokenizer = T5Tokenizer.from_pretrained(HF_MODEL_ID)

    # Load config to get use_kg setting
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(HF_MODEL_ID)
    actual_use_kg = getattr(config, 'use_kg', use_kg_param)

    # Load model from Hub
    if KGEnhancedViT5 is not None:
        # Local class available (e.g. modeling_vit5_kg.py is in the repo)
        # from_pretrained will read config and initialize with correct architecture
        model = KGEnhancedViT5.from_pretrained(HF_MODEL_ID)
    else:
        # Fallback: trust remote code from the Hub (requires transformers>=4.36)
        from transformers import AutoModelForSeq2SeqLM

        model = AutoModelForSeq2SeqLM.from_pretrained(
            HF_MODEL_ID,
            trust_remote_code=True
        )

    model = model.to(device)
    model.eval()

    return model, tokenizer, device, actual_use_kg

@st.cache_data
def load_qa_data():
    """
    Load QA dataset for context retrieval
    Matches model_module/data_loader.py load_qa_dataset() logic
    
    Tries to load from:
    1. data/qa_data.csv (matches training config)
    2. qa_data.csv (fallback)
    """
    # Try paths in order (matching training config)
    possible_paths = [
        'data/qa_data.csv',  # Matches model_module/config.py: self.qa_dataset_path = 'data/qa_data.csv'
        'qa_data.csv',       # Fallback
        'data/6000_samples.csv',  # Alternative name
    ]
    
    qa_data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            qa_data_path = path
            break
    
    if qa_data_path is None:
        return None
    
    try:
        # Load CSV (matching model_module/data_loader.py)
        qa_df = pd.read_csv(qa_data_path)
        qa_samples = []
        
        # Process each row (matching model_module/data_loader.py logic)
        for idx, row in qa_df.iterrows():
            try:
                # Get question
                question = str(row['question']) if 'question' in row and pd.notna(row['question']) else ''
                
                # Get context_text (CRITICAL - matches training)
                # Training uses 'context_text' column (see model_module/data_loader.py line 95)
                context = ''
                if 'context_text' in row and pd.notna(row['context_text']):
                    context = str(row['context_text'])
                else:
                    # Fallback to other column names
                    for col_name in ['context', 'Context', 'CONTEXT']:
                        if col_name in row and pd.notna(row[col_name]):
                            context = str(row[col_name])
                            break
                
                # Skip if no context (matching training logic)
                if not context or len(context.strip()) == 0:
                    continue
                
                # Skip if question too short (matching training: len(question) < 10)
                if not question or len(question) < 10:
                    continue
                
                # Truncate long context (matching training: max 500 chars)
                context = context[:500]
                
                qa_samples.append({
                    'question': question,
                    'context_text': context
                })
            except Exception as e:
                # Skip problematic rows (matching training behavior)
                continue
        
        return qa_samples
    except Exception as e:
        st.sidebar.warning(f"Could not load QA data from {qa_data_path}: {e}")
        return None

def tokenize_vietnamese(text):
    """
    Simple tokenization for Vietnamese text
    Split by whitespace and punctuation
    """
    # Remove punctuation and split by whitespace
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]  # Filter out single character tokens

def retrieve_context(question, qa_samples, top_k=3):
    """
    Retrieve relevant context from QA dataset using BM25
    
    Args:
        question: User's question
        qa_samples: List of QA samples with 'question' and 'context'
        top_k: Number of top contexts to retrieve
    
    Returns:
        str: Retrieved context (concatenated top_k contexts)
    """
    if not qa_samples or len(qa_samples) == 0:
        return None
    
    try:
        # Tokenize all documents (combine question and context for better matching)
        tokenized_corpus = []
        for sample in qa_samples:
            # Combine question and context for better retrieval
            combined_text = sample['question'] + ' ' + sample['context_text'][:300]
            tokens = tokenize_vietnamese(combined_text)
            tokenized_corpus.append(tokens)
        
        # Initialize BM25
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Tokenize query
        query_tokens = tokenize_vietnamese(question)
        
        if not query_tokens:
            return None
        
        # Get BM25 scores
        scores = bm25.get_scores(query_tokens)
        
        # Get top_k most relevant
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Retrieve contexts
        retrieved_contexts = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with positive scores
                retrieved_contexts.append(qa_samples[idx]['context_text'])
        
        if retrieved_contexts:
            # Combine contexts
            combined_context = ' '.join(retrieved_contexts)
            # Limit total length
            if len(combined_context) > 500:
                combined_context = combined_context[:500]
            return combined_context
        else:
            return None
    except Exception as e:
        # Fallback to simple keyword matching
        question_words = set(tokenize_vietnamese(question))
        best_match = None
        best_score = 0
        
        for sample in qa_samples:
            context_words = set(tokenize_vietnamese(sample['context_text']))
            score = len(question_words & context_words)
            if score > best_score:
                best_score = score
                best_match = sample['context_text']
        
        return best_match if best_match else None

# Load model (use_kg will be auto-detected from checkpoint if available)
try:
    model, tokenizer, device, actual_use_kg = load_model(use_kg_param=use_kg)
    # Update use_kg to match actual model configuration
    if actual_use_kg != use_kg:
        st.sidebar.warning(f"⚠ Mô hình được huấn luyện với use_kg={actual_use_kg}. Checkbox đã được cập nhật để khớp.")
        use_kg = actual_use_kg
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    st.stop()

# Load QA data for context retrieval
# Matches model_module/data_loader.py: loads from data/qa_data.csv
qa_samples = load_qa_data()
if qa_samples:
    st.sidebar.success(f"✓ Đã tải {len(qa_samples)} mẫu QA cho việc rút trích ngữ cảnh")
else:
    st.sidebar.warning("⚠️ Không tìm thấy dữ liệu QA trong 'data/qa_data.csv' hoặc 'qa_data.csv'. Việc rút trích ngữ cảnh bị vô hiệu hóa.")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    
    question = st.text_area(
        "Câu hỏi",
        placeholder="Nhập câu hỏi tiếng Việt...",
        height=100,
        help="Nhập câu hỏi tiếng Việt"
    )
    
    context = st.text_area(
        "Ngữ cảnh - không bắt buộc",
        placeholder="Nhập ngữ cảnh hoặc thông tin liên quan (để trống để tự động tìm kiếm)...",
        height=150,
        help="Nhập ngữ cảnh hoặc để trống để tự động rút trích từ dữ liệu"
    )
    
    # Auto-retrieve context option
    auto_retrieve = st.checkbox(
        "Tự động rút trích ngữ cảnh từ dữ liệu (nếu ngữ cảnh không được cung cấp)",
        value=True,
        help="Tự động rút trích ngữ cảnh từ dữ liệu khi ngữ cảnh không được cung cấp"
    )
    
    generate_btn = st.button("Generate Answer", type="primary", use_container_width=True)

with col2:
    st.subheader("Mô hình trả lời")
    
    if generate_btn:
        if not question.strip():
            st.warning("Hãy nhập câu hỏi")
        else:
            with st.spinner("Đang trả lời..."):
                try:
                    # Handle empty context - retrieve from QA dataset
                    retrieved_context_info = None
                    if not context or len(context.strip()) == 0:
                        if auto_retrieve and qa_samples:
                            with st.spinner("Rút trích ngữ cảnh từ dữ liệu..."):
                                retrieved_context = retrieve_context(question, qa_samples, top_k=3)
                                if retrieved_context:
                                    context = retrieved_context
                                    retrieved_context_info = "✓ Đã có ngữ cảnh"
                                else:
                                    context = question  # Fallback to question
                                    retrieved_context_info = "⚠ Không tìm thấy ngữ cảnh phù hợp"
                        else:
                            context = question  # Fallback to question as context
                            retrieved_context_info = "ℹ Sử dụng câu hỏi như ngữ cảnh"
                    
                    # Show retrieved context info
                    if retrieved_context_info:
                        st.info(retrieved_context_info)
                    
                    # Tokenize question and context separately (matching training format)
                    # CRITICAL: Match exactly with model_module/dataset.py and model_module/inference.py
                    # In training: tokenizer.encode() without max_length/truncation, then pad in collate_fn
                    # Here we need to match that format exactly
                    
                    # Tokenize without truncation first (matching training)
                    question_tokens = tokenizer.encode(question, add_special_tokens=False)
                    context_tokens = tokenizer.encode(context, add_special_tokens=False)
                    
                    # Combine tokens (matching model.generate() behavior: torch.cat([question_ids, context_ids], dim=1))
                    combined_tokens = question_tokens + context_tokens
                    
                    # Truncate if total exceeds max_seq_len (matching training max_seq_len)
                    if len(combined_tokens) > max_seq_len:
                        # Prefer to keep more context if possible
                        if len(context_tokens) > len(question_tokens):
                            # Keep full question, truncate context
                            max_q_len = len(question_tokens)
                            max_c_len = max_seq_len - max_q_len
                            question_tokens = question_tokens[:max_q_len]
                            context_tokens = context_tokens[:max_c_len]
                            combined_tokens = question_tokens + context_tokens
                        else:
                            # Keep full context, truncate question
                            max_c_len = len(context_tokens)
                            max_q_len = max_seq_len - max_c_len
                            question_tokens = question_tokens[:max_q_len]
                            context_tokens = context_tokens[:max_c_len]
                            combined_tokens = question_tokens + context_tokens
                    
                    # Convert to tensor (matching training format: no padding here, model handles it)
                    # Shape: (1, seq_len) - batch_size=1
                    input_ids = torch.tensor([combined_tokens], dtype=torch.long).to(device)
                    
                    # Create attention mask (1 for real tokens, 0 for padding)
                    # Matching model.generate() behavior: attention_mask = (input_ids != pad_token_id).long()
                    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                    attention_mask = (input_ids != pad_token_id).long().to(device)
                    
                    # Generate using the same format as training/evaluation
                    # Model from HuggingFace has generate(input_ids, attention_mask, ...) interface
                    # Match parameters with model_module/inference.py defaults: max_len=50, repetition_penalty=1.2, temperature=1.0, num_beams=1
                    # Note: do_sample is automatically calculated in modeling_vit5_kg.py based on temperature and num_beams
                    with torch.no_grad():
                        generated_ids = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            kg_node_features=None,  # Set to None for transformer-only in demo
                            kg_edge_index=None,
                            max_len=max_length,  # Used by custom generate() method
                            max_length=max_length,  # Used by T5's generate() method
                            repetition_penalty=repetition_penalty,
                            temperature=temperature,
                            num_beams=num_beams
                            # do_sample is automatically set in modeling_vit5_kg.py: do_sample=(temperature != 1.0 or num_beams > 1)
                            # pad_token_id, eos_token_id, decoder_start_token_id are also set automatically in modeling_vit5_kg.py
                        )
                    
                    # Decode generated tokens (matching inference.py format exactly)
                    # CRITICAL: Match model_module/inference.py line 69: tokenizer.decode(generated_ids[0].cpu().numpy())
                    if isinstance(generated_ids, torch.Tensor):
                        generated_ids_np = generated_ids[0].cpu().numpy()
                    else:
                        # Handle numpy array or list
                        if isinstance(generated_ids, (list, tuple)):
                            generated_ids_np = generated_ids[0] if len(generated_ids) > 0 else generated_ids
                        else:
                            generated_ids_np = generated_ids[0] if len(generated_ids.shape) > 1 else generated_ids
                    
                    # Decode WITHOUT skip_special_tokens first (matching inference.py exactly)
                    # Then clean up manually if needed
                    answer = tokenizer.decode(generated_ids_np, skip_special_tokens=False)
                    
                    # Clean up: remove special tokens manually if they appear
                    # T5 uses pad_token_id=0, eos_token_id=1 typically
                    # Remove any remaining special token artifacts
                    answer = answer.strip()
                    
                    # Remove common artifacts from generation
                    # Sometimes model generates extra tokens that need cleaning
                    if answer.startswith('<pad>') or answer.startswith('<PAD>'):
                        answer = answer.replace('<pad>', '').replace('<PAD>', '').strip()
                    if answer.startswith('<eos>') or answer.startswith('<EOS>'):
                        answer = answer.replace('<eos>', '').replace('<EOS>', '').strip()
                    
                    # Final cleanup: remove any non-printable characters that might cause display issues
                    import string
                    answer = ''.join(char for char in answer if char.isprintable() or char in string.whitespace)
                    answer = answer.strip()
                    
                    # Clear GPU cache after generation (matching inference.py)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    st.success("Đã trả lời!")
                    st.write(answer)
                    
                    # Show context info
                    with st.expander("Thông tin ngữ cảnh"):
                        st.write(f"**Độ dài ngữ cảnh**: {len(context)} ký tự")
                        st.write(f"**Xem một phần ngữ cảnh**: {context[:200]}...")
                        if retrieved_context_info:
                            st.write(f"**Trạng thái**: {retrieved_context_info}")
                    
                    # Show token info
                    with st.expander("Thông tin token"):
                        st.write(f"**Token của câu hỏi**: {len(question_tokens)}")
                        st.write(f"**Token của ngữ cảnh**: {len(context_tokens)}")
                        st.write(f"**Tổng số token**: {len(combined_tokens)}")
                        st.write(f"**Token đã tạo**: {len(generated_ids[0])}")
                        st.write(f"**Độ dài tối đa của chuỗi (config)**: {max_seq_len}")
                        st.write(f"**Độ dài tối đa của chuỗi đã tạo**: {max_length}")
                        
                except Exception as e:
                    st.error(f"Lỗi khi trả lời: {e}")
                    st.exception(e)
    else:
        st.info("Hãy nhập câu hỏi và click 'Tạo câu trả lời' để bắt đầu.")

# Examples section
st.markdown("---")
st.subheader("Câu hỏi mẫu")

examples = [
    {
        "question": "Bệnh viêm gan B là gì?",
        "context": "Viêm gan B là một bệnh nhiễm trùng gan do virus viêm gan B (HBV) gây ra. Bệnh có thể lây truyền qua đường máu, quan hệ tình dục, hoặc từ mẹ sang con."
    },
    {
        "question": "Triệu chứng của bệnh tiểu đường là gì?",
        "context": "Bệnh tiểu đường có các triệu chứng như khát nước nhiều, đi tiểu thường xuyên, mệt mỏi, giảm cân không rõ nguyên nhân, và vết thương lâu lành."
    },
    {
        "question": "Cách phòng ngừa bệnh cảm cúm?",
        "context": "Để phòng ngừa cảm cúm, bạn nên tiêm vaccine cúm hàng năm, rửa tay thường xuyên, tránh tiếp xúc với người bị bệnh, và giữ gìn vệ sinh cá nhân."
    }
]

for i, example in enumerate(examples):
    with st.expander(f"Example {i+1}: {example['question']}"):
        st.write(f"**Question**: {example['question']}")
        st.write(f"**Context**: {example['context']}")
        if st.button(f"Use Example {i+1}", key=f"example_{i}"):
            st.session_state['question'] = example['question']
            st.session_state['context'] = example['context']
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>@ Authors - Hồ Huỳnh Thư Nhi - Lê Diễm Quỳnh Như - Vũ Đình Nhật
</div>
""", unsafe_allow_html=True)

