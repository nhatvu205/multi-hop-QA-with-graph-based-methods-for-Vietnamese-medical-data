"""Inference and demo functions for ViT5 Encoder-Decoder model"""

import torch


def generate_answer(model, question, context, entities, kg, tokenizer, device, 
                    max_len=50, use_kg=True, repetition_penalty=1.2, temperature=1.0, num_beams=1):
    """
    Generate answer for a single question using ViT5 model
    
    Args:
        model: Trained KGEnhancedViT5 model
        question: Question string
        context: Context string
        entities: List of entity names (used if use_kg=True)
        kg: MedicalKnowledgeGraph instance (required if use_kg=True)
        tokenizer: Tokenizer instance (ViHealthBERT)
        device: torch.device
        max_len: Maximum answer length
        use_kg: Whether to use KG enhancement
        repetition_penalty: Penalty for repeating tokens
        temperature: Sampling temperature
        num_beams: Beam search size (1 = greedy)
    
    Returns:
        str: Generated answer text
    """
    model.eval()
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Handle empty context - use question as context if context is empty
    if not context or len(context.strip()) == 0:
        context = question  # Fallback to question as context
    
    # Tokenize question and context
    question_ids = torch.tensor(tokenizer.encode(question), dtype=torch.long).unsqueeze(0).to(device)
    context_ids = torch.tensor(tokenizer.encode(context), dtype=torch.long).unsqueeze(0).to(device)
    
    # Get KG data (if use_kg=True)
    if use_kg and kg is not None and entities:
        node_features, edge_index, subgraph_entities = kg.get_pyg_data(entities=entities, emb_dim=300)
        # Ensure all tensors are on the correct device
        kg_node_features = node_features.unsqueeze(0).to(device)
        # edge_index is a tensor, convert to device and wrap in list for batch processing
        edge_index = edge_index.to(device)
        kg_edge_index = [edge_index]
    else:
        # Explicitly set to None when use_kg=False
        kg_node_features = None
        kg_edge_index = None
    
    # Generate using ViT5's generate method
    with torch.no_grad():
        generated_ids = model.generate(
            question_ids=question_ids,
            context_ids=context_ids,
            kg_node_features=kg_node_features,
            kg_edge_index=kg_edge_index,
            max_len=max_len,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            num_beams=num_beams
        )
    
    # Decode generated tokens using tokenizer
    # ViT5 generate includes decoder_start_token_id, so we decode directly
    generated_text = tokenizer.decode(generated_ids[0].cpu().numpy())
    
    # Clear GPU cache after generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return generated_text


def predict_batch(model, samples, kg, tokenizer, device, use_kg=True):
    """
    Predict answers for multiple samples
    
    Args:
        model: Trained model
        samples: List of sample dicts with 'question', 'context', 'entities'
        kg: KnowledgeGraph (required if use_kg=True)
        tokenizer: Tokenizer
        device: Device
        use_kg: Whether to use KG
    
    Returns:
        list: Predicted answers
    """
    predictions = []
    
    for sample in samples:
        pred = generate_answer(
            model=model,
            question=sample['question'],
            context=sample['context'],
            entities=sample.get('entities', []) if use_kg else [],
            kg=kg if use_kg else None,
            tokenizer=tokenizer,
            device=device,
            use_kg=use_kg
        )
        predictions.append(pred)
    
    return predictions


def interactive_qa(model, kg, tokenizer, device, use_kg=True):
    """
    Interactive QA system
    
    Args:
        model: Trained model
        kg: KnowledgeGraph (required if use_kg=True)
        tokenizer: Tokenizer
        device: Device
        use_kg: Whether to use KG
    """
    print('=' * 70)
    if use_kg:
        print('Interactive Multi-Hop Medical QA System (KG-Enhanced ViT5)')
    else:
        print('Interactive Medical QA System (ViT5-only)')
    print('=' * 70)
    print('\nEnter your question (or "quit" to exit)')
    if use_kg:
        print('Note: Make sure your question contains medical terms from the KG')
    print('-' * 70)
    
    while True:
        question = input('\nQuestion: ').strip()
        
        if question.lower() in ['quit', 'exit', 'q', '']:
            print('\nGoodbye!')
            break
        
        context = input('Context (optional): ').strip()
        if not context:
            context = question  # Use question as context if not provided
        
        # Find relevant entities (if use_kg=True)
        relevant_entities = []
        if use_kg and kg is not None:
            question_words = question.lower().split()
            for entity in kg.nodes:
                entity_words = entity.lower().split()
                if any(word in question_words for word in entity_words):
                    relevant_entities.append(entity)
            
            if not relevant_entities:
                print('⚠ Could not find relevant entities in the Knowledge Graph.')
                print('Will use Transformer-only mode for this question.')
            else:
                relevant_entities = relevant_entities[:10]
        
        # Predict
        try:
            answer = generate_answer(
                model=model,
                question=question,
                context=context,
                entities=relevant_entities if use_kg else [],
                kg=kg if use_kg else None,
                tokenizer=tokenizer,
                device=device,
                use_kg=use_kg and len(relevant_entities) > 0
            )
            
            print(f'\n✓ Answer: {answer}')
            if use_kg and relevant_entities:
                print(f'  (Based on entities: {", ".join(relevant_entities[:5])}{"..." if len(relevant_entities) > 5 else ""})')
        except Exception as e:
            print(f'⚠ Error during prediction: {str(e)}')
            import traceback
            traceback.print_exc()


def evaluate_samples(model, samples, kg, tokenizer, device, num_samples=5, use_kg=True):
    """
    Evaluate and print predictions for sample questions
    
    Args:
        model: Trained model
        samples: List of QA samples
        kg: KnowledgeGraph (required if use_kg=True)
        tokenizer: Tokenizer
        device: Device
        num_samples: Number of samples to evaluate
        use_kg: Whether to use KG
    """
    print('\n' + '=' * 70)
    print(f'Evaluating model on {num_samples} samples...')
    print('=' * 70)
    
    model.eval()
    
    for i in range(min(num_samples, len(samples))):
        sample = samples[i]
        
        predicted_answer = generate_answer(
            model=model,
            question=sample['question'],
            context=sample['context'],
            entities=sample.get('entities', []) if use_kg else [],
            kg=kg if use_kg else None,
            tokenizer=tokenizer,
            device=device,
            use_kg=use_kg
        )
        
        print(f'\n--- Sample {i+1} ---')
        print(f'Question: {sample["question"][:100]}...')
        if use_kg and 'entities' in sample:
            print(f'Entities: {", ".join(sample["entities"][:5])}{"..." if len(sample.get("entities", [])) > 5 else ""}')
        print(f'True Answer: {sample["answers"][0][:80]}...')
        print(f'Predicted: {predicted_answer[:80]}...')
    
    print('\n' + '=' * 70)
