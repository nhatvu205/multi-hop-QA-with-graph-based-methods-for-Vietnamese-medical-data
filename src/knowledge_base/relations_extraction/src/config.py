import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATHS = {
    'kb': os.path.join(BASE_DIR, 'kb_data', 'knowledge_base.json'),
    'normalized_entities': os.path.join(BASE_DIR, 'entity_data', 'normalized', 'normalized_entity_dict.json'),
    'original_entities': os.path.join(BASE_DIR, 'entity_data', 'original', 'medical_entities_dict.json'),
}

OUTPUT_PATHS = {
    'distant_supervision': os.path.join(BASE_DIR, 'relations_extraction', 'data', 'training_data.json'),
    'train_split': os.path.join(BASE_DIR, 'relations_extraction', 'data', 'train.json'),
    'val_split': os.path.join(BASE_DIR, 'relations_extraction', 'data', 'val.json'),
    'test_split': os.path.join(BASE_DIR, 'relations_extraction', 'data', 'test.json'),
    'model_dir': os.path.join(BASE_DIR, 'relations_extraction', 'models', 'vihealthbert_re'),
    'extracted_relations': os.path.join(BASE_DIR, 'relations_extraction', 'output', 'relations.json'),
    'relation_stats': os.path.join(BASE_DIR, 'relations_extraction', 'output', 'statistics.json'),
}

RELATION_SCHEMA = {
    'TREATED_BY': {
        'description': 'Bệnh được điều trị bởi thuốc/phương pháp',
        'type_pairs': [
            ('DISEASESYMTOM', 'DRUGCHEMICAL'),
            ('DISEASESYMTOM', 'TREATMENT'),
            ('DISEASESYMTOM', 'SURGERY'),
        ]
    },
    'HAS_SYMPTOM': {
        'description': 'Bệnh có triệu chứng',
        'type_pairs': [
            ('DISEASESYMTOM', 'DISEASESYMTOM'),
        ]
    },
    'DIAGNOSED_BY': {
        'description': 'Bệnh được chẩn đoán bằng',
        'type_pairs': [
            ('DISEASESYMTOM', 'DIAGNOSTICS'),
            ('DISEASESYMTOM', 'MEDDEVICETECHNIQUE'),
        ]
    },
    'AFFECTS': {
        'description': 'Bệnh ảnh hưởng đến cơ quan',
        'type_pairs': [
            ('DISEASESYMTOM', 'ORGAN'),
        ]
    },
    'PREVENTED_BY': {
        'description': 'Bệnh được phòng ngừa bởi',
        'type_pairs': [
            ('DISEASESYMTOM', 'PREVENTIVEMED'),
        ]
    },
    'CAUSED_BY': {
        'description': 'Bệnh gây ra bởi',
        'type_pairs': [
            ('DISEASESYMTOM', 'DRUGCHEMICAL'),
            ('DISEASESYMTOM', 'ORGAN'),
        ]
    },
    'LOCATED_IN': {
        'description': 'Thực thể nằm trong cơ quan',
        'type_pairs': [
            ('DISEASESYMTOM', 'ORGAN'),
            ('ORGAN', 'ORGAN'),
        ]
    },
    'HAS_SIDE_EFFECT': {
        'description': 'Thuốc gây tác dụng phụ',
        'type_pairs': [
            ('DRUGCHEMICAL', 'DISEASESYMTOM'),
            ('TREATMENT', 'DISEASESYMTOM'),
        ]
    },
}

RELATION_TYPES = list(RELATION_SCHEMA.keys()) + ['NO_RELATION']
RELATION2ID = {rel: idx for idx, rel in enumerate(RELATION_TYPES)}
ID2RELATION = {idx: rel for rel, idx in RELATION2ID.items()}

MODEL_CONFIG = {
    'model_name': 'demdecuong/vihealthbert-base-word',
    'max_length': 256,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'warmup_steps': 500,
    'weight_decay': 0.01,
}

DISTANT_SUPERVISION_CONFIG = {
    'max_distance': 100,
    'min_confidence': 0.5,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
}

