import os

# KAGGLE PATHS
BASE_DIR = '/kaggle/working'
INPUT_DIR = '/kaggle/input/ds201-deep-learning'

DATA_PATHS = {
    'kb': os.path.join(INPUT_DIR, '6000_sample.json'),  # Hoặc 'knowledge_base.json'
    'original_entities': os.path.join(INPUT_DIR, 'medical_entities_dict.json'),
    'entities_full': os.path.join(INPUT_DIR, 'medical_entities_full.json'),
}

OUTPUT_PATHS = {
    'distant_supervision': os.path.join(BASE_DIR, 'data', 'training_data.json'),
    'train_split': os.path.join(BASE_DIR, 'data', 'train.json'),
    'val_split': os.path.join(BASE_DIR, 'data', 'val.json'),
    'test_split': os.path.join(BASE_DIR, 'data', 'test.json'),
    'model_dir': os.path.join(BASE_DIR, 'models', 'vihealthbert_re'),
    'extracted_relations': os.path.join(BASE_DIR, 'output', 'relations.json'),
    'relation_stats': os.path.join(BASE_DIR, 'output', 'statistics.json'),
}

RELATION_SCHEMA = {
    'TREATED_BY': {
        'description': 'Bệnh/Triệu chứng được điều trị bởi thuốc/phương pháp',
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
            ('ORGAN', 'DISEASESYMTOM'),
        ]
    },
    'DIAGNOSED_BY': {
        'description': 'Bệnh/Triệu chứng được chẩn đoán bằng',
        'type_pairs': [
            ('DISEASESYMTOM', 'DIAGNOSTICS'),
            ('DISEASESYMTOM', 'MEDDEVICETECHNIQUE'),
            ('ORGAN', 'DIAGNOSTICS'),
            ('ORGAN', 'MEDDEVICETECHNIQUE'),
        ]
    },
    'AFFECTS': {
        'description': 'Bệnh/Thuốc/Phương pháp ảnh hưởng/tác động đến cơ quan',
        'type_pairs': [
            ('DISEASESYMTOM', 'ORGAN'),
            ('DRUGCHEMICAL', 'ORGAN'),
            ('TREATMENT', 'ORGAN'),
            ('SURGERY', 'ORGAN'),
        ]
    },
    'PREVENTED_BY': {
        'description': 'Bệnh được phòng ngừa bởi',
        'type_pairs': [
            ('DISEASESYMTOM', 'PREVENTIVEMED'),
            ('PREVENTIVEMED', 'ORGAN'),
        ]
    },
    'CAUSED_BY': {
        'description': 'Bệnh/Triệu chứng gây ra bởi',
        'type_pairs': [
            ('DISEASESYMTOM', 'DRUGCHEMICAL'), 
            ('DISEASESYMTOM', 'ORGAN'), 
            ('DISEASESYMTOM', 'DISEASESYMTOM'),
            ('DISEASESYMTOM', 'ORGAN'), 
            ('DRUGCHEMICAL', 'DISEASESYMTOM'),
        ]
    },
    'LOCATED_IN': {
        'description': 'Thực thể nằm trong cơ quan',
        'type_pairs': [
            ('DISEASESYMTOM', 'ORGAN'), 
            ('ORGAN', 'ORGAN'), 
            ('DRUGCHEMICAL', 'ORGAN'),
        ]
    },
    'HAS_SIDE_EFFECT': {
        'description': 'Thuốc/Phương pháp gây tác dụng phụ',
        'type_pairs': [
            ('DRUGCHEMICAL', 'DISEASESYMTOM'),
            ('TREATMENT', 'DISEASESYMTOM'),
            ('SURGERY', 'DISEASESYMTOM'),
        ]
    },
    'USED_FOR': {
        'description': 'Thuốc/Phương pháp được dùng cho mục đích/bệnh lý',
        'type_pairs': [
            ('DRUGCHEMICAL', 'DISEASESYMTOM'),
            ('TREATMENT', 'DISEASESYMTOM'),
            ('SURGERY', 'DISEASESYMTOM'),
            ('PREVENTIVEMED', 'DISEASESYMTOM'), 
        ]
    },
    'RELIES_ON': {
        'description': 'Phương pháp/Chẩn đoán sử dụng thiết bị/kỹ thuật',
        'type_pairs': [
            ('TREATMENT', 'MEDDEVICETECHNIQUE'),
            ('SURGERY', 'MEDDEVICETECHNIQUE'),
            ('DIAGNOSTICS', 'MEDDEVICETECHNIQUE'),
            ('MEDDEVICETECHNIQUE', 'MEDDEVICETECHNIQUE'),
        ]
    },
    'CONTRAINDICATED_FOR': {
        'description': 'Thuốc/Phương pháp chống chỉ định cho/trong bệnh lý',
        'type_pairs': [
            ('DRUGCHEMICAL', 'DISEASESYMTOM'),
            ('TREATMENT', 'DISEASESYMTOM'),
            ('SURGERY', 'DISEASESYMTOM'),
        ]
    },
}

RELATION_TYPES = list(RELATION_SCHEMA.keys())
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
