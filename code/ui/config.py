"""
Static configuration and field schema for the ICU intake UI.
"""

from typing import Any, Final


FieldSpec = dict[str, Any]
AGGREGATE_SUFFIXES: tuple[str, ...] = ("delta", "first", "last", "max", "mean", "median", "min", "missing", "range", "std")
PROCESSING_REPLY: Final[str] = "Captured processing signal. Assessing risk of acute adverse event based on provided information..."
FIELD_SPECS: list[FieldSpec] = [
    {"name": "age", "question": "Age (years)", "required": True, "type": "int", "min": 0, "max": 120, "direct_column": "Age", "median": 68.0},
    {"name": "bun", "question": "BUN (mg/dL)", "required": True, "type": "float", "min": 0, "feature_base": "BUN", "median": 19.75},
    {"name": "weight", "question": "Weight (kg)", "required": True, "type": "float", "min": 0, "direct_column": "Weight", "median": 80.0},
    {"name": "creatinine", "question": "Creatinine (mg/dL)", "required": True, "type": "float", "min": 0, "feature_base": "Creatinine", "median": 0.95},
    {"name": "gcs", "question": "GCS score", "required": True, "type": "int", "min": 3, "max": 15, "feature_base": "GCS", "median": 12.087121212121213},
    {"name": "height", "question": "Height (cm)", "required": False, "type": "float", "min": 0, "direct_column": "Height", "median": 170.2},
    {"name": "gender", "question": "Gender", "required": False, "type": "choice", "choices": ["male", "female", "other"], "direct_column": "Gender", "median": 1.0},
    {"name": "icu_type", "question": "ICUType", "required": False, "type": "choice", "choices": [("1: Coronary Care Unit", "1"), ("2: Cardiac Surgery Recovery Unit", "2"), ("3: Medical ICU", "3"), ("4: Surgical ICU", "4")], "median": 3.0, "onehot_defaults": {"ICUType_1.0": 0.0, "ICUType_2.0": 0.0, "ICUType_3.0": 0.0, "ICUType_4.0": 0.0, "ICUType_nan": 0.0}},
    {"name": "albumin", "question": "Albumin (g/dL)", "required": False, "type": "float", "feature_base": "Albumin", "median": 2.9},
    {"name": "alp", "question": "ALP (IU/L)", "required": False, "type": "float", "feature_base": "ALP", "median": 76.0},
    {"name": "alt", "question": "ALT (IU/L)", "required": False, "type": "float", "feature_base": "ALT", "median": 31.5},
    {"name": "ast", "question": "AST (IU/L)", "required": False, "type": "float", "feature_base": "AST", "median": 49.0},
    {"name": "bilirubin", "question": "Bilirubin (mg/dL)", "required": False, "type": "float", "feature_base": "Bilirubin", "median": 0.7},
    {"name": "cholesterol", "question": "Cholesterol (mg/dL)", "required": False, "type": "float", "feature_base": "Cholesterol", "median": 151.0},
    {"name": "diasabp", "question": "DiasABP (mmHg)", "required": False, "type": "float", "feature_base": "DiasABP", "median": 58.0},
    {"name": "fio2", "question": "FiO2", "required": False, "type": "float", "min": 0, "max": 1, "feature_base": "FiO2", "median": 0.525},
    {"name": "glucose", "question": "Glucose (mg/dL)", "required": False, "type": "float", "feature_base": "Glucose", "median": 130.0},
    {"name": "hco3", "question": "HCO3 (mmol/L)", "required": True, "type": "float", "feature_base": "HCO3", "median": 23.5},
    {"name": "hct", "question": "HCT (%)", "required": False, "type": "float", "feature_base": "HCT", "median": 30.55},
    {"name": "hr", "question": "HR (bpm)", "required": False, "type": "float", "feature_base": "HR", "median": 86.127441244621},
    {"name": "k", "question": "Potassium K (mEq/L)", "required": False, "type": "float", "feature_base": "K", "median": 4.1},
    {"name": "lactate", "question": "Lactate (mmol/L)", "required": False, "type": "float", "feature_base": "Lactate", "median": 1.94},
    {"name": "mg", "question": "Magnesium Mg (mmol/L)", "required": False, "type": "float", "feature_base": "Mg", "median": 2.0},
    {"name": "map", "question": "MAP (mmHg)", "required": False, "type": "float", "feature_base": "MAP", "median": 78.0},
    {"name": "mechvent", "question": "MechVent", "required": False, "type": "choice", "choices": ["0", "1"], "feature_base": "MechVent", "median": 1.0},
    {"name": "na", "question": "Sodium Na (mEq/L)", "required": False, "type": "float", "feature_base": "Na", "median": 139.0},
    {"name": "nidiasabp", "question": "NIDiasABP (mmHg)", "required": True, "type": "float", "feature_base": "NIDiasABP", "median": 55.2962962962963},
    {"name": "nimap", "question": "NIMAP (mmHg)", "required": False, "type": "float", "feature_base": "NIMAP", "median": 73.85},
    {"name": "nisysabp", "question": "NISysABP (mmHg)", "required": False, "type": "float", "feature_base": "NISysABP", "median": 114.0},
    {"name": "paco2", "question": "PaCO2 (mmHg)", "required": False, "type": "float", "feature_base": "PaCO2", "median": 39.666666666666664},
    {"name": "pao2", "question": "PaO2 (mmHg)", "required": False, "type": "float", "feature_base": "PaO2", "median": 138.5},
    {"name": "ph", "question": "Arterial pH", "required": False, "type": "float", "feature_base": "pH", "median": 7.385},
    {"name": "platelets", "question": "Platelets (cells/nL)", "required": False, "type": "float", "feature_base": "Platelets", "median": 186.0},
    {"name": "resprate", "question": "RespRate (bpm)", "required": False, "type": "float", "feature_base": "RespRate", "median": 19.0},
    {"name": "sao2", "question": "SaO2 (%)", "required": False, "type": "float", "feature_base": "SaO2", "median": 97.2},
    {"name": "sysabp", "question": "SysABP (mmHg)", "required": False, "type": "float", "feature_base": "SysABP", "median": 116.0},
    {"name": "temp", "question": "Temp (C)", "required": True, "type": "float", "feature_base": "Temp", "median": 36.95555555555555},
    {"name": "tropi", "question": "TropI (ug/L)", "required": False, "type": "float", "feature_base": "TroponinI", "median": 2.225},
    {"name": "tropt", "question": "TropT (ug/L)", "required": False, "type": "float", "feature_base": "TroponinT", "median": 0.13333333333333333},
    {"name": "urine", "question": "Urine output (mL)", "required": True, "type": "float", "feature_base": "Urine", "median": 103.33806818181819},
    {"name": "wbc", "question": "WBC (cells/nL)", "required": False, "type": "float", "feature_base": "WBC", "median": 11.558333333333334},
]

REQUIRED_SPECS: list[FieldSpec] = [field for field in FIELD_SPECS if field["required"]]
OPTIONAL_SPECS: list[FieldSpec] = [field for field in FIELD_SPECS if not field["required"]]

REQUIRED_TAB_ID: Final[str] = "required_tab"
OPTIONAL_TAB_ID: Final[str] = "optional_tab"
