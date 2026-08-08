"""Create minimal test data for A2S2D testing

Creates a small synthetic dataset mimicking CGSS structure to test the pipeline
without requiring the full CGSS2023 data file.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Create minimal test data
np.random.seed(2024)
n = 100

data = pd.DataFrame({
    'id': range(1, n + 1),
    'a2': np.random.choice([1, 2], n),  # Gender (1=男, 2=女)
    'a3a': np.random.randint(18, 80, n),  # Age
    'a4': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n),  # Ethnicity
    'a7a': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], n),  # Education
    'a69': np.random.choice([1, 2, 3, 4, 5], n),  # Marital status
    'isurban': np.random.choice([0, 1], n),  # Urban/rural
    'weight2': np.random.uniform(0.5, 2.0, n),  # Survey weight

    # Some outcome variables
    'a15': np.random.choice([1, 2, 3, 4, 5], n),  # Happiness
    'a53': np.random.randint(1, 10, n),  # Trust score
    'a62': np.random.choice([1, 2, 3, 4, 5], n),  # Satisfaction

    # Some explanatory variables
    'a8a': np.random.randint(0, 100000, n),  # Income
    'a1': np.random.choice([1, 2, 3, 4, 5], n),  # Health status
})

# Create minimal codebook
codebook = pd.DataFrame({
    'variable': data.columns,
    'label': [
        'Person ID',
        'Gender (1=Male, 2=Female)',
        'Age in years',
        'Ethnicity',
        'Education level',
        'Marital status',
        'Urban/rural (0=Rural, 1=Urban)',
        'Survey weight',
        'Happiness level',
        'Trust score',
        'Life satisfaction',
        'Annual income',
        'Health status',
    ],
    'question': [
        'What is your ID number?',
        'What is your gender?',
        'What is your age?',
        'What is your ethnicity?',
        'What is your highest education level?',
        'What is your marital status?',
        'Do you live in urban area?',
        'Survey weight for representativeness',
        'How happy are you?',
        'How much do you trust others?',
        'How satisfied are you with your life?',
        'What is your annual income?',
        'How is your health status?',
    ],
    'value_labels': [
        '',
        '1:Male, 2:Female',
        '',
        '1:Han, 2:Mongolian, 3:Hui, 4:Tibetan, 5:Uygur, 6:Miao, 7:Yi, 8:Other',
        '1:No schooling, 2:Primary, 3:Junior high, 4:Senior high, 5:Vocational, 6:Associate, 7:Bachelor, 8:Master+',
        '1:Never married, 2:Married, 3:Widowed, 4:Divorced, 5:Remarried',
        '0:Rural, 1:Urban',
        '',
        '1:Very unhappy, 2:Unhappy, 3:Neutral, 4:Happy, 5:Very happy',
        '1-10 scale',
        '1:Very dissatisfied, 2:Dissatisfied, 3:Neutral, 4:Satisfied, 5:Very satisfied',
        '',
        '1:Very poor, 2:Poor, 3:Fair, 4:Good, 5:Very good',
    ],
    'module': ['demographics'] * 8 + ['attitudes'] * 5,
    'level': ['person'] * 13,
    'missing_values': [''] * 13,
})

# Save test data
output_dir = Path('playground/a2s2d/data')
output_dir.mkdir(parents=True, exist_ok=True)

data_path = output_dir / 'CGSS2023_test.dta'
codebook_path = output_dir / 'CGSS2023编码表_test.xlsx'

data.to_stata(data_path, write_index=False)
codebook.to_excel(codebook_path, index=False)

print(f"✅ Created test data: {data_path}")
print(f"✅ Created test codebook: {codebook_path}")
print(f"   - Variables: {len(data.columns)}")
print(f"   - Rows: {len(data)}")
print("\nUpdate config.yaml:")
print(f"  inputs:")
print(f"    data: '{data_path}'")
print(f"    codebook: '{codebook_path}'")