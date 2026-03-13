import pandas as pd
import numpy as np

np.random.seed(42)

# Generate synthetic data for 1000 students
n_students = 1000

data = {
    'student_id': range(1, n_students + 1),
    'age': np.random.randint(18, 25, size=n_students),
    'gender': np.random.choice(['Male', 'Female', 'Other'], size=n_students, p=[0.48, 0.48, 0.04]),
    'socioeconomic_status': np.random.choice(['Low', 'Medium', 'High'], size=n_students, p=[0.3, 0.5, 0.2]),
    'commute_distance': np.random.uniform(1, 50, size=n_students), # in km
    'attendance_rate': np.random.uniform(50, 100, size=n_students), # percentage
    'previous_grades': np.random.uniform(40, 100, size=n_students), # percentage
    'current_gpa': np.random.uniform(1.0, 4.0, size=n_students), # 4.0 scale
    'financial_aid': np.random.choice([0, 1], size=n_students, p=[0.6, 0.4]),
    'part_time_job': np.random.choice([0, 1], size=n_students, p=[0.7, 0.3]),
    'extracurricular_activities': np.random.choice([0, 1], size=n_students, p=[0.5, 0.5]),
}

df = pd.DataFrame(data)

# Introduce some correlation to define dropout risk
# Lower attendance, lower GPA, no financial aid, part-time job increase risk
risk_score = (
    (100 - df['attendance_rate']) * 0.3 +
    (4.0 - df['current_gpa']) * 15 +
    (df['financial_aid'] == 0) * 10 +
    (df['part_time_job'] == 1) * 10 +
    (df['socioeconomic_status'] == 'Low') * 5 +
    np.random.normal(0, 5, n_students) # Noise
)

# Normalize risk score to probability roughly
risk_prob = 1 / (1 + np.exp(-(risk_score - np.mean(risk_score)) / np.std(risk_score)))

# Assign binary dropout target based on probability threshold
df['next_semester_dropout'] = (risk_prob > 0.6).astype(int)

# Save to CSV
df.to_csv('student_data.csv', index=False)
print("student_data.csv generated successfully with 1000 records.")
