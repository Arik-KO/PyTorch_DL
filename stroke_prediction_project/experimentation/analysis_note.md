# Stroke Data Analysis

---
- There are total 10 columns with training examples: 29065
- So the dataset features (n) are 9, excluding the ground truth and training examples (m) = 29065
- The shape of the dataframe will be (29065, 10)
- Stroke dataset has 5 object style columns: gender, ever_married, work_type, residence_type, smoking_status.
- From the object type column, it is evident that stroke column will be used for prediction ground truth.
- Residence_type column seems less connected to the stroke ground truth, so need to check on that
- The int64 columns are: hypertension and heart_disease and they represent yes or no values using 1 and o
- So, the object types are also needed to be converted to int64 type column using pd.get_dummies()
- Based on the unique strings each object columns have will determine the total number of features 

---

### Column Names
Index(['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status', 'stroke'],
      dtype='object')
---
### Processed Column Names
Index(['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
       'stroke', 'gender_Female', 'gender_Male', 'ever_married_No',
       'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked',
       'work_type_Private', 'work_type_Self-employed', 'work_type_children',
       'Residence_type_Rural', 'Residence_type_Urban',
       'smoking_status_formerly smoked', 'smoking_status_never smoked',
       'smoking_status_smokes'],
      dtype='object')
---
- No missing values are there

#### Now the ground truth will be stroke column. For the remaining columns need to create the correlation matrix and observe the heatmap.