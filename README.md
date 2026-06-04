🌳 Task 7: Decision Tree – Titanic Survival Prediction

Edutech Solution | Data Science Internship


📌 Objective
Build a Decision Tree classification model on the Titanic dataset to predict passenger survival, visualize the tree structure, and evaluate model performance using accuracy and precision.

🛠️ Tools & Libraries
ToolPurposePython 3Programming languageScikit-learnModel building & evaluationPandasData manipulationMatplotlibTree visualizationSeabornDataset loading & plots

📂 Dataset

Name: Titanic Dataset
Source: Built-in via seaborn.load_dataset('titanic')
Size: 891 rows × 15 columns
Target: survived (0 = Not Survived, 1 = Survived)

Features Used
FeatureDescriptionpclassPassenger class (1st, 2nd, 3rd)sexGender (encoded: female=0, male=1)ageAge in yearssibspSiblings/spouses aboardparchParents/children aboardfareTicket fareembarkedPort of embarkation (C=0, Q=1, S=2)

⚙️ Steps Performed

Loaded the Titanic dataset using seaborn
Explored data — checked shape, missing values, class distribution
Preprocessed — filled nulls with median/mode, label-encoded categorical columns
Split data into 80% training / 20% testing
Trained a DecisionTreeClassifier with Gini criterion and max_depth=4
Evaluated the model using accuracy, precision, recall, and F1-score
Visualized the tree structure, confusion matrix, and feature importances


📈 Model Results
MetricScoreAccuracy79.89%Precision83.93%Recall63.51%F1-Score72.31%
Classification Report
              precision    recall  f1-score   support

Not Survived       0.78      0.91      0.84       105
    Survived       0.84      0.64      0.72        74

    accuracy                           0.80       179



🧠 Interview Q&A
Q1: What is Gini Impurity?
Gini Impurity measures how often a randomly chosen element from a set would be incorrectly labelled if it was randomly labelled according to the distribution of labels.
Formula:
Gini = 1 - Σ(pᵢ²)
Gini ValueMeaning0.0Perfectly pure node (one class only)0.5Maximally impure (50/50 split)
The Decision Tree algorithm picks the split that minimizes Gini Impurity at each node.

Q2: What is Pruning in Decision Trees?
Pruning is a technique to reduce overfitting by simplifying the tree structure.
Two types:
TypeDescriptionScikit-learn ParameterPre-PruningStop growing earlymax_depth, min_samples_split, min_samples_leafPost-PruningGrow full tree, then cutccp_alpha (cost-complexity pruning)
In this project, pre-pruning was applied using max_depth=4 and min_samples_split=10.

📁 Repository Structure
task-7-decision-tree/
│
├── decision_tree_titanic.py   # Main Python script
├── tree_plot.png              # Decision Tree visualization
├── confusion_matrix.png       # Confusion matrix heatmap
├── feature_importance.png     # Feature importance bar chart
└── README.md                  # Project documentation

🚀 How to Run
bash# 1. Clone the repository
git clone https://github.com/your-username/task-7-decision-tree.git
cd task-7-decision-tree

# 2. Install dependencies
pip install scikit-learn pandas matplotlib seaborn

# 3. Run the script
python decision_tree_titanic.py

