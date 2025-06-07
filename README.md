**Objective**
The objective of this analysis is to develop a machine learning model that accurately predicts customer attrition (churn) using demographic and behavioural data. By leveraging exploratory data analysis, unsupervised learning, and supervised classification models, the goal is to identify high-risk customers which in turn can inform targeted retention strategies.

**EDA Results**
Kmeans was performed on preprocessed data and was later visualized using PCA to understand general cluster separation. The clusters appeared fairly well separated in the 2D PCA space, with visible boundaries. However, some overlap is present, indicating that boundaries between some clusters aren’t perfectly linear in the original space. This suggests that models that can capture interactions and nonlinearities (e.g., Random Forest, XGBoost) may perform better than purely linear models like Logistic Regression.

Cluster descriptions based on demographic and behavioural data:

Cluster 0: low attrition, predominantly male, slight tilt towards higher income, younger than other clusters, high utlization ratio and revolving balance
Cluster 1: low attrition, mostly female, largely low income, longer bank relationship, low utilization
Cluster 2: high attrition rate, almost equal gender distribution, low credit limit and transaction amount but high utilization ratio and revolving balance
Cluster 3: moderate attrition, mostly male, well-distributed income, high credit limit and number of transactions but low utilization ratio

**Model Building and Results**
Model choice was guided by insights from the EDA and clustering analysis. The KMeans clustering revealed meaningful
customer segments with distinct patterns in transaction behavior, credit limits, and utilization ratios. For example,
Cluster 2 had a high churn rate and showed nonlinear characteristics such as high utilization with low credit limits
and transaction amounts. These patterns indicated that a linear model alone might not be sufficient, prompting the
use of tree-based models like Random Forest and gradient boosting (XGBoost) to capture complex interactions and
non-linear decision boundaries.

Logistic Regression was selected as a baseline due to its interpretability and simplicity. To handle class imbalance, I used class weight=’balanced’ and tuned the regularization parameter C using GridSearchCV with ROC-AUC as the scoring metric. Random Forest was chosen for its ability to model complex, non-linear relationships and capture feature interactions. Key hyperparameters such as n estimators, max depth, and min samples split were tuned using GridSearchCV with 5-fold cross-validation. XGBoost was selected for its robust performance on structured/tabular data and its native support for handling class imbalance via the scale pos weight parameter and help focus more on minority (Attrited Customers) with oversampling or undersampling. I tuned n estimators, max depth, learning rate, and scale pos weight using GridSearchCV. All models were validated using ROC-AUC as the primary metric, with precision, recall, F1-score, and confusion matrices to evaluate performance on the imbalanced test set.

Logistic regression is a good baseline model having high recall but has many false positive which not good for churn prediction as it would lead to over-alerting making it costly. Random Forest does have an improvement compared to Logistic regression but it misses more churners which is not ideal for our objective.

The best performing model was XGBoost:
ROC-AUC: 0.993
Recall: 0.86
F-1 Score: 0.89

It has a high recall and precision and its low false negative rate means it minimizes the cost of missing at-risk customers. Thus, making XGBoost the best model for churn prediction.

The feature importance plot from the XGBoost model reinforces several key findings from the earlier EDA and clustering analysis. Notably, Total Trans Amt and Total Trans Ct, which were prominent in distinguishing customer clusters with lower attrition rates, emerged as the most influential predictors in the model. These features capture the volume and frequency of customer activity, both strong indicators of engagement. Additionally, Total Relationship Count and Total Revolving Bal which were also observed to vary across clusters with differing churn levels further validate that customer-product engagement and usage patterns are central to predicting attrition. This alignment between EDA and model insights strengthens the interpretability and reliability of the final model.
