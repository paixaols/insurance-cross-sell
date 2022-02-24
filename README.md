# insurance-cross-sell	

This project uses Learning to Rank techniques to build a cross-sell strategy for an insurance company.

# 1. Business Problem.

A health insurance company is about to release a new product, an auto insurance. Thus, this company needs to know which of their customers are prone to buy this new product. Based on a survey conducted with part of the company’s client base, a cross-sell strategy was developed. The data used here is available at [kaggle](https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction).

# 2. Solution Strategy
A learning to rank algorithm was used to sort the clients based on their propensity to buy the new insurance offered by the company. The solution was built based on the following steps:

**Step 01. Data description:** Identify relevant data available from the business.

**Step 02. Feature engineering:** Convert and derive new attributes based on the original data to better describe the phenomenon to be modeled.

**Step 03. Data filtering:** Filter data that may not be relevant or available during production.

**Step 04. Exploratory data analysis:** Explore the data to gain insights that may be relevant later during machine learning modeling.

**Step 05. Data preparation:** Prepare the data before applying machine learning models. Given the unbalanced nature of the dataset (12% of the clients were interested in the insurance), a Random Undersampling was performed to balance both classes.

**Step 06. Feature selection:** Select the most relevant features for training the model.

**Step 07. Machine learning modelling:** Train machine learning models.

**Step 08. Hyperparameter fine tunning:** Fine tune the model parameters.

**Step 09. Translate model performance to business intelligence:** Convert the performance of the model to a business result 

**Step 10. Deploy model to production:** Deploy the model in a cloud environment, making it accessible to others.

# 3. Top 3 Data Insights

* Clients previously involved in a car injury are more inclined to purchase the auto insurance.
* Younger people are less inclined to purchase the auto insurance.
* Clients who already an auto insurance are not prone to change their insurance company.

# 4. Machine Learning Model Applied

The following models were trained (all were cross-validated):

* Random Forest Classifier
* Extra Trees Classifier
* K Nearest Neighbors Classifier
* Logistic Regression
* XGBoost Classifier

# 5. Machine Learning Model Performance

The XGBoost Classifier presented the best generalization performance, it was, therefore, chosen as the final model. The graphs below show the cumulative gain curve and the lift curve.

<img src="/reports/figures/gain_lift.png" width="900">

# 6. Business Results

The company’s sales team need to contact those clients more prone to purchase the new insurance in order to make an effective use of resources. Then, a hypothetical situation was imagined where the company’s stakeholders wanted to answer the following questions:

1. What percentage of interested clients can we reach contacting 20,000 clients?
2. What would be that percentage if contacting 40,000 clients?
3. How many contacts the selling team must perform to reach 80% of the clients?

The answers provided by the model were (according to the cumulative gain and lift curves above):

1. 20,000 clients correspond to 26.2% of the client base. In this situation the sales team would reach 55% of the interested clients, a scenario 110% more efficient than contacting random clients.
2. 40,000 clients correspond to 52.5% of the client base. In this situation the sales team would reach 88% of the interested clients, a scenario 67% more efficient than contacting random clients.
3. Contacting 33,500 clients (44% of the client base) the sales team would reach 80% of the interested clients.

# 7. Data product

The company’s sale team will use the prediction proposed here on a regular basis. To facilitate usage, the model was deployed on Heroku and made accessible via an API. Also, a custom Google Sheet spreadsheet can access the API using Google Script to post a request. For the sake of illustration, a screenshot of the spreadsheet is presented below. Note the menu Insurance, which creates the column score, representing the client’s propensity to purchase the auto insurance.

<img src="/reports/figures/gsheet_screenshot.png" width="900">

# 8. Further improvements

One important challenge in this project was the unbalance nature of the dataset. Testing other balancing algorithms, as well as using ML models specialized in handling unbalanced data could improve the results presented.

