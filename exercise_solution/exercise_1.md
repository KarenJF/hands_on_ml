# Chapter 1: The Machine Learning Landscape

**In this chapter, we talked about the machine leanring landscape. The followings are the solutions to this chapter's exercise.**

### <span style="color:blue">1. How would you define Machine Learning?</span>

**Answer:** Machine Learning is the science (and art) of programming computers so they can learn from data.

Here is a slightly more general definition:
- Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed. (Arthur Samuel, 1959)

And a more engineering-oriented one:
- A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. (Tom Mitchell, 1997)

### <span style="color:blue"> 2. Can you name four types of problems where it shines?</span>
**Answer:**
- Problems for which existing solutions require a lot of hand tuning or long lists of rules: one Machine Learning algorithm can often simplify code and perform better.
- Complex problems for which there is no good solution at all using a traditional approach: the best Machine Learning techniques can find a solution.
- Fluctuating environments: a Machine Learning system can adapt to new data.
- Getting insights about complex problems and large amounts of data.

### <span style="color:blue"> 3. What is a labeled training set?</span>
**Answer**:
A labelled training set is the training data which includes the desired solutions/output, called labels. It is fed to the algorithm for supervised learning.

### <span style="color:blue"> 4. What are the two most common supervised tasks?</span>
**Answer**:
Regression and Classification.

### <span style="color:blue">5. Can you name four common unsupervised tasks?</span>
**Answer**:
Clustering, Anomaly Detection and novelty detectin, Visulization and dimensionality reduction and Association rule learning. 

### <span style="color:blue">6. What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains?</span>
**Answer**:
Reinforcement Learning is the Machine Learning algorithm that should be used which will allow the robot to observe the environment, select and perform actions and decide the best strategy (policy).

### <span style="color:blue">7. What type of algorithm would you use to segment your customers into multiple groups?</span>
**Answer**:
Clustering

### <span style="color:blue">8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?</span>
**Answer**:
Supervised because it uses labeled training set for learning. 

### <span style="color:blue">9. What is an online learning system?</span>
**Answer**:
Online learning system is a Machine Learning system that trains the system incrementally by feeding it data instances sequentially, either individually or by small groups called mini-batches so that the system can learn about new data on the fly, as it arrives.

### <span style="color:blue">10. What is out-of-core learning?</span>
**Answer**:
Online learning algorithms that used to train systems incrementally on huge datasets that cannot fit in one machine’s main memory, is called out-of-core learning. This algorithm loads part of the data, runs a training step on that data and repeats the process until it has run on all of the data. Out-of-core learning is usually done offline (i.e., not on the live system), so online learning can be a confusing name. Think of it as incremental learning.

### <span style="color:blue">11. What type of learning algorithm relies on a similarity measure to make predictions?</span>
**Answer**:
Instance-based learning

### <span style="color:blue">12. What is the difference between a model parameter and a learning algorithm’s hyperparameter?</span>
**Answer**:
- Model parameter is a parameter of the function/model which is calculated while training the process so that we get the best fit model.
- Whereas a hyperparameter is a parameter of a learning algorithm and not of the model. It is not affected by the learning algorithm itself rather it must be set prior to training and remains constant during training.

### <span style="color:blue">13. What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?</span>
**Answer**:
- Model-based learning algorithms search for the best model that can be used for prediction purposes.
- The most common strategy is: 
    - You studied the data.
    - You selected a model.
    - You trained it on the training data (i.e., the learning algorithm searched for the model parameter values that minimize a cost function).
    - Finally, you applied the model to make predictions on new cases (this is called inference), hoping that this model will generalize well.
- Make predictions: When input data is passed to the model and based on this and the model parameters the output is predicted.
    
### <span style="color:blue">14. Can you name four of the main challenges in Machine Learning?</span>
**Answer**:
- Insufficient Quantity of Training Data
- Nonrepresentative Training Data
- Poor-Quality Data
- Irrelevant Features

### <span style="color:blue">15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?</span>
**Answer**:
- That is Overfitting
- Three possible solutions are: 
    - To simplify the model by selecting one with fewer parameters (e.g., a linear model rather than a high-degree polynomial model), by reducing the number of attributes in the training data or by constraining the model
    - To gather more training data
    - To reduce the noise in the training data (e.g., fix data errors and remove outliers)

### <span style="color:blue">16. What is a test set and why would you want to use it?</span>
**Answer**:
Test set is the data on which the trained model is tested or evaluated. By evaluating the model on test set we get an estimate of the error rate on new cases also known as generalization error.

### <span style="color:blue">17. What is the purpose of a validation set?</span>
**Answer**:
The purpose of a validation set is to hold out a part of the training set to evaluate several candidate models and select the best model in the end.

### <span style="color:blue">18. What can go wrong if you tune hyperparameters using the test set?</span>
**Answer**:
By tuning hyperparameters using the test set we adapted the model and hyperparameters to produce the best model for that particular set. This means that the model is unlikely to perform as well on new data.

### <span style="color:blue">19. What is repeated cross-validation and why would you prefer it to using a single validation set?</span>
**Answer**:
Repeated cross-validation is evaluating each model on many small validation sets. Each model is evaluated once per validation set after being trained on the rest of the data. Repeated cross-validation is preferred over using a single validation set because by averaging out all the evaluations of a model, we can get a much more accurate measure of its performance.

