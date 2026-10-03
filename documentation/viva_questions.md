# Viva Questions and Simple Answers (30)

**1. What is R?**
R is a free programming language made for statistics, data analysis and graphs.

**2. Why is R used in this project?**
It has ready-made tools for cleaning data (tidyverse), drawing graphs (ggplot2), machine learning (caret, randomForest) and building web dashboards (Shiny), so the whole project can be done in one language.

**3. What is data preprocessing?**
Preparing raw data so that it is clean and usable: removing duplicates, fixing wrong values, filling missing values, converting data types and encoding categories.

**4. How did you handle missing values?**
Numeric columns were filled with the median and category columns with the most common value (mode). The median is used because it is not affected by extreme values.

**5. What is an outlier and how did you detect it?**
An outlier is a value far away from the others. I used two checks: impossible values (for example a speed of 400 km/h) are treated as errors and fixed; and the IQR rule (beyond 1.5 x IQR) is used to report unusual but possible values, which are kept.

**6. What is EDA?**
Exploratory Data Analysis - studying the data with statistics and graphs to understand patterns before building models.

**7. What is classification?**
A type of machine learning where the output is a category. Here the categories are Low, Medium and High congestion.

**8. Why classification and not regression?**
The target is a category, not a continuous number. (Regression predicts numbers such as exact delay minutes.)

**9. What is logistic regression?**
A model that calculates the probability of each class from a weighted combination of inputs. Because we have three classes, we use the multinomial version.

**10. What is a decision tree?**
A flowchart-like model that asks a series of yes/no questions about the inputs (for example "speed below 30?") until it reaches a class.

**11. What is a random forest?**
A collection of many decision trees (200 here), each trained on a random part of the data. The final answer is the majority vote, which is usually more accurate and stable than one tree.

**12. Why did you use multiple models?**
To compare them on the same data and choose the best one using measured metrics instead of guessing.

**13. What is training data?**
The part of the data (80%) the model learns from.

**14. What is testing data?**
The part (20%) kept hidden during learning and used only to check how well the model works on new data.

**15. Why an 80:20 split?**
It is a common balance: enough data to learn from and enough to test fairly. The split is stratified so Low/Medium/High appear in the same proportion in both parts.

**16. What is `set.seed()` and why is it used?**
It fixes the random number generator so the dataset, the split and the models are identical every time, making results reproducible.

**17. What is accuracy?**
Correct predictions divided by total predictions.

**18. What is precision?**
Out of the records the model predicted as a class (say High), the fraction that really were High.

**19. What is recall?**
Out of the records that really were High, the fraction the model found.

**20. What is the F1-score?**
The harmonic mean of precision and recall - a single number that balances both.

**21. What is a confusion matrix?**
A table comparing actual classes with predicted classes. The diagonal shows correct predictions; other cells show the mistakes.

**22. How did you calculate precision/recall for three classes?**
They are calculated for each class and then averaged (macro average), so every class counts equally.

**23. How did you choose the best model?**
The model with the highest F1-score on the test data is chosen; accuracy breaks a tie. The app reads this from the measured results - nothing is typed in by hand.

**24. Why use R Shiny?**
It turns R code into an interactive web app, so a user can enter traffic conditions and see a prediction without writing code.

**25. What is synthetic data and why did you use it?**
Synthetic data is computer-generated data that follows realistic rules. No real dataset was available for this project, and synthetic data can be reproduced, controlled and shared freely.

**26. What are the limitations of synthetic data?**
It is simpler than real traffic, and the labels follow a rule plus noise, so model scores look better than they would on real data.

**27. How do you make sure prediction uses the same preprocessing as training?**
The user input goes through `validate_inputs()` and `build_new_observation()`, which use the same column names, data types and factor levels as the cleaned training data. The logistic-regression scaling is stored inside the trained model and re-applied automatically.

**28. How does the app handle wrong input?**
It checks for empty values, non-numbers, out-of-range numbers and unknown categories and shows a clear red message. Everything is wrapped in `tryCatch()`, so the app does not crash. A missing dataset is regenerated and a missing/damaged model file is retrained.

**29. Which factors affect congestion most in your data?**
Check the Random Forest feature-importance graph and the EDA graphs of your run. Typically vehicle count, speed, hour and accidents matter most. (Answer with the values from your own results.)

**30. What are the limitations and future improvements?**
Limitations: synthetic data, one train/test split, single location, no live data. Improvements: real data, cross-validation and tuning, live weather and traffic APIs, map display and time-series forecasting.

**Bonus 31. What is the difference between training accuracy and test accuracy?**
Training accuracy shows how well the model fits data it has seen; test accuracy shows how well it works on new data. A much higher training accuracy means over-fitting.

**Bonus 32. What is over-fitting?**
When a model memorises training data, including noise, and performs badly on new data.

**Bonus 33. What is a factor in R?**
A data type for categories (for example Low/Medium/High) with a fixed set of allowed levels.

**Bonus 34. What is the pipe operator `%>%`?**
It passes the result of one step into the next, so code reads left to right like a sentence.
