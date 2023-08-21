**BATCH GRADIENT DESCENT**

# First I load my simple dataset which include three columns : area, bedrooms and	price
* Now I need to scale my data, so I import "MinMaxScaler" from "preprocessing", and put inside area and bedroom columns I save it as X_scaled 
* Then I scale price column and I also reshape it into 2D array, I save it as x_scaled

# Now I gonna build from scratch batch gradient descent, I gonna need here X, y_true, epochs and learning_rate which I set to (0.01)
* First I pass "number_of_features", so "area and bedroom", shape of it will be in rows and columns
* Then I initialize my weights with ones and set shape as previous saved "number_of_features", also initialize bias with zero
* And I pass shape of "total_samples" for later which is " X.shape[0]"
* I create two empty list to record cost and epochs and later plot them 

# Then I make for loop in length of my epochs, inside I define "y_predicted", which is dot product of w1, w2 and X transpose, plus bias
* Now I define first derivative, so I pass X transpose dot product of difference between true and predicted value into minus average of total samples 
* Then my second derivative, which is sum of difference between true and predicted value into minus average of total samples
* Next I can calculate my weights, so my w minus learning rate into previous defined w_grad, and this same with bias 
* Then I define cost of my function, which is mean squared error between true and predicted value

# Now I define recording progress, and it will be every tenth iteration
* In my lists will be filling with values recorded values, cost and epochs
* And my function will return weights, bias, cost, cost list and epoch list, so I can plot it later

# Then I call my function and put inside X_scaled, y_scaled_reshaped, pass number of epochs (500)
* And get weights, bias and cost, I can compare it later whith "stochastic_gradient_descent"

# I plot history how my cost change during the epochs, I can see that close to 100th epoch my cost stop decreasing rapidly
* But I can see that "batch_gradient_descent" have smoth curve 

# Then I create prediction function, I gonna use in it "inverse_transform" from my "MinMaxScaler"
* So it return me inverse values, np. when I put 1 into it it will return me max. argument in my column, when i put 0.5, it will return me mean value of column
* I gonna use in my predict function : area, bedrooms, weights and bias 
* Now I must scale my features, so I use transform method from my scaler and put inside area and bedrooms and I supply [0] to get 1D array, all this I save as X_scaled
* Then I fill patter to get "scaled_price", so (w1 into X_scaled of [0] which is area  variable, plus w2 into X_scaled of [1] which is bedrooms variable, plus some bias
* Next I use "inverse_transform" method on "scaled_price" variable to get actual price, and supply [0][0] to get single value

# First prediction is for second house in my data, which is quite well predicted, next I check 5th house which is worst than previous
* My function not perform excellent, but is able to get close real values

**STOCHASTIC GRADIENT DESCENT**
# I import and prepare my dataset as previous and scale all columns

# Now I can build "Mini Batch Gradient Descent", first i put some features, (X and y_true, number of epochs, batch size and learning rate)
* Then I define "number_of_features", which are (area and bedroom), it will be "X.shape[1]"
* Then I initialize weights with "ones" and pass shape which are my previous defined features
* And I also initialize bias with "zero", my columns have equal size, so I define "total_samples" as "X.shape[0]"

# Now I define batches for my "Gradien Descent", so I specify if "total samples" are less than "variable batch_size" which is (5)
* Then if this condition is true "batch_size will be equal total samples", this is a simple python trick, thats the way how it gonna pick batches
* And I create two lists to record epochs and cost

# Then I create for loop in range of my epochs and inside I define variable "random indices"
* "random.permutation" from numpy, it gonna take "all samples" and pick "random sample"
* And I gonna take random samples from "X and y_true", I save it as "X_tmp and y_tmp"

# Then I make another for loop (from 0 to total_samples variable) I also put (batch_size variable which is 5)
* I create two new variables (Xj and yj) I define them as (batches of random samples from X and y)

# Next I define (y_pred) which is dot product between (weights variable and Xj transpose variable) plus bias
* Now I gonna define (w_grad) by taking dot product with (Xj transpose) of (difference between yj variable and y_pred), I take mean multiply by -2 of it 
* Next I define (b_grad) which is sum (of difference between yj variable and y_pred) and -2 mean of it
* Now my derivatives are defined, I gonna use them to calculate (weights and bias)
* They are my (weights and bias variables) minus previou defined (learning_rate which is 0.01) into (w_grad and b_grad variables) defined above

# Now my (weights) are adjusted, I can define (cost) which is (mean squared error)
* So first I use (np.square) to get squared error between (yj - y_pred) and then i take (mean of it) using (np.mean)

# Next I create condition to fill my (cost_list and epoch_list), they will be filling every tenth iteration
* I return from my function (weights, bias, cost and both lists)

# Then I call my function and supply it with (X_scaled, y_scaled reshaped into 1D array)
* I also set number of (epochs at 120 and number of batches at 5)
* I get (both weights, bias and cost) my (mini_batch_gradient_descent) function is working well

# Next I plot on graph (Epochs and Cost) of my function calling both lists, previous returned from my function
* The line look smoother than (Stochastic gradient Descent line) but its not that smooth that in (Batch Gradient Descent)
* Its not perfect smooth line because my Gradient Descent (use batch of randomly picked samples)

# Then I define (predict function) exactly this same as in previous functions
* I supply (predict function) into some values and both variables (weights and bias), it looks that all works well

# Mini Batch Gradient Descent use batch of randomly picked samples for a forward pass and then adjust weights
# In comparison to (Batch Gradient Descent) which use "all" training samples and (Stochastic Gradient Descent) which use "one" random sample for forward pass
# Its also good for big datasets, as well as "SGD", because it use batch of samples to train the model
