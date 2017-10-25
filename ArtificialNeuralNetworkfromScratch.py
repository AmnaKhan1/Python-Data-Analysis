#implementing neural network from scratch
import numpy as np

#----------Data
def base_data():
    #Declaring Input array
    input_X=np.array([[1,0,1,0]
                     ,[1,0,1,1],
                      [0,1,0,1]])
    #declaring utput array
    Output_y=np.array([[1],[1],[0]])
    return input_X,Output_y

#------- Activation Function
def sigmoid_actv(x):
    return 1/(1+np.exp(-x))

#------ Derivative Function
def sigmoid_derivative(x):
    return x*(1-x)

#----- Setting Neral Network environment variables
def neural_Environment_init(X):
    epochs=6000
    learning_rate=0.1
    input_layer_neurons=X.shape[1] #Number of features in the dataset
    hidden_layer_neurons=3
    output_layer_neurons=1
    w_hidden,bais_hidden,w_output,bais_output=weights_bais(0,0,0,0,input_layer_neurons,hidden_layer_neurons,output_layer_neurons)
    return epochs,learning_rate,input_layer_neurons,hidden_layer_neurons,output_layer_neurons,w_hidden,bais_hidden,w_output,bais_output

def weights_bais(wh,bh,wo,bo,input_layer_neurons,hidden_layer_neurons,output_layer_neurons):
    if (wh ==0 and bh==0 and wo==0 and bo==0):
        w_hidden=np.random.uniform(size=(input_layer_neurons,hidden_layer_neurons))
        bais_hidden=np.random.uniform(size=(1,hidden_layer_neurons))
        w_output=np.random.uniform(size=(hidden_layer_neurons,output_layer_neurons))
        bais_output=np.random.uniform(size=(1,output_layer_neurons))
#        print "**********bais_ouput.shape: ", bais_output.shape
    else:
        w_hidden=wh
        bais_hidden=bh
        w_output=wo
        bais_output=bo
#    print "HELP beofre return  bais_ouput.shape: ", bais_output.shape
    return w_hidden,bais_hidden,w_output,bais_output

if __name__=="__main__":
    X,Y=base_data()
    epochs,learning_rate,input_layer_neurons,hidden_layer_neurons,output_layer_neurons,w_hidden,bais_hidden,w_output,bais_output=neural_Environment_init(X)
    
    for i in range(epochs):
        # Forward propogation
        hidden_layer_input1=np.dot(X,w_hidden) # 3x3=(3x4)(4x3)
        hidden_layer_input=hidden_layer_input1 + bais_hidden# 3x3= (3x3)+(1x3)

        hidden_layer_activations=sigmoid_actv(hidden_layer_input) # 3x3
        
        output_layer_input1=np.dot(hidden_layer_activations,w_output) #3x1=(3x3)dot(3x1)
        
        output_layer_input=output_layer_input1+bais_output 
        output=sigmoid_actv(output_layer_input)

        #Backward propogation
        error_output_layer=Y-output
        slope_output_layer=sigmoid_derivative(output)
        delta_output_layer=error_output_layer*slope_output_layer
        
        slope_hidden_layer=sigmoid_derivative(hidden_layer_activations)
        error_hidden_layer=np.dot(np.transpose(delta_output_layer),w_output)
        delta_hidden_layer=error_hidden_layer*slope_hidden_layer
        
        #Updates
        w_output= w_output+np.dot(hidden_layer_activations,delta_output_layer)*learning_rate
        bais_output=bais_output+np.sum(delta_output_layer,axis=0)*learning_rate
        w_hidden=w_hidden+np.dot(np.transpose(X),delta_hidden_layer)*learning_rate
        bais_hidden=bais_hidden+np.sum(delta_hidden_layer,axis=1)*learning_rate

        
    print output
