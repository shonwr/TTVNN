'''
Sean Soliman & Luis Rangel
COMP 282
Project 3 - Neural Networks
'''
import os.path
import webbrowser
import numpy as np
from os import path
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

def selectFile():
    #SELECT STREAMER START********************************************************************
    print("LIST OF STREAMERS\n")
    datasetPath = "./Dataset/Streamers/"
    
    streamers = os.listdir(datasetPath)
    streamers.sort()

    i = 1
    for users in streamers:
        print(str(i) + "." + str(users))
        i += 1

    streamerNum = input("\nSelect a streamer: ")
    while int(streamerNum) < 1 or int(streamerNum) > i-1:
        print("Invalid selection.")
        streamerNum = input("\nSelect a streamer: ")
    print("**********************************************************") 
    streamerList = os.listdir(datasetPath + str(streamers[int(streamerNum)-1]))
    #SELECT STREAMER END**********************************************************************

    #SELECT CLIP START************************************************************************
    print("LIST OF CLIPS FROM",str(streamers[int(streamerNum)-1]),'\n')
    i=1
    for clip in streamerList:
        print(str(i) + "." + str(clip))
        i += 1
    clipNum = input("\nSelect a clip to predict category or enter 'a' to process all clips: ")
    clipPath = []
    if clipNum == 'a':
        count = 0
        for i in streamerList:
            clipPath.append(datasetPath + str(streamers[int(streamerNum)-1]) + "/" + str(streamerList[int(count)])) 
            count += 1;
    else:
        clipPath.append(datasetPath + str(streamers[int(streamerNum)-1]) + "/" + str(streamerList[int(clipNum)-1]))
    print("**********************************************************\n")
    #SELECT CLIP END**************************************************************************

    return clipPath



def processFile(clipPath):
    for p in clipPath: 
        if path.exists(p):
            #can specify txt encoding type in argument e.g. encoding="utf8"
            f = open(p,"r",errors='ignore')

            allData = []

            #empty dictionary for words from file
            d=dict()

            #dictionary to hold results 
            dataDict = {
                "url" : "",
                "funny" : 0,
                "epic" : 0,
                "sad" : 0
            }
            
            #categories - f - funny, e - epic, s - sad
            #list of "keywords"
            f_kw = ["LOL", "LUL", "OMEGALUL", "LOOL", "LOOOL" "LOOOOL", "HAHA", "ROFL", "xD", "LMAO","KEKW", "LULW"]
            #data["funny"] += 0
            
            e_kw = ["WOW", "POGCHAMP", "POG", "POGU", "POGGERS","HOLY", "OMG", "NUTS", "INSANE", "CLIP", "WTF", "EZ", "CLIP"]
            #data["epic"] += 0

            s_kw = ["BIBLETHUMP", "D:", "NOOO", "NOTLIKETHIS", "PEEPOSAD"]
            #data["sad"] += 0
            
            for line in f:
                line = line.strip()
                #line = line.upper()
                words = line.split(" ")
                for word in words:
                    if word in d:
                        #increment + 1 if word exists
                        d[word] = d[word] + 1
                    else:
                        #add word into dictionary and set val to 1  
                        d[word] = 1
                        
            #store url in dictionary
            link = next(iter(d))
            dataDict["url"] = link
            
            for key in list(d.keys()):
                #print( "LUL" == "LUL" ) evals to true
                if str(key).upper() in f_kw:
                    dataDict["funny"] += 1
                    #print("Funny keyword detected: " + str(key ))

                elif str(key).upper() in e_kw:
                    dataDict["epic"] += 1
                    #print("Epic keyword detected: " + str(key ))

                elif str(key).upper() in s_kw:
                    dataDict["sad"] += 1
                    #print("Sad keyword detected: " + str(key ))

            
            allData.append(dataDict)
            
        else:
            print("File does not exist")
        f.close()
        print(allData)
    return allData



def predictCategory(data):
    print("Predict catagory\n")
    print(data)
    for d in data:
        category = ""
        for keyWords in d:
            if d.get("funny") > d.get("epic") and d.get("funny") > d.get("sad"):
                 category = "funny." 
            elif d.get("epic") > d.get("funny") and d.get("epic") > d.get("sad"):
                category = "epic."
            elif d.get("sad") > d.get("funny") and d.get("sad") > d.get("epic"):
                category = "sad."
            elif d.get("funny") > 0 and d.get("funny") >= d.get("epic"):
                category = "funny and epic"
            elif d.get("funny") > 0 and d.get("funny") >= d.get("sad"):
                category = "funny and sad"
            elif d.get("funny") >= d.get("epic") and d.get("epic") >= d.get("sad"):
                category = "funny, epic, and sad- a trifecta!"
            elif d.get("epic") > 0 and d.get("epic") >= d.get("sad"):
                category = "epic and sad"
            
            '''
            print("This video is likely to be", category,
                     "\nfunny: " + str(d.get("funny")),
                     "\nepic: " + str(d.get("epic")),
                     "\nsad: " + str(d.get("sad")),
                     "\nURL:" + d["url"]
            )
            '''
        print("\nThis video is likely to be", category,"\n")
        
        
        openLink = input("Watch clip in a new browser tab?(y/n): ")
         
        while openLink.lower() != 'y' and openLink.lower() != 'n':
            print("\n\nInvalid choice. Please enter either y or n.")
            openLink = input("Watch clip on a new browser tab?(y/n): ")

        if openLink.lower() == "y":
            print("\n\nOpening clip in new browser tab...")
            webbrowser.open_new_tab(d.get("url"))
        


        
#SAMPLE SIMPLE NEURAL NETWORK SOURCE CODE START*********************************************
# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1 - x)

#map scores - [funny, epic, sad]
#model to detect whether a video is funny
#output should result matches first column

def simple_NN(e):
    # input dataset
    training_inputs = np.array([[1,0,1],
                                [1,1,0],
                                [1,0,1],
                                [1,1,0],
                                [1,0,0],])

    # output dataset
    training_outputs = np.array([[1,1,1,1,1]]).T


    '''
    # input dataset
    # first row should match output
                               #[f,e,s]
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,0,1],
                                [1,1,1],
                                [0,1,1]])

    # output dataset
    #output model to predict for funny videos 
    training_outputs = np.array([[0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,1,0]]).T
    '''

    # seed random numbers to make calculation
    np.random.seed(1)

    # initialize weights randomly with mean 0 to create weight matrix, synaptic weights
    synaptic_weights = 2 * np.random.random((3,1)) - 1

    print('Random starting synaptic weights: ')
    print(synaptic_weights)

    # Iterate 10,000 times

    for iteration in range(e):

        # Define input layer
        input_layer = training_inputs
        # Normalize the product of the input layer with the synaptic weights
        outputs = sigmoid(np.dot(input_layer, synaptic_weights))

        # how much did we miss?
        error = training_outputs - outputs

        # multiply how much we missed by the
        # slope of the sigmoid at the values in outputs
        adjustments = error * sigmoid_derivative(outputs)

        # update weights
        synaptic_weights += np.dot(input_layer.T, adjustments)
    print('Synaptic weights after training: ')
    print(synaptic_weights)

    
    print("Output After Training:")
    print(outputs)
    

    print("Error rate after training with:",e,"iterations")
    print(error,"\n\n")
    print("***********************************************************") 
        
        
#SAMPLE SIMPLE NETWORK SOURCE CODE END*********************************************


#SAMPLE DIABETES NEURAL NETWORK SOURCE CODE START*********************************************

def diabetes_NN(e): 
    # random seed for reproducibility
    np.random.seed(2)

    # loading load prima indians diabetes dataset, past 5 years of medical history 
    dataset = np.loadtxt("prima-indians-diabetes.csv", delimiter=",")

    # split into input (X) and output (Y) variables, splitting csv data
    X = dataset[:,0:8]
    Y = dataset[:,8]

    # split X, Y into a train and test set
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # create model, add dense layers one by one specifying activation function
    model = Sequential()
    model.add(Dense(15, input_dim=8, activation='relu')) # input layer requires input_dim param
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1

    # compile the model, adam gradient descent (optimized)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    # call the function to fit to the data (training the network)
    model.fit(x_train, y_train, epochs=e, batch_size=20, validation_data=(x_test, y_test))

    # save the model
    model.save('weights.h5')



#SAMPLE DIABETES NEURAL NETWORK SOURCE CODE END*********************************************



def menu():
    print("**********************************************************")
    print("1. Twitch Sample\n2. Neural Network Sample\n3. Diabetes Neural Network Sample\n")
    print("**********************************************************")
    selection = int(input("Enter selection: "))
    print("**********************************************************") 
    while selection != 1 and selection != 2 and selection != 3:
        print("\nInvalid selection\n")
        print("1. Twitch Neural Network")
        print("2. Simple Neural Network [Sample]")
        print("3. Diabetes Neural Network [Sample]")
        selection = int(input("Enter selection: "))
        selection = int(selection)
    
    if selection == 1:
        clipPath = selectFile()
        data = processFile(clipPath)
        predictCategory(data)
    elif selection ==2:
        simple_NN(50000)
        simple_NN(100000)
        simple_NN(500000)
    else:
        diabetes_NN(1000)
        
'''
#clipPath = selectFile()
#data = processFile(clipPath)

train(50000)
train(100000)
train(500000)

#predictCategory(data)
'''

menu()
