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
    print("LIST OF STREAMERS")
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
    print("******************************************\n")
    streamerList = os.listdir(datasetPath + str(streamers[int(streamerNum)-1]))
    #SELECT STREAMER END**********************************************************************

    #SELECT CLIP START************************************************************************
    print("LIST OF CLIPS FROM",str(streamers[int(streamerNum)-1]))
    i=1
    for clip in streamerList:
        print(str(i) + "." + str(clip))
        i += 1
    clipNum = input("\nSELECT A CLIP TO PREDICT CATEGORY: ")
    print("******************************************\n")
    clipPath = datasetPath + str(streamers[int(streamerNum)-1]) + "/" + str(streamerList[int(clipNum)-1])
    #SELECT CLIP END**************************************************************************

    return clipPath



def processFile(clipPath):
    if path.exists(clipPath):
        #can specify txt encoding type in argument e.g. encoding="utf8"
        f = open(clipPath,"r",errors='ignore')

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

        
        f.close()
        
    else:
        print("File does not exist")

    return dataDict



def predictCategory(data):
    category = ""
    for keyWords in data:
        if data.get("funny") > data.get("epic") and data.get("funny") > data.get("sad"):
             category = "funny." 
        elif data.get("epic") > data.get("funny") and data.get("epic") > data.get("sad"):
            category = "epic."
        elif data.get("sad") > data.get("funny") and data.get("sad") > data.get("epic"):
            category = "sad."
        elif data.get("funny") > 0 and data.get("funny") >= data.get("epic"):
            category = "funny and epic"
        elif data.get("funny") > 0 and data.get("funny") >= data.get("sad"):
            category = "funny and sad"
        elif data.get("funny") >= data.get("epic") and data.get("epic") >= data.get("sad"):
            category = "funny, epic, and sad- a trifecta!"
        elif data.get("epic") > 0 and data.get("epic") >= data.get("sad"):
            category = "epic and sad"
        
        '''
        print("This video is likely to be", category,
                 "\nfunny: " + str(data.get("funny")),
                 "\nepic: " + str(data.get("epic")),
                 "\nsad: " + str(data.get("sad")),
                 "\nURL:" + data["url"]
        )
        '''
    print("This video is likely to be", category)

    openLink = input("Watch clip on a new browser tab?(y/n): ")
     
    while openLink.lower() != 'y' and openLink.lower() != 'n':
        print("\nInvalid choice. Please enter either y or n.")
        openLink = input("Watch clip on a new browser tab?(y/n): ")

    if openLink.lower() == "y":
        print("\n\nOpening clip in new browser tab...")
        webbrowser.open_new_tab(data.get("url"))


        
#SAMPLE NEURAL NETWORK SOURCE CODE START*********************************************
# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1 - x)

#map scores - [funny, epic, sad]
#model to detect whether a video is funny
#output should result matches first column

def train(i):
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

    for iteration in range(i):

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
    

    print("Error rate after training with:",i,"iterations")
    print(error,"\n\n")
    print("***********************************************************") 
        
        
#SAMPLE NEURAL NETWORK SOURCE CODE END*********************************************

def menu():
    print("1. Twitch Sample Run\n2. Neural Network Source Code\n")
    selection = int(input("Enter selection: "))
    print("***********************************************************") 
    while selection != 1 and selection != 2:
        print("\nInvalid selection")
        print("1. Twitch Sample Run\n2. Neural Network Source Code\n")
        selection = int(input("Enter selection: "))
        selection = int(selection)
    
    if selection == 1:
        clipPath = selectFile()
        data = processFile(clipPath)
        predictCategory(data)
    else:
        train(50000)
        train(100000)
        train(500000)
'''
#clipPath = selectFile()
#data = processFile(clipPath)

train(50000)
train(100000)
train(500000)

#predictCategory(data)
'''

menu()
