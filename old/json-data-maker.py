import glob
import os
import json 

'''names_of_dir =  os.listdir(os.path.join(os.getcwd(),"data/TRAIN")) 
names_of_dir.remove(".DS_Store")

names_of_dir =  os.listdir(os.path.join(os.getcwd(),"data/TRAIN")) 

'''

'''import glob
import os

file_list = glob.glob(os.path.join(os.getcwd(), "FolderName", "*.txt"))

corpus = []

for file_path in file_list:
    with open(file_path) as f_input:
        corpus.append(f_input.read())'''


##### 

# To do: Skapa en se till så att det inte är en lista med WAW
# Sätta ut så det blir min egna path - se line 113 för försök till detta! 
# Skriva ut i fil! 


#####

names_of_dir = os.walk(os.path.join(os.getcwd(),"data/TRAIN"))

listOfAllDir=list([x[1] for x in os.walk("data/TRAIN")])

listOfAllDir = [ele for ele in listOfAllDir if ele != []]
### taking away subs
dr_dirr = listOfAllDir[0]
del listOfAllDir[0]
#print(listOfAllDir)

#print(dr_dirr)
#print(dr_dirr[0])
#print(listOfAllDir[0])

PATH = os.getcwd()

file_wav_list = []
file_text_list = []
for drDirr in dr_dirr:
    outsidePath = os.path.join(PATH,os.path.join("data/TRAIN",drDirr))
    #print(outsidePath)
    for smalldir in listOfAllDir:
        #print(os.getcwd(), str(smalldir))
        #print("smalldir",smalldir)
        for smallerdir in smalldir:
            currentPath = os.path.join(outsidePath,smallerdir)
            #print("currentPath",currentPath)
            #print(os.path.join(currentPath,"*.WAV"))
            #print(os.path.join(drDirr,smallerdir)) 
            #print(os.getcwd()+"/"+str(smallerdir))
            if (os.path.exists(currentPath)):
                #print("this is a file: ",currentPath )
                file_wav_list.append(glob.glob(os.path.join(currentPath,"*.WAV")))
                file_text_list.append(glob.glob(os.path.join(currentPath,"*.TXT")))

## Har blir det konstigt! 
#print(file_wav_list)
file_wav_list_no_NA = [ele for ele in file_wav_list if ele != []]



## STILL NEEDS DOING -- get from path to file and remove two fist



textfiles = list()
for promptfileList in file_text_list:
    for promptfile in promptfileList:
        #print("promptfile",promptfile)
        #print(%s,prompt)
        #print(promptfile)
        with open(promptfile) as my_file:
            content = my_file.read()
            #print(content)
            if content:
                textfiles.append(content)
### Getting free text out of corpus

# for sub in textfiles:
#     print(sub)
    #print(textfilesIterator)
    #i+=1
    #print(i)
print("Hallå Eller", len(textfiles))
textfiles = [sub.strip().split(" ")[2:] for sub in textfiles]
### Det finns repeats i denna? 
#print("textfiles efter saker: ", len(textfiles))
textfilesAllCleaned = []
i=0
for i in range(len(textfiles)):
    #print(clearPrompt)
    textfilesAllCleaned.append(textfiles[i])

print(len(textfilesAllCleaned))
print("--------------------")
#print(textfilesAllCleaned)


### Taking away from path to waw, for sloppy work in beginning 


#wavfilesAllCleaned = [sub.replace("/Users/carlake/Desktop/KTH/DD2112 Speech Talteknologi/A.I Sweden Project/transformer-audio/", './') for sub in file_wav_list]


## Sanity check: 
#print(textfilesAllCleaned[0])
#print("wavfilesAllCleaned", len(wavfilesAllCleaned))
#print("wavfilesAllCleaned", wavfilesAllCleaned[0])

## Take all .sph files and put into JSON format

listOfDict = []
#print("wavfilesAllCleaned: ",wavfilesAllCleaned)

#print(len(textfilesAllCleaned))
#print(len(wavfilesAllCleaned))

for ii in range(len(file_wav_list)):
    dictOfJSON = {}
    #print(ii)
    dictOfJSON["Text:"] = " ".join(textfilesAllCleaned[ii])
    dictOfJSON["Ljud:"] = file_wav_list[ii]
    listOfDict.append(dictOfJSON)
#print(listOfDict)

json_object = json.dumps(listOfDict, indent = 4) 

print(json_object)
    
